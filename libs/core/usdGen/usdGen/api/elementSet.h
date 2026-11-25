// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The generic data model: channel tables, element sets and their views.
#pragma once
#include "usdGen/api/arena.h"
#include "usdGen/api/channel.h"
#include "usdGen/api/export.h"
#include "usdGen/api/span.h"
#include "usdGen/exec/pool.h"
#include <cstdint>

namespace usdGen {

class UsdGenElementSet;
class UsdGenWriteSet;
struct UsdGenElementSetView;

/// A chunk: a contiguous range of level-1 elements and the sub-ranges under them.
struct UsdGenChunkRange {
    uint32_t first[kUsdGenMaxLevels]  = {};      // [0] first element, [1] first level-2 item, ...
    uint32_t count[kUsdGenMaxLevels]  = {};      // allocated items at each level
    uint32_t stride[kUsdGenMaxLevels] = {};      // items per parent when uniform, 0 when ragged; [0] unused
    uint32_t live = 0;                           // elements surviving decimation (<= count[0])
    uint16_t group = 0;                          // publication group (a C2 tile for curves)
    uint16_t domain = 0;                         // surface id for surface-rooted primitives
    friend bool operator==(UsdGenChunkRange const &a, UsdGenChunkRange const &b) noexcept
    {
        for (uint8_t l = 0; l < kUsdGenMaxLevels; ++l)
            if (a.first[l] != b.first[l] || a.count[l] != b.count[l] || a.stride[l] != b.stride[l])
                return false;
        return a.live == b.live && a.group == b.group && a.domain == b.domain;
    }
    friend bool operator!=(UsdGenChunkRange const &a, UsdGenChunkRange const &b) noexcept
    {
        return !(a == b);
    }
};

/// Chunks grouped for publication.
struct UsdGenChunkGroup {
    uint16_t id = 0;
    uint32_t firstChunk = 0, chunkCount = 0;
    uint32_t live[kUsdGenMaxLevels] = {};        // live elements, live level-2 items, ...
    float    extentMin[3] = { 3.4e38f,  3.4e38f,  3.4e38f};   // GfRange3f empty state
    float    extentMax[3] = {-3.4e38f, -3.4e38f, -3.4e38f};
    uint32_t dirtyRoles = 0;                     // bit (1 << UsdGenChannelRole) written this run
};

struct UsdGenPartitionParams { uint32_t chunkSize = 512; uint32_t groupTarget = 64; };

/// Registered by a primitive with its TypeDef. Hooks are plain functions:
/// they run on the cook lane, never allocate and never call back into the engine.
struct UsdGenElementLayout {
    UsdGenTypeId primitive = UsdGenTypeId::Invalid;
    uint8_t levels = 1;
    UsdGenSpan<const UsdGenChannelDesc> channels;   // static channels; slot == index
    /// Chunks for one node over 'elements' level-1 elements; 'set' is the node's
    /// effective set (sub-item layout). Returns the chunk count; writes min(count, out.size()).
    uint32_t (*partition)(uint32_t elements, UsdGenElementSetView const &set,
                          UsdGenPartitionParams const &, UsdGenSpan<UsdGenChunkRange> out) = nullptr;
    /// Publication groups over 'chunks' chunks. Returns the group count.
    uint32_t (*groups)(uint32_t chunks, UsdGenPartitionParams const &,
                       UsdGenSpan<UsdGenChunkGroup> out) = nullptr;
};

/// Static channels of a layout followed by dynamic channels ordered by
/// (max(level,1), name): the per-element run, then the deeper runs.
/// Immutable, refcounted, shared by every set that has the same channels.
/// Built at compile or topology time only (allocates from the pools).
class UsdGenChannelTable final : public UsdGenRefCounted, public UsdGenPooled {
public:
    /// Dynamic entries naming a static channel replace its descriptor (same
    /// slot); a repeated dynamic name keeps the last entry.
    USDGEN_CORE_API static UsdGenRef<const UsdGenChannelTable>
        Make(UsdGenElementLayout const &, UsdGenSpan<const UsdGenChannelDesc> dynamic);
    /// This table plus 'added' (same-name entries replaced); returns this when nothing changes.
    USDGEN_CORE_API UsdGenRef<const UsdGenChannelTable>
        With(UsdGenSpan<const UsdGenChannelDesc> added) const;
    UsdGenElementLayout const &Layout() const noexcept { return *_layout; }
    uint16_t Size() const noexcept { return uint16_t(_descs.size()); }
    uint16_t StaticCount() const noexcept { return _staticCount; }
    UsdGenChannelDesc const &operator[](UsdGenSlot s) const noexcept { return _descs[s]; }
    /// Compile/prepare time only (binary search over the dynamic runs).
    USDGEN_CORE_API UsdGenSlot Find(UsdGenToken const &name) const noexcept;
    /// Slots of the dynamic run at 'level' (levels 0 and 1 share a run).
    UsdGenSpan<const UsdGenSlot> DynamicRun(uint8_t level) const noexcept
    {
        uint8_t const r = level < 1 ? 1 : level > kUsdGenMaxLevels ? kUsdGenMaxLevels : level;
        return {_dynamic.data() + _runBegin[r], size_t(_runBegin[r + 1] - _runBegin[r])};
    }
    uint64_t Digest() const noexcept { return _digest; }   // content identity for per-node slot caches

private:
    friend class UsdGenWriteSet;
    UsdGenChannelTable() noexcept = default;
    ~UsdGenChannelTable() override = default;
    static UsdGenRef<const UsdGenChannelTable> _Build(UsdGenElementLayout const &,
                                                      UsdGenSpan<const UsdGenChannelDesc> base,
                                                      uint16_t staticCount,
                                                      UsdGenSpan<const UsdGenChannelDesc> added,
                                                      UsdGenChannelTable const *parent);
    UsdGenElementLayout const *_layout = nullptr;
    UsdGenPoolVector<UsdGenChannelDesc> _descs;
    UsdGenPoolVector<UsdGenSlot> _dynamic;               // dynamic slots in run order
    uint16_t _runBegin[kUsdGenMaxLevels + 2] = {};       // into _dynamic; run r = [r, r+1), r >= 1
    uint16_t _staticCount = 0;
    uint64_t _digest = 0;
    /// With(): the table this one extends and, per slot here, the slot there.
    UsdGenRef<const UsdGenChannelTable> _parent;
    UsdGenPoolVector<UsdGenSlot> _fromParent;
};

/// Raw, non-owning view for workers and op Capture. Valid while the cook holds the set.
struct UsdGenElementSetView {
    UsdGenChannelTable const *channels = nullptr;
    uint32_t count[kUsdGenMaxLevels + 1] = {1, 0, 0, 0};        // count[0] == 1
    uint32_t const *offsets[kUsdGenMaxLevels + 1] = {};          // offsets[L]: count[L-1]+1 entries; null = uniform
    void const *const *columns = nullptr;                        // per slot; null = absent
    uint32_t const *values = nullptr;                            // values per slot (items * arity)
    uint64_t topologyVersion = 0, valueVersion = 0;
    template <class T> T const *Column(UsdGenSlot s) const noexcept { return static_cast<T const *>(columns[s]); }
    bool Has(UsdGenSlot s) const noexcept { return channels && s < channels->Size() && columns[s]; }
    uint32_t Values(UsdGenSlot s) const noexcept { return values[s]; }
    /// Items per parent at 'level' when uniform and exact, else 0.
    uint32_t Stride(uint8_t level) const noexcept
    {
        return offsets[level] || !count[level - 1] || count[level] % count[level - 1]
            ? 0u : count[level] / count[level - 1];
    }
};

/// One primitive's output: immutable once sealed. Columns live in at most
/// kUsdGenMaxArenasPerSet arenas; offsets and the column table are resolved at Seal.
class UsdGenElementSet final : public UsdGenRefCounted, public UsdGenPooled {
public:
    UsdGenChannelTable const &Channels() const noexcept { return *_channels; }
    uint32_t Count(uint8_t level) const noexcept { return level <= kUsdGenMaxLevels ? _count[level] : 0u; }
    /// count[level-1]+1 entries; empty = uniform.
    UsdGenSpan<const uint32_t> Offsets(uint8_t level) const noexcept
    {
        return level >= 1 && level <= kUsdGenMaxLevels && _offsets[level]
            ? UsdGenSpan<const uint32_t>(_offsets[level], size_t(_count[level - 1]) + 1)
            : UsdGenSpan<const uint32_t>();
    }
    /// Size in T: the column's bytes / sizeof(T) (a packed float3 column reads as GfVec3f).
    template <class T>
    UsdGenSpan<const T> Column(UsdGenSlot s, uint8_t sample = 0) const noexcept
    {
        if (sample != 0 || !Has(s)) return {};
        return {static_cast<T const *>(_columns[s]),
                size_t(_values[s]) * UsdGenScalarBytes((*_channels)[s].scalar) / sizeof(T)};
    }
    bool Has(UsdGenSlot s) const noexcept { return s < _columns.size() && _columns[s]; }
    /// False: one sample (every channel today). Hydra 2.0 shape, filled when motion lands.
    bool SampleTimes(UsdGenSlot, double, double, UsdGenSpan<const double> *) const noexcept { return false; }
    uint64_t TopologyVersion() const noexcept { return _topologyVersion; }
    uint64_t ValueVersion() const noexcept { return _valueVersion; }     // bumped by every Seal that wrote a column
    uint64_t ValueVersion(UsdGenSlot s) const noexcept { return s < _versions.size() ? _versions[s] : 0; }
    USDGEN_CORE_API UsdGenElementSetView View() const noexcept;           // no allocation
    uint8_t ArenaCount() const noexcept { return _arenaCount; }
    UsdGenArena const &ArenaAt(uint8_t i) const noexcept { return *_arenas[i]; }
    /// The column's arena and byte offset; a device mirror uses the same offsets.
    uint8_t ColumnArena(UsdGenSlot s) const noexcept { return Has(s) ? _columnArena[s] : uint8_t(0xFF); }
    USDGEN_CORE_API uint64_t ColumnOffset(UsdGenSlot) const noexcept;
    /// Marks every arena shared (cache store, host handoff).
    USDGEN_CORE_API void Export() const noexcept;
    /// No channels written, every count 0 (level 0: 1). Allocates: compile time.
    USDGEN_CORE_API static UsdGenRef<const UsdGenElementSet> Empty(UsdGenElementLayout const &);

private:
    friend class UsdGenWriteSet;
    UsdGenElementSet() noexcept = default;
    ~UsdGenElementSet() override = default;
    UsdGenRef<const UsdGenChannelTable> _channels;
    uint32_t _count[kUsdGenMaxLevels + 1] = {1, 0, 0, 0};
    uint32_t const *_offsets[kUsdGenMaxLevels + 1] = {};
    UsdGenRef<const UsdGenArena> _arenas[kUsdGenMaxArenasPerSet];
    uint8_t _arenaCount = 0;
    uint8_t _delta = 0xFF;                       // arena this set's own Seal wrote, or 0xFF
    uint64_t _topologyVersion = 0, _valueVersion = 0;
    UsdGenPoolVector<void const *> _columns;     // per slot
    UsdGenPoolVector<uint32_t> _values;
    UsdGenPoolVector<uint8_t> _columnArena;
    UsdGenPoolVector<uint64_t> _versions;
};

}  // namespace usdGen
