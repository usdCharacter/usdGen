// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The only write path into an element set.
#pragma once
#include "usdGen/api/elementSet.h"
#include "usdGen/api/export.h"

namespace usdGen {

/// RAII: an unsealed WriteSet returns its delta arena to the pool.
/// Order: Derive or Fresh, [Recycle], [Share/Drop], Plan, Write/WriteOffsets, Seal.
class UsdGenWriteSet {
public:
    /// Same counts as 'base'; unwritten columns are shared with it (matched by
    /// slot, through With()'s parent map, or by name; a layout mismatch is absent).
    USDGEN_API static UsdGenWriteSet Derive(UsdGenRef<const UsdGenElementSet> base,
                                                 UsdGenRef<const UsdGenChannelTable> channels);
    /// New topology: counts per level 1..levels; every column starts absent.
    USDGEN_API static UsdGenWriteSet Fresh(UsdGenRef<const UsdGenChannelTable> channels,
                                                UsdGenSpan<const uint32_t> counts);
    /// Before Plan: reuse 'previous''s delta arena in place when it is unshared,
    /// big enough and held only by 'previous' (the warm path: no pool traffic).
    /// Pass the last reference (moved): a 'previous' held elsewhere is refused.
    USDGEN_API void Recycle(UsdGenRef<const UsdGenElementSet> previous) noexcept;
    /// Before Plan: rewrite 'previous''s delta arena in place (D9), even
    /// while sets of the same cook still share it (they see the new values,
    /// like a column aliased in place). Plan keeps the arena only when it is
    /// not shared outside the cook, big enough, no shared column lives in it,
    /// and every planned column (and offset table) sits where 'previous' had
    /// the same channel with the same size; a kept column keeps its contents
    /// until written. Otherwise Plan takes another arena and 'previous'
    /// keeps its bytes. True after Plan when the arena was kept.
    USDGEN_API void RecycleInPlace(UsdGenRef<const UsdGenElementSet> previous) noexcept;
    USDGEN_API bool RecycledInPlace() const noexcept;
    /// Size one delta arena for every column written, before the first Write.
    /// Copies shared columns into it when the set would need more than
    /// kUsdGenMaxArenasPerSet arenas.
    USDGEN_API void Plan(UsdGenSpan<const UsdGenSlot> written);
    /// Fresh only, after Plan, levels 2..: count[L-1]+1 entries (contents
    /// undefined until written). A level never asked for stays uniform.
    USDGEN_API UsdGenSpan<uint32_t> WriteOffsets(uint8_t level);
    /// After Plan; a planned slot only. Contents undefined until written.
    template <class T> UsdGenSpan<T> Write(UsdGenSlot slot, uint8_t sample = 0)
    {
        return {static_cast<T *>(WriteRaw(slot, sample)), ValueBytes(slot) / sizeof(T)};
    }
    /// Before Plan.
    USDGEN_API void Share(UsdGenSlot slot, UsdGenElementSet const &from, UsdGenSlot fromSlot);
    USDGEN_API void Drop(UsdGenSlot slot);
    /// Before Plan/Share: a free-length column (input records) holds 'values'
    /// values instead of count[level] * arity.
    USDGEN_API void SetValueCount(UsdGenSlot slot, uint32_t values) noexcept;
    /// Overrides the versions Seal stamps (0 keeps the automatic one).
    USDGEN_API void SetVersions(uint64_t topologyVersion, uint64_t valueVersion) noexcept;
    /// The versions Seal stamps, zero included.
    USDGEN_API void SetVersionsExact(uint64_t topologyVersion, uint64_t valueVersion) noexcept;
    USDGEN_API UsdGenRef<const UsdGenElementSet> Seal() &&;
    /// Re-stamps a sealed set's value version (and its delta columns') in
    /// place: the commit thread, after the sweep that wrote it joined, while
    /// the set is still private to the cook.
    USDGEN_API static void Restamp(UsdGenElementSet const &set, uint64_t valueVersion) noexcept;
    USDGEN_API ~UsdGenWriteSet();
    UsdGenWriteSet(UsdGenWriteSet &&o) noexcept : _impl(o._impl) { o._impl = nullptr; }
    UsdGenWriteSet &operator=(UsdGenWriteSet &&o) noexcept
    {
        UsdGenWriteSet tmp(static_cast<UsdGenWriteSet &&>(o));
        Impl *t = _impl; _impl = tmp._impl; tmp._impl = t;
        return *this;
    }
    UsdGenWriteSet(UsdGenWriteSet const &) = delete;
    UsdGenWriteSet &operator=(UsdGenWriteSet const &) = delete;

    /// Values (items * arity) of a slot in this set.
    USDGEN_API size_t ValueCount(UsdGenSlot) const noexcept;

private:
    struct Impl;
    explicit UsdGenWriteSet(Impl *impl) noexcept : _impl(impl) {}
    USDGEN_API void *WriteRaw(UsdGenSlot, uint8_t sample);
    USDGEN_API size_t ValueBytes(UsdGenSlot) const noexcept;
    Impl *_impl;                                 // pool-allocated
};

}  // namespace usdGen
