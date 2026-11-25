// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Bridge (5.3a; 5.4 deletes): element-set views over builder buffers, so the
// scheduler hands ops set views while it still stores buffers.
#pragma once
#include "usdGen/api/elementSet.h"
#include "usdGen/api/export.h"
#include "usdGen/exec/pool.h"
#include "usdGenCurves/types/curveBuffer.h"

#include <cstdint>

namespace usdGen {

/// A UsdGenElementSetView of the CurveSet layout over a UsdGenCurveBuffer.
/// Columns alias the buffer's arrays; view.legacy points at the buffer.
class UsdGenCurveBufferSetView
{
public:
    /// Points the view at 'b'. Rebuilds the channel table only when the
    /// buffer's plane layout changed (allocates then); otherwise it
    /// refreshes pointers and counts and allocates nothing.
    USDGEN_CORE_API void Bind(UsdGenCurveBuffer const &b);
    UsdGenElementSetView const &View() const noexcept { return _view; }
    UsdGenChannelTable const *Channels() const noexcept { return _table.Get(); }

private:
    UsdGenRef<const UsdGenChannelTable> _table;
    UsdGenPoolVector<UsdGenChannelDesc> _dynamic;   // extraCurve then extraCv
    UsdGenPoolVector<UsdGenSlot> _planeSlots;       // same order: the table slot
    UsdGenPoolVector<void const *> _columns;
    UsdGenPoolVector<uint32_t> _values;
    UsdGenElementSetView _view;
};

}  // namespace usdGen
