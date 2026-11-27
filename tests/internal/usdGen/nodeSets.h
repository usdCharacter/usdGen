// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Test helpers: a compiled node's output set (5.4) as a view (column
// identity, no copy) or as a builder buffer (a copy, VtArray semantics), and
// a builder buffer installed as a node's output (fixtures that used to
// assign node.buffer).
#pragma once
#include "usdGen/api/writeSet.h"
#include "usdGen/compile/graph.h"
#include "usdGenCurves/types/curveBuffer.h"
#include "usdGenCurves/types/curveSet.h"

#include <utility>

namespace usdGen {

inline UsdGenCurveSetView NodeView(UsdGenGraph const &g, UsdGenNodeId id)
{
    return UsdGenCurveSetView::Of(g.NodeOutput(id));
}

inline UsdGenCurveSetView OutputView(UsdGenGraph const &g)
{
    return UsdGenCurveSetView::Of(g.Output());
}

inline UsdGenCurveBuffer NodeBuffer(UsdGenGraph const &g, UsdGenNodeId id)
{
    return UsdGenCurveBufferFromSet(g.NodeOutput(id));
}

inline UsdGenCurveBuffer OutputBuffer(UsdGenGraph const &g)
{
    return UsdGenCurveBufferFromSet(g.Output());
}

/// Replaces node 'id''s output with 'b', stamped with b's versions exactly.
/// False when 'b' is not a consistent CurveSet.
inline bool SetNodeBuffer(UsdGenGraph &g, UsdGenNodeId id, UsdGenCurveBuffer const &b)
{
    UsdGenRef<const UsdGenElementSet> set = UsdGenCurveSetFromBuffer(b, 1, 1);
    if (!set) return false;
    UsdGenWriteSet ws = UsdGenWriteSet::Derive(set, {});
    ws.Plan({});
    ws.SetVersionsExact(b.topologyVersion, b.valueVersion);
    UsdGenCompiledNode &n = g.Node(id);
    n.output = std::move(ws).Seal();
    n.topologyVersion = b.topologyVersion;
    n.valueVersion = b.valueVersion;
    return true;
}

}  // namespace usdGen
