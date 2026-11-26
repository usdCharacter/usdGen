// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// In-code graphs for the output goldens and the op-contract snapshot.
// Pure descriptors: no stage, no schema, no randomness.
#ifndef USDGEN_TESTS_GOLDEN_GRAPHS_H
#define USDGEN_TESTS_GOLDEN_GRAPHS_H

#include "usdGenUsd/lower.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"
#include "pxr/usd/sdf/path.h"

#include <string>
#include <vector>

namespace usdGenGolden {

/// Path of the op under test in every MakeOpGraph descriptor.
PXR_NS::SdfPath const &OpPath();

/// The op of `type` fed by the smallest source that lets it cook (a
/// CurveSource, or Scatter/Grow for ops that need a bound surface), with
/// its required inputs. `tuned` adds the hand-picked non-default params the
/// goldens cook; false leaves every optional param unauthored. Empty
/// description when the type has no fixture.
usdGen::UsdGenGraphDesc MakeOpGraph(PXR_NS::TfToken const &type, bool tuned);

struct GoldenGraph
{
    std::string name;
    usdGen::UsdGenGraphDesc desc;
};

/// Every tuned op graph plus a few multi-op chains, in a fixed order.
std::vector<GoldenGraph> MakeGoldenGraphs();

/// Every op type MakeOpGraph knows, sorted by name.
std::vector<PXR_NS::TfToken> FixtureTypes();

}  // namespace usdGenGolden

#endif
