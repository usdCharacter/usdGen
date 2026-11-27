// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Host and engine values, converted explicitly and table-driven over the
// value-type table's rows. ToEngineValue keeps the held C++ type's row
// (double stays double, int stays int; Gf math becomes usdGenMath, TfToken a
// UsdGenToken, a VtArray a UsdGenCowArray (zero copy), a string or asset
// array a std::vector, SdfAssetPath / SdfPathExpression / SdfOpaqueValue the
// engine rows). A host value with no row is boxed: it stays non-empty,
// hashes and compares as itself, and ToHostValue gives it back unchanged.
// Every value the engine reads goes through ToEngineValue: IsHolding is exact.
#pragma once
#include "usdGen/api/value.h"
#include "usdGenUsd/export.h"
#include "usdGenUsd/convert/array.h"
#include "usdGenUsd/convert/math.h"

#include "pxr/base/vt/value.h"

namespace usdGenUsd {

USDGEN_USD_API usdGen::UsdGenValue ToEngineValue(PXR_NS::VtValue const &v);
USDGEN_USD_API PXR_NS::VtValue ToHostValue(usdGen::UsdGenValue const &v);

}  // namespace usdGenUsd
