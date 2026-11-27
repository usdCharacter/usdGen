// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/ext/device/sessionDeviceProvider.h"

namespace usdGen {

// Key function: one typeinfo, in usdGen, for every provider DSO.
UsdGenSessionDeviceProvider::~UsdGenSessionDeviceProvider() = default;

bool UsdGenSessionDeviceProvider::Select(int, UsdGenNodeKey,
    UsdGenSessionDeviceIdentity *identity, UsdGenDiagnostics *)
{
    if (identity) *identity = Identity();
    return true;
}

}  // namespace usdGen
