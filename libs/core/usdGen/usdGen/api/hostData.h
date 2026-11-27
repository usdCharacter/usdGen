// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Opaque host data a spec carries into its generations and routing
// snapshots. The engine copies the reference and never reads it.
#pragma once
#include "usdGen/api/export.h"
#include "usdGen/api/epoch.h"

namespace usdGen {

/// A host derives its own; the last release deletes it through its own
/// virtual destructor.
class USDGEN_API UsdGenHostData : public UsdGenRefCounted {
protected:
    UsdGenHostData() noexcept = default;
    ~UsdGenHostData() override = default;
};
using UsdGenHostDataRef = UsdGenRef<const UsdGenHostData>;

}  // namespace usdGen
