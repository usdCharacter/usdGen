// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen imaging — header-safe static tokens. A TfStaticData is zero-
// initialized and built on first use by CAS: no guarded function static and
// no static destructor.
#ifndef USDGEN_IMAGING_STATIC_TOKEN_H
#define USDGEN_IMAGING_STATIC_TOKEN_H

#include "pxr/pxr.h"
#include "pxr/base/tf/staticData.h"
#include "pxr/base/tf/token.h"

namespace usdGenImaging {

/// TfStaticData factory for a token spelled `Text`.
template <char const *Text>
struct UsdGenTokenFactory {
    static PXR_NS::TfToken *New() { return new PXR_NS::TfToken(Text); }
};
template <char const *Text>
using UsdGenStaticToken = PXR_NS::TfStaticData<PXR_NS::TfToken, UsdGenTokenFactory<Text>>;

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_STATIC_TOKEN_H
