// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Slot reads shared by the G5 stylers.
#ifndef USDGEN_OP_STYLER_PARAMS_H
#define USDGEN_OP_STYLER_PARAMS_H

#include "usdGen/registries/op.h"
#include "usdGen/api/value.h"
#include "usdGen/api/token.h"

namespace usdGen {

/// Decl enumerant index of an authored token param; fallback when unauthored
/// or not a token. -1 when authored with a value that is not an enumerant;
/// *unknown then points at the authored token.
inline int UsdGenStylerEnum(UsdGenParamView const &p, UsdGenParamSlot s, int fallback,
                            UsdGenToken const **unknown = nullptr) noexcept
{
    UsdGenResolvedParam const *r = p.Slot(s);
    if (!r || !r->authored) return fallback;
    UsdGenValue const &v = r->authored->value;
    if (!v.IsHolding<UsdGenToken>() && !v.IsHolding<std::string>()) return fallback;
    if (r->enumIndex < 0 && unknown) *unknown = &r->token;
    return r->enumIndex;
}

}  // namespace usdGen

#endif  // USDGEN_OP_STYLER_PARAMS_H
