// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen - small helpers for ops that read their parameters by slot.
#ifndef USDGEN_OPS_RECORDED_OP_H
#define USDGEN_OPS_RECORDED_OP_H

#include "usdGen/op.h"

namespace usdGen {

/// A copy of the slot's authored value, empty when unauthored.
inline UsdGenValue UsdGenAuthoredValue(UsdGenParamView const &p, UsdGenParamSlot s)
{
    UsdGenResolvedParam const *r = p.Slot(s);
    return r && r->authored ? r->authored->value : UsdGenValue();
}

}  // namespace usdGen

#endif  // USDGEN_OPS_RECORDED_OP_H
