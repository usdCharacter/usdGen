// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/debug.h"
#include "usdGen/exec/debugCodes.h"

#include "pxr/base/tf/stringUtils.h"

#include <cstdarg>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

bool UsdGenDebugEnabled(UsdGenDebugCode c) noexcept
{
    switch (c) {
    case UsdGenDebugCode::Commit: return TfDebug::IsEnabled(USDGEN_COMMIT);
    case UsdGenDebugCode::Schedule: return TfDebug::IsEnabled(USDGEN_SCHEDULE);
    case UsdGenDebugCode::Ingress: return TfDebug::IsEnabled(USDGEN_INGRESS);
    case UsdGenDebugCode::Fur: return TfDebug::IsEnabled(USDGEN_FUR);
    case UsdGenDebugCode::Count: break;
    }
    return false;
}

void UsdGenSetDebugEnabled(UsdGenDebugCode c, bool on) noexcept
{
    switch (c) {
    case UsdGenDebugCode::Commit: on ? TfDebug::Enable(USDGEN_COMMIT) : TfDebug::Disable(USDGEN_COMMIT); break;
    case UsdGenDebugCode::Schedule: on ? TfDebug::Enable(USDGEN_SCHEDULE) : TfDebug::Disable(USDGEN_SCHEDULE); break;
    case UsdGenDebugCode::Ingress: on ? TfDebug::Enable(USDGEN_INGRESS) : TfDebug::Disable(USDGEN_INGRESS); break;
    case UsdGenDebugCode::Fur: on ? TfDebug::Enable(USDGEN_FUR) : TfDebug::Disable(USDGEN_FUR); break;
    case UsdGenDebugCode::Count: break;
    }
}

void UsdGenDebugMsg(char const *fmt, ...) noexcept
{
    va_list ap;
    va_start(ap, fmt);
    try {
        TfDebug::Helper::Msg(TfVStringPrintf(fmt, ap));
    } catch (...) {
    }
    va_end(ap);
}

}  // namespace usdGen
