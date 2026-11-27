// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenStatus (a result code and message) and UsdGenReport, the engine's
// diagnostics door. Hosts install a sink; until one does, reports go to
// Tf's diagnostics (alias phase, step 11.1).
#pragma once
#include "usdGen/api/export.h"
#include "usdGen/api/text.h"

#include <cstdint>
#include <string_view>

#if defined(__GNUC__) || defined(__clang__)
#  define USDGEN_PRINTF_FORMAT(fmt, first) __attribute__((format(printf, fmt, first)))
#else
#  define USDGEN_PRINTF_FORMAT(fmt, first)
#endif

namespace usdGen {

enum class UsdGenStatusCode : uint8_t { Ok, InvalidArgument, Unsupported, Failed, Cancelled, OutOfMemory, DeviceLost };

class UsdGenStatus {
public:
    UsdGenStatus() noexcept = default;                       // Ok
    static UsdGenStatus Error(UsdGenStatusCode code, std::string_view message)
    {
        UsdGenStatus s;
        s._code = code;
        s._message = UsdGenText(message);
        return s;
    }
    bool ok() const noexcept { return _code == UsdGenStatusCode::Ok; }
    UsdGenStatusCode code() const noexcept { return _code; }
    std::string_view message() const noexcept { return _message.View(); }

private:
    UsdGenStatusCode _code = UsdGenStatusCode::Ok;
    UsdGenText _message;
};

enum class UsdGenSeverity : uint8_t { Status, Warning, Error, CodingError };
using UsdGenReportFn = void (*)(void *user, UsdGenSeverity, char const *message) noexcept;
/// Set before cooking; null restores the default.
USDGEN_CORE_API void UsdGenSetReportSink(UsdGenReportFn fn, void *user) noexcept;
/// printf-style; replaces TF_STATUS, TF_WARN, TF_RUNTIME_ERROR, TF_CODING_ERROR.
USDGEN_CORE_API void UsdGenReport(UsdGenSeverity s, char const *fmt, ...) noexcept USDGEN_PRINTF_FORMAT(2, 3);

}  // namespace usdGen
