// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The engine's one text -> UsdGenToken door, with intern counters.
#pragma once
#include "usdGen/api/export.h"
#include "usdGen/api/token.h"
#include <cstdint>
#include <string_view>

namespace usdGen {

// UsdGenInternToken (api/token.h) counts every call.

/// Interns made after UsdGenFreezeRegistries() outside any compile scope.
USDGEN_API uint64_t UsdGenExecInternCount() noexcept;
/// Every intern, compile scope included.
USDGEN_API uint64_t UsdGenTotalInternCount() noexcept;
/// False only when USDGEN_TOKEN_CHECKS=0 (read once at registry creation).
USDGEN_API bool UsdGenTokenChecksEnabled() noexcept;

/// RAII, this thread: interns while alive count as compile interns.
class USDGEN_API UsdGenCompileTokenScope {
public:
    UsdGenCompileTokenScope() noexcept;
    ~UsdGenCompileTokenScope();
    UsdGenCompileTokenScope(UsdGenCompileTokenScope const &) = delete;
    UsdGenCompileTokenScope &operator=(UsdGenCompileTokenScope const &) = delete;
};

/// Brackets one exec phase. Violated(): some thread made an exec intern
/// since construction. Two relaxed loads, no allocation.
class UsdGenCookTokenCheck {
public:
    UsdGenCookTokenCheck() noexcept : _start(UsdGenExecInternCount()) {}
    bool Violated() const noexcept { return UsdGenExecInternCount() != _start; }
private:
    uint64_t _start;
};

}  // namespace usdGen
