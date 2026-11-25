// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/registries/token.h"

#include <atomic>
#include <cstdlib>
#include <string>

namespace usdGen {

bool UsdGenOpRegistryFrozenIfBuilt() noexcept;   // opRegistry.cpp

namespace {
std::atomic<uint64_t> gTotalInterns{0};
std::atomic<uint64_t> gExecInterns{0};
std::atomic<bool> gTokenChecks{true};
thread_local int tCompileScopeDepth = 0;
}  // namespace

// Called once by the op registry constructor.
void UsdGenReadTokenCheckSetting() noexcept
{
    char const *value = std::getenv("USDGEN_TOKEN_CHECKS");
    gTokenChecks.store(!(value && value[0] == '0' && value[1] == '\0'), std::memory_order_relaxed);
}

TfToken UsdGenInternToken(std::string_view text)
{
    gTotalInterns.fetch_add(1, std::memory_order_relaxed);
    if (tCompileScopeDepth == 0 && UsdGenOpRegistryFrozenIfBuilt())
        gExecInterns.fetch_add(1, std::memory_order_relaxed);
    return TfToken(std::string(text));
}

uint64_t UsdGenExecInternCount() noexcept
{
    return gExecInterns.load(std::memory_order_relaxed);
}

uint64_t UsdGenTotalInternCount() noexcept
{
    return gTotalInterns.load(std::memory_order_relaxed);
}

bool UsdGenTokenChecksEnabled() noexcept
{
    return gTokenChecks.load(std::memory_order_relaxed);
}

UsdGenCompileTokenScope::UsdGenCompileTokenScope() noexcept { ++tCompileScopeDepth; }
UsdGenCompileTokenScope::~UsdGenCompileTokenScope() { --tCompileScopeDepth; }

}  // namespace usdGen
