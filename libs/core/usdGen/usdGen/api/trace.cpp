// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/trace.h"

#include <atomic>

namespace usdGen {
namespace {
std::atomic<UsdGenTraceSink const *> gTraceSink{nullptr};
}  // namespace

void UsdGenSetTraceSink(UsdGenTraceSink const *sink) noexcept
{
    gTraceSink.store(sink, std::memory_order_release);
}

UsdGenTraceSink const *UsdGenGetTraceSink() noexcept
{
    return gTraceSink.load(std::memory_order_acquire);
}

}  // namespace usdGen
