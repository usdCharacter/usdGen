// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Production default Vulkan Session provider.
//
// This header is deliberately Vulkan-header-free so the neutral Session
// executor (compiled into libusdGen.so) can call it without importing SDK
// types. The implementation lives in defaultProvider.cpp alongside the rest
// of the Vulkan runtime.
#ifndef USDGEN_VULKAN_DEFAULT_PROVIDER_H
#define USDGEN_VULKAN_DEFAULT_PROVIDER_H

#include <memory>
#include <string>

namespace usdGen {

class UsdGenSessionDeviceProvider;

namespace vulkan {

// Creates a production provider owning its own instance, logical device,
// pipelines (from installed or build-tree SPIR-V), completion service and
// plan executor. Returns null with a precise reason when no suitable
// compute device exists or shader/pipeline bring-up fails; callers must
// fail closed with that reason, never fall back to CPU execution.
std::shared_ptr<UsdGenSessionDeviceProvider> CreateDefaultVulkanSessionProvider(
    std::string* reason = nullptr);

// Lightweight enumeration probe: true when at least one suitable Vulkan
// 1.2+ compute device (discrete or integrated GPU, timeline semaphores)
// is enumerable. Creates and tears down a transient instance; owns nothing.
bool HasDefaultVulkanDevice() noexcept;

} // namespace vulkan
} // namespace usdGen

#endif // USDGEN_VULKAN_DEFAULT_PROVIDER_H
