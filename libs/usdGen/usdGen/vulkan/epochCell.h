// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// The Vulkan backend's name for usdGen/sharedCell.h. New code uses
// UsdGenSharedCell (or UsdGenAtomicRef) directly.
#ifndef USDGEN_VULKAN_EPOCH_CELL_H
#define USDGEN_VULKAN_EPOCH_CELL_H

#include "usdGen/sharedCell.h"

namespace usdGen::vulkan {
template <class T>
using VulkanEpochCell = UsdGenSharedCell<T>;
} // namespace usdGen::vulkan
#endif
