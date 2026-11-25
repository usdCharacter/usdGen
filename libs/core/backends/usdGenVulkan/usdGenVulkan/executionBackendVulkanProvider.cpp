// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Small capability-version bridge for the neutral backend contract.
//
// Mirrors executionBackendCudaProvider.cpp: the neutral registry must not
// include a concrete native adapter header. The host Vulkan backend pins
// capability version 1; the executor refuses any other plan version.
#include <cstdint>

namespace usdGen {

uint32_t UsdGenVulkanCapabilityVersionForBackendContract() noexcept
{
    return 1;
}

} // namespace usdGen
