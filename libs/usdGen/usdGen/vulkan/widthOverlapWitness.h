// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Test-only overlap probe ABI. Vulkan port of the gpu/widthOverlapWitness.h
// contract: one thread records simultaneous probes and waits for the
// expected arrivals, but only until a caller-bounded deadline so a serial
// queue cannot deadlock. The CUDA kernel takes its bound in device clocks;
// this module takes it as a spin-iteration budget (see widthOverlapWitness
// .comp), which keeps the probe portable across Vulkan devices.
#ifndef USDGEN_VULKAN_WIDTH_OVERLAP_WITNESS_H
#define USDGEN_VULKAN_WIDTH_OVERLAP_WITNESS_H

#include <cstdint>

namespace usdGen::vulkan {

struct VulkanWidthOverlapWitnessControls {
    uint32_t expected = 0;        // rendezvous arrivals, must be >= 2
    uint32_t dwellIterations = 0;  // spin budget (register-hash steps), must be nonzero
};
static_assert(sizeof(VulkanWidthOverlapWitnessControls) == 8,
              "witness push-constant ABI");

// Descriptor set 0, binding 0: three uint32 counters, all zeroed before the
// first probe. counters[0] is the live count, counters[1] the maximum
// observed live count, and counters[2] the total arrivals.
constexpr uint32_t kVulkanWidthOverlapWitnessCounters = 3;
constexpr uint32_t kVulkanWidthOverlapWitnessActive = 0;
constexpr uint32_t kVulkanWidthOverlapWitnessMaxActive = 1;
constexpr uint32_t kVulkanWidthOverlapWitnessArrivals = 2;

// Host-side admission, mirroring LaunchWidthOverlapWitness: at least three
// counters, at least two expected arrivals, and a nonzero dwell budget.
inline bool ValidateVulkanWidthOverlapWitnessArgs(uint64_t counterUints,
                                                 uint32_t expected,
                                                 uint32_t dwellIterations) noexcept {
    return counterUints >= kVulkanWidthOverlapWitnessCounters &&
        expected >= 2u && dwellIterations != 0u;
}

} // namespace usdGen::vulkan

#endif // USDGEN_VULKAN_WIDTH_OVERLAP_WITNESS_H
