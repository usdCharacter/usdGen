#ifndef USDGEN_GPU_WIDTH_OVERLAP_WITNESS_H
#define USDGEN_GPU_WIDTH_OVERLAP_WITNESS_H

#include "deviceBuffer.h"

#include <cstdint>

namespace usdGen::gpu {

// Test-only stream probe used by the CUDA execution layer. One thread records
// simultaneous probes and waits for the expected arrivals, but only until a
// caller-bounded device-clock deadline so a serial stream cannot deadlock.
bool LaunchWidthOverlapWitness(DeviceView<uint32_t> counters,
                               uint32_t expected,
                               uint64_t dwellCycles,
                               cudaStream_t stream) noexcept;

} // namespace usdGen::gpu

#endif
