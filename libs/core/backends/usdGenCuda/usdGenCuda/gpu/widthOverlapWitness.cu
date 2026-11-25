#include "widthOverlapWitness.h"

namespace usdGen::gpu {
namespace {

__global__ void
_WidthOverlapWitness(uint32_t* counters, uint32_t expected,
                     uint64_t dwellCycles) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    auto const active = atomicAdd(&counters[0], 1u) + 1u;
    atomicMax(&counters[1], active);
    atomicAdd(&counters[2], 1u);
    // This is a bounded rendezvous, never an unbounded device-side gate. A
    // concurrent peer releases both probes promptly; a serial stream reaches
    // the deadline and observes maxActive of one without stranding a block.
    auto const start = clock64();
    while (atomicAdd(&counters[2], 0u) < expected &&
           clock64() - start < dwellCycles) asm volatile("");
    atomicSub(&counters[0], 1u);
}

} // namespace

bool LaunchWidthOverlapWitness(DeviceView<uint32_t> counters,
                               uint32_t expected,
                               uint64_t dwellCycles,
                               cudaStream_t stream) noexcept {
    if (!counters.data || counters.size < 3 || expected < 2 ||
        !dwellCycles) return false;
    _WidthOverlapWitness<<<1, 1, 0, stream>>>(
        counters.data, expected, dwellCycles);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace usdGen::gpu
