#include "widthBlend.h"

#include <cmath>
#include <limits>

namespace usdGen::gpu {
namespace {

__global__ void WidthBlendKernel(float const* left, float const* right,
                                 float blend, float* output, size_t count) {
    size_t const index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    // Keep the endpoints bit-exact and do not rely on FMA contraction for
    // the public ordered-input contract.
    if (blend == 0.0f) output[index] = left[index];
    else if (blend == 1.0f) output[index] = right[index];
    else output[index] = left[index] + (right[index] - left[index]) * blend;
}

} // namespace

bool LaunchWidthBlend(DeviceView<const float> left,
                      DeviceView<const float> right,
                      float blend,
                      DeviceView<float> output,
                      cudaStream_t stream) {
    if (!std::isfinite(blend) || blend < 0.0f || blend > 1.0f ||
        left.size != right.size || left.size != output.size ||
        (left.size && (!left.data || !right.data || !output.data)) || !stream)
        return false;
    if (!left.size) return true;
    constexpr size_t kThreads = 256u;
    size_t const blocks = 1u + (left.size - 1u) / kThreads;
    if (blocks > std::numeric_limits<unsigned int>::max()) return false;
    WidthBlendKernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
        left.data, right.data, blend, output.data, left.size);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace usdGen::gpu
