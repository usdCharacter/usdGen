#ifndef USDGEN_GPU_WIDTH_BLEND_H
#define USDGEN_GPU_WIDTH_BLEND_H

#include "curveGeometry.h"

namespace usdGen::gpu {

// Ordered, topology-preserving width fan-in.  This is deliberately a small
// leaf primitive: callers own the fresh output and its completion fence.  It
// never aliases either immutable predecessor plane.
bool LaunchWidthBlend(DeviceView<const float> left,
                      DeviceView<const float> right,
                      float blend,
                      DeviceView<float> output,
                      cudaStream_t stream);

} // namespace usdGen::gpu

#endif // USDGEN_GPU_WIDTH_BLEND_H
