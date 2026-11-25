#ifndef USDGEN_GPU_STYLE_OPS_H
#define USDGEN_GPU_STYLE_OPS_H

#include "curveGeometry.h"
#include "usdGen/expressions/context.h"

#include <cstdint>

namespace usdGen::gpu {

// A borrowed, scalar float field. The sample count is the number of values in
// the declared domain (one for Groom, curveCount for Primitive, and pointCount
// for Point). A null data pointer with count zero denotes the literal member;
// construct those fields with ScalarField::Literal so a scalar control remains
// device-only.
struct ScalarField {
    const float *data = nullptr;
    size_t count = 0;
    expr::Domain domain = expr::Domain::Groom;
    float literal = 0.0f;

    static ScalarField Literal(float value) {
        ScalarField result;
        result.literal = value;
        return result;
    }
    static ScalarField Device(DeviceView<const float> values, expr::Domain d) {
        return {values.data, values.size, d, 0.0f};
    }
};

enum class StyleStatus {
    Ok,
    InvalidArgument,
    NonFiniteInput,
    InvalidValue,
    NotSupported,
    CudaError,
    Unsupported = NotSupported
};

// Device-only, topology-preserving style primitives. Inputs and outputs are
// borrowed until Finish returns. Each call queues one operation; callers must
// call Finish (on any stream) before publishing or reusing its output. Finish
// performs only a scalar diagnostic readback. A supplied hairT is a point-
// domain parameter in [0,1]; when omitted, t is inclusive CV-index
// interpolation (0 at the root and 1 at the tip, and 0 for a one-CV curve).
class CudaStyleOps {
public:
    CudaStyleOps() = default;
    ~CudaStyleOps();
    CudaStyleOps(const CudaStyleOps&) = delete;
    CudaStyleOps& operator=(const CudaStyleOps&) = delete;

    StyleStatus WidthRamp(DeviceCurveGeometryView geometry,
                          ScalarField root, ScalarField tip,
                          DeviceView<float> output, cudaStream_t stream,
                          DeviceView<const float> hairT = {});
    StyleStatus WidthRamp(DeviceCurveGeometryView geometry,
                          ScalarField root, ScalarField tip,
                          DeviceView<const float> hairT,
                          DeviceView<float> output, cudaStream_t stream);
    StyleStatus WidthRamp(DeviceCurveGeometryView geometry, float root, float tip,
                          DeviceView<float> output, cudaStream_t stream,
                          DeviceView<const float> hairT = {});

    // Length is root anchored: output[i] = root + (points[i]-root)*scale.
    StyleStatus Length(DeviceCurveGeometryView geometry, ScalarField scale,
                       DeviceView<float3> output, cudaStream_t stream);
    StyleStatus Length(DeviceCurveGeometryView geometry, float scale,
                       DeviceView<float3> output, cudaStream_t stream);

    // Grow uses supplied per-curve roots and normals. The normal is normalized
    // on the device and output = root + normal * length * t.
    StyleStatus Grow(DeviceCurveGeometryView geometry,
                     DeviceView<const float3> roots,
                     DeviceView<const float3> normals, ScalarField length,
                     DeviceView<float3> output, cudaStream_t stream,
                     DeviceView<const float> hairT = {});
    StyleStatus Grow(DeviceCurveGeometryView geometry,
                     DeviceView<const float3> roots,
                     DeviceView<const float3> normals, ScalarField length,
                     DeviceView<const float> hairT,
                     DeviceView<float3> output, cudaStream_t stream);
    StyleStatus Grow(DeviceCurveGeometryView geometry,
                     DeviceView<const float3> roots,
                     DeviceView<const float3> normals, float length,
                     DeviceView<float3> output, cudaStream_t stream,
                     DeviceView<const float> hairT = {});

    // Noise is deliberately not exposed as a fake hash/FBM implementation.
    // The complete planned operator needs explicit rest-noise base, frame,
    // hairT, and exact FBM controls; this foundation reports NotSupported.
    StyleStatus Noise(DeviceCurveGeometryView geometry, float amplitude,
                      uint32_t seed, DeviceView<float3> output,
                      cudaStream_t stream);

    StyleStatus Finish(cudaStream_t stream);
    bool pending() const { return pending_; }

private:
    StyleStatus begin(DeviceCurveGeometryView geometry, size_t outputSize,
                      const void *output, cudaStream_t stream,
                      DeviceView<const float> hairT,
                      bool vectorOutput, bool requirePoints);
    StyleStatus validateField(ScalarField field, DeviceCurveGeometryView geometry) const;
    StyleStatus finishPublication(cudaStream_t stream);

    DeviceBuffer<int> error_;
    DeviceBuffer<float> scalarStaging_;
    DeviceBuffer<float3> vectorStaging_;
    cudaEvent_t ready_ = nullptr;
    DeviceView<float> scalarOutput_{};
    DeviceView<float3> vectorOutput_{};
    size_t pointCount_ = 0;
    bool vectorPending_ = false;
    bool pending_ = false;
};

} // namespace usdGen::gpu

#endif
