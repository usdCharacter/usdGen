#ifndef USDGEN_GPU_NOISE_H
#define USDGEN_GPU_NOISE_H

#include "width.h"
#include "rootFrames.h"

#include <cstdint>

namespace usdGen::gpu {

// Integer parameter fields use the same explicit evaluation-domain contract
// as ScalarField and BoolField.  They are intentionally distinct from float
// controls: CUDA Noise never coerces an expression result to an octave count.
struct IntField {
    const int32_t* data = nullptr;
    size_t count = 0;
    expr::Domain domain = expr::Domain::Groom;
    int32_t literal = 1;

    static IntField Literal(int32_t value) {
        IntField result;
        result.literal = value;
        return result;
    }
    static IntField Device(DeviceView<const int32_t> values, expr::Domain d) {
        return {values.data, values.size, d, 1};
    }
};

// Full device-side controls for the v1 Noise styler.  Scalar controls may be
// groom-, primitive-, or point-domain fields. `enabled` is groom-only (the
// inherited operator toggle), while cumulative is groom/primitive so one
// strand has one well-defined accumulation policy. `octaves` is typed int;
// no float-to-int conversion is permitted.  `magnitudeProfile` contains 257
// samples at canonical hairT.  `mask` IS the operator envelope (usdGen:mask):
// a groom-, primitive- or point-domain field in [0,1]; exactly 0 is a bitwise
// pass-through.
struct NoiseParameters {
    ScalarField magnitude = ScalarField::Literal(.05f);
    ScalarField frequency = ScalarField::Literal(3.0f);
    ScalarField correlation = ScalarField::Literal(.5f);
    IntField octaves = IntField::Literal(1);
    ScalarField lacunarity = ScalarField::Literal(2.0f);
    ScalarField gain = ScalarField::Literal(.5f);
    ScalarField preserveLength = ScalarField::Literal(1.0f);
    ScalarField mask = ScalarField::Literal(1.0f);
    BoolField enabled = BoolField::Literal(true);
    BoolField cumulative = BoolField::Literal(false);
    // One stable-id hash is drawn per curve, so seed may be groom or
    // primitive but is deliberately never point-domain.
    IntField seed = IntField::Literal(0);
    DeviceView<const float> magnitudeProfile; // exactly 257 entries
};

// Device-only rest-frame SeExpr vfbm Noise.  Apply stages all points privately
// and Finish publishes them only after one scalar status readback succeeds.
// Inputs, frame views, controls, and output remain borrowed through Finish.
// Fractional preserveLength blends uncorrected displacement toward a
// sequential root-anchored restoration using authored rest segment lengths;
// 0 is no restoration and 1 restores every nondegenerate segment exactly.
class CudaNoise {
public:
    CudaNoise() = default;
    ~CudaNoise();
    CudaNoise(CudaNoise const&) = delete;
    CudaNoise& operator=(CudaNoise const&) = delete;

    StyleStatus Apply(DeviceCurveGeometryView geometry,
                      DeviceView<const float> hairT,
                      RestRootFrames restFrames,
                      NoiseParameters parameters,
                      DeviceView<float3> output,
                      cudaStream_t stream,
                      UsdGenExecutionMemoryReservation* reservation = nullptr);
    StyleStatus Finish(cudaStream_t stream);
    // Fresh-only nonblocking publication. ApplyFresh writes only to the
    // caller-provided private point plane. FinishFreshAsync publishes that
    // plane after a scalar device-status readback and invokes the native
    // callback; CommitFreshFinish is host-only and must follow a proved
    // cudaSuccess callback. No generated point data is copied to the host.
    StyleStatus ApplyFresh(DeviceCurveGeometryView geometry,
                           DeviceView<const float> hairT,
                           RestRootFrames restFrames,
                           NoiseParameters parameters,
                           DeviceView<float3> output,
                           cudaStream_t stream,
                           UsdGenExecutionMemoryReservation* reservation = nullptr);
    StyleStatus FinishFreshAsync(cudaStream_t stream,
        void (*callback)(cudaStream_t, cudaError_t, void*) noexcept, void* userdata);
    StyleStatus CommitFreshFinish();
    bool HasUnprovenWork() const noexcept { return unprovenWork_; }
    bool pending() const { return pending_; }
    int deviceIndex() const { return deviceIndex_; }

private:
    StyleStatus validateScalar(ScalarField field,
                               DeviceCurveGeometryView geometry) const;
    StyleStatus validateBool(BoolField field, DeviceCurveGeometryView geometry,
                             bool groomOnly, bool allowPrimitive) const;
    StyleStatus validateInt(IntField field, DeviceCurveGeometryView geometry) const;
    StyleStatus begin(DeviceCurveGeometryView geometry,
                      DeviceView<float3> output, cudaStream_t stream,
                      UsdGenExecutionMemoryReservation* reservation);
    StyleStatus finishPublication(cudaStream_t stream);

    DeviceBuffer<int> error_;
    DeviceBuffer<float3> staging_;
    cudaEvent_t ready_ = nullptr;
    DeviceView<float3> output_{};
    size_t pointCount_ = 0;
    int deviceIndex_ = -1;
    bool pending_ = false;
    int* freshHostError_ = nullptr;
    UsdGenExecutionResourcePermit freshHostErrorPermit_;
    bool freshPreparing_ = false;
    bool freshPending_ = false;
    bool freshCallbackArmed_ = false;
    bool freshUploadFailed_ = false;
    bool unprovenWork_ = false;
};

} // namespace usdGen::gpu

#endif // USDGEN_GPU_NOISE_H
