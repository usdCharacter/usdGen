#ifndef USDGEN_GPU_LENGTH_H
#define USDGEN_GPU_LENGTH_H

#include "width.h"

#include <string>

namespace usdGen::gpu {

enum class LengthMode { Set, Scale, Cull };
enum class LengthMethod { Scale, CutExtend };
enum class LengthRebuild { KeepParam, Reparam };

// float2 carries an alignment specifier, so the layout pads after domain;
// MSVC reports that as C4324 even though it is the intended layout.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
struct Vec2Field {
    const float2* data = nullptr;
    size_t count = 0;
    expr::Domain domain = expr::Domain::Groom;
    float2 literal = make_float2(1.0f, 1.0f);
    static Vec2Field Literal(float2 v) { Vec2Field r; r.literal = v; return r; }
    static Vec2Field Device(DeviceView<const float2> v, expr::Domain d) {
        return {v.data, v.size, d, make_float2(1, 1)};
    }
};
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

struct LengthParameters {
    LengthMode mode = LengthMode::Scale;
    LengthMethod method = LengthMethod::Scale;
    LengthRebuild rebuild = LengthRebuild::KeepParam;
    ScalarField value = ScalarField::Literal(1.0f);
    Vec2Field random = Vec2Field::Literal(make_float2(1, 1));
    // usdGen:mask IS the operator envelope; exactly 0 is a pass-through.
    ScalarField mask = ScalarField::Literal(1.0f);
    ScalarField minRemainingLength = ScalarField::Literal(0.0f);
    ScalarField cullThreshold = ScalarField::Literal(0.0f);
    BoolField enabled = BoolField::Literal(true);
    int seed = 0;
};

class CudaLength {
public:
    CudaLength() = default;
    ~CudaLength();
    CudaLength(CudaLength const&) = delete;
    CudaLength& operator=(CudaLength const&) = delete;
    StyleStatus Apply(DeviceCurveGeometryView geometry,
                      DeviceView<const float> hairT,
                      LengthParameters parameters,
                      DeviceView<float3> output,
                      DeviceView<uint8_t> keep,
                      cudaStream_t stream,
                      UsdGenExecutionMemoryReservation* reservation = nullptr);
    StyleStatus Finish(cudaStream_t stream);
    // Fresh-only nonblocking publication.  ApplyFresh rejects capture before
    // device/stream queries or allocation.  FinishFreshAsync conditionally
    // publishes both points and keep only after device validation, copies the
    // scalar status to owned pinned storage, then invokes the caller callback.
    // CommitFreshFinish is host-only and requires the caller relay to prove
    // native cudaSuccess and launcher return.
    StyleStatus ApplyFresh(DeviceCurveGeometryView geometry,
                           DeviceView<const float> hairT,
                           LengthParameters parameters,
                           DeviceView<float3> output,
                           DeviceView<uint8_t> keep,
                           cudaStream_t stream,
                           UsdGenExecutionMemoryReservation* reservation = nullptr);
    StyleStatus FinishFreshAsync(cudaStream_t stream,
        void (*callback)(cudaStream_t, cudaError_t, void*) noexcept, void* userdata);
    StyleStatus CommitFreshFinish();
    bool HasUnprovenWork() const noexcept { return unprovenWork_; }
    bool pending() const { return pending_; }
    int deviceIndex() const { return deviceIndex_; }
    const char *diagnostic() const { return diagnostic_.c_str(); }
private:
    StyleStatus validateField(ScalarField f, DeviceCurveGeometryView g) const;
    StyleStatus validateVec2(Vec2Field f, DeviceCurveGeometryView g) const;
    StyleStatus validateBool(BoolField f, DeviceCurveGeometryView g) const;
    StyleStatus fail(StyleStatus s, const char* text);
    DeviceBuffer<int> error_;
    DeviceBuffer<float3> staging_;
    DeviceBuffer<uint8_t> keepStaging_;
    cudaEvent_t ready_ = nullptr;
    DeviceView<float3> output_{};
    DeviceView<uint8_t> keep_{};
    size_t points_ = 0, curves_ = 0;
    int deviceIndex_ = -1;
    bool pending_ = false;
    int* freshHostError_ = nullptr;
    UsdGenExecutionResourcePermit freshHostErrorPermit_;
    bool freshPreparing_ = false;
    bool freshApplying_ = false;
    bool freshPending_ = false;
    bool freshCallbackArmed_ = false;
    bool freshUploadFailed_ = false;
    bool unprovenWork_ = false;
    std::string diagnostic_;
};
}
#endif
