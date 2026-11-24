#ifndef USDGEN_GPU_EXPRESSION_CONTEXT_H
#define USDGEN_GPU_EXPRESSION_CONTEXT_H

#include "curveGeometry.h"
#include "expression.h"

namespace usdGen::gpu {

struct ExpressionGeometryChannels {
    DeviceView<const float> hairT;      // one value per CV, optional
    DeviceView<const float2> rootUV;    // one value per primitive, optional
    // Rest root frame and surface face index, one value per primitive, all
    // optional. usdGen keeps ONE rest root frame, so both spellings of each
    // variable read the same plane: $N/$Nref <- rootN, $dPdu/$dPduref <- rootT,
    // $dPdv/$dPdvref <- rootB, $faceId <- rootPrim. A null view materializes no
    // field, so the variable stays unavailable exactly as it is today.
    DeviceView<const float3> rootN;
    DeviceView<const float3> rootT;
    DeviceView<const float3> rootB;
    DeviceView<const int32_t> rootPrim;
};

enum class ExpressionContextStatus { Ok, InvalidArgument, InvalidGeometry,
                                     InvalidChannel, CudaError };

// Builds one domain-specialized expression invocation context without a
// geometry readback.  The object owns converted device fields until the next
// Build/destruction and must outlive expression evaluation using Inputs().
class CudaExpressionContext {
public:
    ~CudaExpressionContext();
    CudaExpressionContext() = default;
    CudaExpressionContext(CudaExpressionContext const&) = delete;
    CudaExpressionContext& operator=(CudaExpressionContext const&) = delete;

    ExpressionContextStatus Build(DeviceCurveGeometryView geometry,
                                   ExpressionGeometryChannels channels,
                                   expr::Context context, cudaStream_t stream,
                                   UsdGenExecutionMemoryReservation* reservation = nullptr,
                                   UsdGenExecutionResourceKind kind =
                                       UsdGenExecutionResourceKind::Cache);
    // Fresh async path: inputs remain unpublished until the parent has
    // enqueued this context's status copy, received the shared batch terminal
    // callback, and calls CommitFreshFinish.
    ExpressionContextStatus BuildFresh(DeviceCurveGeometryView geometry,
                                        ExpressionGeometryChannels channels,
                                        expr::Context context, cudaStream_t stream,
                                        UsdGenExecutionMemoryReservation* reservation = nullptr,
                                        UsdGenExecutionResourceKind kind =
                                            UsdGenExecutionResourceKind::Cache);
    // Enqueues this context's status D2H behind its work. The parent owns the
    // single native callback shared by all fresh contexts and must retain this
    // object through that callback and the following commit.
    ExpressionContextStatus EnqueueFreshStatus(cudaStream_t stream);
    // Host-only. Call only after EnqueueFreshStatus returned Ok, the parent
    // returned from installing its native callback, and that callback reported
    // cudaSuccess. This method never calls or synchronizes CUDA.
    ExpressionContextStatus CommitFreshFinish();
    bool HasUnprovenWork() const noexcept { return unprovenWork_; }
    ExpressionInputs const& Inputs() const;
    ExpressionContextStatus Wait(cudaStream_t stream) const;
    ExpressionContextStatus Finish(cudaStream_t stream);

private:
    ExpressionContextStatus BuildImpl(DeviceCurveGeometryView,
                                      ExpressionGeometryChannels, expr::Context,
                                      cudaStream_t, bool fresh,
                                      UsdGenExecutionMemoryReservation*,
                                      UsdGenExecutionResourceKind);
    DeviceBuffer<double> fields_[static_cast<unsigned>(expr::Variable::CountVariables)];
    DeviceBuffer<uint32_t> owners_;
    DeviceBuffer<double> arcLength_;
    DeviceBuffer<int> error_;
    int* freshHostError_ = nullptr;
    UsdGenExecutionResourcePermit freshHostErrorPermit_;
    cudaEvent_t ready_ = nullptr;
    ExpressionInputs inputs_{};
    ExpressionInputs pendingInputs_{};
    int deviceIndex_ = -1;
    bool pending_ = false;
    bool usable_ = false;
    bool freshPending_ = false;
    bool freshStatusEnqueued_ = false;
    bool freshFailed_ = false;
    bool unprovenWork_ = false;
};

} // namespace usdGen::gpu
#endif
