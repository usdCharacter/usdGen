#ifndef USDGEN_CUDA_PARAMETERS_H
#define USDGEN_CUDA_PARAMETERS_H
#ifdef USDGEN_ENABLE_CUDA

#include "usdGenCuda/gpu/expressionContext.h"
#include "usdGenCuda/gpu/expression.h"
#include "usdGen/executionPlan.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/registries/handles.h"

#include <memory>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace usdGen {

struct CudaParameterField {
    TfToken destination;
    const void* data = nullptr;
    size_t count = 0;
    expr::ScalarType type = expr::ScalarType::Invalid;
    uint32_t components = 1;
    expr::Domain domain = expr::Domain::Groom;
    UsdGenParamSlot slot = kUsdGenNoParam;   // the node op's slot for destination
};

struct CudaParameterGeometryMemoryShape {
    uint64_t pointCount = 0, curveCount = 0;
    bool hasRestPoints = false, hasWidths = false;
    bool hasHairT = false, hasRootUV = false;
};

enum class CudaParameterStatus {
    Ok, InvalidArgument, CompileError, UnsupportedType, CudaError, InvalidValue
};

// CudaParameterProgram is immutable CPU IR and literal data. It owns no CUDA
// allocations and can safely be shared by independent evaluators and frames.
class CudaParameterProgram {
public:
    static CudaParameterStatus Compile(
        UsdGenGraphSpec const&, UsdGenOpNode const&,
        std::shared_ptr<const CudaParameterProgram>*,
        std::vector<std::string>* diagnostics = nullptr);

    std::vector<UsdGenParamBinding> const& Bindings() const {
        return bindings_;
    }
    // Exact fresh-candidate delta: context, literals, typed outputs, device
    // IR/status and pinned IR/status. Published/retired cache is a baseline.
    bool EstimateFreshCandidateBytes(CudaParameterGeometryMemoryShape const&,
                                     uint64_t*) const noexcept;

private:
    struct Item {
        UsdGenParamBinding binding;
        UsdGenParamSlot slot = kUsdGenNoParam;
        expr::IRProgram ir;
        std::vector<double> literal;
    };
    std::vector<UsdGenParamBinding> bindings_;
    std::vector<Item> items_;

    friend class CudaParameterEvaluator;
};

// A mutable GPU execution workspace. Every evaluator has independent CUDA
// programs, contexts, and output buffers, while sharing only the immutable
// compiled CudaParameterProgram.
// Fields are borrowed until the next Evaluate or destruction. The work owner
// must finish all consumers before reuse/destruction and select the owning
// CUDA device; a single evaluator is never invoked concurrently.
class CudaParameterEvaluator {
public:
    CudaParameterEvaluator();
    ~CudaParameterEvaluator();
    CudaParameterEvaluator(CudaParameterEvaluator const&) = delete;
    CudaParameterEvaluator& operator=(CudaParameterEvaluator const&) = delete;
    CudaParameterEvaluator(CudaParameterEvaluator&&) = delete;
    CudaParameterEvaluator& operator=(CudaParameterEvaluator&&) = delete;
    CudaParameterStatus Evaluate(
        CudaParameterProgram const&, gpu::DeviceCurveGeometryView,
        gpu::ExpressionGeometryChannels, expr::Context, cudaStream_t,
        std::vector<std::string>* diagnostics = nullptr,
        UsdGenExecutionMemoryReservation* reservation = nullptr,
        UsdGenExecutionResourceKind kind = UsdGenExecutionResourceKind::Cache);
    // Parent-driven fresh phases. No callback is installed here: after each
    // Enqueue batch the parent installs one stream-terminal callback and, on
    // cudaSuccess plus launcher return, calls the matching Commit phase.
    // Geometry/channels are borrowed through final commit and downstream use.
    CudaParameterStatus BeginFreshContexts(std::shared_ptr<const CudaParameterProgram>,
        gpu::DeviceCurveGeometryView, gpu::ExpressionGeometryChannels,
        expr::Context, cudaStream_t, std::vector<std::string>* diagnostics = nullptr,
        UsdGenExecutionMemoryReservation* reservation = nullptr,
        UsdGenExecutionResourceKind kind = UsdGenExecutionResourceKind::Cache);
    CudaParameterStatus EnqueueFreshContextStatus(cudaStream_t);
    CudaParameterStatus CommitFreshContexts();
    CudaParameterStatus BeginFreshPrograms(cudaStream_t,
        UsdGenExecutionMemoryReservation* reservation = nullptr,
        UsdGenExecutionResourceKind kind = UsdGenExecutionResourceKind::Cache);
    CudaParameterStatus EnqueueFreshProgramStatus(cudaStream_t);
    CudaParameterStatus CommitFreshPrograms();
    bool HasUnprovenWork() const noexcept;
    // Host-only abandonment of a fully proved context batch.  This preserves
    // published fields and is only valid before program submission.
    bool DiscardFreshProven() noexcept;

    std::vector<CudaParameterField> const& Fields() const { return fields_; }
    /// By name, with or without "usdGen:"; compares text, never interns.
    CudaParameterField const* Find(TfToken const& destination) const;
    /// By the node op's slot; null for kUsdGenNoParam.
    CudaParameterField const* Find(UsdGenParamSlot slot) const;

private:
    enum class FreshPhase { Idle, ContextBuilt, ContextStatus, ContextCommitted,
                            ProgramsBuilt, ProgramStatus, Failed };
    std::vector<gpu::DeviceBuffer<double>> literals_;
    std::vector<gpu::DeviceBuffer<unsigned char>> outputs_;
    std::vector<std::unique_ptr<gpu::CudaExpressionProgram>> runtime_;
    std::vector<CudaParameterField> fields_;
    std::unique_ptr<gpu::CudaExpressionContext> contexts_[3];
    // Allocated before the first fresh submission.  If terminal proof is
    // lost, AbandonFresh releases this whole object rather than allocating or
    // running CUDA-owning destructors during evaluator teardown.
    struct FreshCandidate;
    std::unique_ptr<FreshCandidate> fresh_;
    // Fresh fields borrow freshPublished_ output storage.  A completion is
    // host-only, so replacing it parks the prior proven candidate here; the
    // next Begin may dispose that one bounded retired candidate.
    std::unique_ptr<FreshCandidate> freshPublished_;
    std::unique_ptr<FreshCandidate> freshRetired_;
    void ResetFreshProven() noexcept;
    void AbandonFresh() noexcept;
    FreshPhase freshPhase_ = FreshPhase::Idle;
    int freshDevice_ = -1;
    bool freshSubmitted_ = false;
};

// slot set: matched against field slots; otherwise by destination name.
struct CudaGroomScalarRequest {
    TfToken destination;
    expr::ScalarType type = expr::ScalarType::Invalid;
    UsdGenParamSlot slot = kUsdGenNoParam;
};

struct CudaGroomScalarValue {
    TfToken destination;
    expr::ScalarType type = expr::ScalarType::Invalid;
    std::array<unsigned char, 8> bytes{};
    UsdGenParamSlot slot = kUsdGenNoParam;
};

// Bounded parent-driven D2H readback for already-proven groom scalar fields.
// The caller retains every field's device storage through native terminal
// proof, including after a partially submitted BeginFresh failure. BeginFresh
// queues compact copies only; the parent owns the native callback and calls
// CommitFreshFinish only after terminal success and launcher return.
class CudaGroomScalarReadback {
public:
    static constexpr size_t Capacity = 8;

    CudaGroomScalarReadback() = default;
    ~CudaGroomScalarReadback();
    CudaGroomScalarReadback(CudaGroomScalarReadback const&) = delete;
    CudaGroomScalarReadback& operator=(CudaGroomScalarReadback const&) = delete;

    CudaParameterStatus BeginFresh(std::vector<CudaParameterField> const&,
                                   std::vector<CudaGroomScalarRequest> const&,
                                   cudaStream_t,
                                   UsdGenExecutionMemoryReservation* reservation = nullptr);
    CudaParameterStatus CommitFreshFinish();
    CudaGroomScalarValue const* Find(TfToken const&) const;
    CudaGroomScalarValue const* Find(UsdGenParamSlot) const;
    bool HasUnprovenWork() const noexcept { return unprovenWork_; }

private:
    std::array<CudaGroomScalarValue, Capacity> committed_{};
    std::array<CudaGroomScalarValue, Capacity> pending_{};
    unsigned char* pinned_ = nullptr;
    UsdGenExecutionResourcePermit pinnedPermit_;
    size_t committedCount_ = 0;
    size_t pendingCount_ = 0;
    int deviceIndex_ = -1;
    bool pendingFresh_ = false;
    bool unprovenWork_ = false;
    bool freshFailed_ = false;
};

// Test-only: makes the next BeginFreshPrograms fail before its first H2D.
void failNextCudaParameterFreshProgramAllocationForTesting();
// Test-only scalar-readback failure seams. The copy seam fails after one D2H
// copy has been accepted, so its object is intentionally unproven.
void failNextCudaGroomScalarReadbackAllocationForTesting();
void failNextCudaGroomScalarReadbackCopyAfterOneForTesting();

// Compatibility facade. Compilation delegates to an immutable program and
// evaluation delegates to one mutable evaluator owned by this facade.
class CudaParameterPlan {
public:
    static CudaParameterStatus Compile(
        UsdGenGraphSpec const&, UsdGenOpNode const&, CudaParameterPlan*,
        std::vector<std::string>* diagnostics = nullptr);
    CudaParameterStatus Evaluate(
        gpu::DeviceCurveGeometryView, gpu::ExpressionGeometryChannels,
        expr::Context, cudaStream_t,
        std::vector<std::string>* diagnostics = nullptr,
        UsdGenExecutionMemoryReservation* reservation = nullptr,
        UsdGenExecutionResourceKind kind = UsdGenExecutionResourceKind::Cache);

    std::vector<UsdGenParamBinding> const& Bindings() const {
        return program_ ? program_->Bindings() : emptyBindings_;
    }
    std::vector<CudaParameterField> const& Fields() const {
        return evaluator_ ? evaluator_->Fields() : emptyFields_;
    }
    CudaParameterField const* Find(TfToken const& destination) const;

private:
    std::shared_ptr<const CudaParameterProgram> program_;
    std::unique_ptr<CudaParameterEvaluator> evaluator_;
    // Always empty: the answer for an unbound plan, without a function-local static.
    std::vector<UsdGenParamBinding> emptyBindings_;
    std::vector<CudaParameterField> emptyFields_;
};
} // namespace usdGen
#endif // USDGEN_ENABLE_CUDA
#endif // USDGEN_CUDA_PARAMETERS_H
