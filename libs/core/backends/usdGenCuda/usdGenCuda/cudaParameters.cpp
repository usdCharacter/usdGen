#ifdef USDGEN_ENABLE_CUDA
#include "cudaParameters.h"
#include "usdGen/gpu/cudaCompat.h"
#include "usdGen/gpu/cudaCompletionLane.h"
#include "usdGen/expressions/frontend.h"
#include "usdGen/registries/opRegistry.h"
#include "pxr/base/gf/half.h"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec4d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec4f.h"
#include "pxr/base/gf/vec2h.h"
#include "pxr/base/gf/vec3h.h"
#include "pxr/base/gf/vec4h.h"
#include "pxr/base/gf/vec2i.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/gf/vec4i.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <string_view>

PXR_NAMESPACE_USING_DIRECTIVE
namespace usdGen {
namespace {
std::atomic<bool> s_failNextFreshProgramAllocation{false};
std::atomic<bool> s_failNextGroomScalarReadbackAllocation{false};
std::atomic<bool> s_failNextGroomScalarReadbackCopyAfterOne{false};
using Type = expr::ScalarType;
using Domain = expr::Domain;

void Error(std::vector<std::string>* diagnostics, std::string const& message) {
    if (diagnostics) diagnostics->push_back(message);
}
// The name without "usdGen:"; a view of the token's text, never interned.
std::string_view CanonicalName(TfToken const& name) {
    std::string_view value = name.GetString();
    if (value.compare(0, 7, "usdGen:") == 0) value.remove_prefix(7);
    return value;
}
// The first slot of the node's op whose canonical name is destination's.
UsdGenParamSlot DestinationSlot(UsdGenOpRecord const* record, TfToken const& destination) {
    if (!record) return kUsdGenNoParam;
    std::string_view const canonical = CanonicalName(destination);
    for (UsdGenParamSlot s = 0; s != record->ParamCount(); ++s)
        if (CanonicalName(record->ParamName(s)) == canonical) return s;
    return kUsdGenNoParam;
}
size_t ScalarBytes(Type type) {
    switch (type) {
    case Type::Bool: return 1;
    case Type::Float16: return 2;
    case Type::Int32: case Type::UInt32: case Type::Float32: return 4;
    case Type::Int64: case Type::UInt64: case Type::Float64: return 8;
    default: return 0;
    }
}
bool SupportedShape(expr::ValueShape const& shape) {
    return ScalarBytes(shape.scalar) && !shape.isArray &&
        shape.elementCount == 1 && shape.rows == 1 && shape.columns == 1 &&
        shape.components >= 1 && shape.components <= 4;
}
bool SameShape(expr::ValueShape const& a, expr::ValueShape const& b) {
    return a.scalar == b.scalar && a.elementCount == b.elementCount &&
        a.components == b.components && a.rows == b.rows &&
        a.columns == b.columns && a.isArray == b.isArray;
}
template<class T>
bool ScalarLiteral(VtValue const& value, std::vector<double>* output) {
    if (!value.IsHolding<T>()) return false;
    output->assign(1, static_cast<double>(value.UncheckedGet<T>()));
    return std::isfinite((*output)[0]);
}
template<class T>
bool VectorLiteral(VtValue const& value, unsigned count, std::vector<double>* output) {
    if (!value.IsHolding<T>() || count != T::dimension) return false;
    output->resize(count);
    auto const& vector = value.UncheckedGet<T>();
    for (unsigned i = 0; i < count; ++i) {
        (*output)[i] = static_cast<double>(vector[i]);
        if (!std::isfinite((*output)[i])) return false;
    }
    return true;
}
bool Literal(VtValue const& value, expr::ValueShape const& shape,
             std::vector<double>* output) {
    unsigned n = shape.components;
    if (n == 1) {
        switch (shape.scalar) {
        case Type::Bool: return ScalarLiteral<bool>(value, output);
        case Type::Int32: return ScalarLiteral<int32_t>(value, output);
        case Type::UInt32: return ScalarLiteral<uint32_t>(value, output);
        case Type::Int64:
            if (!value.IsHolding<int64_t>() ||
                value.UncheckedGet<int64_t>() < -(int64_t(1) << 53) ||
                value.UncheckedGet<int64_t>() > (int64_t(1) << 53)) return false;
            return ScalarLiteral<int64_t>(value, output);
        case Type::UInt64:
            if (!value.IsHolding<uint64_t>() ||
                value.UncheckedGet<uint64_t>() > (uint64_t(1) << 53)) return false;
            return ScalarLiteral<uint64_t>(value, output);
        case Type::Float16: return ScalarLiteral<GfHalf>(value, output);
        case Type::Float32: return ScalarLiteral<float>(value, output);
        case Type::Float64: return ScalarLiteral<double>(value, output);
        default: return false;
        }
    }
    switch (shape.scalar) {
    case Type::Float16:
        return VectorLiteral<GfVec2h>(value, n, output) ||
            VectorLiteral<GfVec3h>(value, n, output) || VectorLiteral<GfVec4h>(value, n, output);
    case Type::Float32:
        return VectorLiteral<GfVec2f>(value, n, output) ||
            VectorLiteral<GfVec3f>(value, n, output) || VectorLiteral<GfVec4f>(value, n, output);
    case Type::Float64:
        return VectorLiteral<GfVec2d>(value, n, output) ||
            VectorLiteral<GfVec3d>(value, n, output) || VectorLiteral<GfVec4d>(value, n, output);
    case Type::Int32:
        return VectorLiteral<GfVec2i>(value, n, output) ||
            VectorLiteral<GfVec3i>(value, n, output) || VectorLiteral<GfVec4i>(value, n, output);
    default: return false;
    }
}
size_t DomainIndex(Domain domain) {
    return domain == Domain::Groom ? 0 : domain == Domain::Primitive ? 1 : 2;
}
bool EstimateAdd(uint64_t* total, uint64_t value) {
    return UsdGenExecutionCheckedBytes::Add(total, value);
}
bool EstimateMultiply(uint64_t count, uint64_t bytes, uint64_t* result) {
    return UsdGenExecutionCheckedBytes::Multiply(count, bytes, result);
}
} // namespace

CudaParameterField const* CudaParameterEvaluator::Find(TfToken const& destination) const {
    auto canonical = CanonicalName(destination);
    for (auto const& field : fields_)
        if (CanonicalName(field.destination) == canonical) return &field;
    return nullptr;
}

CudaParameterField const* CudaParameterEvaluator::Find(UsdGenParamSlot slot) const {
    if (slot == kUsdGenNoParam) return nullptr;
    for (auto const& field : fields_)
        if (field.slot == slot) return &field;
    return nullptr;
}

struct CudaParameterEvaluator::FreshCandidate {
    std::shared_ptr<const CudaParameterProgram> program;
    std::array<std::unique_ptr<gpu::CudaExpressionContext>, 3> contexts;
    std::vector<gpu::DeviceBuffer<double>> literals;
    std::vector<gpu::DeviceBuffer<unsigned char>> outputs;
    std::vector<std::unique_ptr<gpu::CudaExpressionProgram>> runtime;
    std::vector<CudaParameterField> fields;
    // Context and program allocations form one candidate; retain the caller's
    // explicit credit source so a later phase cannot accidentally fall back
    // to an unrelated exact permit.
    UsdGenExecutionMemoryReservation* reservation = nullptr;
    UsdGenExecutionResourceKind kind = UsdGenExecutionResourceKind::Cache;
};

CudaParameterStatus CudaParameterProgram::Compile(
    UsdGenGraphSpec const& graph, UsdGenOpNode const& node,
    std::shared_ptr<const CudaParameterProgram>* output,
    std::vector<std::string>* diagnostics) {
    if (!output) {
        Error(diagnostics, "null parameter plan");
        return CudaParameterStatus::InvalidArgument;
    }
    std::vector<UsdGenParamBinding> bindings;
    std::vector<Item> items;
    std::set<std::string_view> destinations;
    UsdGenOpRegistry const& ops = UsdGenOpRegistry::Get();
    UsdGenOpId const opId = ops.Find(node.type);
    UsdGenOpRecord const* record = opId != UsdGenOpId::Invalid ? &ops.Record(opId) : nullptr;
    for (auto const& binding : node.expressionBindings) {
        auto fail = [&](CudaParameterStatus status, std::string message) {
            Error(diagnostics, binding.destination.GetString() + ": " + message);
            return status;
        };
        if (binding.destination.IsEmpty() || !binding.expression.IsValid() ||
            !destinations.insert(CanonicalName(binding.destination)).second)
            return fail(CudaParameterStatus::InvalidArgument, "empty or duplicate expression destination/path");
        auto expression = std::find_if(graph.expressions.begin(), graph.expressions.end(),
            [&](auto const& candidate) { return candidate.key == binding.expression; });
        if (expression == graph.expressions.end())
            return fail(CudaParameterStatus::CompileError, "missing expression");
        // An empty output is a connection to the expression PRIM; it resolves
        // to outputs:result, or to a single declared output.
        UsdGenExpressionOutputDesc const* result =
            UsdGenFindExpressionOutput(*expression, binding.output);
        if (!result)
            return fail(CudaParameterStatus::CompileError, binding.output.IsEmpty()
                ? "connection to prim " + graph.Label(binding.expression) +
                      " declaring no outputs:result and no single outputs:* attribute"
                : "missing expression output");
        if (binding.domain != Domain::Groom && binding.domain != Domain::Primitive &&
            binding.domain != Domain::Point)
            return fail(CudaParameterStatus::InvalidArgument, "invalid evaluation domain");
        if (!SupportedShape(binding.destinationShape))
            return fail(CudaParameterStatus::UnsupportedType, "unsupported array/matrix/type shape");
        if (binding.nativeType.IsEmpty() || result->nativeType != binding.nativeType ||
            !SameShape(result->shape, binding.destinationShape))
            return fail(CudaParameterStatus::InvalidArgument, "output native type/shape mismatch");
        Item item;
        item.binding = binding;
        item.slot = DestinationSlot(record, binding.destination);
        if (!Literal(binding.literal, binding.destinationShape, &item.literal))
            return fail(CudaParameterStatus::UnsupportedType,
                "literal must have the exact finite native type; int64/uint64 currently limited to 2^53");
        auto compiled = expr::Frontend::Compile(expression->source,
            {binding.domain, binding.destinationShape.scalar, binding.destinationShape.components});
        if (!compiled.ok) {
            for (auto const& message : compiled.diagnostics)
                Error(diagnostics, binding.destination.GetString() + ": " + message);
            return fail(CudaParameterStatus::CompileError, "SeExpr compilation failed");
        }
        // geoSampler()/ptex() read host data through the CPU evaluator's
        // sampler table; the device has none (expressions/irExec.h).
        if (!compiled.program.IR().samplers.empty())
            return fail(CudaParameterStatus::UnsupportedType,
                "geoSampler() and ptex() are only available on the CPU lane");
        item.ir = compiled.program.IR();
        bindings.push_back(binding);
        items.push_back(std::move(item));
    }
    // Only immutable CPU data changes here. An unsuccessful compile leaves
    // both the old program and its published fields usable.
    try {
        auto program = std::make_shared<CudaParameterProgram>();
        program->bindings_ = std::move(bindings);
        program->items_ = std::move(items);
        *output = std::move(program);
    } catch (...) {
        Error(diagnostics, "failed to allocate immutable CUDA parameter program");
        return CudaParameterStatus::CudaError;
    }
    return CudaParameterStatus::Ok;
}

bool CudaParameterProgram::EstimateFreshCandidateBytes(
    CudaParameterGeometryMemoryShape const& shape, uint64_t* result) const noexcept {
    if (!result) return false;
    uint64_t total = 0;
    std::array<bool, 3> domains{};
    for (auto const& item : items_)
        domains[DomainIndex(item.binding.domain)] = true;
    auto context = [&](Domain domain) {
        uint64_t bytes = 0, n = 1, fieldDoubles = 0, value = 0;
        if (domain == Domain::Groom)
            return EstimateAdd(&total, 2 * sizeof(int)); // device + pinned status
        n = domain == Domain::Primitive ? shape.curveCount : shape.pointCount;
        // P and RootP are always materialized. PRef/RootPRef are materialized
        // only when the actual geometry has a rest channel.
        fieldDoubles = shape.hasRestPoints ? 12 : 6;
        // Scalar field set exactly mirrors CudaExpressionContext::BuildImpl.
        fieldDoubles += domain == Domain::Primitive ? 7 : 9;
        if (shape.hasWidths) ++fieldDoubles;
        if (shape.hasRootUV) fieldDoubles += 2;
        // $N/$Nref, $dPdu/$dPduref and $dPdv/$dPdvref are three doubles each,
        // plus the scalar $faceId. They ride on the same C3 root binding that
        // publishes the root UVs (rootT/rootB/rootN and rootPrim are produced
        // with rootUV by scatter/grow), and CudaParameterGeometryMemoryShape
        // carries no separate frame flag, so hasRootUV is the flag that covers
        // them. A path that has the UVs but not the frames over-reserves here,
        // which is the safe direction; under-reserving is not.
        if (shape.hasRootUV) fieldDoubles += 6 * 3 + 1;
        return EstimateMultiply(n, fieldDoubles * sizeof(double), &value) &&
            EstimateAdd(&bytes, value) &&
            // owners + arc are allocated for each non-groom context.
            EstimateMultiply(shape.pointCount, sizeof(uint32_t) + sizeof(double), &value) &&
            EstimateAdd(&bytes, value) && EstimateAdd(&bytes, 2 * sizeof(int)) &&
            EstimateAdd(&total, bytes);
    };
    for (size_t i = 0; i != domains.size(); ++i) if (domains[i]) {
        Domain const domain = i == 0 ? Domain::Groom :
            i == 1 ? Domain::Primitive : Domain::Point;
        if (!context(domain)) return false;
    }
    for (auto const& item : items_) {
        uint64_t count = item.binding.domain == Domain::Groom ? 1 :
            item.binding.domain == Domain::Primitive ? shape.curveCount : shape.pointCount;
        uint64_t literal = 0, output = 0, code = 0, value = 0;
        size_t const scalar = ScalarBytes(item.binding.destinationShape.scalar);
        if (!scalar || !EstimateMultiply(item.binding.destinationShape.components,
                sizeof(double), &literal) ||
            !EstimateMultiply(count, scalar, &value) ||
            !EstimateMultiply(value, item.binding.destinationShape.components, &output) ||
            !EstimateMultiply(item.ir.instructions.size(), sizeof(expr::IRInstruction), &code) ||
            // device + pinned IR and device + pinned error status.
            !EstimateMultiply(code, 2, &code) ||
            !EstimateAdd(&code, 2 * sizeof(int)) ||
            !EstimateAdd(&total, literal) || !EstimateAdd(&total, output) ||
            !EstimateAdd(&total, code)) return false;
    }
    *result = total;
    return true;
}

CudaParameterStatus CudaParameterEvaluator::Evaluate(
    CudaParameterProgram const& program,
    gpu::DeviceCurveGeometryView geometry, gpu::ExpressionGeometryChannels channels,
    expr::Context controls, cudaStream_t stream, std::vector<std::string>* diagnostics,
    UsdGenExecutionMemoryReservation* reservation, UsdGenExecutionResourceKind kind) {
    if (freshPhase_ != FreshPhase::Idle || HasUnprovenWork())
        return CudaParameterStatus::InvalidArgument;
    // A legacy evaluation replaces the published field set synchronously.
    // Fresh candidates are all proven whenever the phase is Idle.
    fresh_.reset();
    freshPublished_.reset();
    freshRetired_.reset();
    fields_.clear();
    // A preceding successful Evaluate finished every GPU reader. No published
    // field survives the next invocation; callers must consume fields first.
    runtime_.clear();
    outputs_.clear();
    literals_.clear();
    for (auto& context : contexts_) context.reset();
    for (auto const& item : program.items_) {
        auto index = DomainIndex(item.binding.domain);
        if (!contexts_[index]) contexts_[index] = std::make_unique<gpu::CudaExpressionContext>();
    }
    for (size_t i = 0; i < 3; ++i) if (contexts_[i]) {
        auto context = controls;
        context.domain = i == 0 ? Domain::Groom : i == 1 ? Domain::Primitive : Domain::Point;
        if (contexts_[i]->Build(geometry, channels, context, stream, reservation, kind) != gpu::ExpressionContextStatus::Ok ||
            contexts_[i]->Finish(stream) != gpu::ExpressionContextStatus::Ok) {
            Error(diagnostics, "expression geometry context validation failed");
            return CudaParameterStatus::CudaError;
        }
    }
    literals_.resize(program.items_.size());
    outputs_.resize(program.items_.size());
    runtime_.reserve(program.items_.size());
    std::vector<CudaParameterField> candidate;
    candidate.reserve(program.items_.size());
    for (size_t i = 0; i < program.items_.size(); ++i) {
        auto const& item = program.items_[i];
        auto const& binding = item.binding;
        auto inputs = contexts_[DomainIndex(binding.domain)]->Inputs();
        size_t const components = binding.destinationShape.components;
        size_t const stride = ScalarBytes(binding.destinationShape.scalar) * components;
        auto fail = [&](CudaParameterStatus status, char const* message) {
            // Error exits must not leave queued readers using context/literal
            // allocations that the next invocation will replace.
            gpu::CudaFenceStream(stream);
            Error(diagnostics, binding.destination.GetString() + ": " + message);
            return status;
        };
        if (inputs.count > std::numeric_limits<size_t>::max() / stride)
            return fail(CudaParameterStatus::InvalidArgument, "output size overflow");
        if (literals_[i].reset(components, reservation, kind) != cudaSuccess ||
            cudaMemcpyAsync(literals_[i].data(), item.literal.data(), components * sizeof(double),
                cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            outputs_[i].reset(inputs.count * stride, reservation, kind) != cudaSuccess)
            return fail(CudaParameterStatus::CudaError, "device allocation/literal upload failed");
        inputs.fields[static_cast<unsigned>(expr::Variable::Value)] =
            {literals_[i].data(), 1, Domain::Groom, static_cast<uint32_t>(components)};
        runtime_.push_back(std::make_unique<gpu::CudaExpressionProgram>());
        auto& runtimeProgram = *runtime_.back();
        if (runtimeProgram.Upload(item.ir, stream, reservation, kind) != gpu::ExpressionStatus::Ok ||
            runtimeProgram.Evaluate(inputs, {outputs_[i].data(), inputs.count,
                binding.destinationShape.scalar, static_cast<uint32_t>(components)}, stream) != gpu::ExpressionStatus::Ok)
            return fail(CudaParameterStatus::CudaError, "CUDA expression launch failed");
        if (runtimeProgram.Finish(stream) != gpu::ExpressionStatus::Ok)
            return fail(CudaParameterStatus::InvalidValue, "CUDA expression value validation failed");
        candidate.push_back({binding.destination, outputs_[i].data(), inputs.count,
            binding.destinationShape.scalar, static_cast<uint32_t>(components), binding.domain,
            item.slot});
    }
    fields_ = std::move(candidate);
    return CudaParameterStatus::Ok;
}

CudaParameterEvaluator::CudaParameterEvaluator() = default;

CudaParameterEvaluator::~CudaParameterEvaluator() {
    if (HasUnprovenWork()) AbandonFresh();
}

void CudaParameterEvaluator::ResetFreshProven() noexcept {
    // This is called only after every child has accepted terminal proof.  It
    // is consequently safe for the candidate's CUDA-owning destructors to
    // run, including contexts retained for program input lifetime.
    fresh_.reset();
    freshPhase_ = FreshPhase::Idle;
    freshDevice_ = -1;
    freshSubmitted_ = false;
}

void CudaParameterEvaluator::AbandonFresh() noexcept {
    // The candidate was allocated before BuildFresh / the first literal copy.
    // Releasing it is intentionally permanent: it retains the immutable host
    // program (pageable literal source), contexts, programs, and buffers
    // without allocating or invoking a CUDA-owning destructor after proof is
    // missing.  Lower-level objects quarantine their own buffers on teardown.
    (void)fresh_.release();
    freshPhase_ = FreshPhase::Failed;
}

bool CudaParameterEvaluator::HasUnprovenWork() const noexcept {
    if (freshSubmitted_) return true;
    if (!fresh_) return false;
    for (auto const& context : fresh_->contexts)
        if (context && context->HasUnprovenWork()) return true;
    for (auto const& program : fresh_->runtime)
        if (program && program->HasUnprovenWork()) return true;
    return false;
}

bool CudaParameterEvaluator::DiscardFreshProven() noexcept {
    if (freshPhase_ != FreshPhase::ContextCommitted || HasUnprovenWork() ||
        !fresh_ || freshRetired_) return false;
    // This is host-only: the proven candidate is parked until the next Begin
    // launcher disposes the one bounded retired candidate.
    freshRetired_ = std::move(fresh_);
    freshPhase_ = FreshPhase::Idle;
    freshDevice_ = -1;
    freshSubmitted_ = false;
    return true;
}

void failNextCudaParameterFreshProgramAllocationForTesting() {
    s_failNextFreshProgramAllocation.store(true, std::memory_order_release);
}

CudaParameterStatus CudaParameterEvaluator::BeginFreshContexts(
    std::shared_ptr<const CudaParameterProgram> program,
    gpu::DeviceCurveGeometryView geometry, gpu::ExpressionGeometryChannels channels,
    expr::Context controls, cudaStream_t stream, std::vector<std::string>* diagnostics,
    UsdGenExecutionMemoryReservation* reservation, UsdGenExecutionResourceKind kind) {
    if (!program || freshPhase_ != FreshPhase::Idle || fresh_ || HasUnprovenWork())
        return CudaParameterStatus::InvalidArgument;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    int current = -1, streamDevice = -1;
    // Capture must be rejected before device/stream queries or allocation,
    // including for an empty binding program.
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
        return CudaParameterStatus::CudaError;
    if (capture != cudaStreamCaptureStatusNone)
        return CudaParameterStatus::InvalidArgument;
    if (cudaGetDevice(&current) != cudaSuccess ||
        (stream && cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess))
        return CudaParameterStatus::CudaError;
    if (stream && streamDevice != current)
        return CudaParameterStatus::InvalidArgument;
    try {
        // Completion never destructs CUDA storage.  This is the next launcher,
        // before any new submission, so dispose the one bounded retired
        // candidate left by a prior proven completion.
        freshRetired_.reset();
        fresh_ = std::make_unique<FreshCandidate>();
        fresh_->program = std::move(program);
        fresh_->reservation = reservation;
        fresh_->kind = kind;
        freshDevice_ = current;
        for (auto const& item : fresh_->program->items_) {
            auto const index = DomainIndex(item.binding.domain);
            if (!fresh_->contexts[index]) fresh_->contexts[index] = std::make_unique<gpu::CudaExpressionContext>();
        }
        for (size_t i = 0; i != fresh_->contexts.size(); ++i) if (fresh_->contexts[i]) {
            auto context = controls;
            context.domain = i == 0 ? Domain::Groom : i == 1 ? Domain::Primitive : Domain::Point;
            auto status = fresh_->contexts[i]->BuildFresh(geometry, channels, context, stream,
                reservation, kind);
            if (status != gpu::ExpressionContextStatus::Ok) {
                if (diagnostics) diagnostics->push_back("fresh expression geometry context validation failed");
                if (HasUnprovenWork()) freshPhase_ = FreshPhase::Failed;
                else ResetFreshProven();
                return status == gpu::ExpressionContextStatus::InvalidArgument ||
                               status == gpu::ExpressionContextStatus::InvalidGeometry ||
                               status == gpu::ExpressionContextStatus::InvalidChannel
                    ? CudaParameterStatus::InvalidArgument : CudaParameterStatus::CudaError;
            }
        }
        freshPhase_ = FreshPhase::ContextBuilt;
        return CudaParameterStatus::Ok;
    } catch (...) {
        if (HasUnprovenWork()) freshPhase_ = FreshPhase::Failed;
        else ResetFreshProven();
        return CudaParameterStatus::CudaError;
    }
}

CudaParameterStatus CudaParameterEvaluator::EnqueueFreshContextStatus(cudaStream_t stream) {
    if (freshPhase_ != FreshPhase::ContextBuilt) return CudaParameterStatus::InvalidArgument;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    int current = -1, streamDevice = -1;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
        return CudaParameterStatus::CudaError;
    if (capture != cudaStreamCaptureStatusNone)
        return CudaParameterStatus::InvalidArgument;
    if (cudaGetDevice(&current) != cudaSuccess ||
        (stream && cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess))
        return CudaParameterStatus::CudaError;
    if (current != freshDevice_ || (stream && streamDevice != current))
        return CudaParameterStatus::InvalidArgument;
    for (auto const& context : fresh_->contexts) if (context &&
        context->EnqueueFreshStatus(stream) != gpu::ExpressionContextStatus::Ok)
        { freshPhase_ = FreshPhase::Failed; return CudaParameterStatus::CudaError; }
    freshPhase_ = FreshPhase::ContextStatus;
    return CudaParameterStatus::Ok;
}

CudaParameterStatus CudaParameterEvaluator::CommitFreshContexts() {
    if (freshPhase_ != FreshPhase::ContextStatus) return CudaParameterStatus::InvalidArgument;
    CudaParameterStatus result = CudaParameterStatus::Ok;
    for (auto const& context : fresh_->contexts) if (context) {
        auto status = context->CommitFreshFinish();
        if (status != gpu::ExpressionContextStatus::Ok && result == CudaParameterStatus::Ok)
            result = status == gpu::ExpressionContextStatus::InvalidArgument ||
                     status == gpu::ExpressionContextStatus::InvalidGeometry ||
                     status == gpu::ExpressionContextStatus::InvalidChannel
                ? CudaParameterStatus::InvalidArgument : CudaParameterStatus::InvalidValue;
    }
    if (HasUnprovenWork()) { freshPhase_ = FreshPhase::Failed; return CudaParameterStatus::CudaError; }
    freshSubmitted_ = false;
    if (result != CudaParameterStatus::Ok) {
        // Host-only completion: park the proven rejected candidate for the
        // next launcher rather than running context destructors here.
        if (freshRetired_) { freshPhase_ = FreshPhase::Failed; return CudaParameterStatus::CudaError; }
        freshRetired_ = std::move(fresh_);
        freshPhase_ = FreshPhase::Idle;
        freshDevice_ = -1;
        return result;
    }
    freshPhase_ = FreshPhase::ContextCommitted;
    return CudaParameterStatus::Ok;
}

CudaParameterStatus CudaParameterEvaluator::BeginFreshPrograms(
    cudaStream_t stream, UsdGenExecutionMemoryReservation* reservation,
    UsdGenExecutionResourceKind kind) {
    if (freshPhase_ != FreshPhase::ContextCommitted || !fresh_ || !fresh_->program)
        return CudaParameterStatus::InvalidArgument;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    int current = -1, streamDevice = -1;
    // Do this before candidate allocation or the pageable literal upload.
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
        return CudaParameterStatus::CudaError;
    if (capture != cudaStreamCaptureStatusNone)
        return CudaParameterStatus::InvalidArgument;
    if (cudaGetDevice(&current) != cudaSuccess ||
        (stream && cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess))
        return CudaParameterStatus::CudaError;
    if (current != freshDevice_ || (stream && streamDevice != current))
        return CudaParameterStatus::InvalidArgument;
    if (!reservation) {
        reservation = fresh_->reservation;
        kind = fresh_->kind;
    } else if (reservation != fresh_->reservation || kind != fresh_->kind) {
        return CudaParameterStatus::InvalidArgument;
    }
    // Test seam: deliberately fail before the candidate allocation loop and
    // therefore before any literal H2D submission.
    if (s_failNextFreshProgramAllocation.exchange(false, std::memory_order_acq_rel))
        return CudaParameterStatus::CudaError;
    try {
        size_t const count = fresh_->program->items_.size();
        fresh_->literals.resize(count); fresh_->outputs.resize(count); fresh_->runtime.resize(count);
        fresh_->fields.clear(); fresh_->fields.reserve(count);
        // Allocate every candidate buffer before the first literal/upload/evaluate submission.
        for (size_t i = 0; i != count; ++i) {
            auto const& item = fresh_->program->items_[i];
            auto const& inputs = fresh_->contexts[DomainIndex(item.binding.domain)]->Inputs();
            size_t const components = item.binding.destinationShape.components;
            size_t const stride = ScalarBytes(item.binding.destinationShape.scalar) * components;
            if (!stride || inputs.count > std::numeric_limits<size_t>::max() / stride) {
                fresh_->runtime.clear(); fresh_->outputs.clear(); fresh_->literals.clear(); fresh_->fields.clear();
                return CudaParameterStatus::InvalidArgument;
            }
            if (fresh_->literals[i].reset(components, reservation, kind) != cudaSuccess ||
                fresh_->outputs[i].reset(inputs.count * stride, reservation, kind) != cudaSuccess) {
                fresh_->runtime.clear(); fresh_->outputs.clear(); fresh_->literals.clear(); fresh_->fields.clear();
                return CudaParameterStatus::CudaError;
            }
            fresh_->runtime[i] = std::make_unique<gpu::CudaExpressionProgram>();
        }
        // This is deliberately before the first pageable H2D: a failure after
        // that point must retain the immutable program and all candidate
        // storage until aggregate terminal proof, even if UploadFresh fails.
        if (count) freshSubmitted_ = true;
        for (size_t i = 0; i != count; ++i) {
            auto const& item = fresh_->program->items_[i];
            auto inputs = fresh_->contexts[DomainIndex(item.binding.domain)]->Inputs();
            size_t const components = item.binding.destinationShape.components;
            if (cudaMemcpyAsync(fresh_->literals[i].data(), item.literal.data(), components * sizeof(double),
                cudaMemcpyHostToDevice, stream) != cudaSuccess ||
                fresh_->runtime[i]->UploadFresh(item.ir, stream, reservation, kind) != gpu::ExpressionStatus::Ok) {
                freshPhase_ = FreshPhase::Failed;
                return CudaParameterStatus::CudaError;
            }
            inputs.fields[static_cast<unsigned>(expr::Variable::Value)] =
                {fresh_->literals[i].data(), 1, Domain::Groom, static_cast<uint32_t>(components)};
            if (fresh_->runtime[i]->EvaluateFresh(inputs, {fresh_->outputs[i].data(), inputs.count,
                    item.binding.destinationShape.scalar, static_cast<uint32_t>(components)}, stream) != gpu::ExpressionStatus::Ok) {
                freshPhase_ = FreshPhase::Failed;
                return CudaParameterStatus::CudaError;
            }
            fresh_->fields.push_back({item.binding.destination, fresh_->outputs[i].data(), inputs.count,
                item.binding.destinationShape.scalar, static_cast<uint32_t>(components), item.binding.domain,
                item.slot});
        }
        freshPhase_ = FreshPhase::ProgramsBuilt;
        return CudaParameterStatus::Ok;
    } catch (...) {
        if (freshSubmitted_ || HasUnprovenWork()) freshPhase_ = FreshPhase::Failed;
        else {
            fresh_->runtime.clear(); fresh_->outputs.clear(); fresh_->literals.clear(); fresh_->fields.clear();
        }
        return CudaParameterStatus::CudaError;
    }
}

CudaParameterStatus CudaParameterEvaluator::EnqueueFreshProgramStatus(cudaStream_t stream) {
    if (freshPhase_ != FreshPhase::ProgramsBuilt) return CudaParameterStatus::InvalidArgument;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    int current = -1, streamDevice = -1;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
        return CudaParameterStatus::CudaError;
    if (capture != cudaStreamCaptureStatusNone)
        return CudaParameterStatus::InvalidArgument;
    if (cudaGetDevice(&current) != cudaSuccess ||
        (stream && cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess))
        return CudaParameterStatus::CudaError;
    if (current != freshDevice_ || (stream && streamDevice != current))
        return CudaParameterStatus::InvalidArgument;
    for (auto const& program : fresh_->runtime)
        if (program->EnqueueFreshStatus(stream) != gpu::ExpressionStatus::Ok) {
            freshPhase_ = FreshPhase::Failed;
            return CudaParameterStatus::CudaError;
        }
    freshPhase_ = FreshPhase::ProgramStatus;
    return CudaParameterStatus::Ok;
}

CudaParameterStatus CudaParameterEvaluator::CommitFreshPrograms() {
    if (freshPhase_ != FreshPhase::ProgramStatus) return CudaParameterStatus::InvalidArgument;
    CudaParameterStatus result = CudaParameterStatus::Ok;
    for (auto const& program : fresh_->runtime)
        if (program->CommitFreshFinish() != gpu::ExpressionStatus::Ok && result == CudaParameterStatus::Ok)
            result = CudaParameterStatus::InvalidValue;
    freshSubmitted_ = false;
    if (HasUnprovenWork()) { freshPhase_ = FreshPhase::Failed; return CudaParameterStatus::CudaError; }
    if (result == CudaParameterStatus::Ok) {
        // Fields borrow output buffers, so transfer that candidate ownership
        // atomically with publication; old published fields remain intact on
        // any semantic failure above.
        if (freshRetired_) { freshPhase_ = FreshPhase::Failed; return CudaParameterStatus::CudaError; }
        freshRetired_ = std::move(freshPublished_);
        freshPublished_ = std::move(fresh_);
        fields_ = std::move(freshPublished_->fields);
        // Contexts only feed the fresh programs; all program proof is now in.
        freshPhase_ = FreshPhase::Idle;
        freshDevice_ = -1;
        return CudaParameterStatus::Ok;
    }
    // Semantic failure is proven and does not publish a partial candidate.
    if (freshRetired_) { freshPhase_ = FreshPhase::Failed; return CudaParameterStatus::CudaError; }
    freshRetired_ = std::move(fresh_);
    freshPhase_ = FreshPhase::Idle;
    freshDevice_ = -1;
    return result;
}

CudaGroomScalarReadback::~CudaGroomScalarReadback() {
    if (unprovenWork_) {
        pinned_ = nullptr;
        pinnedPermit_.Abandon();
        return;
    }
    if (!pinned_) return;
    int previous = -1;
    const bool gotPrevious = cudaGetDevice(&previous) == cudaSuccess;
    const bool selected = deviceIndex_ >= 0 && cudaSetDevice(deviceIndex_) == cudaSuccess;
    if (!selected) {
        pinned_ = nullptr;
        pinnedPermit_.Abandon();
        return;
    }
    if (cudaFreeHost(pinned_) == cudaSuccess) pinnedPermit_.Release();
    else pinnedPermit_.Abandon();
    pinned_ = nullptr;
    if (gotPrevious && previous != deviceIndex_) cudaSetDevice(previous);
}

CudaParameterStatus CudaGroomScalarReadback::BeginFresh(
    std::vector<CudaParameterField> const& fields,
    std::vector<CudaGroomScalarRequest> const& requests, cudaStream_t stream,
    UsdGenExecutionMemoryReservation* reservation) {
    if (pendingFresh_ || unprovenWork_ || requests.size() > Capacity)
        return CudaParameterStatus::InvalidArgument;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
        return CudaParameterStatus::CudaError;
    if (capture != cudaStreamCaptureStatusNone)
        return CudaParameterStatus::InvalidArgument;
    int current = -1, streamDevice = -1;
    if (cudaGetDevice(&current) != cudaSuccess ||
        (stream && cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess))
        return CudaParameterStatus::CudaError;
    if ((stream && streamDevice != current) ||
        (deviceIndex_ >= 0 && deviceIndex_ != current))
        return CudaParameterStatus::InvalidArgument;

    std::array<std::string_view, Capacity> requestNames{};
    std::array<void const*, Capacity> sources{};
    std::array<size_t, Capacity> bytes{};
    // Validate all request identities independently of field presence. A
    // missing duplicate must not be silently accepted merely because it has
    // no D2H copy in this batch.
    for (size_t requestIndex = 0; requestIndex != requests.size(); ++requestIndex) {
        auto const& request = requests[requestIndex];
        const size_t byteCount = ScalarBytes(request.type);
        if (request.destination.IsEmpty() || !byteCount || byteCount > pending_[0].bytes.size())
            return CudaParameterStatus::InvalidArgument;
        std::string_view const destination = CanonicalName(request.destination);
        for (size_t i = 0; i != requestIndex; ++i)
            if (requestNames[i] == destination ||
                (request.slot != kUsdGenNoParam && requests[i].slot == request.slot))
                return CudaParameterStatus::InvalidArgument;
        requestNames[requestIndex] = destination;
    }

    size_t pendingCount = 0;
    for (size_t requestIndex = 0; requestIndex != requests.size(); ++requestIndex) {
        auto const& request = requests[requestIndex];
        std::string_view const destination = requestNames[requestIndex];
        const size_t byteCount = ScalarBytes(request.type);
        CudaParameterField const* field = nullptr;
        for (auto const& candidate : fields) {
            if (request.slot != kUsdGenNoParam ? candidate.slot != request.slot
                                               : CanonicalName(candidate.destination) != destination)
                continue;
            // Published evaluator fields must be unique. Do not select an
            // arbitrary duplicate if a malformed caller bypasses that layer.
            if (field) return CudaParameterStatus::InvalidArgument;
            field = &candidate;
        }
        // Missing controls are intentionally absent from this compact packet;
        // do not create a stale metadata/null-source hole in its D2H layout.
        if (!field) continue;
        if (field->domain != Domain::Groom || field->count != 1 ||
            field->components != 1 || field->type != request.type || !field->data)
            return CudaParameterStatus::InvalidArgument;
        cudaPointerAttributes attributes{};
        if (cudaPointerGetAttributes(&attributes, field->data) != cudaSuccess ||
            attributes.type != cudaMemoryTypeDevice || attributes.device != current)
            return CudaParameterStatus::InvalidArgument;
        pending_[pendingCount].destination = request.destination;
        pending_[pendingCount].type = request.type;
        pending_[pendingCount].slot = request.slot;
        pending_[pendingCount].bytes.fill(0);
        sources[pendingCount] = field->data;
        bytes[pendingCount] = byteCount;
        ++pendingCount;
    }

    if (!pinned_ && pendingCount) {
        if (s_failNextGroomScalarReadbackAllocation.exchange(false, std::memory_order_acq_rel))
            return CudaParameterStatus::CudaError;
        auto permit = gpu::TryReserveCudaExecutionBytes(
            Capacity * pending_[0].bytes.size(), UsdGenExecutionResourceKind::Scratch,
            reservation);
        if (!permit) return CudaParameterStatus::CudaError;
        unsigned char* allocation = nullptr;
        cudaError_t const status = cudaHostAlloc(reinterpret_cast<void**>(&allocation),
            Capacity * pending_[0].bytes.size(), cudaHostAllocDefault);
        if (status != cudaSuccess) {
            if (allocation && cudaFreeHost(allocation) != cudaSuccess) permit->Abandon();
            return CudaParameterStatus::CudaError;
        }
        pinned_ = allocation;
        pinnedPermit_ = std::move(*permit);
        deviceIndex_ = current;
    }

    pendingCount_ = pendingCount;
    pendingFresh_ = true;
    freshFailed_ = false;
    if (!pendingCount) return CudaParameterStatus::Ok;
    unprovenWork_ = true;
    for (size_t i = 0; i != pendingCount; ++i) {
        if (i == 1 && s_failNextGroomScalarReadbackCopyAfterOne.exchange(
                          false, std::memory_order_acq_rel)) {
            freshFailed_ = true;
            return CudaParameterStatus::CudaError;
        }
        if (cudaMemcpyAsync(pinned_ + i * pending_[0].bytes.size(), sources[i],
                bytes[i], cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            freshFailed_ = true;
            return CudaParameterStatus::CudaError;
        }
    }
    return CudaParameterStatus::Ok;
}

CudaParameterStatus CudaGroomScalarReadback::CommitFreshFinish() {
    if (!pendingFresh_) return CudaParameterStatus::InvalidArgument;
    // Native terminal proof is owned by the parent. A successful Begin keeps
    // this flag set until this host-only commit; only a latched enqueue error
    // makes the packet unsafe to inspect.
    if (freshFailed_) return CudaParameterStatus::CudaError;
    for (size_t i = 0; i != pendingCount_; ++i) {
        pending_[i].bytes.fill(0);
        std::memcpy(pending_[i].bytes.data(), pinned_ + i * pending_[0].bytes.size(),
                    ScalarBytes(pending_[i].type));
    }
    committed_ = pending_;
    committedCount_ = pendingCount_;
    pendingCount_ = 0;
    pendingFresh_ = false;
    unprovenWork_ = false;
    freshFailed_ = false;
    return CudaParameterStatus::Ok;
}

void failNextCudaGroomScalarReadbackAllocationForTesting() {
    s_failNextGroomScalarReadbackAllocation.store(true, std::memory_order_release);
}

void failNextCudaGroomScalarReadbackCopyAfterOneForTesting() {
    s_failNextGroomScalarReadbackCopyAfterOne.store(true, std::memory_order_release);
}

CudaGroomScalarValue const* CudaGroomScalarReadback::Find(TfToken const& destination) const {
    std::string_view const canonical = CanonicalName(destination);
    for (size_t i = 0; i != committedCount_; ++i)
        if (CanonicalName(committed_[i].destination) == canonical) return &committed_[i];
    return nullptr;
}

CudaGroomScalarValue const* CudaGroomScalarReadback::Find(UsdGenParamSlot slot) const {
    if (slot == kUsdGenNoParam) return nullptr;
    for (size_t i = 0; i != committedCount_; ++i)
        if (committed_[i].slot == slot) return &committed_[i];
    return nullptr;
}

CudaParameterStatus CudaParameterPlan::Compile(
    UsdGenGraphSpec const& graph, UsdGenOpNode const& node,
    CudaParameterPlan* output, std::vector<std::string>* diagnostics)
{
    if (!output) {
        Error(diagnostics, "null parameter plan");
        return CudaParameterStatus::InvalidArgument;
    }
    std::shared_ptr<const CudaParameterProgram> program;
    CudaParameterStatus const status =
        CudaParameterProgram::Compile(graph, node, &program, diagnostics);
    if (status != CudaParameterStatus::Ok) return status;
    try {
        auto evaluator = std::make_unique<CudaParameterEvaluator>();
        output->program_ = std::move(program);
        output->evaluator_ = std::move(evaluator);
    } catch (...) {
        Error(diagnostics, "failed to allocate CUDA parameter evaluator");
        return CudaParameterStatus::CudaError;
    }
    return CudaParameterStatus::Ok;
}

CudaParameterStatus CudaParameterPlan::Evaluate(
    gpu::DeviceCurveGeometryView geometry, gpu::ExpressionGeometryChannels channels,
    expr::Context controls, cudaStream_t stream,
    std::vector<std::string>* diagnostics,
    UsdGenExecutionMemoryReservation* reservation, UsdGenExecutionResourceKind kind)
{
    if (!program_ || !evaluator_) {
        Error(diagnostics, "parameter plan has no compiled program");
        return CudaParameterStatus::InvalidArgument;
    }
    return evaluator_->Evaluate(*program_, geometry, channels, controls,
                                stream, diagnostics, reservation, kind);
}

CudaParameterField const* CudaParameterPlan::Find(
    TfToken const& destination) const
{
    return evaluator_ ? evaluator_->Find(destination) : nullptr;
}
} // namespace usdGen
#endif
