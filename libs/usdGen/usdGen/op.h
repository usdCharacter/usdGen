// usdGen engine — operator kernel interface (03-execution-engine.md §8.1).
//
// Contract rules (03 §8.1 table):
//  * Evaluate is const, allocates nothing, touches no stage, reads nothing
//    outside view/capture/ctx (S8).
//  * Capture may allocate into its own node's capture and buffer under the
//    graph's scheduled work owner; it runs there or inside a tbb::parallel_for
//    over independent nodes in the private arena (R22).
//  * TopologyParameters() ∪ ValueParameters() == the node's mapped property set
//    (S14); asserted in debug builds (USDGEN_OP_CHECKS=1).
//  * Cross-curve reads come from capture data plus reference buffers, never
//    from another chunk (I3).
//  * Kernels live in usdGenMath as free functions over raw pointers, compiled
//    -ffp-contract=off, deterministic reductions.
#ifndef USDGEN_OP_H
#define USDGEN_OP_H

#include "usdGen/cpuParameters.h"
#include "usdGen/curveBuffer.h"
#include "usdGen/graphDesc.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"

#include <memory>
#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenWorkDispatcher;  // defined in usdGen/scheduler.h

/// Diagnostics collected by a node; the imaging layer is the one that logs
/// (the engine core stays TF_WARN-free on hot paths, 03 §9.2).
struct UsdGenDiagnostics
{
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    bool HasErrors() const { return !errors.empty(); }
    void Error(std::string msg)    { errors.push_back(std::move(msg)); }
    void Warn(std::string msg)     { warnings.push_back(std::move(msg)); }
};

/// This node's resolved parameters and ramps, handed in by the scheduler at
/// Bind time. Ramps arrive already resolved (03 §2.2 UsdGenRampDesc); the
/// adapter does the S11 encoding, the kernel does the LUT (257 entries).
///
/// 26.08 VtValue note: there is no GetAsSafe/SafeGet; the getters below use
/// IsHolding<T>() + UncheckedGet<T>() with the C1 default as fallback.
/// A parameter that may be a literal or a connected expression. `Value` is the
/// ONE indexing rule, and it is the rule gpu::ReadScalar/ReadBool/ReadInt
/// implement: a groom-domain result broadcasts, a primitive-domain result
/// indexes by ABSOLUTE curve index, a point-domain result by ABSOLUTE CV
/// index. Operators pass `view->desc->firstCurve + c` and
/// `view->desc->firstCv + o`, exactly as they already do for whole-buffer
/// capture payloads.
struct UsdGenParamField
{
    double literal = 0.0;
    double const *values = nullptr;
    size_t count = 0;
    uint32_t components = 1;
    expr::Domain domain = expr::Domain::Groom;
    bool connected = false;

    double Value(size_t curve, size_t point, uint32_t component = 0) const
    {
        if (!connected || !values || component >= components) return literal;
        const size_t index = domain == expr::Domain::Primitive ? curve
            : domain == expr::Domain::Point ? point : size_t(0);
        if (index >= count) return literal;
        return values[index * components + component];
    }
    /// True when every element carries the same value, so a kernel may hoist
    /// the read out of its inner loop.
    bool Uniform() const { return !connected || domain == expr::Domain::Groom; }
};

struct UsdGenParamView
{
    UsdGenGraphDesc const *desc = nullptr;
    UsdGenNodeDesc const  *node = nullptr;
    /// This node's evaluated connected parameters, or null when the node has
    /// none (or when the caller is a validator that never evaluates them).
    UsdGenCpuParameters const *expressions = nullptr;

    /// nullptr when the property is not authored and carries no C1 default in
    /// the node description (the builder materializes C1 defaults).
    UsdGenParamValue const *FindParam(TfToken const &name) const
    {
        if (!node) return nullptr;
        for (auto const &p : node->params) if (p.name == name) return &p;
        return nullptr;
    }

    /// The evaluated connected values for `name`, or nullptr.
    UsdGenExpressionValue const *FindExpression(TfToken const &name) const
    {
        return expressions ? expressions->Find(name) : nullptr;
    }

    /// The groom-domain (single-value) result of a connected parameter, when
    /// it has one. Scalars read through GetDouble/GetInt/GetBool pick this up
    /// automatically, so a groom-granularity expression drives every operator
    /// control without the operator knowing that expressions exist.
    bool GroomExpressionValue(TfToken const &name, double *out) const
    {
        UsdGenExpressionValue const *value = FindExpression(name);
        if (!value || value->domain != expr::Domain::Groom || value->values.empty())
            return false;
        *out = value->values.front();
        return true;
    }

    /// A connectable scalar parameter as a field. Falls back to the authored
    /// literal at every element when nothing is connected.
    UsdGenParamField GetScalarField(TfToken const &name, double fallback) const
    {
        UsdGenParamField field;
        field.literal = GetDoubleLiteral(name, fallback);
        UsdGenExpressionValue const *value = FindExpression(name);
        if (!value || value->values.empty()) return field;
        field.connected = true;
        field.values = value->values.data();
        field.count = value->count;
        field.components = value->components;
        field.domain = value->domain;
        return field;
    }
    TfToken GetToken(TfToken const &name, TfToken fallback) const
    {
        if (UsdGenParamValue const *p = FindParam(name)) {
            if (p->value.IsHolding<TfToken>())
                return p->value.UncheckedGet<TfToken>();
            if (p->value.IsHolding<std::string>())
                return TfToken(p->value.UncheckedGet<std::string>().c_str());
        }
        return fallback;
    }
    /// The authored value only, never an expression result. Kernels that read
    /// a connectable parameter per element use GetScalarField instead; this is
    /// the literal that expression gets as $value.
    double GetDoubleLiteral(TfToken const &name, double fallback) const
    {
        if (UsdGenParamValue const *p = FindParam(name)) {
            if (p->value.IsHolding<double>()) return p->value.UncheckedGet<double>();
            if (p->value.IsHolding<float>())  return p->value.UncheckedGet<float>();
        }
        return fallback;
    }
    double GetDouble(TfToken const &name, double fallback) const
    {
        double expression = 0.0;
        if (GroomExpressionValue(name, &expression)) return expression;
        return GetDoubleLiteral(name, fallback);
    }
    int GetInt(TfToken const &name, int fallback) const
    {
        double expression = 0.0;
        if (GroomExpressionValue(name, &expression)) return static_cast<int>(expression);
        if (UsdGenParamValue const *p = FindParam(name)) {
            if (p->value.IsHolding<int>()) return p->value.UncheckedGet<int>();
            // Integer-valued authored VtValues are tolerated (schema declares
            // e.g. usdGen:seed as `uniform int`; callers may author the
            // unsigned spelling of the same value).
            if (p->value.IsHolding<uint32_t>())
                return static_cast<int>(p->value.UncheckedGet<uint32_t>());
        }
        return fallback;
    }
    bool GetBool(TfToken const &name, bool fallback) const
    {
        double expression = 0.0;
        if (GroomExpressionValue(name, &expression)) return expression != 0.0;
        if (UsdGenParamValue const *p = FindParam(name)) {
            if (p->value.IsHolding<bool>()) return p->value.UncheckedGet<bool>();
        }
        return fallback;
    }
    VtValue GetVtValue(TfToken const &name, VtValue const &fallback) const
    {
        if (UsdGenParamValue const *p = FindParam(name)) return p->value;
        return fallback;
    }
};

/// Opaque per-node capture payload; M1 operators derive their own concrete
/// payload types. Lifetime: owned by the compiled node, keyed by the node's
/// capture epoch (03 §3.4).
struct UsdGenCapture
{
    virtual ~UsdGenCapture() = default;
    /// True when this capture is usable for the given upstream topology
    /// (curve/CV counts); false forces a re-capture.
    virtual bool ValidForTopology(UsdGenCurveBuffer const &) const { return true; }
    /// The buffer this capture owns/wraps (a generator's output topology);
    /// default-constructed (empty) for non-generator captures.
    virtual UsdGenCurveBuffer const &Buffer() const { return _buffer; }
    /// True for generator captures: the framework installs Buffer() into the
    /// owning node after a successful Capture.
    virtual bool OwnsBuffer() const { return false; }
    /// True when Capture recorded per-chunk extents into the scheduler's
    /// slots. Consumed only through the interleave's span revalidation, so
    /// a stale true on a reused capture fails closed into the point pass.
    virtual bool RecordedChunkExtents() const { return false; }
    /// True when Capture wrote the node's output point planes direct
    /// (UsdGenCaptureContext::outPx/outPy/outPz) instead of a capture
    /// buffer, so the sweep has nothing to run. The scheduler consults
    /// this only for a capture taken this run.
    virtual bool WroteDirectOutput() const { return false; }
    UsdGenCurveBuffer &MutableBuffer() { return _buffer; }
    /// Type-erased copy for the incremental recompile path (E-6);
    /// concrete payloads override.
    virtual std::unique_ptr<UsdGenCapture> Clone() const { return nullptr; }
protected:
    UsdGenCurveBuffer _buffer;
};

/// M1 shared capture payload base: op-specific per-curve / per-CV scalars
/// resolved once per capture epoch (I4).
/// Copyable (VtArray CoW); every CONCRETE payload must override Clone() with
/// a full copy of its own derived type — a base-level clone would silently
/// slice the op-specific fields (the E-6 reuse defect this closes).
struct UsdGenCapturePayload : public UsdGenCapture
{
    VtFloatArray perCurve;      // op-specific meaning (target length, ...)
    VtFloatArray perCv;         // op-specific meaning (noise displacement field, ...)
    VtFloatArray rampLut;       // 257-entry op-specific LUT over hairT; empty == unused

    /// Abstract: every concrete payload overrides this with a full-type copy
    /// (`return std::make_unique<Self>(*this);` — see ops/*.cpp).
    std::unique_ptr<UsdGenCapture> Clone() const override = 0;
};

struct UsdGenCaptureContext
{
    UsdGenGraphDesc const *desc = nullptr;   // surfaces, maps, density scales
    // Prior committed payload for read-only reuse during a recapture. It
    // remains owned by the node until Capture succeeds and commits.
    UsdGenCapture const *previousCapture = nullptr;
    UsdGenParamView const *params = nullptr; // this node's resolved parameters and ramps
    UsdGenReferenceSet const **references = nullptr; // resolved ReferenceInputs(), evaluated
    UsdGenResolvedReferenceValue const **resolvedReferences = nullptr;
    uint32_t              referenceCount = 0;
    UsdGenResolvedMapValue const **maps = nullptr;
    uint32_t              mapCount = 0;
    // Typed slots are carried by the compiled graph.  Sampling is deliberately
    // not implemented here; legacy maps/mapCount stay available to existing
    // CPU operators until a backend consumes the purpose-aware transport.
    UsdGenMapBindingDesc const *mapBindings = nullptr;
    uint32_t              mapBindingCount = 0;
    UsdGenSurfaceId        surface = 0;
    uint32_t               seed = 0;
    uint64_t               upstreamGeneration = 0; // upstream buffer topologyVersion
    // Complete ordered input list for a multi-input operator.  Unary kernels
    // continue to use the legacy Capture() argument and upstreamGeneration;
    // fan-in kernels opt in by declaring GeometryInputArity() > 1.
    UsdGenCurveBuffer const **upstreams = nullptr;
    uint32_t               upstreamCount = 0;
    UsdGenWorkDispatcher  *dispatcher = nullptr;  // capture may parallelise in the arena
    UsdGenDiagnostics     *diag = nullptr;
    // The node's chunk partition plus the scheduler's per-chunk fused-extent
    // slots. Set only by UsdGenScheduler::Run (slots only for the armed
    // extent writer); every other Capture caller leaves them null, and a
    // capture that records extents must fail closed to not recording unless
    // all four are present and chunkCount == chunkExtentCount.
    UsdGenChunkDesc const *chunks = nullptr;
    size_t                 chunkCount = 0;
    UsdGenChunkExtent     *chunkExtentSlots = nullptr;
    size_t                 chunkExtentCount = 0;
    // The node's previous-run point planes (writable) for capture-direct
    // output, plus their CV count. Set only by UsdGenScheduler::Run when
    // the planes exactly match the node's current totals; a capture that
    // writes direct must additionally verify the count against its own
    // upstream, the planes' mutual distinctness, and that they do not
    // alias the upstream planes. Null everywhere else (including
    // progressive cooks, which keep the sweep path).
    float                 *outPx = nullptr, *outPy = nullptr, *outPz = nullptr;
    size_t                 outPlaneCvs = 0;
};

struct UsdGenEvalContext
{
    double time = 0.0;
    double shutterOffset = 0.0;              // 0 in P0/P1 (03 §7)
    UsdGenGraphDesc const *desc = nullptr;   // surface points already resolved to time+offset
    UsdGenParamView const *params = nullptr;
    UsdGenReferenceSet const **references = nullptr;
    UsdGenResolvedReferenceValue const **resolvedReferences = nullptr;
    uint32_t              referenceCount = 0;
    UsdGenResolvedMapValue const **maps = nullptr;
    uint32_t              mapCount = 0;
    UsdGenMapBindingDesc const *mapBindings = nullptr;
    uint32_t              mapBindingCount = 0;
    uint32_t seed = 0;
    // Kernels cannot append diagnostics from parallel chunk workers.  They
    // publish the first deterministic failure code here; the commit thread
    // converts it to a failed cook after the sweep joins.
    std::atomic<uint32_t> *failureCode = nullptr;
};

/// The operator kernel interface. Five M1 types: UsdGenScatterOp,
/// UsdGenGrowOp, UsdGenNoiseOp, UsdGenLengthOp, UsdGenWidthOp (02 §2.6/§2.7.1).
class UsdGenOp
{
public:
    virtual ~UsdGenOp() = default;

    // ---- static description, read at compile -------------------------------
    /// C1 type name, e.g. "UsdGenScatter".
    virtual TfToken Type() const = 0;
    /// Static TopologyEffect (R14): e.g. UsdGenLengthOp is always CurveCount
    /// because cull mode exists.
    virtual UsdGenTopoFx TopologyEffect() const { return UsdGenTopoFx::None; }
    virtual UsdGenRole Role() const { return UsdGenRole::Curves; }
    /// Evaluate reads the live cook time (and potentially seconds via the
    /// description's timeCodesPerSecond), independent of authored samples.
    virtual bool ReadsTime() const { return false; }
    /// Capture reads scene inputs sampled at the current frame. The imaging
    /// owner must refresh the description on frame-only scene-global dirties.
    virtual bool SamplesFrameInputs() const { return false; }
    /// C1-frozen parameter partition (02 §6 rows; docs/freezes/C1.md):
    /// TopologyParameters = capture/topology-class edits, ValueParameters =
    /// per-frame edits. The union must equal the node's mapped property set
    /// (S14); usdGen:enabled joins whichever set its dirty class implies
    /// (02 §6.2 topology for generators/Length, §6.3 value-toggle for the rest).
    virtual TfSpan<const TfToken> TopologyParameters() const = 0;
    virtual TfSpan<const TfToken> ValueParameters() const = 0;
    /// Named reference inputs (guide sets, clump centres, card roots).
    virtual TfSpan<const TfToken> ReferenceInputs() const { return {}; }
    /// Extra planes this operator writes, in slot order (R24 fixed types/arity).
    virtual TfSpan<const TfToken> OutputPrimvars() const { return {}; }
    /// Upstream planes this operator reads, in slot order.
    virtual TfSpan<const TfToken> InputPrimvars() const { return {}; }
    /// True when Evaluate remaps every inherited vertex-domain extra plane.
    /// The scheduler then materializes private outputs and exposes matching
    /// input/output descriptors through UsdGenChunkView.
    virtual bool RemapsVertexPlanes() const { return false; }
    /// Exact number of geometry-producing usdGen:input edges consumed by
    /// this kernel. This is deliberately separate from IsGenerator(): Grow
    /// owns its generated strand topology, but still consumes one upstream
    /// root geometry. Current kernels are either source operators (0) or
    /// unary operators (1); a future multi-input kernel must declare and
    /// implement its merge semantics before increasing this value.
    virtual size_t GeometryInputArity() const { return 1; }
    /// Whether the authored order of geometry inputs is semantic.  Unary
    /// operators retain the historical canonicalized digest behavior.
    virtual bool GeometryInputsOrdered() const { return false; }
    /// Called by the compiler once per fresh node, before OutputPrimvars() is
    /// read, for an operator whose emitted plane NAMES depend on authored
    /// values (UsdGenClump's clumpId_<level>). The parameters that feed it must
    /// be structural (digest) parameters, so an edit rebuilds the node.
    virtual void Configure(UsdGenParamView const &) {}

    // ---- binding -------------------------------------------------------------
    virtual bool Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag) = 0;

    // ---- capture: topology-dependent, cached by epoch (03 §3.4) --------------
    virtual UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &ctx) const = 0;
    /// Fill *out (derive the concrete payload). A generator (TopologyEffect != None)
    /// must also populate its output topology (chunks, cvCount, totalCurves)
    /// into *out->buffer when it creates curves.
    virtual bool Capture(UsdGenCaptureContext const &ctx,
                         UsdGenCurveBuffer const &upstream,
                         UsdGenCapture *out,
                         UsdGenDiagnostics *diag) = 0;

    // ---- evaluate: per chunk, parallel, allocation-free (03 §8.1) ------------
    virtual void Evaluate(UsdGenEvalContext const &ctx,
                          UsdGenCapture const &capture,
                          UsdGenChunkView *view) const = 0;

    // ---- generators only ------------------------------------------------------
    /// When true, Capture produces the node's topology from scratch (no
    /// upstream buffer). False for all stylers/deformers.
    virtual bool IsGenerator() const { return false; }

    /// Allocate the concrete capture payload for this operator. The payload
    /// owns its per-curve results; generators additionally fill
    /// UsdGenCapture::MutableBuffer() (OwnsBuffer() == true).
    virtual std::unique_ptr<UsdGenCapture> CreateCapture() const
    {
        return std::make_unique<UsdGenCapture>();
    }

    /// Plane bitmask (03 §8.5 envelope + tile-change tracking): which CV
    /// planes Evaluate() may write for this operator type.
    static constexpr uint32_t kPlanePoints = 1u << 0;
    static constexpr uint32_t kPlaneWidths = 1u << 1;
    static constexpr uint32_t kPlaneHairT  = 1u << 2;
    virtual uint32_t PlanesTouched() const { return 0; }
};


}  // namespace usdGen

#endif  // USDGEN_OP_H
