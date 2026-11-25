// usdGen engine — operator kernel interface (03-execution-engine.md §8.1).
//
// Contract rules (03 §8.1 table):
//  * Evaluate is const, allocates nothing, touches no stage, reads nothing
//    outside view/capture/ctx (S8).
//  * Capture may allocate into its own node's capture and buffer under the
//    graph's scheduled work owner; it runs there or inside a parallel region
//    over independent nodes in the private arena (R22).
//  * The record's parameter decls are the node's mapped property set (S14).
//  * Cross-curve reads come from capture data plus reference buffers, never
//    from another chunk (I3).
//  * Kernels live in usdGenMath as free functions over raw pointers, compiled
//    -ffp-contract=off, deterministic reductions.
#ifndef USDGEN_OP_H
#define USDGEN_OP_H

#include "usdGen/cpuParameters.h"
#include "usdGen/curveBuffer.h"
#include "usdGen/graphDesc.h"
#include "usdGen/pool.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/token.h"
#include "usdGen/api/elementSet.h"
#include "usdGen/exec/chunkView.h"

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

/// The exec-phase token rule: true (and an error on diag) when the phase
/// `check` brackets interned a token; with USDGEN_TOKEN_CHECKS=0 a warning
/// on diag and false.
inline bool UsdGenCookInternFails(UsdGenCookTokenCheck const &check, UsdGenDiagnostics *diag)
{
    if (!check.Violated()) return false;
    char const *const message = "usdGen: a token was interned during the cook";
    bool const fail = UsdGenTokenChecksEnabled();
    if (diag) {
        if (fail) diag->Error(message);
        else diag->Warn(message);
    }
    return fail;
}

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

/// One declared parameter resolved at compile (UsdGenResolveParams).
struct UsdGenResolvedParam
{
    UsdGenParamValue const *authored = nullptr;   // first param with exactly the decl name
    TfToken token;                                // token value; a std::string is interned at compile
    int32_t enumIndex = -1;                       // decl enumerant index; -1 none/unmatched
};

struct UsdGenParamView
{
    UsdGenGraphDesc const *desc = nullptr;
    UsdGenNodeDesc const  *node = nullptr;
    /// The node's op record and its resolved slots (op->ParamCount()
    /// entries); null for views built outside the compiler.
    UsdGenOpRecord const *op = nullptr;
    UsdGenResolvedParam const *slots = nullptr;
    /// This node's evaluated connected parameters, or null when the node has
    /// none (or when the caller is a validator that never evaluates them).
    UsdGenCpuParameters const *expressions = nullptr;

    // ---- by slot ---------------------------------------------------------------
    UsdGenResolvedParam const *Slot(UsdGenParamSlot s) const noexcept
    {
        return slots && op && s < op->ParamCount() ? &slots[s] : nullptr;
    }
    bool IsAuthored(UsdGenParamSlot s) const noexcept
    {
        UsdGenResolvedParam const *r = Slot(s);
        return r && r->authored;
    }
    UsdGenExpressionValue const *Expression(UsdGenParamSlot s) const noexcept
    {
        return expressions && Slot(s) ? expressions->FindSlot(s) : nullptr;
    }
    bool GroomExpressionValue(UsdGenParamSlot s, double *out) const
    {
        return GroomValue(Expression(s), out);
    }
    double GetDoubleLiteral(UsdGenParamSlot s, double fallback) const noexcept
    {
        UsdGenResolvedParam const *r = Slot(s);
        return DoubleLiteral(r ? r->authored : nullptr, fallback);
    }
    double GetDouble(UsdGenParamSlot s, double fallback) const
    {
        double expression = 0.0;
        if (GroomExpressionValue(s, &expression)) return expression;
        return GetDoubleLiteral(s, fallback);
    }
    int GetInt(UsdGenParamSlot s, int fallback) const
    {
        double expression = 0.0;
        if (GroomExpressionValue(s, &expression)) return static_cast<int>(expression);
        UsdGenResolvedParam const *r = Slot(s);
        return IntLiteral(r ? r->authored : nullptr, fallback);
    }
    bool GetBool(UsdGenParamSlot s, bool fallback) const
    {
        double expression = 0.0;
        if (GroomExpressionValue(s, &expression)) return expression != 0.0;
        UsdGenResolvedParam const *r = Slot(s);
        return BoolLiteral(r ? r->authored : nullptr, fallback);
    }
    /// The decl enumerant index of an authored token value; fallback when
    /// unauthored or not one of the decl's enumerants.
    int GetEnum(UsdGenParamSlot s, int fallback) const noexcept
    {
        UsdGenResolvedParam const *r = Slot(s);
        return r && r->enumIndex >= 0 ? r->enumIndex : fallback;
    }
    TfToken const &GetToken(UsdGenParamSlot s, TfToken const &fallback) const noexcept
    {
        UsdGenResolvedParam const *r = Slot(s);
        return r && HoldsToken(r->authored) ? r->token : fallback;
    }
    VtValue const &GetValue(UsdGenParamSlot s, VtValue const &fallback) const noexcept
    {
        UsdGenResolvedParam const *r = Slot(s);
        return r && r->authored ? r->authored->value : fallback;
    }
    UsdGenParamField GetScalarField(UsdGenParamSlot s, double fallback) const
    {
        return Field(GetDoubleLiteral(s, fallback), Expression(s));
    }

private:
    static bool HoldsToken(UsdGenParamValue const *p) noexcept
    {
        return p && (p->value.IsHolding<TfToken>() || p->value.IsHolding<std::string>());
    }
    static bool GroomValue(UsdGenExpressionValue const *value, double *out) noexcept
    {
        if (!value || value->domain != expr::Domain::Groom || value->values.empty())
            return false;
        *out = value->values.front();
        return true;
    }
    static UsdGenParamField Field(double literal, UsdGenExpressionValue const *value) noexcept
    {
        UsdGenParamField field;
        field.literal = literal;
        if (!value || value->values.empty()) return field;
        field.connected = true;
        field.values = value->values.data();
        field.count = value->count;
        field.components = value->components;
        field.domain = value->domain;
        return field;
    }
    static double DoubleLiteral(UsdGenParamValue const *p, double fallback) noexcept
    {
        if (p) {
            if (p->value.IsHolding<double>()) return p->value.UncheckedGet<double>();
            if (p->value.IsHolding<float>())  return p->value.UncheckedGet<float>();
        }
        return fallback;
    }
    static int IntLiteral(UsdGenParamValue const *p, int fallback) noexcept
    {
        if (p) {
            if (p->value.IsHolding<int>()) return p->value.UncheckedGet<int>();
            // Integer-valued authored VtValues are tolerated (schema declares
            // e.g. usdGen:seed as `uniform int`; callers may author the
            // unsigned spelling of the same value).
            if (p->value.IsHolding<uint32_t>())
                return static_cast<int>(p->value.UncheckedGet<uint32_t>());
        }
        return fallback;
    }
    static bool BoolLiteral(UsdGenParamValue const *p, bool fallback) noexcept
    {
        if (p && p->value.IsHolding<bool>()) return p->value.UncheckedGet<bool>();
        return fallback;
    }
};

/// Compile phase: resolve node.params into out[0, rec.ParamCount()). A param
/// binds the slot declared with exactly its name; the first one wins.
USDGEN_CORE_API void UsdGenResolveParams(UsdGenOpRecord const &rec, UsdGenNodeDesc const &node,
                                         UsdGenResolvedParam *out);

/// Stack resolution for validators that have no compiled node. Compile
/// phase only (about 3 KB of stack).
class UsdGenStackParams
{
public:
    UsdGenStackParams(UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
                      UsdGenOpRecord const &rec)
    {
        UsdGenResolveParams(rec, node, _slots);
        _view.desc = &desc;
        _view.node = &node;
        _view.op = &rec;
        _view.slots = _slots;
    }
    UsdGenStackParams(UsdGenStackParams const &) = delete;
    UsdGenStackParams &operator=(UsdGenStackParams const &) = delete;
    UsdGenParamView const &View() const noexcept { return _view; }
private:
    UsdGenResolvedParam _slots[kUsdGenMaxParamsPerOp];
    UsdGenParamView _view;
};

/// Opaque per-node capture payload; operators derive their own concrete
/// payload types (curve generators: UsdGenCurveCapture, usdGenCurves/types/
/// curveCapture.h). Lifetime: owned by the compiled node, keyed by the
/// node's capture epoch (03 §3.4).
struct UsdGenCapture : UsdGenPooled   // pool-allocated, recycled across recaptures
{
    virtual ~UsdGenCapture() = default;
    /// True when this capture is usable for the given upstream topology;
    /// false forces a re-capture.
    virtual bool ValidForTopology(UsdGenElementSetView const &) const { return true; }
    /// Owning captures (generators, Resample, frozen Freeze): the set this
    /// capture produced, the node's output base; null for every other capture.
    virtual UsdGenRef<const UsdGenElementSet> Output() const { return {}; }
    /// True when Capture recorded per-chunk extents into the scheduler's
    /// slots. Consumed only through the interleave's span revalidation, so
    /// a stale true on a reused capture fails closed into the point pass.
    virtual bool RecordedChunkExtents() const { return false; }
    /// True when Capture wrote the node's output point planes direct
    /// (UsdGenCaptureContext::outPx/outPy/outPz) instead of a capture
    /// buffer, so the sweep has nothing to run. The scheduler consults
    /// this only for a capture taken this run.
    virtual bool WroteDirectOutput() const { return false; }
    /// The 257-entry LUT over hairT Evaluate reads, or null.
    virtual float const *RampLut() const { return nullptr; }
    /// Type-erased copy for the incremental recompile path (E-6);
    /// concrete payloads override.
    virtual std::unique_ptr<UsdGenCapture> Clone() const { return nullptr; }
    /// Recycling (arenas through retirement): a capture that returns true
    /// here is kept as its node's spare when a newer capture replaces it,
    /// and the next re-capture fills the spare instead of a fresh payload.
    virtual bool Recyclable() const { return false; }
    /// Back to the default-constructed state, keeping array storage that
    /// this capture owns alone (VtArray::clear). Called on the commit thread
    /// right before Capture refills it, so every array a newer generation
    /// still shares has been released by then. False: make a fresh one.
    virtual bool Recycle() { return false; }
};

/// The empty input: what Primary() reads when a node has none.
inline constexpr UsdGenElementSetView kUsdGenNoInput{};

/// The geometry an op's Capture reads: one view per geometry input, in order.
struct UsdGenOpInputs
{
    UsdGenElementSetView const *sets = nullptr;
    uint32_t count = 0;
    UsdGenElementSetView const &Primary() const noexcept
    {
        return sets && count ? sets[0] : kUsdGenNoInput;
    }
};

struct UsdGenCaptureContext
{
    UsdGenGraphDesc const *desc = nullptr;   // surfaces, maps, density scales
    // Prior committed payload for read-only reuse during a recapture. It
    // remains owned by the node until Capture succeeds and commits.
    UsdGenCapture const *previousCapture = nullptr;
    UsdGenParamView const *params = nullptr; // this node's resolved parameters and ramps
    UsdGenReferenceSet const **references = nullptr; // resolved record ReferenceInputs(), evaluated
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
    // The ordered geometry inputs, for CaptureDigest (Capture takes them
    // as its own argument).
    UsdGenOpInputs         inputs;
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

/// What Configure sees: the node's parameters and the compiler's indexed
/// plane allocation (4a).
struct UsdGenConfigureContext
{
    UsdGenParamView const &params;
    int indexedPlaneBase = -1;
};

/// The operator kernel interface. Everything static about an op (type,
/// topology effect, role, arity, traits, parameters, planes) lives in its
/// registered record (registries/opDef.h); the instance holds only state.
class UsdGenOp : public UsdGenPooled
{
public:
    explicit UsdGenOp(UsdGenOpRecord const &record) noexcept : _record(&record) {}
    virtual ~UsdGenOp() = default;
    UsdGenOpRecord const &Record() const noexcept { return *_record; }

    /// Instance state from structural params, once per fresh node and on a
    /// value-only reuse. Plane names come from the record's hooks.
    virtual void Configure(UsdGenConfigureContext const &) {}

    // ---- binding -------------------------------------------------------------
    virtual bool Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag) = 0;

    // ---- capture: topology-dependent, cached by epoch (03 §3.4) --------------
    virtual UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &ctx) const = 0;
    /// Fill *out (derive the concrete payload). A generator (topoFx != None)
    /// also produces its output topology (UsdGenCurveCapture::MutableBuffer).
    virtual bool Capture(UsdGenCaptureContext const &ctx, UsdGenOpInputs const &in,
                         UsdGenCapture *out, UsdGenDiagnostics *diag) = 0;

    // ---- evaluate: per chunk, parallel, allocation-free (03 §8.1) ------------
    virtual void Evaluate(UsdGenEvalContext const &ctx, UsdGenCapture const &capture,
                          UsdGenChunkView const &chunk) const = 0;

    /// Allocate the concrete capture payload for this operator. The payload
    /// owns its per-curve results; generators additionally fill
    /// UsdGenCurveCapture::MutableBuffer() (OwnsBuffer() == true).
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
    // Native clock consumers invalidate only their own evaluation and its
    // descendants. Animated parameters and external inputs have their own
    // value/capture identities and do not need this flag.
    virtual bool ReadsTime(UsdGenParamView const &) const { return false; }

private:
    UsdGenOpRecord const *_record;
};


}  // namespace usdGen

#endif  // USDGEN_OP_H
