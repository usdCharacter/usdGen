// usdGen — UsdGenDeform. plan/examples/rbf-deformation.md.
//
// CUDA lane: the persistent RBF library in cudaExecution.cpp deforms the groom
// by samples of the bound surface; this class only supplies metadata there.
//
// CPU lane: drivers are usdGen:guides, or the bound surface when no guides
// are supplied. Surface-driven input is in the surface's rest-local domain;
// output is transformed into the current description domain. Capture
// resolves the whole result, because every CV reads the same field:
//
//   samples  up to usdGen:rbfSamples driver CVs, farthest-point sampled at
//            rest (primvars:rest, else the Default-time points), in the
//            description's space;
//   field    usdGenMath::rbf::CubicField bound at those rest positions and solved for
//            their current positions: it passes through every sample and
//            reproduces any affine motion of the drivers exactly;
//   strands  every incoming CV x moves to x + D(x), D = F - identity. With
//            usdGen:lockRoots the whole strand is shifted back by D(root), so
//            the root stays where it was and the strand bends above it.
//
// The capture digest carries the driver content, so an animated driver
// re-captures on every frame it moves; the factorization is kept while the
// rest samples do not change. Evaluate applies usdGen:mask.
#include "usdGenCurves/ops/deform.h"
#include "usdGen/api/value.h"
#include "usdGen/api/cowArray.h"
#include "usdGen/api/span.h"
#include "usdGenCurves/types/curveSetInput.h"
#include "usdGenSurface/surfaceInput.h"
#include "usdGenCurves/types/curveSet.h"
#include "usdGenCurves/ops/curveWrap.h"
#include "usdGenCurves/ops/regionMap.h"

#include "usdGenCurves/ops/opUtil.h"
#include "usdGen/api/deformBinding.h"
#include "usdGen/exec/scheduler.h"
#include "usdGen/exec/taskScheduler.h"

#include "usdGenMath/math.h"
#include "usdGen/api/trace.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include "usdGen/registries/backend.h"
#include "usdGen/registries/registrar.h"

namespace usdGen {

static_assert(sizeof(UsdGenVec3d) == 3 * sizeof(double),
              "the selection cache memcmps rest drivers bitwise");

namespace {

using namespace deformP;

// Large displace batches split over the process scheduler.
usdGenMath::rbf::ParallelFor const kDisplaceParallel{&UsdGenParallelForBlocks, nullptr};
using D = expr::Domain;
constexpr char const *kRefs[] = {"guides"};
constexpr char const *kTokens[] = {"curveWrap", "usdGen:regionMap", "periodic", "rbf"};
constexpr UsdGenExprAdmission kBoolGroom{expr::ScalarType::Bool, 1, "bool", D::Groom};
constexpr UsdGenExprAdmission kBoolRoot{expr::ScalarType::Bool, 1, "bool", D::Groom | D::Primitive};
constexpr UsdGenExprAdmission kIntGroom{expr::ScalarType::Int32, 1, "int", D::Groom};
constexpr UsdGenExprAdmission kFloatAll{expr::ScalarType::Float32, 1, "float", D::All};
UsdGenParamDecl const kParams[kCount] = {
    {"input", UsdGenParamClass::Capture},
    {"surface", UsdGenParamClass::Capture},
    {"seed", UsdGenParamClass::Capture},
    {"rbfSamples", UsdGenParamClass::Capture, true, kIntGroom},
    {"mode", UsdGenParamClass::Capture},
    {"regionMap", UsdGenParamClass::Capture},
    {"guideRegions", UsdGenParamClass::Capture},
    {"enabled", UsdGenParamClass::Value, true, kBoolGroom},
    {"lockRoots", UsdGenParamClass::Value, true, kBoolRoot},
    {"mask", UsdGenParamClass::Value, true, kFloatAll},     // operator envelope (02 §2.13)
};
constexpr UsdGenMapSlotDecl kMaps[] = {    // compiler.cpp:428-431, all four as today
    {"usdGen:map"}, {"usdGen:regionMap"}, {"usdGen:densityMap"}, {"usdGen:lengthMap"}};

std::unique_ptr<UsdGenOp> Create(UsdGenOpRecord const &r, void *)
{
    return std::make_unique<UsdGenDeformOp>(r);
}

// The strands loop overwrites every element, so the result buffer must not
// pay for zero-fill: std::vector::resize memsets 3 floats per CV (measured
// 0.17ms at 218k CVs) only to have them overwritten. This buffer resizes
// without initializing; every read below follows a write of the same triple.
class UninitFloatBuffer
{
public:
    UninitFloatBuffer() = default;
    UninitFloatBuffer(UninitFloatBuffer const &other) { assign(other); }
    UninitFloatBuffer(UninitFloatBuffer &&other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_)
    {
        other.data_ = nullptr;
        other.size_ = other.capacity_ = 0;
    }
    UninitFloatBuffer &operator=(UninitFloatBuffer const &other)
    {
        if (this != &other) assign(other);
        return *this;
    }
    UninitFloatBuffer &operator=(UninitFloatBuffer &&other) noexcept
    {
        if (this != &other) {
            std::free(data_);
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = other.capacity_ = 0;
        }
        return *this;
    }
    ~UninitFloatBuffer() { std::free(data_); }
    void clear() { size_ = 0; }
    void resizeUninit(size_t n)
    {
        if (n > capacity_) {
            float *grown =
                static_cast<float *>(std::realloc(data_, n * sizeof(float)));
            if (!grown) throw std::bad_alloc();
            data_ = grown;
            capacity_ = n;
        }
        size_ = n;
    }
    size_t size() const { return size_; }
    float *data() { return data_; }
    float const *data() const { return data_; }
    float &operator[](size_t i) { return data_[i]; }
    float const &operator[](size_t i) const { return data_[i]; }

private:
    void assign(UninitFloatBuffer const &other)
    {
        if (capacity_ < other.size_) {
            float *grown = static_cast<float *>(
                std::realloc(data_, other.size_ * sizeof(float)));
            if (!grown && other.size_) throw std::bad_alloc();
            data_ = grown;
            capacity_ = other.size_;
        }
        size_ = other.size_;
        std::memcpy(data_, other.data_, size_ * sizeof(float));
    }
    float *data_ = nullptr;
    size_t size_ = 0, capacity_ = 0;
};

struct UsdGenDeformCapture final : public UsdGenCapture
{
    UninitFloatBuffer result;                     // 3 * totalCvs, interleaved triples
    uint64_t upstreamTopologyVersion = 0;
    uint64_t upstreamValueVersion = 0;
    uint32_t upstreamCurves = 0;
    uint32_t upstreamCvs = 0;
    size_t samples = 0;                           // RBF samples used
    bool wroteDirect = false;    // output planes written, no result to sweep
    bool extentsRecorded = false;                 // per-chunk slots written

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenDeformCapture>(*this);
    }
    bool RecordedChunkExtents() const override { return extentsRecorded; }
    bool WroteDirectOutput() const override { return wroteDirect; }
    bool ValidForTopology(UsdGenElementSetView const &in) const override
    {
        UsdGenCurveSetView const upstream = UsdGenCurveSetView::Of(in);
        return upstream.topologyVersion == upstreamTopologyVersion &&
               upstream.valueVersion == upstreamValueVersion &&
               upstream.totalCurves == upstreamCurves &&
               upstream.totalCvs == upstreamCvs;
    }
};

template <class F>
void ParallelFor(UsdGenWorkDispatcher *dispatcher, size_t count, F const &body)
{
    // Bodies are strand groups (~19us at 32 strands), so a few groups
    // already outweigh the parallel-region overhead; under 3 groups the
    // groom has at most 64 strands, the old per-strand serial bound.
    // Single-group claims: at ~19us a body the default 4-group quantum
    // strands measurable wall past the last full round, while each group
    // streams its own queries (no cross-group reuse for coarser claims
    // to preserve), so the fetch_add traffic is pure profit.
    if (!dispatcher || count < 3) {
        for (size_t i = 0; i < count; ++i) body(i);
        return;
    }
    struct Payload { F const *body; } payload{&body};
    dispatcher->ParallelFor(count, [](size_t i, void *p) {
        (*static_cast<Payload *>(p)->body)(i);
    }, &payload, 1);
}

void MixArray(opUtil::Digest *d, void const *data, size_t bytes)
{
    d->MixBytes(data, bytes);
}

// Exponent-bit finiteness: all-set exponent bits iff infinite or NaN (sNaN
// included), exactly std::isfinite's verdict on each lane with no FP work,
// so the post loops test stored triples as integers off the FP pipe and
// OR-reduce across the group with one store on failure.
inline uint32_t NonFiniteBits(float f0, float f1, float f2)
{
    uint32_t u0, u1, u2;
    static_assert(sizeof(u0) == sizeof(f0), "");
    std::memcpy(&u0, &f0, sizeof(float));
    std::memcpy(&u1, &f1, sizeof(float));
    std::memcpy(&u2, &f2, sizeof(float));
    uint32_t constexpr kExpMask = 0x7F800000u;
    return ((u0 & kExpMask) == kExpMask) | ((u1 & kExpMask) == kExpMask) |
        ((u2 & kExpMask) == kExpMask);
}

UsdGenInputRecord const *FindCurves(UsdGenGraphSpec const *desc, UsdGenNodeKey path)
{
    return desc ? UsdGenFindCurveSetInput(*desc, path) : nullptr;
}

/// A posed surface point into current groom space. Affine matrices project
/// with w exactly 1, so TransformAffine skips the divide with identical
/// results (its xyz matches Transform term for term; Solve rejects a
/// non-finite pose the same way under either spelling). Projective layouts
/// keep Transform.
inline UsdGenVec3d XformSurfacePoint(UsdGenMatrix4d const &relative, bool affine, UsdGenVec3f const &point)
{
    return affine ? relative.TransformAffine(UsdGenVec3d(point))
                  : relative.Transform(UsdGenVec3d(point));
}

/// Where the drivers' points live relative to the description: their prim's
/// world matrix, brought into the description's space. A transform animation
/// moves rest and pose together; animate the points to deform.
UsdGenMatrix4d DriverToGroom(UsdGenGraphSpec const *desc, UsdGenNodeKey path)
{
    UsdGenInputRecord const *curves = FindCurves(desc, path);
    return curves ? UsdGenCurveSetInputView::Of(*curves).worldMatrix * desc->xformMatrix.GetInverse()
                  : UsdGenMatrix4d(1.0);
}

// The CUDA program owns RBF deform nodes (it reads no other backend).
bool OnCuda(UsdGenGraphSpec const *desc) noexcept
{
    return desc && std::strcmp(UsdGenBackendName(desc->executionBackend), "cuda") == 0;
}

}  // namespace

bool UsdGenDeformOp::Bind(UsdGenParamView const& params, UsdGenDiagnostics* diagnostics) {
    auto fail = [&](char const *message) {
        if (diagnostics) diagnostics->Error(message);
        return false;
    };
    UsdGenOpRecord const &rec = *params.op;
    if (!params.node || (!params.node->mode.IsEmpty() && params.node->mode != rec.Token(kTokCurveWrap)))
        return fail("UsdGenDeform mode must be empty (RBF) or curveWrap");
    bool const cuda = OnCuda(params.desc);
    bool const guided = !params.node->inputSets.empty() || !params.node->references.empty();
    if (params.node->mode == rec.Token(kTokCurveWrap) && (cuda || !guided))
        return fail("UsdGenDeform curveWrap requires guide curves on the CPU lane");
    if (params.node->mode.IsEmpty()) {
        for (auto const &binding:params.node->mapBindings)
            if (binding.relationship==rec.Token(kTokRegionMapRel))
                return fail("UsdGenDeform regionMap is supported only in curveWrap mode");
    }
    if (cuda && guided)
        return fail("UsdGenDeform: guide-driven RBF (usdGen:guides) runs on the CPU lane; "
                    "the CUDA lane samples the bound surface");
    if (!cuda && !guided && (!params.desc || UsdGenSurfaceInputs(*params.desc).empty()))
        return fail("UsdGenDeform: requires usdGen:guides or a bound usdGen:surface");
    return true;
}

UsdGenEpoch UsdGenDeformOp::CaptureDigest(UsdGenCaptureContext const& ctx) const {
    // The CUDA rest cache is keyed by its own surface identity; on the CPU
    // lane the capture follows the drivers' content, which moves every frame
    // they are animated.
    opUtil::Digest d;
    if (UsdGenParamView const *p = ctx.params) {
        d.Mix(uint64_t(p->GetInt(kRbfSamples, 100)));
        d.Mix(uint64_t(p->GetBool(kLockRoots, true)));
        d.Mix(uint64_t(p->node ? p->node->mode.Hash() : 0));
        d.Mix(p->GetValue(kGuideRegions, UsdGenValue()));
    }
    d.Mix(ctx.upstreamGeneration);
    if (ctx.referenceCount == 0 && ctx.desc && ctx.surface < UsdGenSurfaceInputs(*ctx.desc).size()) {
        auto const &surface = UsdGenSurfaceAt(*ctx.desc, ctx.surface);
        // Memoized: the rest points are static while the posed points move.
        // This folds one word instead of the sequential byte mix, so the
        // epoch values differ from before; epochs compare for equality only.
        d.Mix(restDigest_.Digest(surface.data, surface.restPoints));
        if (!MixChosenPosed(&d, ctx, surface))
            MixArray(&d, surface.points.cdata(), surface.points.size() * sizeof(UsdGenVec3f));
        d.Mix(uint64_t(surface.restFromCurrentPoints));
        UsdGenMatrix4d const relative = surface.worldMatrix * ctx.desc->xformMatrix.GetInverse();
        MixArray(&d, relative.GetArray(), 16 * sizeof(double));
    }
    for (uint32_t r = 0; r < ctx.referenceCount; ++r) {
        UsdGenResolvedReferenceValue const *value =
            ctx.resolvedReferences ? ctx.resolvedReferences[r] : nullptr;
        if (!value || !value->value) { d.Mix(uint64_t(~0ull)); continue; }
        UsdGenCurveSetView const b = UsdGenCurveSetView::Of(*value->value);
        d.Mix(uint64_t(b.totalCurves));
        d.Mix(uint64_t(b.totalCvs));
        MixArray(&d, b.px.cdata(), b.px.size() * sizeof(float));
        MixArray(&d, b.py.cdata(), b.py.size() * sizeof(float));
        MixArray(&d, b.pz.cdata(), b.pz.size() * sizeof(float));
        d.Mix(referenceRestDigest_.Digest(value->value->set, b.rest.cdata(),
                                          b.rest.size() * sizeof(UsdGenVec3f)));
        UsdGenMatrix4d const m = DriverToGroom(ctx.desc, value->key);
        MixArray(&d, m.GetArray(), 16 * sizeof(double));
    }
    return d.Epoch(0x446566726dull);
}

bool UsdGenDeformOp::MixChosenPosed(opUtil::Digest *d, UsdGenCaptureContext const &ctx,
                                     UsdGenSurfaceInputView const &surface) const
{
    UsdGenParamView const *p = ctx.params;
    if (p && p->node && p->node->mode == p->op->Token(kTokCurveWrap)) return false;
    // An unbound field (the last bind failed) or a degenerate selection
    // (fewer than four drivers) means Capture fails: the full hash keeps
    // the failure cadence what it was instead of skipping silently.
    if (!field_.Bound()) return false;
    int const budget = p ? p->GetInt(kRbfSamples, 100) : 100;
    // The capture's key, verbatim: a hit there reads posed[chosen] alone,
    // so the same hit here hashes exactly what the capture will read.
    auto const &rp = surface.restPoints;
    bool const hit = surfaceValid_ && surfaceBudget_ == size_t(budget) &&
        surfaceRestOwner_ && rp.size() == surfaceRestRef_.size() &&
        (rp.empty() || rp.cdata() == surfaceRestRef_.cdata());
    if (!hit || surface.points.size() != rp.size()) return false;
    if (surfaceSelection_.size() < 4) return false;
    // cdata, not operator[]: ctx.desc is shared, and a mutating subscript
    // would detach the 1.2MB posed buffer to hash 100 points out of it.
    UsdGenVec3f const *posed = surface.points.cdata();
    d->Mix(uint64_t(surfaceSelection_.size()));
    for (size_t k : surfaceSelection_)
        d->MixBytes(&posed[k], sizeof(UsdGenVec3f));
    ++chosenDigestHits_;
    return true;
}

std::unique_ptr<UsdGenCapture> UsdGenDeformOp::CreateCapture() const {
    return std::make_unique<UsdGenDeformCapture>();
}
uint32_t UsdGenDeformOp::PlanesTouched() const { return kPlanePoints; }

bool UsdGenDeformOp::Capture(UsdGenCaptureContext const& ctx, UsdGenOpInputs const &in,
                             UsdGenCapture* out, UsdGenDiagnostics* diagnostics)
{
    UsdGenCurveSetView const upstream = UsdGenCurveSetView::Of(in.Primary());
    auto fail = [&](std::string const &message) {
        if (diagnostics) diagnostics->Error("UsdGenDeform: " + message);
        return false;
    };
    UsdGenParamView const *p = ctx.params;
    if (p && !Bind(*p,diagnostics)) return false;
    if (OnCuda(ctx.desc))
        return fail("the CUDA RBF executor owns this node; it cannot run through the host scheduler");
    bool const surfaceDriven = ctx.referenceCount == 0;
    if (!surfaceDriven && (ctx.referenceCount != 1 || !ctx.resolvedReferences ||
        !ctx.resolvedReferences[0] || !ctx.resolvedReferences[0]->value))
        return fail("usdGen:guides must resolve to one curve input");
    auto &cap = *static_cast<UsdGenDeformCapture *>(out);
    cap.upstreamTopologyVersion = upstream.topologyVersion;
    cap.upstreamValueVersion = upstream.valueVersion;
    cap.upstreamCurves = upstream.totalCurves;
    cap.upstreamCvs = upstream.totalCvs;
    cap.result.clear();
    cap.samples = 0;

    bool const wrap = p && p->node && p->node->mode == p->op->Token(kTokCurveWrap);
    std::vector<curveWrap::Field> wrapFields;
    std::vector<int> rootRegions;
    bool regionMapped=false;
    UsdGenIntArray guideRegions;
    std::string regionError;
    if (wrap) {
        if (!UsdGenReadRootRegions(ctx,upstream,&rootRegions,&regionMapped,&regionError)) return fail(regionError);
        auto const value=p->GetValue(kGuideRegions,UsdGenValue(UsdGenIntArray{}));
        if (!value.IsHolding<UsdGenIntArray>()) return fail("guideRegions must be an int array");
        guideRegions=value.UncheckedGet<UsdGenIntArray>();
    }
    std::vector<uint32_t> driverSpans;
    int const budget = p ? p->GetInt(kRbfSamples, 100) : 100;
    if (!wrap && budget < 4) return fail("usdGen:rbfSamples must be at least 4");

    // --- driver samples, in the description's space --------------------------
    auto const solveField = [&]() -> bool {
        USDGEN_TRACE_SCOPE("usdGen deform: bind and solve the field");
        // Full driver vectors, for the wrap bind and for re-selection. The
        // steady-state RBF pose path never materializes them: only the
        // chosen samples are converted and transformed (Solve consumes no
        // other driver element, so the values are unchanged).
        std::vector<UsdGenVec3d> driverRest, driverNow;
        std::string driverLabel;
        // Subset sources for the re-selection path below, set by whichever
        // driver branch runs.
        UsdGenSurfaceInputView surfaceView;   // outlives the branch: surfaceSrc points here
        UsdGenSurfaceInputView const *surfaceSrc = nullptr;
        UsdGenMatrix4d surfaceRelative;
        bool surfaceAffine = true;
        UsdGenCurveSetView driverView;   // outlives the branch: guideSrc points here
        UsdGenCurveSetView const *guideSrc = nullptr;
        UsdGenMatrix4d guideToGroom;
        std::vector<UsdGenVec3d> rest, now;
        std::vector<size_t> chosen;
        bool subsetsReady = false;
        if (surfaceDriven) {
            if (!ctx.desc || ctx.surface >= UsdGenSurfaceInputs(*ctx.desc).size())
                return fail("surface-driven RBF requires one bound usdGen:surface");
            surfaceView = UsdGenSurfaceAt(*ctx.desc, ctx.surface);
            UsdGenSurfaceInputView const &surface = surfaceView;
            driverLabel = "usdGen:surface " + ctx.desc->Label(surface.key);
            if (surface.restFromCurrentPoints || surface.restPoints.empty())
                return fail(driverLabel + " requires Default-time rest points and UsdGenRestAPI");
            if (surface.points.size() != surface.restPoints.size())
                return fail(driverLabel + " animated/rest vertex counts differ");
            // Scatter/Grow produce surface rest-local points, independently of
            // hierarchy. Map posed drivers into current groom space. Shared
            // ancestor motion cancels here and is applied once by publication;
            // a separately parented groom receives that motion through the field.
            surfaceSrc = &surface;
            surfaceRelative = surface.worldMatrix * ctx.desc->xformMatrix.GetInverse();
            surfaceAffine = surfaceRelative[0][3] == 0.0 && surfaceRelative[1][3] == 0.0 &&
                surfaceRelative[2][3] == 0.0 && surfaceRelative[3][3] == 1.0;
            if (wrap) {
                driverRest.assign(surface.restPoints.begin(), surface.restPoints.end());
                driverNow.reserve(surface.points.size());
                for (UsdGenVec3f const &point : surface.points)
                    driverNow.push_back(
                        XformSurfacePoint(surfaceRelative, surfaceAffine, point));
            } else {
                // Buffer identity (not operator==, not a 1.2MB memcmp): the
                // cache holds a UsdGenCowArray reference and UsdGenCowArray is
                // copy-on-write, so an in-place edit detaches the writer to
                // a new buffer and the key misses; a hit therefore proves
                // the bytes are unchanged, and float->double conversion is
                // injective so unchanged bytes mean unchanged rest drivers.
                // Identical NaN bits still hit.
                auto const &rp = surface.restPoints;
                bool const hit = surfaceValid_ && surfaceBudget_ == size_t(budget) &&
                    surfaceRestOwner_ && rp.size() == surfaceRestRef_.size() &&
                    (rp.empty() || rp.cdata() == surfaceRestRef_.cdata());
                if (hit) {
                    chosen = surfaceSelection_;
                    rest.resize(chosen.size());
                    now.resize(chosen.size());
                    for (size_t k = 0; k < chosen.size(); ++k) {
                        rest[k] = UsdGenVec3d(rp[chosen[k]]);
                        now[k] = XformSurfacePoint(surfaceRelative, surfaceAffine,
                                                   surface.points[chosen[k]]);
                    }
                    subsetsReady = true;
                } else {
                    driverRest.assign(rp.begin(), rp.end());
                }
            }
        } else {
            UsdGenResolvedReferenceValue const &reference = *ctx.resolvedReferences[0];
            driverLabel = "usdGen:guides " + UsdGenLabel(ctx.desc, reference.key);
            driverView = UsdGenCurveSetView::Of(*reference.value);
            UsdGenCurveSetView const &drivers = driverView;
            if (wrap && drivers.totalCurves != 1 && !regionMapped)
                return fail("multiple curveWrap center curves require regionMap and guideRegions");
            if (wrap) {
                std::string error;
                if (!opUtil::CurveSpans(drivers,&driverSpans,&error)) return fail(error);
                if (regionMapped) {
                    if (guideRegions.size()!=drivers.totalCurves)
                        return fail("guideRegions must label every center curve when regionMap is bound");
                    for (size_t g=0;g<guideRegions.size();++g)
                        if (guideRegions[g]<0 || std::find(guideRegions.begin(),guideRegions.begin()+g,guideRegions[g])!=guideRegions.begin()+g)
                            return fail("guideRegions must contain unique nonnegative region IDs");
                }
                UsdGenInputRecord const *curves = FindCurves(ctx.desc, reference.key);
                if (curves && UsdGenCurveSetInputView::Of(*curves).wrap == p->op->Token(kTokPeriodic))
                    return fail("curveWrap requires an open, non-periodic center curve");
            }
            if (UsdGenInputRecord const *curves = FindCurves(ctx.desc, reference.key);
                curves && UsdGenCurveSetInputView::Of(*curves).restFromCurrentPoints)
                return fail("usdGen:guides " + UsdGenLabel(ctx.desc, reference.key) +
                            " has no rest pose to bind; apply UsdGenCurveAPI to it (its rest is "
                            "primvars:rest, else the Default-time points)");
            if (drivers.px.size() != drivers.totalCvs || drivers.py.size() != drivers.totalCvs ||
                drivers.pz.size() != drivers.totalCvs)
                return fail("the driver curves' point planes do not match their CV count");
            guideSrc = &drivers;
            guideToGroom = DriverToGroom(ctx.desc, reference.key);
            driverRest.resize(drivers.totalCvs);
            for (size_t i = 0; i < drivers.totalCvs; ++i)
                driverRest[i] = guideToGroom.Transform(UsdGenVec3d(opUtil::RestPoint(drivers, i)));
            // The RBF path transforms only the chosen posed samples after
            // selection; wrap binds whole spans, so it keeps the full fill.
            if (wrap) {
                driverNow.resize(drivers.totalCvs);
                for (size_t i = 0; i < drivers.totalCvs; ++i)
                    driverNow[i] = guideToGroom.Transform(UsdGenVec3d(opUtil::Point(drivers, i)));
            }
        }
        if (wrap) {
            std::string error;
            if (driverSpans.size()<2) return fail("curveWrap requires an open center curve");
            wrapFields.resize(driverSpans.size()-1);
            for (size_t g=0;g<wrapFields.size();++g) {
                std::vector<UsdGenVec3d> rest(driverRest.begin()+driverSpans[g],driverRest.begin()+driverSpans[g+1]);
                std::vector<UsdGenVec3d> now(driverNow.begin()+driverSpans[g],driverNow.begin()+driverSpans[g+1]);
                if (!wrapFields[g].Bind(rest,now,&error)) return fail(error);
            }
            return true;
        }
        if (!subsetsReady) {
            // Bitwise compare (not operator==): identical NaN bits still
            // hit. A surface hit takes the subset path above, so reaching
            // here surface-driven always re-selects.
            bool const restUnchanged = surfaceDriven ? false
                : selectionValid_ && selectionBudget_ == size_t(budget) &&
                    selectRest_.size() == driverRest.size() &&
                    std::memcmp(selectRest_.data(), driverRest.data(),
                                driverRest.size() * sizeof(UsdGenVec3d)) == 0;
            // The extent feeds only the re-selection epsilon, so a cache hit
            // skips the pass; the values are unchanged whenever it runs.
            double const epsilon = restUnchanged ? 0.0 : [&] {
                UsdGenRange3d extent;
                for (UsdGenVec3d const &point : driverRest) extent.UnionWith(point);
                double const size = extent.IsEmpty() ? 0.0 : extent.GetSize().GetLength();
                return std::max(1e-12, size * 1e-7);
            }();
            chosen = restUnchanged
                ? selection_
                : usdGenMath::rbf::SelectSamples(driverRest, size_t(budget), epsilon);
            if (surfaceDriven) {
                surfaceRestRef_ = surfaceSrc->restPoints;
                surfaceRestOwner_ = surfaceSrc->data;
                surfaceSelection_ = chosen;
                surfaceBudget_ = size_t(budget);
                surfaceValid_ = true;
            } else if (!restUnchanged) {
                selectRest_ = driverRest;
                selection_ = chosen;
                selectionBudget_ = size_t(budget);
                selectionValid_ = true;
            }
            rest.resize(chosen.size());
            now.resize(chosen.size());
            if (surfaceDriven) {
                for (size_t k = 0; k < chosen.size(); ++k) {
                    rest[k] = driverRest[chosen[k]];
                    now[k] = XformSurfacePoint(surfaceRelative, surfaceAffine,
                                               surfaceSrc->points[chosen[k]]);
                }
            } else {
                // On a hit selectRest_ is bitwise the driver rest, so either
                // source gathers the same values.
                UsdGenVec3d const *restSrc =
                    restUnchanged ? selectRest_.data() : driverRest.data();
                for (size_t k = 0; k < chosen.size(); ++k) {
                    rest[k] = restSrc[chosen[k]];
                    now[k] = guideToGroom.Transform(
                        UsdGenVec3d(opUtil::Point(*guideSrc, chosen[k])));
                }
            }
        }
        if (chosen.size() < 4)
            return fail(driverLabel + " has " +
                        std::to_string(chosen.size()) +
                        " distinct samples; the RBF needs at least four that span 3D");
        std::string error;
        if (!field_.Bound() || rest != boundRest_) {
            boundRest_.clear();
            if (!field_.Bind(rest, &error))
                return fail(driverLabel + ": " + error);
            boundRest_ = rest;
        }
        if (!field_.Solve(now, &error)) return fail(error);
        cap.samples = rest.size();
        return true;
    };
    if (!solveField()) return false;

    // --- strands ---------------------------------------------------------------
    USDGEN_TRACE_SCOPE("usdGen deform: displace strands");
    std::string error;
    size_t const R = upstream.totalCurves;
    size_t const totalCvs = upstream.totalCvs;
    if (R == 0) return true;
    // Uniform grooms (Grow's output: no cvOffsets) need no spans vector:
    // span c is [c*perCurve, (c+1)*perCurve) by the same arithmetic
    // CurveSpans fills the vector with, so spanAt reads the same values
    // without the 27k-entry build. The check mirrors _UsdGenCurveSpans
    // exactly (same condition, same message); ragged buffers keep the
    // vector path below.
    std::vector<uint32_t> spans;
    size_t perCurve = 0;
    bool const uniform = upstream.cvOffsets.empty();
    if (uniform) {
        if (totalCvs % R != 0)
            return fail("uniform extra-plane topology has non-integral CV count");
        perCurve = totalCvs / R;
    } else {
        if (!opUtil::CurveSpans(upstream, &spans, &error)) return fail(error);
    }
    auto spanAt = [&](size_t c) -> size_t {
        return uniform ? c * perCurve : spans[c];
    };
    if (upstream.px.size() != totalCvs || upstream.py.size() != totalCvs ||
        upstream.pz.size() != totalCvs)
        return fail("input point planes do not match the CV count");
    std::vector<size_t> driverForCurve(R,0);
    if (wrap && regionMapped) for (size_t c=0;c<R;++c) {
        auto const found=std::find(guideRegions.begin(),guideRegions.end(),rootRegions[c]);
        if (found==guideRegions.end()) return fail("regionMap region "+std::to_string(rootRegions[c])+" has no center curve");
        driverForCurve[c]=size_t(found-guideRegions.begin());
    }
    UsdGenParamField const lock = p
        ? p->GetScalarField(kLockRoots, p->GetBool(kLockRoots, true) ? 1.0 : 0.0)
        : UsdGenParamField{1.0};

    // Capture-direct output: with a uniform-1 mask the sweep would store
    // the result verbatim, so the capture stores the output planes itself
    // (planar, same floats the sweep would write) and the scheduler skips
    // the sweep. The planes must be exactly this upstream's count, mutual
    // distinct, and unaliased from the upstream planes; the strand spans
    // tile [0, totalCvs), so every output CV is written exactly once.
    // Anything unexpected keeps the result path below. Recording needs the
    // chunk partition and the armed writer's slots on top of that.
    UsdGenParamField const maskField = p
        ? p->GetScalarField(kMask, 1.0) : UsdGenParamField{1.0};
    bool const maskUniformOne =
        maskField.Uniform() && maskField.Value(0, 0) == 1.0;
    bool const direct = !wrap && maskUniformOne && R > 0 && totalCvs > 0 &&
        ctx.outPx != nullptr && ctx.outPy != nullptr && ctx.outPz != nullptr &&
        ctx.outPlaneItems == totalCvs &&
        ctx.outPx != ctx.outPy && ctx.outPy != ctx.outPz && ctx.outPx != ctx.outPz &&
        ctx.outPx != upstream.px.cdata() && ctx.outPy != upstream.py.cdata() &&
        ctx.outPz != upstream.pz.cdata();

    // The result fills even in direct mode: a later sweep without
    // re-capture (a value-only edit reuses the capture) must find the
    // deformed values, not an empty buffer.
    cap.result.resizeUninit(totalCvs * 3);
    usdGenMath::rbf::CubicField const &field = field_;
    UninitFloatBuffer &result = cap.result;
    // The non-finite refusal checks the stored floats while they are still
    // in registers, instead of re-reading the whole result serially: every
    // stored triple is checked exactly once, so the verdict and the message
    // match the retired scan.
    std::atomic<bool> nonFinite{false};
    // Strands displace in groups of kGroupStrands through one
    // DisplaceBatchPlanar call: batch boundaries are bitwise-transparent per
    // query (each query runs the same operations whatever its batch
    // position), so the group pays the per-call, per-resize, and per-task
    // overhead once instead of once per strand. 32 keeps the group's planar
    // displacement scratch (~6KB at 8 CVs per strand) L1-resident.
    size_t constexpr kGroupStrands = 32;
    size_t const groups = (R + kGroupStrands - 1) / kGroupStrands;
    bool const recordExtents = direct && ctx.chunks != nullptr && ctx.chunkCount > 0 &&
        ctx.chunkExtentSlots != nullptr &&
        ctx.chunkCount == ctx.chunkExtentCount;
    // One slot per strand group (disjoint across workers, so no locking);
    // the serial reduce below folds them into chunk slots in group order,
    // which visits every stored CV exactly once in the sweep's order.
    std::vector<UsdGenRange3f> groupExtents;
    if (recordExtents) groupExtents.resize(groups);
    float *outPx = ctx.outPx, *outPy = ctx.outPy, *outPz = ctx.outPz;
    ParallelFor(ctx.dispatcher, groups, [&](size_t g) {
        size_t const c0 = g * kGroupStrands, c1 = std::min(c0 + kGroupStrands, R);
        if (!wrap) {
            // The RBF path displaces the whole group, roots included, in
            // one batch: each strand's root element doubles as its
            // lockRoots shift exactly as in the loop below. The group's
            // CVs tile [spanAt(c0), spanAt(c1)) densely, so the planar
            // entry reads the upstream planes directly (no AoS transpose)
            // and the post loops below index the same CV minus the base.
            size_t const cvBase = spanAt(c0), cvEnd = spanAt(c1);
            size_t const total = cvEnd - cvBase;
            thread_local std::vector<double> batchD3;
            batchD3.resize(total * 3);
            double *bx = batchD3.data(), *by = batchD3.data() + total,
                   *bz = batchD3.data() + total * 2;
            field.DisplaceBatchPlanar(upstream.px.cdata() + cvBase,
                                      upstream.py.cdata() + cvBase,
                                      upstream.pz.cdata() + cvBase, bx, by, bz,
                                      total, kDisplaceParallel);
            if (direct) {
                // Direct stores plus a local range: adjacent groups share
                // cache lines in groupExtents, so the per-CV accumulation
                // stays in registers and publishes once per group. The
                // stored floats are exactly what the full-mask sweep would
                // copy out of a result buffer. The result fills too, for a
                // later sweep without re-capture (a value-only edit reuses
                // this capture's deformed values).
                // Branchless bounds: ExtendBy compiles to six conditional
                // branches per CV (GCC will not if-convert float selects
                // here); comparing the floats but selecting the bit
                // patterns keeps its exact update rule (the incumbent
                // wins ties and NaN, same FLT_MAX/-FLT_MAX start, same
                // CV order) with csel instead of branches, so the
                // published range is bitwise the per-CV ExtendBy
                // sequence, empty groups included (untouched bounds are
                // exactly the default empty range).
                float mn0 = std::numeric_limits<float>::max();
                float mn1 = std::numeric_limits<float>::max();
                float mn2 = std::numeric_limits<float>::max();
                float mx0 = -std::numeric_limits<float>::max();
                float mx1 = -std::numeric_limits<float>::max();
                float mx2 = -std::numeric_limits<float>::max();
                uint32_t mn0b, mn1b, mn2b, mx0b, mx1b, mx2b;
                std::memcpy(&mn0b, &mn0, sizeof(float));
                std::memcpy(&mn1b, &mn1, sizeof(float));
                std::memcpy(&mn2b, &mn2, sizeof(float));
                std::memcpy(&mx0b, &mx0, sizeof(float));
                std::memcpy(&mx1b, &mx1, sizeof(float));
                std::memcpy(&mx2b, &mx2, sizeof(float));
                uint32_t badBits = 0;
                for (size_t c = c0; c < c1; ++c) {
                    size_t const first = spanAt(c), last = spanAt(c + 1);
                    if (first >= last) continue;
                    size_t const ro = first - cvBase;
                    UsdGenVec3d const shift = lock.Value(c, first) != 0.0
                        ? UsdGenVec3d(bx[ro], by[ro], bz[ro])
                        : UsdGenVec3d(0.0);
                    for (size_t cv = first; cv < last; ++cv) {
                        UsdGenVec3d const x(opUtil::Point(upstream, cv));
                        size_t const k = cv - cvBase;
                        UsdGenVec3d const moved =
                            x + UsdGenVec3d(bx[k], by[k], bz[k]) - shift;
                        float const f0 = float(moved[0]);
                        float const f1 = float(moved[1]);
                        float const f2 = float(moved[2]);
                        outPx[cv] = f0;
                        outPy[cv] = f1;
                        outPz[cv] = f2;
                        result[cv * 3] = f0;
                        result[cv * 3 + 1] = f1;
                        result[cv * 3 + 2] = f2;
                        if (recordExtents) {
                            uint32_t p0, p1, p2;
                            std::memcpy(&p0, &f0, sizeof(float));
                            std::memcpy(&p1, &f1, sizeof(float));
                            std::memcpy(&p2, &f2, sizeof(float));
                            mn0b = f0 < mn0 ? p0 : mn0b;
                            mn1b = f1 < mn1 ? p1 : mn1b;
                            mn2b = f2 < mn2 ? p2 : mn2b;
                            mx0b = f0 > mx0 ? p0 : mx0b;
                            mx1b = f1 > mx1 ? p1 : mx1b;
                            mx2b = f2 > mx2 ? p2 : mx2b;
                            std::memcpy(&mn0, &mn0b, sizeof(float));
                            std::memcpy(&mn1, &mn1b, sizeof(float));
                            std::memcpy(&mn2, &mn2b, sizeof(float));
                            std::memcpy(&mx0, &mx0b, sizeof(float));
                            std::memcpy(&mx1, &mx1b, sizeof(float));
                            std::memcpy(&mx2, &mx2b, sizeof(float));
                        }
                        badBits |= NonFiniteBits(f0, f1, f2);
                    }
                }
                if (recordExtents)
                    groupExtents[g] = UsdGenRange3f(UsdGenVec3f(mn0, mn1, mn2),
                                                UsdGenVec3f(mx0, mx1, mx2));
                if (badBits) nonFinite.store(true, std::memory_order_relaxed);
                return;
            }
            uint32_t badBits = 0;
            for (size_t c = c0; c < c1; ++c) {
                size_t const first = spanAt(c), last = spanAt(c + 1);
                if (first >= last) continue;
                size_t const ro = first - cvBase;
                UsdGenVec3d const shift = lock.Value(c, first) != 0.0
                    ? UsdGenVec3d(bx[ro], by[ro], bz[ro])
                    : UsdGenVec3d(0.0);
                for (size_t cv = first; cv < last; ++cv) {
                    UsdGenVec3d const x(opUtil::Point(upstream, cv));
                    size_t const k = cv - cvBase;
                    UsdGenVec3d const moved =
                        x + UsdGenVec3d(bx[k], by[k], bz[k]) - shift;
                    float const f0 = float(moved[0]);
                    float const f1 = float(moved[1]);
                    float const f2 = float(moved[2]);
                    result[cv * 3] = f0;
                    result[cv * 3 + 1] = f1;
                    result[cv * 3 + 2] = f2;
                    badBits |= NonFiniteBits(f0, f1, f2);
                }
            }
            if (badBits) nonFinite.store(true, std::memory_order_relaxed);
            return;
        }
        uint32_t badBits = 0;
        for (size_t c = c0; c < c1; ++c) {
            size_t const first = spanAt(c), last = spanAt(c + 1);
            if (first >= last) continue;
            auto displacement = [&](UsdGenVec3d const &x) {
                return wrapFields[driverForCurve[c]].Map(x)-x;
            };
            // The root's displacement serves as both the root CV's own and,
            // with usdGen:lockRoots, the shift of the whole strand.
            UsdGenVec3d const rootDisplacement = displacement(UsdGenVec3d(opUtil::Point(upstream, first)));
            UsdGenVec3d const shift = lock.Value(c, first) != 0.0 ? rootDisplacement : UsdGenVec3d(0.0);
            for (size_t cv = first; cv < last; ++cv) {
                UsdGenVec3d const x(opUtil::Point(upstream, cv));
                UsdGenVec3d const moved =
                    x + (cv == first ? rootDisplacement : displacement(x)) - shift;
                float const f0 = float(moved[0]);
                float const f1 = float(moved[1]);
                float const f2 = float(moved[2]);
                result[cv * 3] = f0;
                result[cv * 3 + 1] = f1;
                result[cv * 3 + 2] = f2;
                badBits |= NonFiniteBits(f0, f1, f2);
            }
        }
        if (badBits) nonFinite.store(true, std::memory_order_relaxed);
    });
    if (nonFinite.load(std::memory_order_relaxed)) {
        // Failure atomicity: direct mode leaves a partial mix of old and
        // new values in the output planes; restore the input (the sweep's
        // identity fallback) so a failed run never publishes mixed
        // garbage. Failure-only cost.
        if (direct) {
            std::memcpy(outPx, upstream.px.cdata(), totalCvs * sizeof(float));
            std::memcpy(outPy, upstream.py.cdata(), totalCvs * sizeof(float));
            std::memcpy(outPz, upstream.pz.cdata(), totalCvs * sizeof(float));
        }
        return fail("the deformation produced a non-finite point");
    }
    // Publish only on success: a failed capture claims neither flag, and
    // the reduce below publishes the slots only on full success.
    if (direct) cap.wroteDirect = true;
    if (recordExtents) {
        // Fold group ranges into the scheduler's chunk slots. Groups tile
        // [0, R) and chunks must tile [0, R) contiguously, so every group
        // nests in exactly one chunk; verify both before writing any slot
        // so a future chunking change fails closed into the sweep's own
        // recording. Unioning group ranges in group order reproduces the
        // sweep's per-CV ExtendBy sequence bitwise (UsdGenRange3d keeps the
        // incumbent on ties under either spelling).
        size_t chunk = 0;
        bool nested = ctx.chunkCount > 0 && ctx.chunks[0].first[0] == 0;
        for (size_t k = 1; nested && k < ctx.chunkCount; ++k)
            nested = ctx.chunks[k].first[0] == ctx.chunks[k - 1].first[0] +
                ctx.chunks[k - 1].count[0];
        if (nested) {
            UsdGenChunkRange const &last = ctx.chunks[ctx.chunkCount - 1];
            nested = size_t(last.first[0] + last.count[0]) == R;
        }
        for (size_t g = 0; nested && g < groups; ++g) {
            size_t const gFirst = g * kGroupStrands;
            size_t const gLast = std::min(gFirst + kGroupStrands, R);
            while (chunk < ctx.chunkCount &&
                   size_t(ctx.chunks[chunk].first[0] + ctx.chunks[chunk].count[0]) <=
                       gFirst)
                ++chunk;
            nested = chunk < ctx.chunkCount &&
                ctx.chunks[chunk].first[0] <= gFirst &&
                gLast <= size_t(ctx.chunks[chunk].first[0] +
                                ctx.chunks[chunk].count[0]);
        }
        if (nested) {
            // Reduce into locals first: the slots publish only on full
            // success, so an abandoned walk can never leave partial
            // extents behind valid spans.
            std::vector<UsdGenRange3f> local(ctx.chunkCount);
            std::vector<UsdGenChunkRange> ranges(ctx.chunkCount);
            chunk = 0;
            for (size_t g = 0; g < groups; ++g) {
                size_t const gFirst = g * kGroupStrands;
                while (chunk < ctx.chunkCount &&
                       size_t(ctx.chunks[chunk].first[0] +
                              ctx.chunks[chunk].count[0]) <= gFirst)
                    ++chunk;
                if (chunk >= ctx.chunkCount) { nested = false; break; }
                UsdGenChunkRange const &cd = ctx.chunks[chunk];
                local[chunk].UnionWith(groupExtents[g]);
                UsdGenChunkRange &span = ranges[chunk];
                span.first[0] = cd.first[0];
                span.count[0] = cd.count[0];
                span.live = cd.live;
                span.first[1] = cd.first[1];
                span.stride[1] = cd.stride[1];
            }
            if (nested) {
                for (size_t k = 0; k < ctx.chunkCount; ++k) {
                    UsdGenChunkExtent &slot = ctx.chunkExtentSlots[k];
                    UsdGenVec3f const &mn = local[k].GetMin(), &mx = local[k].GetMax();
                    for (int a = 0; a < 3; ++a) { slot.min[a] = mn[a]; slot.max[a] = mx[a]; }
                    slot.range = ranges[k];
                }
                cap.extentsRecorded = true;
            }
        }
    }
    return true;
}

void UsdGenDeformOp::Evaluate(UsdGenEvalContext const& ctx, UsdGenCapture const& captureIn,
                              UsdGenChunkView const &chunk) const {
    UsdGenCurveChunk chunkCurves(chunk);
    UsdGenCurveChunk const *view = &chunkCurves;
    auto const &cap = static_cast<UsdGenDeformCapture const &>(captureIn);
    UsdGenParamView const *p = ctx.params;
    UsdGenParamField const maskField = p ? p->GetScalarField(kMask, 1.0) : UsdGenParamField{1.0};
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const firstCv = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;
    bool const ready = cap.result.size() == size_t(cap.upstreamCvs) * 3;
    // Fused tile extents: extend over exactly the floats stored below, in
    // store order, so the interleave can union per-chunk records instead
    // of re-reading the points. The stored values are NaN-free whenever
    // this runs (capture success pins the upstream and the result finite,
    // and the lerp below closes over finite/Inf without producing NaN),
    // which is what makes the chunked union bitwise exact. Null in every
    // graph the gate does not select: one predictable branch per CV.
    UsdGenChunkExtent *extSlot = view->extentSlot;
    UsdGenRange3f chunkExt;
    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        size_t const curve = curveBase + c;
        for (size_t i = 0; i < n; ++i) {
            size_t const o = g + i;
            size_t const cv = firstCv + o;
            float const m = ready
                ? std::clamp(float(maskField.Value(curve, cv)), 0.0f, 1.0f) : 0.0f;
            if (m == 0.0f || cv * 3 + 2 >= cap.result.size()) {
                float const e0 = view->inPx[o];
                float const e1 = view->inPy[o];
                float const e2 = view->inPz[o];
                view->px[o] = e0;
                view->py[o] = e1;
                view->pz[o] = e2;
                if (extSlot) chunkExt.ExtendBy(UsdGenVec3f(e0, e1, e2));
                continue;
            }
            float const *r = &cap.result[cv * 3];
            // The full-mask store skips the input planes: the old loop
            // loaded in[3] before the mask test and discarded it here, so
            // the stores are unchanged while the loop reads a third less.
            if (m == 1.0f) {
                view->px[o] = r[0]; view->py[o] = r[1]; view->pz[o] = r[2];
                if (extSlot) chunkExt.ExtendBy(UsdGenVec3f(r[0], r[1], r[2]));
                continue;
            }
            float const in0 = view->inPx[o];
            float const in1 = view->inPy[o];
            float const in2 = view->inPz[o];
            float const e0 = in0 + (r[0] - in0) * m;
            float const e1 = in1 + (r[1] - in1) * m;
            float const e2 = in2 + (r[2] - in2) * m;
            view->px[o] = e0;
            view->py[o] = e1;
            view->pz[o] = e2;
            if (extSlot) chunkExt.ExtendBy(UsdGenVec3f(e0, e1, e2));
        }
    }
    if (extSlot && view->desc) {
        for (int k = 0; k < 3; ++k) {
            extSlot->min[k] = chunkExt.GetMin()[k];
            extSlot->max[k] = chunkExt.GetMax()[k];
        }
        extSlot->range.first[0] = view->desc->firstCurve;
        extSlot->range.count[0] = view->desc->curveCount;
        extSlot->range.live = view->desc->liveCount;
        extSlot->range.first[1] = view->desc->firstCv;
        extSlot->range.stride[1] = view->desc->cvCount;
    }
}

namespace {

// Only a static styling stack followed by one surface-driven RBF can retain
// its upstream geometry while the surface poses (the tail-deform binding).
bool MakeAnimationBinding(UsdGenAnimationBindingQuery const &q, UsdGenOp const &op,
                          UsdGenDeformBinding *out)
{
    UsdGenOpNode const *n = q.terminalDesc;
    UsdGenParamView const *p = q.terminalParams;
    if (!n || !p || !p->op || !out || !n->enabled) return false;
    if (!n->mode.IsEmpty() && n->mode != p->op->Token(kTokRbf)) return false;
    if (q.terminalInputCount != 1 || !n->inputSets.empty() || !n->references.empty() ||
        !n->expressionBindings.empty() || !n->maps.empty() || !n->mapBindings.empty() ||
        !q.surface || p->GetBool(kLockRoots, true) || p->GetDouble(kMask, 1.0) != 1.0 ||
        q.terminalHasOutputPlanes)
        return false;
    // The hook is registered on Deform only, so the terminal op is one.
    auto const &deform = static_cast<UsdGenDeformOp const &>(op);
    if (deform.AnimationSamples().empty() || !q.graphStatic || q.nodesWithTerminalOp != 1)
        return false;
    out->sampleIndices = deform.AnimationSamples();
    out->field = deform.AnimationField();
    return true;
}

}  // namespace

UsdGenOpId UsdGenRegisterDeformOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenDeform";
    d.displayName = "Deform";
    d.create = &Create;
    d.countChanges = 0;
    d.traits = kUsdGenOpRecordsChunkExtents | kUsdGenOpPosesSpace;   // scheduler.cpp:1031, 1076
    d.params = UsdGenSpan(kParams);
    d.referenceInputs = UsdGenSpan(kRefs);
    d.mapSlots = UsdGenSpan(kMaps);
    d.tokens = UsdGenSpan(kTokens);
    d.hooks.makeAnimationBinding = &MakeAnimationBinding;
    return r.RegisterOp(d);
}

} // namespace usdGen
