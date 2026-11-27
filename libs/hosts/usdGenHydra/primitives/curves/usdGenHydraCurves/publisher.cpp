// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Curve publisher: a curve terminal's tiles (look bake, preview colours,
// carry-over and points-only refresh), fur occlusion and the scalp cap, the
// RBF playback binding and the prim-set signature; the diff of two
// publications and their cache charge. Moved out of sessionCooker (6.3).
#include "usdGenHydraCurves/publisher.h"
#include "usdGenHydraCurves/curveColumns.h"
#include "usdGenHydraCurves/furOcclusion.h"
#include "usdGenHydraCurves/previewColor.h"
#include "usdGenHydraCurves/publication.h"
#include "usdGenHydraCurves/register.h"
#include "usdGenHydraCurves/tokens.h"
#include "usdGenUsd/lower.h"
#include "usdGenUsd/convert/math.h"
#include "usdGen/api/deformBinding.h"
#include "usdGen/api/publishDesc.h"
#include "usdGen/compile/graph.h"
#include "usdGen/exec/debugCodes.h"
#include "usdGen/exec/generationStore.h"
#include "usdGen/exec/scheduler.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/token.h"
#include "usdGen/registries/typeRegistry.h"

#include "usdGenMath/usdGenMath/hash.h"

#include "pxr/base/tf/envSetting.h"
#include "pxr/base/trace/trace.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

// Hair shadowing the scalp is on by default and costs one extra sweep of the
// density volume plus a small translucent mesh. Set USDGEN_SCALP_SHADOW=0 to
// publish no cap at all: nothing else in the groom changes.
TF_DEFINE_ENV_SETTING(USDGEN_SCALP_SHADOW, true,
                      "Publish the usdGen scalp-shadow cap (hair shadowing the "
                      "groom's emitting surface).");

// Per-CV local hair density (`furDensity`), read by a hair shading model that
// darkens ambient light by local hair density. On by default only in a build
// that selects that model (USDGEN_FUR_LOCAL_DENSITY); USDGEN_FUR_DENSITY=0/1
// overrides either way. Off, tiles carry no extra primvar.
#if defined(USDGEN_FUR_LOCAL_DENSITY)
static constexpr bool _furDensityDefault = true;
#else
static constexpr bool _furDensityDefault = false;
#endif
TF_DEFINE_ENV_SETTING(USDGEN_STORM_GPU_DEFORM, false,
    "Use resident Storm GPU points for compatible terminal RBF pose previews.");

TF_DEFINE_ENV_SETTING(USDGEN_FUR_DENSITY, _furDensityDefault,
                      "Publish the per-CV hair-only local density primvar "
                      "furDensity on usdGen fur tiles.");

namespace usdGen {

bool UsdGenPrimSetSignature::operator==(
    UsdGenPrimSetSignature const &rhs) const
{
    return tileCount == rhs.tileCount &&
           instancerCount == rhs.instancerCount &&
           prims == rhs.prims &&
           primvarNames == rhs.primvarNames &&
           primTypes == rhs.primTypes;
}

namespace {

// Returned for a null or foreign publication; never referenced by a UsdGenRef.
// Built at registration (UsdGenRegisterCurvePublisher), never at load: their
// constructors read the CurveSet type id, and that read builds the registries.
// A registry built while this library's initializers run loads this library
// before its later translation units are initialized (glibc returns a library
// that is mid-initialization), so ops whose tables are dynamically initialized
// (Scatter, ExprOp) would register with empty tables. CAS-published, never freed.
std::atomic<UsdGenCurvePublication const *> gEmptyPublication{nullptr};
std::atomic<UsdGenCurveDirtyReport const *> gEmptyReport{nullptr};

template <class T>
T const &EmptyPublication(std::atomic<T const *> &slot)
{
    if (T const *p = slot.load(std::memory_order_acquire)) return *p;
    T const *fresh = new T;
    T const *winner = nullptr;
    if (slot.compare_exchange_strong(winner, fresh, std::memory_order_acq_rel,
                                     std::memory_order_acquire))
        return *fresh;
    delete fresh;
    return *winner;
}

// Schema-default look and presentation, made at load rather than behind a
// static's guard. A spec without host data publishes the defaults.
UsdGenLookDesc const _lookFallbackDesc;
usdGenUsd::UsdGenPresentation const _presentationFallback;

// The spec's presentation (usdGenUsd host data).
usdGenUsd::UsdGenPresentation const &PresentationOf(UsdGenGraphSpec const &spec) noexcept
{
    usdGenUsd::UsdGenPresentation const *p = usdGenUsd::Presentation(spec);
    return p ? *p : _presentationFallback;
}

// Look constants published on tiles (addLookConstant below).
TfToken const kHairTipColor = UsdGenInternToken("hairTipColor");
TfToken const kHairColorRamp = UsdGenInternToken("hairColorRamp");
TfToken const kHairRandomHue = UsdGenInternToken("hairRandomHue");
TfToken const kHairRandomValue = UsdGenInternToken("hairRandomValue");

/// The preview colours of one cook, indexed like the terminal buffer.
struct UsdGenPreviewColors
{
    bool active = false;
    bool perCv = false;            // one colour per CV, else one per curve
    std::vector<GfVec3f> colors;   // linear
    uint64_t digest = 0;           // 0 exactly when inactive
};

// The CurveSet's publish hooks (curvePublishDesc.cpp); null before registration.
UsdGenPublishDesc const *CurvePublishDesc()
{
    UsdGenTypeId const type = UsdGenHydraCurveSetType();
    return type == UsdGenTypeId::Invalid ? nullptr : UsdGenTypeRegistry::Get().Def(type).publish;
}

// What the publisher keeps across one cooker's cooks.
struct _State final : UsdGenPublisherState
{
    std::unique_ptr<UsdGenPublishHookState> hooks;   // the publish hooks' state
    UsdGenCurveSlots slots;         // the terminal's static channel slots
    UsdGenPreviewColors colors;     // the last build's usdGen:preview colours
    // Identity of the fur density volume's non-tile inputs (occluder meshes,
    // grid parameters): the tile COW check alone cannot see them change.
    uint64_t furVolumeKey = 0;
    // Occluder work reused across bakes (furOcclusion.h); made on first use.
    std::shared_ptr<UsdGenFurOcclusionCache> furCache;
};

// What one tile build reads; shared read-only by concurrent builds.
struct TileInputs
{
    UsdGenGraphSpec const &spec;
    UsdGenCurveColumns const &term;
    UsdGenSpan<const UsdGenChunkRange> chunks;
    UsdGenPreviewColors const &colors;
};

// Colours of the previewed values, indexed like the terminal buffer.
void PreviewColors(UsdGenGraphSpec const &spec, UsdGenCurveColumns const &term,
                   UsdGenPreviewValues const &values, UsdGenPreviewColors *out)
{
    out->active = values.active;
    out->perCv = false;
    out->colors.clear();
    out->digest = 0;
    if (!values.active) return;
    usdGenUsd::UsdGenPresentation const &preview = PresentationOf(spec);
    out->perCv = values.level == 2;
    size_t const elements = out->perCv ? term.totalCvs : term.totalCurves;
    out->colors.assign(elements, UsdGenPreviewMissingColor());
    unsigned const k = values.components;
    if (k) {
        for (size_t i = 0; i < elements && (i + 1) * k <= values.values.size(); ++i) {
            if (!values.present.empty() && (i >= values.present.size() || !values.present[i]))
                continue;
            out->colors[i] =
                UsdGenPreviewColor(preview.previewColorMap, preview.previewRange,
                                   values.values.data() + i * k, k);
        }
    }
    auto mix = [](uint64_t h, uint64_t value) {
        return h ^ (value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    };
    uint64_t digest = mix(1469598103934665603ULL, out->perCv ? 2u : 1u);
    digest = mix(digest, preview.previewShading.Hash());
    for (GfVec3f const &c : out->colors) {
        uint32_t bits[3];
        std::memcpy(bits, c.data(), sizeof(bits));
        digest = mix(mix(mix(digest, bits[0]), bits[1]), bits[2]);
    }
    out->digest = digest | 1u;
}

void Phase(UsdGenPublishContext const &ctx, char const *name,
           size_t rebuilt = SIZE_MAX, size_t total = 0)
{
    if (ctx.phase) ctx.phase(ctx.phaseUser, name, rebuilt, total);
}

// One tile as a one-group publication (progress).
void EmitOne(UsdGenPublishContext const &ctx, UsdGenTilePublication const &tile)
{
    UsdGenRef<UsdGenCurvePublication> one = UsdGenNewCurvePublication();
    one->tiles.push_back(tile);
    one->groups = 1;
    ctx.emit(ctx.emitUser, std::move(one));
}

bool _AddBytes(size_t *total, size_t count, size_t element)
{
    if (count != 0 && element > std::numeric_limits<size_t>::max() / count)
        return false;
    size_t const bytes = count * element;
    if (bytes > std::numeric_limits<size_t>::max() - *total) return false;
    *total += bytes;
    return true;
}

struct _PresentationScalars
{
    GfMatrix4d xformMatrix{1.0};
    TfToken purpose;
    TfToken visibility;
    UsdGenPrimRef material;
    TfToken materialPurpose{hydraCurvesTokens::allPurpose()};
    int refineLevel = 2;
    UsdGenPrimRef origin;
    UsdGenPrimRef dependency;
};

_PresentationScalars _BuildPresentationScalars(
    UsdGenGraphSpec const &desc, UsdGenNodeKey dependencySurface)
{
    usdGenUsd::UsdGenPresentation const &p = PresentationOf(desc);
    _PresentationScalars result;
    result.xformMatrix = usdGenUsd::ToHost(desc.xformMatrix);
    result.purpose = p.purpose;
    result.visibility = p.visibility;
    result.material = UsdGenPrimRef{p.material};
    result.origin = UsdGenPrimRef{desc.description};
    result.dependency = UsdGenPrimRef{dependencySurface};
    return result;
}

// CUDA admission permits at most one CurveSource root-binding surface and
// ExecuteCudaGraph rejects a named binding that is absent from desc.surfaces.

// Sample a named curve buffer plane into the publication's uniform array.
// "uniform" planes index by curve, "vertex" planes by CV.
template <class Plane>
void _GatherPlane(const Plane &plane, uint32_t curveIdx, uint32_t cvIdx,
                  bool wantUniform, VtFloatArray *outF)
{
    if (plane.type == UsdGenTokens().int_ || plane.arity == 0) return;
    const size_t idx =
        (wantUniform && plane.interpolation != UsdGenTokens().vertex)
            ? curveIdx : cvIdx;
    if ((idx + 1) * plane.arity > plane.f.size()) return;
    for (uint8_t k = 0; k < plane.arity; ++k)
        outF->push_back(plane.f[idx * plane.arity + k]);
}

template <class Plane>
bool _GatherPlaneElement(Plane const& plane, size_t index,
                         UsdGenPlane* out)
{
    if (!out || plane.arity == 0) return false;
    size_t const begin = index * plane.arity;
    if (plane.type == UsdGenTokens().int_) {
        if (begin + plane.arity > plane.i.size()) return false;
        for (uint8_t component = 0; component != plane.arity; ++component)
            out->i.push_back(plane.i[begin + component]);
        return true;
    }
    if (plane.type != UsdGenTokens().float_ || begin + plane.arity > plane.f.size())
        return false;
    for (uint8_t component = 0; component != plane.arity; ++component)
        out->f.push_back(plane.f[begin + component]);
    return true;
}

// Color planes may be float3 (arity 3) or float (arity 1, grey).
template <class Plane>
void _GatherColor(const Plane &plane, uint32_t curveIdx, uint32_t cvIdx,
                  VtVec3fArray *out)
{
    VtFloatArray tmp;
    _GatherPlane(plane, curveIdx, cvIdx, /*wantUniform=*/true, &tmp);
    if (plane.arity >= 3) {
        for (size_t q = 0; q + 3 <= tmp.size(); q += 3)
            out->push_back(GfVec3f(tmp[q], tmp[q + 1], tmp[q + 2]));
    } else if (plane.arity == 1) {
        for (float v : tmp) out->push_back(GfVec3f(v, v, v));
    }
}

// Group dirty bits the scheduler's interleave sets (UsdGenChunkGroup::dirtyRoles).
constexpr uint32_t kGroupPositionDirty = 1u << uint32_t(UsdGenChannelRole::Position);
constexpr uint32_t kGroupWidthDirty = 1u << uint32_t(UsdGenChannelRole::Width);

// A named per-curve (cv false) or per-CV plane of the terminal.
std::optional<UsdGenCurvePlane> _FindPlane(UsdGenCurveColumns const &term, bool cv,
                                          TfToken const &name)
{
    uint32_t const n = cv ? term.ExtraCvCount() : term.ExtraCurveCount();
    for (uint32_t k = 0; k < n; ++k) {
        UsdGenCurvePlane const p = cv ? term.ExtraCv(k) : term.ExtraCurve(k);
        if (p.name == name) return p;
    }
    return std::nullopt;
}

// Bulk tile path (bit-identical): when the tile's chunks are all uniform,
// the terminal planes cover every emitted element, no extra plane needs a
// per-element gather, and the color mode is a per-tile constant (Root) or
// empty (NoColor), the serial push_back loop becomes per-chunk bulk copies
// (points interleave, widths/hairT/st memcpy, counts fill, displayColor
// assign) plus a branch-free per-curve hairId loop, with the extent
// reduction fused into the interleave. Coverage is verified up front (any
// gap returns false with `pub` untouched and the serial loop runs); sizes
// are proven against the tile view, so every no-init element is
// overwritten and no garbage is observed. wantColor selects Root (NoColor
// passes false); every other color mode stays on the serial loop.
bool _BuildTilePublicationBulk(
    UsdGenChunkGroup const &tv, UsdGenCurveColumns const &term,
    UsdGenSpan<const UsdGenChunkRange> chunks, bool wantColor, GfVec3f const &rootColor,
    UsdGenTilePublication *pub, uint32_t *depSurface, bool *hasDep)
{
    // Pre-scan: uniform chunks, plane coverage, view-size agreement. The
    // serial loop breaks the CV loop on px overrun and falls back per
    // element for short width/hairT/curveId/rootUV; any of that needs the
    // serial loop, so coverage must be total here.
    uint64_t scanCurves = 0, scanCvs = 0;
    bool const needWidth = !term.width.empty();
    bool const needHairT = !term.hairT.empty();
    bool const needSt = !term.rootUV.empty();
    for (uint32_t i = 0; i < tv.chunkCount; ++i) {
        UsdGenChunkRange const &cd = chunks[tv.firstChunk + i];
        if (cd.stride[1] == 0)
            return false;
        uint64_t const n = uint64_t(cd.live) * cd.stride[1];
        uint64_t const cvEnd = uint64_t(cd.first[1]) + n;
        if (cvEnd > term.px.size() || cvEnd > term.py.size() ||
            cvEnd > term.pz.size())
            return false;
        if (needWidth && cvEnd > term.width.size())
            return false;
        if (needHairT && cvEnd > term.hairT.size())
            return false;
        uint64_t const gEnd = uint64_t(cd.first[0]) + cd.live;
        if (gEnd > term.curveId.size())
            return false;
        if (needSt && gEnd > term.rootUV.size())
            return false;
        scanCurves += cd.live;
        scanCvs += n;
    }
    if (scanCurves != tv.live[0] || scanCvs != tv.live[1])
        return false;
    size_t const nCurves = tv.live[0], nCvs = tv.live[1];
    auto noInit = [](auto *b, auto *e) {
        (void)b;
        (void)e;
    };
    pub->curveVertexCounts.resize(nCurves, noInit);
    pub->points.resize(nCvs, noInit);
    pub->widths.resize(nCvs, noInit);
    pub->hairT.resize(nCvs, noInit);
    pub->hairId.resize(nCurves, noInit);
    if (needSt)
        pub->st.resize(nCurves, noInit);
    if (wantColor)
        pub->displayColor.assign(nCurves, rootColor);
    int *counts = pub->curveVertexCounts.data();
    GfVec3f *points = nCvs ? pub->points.data() : nullptr;
    float *widths = nCvs ? pub->widths.data() : nullptr;
    float *hairT = nCvs ? pub->hairT.data() : nullptr;
    float *hairIds = nCurves ? pub->hairId.data() : nullptr;
    GfVec2f *st = (needSt && nCurves) ? pub->st.data() : nullptr;
    float const *px = term.px.cdata(), *py = term.py.cdata(),
                *pz = term.pz.cdata();
    // Fused extent: same init (SetEmpty) and per-point compares as the
    // serial GfRange3f pass over the identical emission order, including
    // the empty-range tail below (NaN points stay ignored, as there).
    float const inf = std::numeric_limits<float>::max();
    float minX = inf, minY = inf, minZ = inf;
    float maxX = -inf, maxY = -inf, maxZ = -inf;
    size_t curOff = 0, cvOff = 0;
    for (uint32_t i = 0; i < tv.chunkCount; ++i) {
        UsdGenChunkRange const &cd = chunks[tv.firstChunk + i];
        uint32_t const L = cd.live;
        if (L == 0)
            continue;
        size_t const n = size_t(L) * cd.stride[1];
        uint32_t const G = cd.first[0], P = cd.first[1];
        std::fill(counts + curOff, counts + curOff + L, int(cd.stride[1]));
        for (uint32_t c = 0; c < L; ++c)
            hairIds[curOff + c] = UsdGenHairId(term.curveId[G + c]);
        if (needSt)
            std::memcpy(st + curOff, term.rootUV.cdata() + G,
                        L * sizeof(GfVec2f));
        if (needWidth)
            std::memcpy(widths + cvOff, term.width.cdata() + P,
                        n * sizeof(float));
        else
            std::fill(widths + cvOff, widths + cvOff + n, 0.01f);
        if (needHairT)
            std::memcpy(hairT + cvOff, term.hairT.cdata() + P,
                        n * sizeof(float));
        else
            std::fill(hairT + cvOff, hairT + cvOff + n, 0.0f);
        for (size_t k = 0; k < n; ++k) {
            float const x = px[P + k], y = py[P + k], z = pz[P + k];
            points[cvOff + k] = GfVec3f(x, y, z);
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            if (z < minZ) minZ = z;
            if (z > maxZ) maxZ = z;
        }
        curOff += L;
        cvOff += n;
    }
    GfRange3f e(GfVec3f(minX, minY, minZ), GfVec3f(maxX, maxY, maxZ));
    if (e.IsEmpty()) e = GfRange3f(GfVec3f(0.f), GfVec3f(0.f));
    pub->extentMin = GfVec3d(e.GetMin()[0], e.GetMin()[1], e.GetMin()[2]);
    pub->extentMax = GfVec3d(e.GetMax()[0], e.GetMax()[1], e.GetMax()[2]);
    if (tv.chunkCount != 0) {
        *depSurface = chunks[tv.firstChunk].domain;
        *hasDep = true;
    }
    return true;
}

UsdGenTilePublication BuildTile(TileInputs const &in, UsdGenChunkGroup const &tv)
{
    UsdGenGraphSpec const &spec = in.spec;
    usdGenUsd::UsdGenPresentation const &presentation = PresentationOf(spec);
    UsdGenLookDesc const &look = presentation.look;
    UsdGenCurveColumns const &term = in.term;
    UsdGenSpan<const UsdGenChunkRange> const chunks = in.chunks;
    UsdGenPreviewColors const &previewColors = in.colors;
    UsdGenTilePublication pub;
    pub.tile = tv.id;
    pub.prim = UsdGenPrimRef{spec.description, UsdGenPublishRole::Tile, uint32_t(tv.id)};
    if (!presentation.basis.IsEmpty())
        pub.basis = presentation.basis.GetString();
    pub.refineLevel = 2;

    pub.curveVertexCounts.reserve(tv.live[0]);
    pub.points.reserve(tv.live[1]);
    pub.widths.reserve(tv.live[1]);
    pub.hairT.reserve(tv.live[1]);
    pub.hairId.reserve(tv.live[0]);

    std::optional<UsdGenCurvePlane> const displayColor =
        _FindPlane(term, false, hydraCurvesTokens::displayColor());
    struct ExtraPlanePublication {
        std::optional<UsdGenCurvePlane> source;
        UsdGenPlane output;
    };
    std::vector<ExtraPlanePublication> extraPlanes;
    // The forwarded source colour is consumed into the tile's displayColor
    // below, never published: it is an input to the colour decision, not a
    // primvar of its own.
    std::optional<UsdGenCurvePlane> sourceColor =
        _FindPlane(term, false, UsdGenHydraSourceColorPlane());
    if (!sourceColor) sourceColor = _FindPlane(term, true, UsdGenHydraSourceColorPlane());
    auto addExtraPlane = [&](UsdGenCurvePlane const& plane, TfToken expectedInterpolation) {
        if (plane.name == hydraCurvesTokens::displayColor() ||
            plane.name == UsdGenHydraSourceColorPlane()) return;
        size_t const elements = expectedInterpolation == UsdGenTokens().constant ? 1 :
            expectedInterpolation == UsdGenTokens().vertex ? term.totalCvs : term.totalCurves;
        size_t const values = elements * plane.arity;
        bool const validPayload = plane.type == UsdGenTokens().int_
            ? plane.i.size() == values && plane.f.empty()
            : plane.type == UsdGenTokens().float_ && plane.f.size() == values && plane.i.empty();
        if (plane.name.IsEmpty() || plane.arity == 0 ||
            plane.arity > kUsdGenMaxExtraPlaneSlots ||
            plane.interpolation != expectedInterpolation || !validPayload)
            throw std::runtime_error("invalid extra-plane publication layout for '" +
                                     plane.name.GetString() + "'");
        if (std::any_of(extraPlanes.begin(), extraPlanes.end(), [&](auto const& entry) {
                return entry.output.name == plane.name;
            }))
            throw std::runtime_error("duplicate extra-plane publication name '" +
                                     plane.name.GetString() + "'");
        ExtraPlanePublication entry;
        entry.source = plane;
        entry.output.name = plane.name;
        entry.output.interpolation = plane.interpolation;
        entry.output.type = plane.type;
        entry.output.arity = plane.arity;
        if (plane.interpolation == UsdGenTokens().constant)
            _GatherPlaneElement(plane, 0, &entry.output);
        extraPlanes.push_back(std::move(entry));
    };
    for (uint32_t k = 0; k < term.ExtraCurveCount(); ++k) {
        UsdGenCurvePlane const p = term.ExtraCurve(k);
        if (p.interpolation == UsdGenTokens().constant)
            addExtraPlane(p, UsdGenTokens().constant);
        else if (p.interpolation == UsdGenTokens().uniform)
            addExtraPlane(p, UsdGenTokens().uniform);
        else if (p.name != hydraCurvesTokens::displayColor())
            throw std::runtime_error("invalid per-curve extra-plane interpolation for '" +
                                     p.name.GetString() + "'");
    }
    for (uint32_t k = 0; k < term.ExtraCvCount(); ++k)
        addExtraPlane(term.ExtraCv(k), UsdGenTokens().vertex);

    // Precedence (02 §2.14): an AUTHORED look on the description wins;
    // otherwise the source curves' own displayColor, which is what a
    // character asset already carries; otherwise the look's schema
    // defaults. "Authored" has to mean an opinion that differs from the
    // fallback -- a description that merely applies UsdGenLookAPI has a full
    // set of schema-default look values, and treating those as authored would
    // silently discard every asset's own colour.
    UsdGenLookDesc const &lookFallback = _lookFallbackDesc;
    bool const authoredLook =
        look.rootColor != lookFallback.rootColor ||
        look.tipColor != lookFallback.tipColor ||
        !look.rampColors.empty();

    // The rest of the look as CONSTANT primvars, so the shading of a groom
    // does not depend on the tile keeping the synthetic material: a user who
    // binds their own UsdGenHairStrands material gets the Sdr defaults for
    // everything the tile does not carry, which used to mean blond tips on a
    // black-haired groom. rootColor already travels as the baked per-curve
    // displayColor, so it is deliberately NOT repeated here -- a second
    // channel for it would only be a way for the two to disagree.
    //
    // Each one is published only where the look RESOLVES it to something
    // other than its schema fallback. An unauthored look publishes none of
    // them, the shader's `#ifdef HD_HAS_*` is false, and the material input
    // applies exactly as before: the default-material path is unchanged by
    // construction, and a forwarded asset colour (one flat colour per curve)
    // never acquires an invented tip.
    auto addLookConstant = [&](TfToken const& name, std::initializer_list<float> v) {
        // An authored plane of the same name wins: the user asked for it by
        // hand, and two planes with one name is not a publishable tile.
        if (std::any_of(extraPlanes.begin(), extraPlanes.end(),
                        [&](auto const& e) { return e.output.name == name; }))
            return;
        UsdGenPlane plane;
        plane.name = name;
        plane.interpolation = UsdGenTokens().constant;
        plane.type = UsdGenTokens().float_;
        plane.arity = static_cast<decltype(plane.arity)>(v.size());
        plane.f.assign(v.begin(), v.end());
        ExtraPlanePublication entry;
        entry.source.reset();     // synthesized; the uniform gather never runs
        entry.output = std::move(plane);
        extraPlanes.push_back(std::move(entry));
    };
    if (authoredLook) {
        // Ramp-aware, matching UsdGenTilePublisher::_LookColors: the shader
        // has one lerp, so a multi-stop ramp presents as its last stop.
        GfVec3f const tip = look.rampColors.size() >= 2
            ? look.rampColors[look.rampColors.size() - 1]
            : look.tipColor;
        addLookConstant(kHairTipColor, {tip[0], tip[1], tip[2]});
    }
    if (look.rampExponent != lookFallback.rampExponent)
        addLookConstant(kHairColorRamp,
                        {std::max(look.rampExponent, 0.001f)});
    if (look.hueJitter != lookFallback.hueJitter)
        addLookConstant(kHairRandomHue, {std::max(look.hueJitter, 0.0f)});
    if (look.valueJitter != lookFallback.valueJitter)
        addLookConstant(kHairRandomValue,
                        {std::max(look.valueJitter, 0.0f)});

    GfRange3f bounds;
    uint32_t depSurface = 0;
    bool hasDep = false;
    bool firstChunk = true;

    // Hoisted tile-uniform color dispatch (bit-identical): preview, the
    // displayColor/sourceColor planes, bakeTarget and authoredLook never
    // change within a tile, but the curve loop re-evaluated
    // `bakeTarget != TfToken("none")` -- a token-registry lookup -- for
    // every curve (1M lookups at bench scale, and a mutex convoy under
    // tile threading). The mode below selects the identical branch.
    enum ColorMode { NoColor, Preview, Plane, SourcePlane, Root };
    ColorMode colorMode = NoColor;
    if (previewColors.active)
        colorMode = Preview;
    else if (displayColor)
        colorMode = Plane;
    else if (look.bakeTarget != hydraCurvesTokens::none())
        colorMode = (!authoredLook && sourceColor) ? SourcePlane : Root;
    // Extra planes partitioned once per tile (bit-identical): the CV loop
    // gathered the vertex-interpolated planes and the curve loop the
    // uniform ones, each re-comparing every plane's interpolation token
    // per element. Same planes, same relative order, no per-element
    // token work; constant planes stay out of both loops, as before.
    std::vector<ExtraPlanePublication *> vertexPlanes, uniformPlanes;
    for (auto &extra : extraPlanes) {
        if (extra.output.interpolation == UsdGenTokens().vertex)
            vertexPlanes.push_back(&extra);
        else if (extra.output.interpolation == UsdGenTokens().uniform)
            uniformPlanes.push_back(&extra);
    }
    // Pre-size the remaining per-curve/per-CV arrays (bit-identical: same
    // pushed values, no geometric-growth copies): st takes one root UV per
    // curve when the terminal carries rootUV; displayColor takes one color
    // per curve except under per-CV preview (one per CV), and nothing at
    // all when the mode is NoColor; extra planes take their live element
    // count times arity on the matching payload. Over-reserve is
    // capacity-only: short gathers (failed bounds checks, multi-push
    // color arities resuming past their share) just push less.
    if (!term.rootUV.empty())
        pub.st.reserve(tv.live[0]);
    if (colorMode == Preview && previewColors.perCv)
        pub.displayColor.reserve(tv.live[1]);
    else if (colorMode != NoColor)
        pub.displayColor.reserve(tv.live[0]);
    for (ExtraPlanePublication *extra : vertexPlanes) {
        size_t const n = size_t(tv.live[1]) * extra->output.arity;
        if (extra->output.type == UsdGenTokens().int_)
            extra->output.i.reserve(n);
        else
            extra->output.f.reserve(n);
    }
    for (ExtraPlanePublication *extra : uniformPlanes) {
        size_t const n = size_t(tv.live[0]) * extra->output.arity;
        if (extra->output.type == UsdGenTokens().int_)
            extra->output.i.reserve(n);
        else
            extra->output.f.reserve(n);
    }

    // Bulk uniform path (bit-identical, see _BuildTilePublicationBulk):
    // per-chunk bulk copies instead of the push_back loop below, with the
    // extent fused in. Any uncovered element falls back to the serial loop
    // (ragged chunks, short planes, per-element color or extra gathers).
    bool useBulk = vertexPlanes.empty() && uniformPlanes.empty() &&
        (colorMode == Root || colorMode == NoColor);
    if (useBulk)
        useBulk = _BuildTilePublicationBulk(tv, term, chunks,
                                            colorMode == Root,
                                            look.rootColor, &pub,
                                            &depSurface, &hasDep);
    if (!useBulk) {
        for (uint32_t i = 0; i < tv.chunkCount; ++i) {
            UsdGenChunkRange const &cd = chunks[tv.firstChunk + i];
            // id 0 is a legitimate surface (compiler.cpp assigns dense indices
            // from 0): the first chunk of the tile always establishes the
            // dependency, never a `!= 0` sentinel.
            if (firstChunk) { depSurface = cd.domain; hasDep = true; firstChunk = false; }
            for (uint32_t c = 0; c < cd.live; ++c) {
                const uint32_t g = cd.first[0] + c;
                uint32_t p0 = 0, len = 0;
                if (cd.stride[1] != 0) {
                    len = cd.stride[1];
                    p0 = cd.first[1] + c * cd.stride[1];
                } else if (g + 1 < term.cvOffsets.size()) {   // ragged chunk
                    p0 = static_cast<uint32_t>(term.cvOffsets[g]);
                    len = static_cast<uint32_t>(term.cvOffsets[g + 1]) - p0;
                }
                pub.curveVertexCounts.push_back(static_cast<int>(len));
                for (uint32_t v = 0; v < len; ++v) {
                    const uint32_t p = p0 + v;
                    if (p >= term.px.size()) break;
                    pub.points.emplace_back(term.px[p], term.py[p], term.pz[p]);
                    // C2 (06 §4.1): widths/hairT are vertex channels on every
                    // tile. A chain whose terminal never wrote them (grow-only:
                    // kPlanePoints|kPlaneHairT) still publishes FULL-SIZE planes
                    // from desc defaults (02 §2.6/§2.14: width 0.01, look bake),
                    // never a wrong-size array — SI-1 sizes every non-empty
                    // vertex plane against points.
                    float w = 0.01f;
                    if (p < term.width.size()) w = term.width[p];
                    else if (!term.width.empty()) w = term.width.back();
                    pub.widths.push_back(w);
                    pub.hairT.push_back(p < term.hairT.size() ? term.hairT[p] : 0.0f);
                    for (ExtraPlanePublication *extra : vertexPlanes)
                        _GatherPlaneElement(*extra->source, p, &extra->output);
                }
                // hairId: UsdGenHash32(curveId, 0) / 2^32 in [0,1) (06, S29).
                pub.hairId.push_back(g < term.curveId.size()
                    ? UsdGenHairId(term.curveId[g]) : 0.0f);
                if (!term.rootUV.empty() && g < term.rootUV.size())
                    pub.st.push_back(term.rootUV[g]);
                if (colorMode == Preview) {
                    std::vector<GfVec3f> const &colors = previewColors.colors;
                    if (previewColors.perCv) {
                        for (uint32_t v = 0; v < len; ++v)
                            pub.displayColor.push_back(size_t(p0) + v < colors.size()
                                ? colors[size_t(p0) + v] : UsdGenPreviewMissingColor());
                    } else {
                        pub.displayColor.push_back(g < colors.size()
                            ? colors[g] : UsdGenPreviewMissingColor());
                    }
                } else if (colorMode == Plane)
                    _GatherColor(*displayColor, g, p0, &pub.displayColor);
                else if (colorMode == SourcePlane)
                    _GatherColor(*sourceColor, g, p0, &pub.displayColor);
                else if (colorMode == Root)
                    pub.displayColor.push_back(look.rootColor);
                for (ExtraPlanePublication *extra : uniformPlanes)
                    _GatherPlaneElement(*extra->source, g, &extra->output);
            }
        }
    }
    for (auto& extra : extraPlanes)
        pub.extraUniform.push_back(std::move(extra.output));
    std::sort(pub.extraUniform.begin(), pub.extraUniform.end(),
              [](UsdGenPlane const& left, UsdGenPlane const& right) {
                  return left.name < right.name;
              });

        if (!useBulk) {
        // Extent is a pure function of the published points (03 §6.3: min/max
        // fused into the interleave loop — but InterleaveTile skips untouched
        // tiles, so tv.extent is empty on any tile this commit did not evaluate
        // while its points are real). Reduce over pub.points in memory: same
        // (g,p0,len) ragged walk already emitted them above, one cheap pass.
        GfRange3f e;
        for (GfVec3f const &pt : pub.points) e.ExtendBy(pt);
        if (e.IsEmpty()) e = GfRange3f(GfVec3f(0.f), GfVec3f(0.f));
        pub.extentMin = GfVec3d(e.GetMin()[0], e.GetMin()[1], e.GetMin()[2]);
        pub.extentMax = GfVec3d(e.GetMax()[0], e.GetMax()[1], e.GetMax()[2]);
    }

    UsdGenNodeKey dependencySurface;
    if (hasDep) {
        for (UsdGenInputRecord const &record : UsdGenHydraSurfaceInputs(spec))
            if (UsdGenHydraSurfaceId(record) == depSurface) {
                dependencySurface = record.key;
                break;
            }
    }
    _PresentationScalars const scalars = _BuildPresentationScalars(
        spec, dependencySurface);
    pub.xformMatrix = scalars.xformMatrix;
    pub.purpose = scalars.purpose;
    pub.visibility = scalars.visibility;
    pub.material = scalars.material;
    pub.materialPurpose = scalars.materialPurpose;
    if (previewColors.active) {
        // The look's shader blends toward the tip colour; the preview shows
        // exactly the published displayColor, over any authored material.
        pub.material = UsdGenPrimRef{spec.description,
            UsdGenPreviewMaterialRole(presentation.previewShading), 0};
        pub.materialPurpose = hydraCurvesTokens::allPurpose();
    }
    pub.refineLevel = scalars.refineLevel;
    pub.origin = scalars.origin;
    pub.dependency = scalars.dependency;
    return pub;
}

bool RefreshTilePoints(UsdGenChunkGroup const &tv, UsdGenCurveColumns const &term,
                       UsdGenSpan<const UsdGenChunkRange> chunks,
                       UsdGenTilePublication const &previous, UsdGenTilePublication *out)
{
    // The same chunk -> live curve -> CV walk as BuildTile, so
    // the points land in the order the previous build published them; any
    // disagreement with the previous curve layout refuses the refresh.
    size_t const count = previous.points.size();
    VtVec3fArray points(count);
    GfVec3f *dst = points.data();
    size_t k = 0, curve = 0;
    GfRange3f e;
    for (uint32_t i = 0; i < tv.chunkCount; ++i) {
        if (size_t(tv.firstChunk) + i >= chunks.size()) return false;
        UsdGenChunkRange const &cd = chunks[tv.firstChunk + i];
        for (uint32_t c = 0; c < cd.live; ++c) {
            uint32_t const g = cd.first[0] + c;
            uint32_t p0 = 0, len = 0;
            if (cd.stride[1] != 0) {
                len = cd.stride[1];
                p0 = cd.first[1] + c * cd.stride[1];
            } else if (g + 1 < term.cvOffsets.size()) {   // ragged chunk
                p0 = static_cast<uint32_t>(term.cvOffsets[g]);
                len = static_cast<uint32_t>(term.cvOffsets[g + 1]) - p0;
            }
            if (curve >= previous.curveVertexCounts.size() ||
                previous.curveVertexCounts[curve] != static_cast<int>(len) ||
                size_t(p0) + len > term.px.size() || k + len > count)
                return false;
            ++curve;
            for (uint32_t v = 0; v < len; ++v) {
                uint32_t const p = p0 + v;
                GfVec3f const pt(term.px[p], term.py[p], term.pz[p]);
                dst[k++] = pt;
                e.ExtendBy(pt);
            }
        }
    }
    if (k != count || curve != previous.curveVertexCounts.size()) return false;
    *out = previous;
    out->points = std::move(points);
    if (e.IsEmpty()) e = GfRange3f(GfVec3f(0.f), GfVec3f(0.f));
    out->extentMin = GfVec3d(e.GetMin()[0], e.GetMin()[1], e.GetMin()[2]);
    out->extentMax = GfVec3d(e.GetMax()[0], e.GetMax()[1], e.GetMax()[2]);
    return true;
}

// The curve type's playback binding when every group matches the tile the
// publication carries for it, point for point; else null.
UsdGenRef<const UsdGenDeformBinding> BuildAnimationBinding(
    UsdGenPublishContext const &ctx, UsdGenPublishDesc const *publish,
    std::vector<UsdGenTilePublication> const &tiles)
{
    UsdGenRef<const UsdGenDeformBinding> binding;
    if (!publish || !publish->deformBinding || !publish->deformBinding(ctx, &binding) || !binding)
        return {};
    for (UsdGenDeformBindingGroup const &group : binding->groups) {
        auto tile = std::find_if(tiles.begin(), tiles.end(),
                                 [&](auto const &t) { return t.tile == group.group; });
        if (tile == tiles.end() || tile->points.size() != group.restPoints.size()) return {};
    }
    return binding;
}

// Per-prim primvar-name list from a signature; null when the path is new.
std::vector<std::string> const *
_namesFor(UsdGenPrimSetSignature const &sig, UsdGenPrimRef const &prim)
{
    for (size_t i = 0; i < sig.prims.size(); ++i)
        if (sig.prims[i] == prim)
            return &sig.primvarNames[i];
    return nullptr;
}

// next-names minus prev-names, both kept sorted by the builder.
std::vector<TfToken>
_newPrimvars(UsdGenPrimSetSignature const &prev, UsdGenPrimRef const &prevPath,
             UsdGenPrimSetSignature const &next, UsdGenPrimRef const &nextPath)
{
    std::vector<std::string> const *a = _namesFor(prev, prevPath);
    std::vector<std::string> const *b = _namesFor(next, nextPath);
    std::vector<TfToken> out;
    size_t const na = a ? a->size() : 0, nb = b ? b->size() : 0;
    size_t i = 0, j = 0;
    while (i < na && j < nb) {
        if ((*a)[i] < (*b)[j])          ++i;
        else if ((*b)[j] < (*a)[i])     out.emplace_back((*b)[j++]);
        else                            { ++i; ++j; }
    }
    while (j < nb) out.emplace_back((*b)[j++]);
    return out;
}

template <typename T>
bool _changed(VtArray<T> const &a, VtArray<T> const &b)
{
    return !a.IsIdentical(b);   // data-pointer compare (03 §6.1), never memcmp
}


UsdGenRef<const UsdGenPublication> BuildPublication(UsdGenPublishContext const &ctx)
{
    if (!ctx.spec || !ctx.graph || !ctx.terminal || !ctx.state) return {};
    UsdGenGraphSpec const &spec = *ctx.spec;
    _State &state = static_cast<_State &>(*ctx.state);
    UsdGenCurveColumns const term = UsdGenCurveColumns::Of(*ctx.terminal, &state.slots);
    UsdGenCurvePublication const *const prev =
        ctx.previous ? &UsdGenCurvePublicationOf(ctx.previous) : nullptr;
    UsdGenRef<UsdGenCurvePublication> const out = UsdGenNewCurvePublication();
    UsdGenCurvePublication &gen = *out;
    gen.tiles.reserve(ctx.groups.size());
    TileInputs const inputs{spec, term, ctx.chunks, state.colors};

    // usdGen:preview recolours every tile, including those whose geometry
    // this cook did not touch.
    UsdGenPublishDesc const *const publish = CurvePublishDesc();
    {
        UsdGenPreviewValues values;
        if (!publish || !publish->previewValues ||
            !publish->previewValues(ctx, state.hooks.get(), &values, ctx.diag))
            values = UsdGenPreviewValues{};
        PreviewColors(spec, term, values, &state.colors);
    }
    gen.colorDigest = state.colors.digest;
    // ...and so does usdGen:look, which is baked into every tile's
    // displayColor and (since the look travels as constant primvars too)
    // into its extra planes. A look-only edit touches no point and no
    // topology, so without this it would reach the synthetic material and
    // nothing else -- the strands would keep the previous root colour.
    {
        auto mix = [](uint64_t *h, uint64_t v) {
            *h ^= v + 0x9e3779b97f4a7c15ull + (*h << 6) + (*h >> 2);
            *h ^= *h >> 30; *h *= 0xbf58476d1ce4e5b9ull;
            *h ^= *h >> 27; *h *= 0x94d049bb133111ebull;
            *h ^= *h >> 31;
        };
        auto mixF = [&](uint64_t *h, float v) {
            uint32_t bits = 0; std::memcpy(&bits, &v, sizeof(bits));
            mix(h, bits);
        };
        UsdGenLookDesc const &lk = PresentationOf(spec).look;
        uint64_t h = gen.colorDigest;
        for (int i = 0; i < 3; ++i) { mixF(&h, lk.rootColor[i]); mixF(&h, lk.tipColor[i]); }
        for (GfVec3f const &c : lk.rampColors)
            for (int i = 0; i < 3; ++i) mixF(&h, c[i]);
        for (float p : lk.rampPositions) mixF(&h, p);
        mixF(&h, lk.rampExponent);
        mixF(&h, lk.hueJitter);
        mixF(&h, lk.valueJitter);
        mix(&h, static_cast<uint64_t>(static_cast<uint32_t>(lk.jitterSeed)));
        mix(&h, lk.rampInterpolation.Hash());
        mix(&h, lk.bakeMode.Hash());
        mix(&h, lk.bakeTarget.Hash());
        mix(&h, lk.bakePrimvar.Hash());
        gen.colorDigest = h;
    }
    const bool recolour = prev && prev->colorDigest != gen.colorDigest;
    Phase(ctx, "preview");

    TRACE_SCOPE("usdGen build tile publications");
    // Serial triage (cheap): carry-over and streamed reuse resolve here in
    // ctx.groups order; only the BuildTile calls fan out.
    // Every tile reads only its own view plus shared-immutable run state
    // (terminal buffer, graph, desc, preview colors) and writes only its
    // own publication, so any build order fills identical tiles -- and the
    // streaming path already invokes BuildTile concurrently
    // from scheduler workers, so the function is production-hardened for
    // it. Assembly below restores ctx.groups order, the progress emits
    // keep their serial sequence, and the first failing tile's exception
    // still wins (captured per slot, rethrown in order). One deliberate
    // edge difference: on a layout-corruption throw with a progress
    // callback set, no partial progress emits (serial interleaved them);
    // success paths -- the only contracted case -- are identical.
    // The engine joined every group callback: the streamed set is final.
    auto findStreamed = [&ctx](UsdGenTileId id) -> UsdGenTilePublication const * {
        auto it = std::lower_bound(ctx.streamed.begin(), ctx.streamed.end(), id,
            [](UsdGenPublishedGroup const &e, UsdGenTileId t) { return e.group < t; });
        if (it == ctx.streamed.end() || it->group != id) return nullptr;
        UsdGenCurvePublication const &one = UsdGenCurvePublicationOf(it->publication);
        return one.tiles.empty() ? nullptr : &one.tiles.front();
    };
    size_t rebuiltTiles = 0;
    size_t rebuildCvs = 0;
    std::vector<UsdGenChunkGroup const *> buildViews;
    // Points-only republication: the previous tile with fresh points (every
    // other channel shared), when nothing but points moved since it was built.
    bool const refreshPoints = ctx.pointsOnly && prev && !recolour && !ctx.rebuildAll &&
        !state.colors.active;
    size_t refreshedTiles = 0;
    for (UsdGenChunkGroup const &tv : ctx.groups) {
        // E-4: untouched tiles carry over their publication wholesale (the
        // VtArray copies share buffers, so step 7 sees IsIdentical == true).
        const bool rebuild =
            ctx.topologyChanged || (tv.dirtyRoles & kGroupPositionDirty) ||
            (tv.dirtyRoles & kGroupWidthDirty) || recolour ||
            ctx.rebuildAll;
        const UsdGenTilePublication *previous = nullptr;
        if (prev) {
            auto it = std::lower_bound(
                prev->tiles.begin(), prev->tiles.end(), tv.id,
                [](UsdGenTilePublication const &p, UsdGenTileId t) {
                    return p.tile < t;
                });
            if (it != prev->tiles.end() && it->tile == tv.id) previous = &*it;
        }
        if (!rebuild && previous) {
            gen.tiles.push_back(*previous);
            continue;
        }
        ++rebuiltTiles;
        if (refreshPoints && previous && !(tv.dirtyRoles & kGroupWidthDirty)) {
            UsdGenTilePublication refreshed;
            if (RefreshTilePoints(tv, term, ctx.chunks, *previous, &refreshed)) {
                gen.tiles.push_back(std::move(refreshed));
                ++refreshedTiles;
                continue;
            }
        }
        if (UsdGenTilePublication const *streamed = findStreamed(tv.id)) {
            gen.tiles.push_back(*streamed);
            continue;
        }
        rebuildCvs += tv.live[1];
        buildViews.push_back(&tv);
    }
    std::vector<UsdGenTilePublication> built(buildViews.size());
    // Threaded over tiles for big rebuilds on the cook's worker pool
    // (idle here: the run has returned): each worker builds whole tiles into
    // indexed slots. Small rebuilds keep the serial driver below. Chunks
    // cap at 8 like every other site: uncapped auto-partitioning over the
    // 20-thread arena regressed this loop 5x (oversubscription on the
    // pinned cores plus co-tenant contention).
    int const tileWorkers = ctx.dispatcher ? ctx.dispatcher->MaxConcurrency() : 1;
    size_t const tileChunks =
        (tileWorkers > 1 && rebuildCvs > 32768)
            ? std::min({size_t(tileWorkers), size_t(8), buildViews.size()})
            : 1;
    if (tileChunks > 1) {
        std::vector<std::exception_ptr> buildError(buildViews.size());
        UsdGenChunkGroup const *const *views = buildViews.data();
        UsdGenTilePublication *slots = built.data();
        std::exception_ptr *errors = buildError.data();
        size_t const jobs = buildViews.size();
        struct Build {
            TileInputs const *in;
            UsdGenChunkGroup const *const *views;
            UsdGenTilePublication *slots;
            std::exception_ptr *errors;
            size_t jobs, chunks;
        } build{&inputs, views, slots, errors, jobs, tileChunks};
        ctx.dispatcher->ParallelFor(tileChunks, [](size_t c, void *p) {
            auto const &b = *static_cast<Build const *>(p);
            size_t const i0 = (c * b.jobs) / b.chunks;
            size_t const i1 = ((c + 1) * b.jobs) / b.chunks;
            for (size_t i = i0; i < i1; ++i) {
                try {
                    b.slots[i] = BuildTile(*b.in, *b.views[i]);
                } catch (...) {
                    b.errors[i] = std::current_exception();
                }
            }
        }, &build, 1);
        for (std::exception_ptr const &e : buildError)
            if (e)
                std::rethrow_exception(e);
    } else {
        for (size_t i = 0; i < buildViews.size(); ++i)
            built[i] = BuildTile(inputs, *buildViews[i]);
    }
    for (size_t i = 0; i < built.size(); ++i) {
        gen.tiles.push_back(std::move(built[i]));
        if (ctx.emit)
            EmitOne(ctx, gen.tiles.back());
    }
    std::sort(gen.tiles.begin(), gen.tiles.end(),
              [](UsdGenTilePublication const &a, UsdGenTilePublication const &b) {
                  return a.tile < b.tile;
              });
    Phase(ctx, refreshedTiles == rebuiltTiles && refreshedTiles ? "tile points" : "tiles",
          rebuiltTiles, gen.tiles.size());
    if (UsdGenRbfAnimationEnabled())
        gen.animationBinding = BuildAnimationBinding(ctx, publish, gen.tiles);

    // Whole-groom optical depth includes neighboring tiles. Unchanged
    // geometry reuses the immutable planes, including on look-only edits.
    {
        UsdGenFurOcclusionParams occlusion;
        occlusion.dispatcher = ctx.dispatcher;
        if (!state.furCache) state.furCache = std::make_shared<UsdGenFurOcclusionCache>();
        occlusion.cache = state.furCache.get();
        // The emitting surfaces are opaque: without them light reaches hair
        // through the scalp (a host renderer injects the opaque depth for the same
        // reason). Deformed points at this frame, in the same world space the
        // tiles were transformed into.
        // A GeomSubset surface carries its whole parent mesh, which occludes
        // whole and once, however many subsets of it the groom binds.
        occlusion.occluders.reserve(UsdGenHydraSurfaceInputs(spec).size());
        for (uint32_t i = 0; publish && publish->occluder; ++i) {
            UsdGenNodeKey const key = publish->occluder(ctx, i);
            if (!key.IsValid()) break;
            UsdGenInputRecord const *const surfaceRecord = UsdGenHydraFindSurface(spec, key);
            if (!surfaceRecord) continue;
            UsdGenHydraSurfaceMesh const surface = UsdGenHydraSurfaceMeshOf(*surfaceRecord);
            UsdGenFurOccluder occluder;
            occluder.points.assign(surface.points.begin(), surface.points.end());
            occluder.faceVertexCounts.assign(surface.faceVertexCounts.begin(),
                                             surface.faceVertexCounts.end());
            occluder.faceVertexIndices.assign(surface.faceVertexIndices.begin(),
                                              surface.faceVertexIndices.end());
            occluder.worldMatrix = surface.worldMatrix;
            occlusion.occluders.push_back(std::move(occluder));
        }
        // Hair shadowing the scalp. Stock Storm casts no shadow from any
        // UsdLux light, so a synthetic cap over the emitting surface is the
        // only route; it is built from the same volume, at no extra sweep
        // beyond the hair-only one it needs.
        occlusion.scalpShadow = !occlusion.occluders.empty() &&
                                TfGetEnvSetting(USDGEN_SCALP_SHADOW);
        // Local density for ambient darkening: a gather of a grid the bake
        // already has, so it needs no occluders and no extra sweep.
        occlusion.localDensity = TfGetEnvSetting(USDGEN_FUR_DENSITY);
        // The reuse path leaves the cap untouched, so seed it: unchanged
        // geometry shares the previous immutable arrays.
        if (prev) gen.scalpShadow = prev->scalpShadow;
        UsdGenBuildFurOcclusion(&gen.tiles, prev ? &prev->tiles : nullptr,
                                occlusion, &state.furVolumeKey, &gen.scalpShadow);
        if (!gen.scalpShadow.IsEmpty()) {
            _PresentationScalars const scalars = _BuildPresentationScalars(
                spec, UsdGenNodeKey());
            gen.scalpShadow.prim =
                UsdGenPrimRef{spec.description, UsdGenPublishRole::ScalpShadow, 0};
            gen.scalpShadow.purpose = scalars.purpose;
            gen.scalpShadow.visibility = scalars.visibility;
            gen.scalpShadow.material =
                UsdGenPrimRef{spec.description, UsdGenPublishRole::ScalpShadowMaterial, 0};
        }
    }
    Phase(ctx, "occlusion");

    // Signature: the prim-set identity step 7 diffs structurally (03 §6.1).
    gen.signature.tileCount = static_cast<uint32_t>(gen.tiles.size());
    gen.signature.instancerCount = 0;
    // The end of the look's path into Storm: the colour a tile actually
    // publishes. It is baked from the presentation's look, so this and the
    // INGRESS capture line together say whether an authored look survived the whole trip --
    // the failure mode is silent, because a defaulted look is still a
    // plausible dark brown.
    if (TfDebug::IsEnabled(USDGEN_INGRESS) && !gen.tiles.empty() &&
        !gen.tiles.front().displayColor.empty()) {
        GfVec3f const &c = gen.tiles.front().displayColor.front();
        TF_DEBUG(USDGEN_INGRESS).Msg(
            "usdGen tile displayColor[0] = %.4f %.4f %.4f (look rootColor "
            "%.4f %.4f %.4f)\n",
            double(c[0]), double(c[1]), double(c[2]),
            double(PresentationOf(spec).look.rootColor[0]),
            double(PresentationOf(spec).look.rootColor[1]),
            double(PresentationOf(spec).look.rootColor[2]));
    }
    gen.signature.prims.reserve(gen.tiles.size());
    gen.signature.primTypes.assign(gen.tiles.size(), "basisCurves");
    gen.signature.primvarNames.reserve(gen.tiles.size());
    for (UsdGenTilePublication const &t : gen.tiles) {
        gen.signature.prims.push_back(t.prim);
        std::vector<std::string> names{"points", "widths", "hairT", "hairId"};
        if (!t.st.empty())            names.emplace_back("st");
        if (!t.displayColor.empty())  names.emplace_back("displayColor");
        if (!t.bakeColor.empty())     names.emplace_back("bakeColor");
        if (!t.velocities.empty())    names.emplace_back("velocities");
        for (UsdGenPlane const &p : t.extraUniform)
            names.push_back(p.name.GetString());
        std::sort(names.begin(), names.end());
        names.erase(std::unique(names.begin(), names.end()), names.end());
        gen.signature.primvarNames.push_back(std::move(names));
    }

    gen.groups = static_cast<uint32_t>(gen.tiles.size());
    return out;
}

UsdGenRef<const UsdGenPublication> BuildGroupPublication(UsdGenPublishContext const &ctx,
                                                         uint16_t group)
{
    if (!ctx.spec || !ctx.terminal || group >= ctx.groups.size()) return {};
    // Streamed groups never carry preview colours: the engine streams only
    // when no preview is active.
    UsdGenPreviewColors const none{};
    UsdGenCurveColumns const term = UsdGenCurveColumns::Of(*ctx.terminal);
    TileInputs const inputs{*ctx.spec, term, ctx.chunks, none};
    UsdGenRef<UsdGenCurvePublication> one = UsdGenNewCurvePublication();
    one->tiles.push_back(BuildTile(inputs, ctx.groups[group]));
    one->groups = 1;
    return one;
}

UsdGenRef<const UsdGenPublication> Diff(UsdGenPublication const *prevPublication,
                                        UsdGenPublication const &nextPublication)
{
    TRACE_FUNCTION();
    UsdGenCurvePublication const &prev = UsdGenCurvePublicationOf(prevPublication);
    UsdGenCurvePublication const &next = UsdGenCurvePublicationOf(&nextPublication);
    // Both tile vectors are sorted by tile id (publication invariant), so a
    // single merge walk classifies every prim.
    UsdGenRef<UsdGenCurveDirtyReport> const out(new UsdGenCurveDirtyReport);
    UsdGenCurveDirtyReport &report = *out;
    size_t i = 0, j = 0;
    while (i < prev.tiles.size() || j < next.tiles.size()) {
        UsdGenTileDirty td;
        if (j < next.tiles.size() &&
            (i >= prev.tiles.size() || next.tiles[j].tile < prev.tiles[i].tile))
        {   // prim appears only in next -> added.
            td.tile = next.tiles[j].tile;
            td.prim = next.tiles[j].prim;
            td.added = true;
            td.pointsDirty = true;   // fresh prim: everything is new content
            td.newPrimvars = _newPrimvars(prev.signature, td.prim,
                                          next.signature, td.prim);
            ++j;
        }
        else if (i < prev.tiles.size() &&
                 (j >= next.tiles.size() || prev.tiles[i].tile < next.tiles[j].tile))
        {   // prim disappears -> removed.
            td.tile = prev.tiles[i].tile;
            td.prim = prev.tiles[i].prim;
            td.removed = true;
            ++i;
        }
        else {
            UsdGenTilePublication const &a = prev.tiles[i];
            UsdGenTilePublication const &b = next.tiles[j];
            td.tile = b.tile;
            td.prim = b.prim;
            td.pointsDirty = _changed(a.points, b.points);
            td.widthsDirty = _changed(a.widths, b.widths);
            td.xformDirty  = !(a.xformMatrix == b.xformMatrix);
            auto mark = [&td](TfToken const &name, bool dirty) {
                if (dirty) td.dirtyPrimvars.push_back(name);
            };
            mark(hydraCurvesTokens::hairT(),  _changed(a.hairT,  b.hairT));
            mark(hydraCurvesTokens::hairId(), _changed(a.hairId, b.hairId));
            mark(hydraCurvesTokens::st(),     _changed(a.st,     b.st));
            // A colour that moves between one-per-curve and one-per-CV (the
            // look's bakeMode, or a usdGen:preview) changes the primvar's
            // interpolation. Hydra keeps its primvar descriptors across a
            // value-only dirty, so that is a container-level dirty.
            auto colorRate = [](UsdGenTilePublication const &t) {
                if (t.displayColor.empty()) return 0;
                if (t.displayColor.size() == t.curveVertexCounts.size()) return 1;
                return t.displayColor.size() == t.points.size() ? 2 : 3;
            };
            if (colorRate(a) != colorRate(b))
                td.newPrimvars.push_back(hydraCurvesTokens::displayColor());
            else
                mark(hydraCurvesTokens::displayColor(), _changed(a.displayColor, b.displayColor));
            mark(hydraCurvesTokens::bakeColor(),    _changed(a.bakeColor,    b.bakeColor));
            mark(hydraCurvesTokens::velocities(),   _changed(a.velocities,   b.velocities));
            // Named uniform planes (both vectors sorted by name): common names
            // compared by BUFFER IDENTITY like every other payload above,
            // next-only names are new primvars. Content comparison here
            // re-scanned megabytes of tau planes per cook (~2ms at bench
            // scale); rebuilt tiles always take fresh buffers, so identity
            // reports them dirty -- conservative-correct (re-upload
            // identical bytes), and identical recomputes always co-dirty
            // displayColor or the topology, which Storm re-uploads whole
            // anyway, so no extra traffic results. Carried tiles share
            // buffers and stay clean, as before.
            for (UsdGenPlane const &pb : b.extraUniform) {
                bool paired = false;
                for (UsdGenPlane const &pa : a.extraUniform) {
                    if (pa.name != pb.name) continue;
                    paired = true;
                    if (pa.interpolation != pb.interpolation || pa.arity != pb.arity ||
                        pa.type != pb.type)
                        td.newPrimvars.push_back(pb.name);
                    else
                        mark(pb.name, !pa.f.IsIdentical(pb.f) ||
                                     !pa.i.IsIdentical(pb.i));
                    break;
                }
                if (!paired) td.newPrimvars.push_back(pb.name);
            }
            // Primvar-set growth not visible in payloads (e.g. empty->empty)
            // still comes from the signatures.
            std::vector<TfToken> sigNew =
                _newPrimvars(prev.signature, a.prim, next.signature, b.prim);
            for (TfToken const &t : sigNew) {
                if (std::find(td.newPrimvars.begin(), td.newPrimvars.end(), t)
                    == td.newPrimvars.end())
                    td.newPrimvars.push_back(t);
            }
            ++i;
            ++j;
        }
        report.tiles.push_back(std::move(td));
    }
    // Surface xform dirt is a scene-index concern the engine cannot observe;
    // the imaging dirty router sets this bit itself.
    report.surfaceXformDirty = false;
    return out;
}

size_t Bytes(UsdGenPublication const &publication)
{
    UsdGenCurvePublication const &p = UsdGenCurvePublicationOf(&publication);
    size_t total = sizeof(UsdGenCurvePublication);
    auto addTile = [&total](UsdGenTilePublication const &tile) {
        auto addPlane = [&total](UsdGenPlane const &plane) {
            return _AddBytes(&total, 1, sizeof(UsdGenPlane)) &&
                _AddBytes(&total, plane.f.size(), sizeof(float)) &&
                _AddBytes(&total, plane.i.size(), sizeof(int));
        };
        if (!_AddBytes(&total, 1, sizeof(UsdGenTilePublication)) ||
            !_AddBytes(&total, tile.basis.size(), sizeof(char)) ||
            !_AddBytes(&total, tile.curveVertexCounts.size(), sizeof(int)) ||
            !_AddBytes(&total, tile.points.size(), sizeof(GfVec3f)) ||
            !_AddBytes(&total, tile.widths.size(), sizeof(float)) ||
            !_AddBytes(&total, tile.hairT.size(), sizeof(float)) ||
            !_AddBytes(&total, tile.hairId.size(), sizeof(float)) ||
            !_AddBytes(&total, tile.st.size(), sizeof(GfVec2f)) ||
            !_AddBytes(&total, tile.displayColor.size(), sizeof(GfVec3f)) ||
            !_AddBytes(&total, tile.bakeColor.size(), sizeof(GfVec3f)) ||
            !_AddBytes(&total, tile.velocities.size(), sizeof(GfVec3f)))
            return false;
        for (UsdGenPlane const &plane : tile.extraUniform)
            if (!addPlane(plane)) return false;
        return true;
    };
    for (auto const &tile : p.tiles)
        if (!addTile(tile)) return 0;
    for (auto const &tile : p.guides)
        if (!addTile(tile)) return 0;
    if (p.animationBinding) {
        auto const& binding = *p.animationBinding;
        size_t const n = binding.sampleIndices.size(), order = n + 4;
        if (!_AddBytes(&total, 1, sizeof(UsdGenDeformBinding)) ||
            !_AddBytes(&total, order * order + 9 * order, sizeof(double)) ||
            !_AddBytes(&total, n + order, sizeof(size_t))) return 0;
        for (auto const& group : binding.groups)
            if (!_AddBytes(&total, 1, sizeof(UsdGenDeformBindingGroup)) ||
                !_AddBytes(&total, group.restPoints.size(), sizeof(GfVec3f))) return 0;
    }
    for (UsdGenInstancerPublication const &instancer : p.instancers) {
        if (!_AddBytes(&total, 1, sizeof(UsdGenInstancerPublication)) ||
            !_AddBytes(&total, instancer.prototypes.size(), sizeof(UsdGenPrimRef)) ||
            !_AddBytes(&total, instancer.instanceIndices.size(), sizeof(VtIntArray)) ||
            !_AddBytes(&total, instancer.mask.size(), sizeof(bool)))
            return 0;
        for (VtIntArray const &indices : instancer.instanceIndices)
            if (!_AddBytes(&total, indices.size(), sizeof(int))) return 0;
    }
    if (!_AddBytes(&total, 1, sizeof(UsdGenPrimSetSignature)) ||
        !_AddBytes(&total, p.signature.prims.size(), sizeof(UsdGenPrimRef)) ||
        !_AddBytes(&total, p.signature.primTypes.size(), sizeof(std::string)) ||
        !_AddBytes(&total, p.signature.primvarNames.size(),
                   sizeof(std::vector<std::string>))) return 0;
    for (std::string const &type : p.signature.primTypes)
        if (!_AddBytes(&total, type.size(), sizeof(char))) return 0;
    for (auto const &names : p.signature.primvarNames) {
        if (!_AddBytes(&total, names.size(), sizeof(std::string))) return 0;
        for (std::string const &name : names)
            if (!_AddBytes(&total, name.size(), sizeof(char))) return 0;
    }
    return total;
}

std::unique_ptr<UsdGenPublisherState> CreateState()
{
    auto state = std::make_unique<_State>();
    UsdGenPublishDesc const *const publish = CurvePublishDesc();
    if (publish && publish->createState) state->hooks.reset(publish->createState());
    return state;
}

}  // namespace

bool UsdGenRbfAnimationEnabled() { return TfGetEnvSetting(USDGEN_STORM_GPU_DEFORM); }

UsdGenCurvePublication::UsdGenCurvePublication() noexcept
    : UsdGenPublication(UsdGenHydraCurveSetType(), kUsdGenCurvePublicationKind)
{
}

UsdGenCurveDirtyReport::UsdGenCurveDirtyReport() noexcept
    : UsdGenPublication(UsdGenHydraCurveSetType(), kUsdGenCurveDirtyReportKind)
{
}

UsdGenRef<UsdGenCurvePublication> UsdGenNewCurvePublication()
{
    // The type id exists once this adapter registered.
    (void)usdGenHydra::UsdGenHydraCurvesEnsureRegistered();
    return UsdGenRef<UsdGenCurvePublication>(new UsdGenCurvePublication);
}

UsdGenCurvePublication const &UsdGenCurvePublicationOf(UsdGenPublication const *p) noexcept
{
    return p && p->kind == kUsdGenCurvePublicationKind && p->type == UsdGenHydraCurveSetType()
        ? static_cast<UsdGenCurvePublication const &>(*p) : EmptyPublication(gEmptyPublication);
}

UsdGenCurvePublication const &UsdGenCurvePublicationOf(UsdGenGeneration const &generation) noexcept
{
    return UsdGenCurvePublicationOf(generation.publication.Get());
}

UsdGenCurveDirtyReport const &UsdGenCurveDirtyReportOf(UsdGenPublication const *p) noexcept
{
    return p && p->kind == kUsdGenCurveDirtyReportKind && p->type == UsdGenHydraCurveSetType()
        ? static_cast<UsdGenCurveDirtyReport const &>(*p) : EmptyPublication(gEmptyReport);
}

UsdGenRef<const UsdGenPublication> UsdGenDiffCurvePublications(UsdGenPublication const *prev,
                                                               UsdGenPublication const &next)
{
    return Diff(prev, next);
}

bool UsdGenCurvePublisherRegistered() noexcept
{
    UsdGenTypeId const type = UsdGenHydraCurveSetType();
    UsdGenPublisherDef const *const def =
        type == UsdGenTypeId::Invalid ? nullptr : UsdGenFindPublisher(type);
    return def && def->build == &BuildPublication;
}

bool UsdGenRegisterCurvePublisher(UsdGenRegistrar &r)
{
    UsdGenPublisherDef def;
    def.primitive = UsdGenHydraCurveSetType();
    def.build = &BuildPublication;
    def.buildGroup = &BuildGroupPublication;
    def.diff = &Diff;
    def.bytes = &Bytes;
    def.createState = &CreateState;
    // The CurveSet type registered first, so these read its id without
    // building the registries.
    (void)EmptyPublication(gEmptyPublication);
    (void)EmptyPublication(gEmptyReport);
    return r.RegisterPublisher(def);
}

}  // namespace usdGen
