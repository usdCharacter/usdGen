// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenCardSourceOp implementation. See cardSource.h for the
// behaviour and ops/cardGeometry.h for the card grids.
#include "usdGen/ops/cardSource.h"

#include "usdGen/opParams.h"
#include "usdGen/ops/cardGeometry.h"
#include "usdGen/ops/opUtil.h"
#include "usdGen/ops/strandUtil.h"
#include "usdGen/scheduler.h"
#include "usdGenMath/usdGenMath/kernels.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace {

struct UsdGenCardSourceCapture final : public UsdGenCapture
{
    bool OwnsBuffer() const override { return true; }
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenCardSourceCapture>(*this);
    }
};

const TfToken sCvCount{"cvCount"}, sLayer{"cardSource:layer"}, sUvId{"cardSource:uvId"},
    sTile{"cardSource:tile"}, sBraid{"cardSource:braid"}, sAmount{"cardSource:amount"},
    sRootFlatten{"cardSource:rootFlatten"}, sSmooth{"cardSource:smooth"},
    sMaxStrands{"cardSource:maxStrands"}, sBraidWidth{"cardSource:braidWidth"},
    sBraidFlatness{"cardSource:braidFlatness"};
const TfToken sStrandIndex{"strandIndex"}, sStrandSeed{"strandSeed"}, sCardIndex{"cardIndex"};

TfTokenVector const kReferenceInputs{TfToken("cards"), TfToken("cardStrands")};
TfTokenVector const kOutputs{sCardIndex, sStrandIndex, sStrandSeed};
TfTokenVector const kTopo = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("enabled"));   // generator: topology-class (02 §6.2)
    v.push_back(TfToken("cards"));
    v.push_back(TfToken("cardStrands"));
    for (TfToken const &t : {sLayer, sUvId, sTile, sBraid, sAmount, sRootFlatten, sSmooth,
                             sMaxStrands, sBraidWidth, sBraidFlatness, sCvCount})
        v.push_back(t);
    return v;
}();
TfTokenVector const kValue = UsdGenBaseValueParams();

struct Settings
{
    int layer = -1, uvId = -1, tile = -1;
    bool braid = false, smooth = true;
    float amount = 1.0f, rootFlatten = 0.0f, braidWidth = 1.0f, braidFlatness = 0.55f;
    int cvCount = 8, maxStrands = 2000000;
};

bool ReadSettings(UsdGenParamView const *p, Settings *s, UsdGenDiagnostics *diag)
{
    if (p) {
        s->layer = p->GetInt(sLayer, -1);
        s->uvId = p->GetInt(sUvId, -1);
        s->tile = p->GetInt(sTile, -1);
        s->braid = p->GetBool(sBraid, false);
        s->amount = float(p->GetDouble(sAmount, 1.0));
        s->rootFlatten = float(p->GetDouble(sRootFlatten, 0.0));
        s->smooth = p->GetBool(sSmooth, true);
        s->cvCount = p->GetInt(sCvCount, 8);
        s->maxStrands = p->GetInt(sMaxStrands, 2000000);
        s->braidWidth = float(p->GetDouble(sBraidWidth, 1.0));
        s->braidFlatness = float(p->GetDouble(sBraidFlatness, 0.55));
    }
    auto fail = [&](std::string const &message) {
        if (diag) diag->Error("UsdGenCardSource: " + message);
        return false;
    };
    if (s->cvCount < 2 || uint32_t(s->cvCount) > kUsdGenMaxCvsPerCurve)
        return fail("usdGen:cvCount must be in [2, " + std::to_string(kUsdGenMaxCvsPerCurve) + "]");
    if (s->maxStrands < 1) return fail("usdGen:cardSource:maxStrands must be >= 1");
    if (!std::isfinite(s->amount) || !std::isfinite(s->rootFlatten) ||
        !std::isfinite(s->braidWidth) || !std::isfinite(s->braidFlatness))
        return fail("card source amounts must be finite");
    s->amount = strand::Clamp01(s->amount);
    s->rootFlatten = strand::Clamp01(s->rootFlatten);
    s->braidFlatness = strand::Clamp01(s->braidFlatness);
    return true;
}

template <class F>
void ParallelFor(UsdGenWorkDispatcher *dispatcher, size_t count, F const &body)
{
    if (!dispatcher || count < 64) {
        for (size_t i = 0; i < count; ++i) body(i);
        return;
    }
    struct Payload { F const *body; } payload{&body};
    dispatcher->ParallelFor(count, [](size_t i, void *p) {
        (*static_cast<Payload *>(p)->body)(i);
    }, &payload);
}

using card::CurvePlane;
using card::DecodeCards;
using card::IsLibrarySet;
using card::IsSpineSet;
using Column = card::PlaneColumn;

// --- strand library --------------------------------------------------------------------

struct Library
{
    UsdGenCurveBuffer const *buffer = nullptr;
    std::vector<uint32_t> spans;
    struct Entry { std::vector<uint32_t> curves; uint64_t strandCount = 0; };
    std::map<std::pair<int, int>, Entry> entries;   // (uvId, tile) -> strands in order
    std::vector<int> index;                          // local strand index per curve

    Entry const *Find(int uvId, int tile) const
    {
        auto const it = entries.find({uvId, tile});
        return it == entries.end() ? nullptr : &it->second;
    }
};

bool DecodeLibrary(UsdGenCurveBuffer const &b, Library *lib, std::string *error)
{
    lib->buffer = &b;
    if (!opUtil::CurveSpans(b, &lib->spans, error)) return false;
    Column const uvId(b, "uvId"), tile(b, "tile"), index(b, "index"), count(b, "tileStrandCount");
    lib->index.assign(b.totalCurves, 0);
    std::map<std::pair<int, int>, uint64_t> listed;
    for (uint32_t c = 0; c < b.totalCurves; ++c) {
        std::pair<int, int> const key{uvId.Int(c, -1), tile.Int(c, -1)};
        auto &entry = lib->entries[key];
        uint64_t &seen = listed[key];
        lib->index[c] = index.Int(c, int(seen));
        ++seen;
        // The entry size counts strands the library omitted (<= 1 point);
        // without the plane, the listed strands are the entry.
        entry.strandCount = count.Has()
            ? std::max(entry.strandCount, uint64_t(std::max(count.Int(c, 0), 0))) : seen;
        if (lib->spans[c + 1] - lib->spans[c] >= 2) entry.curves.push_back(c);
    }
    return true;
}

// --- per-strand placement --------------------------------------------------------------

struct Root
{
    uint32_t span = 0;         // index into the built span list (non-braid)
    uint32_t source = 0;       // library curve (non-braid) or braid strand number
    uint32_t card = 0;         // spine curve
    uint64_t id = 0;
    GfVec3f offset{0.0f};      // braid: root offset in the card root frame
};

struct Scalp
{
    strand::TriangleBvh bvh;
    std::vector<GfVec3f> faceNormal;   // per polygon, area-weighted fan normal
    bool valid = false;
};

void RootFrameAt(Scalp const &scalp, strand::ClosestHit const &hit, GfVec3f const &fallbackN,
                 int *face, GfVec2f *uv, GfVec3f *T, GfVec3f *N, GfVec3f *B)
{
    *face = -1;
    *uv = GfVec2f(0.0f);
    GfVec3f n = fallbackN;
    GfVec3f e0(1.0f, 0.0f, 0.0f), e1(0.0f, 0.0f, 1.0f);
    if (scalp.valid && hit.hit) {
        strand::FanTriangle const &t = scalp.bvh.Triangles()[size_t(hit.triangle)];
        *face = t.face;
        *uv = scalp.bvh.Uv(hit.triangle, hit.bary);
        n = (t.face >= 0 && size_t(t.face) < scalp.faceNormal.size())
            ? scalp.faceNormal[size_t(t.face)] : scalp.bvh.FaceNormal(hit.triangle);
        VtVec3fArray const &P = scalp.bvh.Points();
        e0 = P[size_t(t.v[1])] - P[size_t(t.v[0])];
        e1 = P[size_t(t.v[2])] - P[size_t(t.v[0])];
    }
    // Scatter's frame: N the face normal, T a fan edge Gram-Schmidt'd against
    // N (the other edge when the first is nearly normal), B = N x T.
    float const d = GfDot(e0, n);
    GfVec3f tangent = d * d > 0.81f * GfDot(e0, e0) ? e1 - n * GfDot(e1, n) : e0 - n * d;
    float const tl = tangent.GetLength();
    if (tl < 1e-9f) {
        tangent = std::fabs(n[0]) > 0.9f ? GfVec3f(0.0f, 1.0f, 0.0f) : GfVec3f(1.0f, 0.0f, 0.0f);
        tangent = strand::Normalize(tangent - n * GfDot(tangent, n), GfVec3f(1.0f, 0.0f, 0.0f));
    } else {
        tangent /= tl;
    }
    *T = tangent;
    *N = n;
    *B = strand::Normalize(GfCross(n, tangent), GfVec3f(0.0f, 0.0f, 1.0f));
}

}  // namespace

TfSpan<const TfToken> UsdGenCardSourceOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(kTopo.data(), kTopo.size());
}
TfSpan<const TfToken> UsdGenCardSourceOp::ValueParameters() const
{
    return TfSpan<const TfToken>(kValue.data(), kValue.size());
}
TfSpan<const TfToken> UsdGenCardSourceOp::ReferenceInputs() const
{
    return TfSpan<const TfToken>(kReferenceInputs.data(), kReferenceInputs.size());
}
TfSpan<const TfToken> UsdGenCardSourceOp::OutputPrimvars() const
{
    return TfSpan<const TfToken>(kOutputs.data(), kOutputs.size());
}

bool UsdGenCardSourceOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    Settings s;
    return ReadSettings(&params, &s, diag);
}

UsdGenEpoch UsdGenCardSourceOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    opUtil::Digest d;
    d.Mix(Type());
    if (UsdGenParamView const *p = ctx.params) {
        for (TfToken const &name : kTopo) {
            d.Mix(name);
            d.Mix(p->GetVtValue(name, VtValue()));
        }
    }
    d.Mix(uint64_t(ctx.seed));
    if (ctx.desc && ctx.surface < ctx.desc->surfaces.size())
        d.Mix(ctx.desc->surfaces[ctx.surface].surfaceGeneration);
    for (uint32_t i = 0; i < ctx.referenceCount; ++i) {
        if (ctx.resolvedReferences && ctx.resolvedReferences[i]) {
            d.Mix(ctx.resolvedReferences[i]->identity);
            d.Mix(ctx.resolvedReferences[i]->curveGeneration);
        } else if (ctx.references && ctx.references[i]) {
            d.Mix(ctx.references[i]->generation);
        }
    }
    return d.Epoch(0x43617264ull);  // "Card"
}

std::unique_ptr<UsdGenCapture> UsdGenCardSourceOp::CreateCapture() const
{
    return std::make_unique<UsdGenCardSourceCapture>();
}

bool UsdGenCardSourceOp::Capture(UsdGenCaptureContext const &ctx, UsdGenCurveBuffer const &,
                                 UsdGenCapture *out, UsdGenDiagnostics *diag)
{
    auto fail = [&](std::string const &message) {
        if (diag) diag->Error("UsdGenCardSource: " + message);
        return false;
    };
    if (!out) return fail("no capture");
    Settings s;
    if (!ReadSettings(ctx.params, &s, diag)) return false;
    UsdGenCurveBuffer &buf = static_cast<UsdGenCardSourceCapture *>(out)->MutableBuffer();
    buf = UsdGenCurveBuffer();

    // --- reference sets, told apart by their planes ------------------------------------
    UsdGenCurveBuffer const *spineSet = nullptr, *librarySet = nullptr;
    bool emptySet = false;
    for (uint32_t i = 0; i < ctx.referenceCount; ++i) {
        UsdGenReferenceSet const *ref = nullptr;
        if (ctx.resolvedReferences && ctx.resolvedReferences[i] && ctx.resolvedReferences[i]->value)
            ref = ctx.resolvedReferences[i]->value.get();
        else if (ctx.references)
            ref = ctx.references[i];
        if (!ref) continue;
        UsdGenCurveBuffer const &b = ref->buffer;
        if (IsSpineSet(b)) {
            if (spineSet) return fail("more than one card spine set (curves with a cardType plane)");
            spineSet = &b;
        } else if (IsLibrarySet(b)) {
            if (librarySet) return fail("more than one tile strand library");
            librarySet = &b;
        } else if (b.totalCurves == 0) {
            emptySet = true;   // an empty set carries no planes: no cards or no strands
        }
    }
    if ((!spineSet || !librarySet) && emptySet) {
        if (diag) diag->Warn("UsdGenCardSource: an empty card or strand set grows no strands");
        return true;
    }
    if (!spineSet) return fail("usdGen:cards names no card spine set (a curve set with a cardType plane)");
    if (!librarySet)
        return fail("usdGen:cardStrands names no tile strand library (curves with tile and index planes)");

    std::string error;
    std::vector<card::Card> cards;
    if (!DecodeCards(*spineSet, &cards, &error)) return fail(error);
    Library library;
    if (!DecodeLibrary(*librarySet, &library, &error)) return fail(error);

    // --- the scalp: the whole mesh (card bindings index all its fan triangles) ----------
    Scalp scalp;
    card::Context gctx;
    gctx.cards = &cards;
    gctx.smooth = s.smooth;
    float sceneLength = 1.0f;
    if (ctx.desc && ctx.surface < ctx.desc->surfaces.size()) {
        UsdGenSurfaceDesc whole = ctx.desc->surfaces[ctx.surface];
        whole.isSubset = false;
        whole.subsetFaces = VtIntArray();
        strand::BuildSurfaceBvh(whole, true, &scalp.bvh);
        scalp.valid = !scalp.bvh.Empty();
        VtVec3fArray const &P = scalp.bvh.Points();
        if (scalp.valid && !P.empty()) {
            GfVec3f lo = P[0], hi = P[0];
            for (GfVec3f const &p : P)
                for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], p[k]); hi[k] = std::max(hi[k], p[k]); }
            gctx.bboxCentre = (lo + hi) * 0.5f;
            float const diagonal = (hi - lo).GetLength();
            sceneLength = std::isfinite(diagonal) && diagonal > 1e-4f ? diagonal : 1.0f;
            int maxFace = -1;
            for (strand::FanTriangle const &t : scalp.bvh.Triangles()) maxFace = std::max(maxFace, t.face);
            scalp.faceNormal.assign(size_t(maxFace + 1), GfVec3f(0.0f));
            for (strand::FanTriangle const &t : scalp.bvh.Triangles())
                scalp.faceNormal[size_t(t.face)] +=
                    GfCross(P[size_t(t.v[1])] - P[size_t(t.v[0])], P[size_t(t.v[2])] - P[size_t(t.v[0])]);
            for (GfVec3f &n : scalp.faceNormal) n = strand::Normalize(n, GfVec3f(0.0f, 1.0f, 0.0f), 1e-20f);
        }
    }
    if (!scalp.valid && diag && !cards.empty())
        diag->Warn("UsdGenCardSource: no bound scalp surface; roots are not snapped");
    gctx.scalp = scalp.valid ? &scalp.bvh : nullptr;
    gctx.bindTolerance = 1e-3f * sceneLength;

    // --- cards in emission order: layer ascending, then spine order --------------------
    auto passes = [&](card::Card const &k) {
        return (s.layer == -1 || k.layer == s.layer) && (s.uvId == -1 || k.uvId == s.uvId);
    };
    std::vector<uint32_t> order;
    for (uint32_t c = 0; c < cards.size(); ++c) {
        card::Card const &k = cards[c];
        bool const braid = k.cardType == card::kBraid;
        if (braid != s.braid || !passes(k)) continue;
        if (braid ? (k.uvId < 0 || k.uvId > 9 || k.erased || k.distributionMask < 0.5f ||
                     k.points.size() < 2)
                  : !card::Included(k))
            continue;
        order.push_back(c);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return cards[a].layer < cards[b].layer; });

    std::vector<Root> roots;
    std::vector<card::Span> spans;
    if (!s.braid) {
        std::vector<std::vector<card::Span>> built(order.size());
        ParallelFor(ctx.dispatcher, order.size(),
                    [&](size_t i) { built[i] = card::BuildCard(order[i], gctx); });
        for (auto &list : built)
            for (card::Span &span : list) {
                if (span.uvId < 0 || span.tile < 0 || span.rows < 2 || span.cols < 2) continue;
                if (s.tile != -1 && span.tile != s.tile) continue;
                spans.push_back(std::move(span));
            }
        uint64_t total = 0;
        for (card::Span const &span : spans)
            if (Library::Entry const *e = library.Find(span.uvId, span.tile)) total += e->strandCount;
        double const frac = (total <= uint64_t(s.maxStrands) ? 1.0 : double(s.maxStrands) / double(total)) *
            double(s.amount);
        for (uint32_t si = 0; si < spans.size(); ++si) {
            card::Span const &span = spans[si];
            Library::Entry const *e = library.Find(span.uvId, span.tile);
            if (!e) continue;
            uint64_t const cardId = cards[span.card].id;
            for (uint32_t curve : e->curves) {
                uint64_t const id = card::StrandId(cardId, uint64_t(int64_t(library.index[curve])), span.strip);
                if (frac < 1.0 &&
                    double(strand::SplitMix64(id ^ 0x5a17c4ull) >> 11) * (1.0 / 9007199254740992.0) >= frac)
                    continue;
                roots.push_back({si, curve, span.card, id, GfVec3f(0.0f)});
            }
        }
    } else {
        // Braid cards: R2 low-discrepancy root spread around the spine root
        // in the card's root frame; the strand follows the length-scaled
        // spine (the reference braid weave is not reproduced).
        size_t const nCards = order.size();
        float const W = std::max(0.01f, s.braidWidth) * 0.045f * sceneLength;
        float const f = s.braidFlatness;
        for (uint32_t c : order) {
            card::Card const &k = cards[c];
            if (k.tile[0] < 0 || (s.tile != -1 && k.tile[0] != s.tile)) continue;
            Library::Entry const *e = library.Find(k.uvId, k.tile[0]);
            if (!e) continue;
            long const want = std::lround(float(e->strandCount) * 3.0f * s.amount);
            long const cap = std::max<long>(3, long(double(s.maxStrands) / double(std::max<size_t>(nCards, 1))));
            long const n = std::min(std::max<long>(3, want), cap);
            for (long q = 0; q < n; ++q) {
                double const a = std::fmod(0.7548776662466927 * double(q) + 0.5, 1.0);
                double const b = std::fmod(0.5698402909980532 * double(q) + 0.5, 1.0);
                GfVec3f const offset(float((2.0 * a - 1.0) * 1.5 * W),
                                     float((2.0 * b - 1.0) * ((1.0 - f) + 0.35 * f) * 0.5 * W), 0.0f);
                roots.push_back({0, uint32_t(q), c, card::StrandId(k.id, uint64_t(q), 0), offset});
            }
        }
    }

    // --- strands ---------------------------------------------------------------------------
    size_t const R = roots.size();
    size_t const n = size_t(s.cvCount);
    std::vector<float> points(R * n * 3);
    std::vector<GfVec3f> rootT(R), rootN(R), rootB(R);
    std::vector<int> rootPrim(R, -1);
    std::vector<GfVec2f> rootUV(R, GfVec2f(0.0f));
    float const depthScale = 1.0f - s.rootFlatten;
    UsdGenCurveBuffer const &lb = *library.buffer;

    // Braid strands: the processed spine and its root frame per card.
    std::map<uint32_t, std::pair<std::vector<GfVec3f>, std::vector<GfVec3f>>> braidPaths;
    if (s.braid) {
        for (Root const &r : roots) {
            if (braidPaths.count(r.card)) continue;
            card::Card const &k = cards[r.card];
            std::vector<GfVec3f> path = card::ProcessSpine(k, gctx);
            GfVec3f n0, s0;
            bool found = false;
            card::RootFrame(k, gctx, &n0, &s0, &found);
            float const a = strand::BezierProfile(0.0f, k.twistRoot, k.twistMid, k.twistMidShift,
                                                  k.twistTip) * 0.017453292f *
                (k.MirrorSecondary() ? -1.0f : 1.0f);
            // Transported (axis1, axis2) frame along the path.
            std::vector<GfVec3f> T;
            std::vector<float> flat(3 * path.size());
            for (size_t i = 0; i < path.size(); ++i) strand::Store(flat.data(), i, path[i]);
            strand::PolylineTangents(flat.data(), path.size(), &T);
            GfVec3f axis1 = strand::Rotate(s0, T.empty() ? GfVec3f(0, 1, 0) : T[0], a);
            std::vector<GfVec3f> frame(2 * path.size());
            for (size_t i = 0; i < path.size(); ++i) {
                axis1 = strand::Normalize(axis1 - T[i] * GfDot(axis1, T[i]), axis1);
                frame[2 * i] = axis1;
                frame[2 * i + 1] = strand::Normalize(GfCross(T[i], axis1), GfVec3f(0, 0, 1));
            }
            braidPaths[r.card] = {std::move(path), std::move(frame)};
        }
    }

    ParallelFor(ctx.dispatcher, R, [&](size_t k) {
        Root const &r = roots[k];
        std::vector<float> mapped;
        GfVec3f mappedNormal(0.0f, 1.0f, 0.0f);
        if (!s.braid) {
            card::Span const &span = spans[r.span];
            uint32_t const first = library.spans[r.source], count = library.spans[r.source + 1] - first;
            mapped.resize(3 * size_t(count));
            for (uint32_t i = 0; i < count; ++i) {
                GfVec3f const q = opUtil::RestPoint(lb, first + i);
                GfVec3f p, nrm;
                card::CardSurface(span, q[0] + 0.5f, q[1] + 0.5f, q[2] * depthScale, &p, &nrm);
                strand::Store(mapped.data(), i, p);
                if (i == 0) mappedNormal = nrm;
            }
        } else {
            auto const &bp = braidPaths.at(r.card);
            mapped.resize(3 * bp.first.size());
            for (size_t i = 0; i < bp.first.size(); ++i)
                strand::Store(mapped.data(), i, bp.first[i] + bp.second[2 * i] * r.offset[0] +
                                                    bp.second[2 * i + 1] * r.offset[1]);
            mappedNormal = bp.second.size() > 1 ? bp.second[1] : mappedNormal;
        }
        size_t const m = mapped.size() / 3;
        // Snap the root to the scalp; the strand base follows (falloff shift).
        strand::ClosestHit hit;
        if (scalp.valid && m) {
            hit = scalp.bvh.Closest(strand::Load(mapped.data(), 0));
            if (hit.hit) strand::ResnapRoot(mapped.data(), m, hit.point - strand::Load(mapped.data(), 0));
        }
        RootFrameAt(scalp, hit, mappedNormal, &rootPrim[k], &rootUV[k], &rootT[k], &rootN[k], &rootB[k]);
        std::vector<float> cum;
        float const L = strand::ArcLengths(mapped.data(), m, &cum);
        strand::ResampleArc(mapped.data(), m, cum, 0.0f, L, n, &points[k * n * 3]);
    });

    // --- Morton order over roots, ties in generation order ------------------------------
    std::vector<uint64_t> keys(R);
    for (size_t k = 0; k < R; ++k)
        keys[k] = UsdGenMortonKey3(points[k * n * 3], points[k * n * 3 + 1], points[k * n * 3 + 2], 64.0f);
    std::vector<uint32_t> perm(R);
    std::iota(perm.begin(), perm.end(), 0u);
    std::stable_sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });

    // --- the output buffer ----------------------------------------------------------------
    buf.totalCurves = uint32_t(R);
    buf.totalCvs = uint32_t(R * n);
    buf.topologyVersion = 1;
    buf.px = VtFloatArray(R * n);
    buf.py = VtFloatArray(R * n);
    buf.pz = VtFloatArray(R * n);
    buf.rest = VtVec3fArray(R * n);
    buf.hairT = VtFloatArray(R * n);
    buf.curveId = VtArray<uint64_t>(R);
    buf.rootPrim = VtIntArray(R);
    buf.rootUV = VtVec2fArray(R);
    buf.rootT = VtVec3fArray(R);
    buf.rootN = VtVec3fArray(R);
    buf.rootB = VtVec3fArray(R);
    UsdGenPlane cardIndex, strandIndex, strandSeed;
    for (auto *plane : {&cardIndex, &strandIndex, &strandSeed}) {
        plane->interpolation = TfToken("uniform");
        plane->type = TfToken("int");
        plane->arity = 1;
        plane->i = VtIntArray(R);
    }
    cardIndex.name = sCardIndex;
    strandIndex.name = sStrandIndex;
    strandSeed.name = sStrandSeed;
    for (size_t c = 0; c < R; ++c) {
        uint32_t const k = perm[c];
        for (size_t i = 0; i < n; ++i) {
            float const *p = &points[(k * n + i) * 3];
            size_t const o = c * n + i;
            buf.px[o] = p[0];
            buf.py[o] = p[1];
            buf.pz[o] = p[2];
            buf.rest[o] = GfVec3f(p[0], p[1], p[2]);
            buf.hairT[o] = n > 1 ? float(i) / float(n - 1) : 0.0f;
        }
        buf.curveId[c] = roots[k].id;
        buf.rootPrim[c] = rootPrim[k];
        buf.rootUV[c] = rootUV[k];
        buf.rootT[c] = rootT[k];
        buf.rootN[c] = rootN[k];
        buf.rootB[c] = rootB[k];
        cardIndex.i[c] = int(roots[k].card);
        strandIndex.i[c] = int(k);
        strandSeed.i[c] = int(strand::SplitMix64(roots[k].id) & 0x7fffffffull);
    }
    buf.extraCurve.push_back(std::move(cardIndex));
    buf.extraCurve.push_back(std::move(strandIndex));
    buf.extraCurve.push_back(std::move(strandSeed));
    return true;
}

}  // namespace usdGen
