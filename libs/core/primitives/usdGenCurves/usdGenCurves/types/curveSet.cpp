// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenCurves/types/curveSet.h"
#include "usdGenCurves/types/curveBuffer.h"
#include "usdGenCurves/types/curveCapture.h"

#include "usdGen/api/writeSet.h"
#include "usdGen/api/engineTokens.h"
#include "usdGen/api/ext/opRegistry.h"
#include "usdGen/api/ext/registrar.h"
#include "usdGen/api/ext/tokenScope.h"
#include "usdGen/api/cowArray.h"
#include "usdGen/api/token.h"
#include "usdGen/api/span.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <utility>

namespace usdGen {

namespace {

/// One per process, published once by the registration that registered the type.
struct CurveSetState {
    UsdGenChannelDesc statics[kUsdGenCurveStaticSlots];
    UsdGenElementLayout layout;
    UsdGenTypeId type = UsdGenTypeId::Invalid;
};

std::atomic<CurveSetState *> gCurveSet{nullptr};
UsdGenElementLayout const kNoLayout{};

CurveSetState const *State() noexcept
{
    if (CurveSetState const *s = gCurveSet.load(std::memory_order_acquire)) return s;
    (void)UsdGenOpRegistry::Get();   // its constructor registers the built-ins
    return gCurveSet.load(std::memory_order_acquire);
}

UsdGenChannelDesc Static(char const *name, UsdGenScalar scalar, uint8_t arity, uint8_t level,
                         UsdGenChannelRole role, uint8_t component = 0, uint8_t flags = 0)
{
    UsdGenChannelDesc d;
    d.name = UsdGenInternToken(name);
    d.scalar = scalar;
    d.arity = arity;
    d.level = level;
    d.role = role;
    d.component = component;
    d.flags = flags;
    return d;
}

// C2 tile arithmetic (ADR R21), formerly types.h ComputeChunksPerTile/ComputeNumTiles
// with Repartition's tileTarget clamp.
constexpr uint32_t kTileMin = 32, kTileMax = 256;

struct TileLayout {
    uint32_t chunks = 0, perTile = 0, tiles = 0;
    /// First chunk of tile t: one chunk reserved for each remaining tile
    /// before the packed bound, so no tile is empty.
    uint32_t Boundary(uint32_t t) const noexcept
    {
        int64_t const packed = int64_t(t) * int64_t(perTile);
        int64_t const tail = int64_t(chunks) - int64_t(tiles - t);
        return uint32_t(std::min(packed, tail));
    }
    uint16_t TileOf(uint32_t chunk) const noexcept
    {
        uint32_t lo = 0, hi = tiles;   // last t with Boundary(t) <= chunk
        while (hi - lo > 1) {
            uint32_t const mid = lo + (hi - lo) / 2;
            if (Boundary(mid) <= chunk) lo = mid;
            else hi = mid;
        }
        return uint16_t(lo);
    }
};

TileLayout Tiles(uint32_t chunks, uint32_t groupTarget) noexcept
{
    TileLayout l;
    l.chunks = chunks;
    if (!chunks) return l;
    uint32_t const target = std::clamp(groupTarget, kTileMin, kTileMax);
    l.perTile = std::max(1u, (chunks + target - 1) / target);
    uint32_t n = (chunks + l.perTile - 1) / l.perTile;
    n = std::max(kTileMin, std::min(kTileMax, n));
    l.tiles = std::min(chunks, n);
    return l;
}

bool StartsWith(char const *s, char const *prefix) noexcept
{
    return std::strncmp(s, prefix, std::strlen(prefix)) == 0;
}

/// First-producer layouts of the named channels curve ops emit without
/// declaring them (the layout's namedChannel hook). Text compares only.
bool CurveNamedChannel(UsdGenToken const &name, UsdGenChannelDesc *d, bool *fixedFamily) noexcept
{
    char const *const s = name.GetText();
    auto is = [s](char const *text) { return std::strcmp(s, text) == 0; };
    bool const clumpId = StartsWith(s, "clumpId_"), centre = StartsWith(s, "clumpCenter_"),
               centreId = StartsWith(s, "clumpCenterId_"), weight = StartsWith(s, "clumpWeight_");
    *fixedFamily = clumpId || centre || centreId || weight;
    *d = UsdGenChannelDesc{};
    d->name = name;
    d->level = 1;
    if (is("guideIndex") || is("partId") || clumpId || centreId) d->scalar = UsdGenScalar::I32;
    if (centre) d->arity = 3;
    if (centreId) d->arity = 2;
    // Whole-strand identities are ints; the braid knot mask is per CV.
    if (is("primaryGuide") || is("nearestGuide") || is("strandSeed") || is("strandIndex") ||
        is("mirrorSign"))
        d->scalar = UsdGenScalar::I32;
    if (weight || is("braidKnotMask")) d->level = 2;
    if (is("guideIndex") || is("guideWeight")) d->arity = 3;
    d->flags = d->arity > 1 ? uint8_t(kUsdGenChannelPacked) : uint8_t(0);
    return true;
}

template <class T>
UsdGenSpan<const T> Col(UsdGenElementSetView const &s, UsdGenSlot slot, uint32_t perItem = 1) noexcept
{
    if (!s.Has(slot)) return {};
    return {s.Column<T>(slot), size_t(s.Values(slot) / perItem)};
}

template <class T>
UsdGenSpan<const T> Arr(UsdGenCowArray<T> const &a) noexcept
{
    return {a.cdata(), a.size()};
}

UsdGenPlaneView PlaneOf(UsdGenElementSetView const &s, UsdGenSlot slot) noexcept
{
    UsdGenEngineTokens const &tk = UsdGenTokens();
    UsdGenChannelDesc const &d = (*s.channels)[slot];
    UsdGenPlaneView p;
    p.name = d.name;
    p.interpolation = d.level == 0 ? tk.constant : d.level == 1 ? tk.uniform : tk.vertex;
    p.arity = d.arity;
    if (d.scalar == UsdGenScalar::F32) {
        p.type = tk.float_;
        p.f = Col<float>(s, slot);
    } else if (d.scalar == UsdGenScalar::I32) {
        p.type = tk.int_;
        p.i = Col<int>(s, slot);
    }
    return p;
}

UsdGenPlaneView PlaneOf(UsdGenPlane const &plane) noexcept
{
    UsdGenPlaneView p;
    p.name = plane.name;
    p.interpolation = plane.interpolation;
    p.type = plane.type;
    p.arity = plane.arity;
    p.f = Arr(plane.f);
    p.i = Arr(plane.i);
    return p;
}

template <class T>
bool Fits(UsdGenCowArray<T> const &a, size_t n) noexcept { return a.empty() || a.size() == n; }

template <class T>
void Put(UsdGenWriteSet &w, UsdGenSlot slot, T const *data, size_t count)
{
    UsdGenSpan<T> out = w.Write<T>(slot);
    if (count && out.size() == count) std::memcpy(out.data(), data, count * sizeof(T));
}

template <class T>
void Take(UsdGenCowArray<T> *dst, UsdGenSpan<const T> src)
{
    if (src.empty()) return;
    dst->resize(src.size());
    std::memcpy(dst->data(), src.data(), src.size() * sizeof(T));
}

}  // namespace

UsdGenTypeId UsdGenRegisterCurveSetType(UsdGenRegistrar &r, UsdGenPublishDesc const *publish)
{
    if (CurveSetState const *s = gCurveSet.load(std::memory_order_acquire)) return s->type;
    CurveSetState *fresh = new CurveSetState;   // startup; never freed (the registry points at it)
    {
        UsdGenCompileTokenScope scope;
        using S = UsdGenScalar;
        using R = UsdGenChannelRole;
        UsdGenChannelDesc *d = fresh->statics;
        d[kUsdGenCurvePx] = Static("__px", S::F32, 1, 2, R::Position, 0);
        d[kUsdGenCurvePy] = Static("__py", S::F32, 1, 2, R::Position, 1);
        d[kUsdGenCurvePz] = Static("__pz", S::F32, 1, 2, R::Position, 2);
        d[kUsdGenCurveRest] = Static("__rest", S::F32, 3, 2, R::Rest, 0,
                                     kUsdGenChannelPacked | kUsdGenChannelPassThrough);
        d[kUsdGenCurveWidth] = Static("__width", S::F32, 1, 2, R::Width);
        d[kUsdGenCurveHairT] = Static("__hairT", S::F32, 1, 2, R::Parameter);
        d[kUsdGenCurveId] = Static("__curveId", S::U64, 1, 1, R::Id);
        d[kUsdGenCurveRootPrim] = Static("__rootPrim", S::I32, 1, 1, R::Other);
        d[kUsdGenCurveRootUV] = Static("__rootUV", S::F32, 2, 1, R::Uv, 0, kUsdGenChannelPacked);
        d[kUsdGenCurveRootT] = Static("__rootT", S::F32, 3, 1, R::Frame, 0, kUsdGenChannelPacked);
        d[kUsdGenCurveRootN] = Static("__rootN", S::F32, 3, 1, R::Frame, 1, kUsdGenChannelPacked);
        d[kUsdGenCurveRootB] = Static("__rootB", S::F32, 3, 1, R::Frame, 2, kUsdGenChannelPacked);
    }
    fresh->layout.levels = 2;
    fresh->layout.channels = UsdGenSpan<const UsdGenChannelDesc>(fresh->statics, kUsdGenCurveStaticSlots);
    fresh->layout.partition = &UsdGenCurveSetPartition;
    fresh->layout.groups = &UsdGenCurveSetGroups;
    fresh->layout.namedChannel = &CurveNamedChannel;
    UsdGenTypeDef def;
    def.name = "UsdGenCurveSet";
    def.traits = kUsdGenTypeDefaultPrimitive;
    def.layout = &fresh->layout;
    def.limits[1] = kUsdGenMaxCvsPerCurve;
    def.publish = publish;
    UsdGenTypeId const id = r.RegisterType(def);
    if (id == UsdGenTypeId::Invalid) {
        delete fresh;
        CurveSetState const *s = gCurveSet.load(std::memory_order_acquire);
        return s ? s->type : UsdGenTypeId::Invalid;
    }
    fresh->type = id;
    fresh->layout.primitive = id;
    CurveSetState *expected = nullptr;
    gCurveSet.compare_exchange_strong(expected, fresh, std::memory_order_acq_rel,
                                      std::memory_order_acquire);
    return id;
}

UsdGenTypeId UsdGenCurveSetType() noexcept
{
    CurveSetState const *s = State();
    return s ? s->type : UsdGenTypeId::Invalid;
}

UsdGenElementLayout const &UsdGenCurveSetLayout() noexcept
{
    CurveSetState const *s = State();
    return s ? s->layout : kNoLayout;
}

uint32_t UsdGenCurveSetPartition(uint32_t elements, UsdGenElementSetView const &set,
                                 UsdGenPartitionParams const &params, UsdGenSpan<UsdGenChunkRange> out)
{
    uint32_t const size = std::max(1u, params.chunkSize);
    uint32_t const chunks = elements ? uint32_t((uint64_t(elements) + size - 1) / size) : 0u;
    TileLayout const tiles = Tiles(chunks, params.groupTarget);
    // CVs per curve from the set when it is uniform and exact; ragged carries
    // its offsets verbatim (stride 0, first CV from the offsets, clamped).
    uint32_t const *offs = set.offsets[2];
    size_t const offsCount = offs ? size_t(set.count[1]) + 1 : 0;
    uint32_t const cvp = offs || !set.count[1] || set.count[2] % set.count[1]
        ? 0u : set.count[2] / set.count[1];
    size_t const n = std::min<size_t>(chunks, out.size());
    for (size_t c = 0; c < n; ++c) {
        UsdGenChunkRange r;
        r.first[0] = uint32_t(c) * size;
        r.count[0] = std::min(size, elements - r.first[0]);
        if (!offs) {
            r.first[1] = r.first[0] * cvp;
            r.count[1] = r.count[0] * cvp;
            r.stride[1] = cvp;
        } else {
            uint32_t const begin = offs[std::min<size_t>(r.first[0], offsCount - 1)];
            uint32_t const end = offs[std::min<size_t>(size_t(r.first[0]) + r.count[0], offsCount - 1)];
            r.first[1] = begin;
            r.count[1] = end >= begin ? end - begin : 0u;
        }
        r.live = r.count[0];
        r.group = tiles.TileOf(uint32_t(c));
        out[c] = r;
    }
    return chunks;
}

uint32_t UsdGenCurveSetGroups(uint32_t chunks, UsdGenPartitionParams const &params,
                              UsdGenSpan<UsdGenChunkGroup> out)
{
    TileLayout const tiles = Tiles(chunks, params.groupTarget);
    size_t const n = std::min<size_t>(tiles.tiles, out.size());
    for (size_t t = 0; t < n; ++t) {
        UsdGenChunkGroup g;
        g.id = uint16_t(t);
        g.firstChunk = tiles.Boundary(uint32_t(t));
        g.chunkCount = tiles.Boundary(uint32_t(t) + 1) - g.firstChunk;
        out[t] = g;
    }
    return tiles.tiles;
}

// ---- views -----------------------------------------------------------------------

UsdGenCurveSetView UsdGenCurveSetView::Of(UsdGenCurveBuffer const &b) noexcept
{
    UsdGenCurveSetView v;
    v._buffer = &b;
    v.px = Arr(b.px);
    v.py = Arr(b.py);
    v.pz = Arr(b.pz);
    v.rest = Arr(b.rest);
    v.width = Arr(b.width);
    v.hairT = Arr(b.hairT);
    v.curveId = Arr(b.curveId);
    v.rootPrim = Arr(b.rootPrim);
    v.rootUV = Arr(b.rootUV);
    v.rootT = Arr(b.rootT);
    v.rootN = Arr(b.rootN);
    v.rootB = Arr(b.rootB);
    v.cvOffsets = Arr(b.cvOffsets);
    v.totalCurves = b.totalCurves;
    v.totalCvs = b.totalCvs;
    v.topologyVersion = b.topologyVersion;
    v.valueVersion = b.valueVersion;
    return v;
}

uint32_t UsdGenCurveSetView::ExtraCurveCount() const noexcept
{
    if (_buffer) return uint32_t(_buffer->extraCurve.size());
    return _set.channels ? uint32_t(_set.channels->DynamicRun(1).size()) : 0u;
}

UsdGenPlaneView UsdGenCurveSetView::ExtraCurve(uint32_t k) const noexcept
{
    if (_buffer) return PlaneOf(_buffer->extraCurve[k]);
    return PlaneOf(_set, _set.channels->DynamicRun(1)[k]);
}

uint32_t UsdGenCurveSetView::ExtraCvCount() const noexcept
{
    if (_buffer) return uint32_t(_buffer->extraCv.size());
    return _set.channels ? uint32_t(_set.channels->DynamicRun(2).size()) : 0u;
}

UsdGenPlaneView UsdGenCurveSetView::ExtraCv(uint32_t k) const noexcept
{
    if (_buffer) return PlaneOf(_buffer->extraCv[k]);
    return PlaneOf(_set, _set.channels->DynamicRun(2)[k]);
}

bool UsdGenCurveSetView::FindPlane(UsdGenToken const &name, UsdGenPlaneView *out, bool *perCv) const noexcept
{
    if (_buffer) {
        for (int cv = 1; cv >= 0; --cv)
            for (UsdGenPlane const &p : cv ? _buffer->extraCv : _buffer->extraCurve)
                if (p.name == name) {
                    if (out) *out = PlaneOf(p);
                    if (perCv) *perCv = cv != 0;
                    return true;
                }
        return false;
    }
    if (!_set.channels) return false;
    UsdGenSlot const slot = _set.channels->Find(name);
    if (slot == kUsdGenNoSlot || slot < _set.channels->StaticCount()) return false;
    if (out) *out = PlaneOf(_set, slot);
    if (perCv) *perCv = (*_set.channels)[slot].level >= 2;
    return true;
}

// ---- converters ------------------------------------------------------------------

namespace {

UsdGenRef<const UsdGenElementSet>
SetFromBuffer(UsdGenCurveBuffer const &b, uint64_t topologyVersion, uint64_t valueVersion, bool exact)
{
    UsdGenElementLayout const &layout = UsdGenCurveSetLayout();
    if (!layout.channels.size()) return {};
    size_t const curves = b.totalCurves, cvs = b.totalCvs;
    if (!Fits(b.px, cvs) || !Fits(b.py, cvs) || !Fits(b.pz, cvs) || !Fits(b.rest, cvs) ||
        !Fits(b.width, cvs) || !Fits(b.hairT, cvs) || !Fits(b.curveId, curves) ||
        !Fits(b.rootPrim, curves) || !Fits(b.rootUV, curves) || !Fits(b.rootT, curves) ||
        !Fits(b.rootN, curves) || !Fits(b.rootB, curves) || !Fits(b.cvOffsets, curves + 1))
        return {};

    UsdGenEngineTokens const &tk = UsdGenTokens();
    UsdGenPoolVector<UsdGenChannelDesc> dynamic;
    dynamic.reserve(b.extraCurve.size() + b.extraCv.size());
    auto plane = [&](UsdGenPlane const &p, bool cv) {
        UsdGenChannelDesc d;
        d.name = p.name;
        d.arity = p.arity;
        d.level = cv ? 2 : p.interpolation == tk.constant ? 0 : 1;
        if (cv ? p.interpolation != tk.vertex : d.level == 1 && p.interpolation != tk.uniform)
            return false;
        if (p.name.IsEmpty() || !p.arity) return false;
        size_t const values = (d.level == 0 ? 1 : d.level == 1 ? curves : cvs) * p.arity;
        if (p.type == tk.float_) {
            d.scalar = UsdGenScalar::F32;
            if (!p.i.empty() || p.f.size() != values) return false;
        } else if (p.type == tk.int_) {
            d.scalar = UsdGenScalar::I32;
            if (!p.f.empty() || p.i.size() != values) return false;
        } else {
            return false;
        }
        d.flags = p.arity > 1 ? uint8_t(kUsdGenChannelPacked) : uint8_t(0);
        dynamic.push_back(d);
        return true;
    };
    for (UsdGenPlane const &p : b.extraCurve)
        if (!plane(p, false)) return {};
    for (UsdGenPlane const &p : b.extraCv)
        if (!plane(p, true)) return {};

    UsdGenRef<const UsdGenChannelTable> table =
        UsdGenChannelTable::Make(layout, UsdGenSpan<const UsdGenChannelDesc>(dynamic.data(), dynamic.size()));
    if (table->Size() != kUsdGenCurveStaticSlots + dynamic.size()) return {};   // a repeated name

    uint32_t const counts[2] = {b.totalCurves, b.totalCvs};
    UsdGenWriteSet w = UsdGenWriteSet::Fresh(table, UsdGenSpan<const uint32_t>(counts, 2));
    UsdGenPoolVector<UsdGenSlot> written;
    written.reserve(table->Size());
    bool const present[kUsdGenCurveStaticSlots] = {
        !b.px.empty(), !b.py.empty(), !b.pz.empty(), !b.rest.empty(), !b.width.empty(),
        !b.hairT.empty(), !b.curveId.empty(), !b.rootPrim.empty(), !b.rootUV.empty(),
        !b.rootT.empty(), !b.rootN.empty(), !b.rootB.empty()};
    for (UsdGenSlot s = 0; s < kUsdGenCurveStaticSlots; ++s)
        if (present[s]) written.push_back(s);
    for (UsdGenSlot s = kUsdGenCurveStaticSlots; s < table->Size(); ++s) written.push_back(s);
    w.Plan(UsdGenSpan<const UsdGenSlot>(written.data(), written.size()));

    if (!b.cvOffsets.empty()) {
        UsdGenSpan<uint32_t> offs = w.WriteOffsets(2);
        if (offs.size() == b.cvOffsets.size())
            std::memcpy(offs.data(), b.cvOffsets.cdata(), offs.size() * sizeof(uint32_t));
    }
    Put(w, kUsdGenCurvePx, b.px.cdata(), b.px.size());
    Put(w, kUsdGenCurvePy, b.py.cdata(), b.py.size());
    Put(w, kUsdGenCurvePz, b.pz.cdata(), b.pz.size());
    Put(w, kUsdGenCurveRest, reinterpret_cast<float const *>(b.rest.cdata()), b.rest.size() * 3);
    Put(w, kUsdGenCurveWidth, b.width.cdata(), b.width.size());
    Put(w, kUsdGenCurveHairT, b.hairT.cdata(), b.hairT.size());
    Put(w, kUsdGenCurveId, b.curveId.cdata(), b.curveId.size());
    Put(w, kUsdGenCurveRootPrim, b.rootPrim.cdata(), b.rootPrim.size());
    Put(w, kUsdGenCurveRootUV, reinterpret_cast<float const *>(b.rootUV.cdata()), b.rootUV.size() * 2);
    Put(w, kUsdGenCurveRootT, reinterpret_cast<float const *>(b.rootT.cdata()), b.rootT.size() * 3);
    Put(w, kUsdGenCurveRootN, reinterpret_cast<float const *>(b.rootN.cdata()), b.rootN.size() * 3);
    Put(w, kUsdGenCurveRootB, reinterpret_cast<float const *>(b.rootB.cdata()), b.rootB.size() * 3);
    auto putPlane = [&](UsdGenPlane const &p) {
        UsdGenSlot const s = table->Find(p.name);
        if (p.type == tk.int_) Put(w, s, p.i.cdata(), p.i.size());
        else Put(w, s, p.f.cdata(), p.f.size());
    };
    for (UsdGenPlane const &p : b.extraCurve) putPlane(p);
    for (UsdGenPlane const &p : b.extraCv) putPlane(p);
    if (exact) w.SetVersionsExact(topologyVersion, valueVersion);
    else w.SetVersions(topologyVersion, valueVersion);
    return std::move(w).Seal();
}

}  // namespace

UsdGenRef<const UsdGenElementSet>
UsdGenCurveSetFromBuffer(UsdGenCurveBuffer const &b, uint64_t topologyVersion, uint64_t valueVersion,
                         bool exactVersions)
{
    return SetFromBuffer(b, topologyVersion, valueVersion, exactVersions);
}

UsdGenRef<const UsdGenElementSet> UsdGenCurveCapture::Output() const
{
    if (!OwnsBuffer()) return {};
    return SetFromBuffer(_buffer, _buffer.topologyVersion, _buffer.valueVersion, true);
}

UsdGenCurveBuffer UsdGenCurvePlanesBuffer(UsdGenCurveSetView const &v)
{
    UsdGenCurveBuffer b;
    Take(&b.cvOffsets, v.cvOffsets);
    auto plane = [](UsdGenPlaneView const &p) {
        UsdGenPlane out;
        out.name = p.name;
        out.interpolation = p.interpolation;
        out.type = p.type;
        out.arity = p.arity;
        Take(&out.f, p.f);
        Take(&out.i, p.i);
        return out;
    };
    b.extraCurve.reserve(v.ExtraCurveCount());
    for (uint32_t k = 0; k < v.ExtraCurveCount(); ++k) b.extraCurve.push_back(plane(v.ExtraCurve(k)));
    b.extraCv.reserve(v.ExtraCvCount());
    for (uint32_t k = 0; k < v.ExtraCvCount(); ++k) b.extraCv.push_back(plane(v.ExtraCv(k)));
    b.totalCurves = v.totalCurves;
    b.totalCvs = v.totalCvs;
    b.topologyVersion = v.topologyVersion;
    b.valueVersion = v.valueVersion;
    return b;
}

bool UsdGenResampleExtraPlanes(UsdGenCurveSetView const &source, UsdGenCurveBuffer *target,
                               std::string *error)
{
    return UsdGenResampleExtraPlanes(UsdGenCurvePlanesBuffer(source), target, error);
}

bool UsdGenCompactExtraPlanes(UsdGenCurveSetView const &source, std::vector<uint32_t> const &survivors,
                              UsdGenCurveBuffer *target, std::string *error)
{
    return UsdGenCompactExtraPlanes(UsdGenCurvePlanesBuffer(source), survivors, target, error);
}

UsdGenCurveBuffer UsdGenCurveBufferFromSet(UsdGenElementSet const &set)
{
    UsdGenCurveBuffer b;
    UsdGenCurveSetView const v = UsdGenCurveSetView::Of(set);
    Take(&b.px, v.px);
    Take(&b.py, v.py);
    Take(&b.pz, v.pz);
    Take(&b.rest, v.rest);
    Take(&b.width, v.width);
    Take(&b.hairT, v.hairT);
    Take(&b.curveId, v.curveId);
    Take(&b.rootPrim, v.rootPrim);
    Take(&b.rootUV, v.rootUV);
    Take(&b.rootT, v.rootT);
    Take(&b.rootN, v.rootN);
    Take(&b.rootB, v.rootB);
    Take(&b.cvOffsets, v.cvOffsets);
    auto plane = [](UsdGenPlaneView const &p) {
        UsdGenPlane out;
        out.name = p.name;
        out.interpolation = p.interpolation;
        out.type = p.type;
        out.arity = p.arity;
        Take(&out.f, p.f);
        Take(&out.i, p.i);
        return out;
    };
    b.extraCurve.reserve(v.ExtraCurveCount());
    for (uint32_t k = 0; k < v.ExtraCurveCount(); ++k) b.extraCurve.push_back(plane(v.ExtraCurve(k)));
    b.extraCv.reserve(v.ExtraCvCount());
    for (uint32_t k = 0; k < v.ExtraCvCount(); ++k) b.extraCv.push_back(plane(v.ExtraCv(k)));
    b.totalCurves = set.Count(1);
    b.totalCvs = set.Count(2);
    b.topologyVersion = set.TopologyVersion();
    b.valueVersion = set.ValueVersion();
    return b;
}

}  // namespace usdGen
