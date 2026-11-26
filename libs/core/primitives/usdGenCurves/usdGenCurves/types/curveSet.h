// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The CurveSet primitive: its element layout (level 1 curves, level 2 CVs),
// partition and group hooks, read views and the builder converters.
// No pxr include and no curveBuffer.h here.
#pragma once
#include "usdGen/api/elementSet.h"
#include "usdGenCurves/export.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/span.h"
#include "usdGen/api/token.h"
#include "usdGen/exec/chunkView.h"
#include <cstddef>
#include <cstdint>

PXR_NAMESPACE_OPEN_SCOPE
class GfVec2f;
class GfVec3f;
class GfVec3d;
PXR_NAMESPACE_CLOSE_SCOPE

namespace usdGen {

using PXR_NS::GfVec2f;
using PXR_NS::GfVec3f;
using PXR_NS::GfVec3d;

struct UsdGenCurveBuffer;     // usdGenCurves/types/curveBuffer.h
struct UsdGenChunkView;       // usdGen/exec/chunkView.h
struct UsdGenChunkExtent;     // usdGen/exec/chunkView.h
class UsdGenRegistrar;

/// CPU-lane ceiling on CVs per generated or resampled strand (Grow segments,
/// GuideInterpolate/CardSource cvCount, Resample cvCount/spacing). The CUDA
/// and Vulkan lanes keep their own, smaller kernel limits and refuse a graph
/// above them at admission. Also the CurveSet type's level-2 limit.
constexpr uint32_t kUsdGenMaxCvsPerCurve = 4096u;

/// Static channels of the CurveSet layout (slot == enumerator).
enum UsdGenCurveSlot : UsdGenSlot {
    kUsdGenCurvePx, kUsdGenCurvePy, kUsdGenCurvePz,           // L2 F32 Position, comp 0/1/2 (planar)
    kUsdGenCurveRest,                                         // L2 F32x3 packed, Rest, PassThrough
    kUsdGenCurveWidth, kUsdGenCurveHairT,                     // L2 F32 Width / Parameter
    kUsdGenCurveId,                                           // L1 U64 Id
    kUsdGenCurveRootPrim,                                     // L1 I32
    kUsdGenCurveRootUV,                                       // L1 F32x2 packed Uv
    kUsdGenCurveRootT, kUsdGenCurveRootN, kUsdGenCurveRootB,  // L1 F32x3 packed Frame
    kUsdGenCurveStaticSlots
};

/// Valid after usdGenRegister_usdGenCurves ran (the op registry's first use).
USDGEN_CURVES_API UsdGenTypeId UsdGenCurveSetType() noexcept;
USDGEN_CURVES_API UsdGenElementLayout const &UsdGenCurveSetLayout() noexcept;
/// Interns the static channel names and registers "UsdGenCurveSet".
USDGEN_CURVES_API UsdGenTypeId UsdGenRegisterCurveSetType(UsdGenRegistrar &);

/// Partition hook: chunks of params.chunkSize curves; CVs from the set's
/// level-2 offsets or uniform stride; group = the chunk's tile.
USDGEN_CURVES_API uint32_t UsdGenCurveSetPartition(uint32_t elements, UsdGenElementSetView const &set,
                                                 UsdGenPartitionParams const &,
                                                 UsdGenSpan<UsdGenChunkRange> out);
/// Group hook: the C2 tile layout (groupTarget clamped to [32, 256]).
USDGEN_CURVES_API uint32_t UsdGenCurveSetGroups(uint32_t chunks, UsdGenPartitionParams const &,
                                              UsdGenSpan<UsdGenChunkGroup> out);

/// Read-side replacement for UsdGenPlane.
struct UsdGenPlaneView {
    UsdGenToken name, interpolation, type;
    uint8_t arity = 1;
    UsdGenSpan<const float> f;
    UsdGenSpan<const int> i;
};

/// Read view with UsdGenCurveBuffer's field names, so Capture bodies compile unchanged.
struct UsdGenCurveSetView {
    /// Inline: usdGenExpr and usdGenSurface read curve sets without linking usdGenCurves.
    static UsdGenCurveSetView Of(UsdGenElementSetView const &) noexcept;
    static UsdGenCurveSetView Of(UsdGenElementSet const &s) noexcept { return Of(s.View()); }
    /// A reference set's curves; empty when unset.
    static UsdGenCurveSetView Of(UsdGenReferenceSet const &r) noexcept
    {
        return r.set ? Of(*r.set) : UsdGenCurveSetView{};
    }
    USDGEN_CURVES_API static UsdGenCurveSetView Of(UsdGenCurveBuffer const &) noexcept;   // tests, builders
    UsdGenSpan<const float> px, py, pz, width, hairT;
    UsdGenSpan<const GfVec3f> rest, rootT, rootN, rootB;
    UsdGenSpan<const uint64_t> curveId;
    UsdGenSpan<const int> rootPrim;
    UsdGenSpan<const GfVec2f> rootUV;
    UsdGenSpan<const int> cvOffsets;       // empty = uniform (stored uint32, same bits)
    uint32_t totalCurves = 0, totalCvs = 0;
    uint64_t topologyVersion = 0, valueVersion = 0;
    USDGEN_CURVES_API uint32_t ExtraCurveCount() const noexcept;        // levels 0-1 run, name order
    USDGEN_CURVES_API UsdGenPlaneView ExtraCurve(uint32_t) const noexcept;
    USDGEN_CURVES_API uint32_t ExtraCvCount() const noexcept;           // level-2 run, name order
    USDGEN_CURVES_API UsdGenPlaneView ExtraCv(uint32_t) const noexcept;
    /// Compile/prepare time (name search). perCv: found among the CV planes.
    USDGEN_CURVES_API bool FindPlane(UsdGenToken const &, UsdGenPlaneView *out, bool *perCv) const noexcept;
private:
    template <class T>
    static UsdGenSpan<const T> _Col(UsdGenElementSetView const &s, UsdGenSlot slot,
                                    uint32_t perItem = 1) noexcept
    {
        if (!s.Has(slot)) return {};
        return {s.Column<T>(slot), size_t(s.Values(slot) / perItem)};
    }
    UsdGenElementSetView _set{};
    UsdGenCurveBuffer const *_buffer = nullptr;   // set by Of(UsdGenCurveBuffer)
};

inline UsdGenCurveSetView UsdGenCurveSetView::Of(UsdGenElementSetView const &s) noexcept
{
    UsdGenCurveSetView v;
    if (!s.channels || s.channels->StaticCount() < kUsdGenCurveStaticSlots || !s.columns || !s.values)
        return v;
    v._set = s;
    v.px = _Col<float>(s, kUsdGenCurvePx);
    v.py = _Col<float>(s, kUsdGenCurvePy);
    v.pz = _Col<float>(s, kUsdGenCurvePz);
    v.rest = _Col<GfVec3f>(s, kUsdGenCurveRest, 3);
    v.width = _Col<float>(s, kUsdGenCurveWidth);
    v.hairT = _Col<float>(s, kUsdGenCurveHairT);
    v.curveId = _Col<uint64_t>(s, kUsdGenCurveId);
    v.rootPrim = _Col<int>(s, kUsdGenCurveRootPrim);
    v.rootUV = _Col<GfVec2f>(s, kUsdGenCurveRootUV, 2);
    v.rootT = _Col<GfVec3f>(s, kUsdGenCurveRootT, 3);
    v.rootN = _Col<GfVec3f>(s, kUsdGenCurveRootN, 3);
    v.rootB = _Col<GfVec3f>(s, kUsdGenCurveRootB, 3);
    if (s.offsets[2])
        v.cvOffsets = {reinterpret_cast<int const *>(s.offsets[2]), size_t(s.count[1]) + 1};
    v.totalCurves = s.count[1];
    v.totalCvs = s.count[2];
    v.topologyVersion = s.topologyVersion;
    v.valueVersion = s.valueVersion;
    return v;
}

/// Evaluate's view: the legacy UsdGenChunkView member for member, built on
/// the worker's stack from the generic view. No allocation, no name lookup.
/// cvOffsets/inCvOffsets keep the legacy meaning (absolute values, pointer at
/// the chunk's first curve); extraCvF/I are whole-column bases as before.
class UsdGenCurveChunk {
public:
    USDGEN_CURVES_API explicit UsdGenCurveChunk(UsdGenChunkView const &) noexcept;
    UsdGenCurveChunk(UsdGenCurveChunk const &) = delete;
    UsdGenCurveChunk &operator=(UsdGenCurveChunk const &) = delete;
    static constexpr uint32_t kMaxSlots = 16;   // kUsdGenMaxExtraPlaneSlots (graph.h)
    struct Desc { uint32_t firstCurve, curveCount, liveCount, firstCv, cvCount; uint16_t tile, surface; };
    struct PlaneInfo { UsdGenToken name, interpolation, type; uint8_t arity = 1; };
    Desc const *desc = nullptr;
    float *px = nullptr, *py = nullptr, *pz = nullptr;
    float const *inPx = nullptr, *inPy = nullptr, *inPz = nullptr, *inWidth = nullptr, *inWidth2 = nullptr;
    float *width = nullptr, *hairT = nullptr;
    uint64_t const *curveId = nullptr;
    int const *rootPrim = nullptr, *cvOffsets = nullptr;
    GfVec2f const *rootUV = nullptr;
    GfVec3f const *rootT = nullptr, *rootN = nullptr, *rootB = nullptr;
    float const *rampLut = nullptr;
    float **outF = nullptr; int **outI = nullptr; float const **inF = nullptr; int const **inI = nullptr;
    uint32_t outCount = 0, inCount = 0;
    PlaneInfo const *extraCv = nullptr; float *const *extraCvF = nullptr; int *const *extraCvI = nullptr;
    PlaneInfo const *inExtraCv = nullptr; uint32_t extraCvCount = 0;
    float const *const *inExtraCvF = nullptr; int const *const *inExtraCvI = nullptr;   // whole-column bases, like extraCvF/I
    GfVec3d *pointScratch = nullptr; double *scalarScratch = nullptr; UsdGenChunkExtent *extentSlot = nullptr;
    uint32_t curveCount = 0, cvCount = 0, inCvCount = 0;
    int const *inCvOffsets = nullptr; uint32_t inFirstCv = 0;
    size_t Cv(uint32_t c, uint32_t i) const { return size_t(c) * cvCount + i; }
    size_t CvRagged(uint32_t c, uint32_t i) const { return size_t(cvOffsets[c]) + i; }
    float *OutF(uint32_t s) const { return outF[s]; }
    int *OutI(uint32_t s) const { return outI[s]; }
    float const *InF(uint32_t s) const { return inF[s]; }
    int const *InI(uint32_t s) const { return inI[s]; }
private:
    Desc _desc{};
    float *_outF[kMaxSlots] = {}; int *_outI[kMaxSlots] = {};
    float const *_inF[kMaxSlots] = {}; int const *_inI[kMaxSlots] = {};
    PlaneInfo _extraCv[kMaxSlots], _inExtraCv[kMaxSlots];
    float *_extraCvF[kMaxSlots] = {}; int *_extraCvI[kMaxSlots] = {};
    float const *_inExtraCvF[kMaxSlots] = {}; int const *_inExtraCvI[kMaxSlots] = {};
};

/// Builder <-> set. FromBuffer copies into one arena (generators: once per
/// capture); null when the buffer is inconsistent (a column or plane whose
/// size disagrees with its counts, planes not "float"/"int" or in the wrong
/// rate, ragged offsets not totalCurves+1). Chunks are not carried: the
/// graph owns the partition. exactVersions stamps the versions as given
/// (zero included). BufferFromSet copies (tests and tools).
USDGEN_CURVES_API UsdGenRef<const UsdGenElementSet>
UsdGenCurveSetFromBuffer(UsdGenCurveBuffer const &, uint64_t topologyVersion, uint64_t valueVersion,
                         bool exactVersions = false);
USDGEN_CURVES_API UsdGenCurveBuffer UsdGenCurveBufferFromSet(UsdGenElementSet const &);

}  // namespace usdGen
