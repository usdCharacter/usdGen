// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenCurveChunk over the generic chunk view, or over the old family's
// view it wraps (bridge, 5.3a-f).
#include "usdGenCurves/types/curveSet.h"
#include "usdGenCurves/types/curveBuffer.h"

#include "usdGen/exec/chunkView.h"
#include "usdGen/registries/engineTokens.h"

#include <algorithm>

namespace usdGen {

namespace {

template <class T>
T *At(void *const *table, UsdGenSlot slot) noexcept
{
    return table && slot != kUsdGenNoSlot ? static_cast<T *>(table[slot]) : nullptr;
}

template <class T>
T const *AtIn(void const *const *table, UsdGenSlot slot) noexcept
{
    return table && slot != kUsdGenNoSlot ? static_cast<T const *>(table[slot]) : nullptr;
}

UsdGenCurveChunk::PlaneInfo Info(UsdGenChannelDesc const &d) noexcept
{
    UsdGenEngineTokens const &tk = UsdGenTokens();
    UsdGenCurveChunk::PlaneInfo p;
    p.name = d.name;
    p.interpolation = d.level == 0 ? tk.constant : d.level == 1 ? tk.uniform : tk.vertex;
    p.type = d.scalar == UsdGenScalar::I32 ? tk.int_ : d.scalar == UsdGenScalar::F32 ? tk.float_ : UsdGenToken();
    p.arity = d.arity;
    return p;
}

UsdGenCurveChunk::PlaneInfo Info(UsdGenPlane const &p) noexcept
{
    UsdGenCurveChunk::PlaneInfo i;
    i.name = p.name;
    i.interpolation = p.interpolation;
    i.type = p.type;
    i.arity = p.arity;
    return i;
}

}  // namespace

void UsdGenCurveChunk::_Wrap(UsdGenLegacyChunkView const &l) noexcept
{
    if (l.desc) {
        UsdGenChunkDesc const &d = *l.desc;
        _desc = {d.firstCurve, d.curveCount, d.liveCount, d.firstCv, d.cvCount,
                 uint16_t(d.tile), uint16_t(d.surface)};
    }
    desc = &_desc;
    px = l.px; py = l.py; pz = l.pz;
    inPx = l.inPx; inPy = l.inPy; inPz = l.inPz; inWidth = l.inWidth; inWidth2 = l.inWidth2;
    width = l.width; hairT = l.hairT;
    curveId = l.curveId; rootPrim = l.rootPrim; cvOffsets = l.cvOffsets;
    rootUV = l.rootUV; rootT = l.rootT; rootN = l.rootN; rootB = l.rootB;
    rampLut = l.rampLut;
    outF = l.outF; outI = l.outI; inF = l.inF; inI = l.inI;
    outCount = l.outCount; inCount = l.inCount;
    extraCvCount = std::min<uint32_t>(l.extraCvCount, kMaxSlots);
    for (uint32_t k = 0; k < extraCvCount; ++k) {
        if (l.extraCv) _extraCv[k] = Info(l.extraCv[k]);
        if (l.inExtraCv) {
            _inExtraCv[k] = Info(l.inExtraCv[k]);
            _inExtraCvF[k] = l.inExtraCv[k].f.cdata();
            _inExtraCvI[k] = l.inExtraCv[k].i.cdata();
        }
    }
    extraCv = l.extraCv ? _extraCv : nullptr;
    inExtraCv = l.inExtraCv ? _inExtraCv : nullptr;
    inExtraCvF = l.inExtraCv ? _inExtraCvF : nullptr;
    inExtraCvI = l.inExtraCv ? _inExtraCvI : nullptr;
    extraCvF = l.extraCvF; extraCvI = l.extraCvI;
    pointScratch = l.pointScratch; scalarScratch = l.scalarScratch;
    curveCount = l.curveCount; cvCount = l.cvCount; inCvCount = l.inCvCount;
    inCvOffsets = l.inCvOffsets; inFirstCv = l.inFirstCv;
}

UsdGenCurveChunk::UsdGenCurveChunk(UsdGenChunkView const &v) noexcept
{
    if (!v.range && v.legacy) {
        _Wrap(*v.legacy);
        extentSlot = v.extentSlot;
        return;
    }
    if (v.range) {
        UsdGenChunkRange const &r = *v.range;
        _desc = {r.first[0], r.count[0], r.live, r.first[1], r.stride[1], r.group, r.domain};
    }
    desc = &_desc;
    curveCount = _desc.curveCount;
    cvCount = _desc.cvCount;

    px = At<float>(v.out, kUsdGenCurvePx);
    py = At<float>(v.out, kUsdGenCurvePy);
    pz = At<float>(v.out, kUsdGenCurvePz);
    width = At<float>(v.out, kUsdGenCurveWidth);
    hairT = At<float>(v.out, kUsdGenCurveHairT);
    curveId = At<uint64_t>(v.out, kUsdGenCurveId);
    rootPrim = At<int>(v.out, kUsdGenCurveRootPrim);
    rootUV = At<GfVec2f>(v.out, kUsdGenCurveRootUV);
    rootT = At<GfVec3f>(v.out, kUsdGenCurveRootT);
    rootN = At<GfVec3f>(v.out, kUsdGenCurveRootN);
    rootB = At<GfVec3f>(v.out, kUsdGenCurveRootB);
    cvOffsets = reinterpret_cast<int const *>(v.offsets[2]);

    inPx = AtIn<float>(v.in, kUsdGenCurvePx);
    inPy = AtIn<float>(v.in, kUsdGenCurvePy);
    inPz = AtIn<float>(v.in, kUsdGenCurvePz);
    inWidth = AtIn<float>(v.in, kUsdGenCurveWidth);
    inWidth2 = AtIn<float>(v.in2, kUsdGenCurveWidth);
    inCvCount = v.inStride[2];
    inFirstCv = v.inFirst[2];
    inCvOffsets = reinterpret_cast<int const *>(v.inOffsets[2]);
    rampLut = v.rampLut;

    // Declared planes in declaration order; each slot is float or int.
    outCount = std::min<uint32_t>(v.emittedCount, kMaxSlots);
    for (uint32_t k = 0; k < outCount && v.channels; ++k) {
        UsdGenSlot const s = v.emitted[k];
        if (s == kUsdGenNoSlot) continue;
        if ((*v.channels)[s].scalar == UsdGenScalar::I32) _outI[k] = At<int>(v.out, s);
        else _outF[k] = At<float>(v.out, s);
    }
    inCount = std::min<uint32_t>(v.consumedCount, kMaxSlots);
    for (uint32_t k = 0; k < inCount && v.inChannels; ++k) {
        UsdGenSlot const s = v.consumed[k];
        if (s == kUsdGenNoSlot) continue;
        if ((*v.inChannels)[s].scalar == UsdGenScalar::I32) _inI[k] = AtIn<int>(v.in, s);
        else _inF[k] = AtIn<float>(v.in, s);
    }
    outF = outCount ? _outF : nullptr;
    outI = outCount ? _outI : nullptr;
    inF = inCount ? _inF : nullptr;
    inI = inCount ? _inI : nullptr;

    // remapsDeepestDynamic: the node's CV planes against the input's, by run
    // position; bases are whole-column (absolute CV index), as before.
    if (v.remappedCount && v.channels && v.inChannels) {
        UsdGenSpan<const UsdGenSlot> const inRun = v.inChannels->DynamicRun(2);
        extraCvCount = std::min<uint32_t>({uint32_t(v.remappedCount), uint32_t(inRun.size()), kMaxSlots});
        size_t const first = _desc.firstCv;
        for (uint32_t k = 0; k < extraCvCount; ++k) {
            UsdGenChannelDesc const &d = (*v.channels)[v.remapped[k]];
            _extraCv[k] = Info(d);
            UsdGenChannelDesc const &inDesc = (*v.inChannels)[inRun[k]];
            _inExtraCv[k] = Info(inDesc);
            size_t const inShift = size_t(v.inFirst[2]) * inDesc.arity;
            if (inDesc.scalar == UsdGenScalar::I32) {
                if (int const *p = AtIn<int>(v.in, inRun[k])) _inExtraCvI[k] = p - inShift;
            } else if (float const *p = AtIn<float>(v.in, inRun[k])) {
                _inExtraCvF[k] = p - inShift;
            }
            size_t const shift = first * d.arity;
            if (d.scalar == UsdGenScalar::I32) {
                if (int *p = At<int>(v.out, v.remapped[k])) _extraCvI[k] = p - shift;
            } else if (float *p = At<float>(v.out, v.remapped[k])) {
                _extraCvF[k] = p - shift;
            }
        }
        extraCv = _extraCv;
        extraCvF = _extraCvF;
        extraCvI = _extraCvI;
        inExtraCv = _inExtraCv;
        inExtraCvF = _inExtraCvF;
        inExtraCvI = _inExtraCvI;
    }
    pointScratch = static_cast<GfVec3d *>(v.pointScratch);
    scalarScratch = v.scalarScratch;
    extentSlot = v.extentSlot;
}

}  // namespace usdGen
