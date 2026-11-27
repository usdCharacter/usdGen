// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// testUsdGenCurveSet, part 2: UsdGenCurveChunk over a generic chunk view.
// A TU of its own: curveBuffer.h's legacy UsdGenChunkView must not be visible.

#include "usdGenCurves/types/curveSet.h"

#include "usdGen/exec/chunkView.h"
#include "usdGen/registries/engineTokens.h"

#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"

#include <cstdint>
#include <cstdio>
#include <vector>

#include "pxr/pxr.h"
#include "pxr/base/vt/types.h"

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;

namespace {

int failures = 0;
void Expect(bool ok, char const *what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

UsdGenChannelDesc Dyn(char const *name, UsdGenScalar scalar, uint8_t arity, uint8_t level)
{
    UsdGenChannelDesc d;
    d.name = UsdGenToken(name);
    d.scalar = scalar;
    d.arity = arity;
    d.level = level;
    return d;
}

}  // namespace

int RunCurveChunkTests()
{
    UsdGenElementLayout const &layout = UsdGenCurveSetLayout();
    UsdGenChannelDesc const nodeDyn[] = {
        Dyn("clumpId_0", UsdGenScalar::I32, 1, 1),
        Dyn("clumpWeight_0", UsdGenScalar::F32, 1, 2),
        Dyn("mask", UsdGenScalar::I32, 2, 2),
    };
    UsdGenChannelDesc const inDyn[] = {
        Dyn("guideWeight", UsdGenScalar::F32, 3, 1),
        Dyn("clumpWeight_0", UsdGenScalar::F32, 1, 2),
        Dyn("mask", UsdGenScalar::I32, 2, 2),
    };
    auto node = UsdGenChannelTable::Make(layout, UsdGenSpan<const UsdGenChannelDesc>(nodeDyn, 3));
    auto in = UsdGenChannelTable::Make(layout, UsdGenSpan<const UsdGenChannelDesc>(inDyn, 3));
    UsdGenSlot const clumpId = node->Find(UsdGenToken("clumpId_0"));
    UsdGenSlot const weight = node->Find(UsdGenToken("clumpWeight_0"));
    UsdGenSlot const mask = node->Find(UsdGenToken("mask"));
    UsdGenSlot const guideWeight = in->Find(UsdGenToken("guideWeight"));
    Expect(clumpId != kUsdGenNoSlot && weight != kUsdGenNoSlot && mask != kUsdGenNoSlot &&
               guideWeight != kUsdGenNoSlot,
           "chunk tables");

    // Chunk 1 of a 10-curve set, curves 4..7, 3 CVs each (uniform).
    UsdGenChunkRange range;
    range.first[0] = 4; range.count[0] = 4;
    range.first[1] = 12; range.count[1] = 12; range.stride[1] = 3;
    range.live = 3; range.group = 5; range.domain = 2;
    std::vector<float> px(30), py(30), pz(30), width(30), hairT(30), w(30), inPx(30), inW(30), inW2(30);
    std::vector<int> mk(60), ids(10), prim(10);
    std::vector<uint64_t> cid(10);
    std::vector<GfVec2f> uv(10);
    std::vector<GfVec3f> rt(10), rn(10), rb(10);
    std::vector<float> gw(30);
    std::vector<GfVec3d> scratch(30);
    std::vector<double> scalar(30);
    float const lut[257] = {};
    std::vector<void *> out(node->Size(), nullptr);
    std::vector<void const *> inCols(in->Size(), nullptr), in2Cols(in->Size(), nullptr);
    out[kUsdGenCurvePx] = px.data() + 12;
    out[kUsdGenCurvePy] = py.data() + 12;
    out[kUsdGenCurvePz] = pz.data() + 12;
    out[kUsdGenCurveWidth] = width.data() + 12;
    out[kUsdGenCurveHairT] = hairT.data() + 12;
    out[kUsdGenCurveId] = cid.data() + 4;
    out[kUsdGenCurveRootPrim] = prim.data() + 4;
    out[kUsdGenCurveRootUV] = uv.data() + 4;
    out[kUsdGenCurveRootT] = rt.data() + 4;
    out[kUsdGenCurveRootN] = rn.data() + 4;
    out[kUsdGenCurveRootB] = rb.data() + 4;
    out[clumpId] = ids.data() + 4;
    out[weight] = w.data() + 12;
    out[mask] = mk.data() + 24;
    inCols[kUsdGenCurvePx] = inPx.data() + 12;
    inCols[kUsdGenCurveWidth] = inW.data() + 12;
    inCols[guideWeight] = gw.data() + 12;
    in2Cols[kUsdGenCurveWidth] = inW2.data() + 12;
    UsdGenSlot const emitted[] = {clumpId, weight};
    UsdGenSlot const consumed[] = {guideWeight};
    UsdGenSlot const remapped[] = {weight, mask};
    UsdGenChunkExtent extent{};

    UsdGenChunkView v;
    v.range = &range;
    v.inFirst[2] = 12;
    v.inStride[2] = 3;
    v.out = out.data();
    v.in = inCols.data();
    v.in2 = in2Cols.data();
    v.emitted = emitted;
    v.consumed = consumed;
    v.remapped = remapped;
    v.channels = node.Get();
    v.inChannels = in.Get();
    v.emittedCount = 2;
    v.consumedCount = 1;
    v.remappedCount = 2;
    v.pointScratch = scratch.data() + 12;
    v.scalarScratch = scalar.data() + 12;
    v.extentSlot = &extent;
    v.rampLut = lut;

    UsdGenCurveChunk const c(v);
    Expect(c.desc && c.desc->firstCurve == 4 && c.desc->curveCount == 4 && c.desc->liveCount == 3 &&
               c.desc->firstCv == 12 && c.desc->cvCount == 3 && c.desc->tile == 5 && c.desc->surface == 2,
           "desc from the range");
    Expect(c.curveCount == 4 && c.cvCount == 3 && c.inCvCount == 3 && c.inFirstCv == 12 && c.Cv(1, 2) == 5,
           "counts");
    Expect(c.px == px.data() + 12 && c.py == py.data() + 12 && c.pz == pz.data() + 12 &&
               c.width == width.data() + 12 && c.hairT == hairT.data() + 12,
           "per-CV outputs are chunk-local");
    Expect(c.curveId == cid.data() + 4 && c.rootPrim == prim.data() + 4 && c.rootUV == uv.data() + 4 &&
               c.rootT == rt.data() + 4 && c.rootN == rn.data() + 4 && c.rootB == rb.data() + 4,
           "per-curve columns are chunk-local");
    Expect(c.inPx == inPx.data() + 12 && !c.inPy && c.inWidth == inW.data() + 12 && c.inWidth2 == inW2.data() + 12,
           "inputs");
    Expect(!c.cvOffsets && !c.inCvOffsets, "uniform: no offsets");
    Expect(c.outCount == 2 && c.OutI(0) == ids.data() + 4 && !c.outF[0] && c.OutF(1) == w.data() + 12 &&
               !c.outI[1],
           "emitted planes in declaration order, typed");
    Expect(c.inCount == 1 && c.InF(0) == gw.data() + 12 && !c.inI[0], "consumed planes");
    UsdGenEngineTokens const &tk = UsdGenTokens();
    Expect(c.extraCvCount == 2 && c.extraCv[0].name == UsdGenToken("clumpWeight_0") &&
               c.extraCv[0].interpolation == tk.vertex && c.extraCv[0].type == tk.float_ &&
               c.extraCv[1].type == tk.int_ && c.extraCv[1].arity == 2 &&
               c.inExtraCv[1].name == UsdGenToken("mask"),
           "remapped plane descriptors");
    Expect(c.extraCvF[0] == w.data() && !c.extraCvI[0] && c.extraCvI[1] == mk.data() && !c.extraCvF[1],
           "remapped bases are whole-column");
    Expect(c.pointScratch == scratch.data() + 12 && c.scalarScratch == scalar.data() + 12 &&
               c.extentSlot == &extent && c.rampLut == lut,
           "scratch, extent and ramp");

    // Ragged: offsets keep absolute values, pointer at the chunk's first curve.
    uint32_t offs[11] = {0, 2, 5, 6, 9, 13, 14, 16, 20, 21, 25};
    UsdGenChunkRange ragged = range;
    ragged.stride[1] = 0;
    ragged.first[1] = offs[4];
    UsdGenChunkView r = v;
    r.range = &ragged;
    r.offsets[2] = offs + 4;
    r.inOffsets[2] = offs + 4;
    r.emittedCount = r.consumedCount = r.remappedCount = 0;
    UsdGenCurveChunk const rc(r);
    Expect(rc.cvCount == 0 && rc.cvOffsets == reinterpret_cast<int const *>(offs + 4) &&
               rc.CvRagged(1, 2) == 15 && rc.inCvOffsets == rc.cvOffsets,
           "ragged offsets");
    Expect(!rc.outF && !rc.outI && !rc.inF && !rc.inI && !rc.extraCv && rc.extraCvCount == 0,
           "no planes: null tables");

    UsdGenChunkView empty;
    UsdGenCurveChunk const e(empty);
    Expect(e.desc && e.desc->curveCount == 0 && !e.px && !e.inPx && e.outCount == 0, "empty view");
    return failures;
}
