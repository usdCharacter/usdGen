// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Test helpers: drive an op's Capture/Evaluate by concrete type over builder
// buffers and raw chunk-local planes, the way the scheduler does.
#pragma once
#include "usdGen/api/ext/chunkView.h"
#include "usdGen/api/ext/op.h"
#include "curveBufferSetView.h"
#include "usdGenCurves/types/curveBuffer.h"
#include "usdGenCurves/types/curveSet.h"

#include <cstdint>

#include "pxr/pxr.h"
#include "pxr/base/vt/types.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// A UsdGenOpInputs over one builder buffer (the buffer must outlive it).
class UsdGenBufferInputs
{
public:
    explicit UsdGenBufferInputs(UsdGenCurveBuffer const &b)
    {
        _set.Bind(b);
        _in.sets = &_set.View();
        _in.count = 1;
    }
    UsdGenBufferInputs(UsdGenBufferInputs const &) = delete;
    UsdGenBufferInputs &operator=(UsdGenBufferInputs const &) = delete;
    operator UsdGenOpInputs const &() const { return _in; }

private:
    UsdGenCurveBufferSetView _set;
    UsdGenOpInputs _in;
};

/// One chunk over raw chunk-local CurveSet planes (static channels only).
/// Fill the fields, then pass View(); the view points into this object.
/// The range takes first/live/tile/surface from desc (zero, live =
/// curveCount without one) and its counts from curveCount/cvCount.
struct UsdGenTestChunk
{
    UsdGenChunkDesc const *desc = nullptr;
    float *px = nullptr, *py = nullptr, *pz = nullptr, *width = nullptr, *hairT = nullptr;
    float const *inPx = nullptr, *inPy = nullptr, *inPz = nullptr;
    float const *inWidth = nullptr, *inWidth2 = nullptr;
    uint64_t const *curveId = nullptr;
    int const *rootPrim = nullptr;
    int const *cvOffsets = nullptr;     // absolute, at the chunk's first curve; null = uniform
    usdGen::UsdGenVec2f const *rootUV = nullptr;
    usdGen::UsdGenVec3f const *rootT = nullptr, *rootN = nullptr, *rootB = nullptr;
    float const *rampLut = nullptr;
    usdGen::UsdGenVec3d *pointScratch = nullptr;
    double *scalarScratch = nullptr;
    UsdGenChunkExtent *extentSlot = nullptr;
    uint32_t curveCount = 0, cvCount = 0, inCvCount = 0, inFirstCv = 0;
    int const *inCvOffsets = nullptr;

    UsdGenChunkView const &View()
    {
        _range = UsdGenChunkRange{};
        _range.first[0] = desc ? desc->firstCurve : 0;
        _range.count[0] = curveCount;
        _range.first[1] = desc ? desc->firstCv : 0;
        _range.stride[1] = cvCount;
        _range.live = desc ? desc->liveCount : curveCount;
        _range.group = desc ? uint16_t(desc->tile) : 0;
        _range.domain = desc ? uint16_t(desc->surface) : 0;
        for (uint32_t s = 0; s < kUsdGenCurveStaticSlots; ++s) {
            _out[s] = nullptr;
            _in[s] = _in2[s] = nullptr;
        }
        _out[kUsdGenCurvePx] = px;
        _out[kUsdGenCurvePy] = py;
        _out[kUsdGenCurvePz] = pz;
        _out[kUsdGenCurveWidth] = width;
        _out[kUsdGenCurveHairT] = hairT;
        _out[kUsdGenCurveId] = const_cast<uint64_t *>(curveId);
        _out[kUsdGenCurveRootPrim] = const_cast<int *>(rootPrim);
        _out[kUsdGenCurveRootUV] = const_cast<usdGen::UsdGenVec2f *>(rootUV);
        _out[kUsdGenCurveRootT] = const_cast<usdGen::UsdGenVec3f *>(rootT);
        _out[kUsdGenCurveRootN] = const_cast<usdGen::UsdGenVec3f *>(rootN);
        _out[kUsdGenCurveRootB] = const_cast<usdGen::UsdGenVec3f *>(rootB);
        _in[kUsdGenCurvePx] = inPx;
        _in[kUsdGenCurvePy] = inPy;
        _in[kUsdGenCurvePz] = inPz;
        _in[kUsdGenCurveWidth] = inWidth;
        _in2[kUsdGenCurveWidth] = inWidth2;
        _view = UsdGenChunkView{};
        _view.range = &_range;
        _view.offsets[2] = reinterpret_cast<uint32_t const *>(cvOffsets);
        _view.inOffsets[2] = reinterpret_cast<uint32_t const *>(inCvOffsets);
        _view.inFirst[2] = inFirstCv;
        _view.inStride[2] = inCvCount;
        _view.out = _out;
        _view.in = _in;
        _view.in2 = inWidth2 ? _in2 : nullptr;
        _view.pointScratch = pointScratch;
        _view.scalarScratch = scalarScratch;
        _view.extentSlot = extentSlot;
        _view.rampLut = rampLut;
        return _view;
    }

private:
    UsdGenChunkRange _range;
    void *_out[kUsdGenCurveStaticSlots] = {};
    void const *_in[kUsdGenCurveStaticSlots] = {};
    void const *_in2[kUsdGenCurveStaticSlots] = {};
    UsdGenChunkView _view{};
};

}  // namespace usdGen
