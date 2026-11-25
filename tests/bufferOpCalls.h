// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Test helpers (5.3b-e): drive a new-family op's Capture/Evaluate by concrete
// type over builder buffers and legacy chunk views. 5.3f converts the callers.
#pragma once
#include "usdGen/registries/op.h"
#include "usdGenCurves/types/curveBridge.h"
#include "usdGenCurves/types/curveBuffer.h"

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

/// A chunk view that wraps the old family's view for UsdGenCurveChunk.
class UsdGenLegacyChunkWrap
{
public:
    explicit UsdGenLegacyChunkWrap(UsdGenLegacyChunkView *legacy) { _view.legacy = legacy; }
    UsdGenLegacyChunkWrap(UsdGenLegacyChunkWrap const &) = delete;
    UsdGenLegacyChunkWrap &operator=(UsdGenLegacyChunkWrap const &) = delete;
    operator UsdGenChunkView const &() const { return _view; }

private:
    UsdGenChunkView _view{};
};

}  // namespace usdGen
