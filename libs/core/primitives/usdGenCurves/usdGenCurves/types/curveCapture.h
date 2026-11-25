// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Curve captures: generators fill a builder buffer; payload captures carry
// per-curve / per-CV scalars resolved once per capture epoch (I4).
#pragma once
#include "usdGen/registries/op.h"
#include "usdGenCurves/types/curveBuffer.h"

#include <memory>

namespace usdGen {

/// A capture that may own a builder buffer (a generator's output topology).
struct UsdGenCurveCapture : UsdGenCapture
{
    UsdGenCurveBuffer const &Buffer() const { return _buffer; }
    UsdGenCurveBuffer &MutableBuffer() { return _buffer; }
    /// True for owning captures (generators, Resample, frozen Freeze): the
    /// scheduler installs Output() as the node's base after a successful Capture.
    virtual bool OwnsBuffer() const { return false; }
    /// OwnsBuffer(): the buffer as a set with the buffer's own versions (a
    /// copy per call; the scheduler converts once per capture). Null
    /// otherwise or when inconsistent.
    USDGEN_CORE_API UsdGenRef<const UsdGenElementSet> Output() const override;

protected:
    UsdGenCurveBuffer _buffer;
};

/// Shared payload base. Copyable (VtArray CoW); every CONCRETE payload must
/// override Clone() with a full copy of its own derived type: a base-level
/// clone would silently slice the op-specific fields (the E-6 reuse defect).
struct UsdGenCapturePayload : UsdGenCurveCapture
{
    VtFloatArray perCurve;      // op-specific meaning (target length, ...)
    VtFloatArray perCv;         // op-specific meaning (noise displacement field, ...)
    VtFloatArray rampLut;       // 257-entry op-specific LUT over hairT; empty == unused

    float const *RampLut() const override { return rampLut.empty() ? nullptr : rampLut.cdata(); }
    std::unique_ptr<UsdGenCapture> Clone() const override = 0;
};

/// The builder buffer of a curve capture held as a UsdGenCapture (tests,
/// backends). 'c' must be a UsdGenCurveCapture.
inline UsdGenCurveBuffer const &UsdGenCaptureBuffer(UsdGenCapture const &c)
{
    return static_cast<UsdGenCurveCapture const &>(c).Buffer();
}
inline UsdGenCurveBuffer &UsdGenCaptureMutableBuffer(UsdGenCapture &c)
{
    return static_cast<UsdGenCurveCapture &>(c).MutableBuffer();
}

}  // namespace usdGen
