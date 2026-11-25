// C3 BasisCurves loader.  This is deliberately an engine-only value loader:
// it consumes the graph descriptor snapshot and never reads a USD stage.
#ifndef USDGEN_CURVE_LOADER_H
#define USDGEN_CURVE_LOADER_H

#include "usdGen/curveBuffer.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include <memory>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenRestSurfaceBindingCache;

/// Loads one authored C3 curve set into a fresh, canonical CurveBuffer.
/// `out` is unchanged on failure. Root binding/frame capture is portable and
/// immutable; an optional owner slot reuses validated rest-surface indices.
class UsdGenCurveLoader final
{
public:
    static bool Load(UsdGenCaptureContext const &ctx,
                     UsdGenCurveSetDesc const &source,
                     UsdGenCurveBuffer *out,
                     UsdGenDiagnostics *diag,
                     std::shared_ptr<const UsdGenRestSurfaceBindingCache> *bindingCache = nullptr);
};

} // namespace usdGen

#endif // USDGEN_CURVE_LOADER_H
