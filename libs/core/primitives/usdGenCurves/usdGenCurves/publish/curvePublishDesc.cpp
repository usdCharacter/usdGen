// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenCurves/publish/curvePublishDesc.h"
#include "usdGenCurves/preview/valuePreview.h"
#include "usdGenCurves/types/curveSet.h"
#include "usdGenSurface/surfaceInput.h"
#include "usdGen/registries/publisher.h"
#include "usdGen/registries/token.h"

namespace usdGen {

namespace {

// Filled at registration (UsdGenMakeCurvePublishDesc).
UsdGenPublishDesc gCurvePublish;

class CurveHookState final : public UsdGenPublishHookState
{
public:
    ~CurveHookState() override;
    UsdGenValuePreview preview;
};

CurveHookState::~CurveHookState() = default;

UsdGenPublishHookState *CreateHookState()
{
    return new CurveHookState;
}

bool PreviewValues(UsdGenPublishContext const &ctx, UsdGenPublishHookState *state,
                   UsdGenPreviewValues *out, UsdGenDiagnostics *diag)
{
    if (!ctx.spec || !ctx.graph || !ctx.terminal || !state || !out) return false;
    static_cast<CurveHookState *>(state)->preview.Build(
        *ctx.spec, *ctx.graph, UsdGenCurveSetView::Of(*ctx.terminal), ctx.frame, out, diag);
    return true;
}

bool Occludes(UsdGenSurfaceInputView const &surface)
{
    return !surface.points.empty() && !surface.faceVertexIndices.empty();
}

// The emitting surfaces are opaque. A GeomSubset surface carries its whole
// parent mesh, which occludes whole and once, however many subsets of it
// the groom binds: the first input of each mesh stands for it.
UsdGenNodeKey Occluder(UsdGenPublishContext const &ctx, uint32_t index)
{
    if (!ctx.spec) return {};
    UsdGenSpan<const UsdGenInputRecord> const records = UsdGenSurfaceInputs(*ctx.spec);
    uint32_t found = 0;
    for (size_t i = 0; i < records.size(); ++i) {
        UsdGenSurfaceInputView const surface = UsdGenSurfaceInputView::Of(records[i]);
        if (!Occludes(surface)) continue;
        UsdGenNodeKey const mesh = UsdGenSurfaceMeshKey(surface);
        bool repeat = false;
        for (size_t j = 0; j < i && !repeat; ++j) {
            UsdGenSurfaceInputView const earlier = UsdGenSurfaceInputView::Of(records[j]);
            repeat = Occludes(earlier) && UsdGenSurfaceMeshKey(earlier) == mesh;
        }
        if (repeat) continue;
        if (found++ == index) return records[i].key;
    }
    return {};
}

}  // namespace

UsdGenPublishDesc const *UsdGenMakeCurvePublishDesc()
{
    UsdGenCompileTokenScope scope;
    UsdGenPublishDesc &d = gCurvePublish;
    d.kind = UsdGenPublishKind::Lines;
    d.type = UsdGenInternToken("cubic");
    d.basis = UsdGenInternToken("bspline");
    d.wrap = UsdGenInternToken("pinned");
    d.materialClass = UsdGenInternToken("hair");
    d.createState = &CreateHookState;
    d.previewValues = &PreviewValues;
    d.occluder = &Occluder;
    return &d;
}

}  // namespace usdGen
