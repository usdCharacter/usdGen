// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenCurves/publish/curvePublishDesc.h"
#include "usdGenCurves/preview/valuePreview.h"
#include "usdGenCurves/types/curveSet.h"
#include "usdGenSurface/surfaceInput.h"
#include "usdGen/api/deformBinding.h"
#include "usdGen/compile/graph.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/publisher.h"
#include "usdGen/registries/token.h"

#include <algorithm>
#include <utility>

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

// The rest points of one group over 'rest' (the terminal's input), walked
// chunk by chunk like the tile build over 'term'; false when the two curve
// layouts disagree. *padding: half the widest published width.
bool GroupRest(UsdGenChunkGroup const &tv, UsdGenCurveSetView const &rest,
               UsdGenCurveSetView const &term, UsdGenSpan<const UsdGenChunkRange> chunks,
               UsdGenVec3fArray *out, double *padding)
{
    auto span = [&](UsdGenCurveSetView const &v, UsdGenChunkRange const &cd, uint32_t c,
                    uint32_t *p0, uint32_t *len) {
        uint32_t const g = cd.first[0] + c;
        *p0 = 0;
        *len = 0;
        if (cd.stride[1] != 0) {
            *len = cd.stride[1];
            *p0 = cd.first[1] + c * cd.stride[1];
        } else if (g + 1 < v.cvOffsets.size()) {   // ragged chunk
            *p0 = static_cast<uint32_t>(v.cvOffsets[g]);
            *len = static_cast<uint32_t>(v.cvOffsets[g + 1]) - *p0;
        }
    };
    if (size_t(tv.firstChunk) + tv.chunkCount > chunks.size()) return false;
    UsdGenVec3fArray points;
    double pad = 0.0;
    for (uint32_t i = 0; i < tv.chunkCount; ++i) {
        UsdGenChunkRange const &cd = chunks[tv.firstChunk + i];
        for (uint32_t c = 0; c < cd.live; ++c) {
            uint32_t r0 = 0, rlen = 0, t0 = 0, tlen = 0;
            span(rest, cd, c, &r0, &rlen);
            span(term, cd, c, &t0, &tlen);
            if (rlen != tlen || size_t(r0) + rlen > rest.px.size() ||
                size_t(t0) + tlen > term.px.size())
                return false;
            for (uint32_t v = 0; v < rlen; ++v) {
                uint32_t const p = r0 + v;
                points.push_back(UsdGenVec3f(rest.px[p], rest.py[p], rest.pz[p]));
                uint32_t const q = t0 + v;
                float w = 0.01f;
                if (q < term.width.size()) w = term.width[q];
                else if (!term.width.empty()) w = term.width.back();
                pad = std::max(pad, double(w) * 0.5);
            }
        }
    }
    *out = std::move(points);
    *padding = pad;
    return true;
}

// Only a static styling stack ending in one surface-driven RBF retains its
// upstream geometry while the surface poses (the Deform op's hook decides).
bool DeformBinding(UsdGenPublishContext const &ctx, UsdGenRef<const UsdGenDeformBinding> *out)
{
    if (!ctx.spec || !ctx.graph || !out) return false;
    UsdGenGraph const &graph = *ctx.graph;
    auto const &terminal = graph.Node(graph.TerminalNodeId());
    if (!terminal.op || !terminal.record || !terminal.desc || terminal.inputs.empty()) return false;
    auto const makeBinding = terminal.record->Def().hooks.makeAnimationBinding;
    if (!makeBinding) return false;
    UsdGenAnimationBindingQuery query;
    query.terminalDesc = terminal.desc;
    query.terminalParams = &terminal.paramView;
    UsdGenSpan<const UsdGenInputRecord> const surfaceRecords = UsdGenSurfaceInputs(*ctx.spec);
    query.surface = terminal.surface < surfaceRecords.size() ? &surfaceRecords[terminal.surface] : nullptr;
    query.terminalInputCount = uint32_t(terminal.inputs.size());
    query.terminalHasOutputPlanes = !terminal.outputPrimvars.empty();
    query.graphStatic = true;
    for (int i = 0; i != graph.NodeCount(); ++i) {
        auto const &node = graph.Node(i);
        if (!node.op || !node.record || node.record->Has(kUsdGenOpReadsTime) ||
            !node.desc->expressionBindings.empty())
            query.graphStatic = false;
        else
            for (auto const &param : node.desc->params) if (param.animated) query.graphStatic = false;
        if (node.opId == terminal.opId) ++query.nodesWithTerminalOp;
    }
    UsdGenRef<UsdGenDeformBinding> binding(new UsdGenDeformBinding);
    if (!makeBinding(query, *terminal.op, binding.Get()) || !query.surface) return false;
    binding->surface = query.surface->key;
    UsdGenCurveSetView const rest = UsdGenCurveSetView::Of(graph.NodeOutput(terminal.inputs.front()));
    UsdGenCurveSetView const term =
        ctx.terminal ? UsdGenCurveSetView::Of(*ctx.terminal) : UsdGenCurveSetView{};
    binding->groups.reserve(ctx.groups.size());
    for (UsdGenChunkGroup const &tv : ctx.groups) {
        UsdGenDeformBindingGroup group;
        group.group = tv.id;
        if (!GroupRest(tv, rest, term, ctx.chunks, &group.restPoints, &group.widthPadding))
            return false;
        binding->groups.push_back(std::move(group));
    }
    *out = std::move(binding);
    return true;
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
    d.deformBinding = &DeformBinding;
    return &d;
}

}  // namespace usdGen
