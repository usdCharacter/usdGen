// usdGenPomade — C ABI implementation (P0 model + P1 commit + P2 graph/bake).
#include "usdGenPomade/pomadeApi.h"
#include "usdGenPomade/pomadeApiImpl.h"
#include "usdGenPomade/pomadeBake.h"
#include "usdGenPomade/pomadeCommit.h"
#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadeRegistry.h"
#include "usdGenPomade/pomadeTransport.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

thread_local std::string _tlError;

void _SetError(char const *what)
{
    _tlError = what ? what : "unknown pomade error";
}

using usdGenPomade::PomadeModelContextImpl;
using usdGenPomade::PomadeCommitterContextImpl;
using usdGenPomade::PomadeBakeContextImpl;

PomadeModelContextImpl *_Impl(PomadeModelContext *ctx)
{
    return reinterpret_cast<PomadeModelContextImpl *>(ctx);
}

PomadeModelContextImpl const *_Impl(PomadeModelContext const *ctx)
{
    return reinterpret_cast<PomadeModelContextImpl const *>(ctx);
}

PomadeCommitterContextImpl *_CImpl(PomadeCommitterContext *cc)
{
    return reinterpret_cast<PomadeCommitterContextImpl *>(cc);
}

PomadeCommitterContextImpl const *_CImpl(PomadeCommitterContext const *cc)
{
    return reinterpret_cast<PomadeCommitterContextImpl const *>(cc);
}

PomadeBakeContextImpl *_BImpl(PomadeBakeContext *bake)
{
    return reinterpret_cast<PomadeBakeContextImpl *>(bake);
}

PomadeBakeContextImpl const *_BImpl(PomadeBakeContext const *bake)
{
    return reinterpret_cast<PomadeBakeContextImpl const *>(bake);
}

// Locate (faceId, u, v) on the model's scalp (positions via
// PomadeFacePosition, normals from the face frame). A face outside a bound
// face subset is off the scalp, exactly as a raycast would report it.
bool _Locate(usdGenPomade::PomadeModel &model, int faceId, float u, float v,
             usdGenPomade::PomadeHit *hit)
{
    std::shared_ptr<usdGenPomade::PomadeScalpMesh const> scalp =
        model.GetScalp();
    if (!scalp || !scalp->finalized || !hit) {
        return false;
    }
    if (!usdGenPomade::PomadeScalpFaceActive(*scalp, faceId)) {
        return false;
    }
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    if (!usdGenPomade::PomadeFacePosition(*scalp, faceId, u, v, &px, &py, &pz)) {
        return false;
    }
    hit->hit = true;
    hit->faceId = faceId;
    hit->u = u;
    hit->v = v;
    hit->px = px;
    hit->py = py;
    hit->pz = pz;
    hit->nx = scalp->faceNormals[size_t(faceId) * 3 + 0];
    hit->ny = scalp->faceNormals[size_t(faceId) * 3 + 1];
    hit->nz = scalp->faceNormals[size_t(faceId) * 3 + 2];
    return true;
}

void _WriteHit(usdGenPomade::PomadeHit const &hit, int *outHit, int *outFace,
               float outUV[2], float outP[3], float outN[3])
{
    if (outHit) {
        *outHit = hit.hit ? 1 : 0;
    }
    if (!hit.hit) {
        return;
    }
    if (outFace) {
        *outFace = hit.faceId;
    }
    if (outUV) {
        outUV[0] = hit.u;
        outUV[1] = hit.v;
    }
    if (outP) {
        outP[0] = hit.px;
        outP[1] = hit.py;
        outP[2] = hit.pz;
    }
    if (outN) {
        outN[0] = hit.nx;
        outN[1] = hit.ny;
        outN[2] = hit.nz;
    }
}

} // namespace

extern "C" {

int
Pomade_Create(PomadeModelContext **outCtx)
{
    try {
        if (!outCtx) {
            _SetError("Pomade_Create: outCtx is null");
            return POMADE_ERROR;
        }
        *outCtx = reinterpret_cast<PomadeModelContext *>(
            new (std::nothrow) PomadeModelContextImpl());
        if (!*outCtx) {
            _SetError("Pomade_Create: allocation failed");
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_Create: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_Destroy(PomadeModelContext *ctx)
{
    try {
        delete _Impl(ctx);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_Destroy: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_BuildTestTube(PomadeModelContext *ctx, int rings, int ringVerts,
                    float radius, float length)
{
    try {
        if (!ctx) {
            _SetError("Pomade_BuildTestTube: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeTubeShape shape;
        if (rings > 0) {
            shape.rings = rings;
        }
        if (ringVerts > 0) {
            shape.ringVerts = ringVerts;
        }
        if (radius > 0.0f) {
            shape.radius = radius;
        }
        if (length > 0.0f) {
            shape.length = length;
        }
        if (!_Impl(ctx)->model.BuildTestTube(shape)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_BuildTestTube: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_MoveCenterRing(PomadeModelContext *ctx, int ring, float dx, float dz)
{
    try {
        if (!ctx) {
            _SetError("Pomade_MoveCenterRing: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.MoveCenterRing(ring, dx, dz)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_MoveCenterRing: unknown exception");
        return POMADE_ERROR;
    }
}

unsigned long long
Pomade_GetVersion(PomadeModelContext const *ctx)
{
    try {
        if (!ctx) {
            return 0;
        }
        return _Impl(ctx)->model.GetVersion();
    } catch (...) {
        return 0;
    }
}

int
Pomade_TakeDirty(PomadeModelContext *ctx)
{
    try {
        if (!ctx) {
            return 0;
        }
        return int(_Impl(ctx)->model.TakeDirty());
    } catch (...) {
        return 0;
    }
}

int
Pomade_GetVertexCount(PomadeModelContext const *ctx)
{
    try {
        if (!ctx) {
            return 0;
        }
        usdGenPomade::PomadeModel::HostTubeMesh const &mesh =
            _Impl(ctx)->model.GetHostMesh();
        return int(mesh.positions.size() / 3);
    } catch (...) {
        return 0;
    }
}

int
Pomade_GetQuadCount(PomadeModelContext const *ctx)
{
    try {
        if (!ctx) {
            return 0;
        }
        usdGenPomade::PomadeModel::HostTubeMesh const &mesh =
            _Impl(ctx)->model.GetHostMesh();
        return int(mesh.faceVertexCounts.size());
    } catch (...) {
        return 0;
    }
}

int
Pomade_ReadTubePoints(PomadeModelContext *ctx, float *outXYZ, int xyzLen)
{
    try {
        if (!ctx || !outXYZ) {
            _SetError("Pomade_ReadTubePoints: null argument");
            return POMADE_ERROR;
        }
        PomadeModelContextImpl *impl = _Impl(ctx);
        usdGenPomade::PomadeStagedTubeMesh staged;
        if (!usdGenPomade::PomadeStageTubeMesh(impl->model, &impl->positions,
                                             &impl->normals, &staged)) {
            _SetError("Pomade_ReadTubePoints: no tube staged");
            return POMADE_ERROR;
        }
        int const need = int(staged.points.size()) * 3;
        if (xyzLen < need) {
            _SetError("Pomade_ReadTubePoints: output too small");
            return POMADE_ERROR;
        }
        std::memcpy(outXYZ, staged.points.data(),
                    size_t(need) * sizeof(float));
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadTubePoints: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_HasCudaMirror(PomadeModelContext const *ctx)
{
    try {
        if (!ctx) {
            return 0;
        }
        return _Impl(ctx)->model.HasCudaMirror() ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

char const *
Pomade_GetDeviceFallbackReason(PomadeModelContext const *ctx)
{
    try {
        if (!ctx) {
            return "";
        }
        return _Impl(ctx)->model.DeviceFallbackReason();
    } catch (...) {
        return "";
    }
}

int
Pomade_Undo(PomadeModelContext *ctx, unsigned int *outDirty)
{
    try {
        if (!ctx) {
            _SetError("Pomade_Undo: null context");
            return POMADE_ERROR;
        }
        uint32_t dirty = 0;
        if (!_Impl(ctx)->model.Undo(&dirty)) {
            return POMADE_ERROR;
        }
        if (outDirty) {
            *outDirty = dirty;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_Undo: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetUndoDepth(PomadeModelContext const *ctx)
{
    try {
        if (!ctx) {
            return 0;
        }
        return _Impl(ctx)->model.GetUndoDepth();
    } catch (...) {
        return 0;
    }
}

unsigned long long
Pomade_GetUndoBytes(PomadeModelContext const *ctx)
{
    try {
        if (!ctx) {
            return 0;
        }
        return _Impl(ctx)->model.GetUndoBytes();
    } catch (...) {
        return 0;
    }
}

int
Pomade_SetUndoBudget(PomadeModelContext *ctx, int maxDepth,
                    unsigned long long maxBytes)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetUndoBudget: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SetUndoBudget(
                maxDepth, uint64_t(maxBytes))) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetUndoBudget: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ClearUndo(PomadeModelContext *ctx)
{
    try {
        if (!ctx) {
            _SetError("Pomade_ClearUndo: null context");
            return POMADE_ERROR;
        }
        _Impl(ctx)->model.ClearUndo();
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ClearUndo: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_CommitterCreate(PomadeModelContext *modelCtx, const char *groomPath,
                      const char *descPath, PomadeCommitterContext **outCc)
{
    try {
        if (!modelCtx || !outCc) {
            _SetError("Pomade_CommitterCreate: null context or out-param");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeCommitPaths paths;
        if (groomPath && *groomPath) {
            paths.groomPath = SdfPath(groomPath);
        }
        if (descPath && *descPath) {
            paths.descriptionPath = SdfPath(descPath);
        }
        if (paths.groomPath.IsEmpty() ||
            !paths.groomPath.IsAbsolutePath()) {
            _SetError("Pomade_CommitterCreate: groom path must be absolute");
            return POMADE_ERROR;
        }
        std::unique_ptr<PomadeCommitterContextImpl> impl(
            new (std::nothrow) PomadeCommitterContextImpl());
        if (!impl) {
            _SetError("Pomade_CommitterCreate: allocation failed");
            return POMADE_ERROR;
        }
        impl->committer.reset(new usdGenPomade::PomadeCommitter(
            &_Impl(modelCtx)->model, paths));
        *outCc = reinterpret_cast<PomadeCommitterContext *>(impl.release());
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_CommitterCreate: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_CommitterDestroy(PomadeCommitterContext *cc)
{
    try {
        delete _CImpl(cc);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_CommitterDestroy: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_CommitterEnqueue(PomadeCommitterContext *cc, int createOp, int setGuides,
                       int setRegion, const char *opPath)
{
    try {
        if (!cc) {
            _SetError("Pomade_CommitterEnqueue: null committer");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeFillPlan plan;
        plan.createInterpOp = createOp != 0;
        plan.setInterpGuides = setGuides != 0;
        plan.setInterpRegion = setRegion != 0;
        if (opPath && *opPath) {
            plan.opPath = SdfPath(opPath);
        }
        _CImpl(cc)->committer->EnqueuePlan(plan);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_CommitterEnqueue: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_CommitterSwap(PomadeCommitterContext *cc, const char *liveIdentifier,
                    int gestureActive)
{
    try {
        // PomadeCommitter_Error, never POMADE_ERROR: POMADE_ERROR is 1, which
        // is also PomadeCommitter_PartialProgress, so a failed swap used to
        // read as "call again next idle slot" and nobody ever heard of it.
        if (!cc || !liveIdentifier) {
            _SetError("Pomade_CommitterSwap: null argument");
            return PomadeCommitter_Error;
        }
        SdfLayerHandle live =
            SdfLayer::Find(std::string(liveIdentifier));
        if (!live) {
            std::string const what =
                "Pomade_CommitterSwap: unknown live layer " +
                std::string(liveIdentifier);
            _SetError(what.c_str());
            return PomadeCommitter_Error;
        }
        return int(_CImpl(cc)->committer->SwapIfIdle(
            live, gestureActive != 0));
    } catch (std::exception const &e) {
        _SetError(e.what());
        return PomadeCommitter_Error;
    } catch (...) {
        _SetError("Pomade_CommitterSwap: unknown exception");
        return PomadeCommitter_Error;
    }
}

int
Pomade_CommitterTakeDiagnostic(PomadeCommitterContext *cc, char *out, int cap)
{
    try {
        if (!cc) {
            _SetError("Pomade_CommitterTakeDiagnostic: null committer");
            return -1;
        }
        std::string const text = _CImpl(cc)->committer->TakeDiagnostic();
        if (out && cap > 0) {
            size_t const n = std::min(text.size(), size_t(cap - 1));
            std::memcpy(out, text.data(), n);
            out[n] = '\0';
        }
        return int(std::min(text.size(), size_t(0x7fffffff)));
    } catch (std::exception const &e) {
        _SetError(e.what());
        return -1;
    } catch (...) {
        _SetError("Pomade_CommitterTakeDiagnostic: unknown exception");
        return -1;
    }
}

unsigned long long
Pomade_CommitterFailedVersion(PomadeCommitterContext const *cc)
{
    try {
        if (!cc) {
            return 0;
        }
        return _CImpl(cc)->committer->FailedVersion();
    } catch (...) {
        return 0;
    }
}

unsigned long long
Pomade_CommitterCommittedVersion(PomadeCommitterContext const *cc)
{
    try {
        if (!cc) {
            return 0;
        }
        return _CImpl(cc)->committer->CommittedVersion();
    } catch (...) {
        return 0;
    }
}

unsigned long long
Pomade_CommitterPendingVersion(PomadeCommitterContext const *cc)
{
    try {
        if (!cc) {
            return 0;
        }
        return _CImpl(cc)->committer->PendingVersion();
    } catch (...) {
        return 0;
    }
}

double
Pomade_CommitterLastSwapMs(PomadeCommitterContext const *cc)
{
    try {
        if (!cc) {
            return 0.0;
        }
        return _CImpl(cc)->committer->LastSwapMs();
    } catch (...) {
        return 0.0;
    }
}

int
Pomade_CommitterPartialMode(PomadeCommitterContext const *cc)
{
    try {
        if (!cc) {
            return 0;
        }
        return _CImpl(cc)->committer->PartialMode() ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_CommitterCancelCooks(PomadeCommitterContext *cc)
{
    try {
        if (!cc) {
            _SetError("Pomade_CommitterCancelCooks: null committer");
            return -1;
        }
        return int(_CImpl(cc)->committer->CancelDescriptionCooks());
    } catch (std::exception const &e) {
        _SetError(e.what());
        return -1;
    } catch (...) {
        _SetError("Pomade_CommitterCancelCooks: unknown exception");
        return -1;
    }
}

int
Pomade_CommitterSetSwapBudgetMs(PomadeCommitterContext *cc, double ms)
{
    try {
        if (!cc) {
            _SetError("Pomade_CommitterSetSwapBudgetMs: null committer");
            return POMADE_ERROR;
        }
        _CImpl(cc)->committer->SetSwapBudgetMs(ms);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_CommitterSetSwapBudgetMs: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_CommitterDetach(PomadeCommitterContext *cc)
{
    try {
        if (!cc) {
            _SetError("Pomade_CommitterDetach: null committer");
            return POMADE_ERROR;
        }
        _CImpl(cc)->committer->Detach();
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_CommitterDetach: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_CommitterReattach(PomadeCommitterContext *cc)
{
    try {
        if (!cc) {
            _SetError("Pomade_CommitterReattach: null committer");
            return POMADE_ERROR;
        }
        _CImpl(cc)->committer->Reattach();
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_CommitterReattach: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_BindScalp(PomadeModelContext *ctx, float const *points, int pointFloats,
                int const *faceVertexCounts, int faceCount,
                int const *faceVertexIndices, int indexCount)
{
    try {
        if (!ctx || !points || !faceVertexCounts || !faceVertexIndices ||
            pointFloats <= 0 || faceCount <= 0 || indexCount <= 0) {
            _SetError("Pomade_BindScalp: null or empty scalp input");
            return POMADE_ERROR;
        }
        std::vector<float> pts(points, points + pointFloats);
        std::vector<int> counts(faceVertexCounts,
                                faceVertexCounts + faceCount);
        std::vector<int> indices(faceVertexIndices,
                                 faceVertexIndices + indexCount);
        if (!_Impl(ctx)->model.BindScalp(pts, counts, indices)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_BindScalp: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_BindScalpSubset(PomadeModelContext *ctx, float const *points,
                      int pointFloats, int const *faceVertexCounts,
                      int faceCount, int const *faceVertexIndices,
                      int indexCount, int const *activeFaces, int activeCount)
{
    try {
        if (!ctx || !points || !faceVertexCounts || !faceVertexIndices ||
            pointFloats <= 0 || faceCount <= 0 || indexCount <= 0) {
            _SetError("Pomade_BindScalpSubset: null or empty scalp input");
            return POMADE_ERROR;
        }
        // An empty subset is a GeomSubset that names no face: an authoring
        // error, never "the whole mesh" (that is Pomade_BindScalp).
        if (!activeFaces || activeCount <= 0) {
            _SetError("Pomade_BindScalpSubset: the face subset names no faces");
            return POMADE_ERROR;
        }
        std::vector<float> pts(points, points + pointFloats);
        std::vector<int> counts(faceVertexCounts,
                                faceVertexCounts + faceCount);
        std::vector<int> indices(faceVertexIndices,
                                 faceVertexIndices + indexCount);
        std::vector<int> active(activeFaces, activeFaces + activeCount);
        if (!_Impl(ctx)->model.BindScalp(pts, counts, indices, active)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_BindScalpSubset: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadScalpFaceActive(PomadeModelContext *ctx, int *out, int maxOut,
                          int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadScalpFaceActive: null argument");
            return POMADE_ERROR;
        }
        std::shared_ptr<usdGenPomade::PomadeScalpMesh const> const scalp =
            _Impl(ctx)->model.GetScalp();
        if (!scalp || !scalp->finalized) {
            *outCount = 0;
            return POMADE_OK;
        }
        int const faceCount = int(scalp->faceVertexCounts.size());
        *outCount = faceCount;
        if (out) {
            if (maxOut < faceCount) {
                _SetError("Pomade_ReadScalpFaceActive: output too small");
                return POMADE_ERROR;
            }
            for (int f = 0; f < faceCount; ++f) {
                out[f] = usdGenPomade::PomadeScalpFaceActive(*scalp, f) ? 1 : 0;
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadScalpFaceActive: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_HasScalp(PomadeModelContext const *ctx)
{
    try {
        return (ctx && _Impl(ctx)->model.HasScalp()) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_Raycast(PomadeModelContext *ctx, float const origin[3],
              float const dir[3], int *outHit, int *outFace, float outUV[2],
              float outP[3], float outN[3])
{
    try {
        if (!ctx || !origin || !dir) {
            _SetError("Pomade_Raycast: null argument");
            return POMADE_ERROR;
        }
        _WriteHit(_Impl(ctx)->model.Raycast(origin, dir), outHit, outFace,
                  outUV, outP, outN);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_Raycast: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ClosestPoint(PomadeModelContext *ctx, float const p[3], int *outHit,
                   int *outFace, float outUV[2], float outP[3], float outN[3])
{
    try {
        if (!ctx || !p) {
            _SetError("Pomade_ClosestPoint: null argument");
            return POMADE_ERROR;
        }
        _WriteHit(_Impl(ctx)->model.ClosestPoint(p), outHit, outFace, outUV,
                  outP, outN);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ClosestPoint: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphAddNode(PomadeModelContext *ctx, int faceId, float u, float v,
                   int *outId)
{
    try {
        if (!ctx || !outId) {
            _SetError("Pomade_GraphAddNode: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeHit hit;
        if (!_Locate(_Impl(ctx)->model, faceId, u, v, &hit)) {
            _SetError("Pomade_GraphAddNode: (face, uv) is off the scalp");
            return POMADE_ERROR;
        }
        int const id = _Impl(ctx)->model.GraphAddNode(hit);
        if (id < 0) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        *outId = id;
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphAddNode: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphCreateRegion(PomadeModelContext *ctx, int const *nodeIds,
                        int const *faceIds, float const *uvs, int count,
                        int *outRegionId)
{
    try {
        if (!ctx || !nodeIds || !faceIds || !uvs || count < 3) {
            _SetError("Pomade_GraphCreateRegion: invalid argument");
            return POMADE_ERROR;
        }
        std::vector<int> ids(nodeIds, nodeIds + count);
        std::vector<usdGenPomade::PomadeHit> hits(
            size_t(count), usdGenPomade::PomadeHit{});
        for (int i = 0; i < count; ++i) {
            if (ids[size_t(i)] >= 0) {
                continue;  // Exact stable id: its supplied (face, uv) is ignored.
            }
            if (ids[size_t(i)] != -1 ||
                !_Locate(_Impl(ctx)->model, faceIds[i], uvs[i * 2 + 0],
                         uvs[i * 2 + 1], &hits[size_t(i)])) {
                _SetError("Pomade_GraphCreateRegion: (face, uv) is off the scalp");
                return POMADE_ERROR;
            }
        }
        int const regionId = _Impl(ctx)->model.GraphCreateRegion(ids, hits);
        if (regionId < 0) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        if (outRegionId) {
            *outRegionId = regionId;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphCreateRegion: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphMoveNode(PomadeModelContext *ctx, int nodeId, int faceId, float u,
                    float v)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GraphMoveNode: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeHit hit;
        if (!_Locate(_Impl(ctx)->model, faceId, u, v, &hit)) {
            _SetError("Pomade_GraphMoveNode: (face, uv) is off the scalp");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.GraphMoveNode(nodeId, hit)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphMoveNode: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphMoveNodes(PomadeModelContext *ctx, int const *nodeIds,
                     int const *faceIds, float const *uvs, int count)
{
    try {
        if (!ctx || !nodeIds || !faceIds || !uvs || count <= 0) {
            _SetError("Pomade_GraphMoveNodes: invalid argument");
            return POMADE_ERROR;
        }
        std::vector<int> ids(nodeIds, nodeIds + count);
        std::vector<usdGenPomade::PomadeHit> hits(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            if (!_Locate(_Impl(ctx)->model, faceIds[i], uvs[i * 2 + 0],
                         uvs[i * 2 + 1], &hits[size_t(i)])) {
                _SetError("Pomade_GraphMoveNodes: (face, uv) is off the scalp");
                return POMADE_ERROR;
            }
        }
        if (!_Impl(ctx)->model.GraphMoveNodes(ids, hits)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphMoveNodes: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphConnect(PomadeModelContext *ctx, int a, int b, int *outEdge)
{
    try {
        if (!ctx || !outEdge) {
            _SetError("Pomade_GraphConnect: null argument");
            return POMADE_ERROR;
        }
        int const id = _Impl(ctx)->model.GraphConnect(a, b);
        if (id < 0) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        *outEdge = id;
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphConnect: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphSplitEdge(PomadeModelContext *ctx, int edgeId, int faceId, float u,
                     float v, int *outNode)
{
    try {
        if (!ctx || !outNode) {
            _SetError("Pomade_GraphSplitEdge: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeHit hit;
        if (!_Locate(_Impl(ctx)->model, faceId, u, v, &hit)) {
            _SetError("Pomade_GraphSplitEdge: (face, uv) is off the scalp");
            return POMADE_ERROR;
        }
        int const id = _Impl(ctx)->model.GraphSplitEdge(edgeId, hit);
        if (id < 0) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        *outNode = id;
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphSplitEdge: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphWeld(PomadeModelContext *ctx, int keep, int drop)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GraphWeld: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.GraphWeld(keep, drop)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphWeld: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphWeldAll(PomadeModelContext *ctx, float radius, int *outWelds)
{
    try {
        if (!ctx || !outWelds) {
            _SetError("Pomade_GraphWeldAll: null argument");
            return POMADE_ERROR;
        }
        int const welds = _Impl(ctx)->model.GraphWeldAll(radius);
        if (welds < 0) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        *outWelds = welds;
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphWeldAll: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphUnweld(PomadeModelContext *ctx, int nodeId, int *outIds, int maxOut,
                  int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_GraphUnweld: null argument");
            return POMADE_ERROR;
        }
        std::vector<int> created = _Impl(ctx)->model.GraphUnweld(nodeId);
        *outCount = int(created.size());
        if (outIds) {
            for (size_t i = 0;
                 i < created.size() && int(i) < maxOut; ++i) {
                outIds[i] = created[i];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphUnweld: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphDeleteEdge(PomadeModelContext *ctx, int edgeId)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GraphDeleteEdge: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.GraphDeleteEdge(edgeId)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphDeleteEdge: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphDeleteNode(PomadeModelContext *ctx, int nodeId)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GraphDeleteNode: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.GraphDeleteNode(nodeId)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphDeleteNode: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphSnapNode(PomadeModelContext const *ctx, float const p[3],
                    float radius)
{
    try {
        if (!ctx || !p) {
            return -1;
        }
        return _Impl(ctx)->model.GraphSnapNode(p, radius);
    } catch (...) {
        return -1;
    }
}

int
Pomade_GraphSnapEdge(PomadeModelContext const *ctx, float const p[3],
                    float radius)
{
    try {
        if (!ctx || !p) {
            return -1;
        }
        return _Impl(ctx)->model.GraphSnapEdge(p, radius);
    } catch (...) {
        return -1;
    }
}

int
Pomade_GraphGetNode(PomadeModelContext const *ctx, int nodeId,
                   int *outFaceId, float outUV[2], float outP[3])
{
    try {
        if (!ctx || !outFaceId || !outUV || !outP) {
            _SetError("Pomade_GraphGetNode: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeGraphNode node;
        if (!_Impl(ctx)->model.GraphGetNode(nodeId, &node)) {
            _SetError("Pomade_GraphGetNode: unknown node id");
            return POMADE_ERROR;
        }
        *outFaceId = node.faceId;
        outUV[0] = node.u;
        outUV[1] = node.v;
        outP[0] = node.p[0];
        outP[1] = node.p[1];
        outP[2] = node.p[2];
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphGetNode: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphGetEdge(PomadeModelContext const *ctx, int edgeId,
                   int outNodeIds[2])
{
    try {
        if (!ctx || !outNodeIds) {
            _SetError("Pomade_GraphGetEdge: null argument");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.GraphGetEdge(edgeId, outNodeIds)) {
            _SetError("Pomade_GraphGetEdge: unknown edge id");
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphGetEdge: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphGetNodeDisplayPosition(PomadeModelContext const *ctx, int nodeId,
                                  float outP[3])
{
    try {
        if (!ctx || !outP) {
            _SetError("Pomade_GraphGetNodeDisplayPosition: null argument");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.GraphGetNodeDisplayPosition(nodeId, outP)) {
            _SetError("Pomade_GraphGetNodeDisplayPosition: unknown node id");
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphGetNodeDisplayPosition: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphLinkRegions(PomadeModelContext *ctx, int r0, int r1)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GraphLinkRegions: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.GraphLinkRegions(r0, r1)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphLinkRegions: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphUnlinkRegions(PomadeModelContext *ctx, int r0, int r1)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GraphUnlinkRegions: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.GraphUnlinkRegions(r0, r1)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphUnlinkRegions: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphStroke(PomadeModelContext *ctx, int const *faceIds,
                  float const *uvs, int samples, float snapRadius,
                  float simplifyEps, int *outNodes, int maxOut, int *outCount,
                  int *outClosed, int *outWeldedStart, int *outWeldedEnd)
{
    try {
        if (!ctx || !faceIds || !uvs || samples < 0 || !outCount) {
            _SetError("Pomade_GraphStroke: null argument");
            return POMADE_ERROR;
        }
        std::vector<usdGenPomade::PomadeHit> hits;
        hits.reserve(size_t(samples));
        for (int i = 0; i < samples; ++i) {
            usdGenPomade::PomadeHit hit;
            if (!_Locate(_Impl(ctx)->model, faceIds[i], uvs[i * 2],
                         uvs[i * 2 + 1], &hit)) {
                _SetError("Pomade_GraphStroke: a sample is off the scalp");
                return POMADE_ERROR;
            }
            hits.push_back(hit);
        }
        usdGenPomade::PomadeStrokeResult result =
            _Impl(ctx)->model.GraphStroke(hits, snapRadius, simplifyEps);
        *outCount = int(result.nodeIds.size());
        if (outNodes) {
            for (size_t i = 0;
                 i < result.nodeIds.size() && int(i) < maxOut; ++i) {
                outNodes[i] = result.nodeIds[i];
            }
        }
        if (outClosed) {
            *outClosed = result.closedLoop ? 1 : 0;
        }
        if (outWeldedStart) {
            *outWeldedStart = result.weldedStart ? 1 : 0;
        }
        if (outWeldedEnd) {
            *outWeldedEnd = result.weldedEnd ? 1 : 0;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphStroke: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GraphMirrorX(PomadeModelContext *ctx, int *outPairs, int maxPairs,
                   int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_GraphMirrorX: null argument");
            return POMADE_ERROR;
        }
        std::vector<std::pair<int, int>> mapping =
            _Impl(ctx)->model.GraphMirrorX();
        *outCount = int(mapping.size());
        if (outPairs) {
            for (size_t i = 0;
                 i < mapping.size() && int(i) < maxPairs; ++i) {
                outPairs[i * 2 + 0] = mapping[i].first;
                outPairs[i * 2 + 1] = mapping[i].second;
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GraphMirrorX: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetSnapRadius(PomadeModelContext *ctx, float radius)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetSnapRadius: null context");
            return POMADE_ERROR;
        }
        _Impl(ctx)->model.SetSnapRadius(radius);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetSnapRadius: unknown exception");
        return POMADE_ERROR;
    }
}

float
Pomade_GetSnapRadius(PomadeModelContext const *ctx)
{
    try {
        return ctx ? _Impl(ctx)->model.GetSnapRadius() : 0.0f;
    } catch (...) {
        return 0.0f;
    }
}

int
Pomade_SetMirrorX(PomadeModelContext *ctx, int on)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetMirrorX: null context");
            return POMADE_ERROR;
        }
        _Impl(ctx)->model.SetMirrorX(on != 0);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetMirrorX: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetMirrorX(PomadeModelContext const *ctx)
{
    try {
        return (ctx && _Impl(ctx)->model.GetMirrorX()) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_Rasterise(PomadeModelContext *ctx)
{
    try {
        if (!ctx) {
            _SetError("Pomade_Rasterise: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.Rasterise()) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_Rasterise: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_RegionAtSurface(PomadeModelContext const *ctx, int faceId, float u,
                      float v)
{
    try {
        return ctx ? _Impl(ctx)->model.RegionAtSurface(faceId, u, v) : -1;
    } catch (...) {
        return -1;
    }
}

unsigned long long
Pomade_GetMapVersion(PomadeModelContext const *ctx)
{
    try {
        return ctx ? _Impl(ctx)->model.GetMapVersion() : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_GetGraphCounts(PomadeModelContext const *ctx, int *outNodes,
                     int *outEdges, int *outRegions)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GetGraphCounts: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::GraphSnapshot snap =
            _Impl(ctx)->model.SnapshotGraph();
        if (outNodes) {
            *outNodes = int(snap.nodes.size());
        }
        if (outEdges) {
            *outEdges = int(snap.edges.size());
        }
        if (outRegions) {
            *outRegions = int(snap.regionLoops.size());
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetGraphCounts: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetRegionStats(PomadeModelContext const *ctx, int *outRegions,
                     int *outUncovered, int *outIntersected)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GetRegionStats: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::GraphSnapshot snap =
            _Impl(ctx)->model.SnapshotGraph();
        if (outRegions) {
            *outRegions = int(snap.regionLoops.size());
        }
        if (outUncovered) {
            *outUncovered = snap.uncoveredCount;
        }
        if (outIntersected) {
            *outIntersected = snap.intersectedCount;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetRegionStats: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadFaceRegions(PomadeModelContext *ctx, int *out, int maxOut,
                      int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadFaceRegions: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::GraphSnapshot snap =
            _Impl(ctx)->model.SnapshotGraph();
        *outCount = int(snap.faceRegions.size());
        if (out) {
            if (maxOut < int(snap.faceRegions.size())) {
                _SetError("Pomade_ReadFaceRegions: output too small");
                return POMADE_ERROR;
            }
            std::memcpy(out, snap.faceRegions.data(),
                        snap.faceRegions.size() * sizeof(int));
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadFaceRegions: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadFaceRegionIds(PomadeModelContext *ctx, int *out, int maxOut,
                        int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadFaceRegionIds: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::GraphSnapshot snap =
            _Impl(ctx)->model.SnapshotGraph();
        *outCount = int(snap.faceRegionIds.size());
        if (out) {
            if (maxOut < int(snap.faceRegionIds.size())) {
                _SetError("Pomade_ReadFaceRegionIds: output too small");
                return POMADE_ERROR;
            }
            std::memcpy(out, snap.faceRegionIds.data(),
                        snap.faceRegionIds.size() * sizeof(int));
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadFaceRegionIds: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadGraphNodes(PomadeModelContext *ctx, int *outFaceIds, float *outUV,
                     float *outP, int maxNodes, int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadGraphNodes: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::GraphSnapshot snap =
            _Impl(ctx)->model.SnapshotGraph();
        *outCount = int(snap.nodes.size());
        if ((outFaceIds || outUV || outP) &&
            maxNodes < int(snap.nodes.size())) {
            _SetError("Pomade_ReadGraphNodes: output too small");
            return POMADE_ERROR;
        }
        for (size_t i = 0; i < snap.nodes.size(); ++i) {
            if (outFaceIds) {
                outFaceIds[i] = snap.nodes[i].faceId;
            }
            if (outUV) {
                outUV[i * 2 + 0] = snap.nodes[i].u;
                outUV[i * 2 + 1] = snap.nodes[i].v;
            }
            if (outP) {
                outP[i * 3 + 0] = snap.nodes[i].p[0];
                outP[i * 3 + 1] = snap.nodes[i].p[1];
                outP[i * 3 + 2] = snap.nodes[i].p[2];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadGraphNodes: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadGraphEdges(PomadeModelContext *ctx, int *outPairs, int maxEdges,
                     int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadGraphEdges: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::GraphSnapshot snap =
            _Impl(ctx)->model.SnapshotGraph();
        *outCount = int(snap.edges.size());
        if (outPairs) {
            if (maxEdges < int(snap.edges.size())) {
                _SetError("Pomade_ReadGraphEdges: output too small");
                return POMADE_ERROR;
            }
            for (size_t i = 0; i < snap.edges.size(); ++i) {
                outPairs[i * 2 + 0] = snap.edges[i].first;
                outPairs[i * 2 + 1] = snap.edges[i].second;
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadGraphEdges: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadRegionColors(PomadeModelContext *ctx, float *outRGB, int maxRegions,
                       int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadRegionColors: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::GraphSnapshot snap =
            _Impl(ctx)->model.SnapshotGraph();
        int const regions = int(snap.regionColors.size() / 3);
        *outCount = regions;
        if (outRGB) {
            if (maxRegions < regions) {
                _SetError("Pomade_ReadRegionColors: output too small");
                return POMADE_ERROR;
            }
            std::memcpy(outRGB, snap.regionColors.data(),
                        snap.regionColors.size() * sizeof(float));
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadRegionColors: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadRegionLoops(PomadeModelContext *ctx, int *outCounts, int maxRegions,
                      int *outIndices, int maxIndices, int *outRegionCount,
                      int *outIndexCount)
{
    try {
        if (!ctx || !outRegionCount || !outIndexCount) {
            _SetError("Pomade_ReadRegionLoops: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::GraphSnapshot snap =
            _Impl(ctx)->model.SnapshotGraph();
        int total = 0;
        for (auto const &loop : snap.regionLoops) {
            total += int(loop.size());
        }
        *outRegionCount = int(snap.regionLoops.size());
        *outIndexCount = total;
        if (outCounts) {
            if (maxRegions < int(snap.regionLoops.size())) {
                _SetError("Pomade_ReadRegionLoops: counts too small");
                return POMADE_ERROR;
            }
            for (size_t r = 0; r < snap.regionLoops.size(); ++r) {
                outCounts[r] = int(snap.regionLoops[r].size());
            }
        }
        if (outIndices) {
            if (maxIndices < total) {
                _SetError("Pomade_ReadRegionLoops: indices too small");
                return POMADE_ERROR;
            }
            int o = 0;
            for (auto const &loop : snap.regionLoops) {
                for (int nid : loop) {
                    outIndices[o++] = nid;
                }
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadRegionLoops: unknown exception");
        return POMADE_ERROR;
    }
}

// -- P3 Tube + Fill modes + pick --------------------------------------------

int
Pomade_BuildTubeFromRegion(PomadeModelContext *ctx, int regionId,
                          int centerCount, int ringVerts, float length)
{
    try {
        if (!ctx) {
            _SetError("Pomade_BuildTubeFromRegion: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.BuildTubeFromRegion(regionId, centerCount,
                                                   ringVerts, length)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_BuildTubeFromRegion: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_MoveCenterCV(PomadeModelContext *ctx, int cv, float dx, float dy,
                   float dz)
{
    try {
        if (!ctx) {
            _SetError("Pomade_MoveCenterCV: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.MoveCenterCV(cv, dx, dy, dz)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_MoveCenterCV: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_InsertCenterCV(PomadeModelContext *ctx, int atIndex)
{
    try {
        if (!ctx) {
            _SetError("Pomade_InsertCenterCV: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.InsertCenterCV(atIndex)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_InsertCenterCV: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_DeleteCenterCV(PomadeModelContext *ctx, int index)
{
    try {
        if (!ctx) {
            _SetError("Pomade_DeleteCenterCV: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.DeleteCenterCV(index)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_DeleteCenterCV: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetTubeLength(PomadeModelContext *ctx, float length)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetTubeLength: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SetTubeLength(length)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetTubeLength: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_MatchSurface(PomadeModelContext *ctx)
{
    try {
        if (!ctx) {
            _SetError("Pomade_MatchSurface: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.MatchSurface()) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_MatchSurface: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetCenterCVCount(PomadeModelContext const *ctx)
{
    try {
        return ctx ? _Impl(ctx)->model.GetCenterCVCount() : -1;
    } catch (...) {
        return -1;
    }
}

int
Pomade_GetCenterCV(PomadeModelContext *ctx, int cv, float outXYZ[3])
{
    try {
        if (!ctx || !outXYZ) {
            _SetError("Pomade_GetCenterCV: null argument");
            return POMADE_ERROR;
        }
        float x = 0, y = 0, z = 0;
        if (!_Impl(ctx)->model.GetCenterCV(cv, &x, &y, &z)) {
            _SetError("Pomade_GetCenterCV: CV out of range");
            return POMADE_ERROR;
        }
        outXYZ[0] = x;
        outXYZ[1] = y;
        outXYZ[2] = z;
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetCenterCV: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetSectionCount(PomadeModelContext const *ctx)
{
    try {
        return ctx ? _Impl(ctx)->model.GetSectionCount() : -1;
    } catch (...) {
        return -1;
    }
}

int
Pomade_GetSection(PomadeModelContext *ctx, int ring, float *outT, float *outUV,
                 int uvLen, int *outCount, float *outScale, float *outTwist)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_GetSection: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeTubeSection section;
        if (!_Impl(ctx)->model.GetSection(ring, &section)) {
            _SetError("Pomade_GetSection: ring out of range");
            return POMADE_ERROR;
        }
        *outCount = int(section.u.size());
        if (outT) {
            *outT = section.t;
        }
        if (outScale) {
            *outScale = section.scale;
        }
        if (outTwist) {
            *outTwist = section.twist;
        }
        if (outUV) {
            if (uvLen < int(section.u.size()) * 2) {
                _SetError("Pomade_GetSection: output too small");
                return POMADE_ERROR;
            }
            for (size_t i = 0; i < section.u.size(); ++i) {
                outUV[i * 2 + 0] = section.u[i];
                outUV[i * 2 + 1] = section.v[i];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetSection: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_MoveSectionRing(PomadeModelContext *ctx, int ring, float du, float dv)
{
    try {
        if (!ctx) {
            _SetError("Pomade_MoveSectionRing: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.MoveSectionRing(ring, du, dv)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_MoveSectionRing: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ScaleSectionRing(PomadeModelContext *ctx, int ring, float scale)
{
    try {
        if (!ctx) {
            _SetError("Pomade_ScaleSectionRing: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.ScaleSectionRing(ring, scale)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ScaleSectionRing: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_TwistSectionRing(PomadeModelContext *ctx, int ring, float radians)
{
    try {
        if (!ctx) {
            _SetError("Pomade_TwistSectionRing: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.TwistSectionRing(ring, radians)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_TwistSectionRing: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_MoveSectionCV(PomadeModelContext *ctx, int ring, int slot, float du,
                    float dv)
{
    try {
        if (!ctx) {
            _SetError("Pomade_MoveSectionCV: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.MoveSectionCV(ring, slot, du, dv)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_MoveSectionCV: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_AddSectionRing(PomadeModelContext *ctx, float t, int *outRing)
{
    try {
        if (!ctx) {
            _SetError("Pomade_AddSectionRing: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.AddSectionRing(t)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        if (outRing) {
            // Inserted rings sort by t; report the new ring count so the
            // caller re-reads census.
            *outRing = _Impl(ctx)->model.GetSectionCount();
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_AddSectionRing: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_RemoveSectionRing(PomadeModelContext *ctx, int ring)
{
    try {
        if (!ctx) {
            _SetError("Pomade_RemoveSectionRing: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.RemoveSectionRing(ring)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_RemoveSectionRing: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_CopySectionRing(PomadeModelContext *ctx, int src, int dst)
{
    try {
        if (!ctx) {
            _SetError("Pomade_CopySectionRing: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.CopySectionRing(src, dst)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_CopySectionRing: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetSoftSelection(PomadeModelContext *ctx, float center, float radius)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetSoftSelection: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SetSoftSelection(center, radius)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetSoftSelection: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetSoftSelection(PomadeModelContext *ctx, float *outCenter,
                       float *outRadius)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GetSoftSelection: null context");
            return POMADE_ERROR;
        }
        _Impl(ctx)->model.GetSoftSelection(outCenter, outRadius);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetSoftSelection: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_RelaxCenter(PomadeModelContext *ctx, float strength, int iterations)
{
    try {
        if (!ctx) {
            _SetError("Pomade_RelaxCenter: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.RelaxCenter(strength, iterations)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_RelaxCenter: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SnapRootToScalp(PomadeModelContext *ctx)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SnapRootToScalp: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SnapRootToScalp()) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SnapRootToScalp: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetDisplaySegments(PomadeModelContext *ctx, int segments)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetDisplaySegments: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SetDisplaySegments(segments)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetDisplaySegments: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetDisplaySegments(PomadeModelContext const *ctx)
{
    try {
        return ctx ? _Impl(ctx)->model.GetDisplaySegments() : -1;
    } catch (...) {
        return -1;
    }
}

int
Pomade_SetTubeRegionId(PomadeModelContext *ctx, int regionId)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetTubeRegionId: null context");
            return POMADE_ERROR;
        }
        _Impl(ctx)->model.SetTubeRegionId(regionId);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetTubeRegionId: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetTubeRegionId(PomadeModelContext const *ctx)
{
    try {
        return ctx ? _Impl(ctx)->model.GetTubeRegionId() : -1;
    } catch (...) {
        return -1;
    }
}

int
Pomade_ReadTubeRegionFaces(PomadeModelContext *ctx, int *out, int maxOut,
                          int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadTubeRegionFaces: null argument");
            return POMADE_ERROR;
        }
        std::vector<int> faces = _Impl(ctx)->model.TubeRegionFaces();
        *outCount = int(faces.size());
        if (out) {
            if (maxOut < int(faces.size())) {
                _SetError("Pomade_ReadTubeRegionFaces: output too small");
                return POMADE_ERROR;
            }
            for (size_t i = 0; i < faces.size(); ++i) {
                out[i] = faces[i];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadTubeRegionFaces: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetFillParams(PomadeModelContext *ctx, float density, int cvCount,
                    int seed, float edgeBias, float const *profilePairs,
                    int pairFloats)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetFillParams: null context");
            return POMADE_ERROR;
        }
        if (pairFloats < 0 || pairFloats % 2 != 0 ||
            (pairFloats > 0 && !profilePairs)) {
            _SetError("Pomade_SetFillParams: profile holds pairs");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::FillParams params;
        params.density = density;
        params.cvCount = cvCount;
        params.seed = seed;
        params.edgeBias = edgeBias;
        params.lengthProfile.assign(
            profilePairs ? profilePairs : nullptr,
            profilePairs ? profilePairs + pairFloats : nullptr);
        if (!_Impl(ctx)->model.SetFillParams(std::move(params))) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetFillParams: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetFillParams(PomadeModelContext *ctx, float *outDensity, int *outCvCount,
                    int *outSeed, float *outEdgeBias, float *outProfile,
                    int maxFloats, int *outFloats)
{
    try {
        if (!ctx || !outFloats) {
            _SetError("Pomade_GetFillParams: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::FillParams params =
            _Impl(ctx)->model.GetFillParams();
        if (outDensity) {
            *outDensity = params.density;
        }
        if (outCvCount) {
            *outCvCount = params.cvCount;
        }
        if (outSeed) {
            *outSeed = params.seed;
        }
        if (outEdgeBias) {
            *outEdgeBias = params.edgeBias;
        }
        *outFloats = int(params.lengthProfile.size());
        if (outProfile) {
            if (maxFloats < int(params.lengthProfile.size())) {
                _SetError("Pomade_GetFillParams: profile output too small");
                return POMADE_ERROR;
            }
            for (size_t i = 0; i < params.lengthProfile.size(); ++i) {
                outProfile[i] = params.lengthProfile[i];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetFillParams: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetOutputSettings(PomadeModelContext *ctx, int enabled,
                        float densityMultiplier, float width)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetOutputSettings: null context");
            return POMADE_ERROR;
        }
        if (enabled != 0 && enabled != 1) {
            _SetError("Pomade_SetOutputSettings: enabled must be 0 or 1");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::OutputSettings settings =
            _Impl(ctx)->model.GetOutputSettings();
        settings.enabled = enabled != 0;
        settings.densityMultiplier = densityMultiplier;
        settings.width = width;
        if (!_Impl(ctx)->model.SetOutputSettings(settings)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetOutputSettings: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetOutputSettings(PomadeModelContext const *ctx, int *outEnabled,
                        float *outDensityMultiplier, float *outWidth)
{
    try {
        if (!ctx || !outEnabled || !outDensityMultiplier || !outWidth) {
            _SetError("Pomade_GetOutputSettings: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::OutputSettings const settings =
            _Impl(ctx)->model.GetOutputSettings();
        *outEnabled = settings.enabled ? 1 : 0;
        *outDensityMultiplier = settings.densityMultiplier;
        *outWidth = settings.width;
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetOutputSettings: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetPreviewFraction(PomadeModelContext *ctx, float fraction)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetPreviewFraction: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SetPreviewFraction(fraction)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetPreviewFraction: unknown exception");
        return POMADE_ERROR;
    }
}

float
Pomade_GetPreviewFraction(PomadeModelContext const *ctx)
{
    try {
        return ctx ? _Impl(ctx)->model.GetPreviewFraction() : 0.0f;
    } catch (...) {
        return 0.0f;
    }
}

int
Pomade_SetFreezeRoots(PomadeModelContext *ctx, int on)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetFreezeRoots: null context");
            return POMADE_ERROR;
        }
        _Impl(ctx)->model.SetFreezeRoots(on != 0);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetFreezeRoots: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetFreezeRoots(PomadeModelContext const *ctx)
{
    try {
        return ctx && _Impl(ctx)->model.GetFreezeRoots() ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_RefillGuides(PomadeModelContext *ctx, float fraction)
{
    try {
        if (!ctx) {
            _SetError("Pomade_RefillGuides: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.RefillGuides(fraction)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_RefillGuides: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GenerateGuides(PomadeModelContext *ctx, float fraction)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GenerateGuides: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.GenerateGuides(fraction)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    }
}

int
Pomade_ClearGeneratedCurves(PomadeModelContext *ctx)
{
    try {
        if (!ctx) {
            _SetError("Pomade_ClearGeneratedCurves: null context");
            return POMADE_ERROR;
        }
        return _Impl(ctx)->model.ClearGeneratedCurves() ? POMADE_OK
                                                         : POMADE_ERROR;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    }
}

int
Pomade_SetGeneratedCurvesVisible(PomadeModelContext *ctx, int visible)
{
    if (!ctx) {
        _SetError("Pomade_SetGeneratedCurvesVisible: null context");
        return POMADE_ERROR;
    }
    return _Impl(ctx)->model.SetGeneratedCurvesVisible(visible != 0)
               ? POMADE_OK
               : POMADE_ERROR;
}

int
Pomade_GetGeneratedCurvesVisible(PomadeModelContext const *ctx, int *outVisible)
{
    if (!ctx || !outVisible) {
        _SetError("Pomade_GetGeneratedCurvesVisible: null argument");
        return POMADE_ERROR;
    }
    *outVisible = _Impl(ctx)->model.GetGeneratedCurvesVisible() ? 1 : 0;
    return POMADE_OK;
}

int
Pomade_GetGuideCounts(PomadeModelContext const *ctx, int *outGuides, int *outCv)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GetGuideCounts: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::GuidePreview preview =
            _Impl(ctx)->model.GetGuidePreview();
        if (outGuides) {
            *outGuides = preview.guideCount;
        }
        if (outCv) {
            *outCv = preview.cvCount;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetGuideCounts: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadRefillDrops(PomadeModelContext const *ctx, int *out, int outCap,
                      int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadRefillDrops: null context or count");
            return POMADE_ERROR;
        }
        std::vector<std::pair<int, std::string>> const drops =
            _Impl(ctx)->model.RefillDrops();
        *outCount = int(drops.size());
        if (out && outCap >= *outCount) {
            for (size_t i = 0; i < drops.size(); ++i) {
                out[i] = drops[i].first;
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadRefillDrops: unknown exception");
        return POMADE_ERROR;
    }
}

const char *
Pomade_GetRefillDropReason(PomadeModelContext const *ctx, int index)
{
    // A copy per thread: the model's list is replaced by the next refill,
    // possibly on another thread, while the caller still reads this one.
    static thread_local std::string reason;
    try {
        reason.clear();
        if (ctx && index >= 0) {
            std::vector<std::pair<int, std::string>> const drops =
                _Impl(ctx)->model.RefillDrops();
            if (size_t(index) < drops.size()) {
                reason = drops[size_t(index)].second;
            }
        }
        return reason.c_str();
    } catch (...) {
        return "";
    }
}

int
Pomade_ReadGuidePreview(PomadeModelContext *ctx, float *outXYZ, int xyzLen,
                       int *outCounts, int countsLen, int *outGuideCount)
{
    try {
        if (!ctx || !outGuideCount) {
            _SetError("Pomade_ReadGuidePreview: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::GuidePreview preview =
            _Impl(ctx)->model.GetGuidePreview();
        *outGuideCount = preview.guideCount;
        if (outXYZ) {
            if (xyzLen < int(preview.points.size())) {
                _SetError("Pomade_ReadGuidePreview: points too small");
                return POMADE_ERROR;
            }
            for (size_t i = 0; i < preview.points.size(); ++i) {
                outXYZ[i] = preview.points[i];
            }
        }
        if (outCounts) {
            if (countsLen < int(preview.counts.size())) {
                _SetError("Pomade_ReadGuidePreview: counts too small");
                return POMADE_ERROR;
            }
            for (size_t i = 0; i < preview.counts.size(); ++i) {
                outCounts[i] = preview.counts[i];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadGuidePreview: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadGuideRoots(PomadeModelContext *ctx, int *outFaceIds, float *outXYZ,
                     float *outRU, int maxRoots, int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadGuideRoots: null argument");
            return POMADE_ERROR;
        }
        // Snapshot the roots under the model's mutex (GetRoots is
        // UI-thread-only by convention; the vector copy is the snapshot).
        std::vector<usdGenPomade::PomadeGuideRoot> roots =
            _Impl(ctx)->model.GetRoots();
        *outCount = int(roots.size());
        if ((outFaceIds || outXYZ || outRU) && maxRoots < int(roots.size())) {
            _SetError("Pomade_ReadGuideRoots: output too small");
            return POMADE_ERROR;
        }
        for (size_t i = 0; i < roots.size(); ++i) {
            if (outFaceIds) {
                outFaceIds[i] = roots[i].faceId;
            }
            if (outXYZ) {
                outXYZ[i * 3 + 0] = roots[i].px;
                outXYZ[i * 3 + 1] = roots[i].py;
                outXYZ[i * 3 + 2] = roots[i].pz;
            }
            if (outRU) {
                outRU[i * 2 + 0] = roots[i].ru;
                outRU[i * 2 + 1] = roots[i].rv;
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadGuideRoots: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_Pick(PomadeModelContext *ctx, float const viewProj[16], int w, int h,
           float x, float y, float radiusPx, unsigned int kindMask,
           int *outHit, unsigned int *outKind, int *outIndex, int *outSubIndex,
           float *outDistPx, float *outDepth)
{
    try {
        if (!ctx || !viewProj || !outHit) {
            _SetError("Pomade_Pick: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadePickHit hit = _Impl(ctx)->model.Pick(
            viewProj, w, h, x, y, radiusPx, uint32_t(kindMask));
        *outHit = hit.hit ? 1 : 0;
        if (!hit.hit) {
            return POMADE_OK;
        }
        if (outKind) {
            *outKind = hit.kind;
        }
        if (outIndex) {
            *outIndex = hit.index;
        }
        if (outSubIndex) {
            *outSubIndex = hit.subIndex;
        }
        if (outDistPx) {
            *outDistPx = hit.distPx;
        }
        if (outDepth) {
            *outDepth = hit.depth;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_Pick: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_CommitterSetScalpPath(PomadeCommitterContext *cc, const char *scalpPath)
{
    try {
        if (!cc) {
            _SetError("Pomade_CommitterSetScalpPath: null committer");
            return POMADE_ERROR;
        }
        if (!scalpPath || !*scalpPath) {
            _CImpl(cc)->committer->SetScalpPath(SdfPath());
        } else {
            SdfPath const path(scalpPath);
            if (!path.IsAbsolutePath()) {
                _SetError("Pomade_CommitterSetScalpPath: path must be absolute");
                return POMADE_ERROR;
            }
            _CImpl(cc)->committer->SetScalpPath(path);
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_CommitterSetScalpPath: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_CommitterSetScalpTarget(PomadeCommitterContext *cc,
                              const char *scalpPath, const char *meshPath)
{
    try {
        if (!cc) {
            _SetError("Pomade_CommitterSetScalpTarget: null committer");
            return POMADE_ERROR;
        }
        if (!scalpPath || !*scalpPath) {
            _CImpl(cc)->committer->SetScalpPath(SdfPath());
            return POMADE_OK;
        }
        SdfPath const path(scalpPath);
        SdfPath const mesh =
            (meshPath && *meshPath) ? SdfPath(meshPath) : SdfPath();
        if (!path.IsAbsolutePath() ||
            (!mesh.IsEmpty() && !mesh.IsAbsolutePath())) {
            _SetError("Pomade_CommitterSetScalpTarget: paths must be absolute");
            return POMADE_ERROR;
        }
        // A face GeomSubset is a namespace child of its Mesh (plan/02
        // §2.20 rule 1); anything else would put the rest binding on a
        // prim that does not carry the subset's geometry.
        if (!mesh.IsEmpty() && mesh != path && path.GetParentPath() != mesh) {
            _SetError(("Pomade_CommitterSetScalpTarget: " + mesh.GetString() +
                       " is not the parent of the subset " + path.GetString())
                          .c_str());
            return POMADE_ERROR;
        }
        _CImpl(cc)->committer->SetScalpPath(path, mesh);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_CommitterSetScalpTarget: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_BakeCreate(PomadeModelContext *modelCtx, const char *outDir,
                 const char *baseName, PomadeBakeContext **outBake)
{
    try {
        if (!modelCtx || !outDir || !*outDir || !outBake) {
            _SetError("Pomade_BakeCreate: null argument");
            return POMADE_ERROR;
        }
        std::unique_ptr<PomadeBakeContextImpl> impl(
            new (std::nothrow) PomadeBakeContextImpl());
        if (!impl) {
            _SetError("Pomade_BakeCreate: allocation failed");
            return POMADE_ERROR;
        }
        impl->model = &_Impl(modelCtx)->model;
        impl->worker.reset(new usdGenPomade::PomadeBakeWorker());
        impl->outDir = outDir;
        if (baseName && *baseName) {
            impl->baseName = baseName;
        }
        *outBake = reinterpret_cast<PomadeBakeContext *>(impl.release());
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_BakeCreate: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_BakeDestroy(PomadeBakeContext *bake)
{
    try {
        delete _BImpl(bake);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_BakeDestroy: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_BakeSetOptions(PomadeBakeContext *bake, int resOverride, int levelCount)
{
    try {
        if (!bake) {
            _SetError("Pomade_BakeSetOptions: null bake context");
            return POMADE_ERROR;
        }
        if (levelCount < 1) {
            _SetError("Pomade_BakeSetOptions: levelCount must be >= 1");
            return POMADE_ERROR;
        }
        if (!_BImpl(bake)->model->SetOutputPtexResolution(resOverride)) {
            _SetError(_BImpl(bake)->model->GetDiagnostic());
            return POMADE_ERROR;
        }
        _BImpl(bake)->resOverride = resOverride;
        _BImpl(bake)->levelCount = levelCount;
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_BakeSetOptions: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_BakeEnqueue(PomadeBakeContext *bake)
{
    try {
        if (!bake) {
            _SetError("Pomade_BakeEnqueue: null bake context");
            return POMADE_ERROR;
        }
        PomadeBakeContextImpl *impl = _BImpl(bake);
        if (!impl->model->HasScalp()) {
            _SetError("Pomade_BakeEnqueue: no scalp bound");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeBakeInput input;
        input.scalp = impl->model->GetScalp();
        input.graph = impl->model->GetGraph();
        input.levelCount = impl->levelCount;
        input.resOverride = impl->resOverride;
        input.outDir = impl->outDir;
        input.baseName = impl->baseName;
        impl->worker->Enqueue(impl->model->GetMapVersion(), std::move(input));
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_BakeEnqueue: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_BakeTakeCompleted(PomadeBakeContext *bake, unsigned long long *outVersion,
                        char *outPath, int pathLen)
{
    try {
        if (!bake) {
            return 0;
        }
        uint64_t version = 0;
        std::string path;
        if (!_BImpl(bake)->worker->TakeCompleted(&version, &path)) {
            return 0;
        }
        if (outVersion) {
            *outVersion = version;
        }
        if (outPath && pathLen > 0) {
            std::strncpy(outPath, path.c_str(), size_t(pathLen) - 1);
            outPath[pathLen - 1] = '\0';
        }
        return 1;
    } catch (...) {
        return 0;
    }
}

int
Pomade_BakeSwap(PomadeBakeContext *bake, unsigned long long mapVersion,
               const char *liveIdentifier, const char *regionMapPath,
               const char *file)
{
    try {
        if (!bake || !liveIdentifier || !regionMapPath || !file) {
            _SetError("Pomade_BakeSwap: null argument");
            return POMADE_ERROR;
        }
        SdfLayerHandle live = SdfLayer::Find(std::string(liveIdentifier));
        if (!live) {
            _SetError("Pomade_BakeSwap: unknown live layer");
            return POMADE_ERROR;
        }
        std::string err;
        if (!usdGenPomade::PomadeBakeSwapMapFile(
                live, SdfPath(regionMapPath), file, &err)) {
            _SetError(err.c_str());
            return POMADE_ERROR;
        }
        _BImpl(bake)->model->NoteBakedMapFile(uint64_t(mapVersion), file);
        _BImpl(bake)->worker->NoteSwapped(uint64_t(mapVersion), file);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_BakeSwap: unknown exception");
        return POMADE_ERROR;
    }
}

unsigned long long
Pomade_BakePendingVersion(PomadeBakeContext const *bake)
{
    try {
        return bake ? _BImpl(bake)->worker->PendingVersion() : 0;
    } catch (...) {
        return 0;
    }
}

unsigned long long
Pomade_BakeCompletedVersion(PomadeBakeContext const *bake)
{
    try {
        return bake ? _BImpl(bake)->worker->CompletedVersion() : 0;
    } catch (...) {
        return 0;
    }
}

const char *
Pomade_GetLastError(void)
{
    return _tlError.c_str();
}

// -- P4: hierarchy + sculpt -----------------------------------------------

int
Pomade_SubdivideTube(PomadeModelContext *ctx, int tubeId, int count,
                    const char *splitMode, int seed, int *outIds, int outCap,
                    int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_SubdivideTube: null context or count");
            return POMADE_ERROR;
        }
        std::vector<int> kids;
        if (!_Impl(ctx)->model.SubdivideTube(tubeId, count, splitMode, seed,
                                             &kids)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        *outCount = int(kids.size());
        if (outIds && outCap >= *outCount) {
            for (size_t i = 0; i < kids.size(); ++i) {
                outIds[i] = kids[i];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SubdivideTube: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_MergeChildren(PomadeModelContext *ctx, int tubeId)
{
    try {
        if (!ctx) {
            _SetError("Pomade_MergeChildren: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.MergeChildren(tubeId)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_MergeChildren: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_MergeSelected(PomadeModelContext *ctx, int const *tubeIds, int idCount,
                    int *outKept)
{
    try {
        if (!ctx || !outKept) {
            _SetError("Pomade_MergeSelected: null context or output");
            return POMADE_ERROR;
        }
        if (!tubeIds || idCount <= 0) {
            _SetError("Pomade_MergeSelected: want tube ids");
            return POMADE_ERROR;
        }
        auto ids = std::vector<int>(size_t(idCount), 0);
        for (int i = 0; i < idCount; ++i) {
            ids[size_t(i)] = tubeIds[i];
        }
        if (!_Impl(ctx)->model.MergeSelected(ids, outKept)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_MergeSelected: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetTubeCount(PomadeModelContext const *ctx)
{
    try {
        if (!ctx) {
            return 0;
        }
        return _Impl(ctx)->model.GetTubeCount();
    } catch (...) {
        return 0;
    }
}

int
Pomade_GetTubeLevel(PomadeModelContext const *ctx, int tubeId)
{
    try {
        if (!ctx) {
            return -1;
        }
        return _Impl(ctx)->model.GetTubeLevel(tubeId);
    } catch (...) {
        return -1;
    }
}

int
Pomade_GetTubeChildren(PomadeModelContext const *ctx, int tubeId, int *outIds,
                      int outCap, int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_GetTubeChildren: null context or count");
            return POMADE_ERROR;
        }
        std::vector<int> kids = _Impl(ctx)->model.GetTubeChildren(tubeId);
        *outCount = int(kids.size());
        if (outIds && outCap >= *outCount) {
            for (size_t i = 0; i < kids.size(); ++i) {
                outIds[i] = kids[i];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetTubeChildren: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetTubeCenterCount(PomadeModelContext const *ctx, int tubeId)
{
    try {
        if (!ctx) {
            return -1;
        }
        return _Impl(ctx)->model.GetTubeCenterCount(tubeId);
    } catch (...) {
        return -1;
    }
}

int
Pomade_GetTubeCenterCV(PomadeModelContext const *ctx, int tubeId, int cv,
                      float *outXYZ)
{
    try {
        if (!ctx || !outXYZ) {
            _SetError("Pomade_GetTubeCenterCV: null context or output");
            return POMADE_ERROR;
        }
        float x = 0.0f, y = 0.0f, z = 0.0f;
        if (!_Impl(ctx)->model.GetTubeCenterCV(tubeId, cv, &x, &y, &z)) {
            _SetError("Pomade_GetTubeCenterCV: unknown tube or CV");
            return POMADE_ERROR;
        }
        outXYZ[0] = x;
        outXYZ[1] = y;
        outXYZ[2] = z;
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetTubeCenterCV: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetTubeCenterHandle(PomadeModelContext const *ctx, int tubeId, int cv,
                          float *outXYZ)
{
    try {
        if (!ctx || !outXYZ) {
            _SetError("Pomade_GetTubeCenterHandle: null context or output");
            return POMADE_ERROR;
        }
        float x = 0.0f, y = 0.0f, z = 0.0f;
        if (!_Impl(ctx)->model.GetTubeCenterHandle(tubeId, cv, &x, &y, &z)) {
            _SetError("Pomade_GetTubeCenterHandle: unknown tube, CV or "
                      "malformed section");
            return POMADE_ERROR;
        }
        outXYZ[0] = x;
        outXYZ[1] = y;
        outXYZ[2] = z;
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetTubeCenterHandle: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_MoveTubeCenterCV(PomadeModelContext *ctx, int tubeId, int cv, float dx,
                       float dy, float dz)
{
    try {
        if (!ctx) {
            _SetError("Pomade_MoveTubeCenterCV: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.MoveTubeCenterCV(tubeId, cv, dx, dy, dz)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_MoveTubeCenterCV: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_TranslateTube(PomadeModelContext *ctx, int tubeId, float dx, float dy,
                    float dz)
{
    try {
        if (!ctx) {
            _SetError("Pomade_TranslateTube: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.TranslateTube(tubeId, dx, dy, dz)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_TranslateTube: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadTubeDeltas(PomadeModelContext const *ctx, int tubeId, float *out,
                     int outCap, int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadTubeDeltas: null context or count");
            return POMADE_ERROR;
        }
        std::vector<float> flat = _Impl(ctx)->model.ReadTubeDeltas(tubeId);
        *outCount = int(flat.size());
        if (out && outCap >= *outCount) {
            for (size_t i = 0; i < flat.size(); ++i) {
                out[i] = flat[i];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadTubeDeltas: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetLockParents(PomadeModelContext *ctx, int tubeId, int on)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetLockParents: null context");
            return POMADE_ERROR;
        }
        _Impl(ctx)->model.SetTubeLockParents(tubeId, on != 0);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetLockParents: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetLockChildren(PomadeModelContext *ctx, int tubeId, int on)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetLockChildren: null context");
            return POMADE_ERROR;
        }
        _Impl(ctx)->model.SetTubeLockChildren(tubeId, on != 0);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetLockChildren: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GroupTubes(PomadeModelContext *ctx, int const *tubeIds, int idCount,
                 int makeTransient, int *outParent)
{
    try {
        if (!ctx || !outParent) {
            _SetError("Pomade_GroupTubes: null context or output");
            return POMADE_ERROR;
        }
        if (!tubeIds || idCount <= 0) {
            _SetError("Pomade_GroupTubes: want tube ids");
            return POMADE_ERROR;
        }
        auto ids = std::vector<int>(size_t(idCount), 0);
        for (int i = 0; i < idCount; ++i) {
            ids[size_t(i)] = tubeIds[i];
        }
        if (!_Impl(ctx)->model.GroupTubes(ids, makeTransient != 0,
                                          outParent)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GroupTubes: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_MakePersistent(PomadeModelContext *ctx, int tubeId)
{
    try {
        if (!ctx) {
            _SetError("Pomade_MakePersistent: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.MakeTubePersistent(tubeId)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_MakePersistent: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SculptStroke(PomadeModelContext *ctx, int tubeId, const char *brush,
                   int const *cvIds, float const *deltas, int cvCount,
                   int preserveLength, int mirrorX)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SculptStroke: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SculptStroke(tubeId, brush, cvIds, deltas,
                                            cvCount, preserveLength != 0,
                                            mirrorX != 0)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SculptStroke: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadSmoothnessScores(PomadeModelContext const *ctx, float *out,
                           int outCap, int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadSmoothnessScores: null context or count");
            return POMADE_ERROR;
        }
        std::vector<float> scores = _Impl(ctx)->model.SmoothnessScores();
        *outCount = int(scores.size());
        if (out && outCap >= *outCount) {
            for (size_t i = 0; i < scores.size(); ++i) {
                out[i] = scores[i];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadSmoothnessScores: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_CheckRootIntersections(PomadeModelContext *ctx)
{
    try {
        if (!ctx) {
            _SetError("Pomade_CheckRootIntersections: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.CheckRootIntersections()) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_CheckRootIntersections: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadIntersectedTubes(PomadeModelContext const *ctx, int *out, int outCap,
                           int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadIntersectedTubes: null context or count");
            return POMADE_ERROR;
        }
        std::vector<int> ids = _Impl(ctx)->model.IntersectedTubes();
        *outCount = int(ids.size());
        if (out && outCap >= *outCount) {
            for (size_t i = 0; i < ids.size(); ++i) {
                out[i] = ids[i];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadIntersectedTubes: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadTubeIds(PomadeModelContext const *ctx, int *out, int outCap,
                  int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadTubeIds: null context or count");
            return POMADE_ERROR;
        }
        std::vector<int> ids = _Impl(ctx)->model.TubeIds();
        *outCount = int(ids.size());
        if (out && outCap >= *outCount) {
            for (size_t i = 0; i < ids.size(); ++i) {
                out[i] = ids[i];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadTubeIds: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadL1TubeIds(PomadeModelContext const *ctx, int *out, int outCap,
                    int *outCount)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_ReadL1TubeIds: null context or count");
            return POMADE_ERROR;
        }
        std::vector<int> ids = _Impl(ctx)->model.L1TubeIds();
        *outCount = int(ids.size());
        if (out && outCap >= *outCount) {
            for (size_t i = 0; i < ids.size(); ++i) {
                out[i] = ids[i];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadL1TubeIds: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_TubeForRegion(PomadeModelContext const *ctx, int regionId)
{
    try {
        return ctx ? _Impl(ctx)->model.TubeForRegion(regionId) : -1;
    } catch (...) {
        return -1;
    }
}

int
Pomade_RegionForTube(PomadeModelContext const *ctx, int tubeId)
{
    try {
        return ctx ? _Impl(ctx)->model.RegionForTube(tubeId) : -1;
    } catch (...) {
        return -1;
    }
}

int
Pomade_SyncRegionTubes(PomadeModelContext *ctx, int *outRebuilt,
                      int *outRemoved)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SyncRegionTubes: null context");
            return POMADE_ERROR;
        }
        std::vector<int> rebuilt, removed;
        if (!_Impl(ctx)->model.SyncRegionTubes(&rebuilt, &removed)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        if (outRebuilt) {
            *outRebuilt = int(rebuilt.size());
        }
        if (outRemoved) {
            *outRemoved = int(removed.size());
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SyncRegionTubes: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ImportLockedTube(PomadeModelContext *ctx, int parentId, float const *cx,
                       float const *cy, float const *cz, int nCv,
                       float const *secT, float const *secU, float const *secV,
                       int nSec, int ringVerts, int *outTubeId)
{
    try {
        if (!ctx) {
            _SetError("Pomade_ImportLockedTube: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.ImportLockedTube(
                parentId, cx, cy, cz, nCv, secT, secU, secV, nSec, ringVerts,
                outTubeId)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ImportLockedTube: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ImportSweptMesh(PomadeModelContext *ctx, int parentId,
                      float const *points, int ringCount, int ringVerts,
                      int *outTubeId)
{
    try {
        if (!ctx) {
            _SetError("Pomade_ImportSweptMesh: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.ImportSweptMesh(parentId, points, ringCount,
                                               ringVerts, outTubeId)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ImportSweptMesh: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetTubeSectionCount(PomadeModelContext const *ctx, int tubeId)
{
    try {
        return ctx ? _Impl(ctx)->model.GetTubeSectionCount(tubeId) : -1;
    } catch (...) {
        return -1;
    }
}

int
Pomade_GetTubeSection(PomadeModelContext *ctx, int tubeId, int ring, float *outT,
                     float *outUV, int uvLen, int *outCount, float *outScale,
                     float *outTwist)
{
    try {
        if (!ctx || !outCount) {
            _SetError("Pomade_GetTubeSection: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeTubeSection section;
        if (!_Impl(ctx)->model.GetTubeSection(tubeId, ring, &section)) {
            _SetError("Pomade_GetTubeSection: unknown tube or ring");
            return POMADE_ERROR;
        }
        *outCount = int(section.u.size());
        if (outT) {
            *outT = section.t;
        }
        if (outScale) {
            *outScale = section.scale;
        }
        if (outTwist) {
            *outTwist = section.twist;
        }
        if (outUV) {
            if (uvLen < int(section.u.size()) * 2) {
                _SetError("Pomade_GetTubeSection: output too small");
                return POMADE_ERROR;
            }
            for (size_t i = 0; i < section.u.size(); ++i) {
                outUV[i * 2 + 0] = section.u[i];
                outUV[i * 2 + 1] = section.v[i];
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetTubeSection: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_IsTubeImported(PomadeModelContext const *ctx, int tubeId)
{
    try {
        return ctx && _Impl(ctx)->model.IsTubeImported(tubeId) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

/* -- V0 viewport publication (plan/18 §2.1, §2.2) --------------------------- */

int
Pomade_GetModelId(PomadeModelContext const *ctx)
{
    try {
        return ctx ? _Impl(ctx)->modelId : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_Activate(PomadeModelContext *ctx)
{
    try {
        if (!ctx) {
            _SetError("Pomade_Activate: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeRegistry &registry =
            usdGenPomade::PomadeRegistry::Get();
        if (!registry.SetActive(_Impl(ctx)->modelId)) {
            _SetError("Pomade_Activate: model is not registered");
            return POMADE_ERROR;
        }
        /* Publish everything: this is where the test tube goes away and
         * the model's own levels appear. */
        registry.Publish(~0u);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_Activate: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_Deactivate(PomadeModelContext *ctx)
{
    try {
        (void)ctx;
        usdGenPomade::PomadeRegistry &registry =
            usdGenPomade::PomadeRegistry::Get();
        registry.SetActive(0);
        registry.Publish(~0u);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_Deactivate: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_Publish(PomadeModelContext *ctx, unsigned int dirtyMask)
{
    try {
        if (!ctx) {
            _SetError("Pomade_Publish: null context");
            return -1;
        }
        return usdGenPomade::PomadeRegistry::Get().Publish(
            uint32_t(dirtyMask));
    } catch (std::exception const &e) {
        _SetError(e.what());
        return -1;
    } catch (...) {
        _SetError("Pomade_Publish: unknown exception");
        return -1;
    }
}

int
Pomade_SetRingDisplay(PomadeModelContext *ctx, int mode)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetRingDisplay: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SetRingDisplay(mode)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetRingDisplay: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetRingDisplay(PomadeModelContext const *ctx)
{
    try {
        return ctx ? _Impl(ctx)->model.GetRingDisplay() : -1;
    } catch (...) {
        return -1;
    }
}

int
Pomade_SetAmplifiedHair(PomadeModelContext *ctx, int show)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetAmplifiedHair: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SetAmplifiedHair(show != 0)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetAmplifiedHair: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetAmplifiedHair(PomadeModelContext const *ctx)
{
    try {
        return (ctx && _Impl(ctx)->model.GetAmplifiedHair()) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_SetLevelDisplay(PomadeModelContext *ctx, int level, int visible,
                      int xray)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetLevelDisplay: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SetLevelDisplay(level, visible != 0,
                                               xray != 0)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetLevelDisplay: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetLevelDisplay(PomadeModelContext const *ctx, int level,
                      int *outVisible, int *outXray)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GetLevelDisplay: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::LevelDisplay const display =
            _Impl(ctx)->model.GetLevelDisplay(level);
        if (outVisible) {
            *outVisible = display.visible ? 1 : 0;
        }
        if (outXray) {
            *outXray = display.xray ? 1 : 0;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetLevelDisplay: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetFocusLevel(PomadeModelContext *ctx, int level)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetFocusLevel: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SetFocusLevel(level)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetFocusLevel: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetFocusLevel(PomadeModelContext const *ctx)
{
    try {
        return ctx ? _Impl(ctx)->model.GetFocusLevel() : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_SetActiveCutEnabled(PomadeModelContext *ctx, int enabled)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetActiveCutEnabled: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SetActiveCutEnabled(enabled != 0)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetActiveCutEnabled: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetActiveCutEnabled(PomadeModelContext const *ctx)
{
    try {
        return ctx && _Impl(ctx)->model.GetActiveCutEnabled() ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_SetTubeExpanded(PomadeModelContext *ctx, int tubeId, int expanded)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetTubeExpanded: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SetTubeExpanded(tubeId, expanded != 0)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetTubeExpanded: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetTubeExpanded(PomadeModelContext const *ctx, int tubeId)
{
    try {
        return ctx && _Impl(ctx)->model.GetTubeExpanded(tubeId) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_IsTubeVisible(PomadeModelContext const *ctx, int tubeId)
{
    try {
        return ctx && _Impl(ctx)->model.IsTubeVisibleInActiveCut(tubeId)
                   ? 1
                   : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_SetDisplayScale(PomadeModelContext *ctx, float worldPerPixel)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetDisplayScale: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SetDisplayScale(worldPerPixel)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetDisplayScale: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetDisplayScale(PomadeModelContext const *ctx, float *outWorldPerPixel)
{
    try {
        if (!ctx || !outWorldPerPixel) {
            _SetError("Pomade_GetDisplayScale: null argument");
            return POMADE_ERROR;
        }
        *outWorldPerPixel = _Impl(ctx)->model.GetDisplayScale();
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetDisplayScale: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetDisplayPolicy(PomadeModelContext *ctx, char const *mode,
                       char const *subMode, int focusLevel)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetDisplayPolicy: null context");
            return POMADE_ERROR;
        }
        int const displayMode =
            usdGenPomade::PomadeDisplayModeFromNames(mode, subMode);
        if (displayMode < 0) {
            _SetError("Pomade_SetDisplayPolicy: unknown mode");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel &model = _Impl(ctx)->model;
        int const maxLevel = std::max(1, model.GetMaxTubeLevel());
        for (int level = 1; level <= maxLevel; ++level) {
            usdGenPomade::PomadeModel::LevelDisplay const draw =
                usdGenPomade::PomadePolicyLevelDisplay(displayMode, level,
                                                     focusLevel);
            if (!model.SetLevelDraw(level, draw)) {
                _SetError(model.GetDiagnostic());
                return POMADE_ERROR;
            }
        }
        if (!model.SetRingDisplay(
                usdGenPomade::PomadePolicyRingDisplay(displayMode))) {
            _SetError(model.GetDiagnostic());
            return POMADE_ERROR;
        }
        if (!model.SetFocusLevel(std::max(0, focusLevel))) {
            _SetError(model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetDisplayPolicy: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetLevelDraw(PomadeModelContext const *ctx, int level,
                   float *outXrayOpacity, int *outCenters)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GetLevelDraw: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeModel::LevelDisplay const draw =
            _Impl(ctx)->model.GetLevelDisplay(level);
        if (outXrayOpacity) {
            *outXrayOpacity = draw.xray ? draw.xrayOpacity : 0.0f;
        }
        if (outCenters) {
            *outCenters = draw.centers ? 1 : 0;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetLevelDraw: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetGroomPath(PomadeModelContext *ctx, char const *path)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetGroomPath: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SetGroomPath(path ? std::string(path)
                                                 : std::string())) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetGroomPath: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetGroomPath(PomadeModelContext const *ctx, char *out, int cap)
{
    try {
        if (!ctx || !out || cap <= 0) {
            _SetError("Pomade_GetGroomPath: null argument");
            return POMADE_ERROR;
        }
        std::string const path = _Impl(ctx)->model.GetGroomPath();
        if (int(path.size()) + 1 > cap) {
            _SetError("Pomade_GetGroomPath: output too small");
            return POMADE_ERROR;
        }
        std::memcpy(out, path.c_str(), path.size() + 1);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetGroomPath: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetPublishedLevelInfo(PomadeModelContext const *ctx, int level,
                            int *outFaceCount, int *outPointCount,
                            int *outTubeCount)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GetPublishedLevelInfo: null context");
            return POMADE_ERROR;
        }
        if (!usdGenPomade::PomadeRegistry::Get().QueryPublishedLevel(
                level, outFaceCount, outPointCount, outTubeCount)) {
            _SetError("Pomade_GetPublishedLevelInfo: no attached scene index "
                      "publishes that level");
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetPublishedLevelInfo: unknown exception");
        return POMADE_ERROR;
    }
}

/* -- V1: selection, overlays, gestures (plan/18 §2.3-§2.5) --------------- */

namespace {

/* Turn the three parallel id arrays into selection items of one kind.
 * subIds / subSubIds may be NULL for the kinds that do not use them. */
bool
_CollectItems(unsigned int kind, const int *ids, const int *subIds,
              const int *subSubIds, int n,
              std::vector<usdGenPomade::PomadeSelectionItem> *out)
{
    if (n < 0 || (n > 0 && !ids)) {
        return false;
    }
    if (kind == 0 || (kind & ~uint32_t(usdGenPomade::PomadeSelect_All)) ||
        (kind & (kind - 1)) != 0) {
        return false;  /* exactly one kind per call */
    }
    out->reserve(size_t(n));
    for (int i = 0; i < n; ++i) {
        usdGenPomade::PomadeSelectionItem item;
        item.kind = kind;
        item.id = ids[i];
        item.subId = subIds ? subIds[i] : -1;
        item.subSubId = subSubIds ? subSubIds[i] : -1;
        out->push_back(item);
    }
    return true;
}

int
_ApplySelect(PomadeModelContext *ctx, usdGenPomade::PomadeSelectMode mode,
             unsigned int kind, const int *ids, const int *subIds,
             const int *subSubIds, int n, char const *who)
{
    if (!ctx) {
        _SetError(who);
        return POMADE_ERROR;
    }
    std::vector<usdGenPomade::PomadeSelectionItem> items;
    if (!_CollectItems(kind, ids, subIds, subSubIds, n, &items)) {
        _SetError(who);
        return POMADE_ERROR;
    }
    _Impl(ctx)->model.SelectionApply(mode, items);
    return POMADE_OK;
}

usdGenPomade::PomadeSelectMode
_SelectModeFrom(int mode)
{
    switch (mode) {
    case POMADE_SELECT_ADD:
        return usdGenPomade::PomadeSelect_Add;
    case POMADE_SELECT_TOGGLE:
        return usdGenPomade::PomadeSelect_Toggle;
    default:
        return usdGenPomade::PomadeSelect_Set;
    }
}

}  /* namespace */

int
Pomade_SelectClear(PomadeModelContext *ctx, unsigned int kindMask)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SelectClear: null context");
            return POMADE_ERROR;
        }
        _Impl(ctx)->model.SelectionClear(kindMask);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SelectClear: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SelectSet(PomadeModelContext *ctx, unsigned int kind, const int *ids,
                const int *subIds, const int *subSubIds, int n)
{
    try {
        return _ApplySelect(ctx, usdGenPomade::PomadeSelect_Set, kind, ids,
                            subIds, subSubIds, n,
                            "Pomade_SelectSet: null context or bad kind");
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SelectSet: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SelectAdd(PomadeModelContext *ctx, unsigned int kind, const int *ids,
                const int *subIds, const int *subSubIds, int n)
{
    try {
        return _ApplySelect(ctx, usdGenPomade::PomadeSelect_Add, kind, ids,
                            subIds, subSubIds, n,
                            "Pomade_SelectAdd: null context or bad kind");
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SelectAdd: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SelectToggle(PomadeModelContext *ctx, unsigned int kind, const int *ids,
                   const int *subIds, const int *subSubIds, int n)
{
    try {
        return _ApplySelect(ctx, usdGenPomade::PomadeSelect_Toggle, kind, ids,
                            subIds, subSubIds, n,
                            "Pomade_SelectToggle: null context or bad kind");
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SelectToggle: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SelectRect(PomadeModelContext *ctx, const float *viewProj, int w, int h,
                 float x0, float y0, float x1, float y1,
                 unsigned int kindMask, int mode)
{
    try {
        if (!ctx || !viewProj) {
            _SetError("Pomade_SelectRect: null argument");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SelectRect(viewProj, w, h, x0, y0, x1, y1,
                                          kindMask, _SelectModeFrom(mode))) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SelectRect: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SelectPolygon(PomadeModelContext *ctx, const float *viewProj, int w,
                    int h, const float *xy, int n, unsigned int kindMask,
                    int mode)
{
    try {
        if (!ctx || !viewProj || !xy) {
            _SetError("Pomade_SelectPolygon: null argument");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SelectPolygon(viewProj, w, h, xy, n, kindMask,
                                             _SelectModeFrom(mode))) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SelectPolygon: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetHover(PomadeModelContext *ctx, unsigned int kind, int id, int subId,
               int subSubId)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetHover: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeSelectionItem item;
        if (kind) {
            if ((kind & ~uint32_t(usdGenPomade::PomadeSelect_All)) ||
                (kind & (kind - 1)) != 0) {
                _SetError("Pomade_SetHover: hover takes exactly one kind");
                return POMADE_ERROR;
            }
            item.kind = kind;
            item.id = id;
            item.subId = subId;
            item.subSubId = subSubId;
        }
        _Impl(ctx)->model.SelectionSetHover(item);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetHover: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetHover(PomadeModelContext const *ctx, unsigned int *outKind,
               int *outId, int *outSubId, int *outSubSubId)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GetHover: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeSelectionItem const item =
            _Impl(ctx)->model.SelectionHover();
        if (outKind) {
            *outKind = item.kind;
        }
        if (outId) {
            *outId = item.id;
        }
        if (outSubId) {
            *outSubId = item.subId;
        }
        if (outSubSubId) {
            *outSubSubId = item.subSubId;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetHover: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_ReadSelection(PomadeModelContext const *ctx, unsigned int kind,
                    int *outIds, int *outSubIds, int *outSubSubIds, int cap,
                    int *outCount)
{
    try {
        if (!ctx) {
            _SetError("Pomade_ReadSelection: null context");
            return POMADE_ERROR;
        }
        std::vector<usdGenPomade::PomadeSelectionItem> const items =
            _Impl(ctx)->model.SelectionItems(kind);
        if (outCount) {
            *outCount = int(items.size());
        }
        bool const wantsArrays = outIds || outSubIds || outSubSubIds;
        if (!wantsArrays) {
            return POMADE_OK;  /* sizing call */
        }
        if (cap < int(items.size())) {
            _SetError("Pomade_ReadSelection: buffer too small");
            return POMADE_ERROR;
        }
        for (size_t i = 0; i < items.size(); ++i) {
            if (outIds) {
                outIds[i] = items[i].id;
            }
            if (outSubIds) {
                outSubIds[i] = items[i].subId;
            }
            if (outSubSubIds) {
                outSubSubIds[i] = items[i].subSubId;
            }
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_ReadSelection: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetSelectionCount(PomadeModelContext const *ctx, unsigned int kindMask)
{
    try {
        return ctx ? int(_Impl(ctx)->model.SelectionCount(kindMask)) : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_GetSelectionBounds(PomadeModelContext const *ctx, float *outMin,
                         float *outMax)
{
    try {
        if (!ctx || !outMin || !outMax) {
            _SetError("Pomade_GetSelectionBounds: null argument");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.SelectionBounds(outMin, outMax)) {
            _SetError("Pomade_GetSelectionBounds: nothing selected has a "
                      "position");
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetSelectionBounds: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_PickItem(PomadeModelContext *ctx, const float *viewProj, int w, int h,
               float x, float y, float radiusPx, unsigned int kindMask,
               int *outHit, unsigned int *outKind, int *outId, int *outSubId,
               int *outSubSubId)
{
    try {
        if (!ctx || !viewProj || !outHit) {
            _SetError("Pomade_PickItem: null argument");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadePickHit const hit = _Impl(ctx)->model.PickItem(
            viewProj, w, h, x, y, radiusPx, uint32_t(kindMask));
        usdGenPomade::PomadeSelectionItem const item =
            _Impl(ctx)->model.SelectionItemFromHit(hit);
        *outHit = (hit.hit && item.kind) ? 1 : 0;
        if (!*outHit) {
            return POMADE_OK;
        }
        if (outKind) {
            *outKind = item.kind;
        }
        if (outId) {
            *outId = item.id;
        }
        if (outSubId) {
            *outSubId = item.subId;
        }
        if (outSubSubId) {
            *outSubSubId = item.subSubId;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_PickItem: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetGizmo(PomadeModelContext *ctx, int kind, const float *origin,
               const float *frame, float sizeWorld, int activeHandle)
{
    return Pomade_SetGizmoEx(ctx, kind, origin, frame, sizeWorld,
                            activeHandle, usdGenPomade::PomadeGizmoAllHandles);
}

int
Pomade_SetGizmoEx(PomadeModelContext *ctx, int kind, const float *origin,
                 const float *frame, float sizeWorld, int activeHandle,
                 unsigned int allowedMask)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetGizmo: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeGizmoRecord record;
        record.kind = kind;
        record.allowedMask = allowedMask;
        if (origin) {
            for (int i = 0; i < 3; ++i) {
                record.origin[i] = origin[i];
            }
        }
        if (frame) {
            for (int i = 0; i < 9; ++i) {
                record.frame[i] = frame[i];
            }
        }
        record.sizeWorld = sizeWorld;
        record.activeHandle = activeHandle;
        if (!_Impl(ctx)->model.SetGizmo(record)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetGizmo: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetGizmo(PomadeModelContext const *ctx, int *outKind, float *outOrigin,
               float *outFrame, float *outSizeWorld, int *outActiveHandle)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GetGizmo: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeGizmoRecord const record =
            _Impl(ctx)->model.GetGizmo();
        if (outKind) {
            *outKind = record.kind;
        }
        if (outOrigin) {
            for (int i = 0; i < 3; ++i) {
                outOrigin[i] = record.origin[i];
            }
        }
        if (outFrame) {
            for (int i = 0; i < 9; ++i) {
                outFrame[i] = record.frame[i];
            }
        }
        if (outSizeWorld) {
            *outSizeWorld = record.sizeWorld;
        }
        if (outActiveHandle) {
            *outActiveHandle = record.activeHandle;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetGizmo: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetGizmoAllowedMask(PomadeModelContext const *ctx, unsigned int *outMask)
{
    try {
        if (!ctx || !outMask) {
            _SetError("Pomade_GetGizmoAllowedMask: null argument");
            return POMADE_ERROR;
        }
        *outMask = _Impl(ctx)->model.GetGizmo().allowedMask;
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetGizmoAllowedMask: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_SetBrushRing(PomadeModelContext *ctx, const float *center,
                   const float *normal, float radiusWorld)
{
    try {
        if (!ctx) {
            _SetError("Pomade_SetBrushRing: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeBrushRingRecord record;
        record.active = radiusWorld > 0.0f;
        if (center) {
            for (int i = 0; i < 3; ++i) {
                record.center[i] = center[i];
            }
        }
        if (normal) {
            for (int i = 0; i < 3; ++i) {
                record.normal[i] = normal[i];
            }
        }
        record.radiusWorld = radiusWorld > 0.0f ? radiusWorld : 0.0f;
        if (!_Impl(ctx)->model.SetBrushRing(record)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_SetBrushRing: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetBrushRing(PomadeModelContext const *ctx, int *outActive,
                   float *outCenter, float *outNormal, float *outRadius)
{
    try {
        if (!ctx) {
            _SetError("Pomade_GetBrushRing: null context");
            return POMADE_ERROR;
        }
        usdGenPomade::PomadeBrushRingRecord const record =
            _Impl(ctx)->model.GetBrushRing();
        if (outActive) {
            *outActive = record.active ? 1 : 0;
        }
        if (outCenter) {
            for (int i = 0; i < 3; ++i) {
                outCenter[i] = record.center[i];
            }
        }
        if (outNormal) {
            for (int i = 0; i < 3; ++i) {
                outNormal[i] = record.normal[i];
            }
        }
        if (outRadius) {
            *outRadius = record.radiusWorld;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetBrushRing: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_BeginGesture(PomadeModelContext *ctx, const char *label)
{
    try {
        if (!ctx) {
            _SetError("Pomade_BeginGesture: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.BeginGesture(label)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_BeginGesture: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_EndGesture(PomadeModelContext *ctx)
{
    try {
        if (!ctx) {
            _SetError("Pomade_EndGesture: null context");
            return POMADE_ERROR;
        }
        if (!_Impl(ctx)->model.EndGesture()) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_EndGesture: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_CancelGesture(PomadeModelContext *ctx, unsigned int *outDirty)
{
    try {
        if (!ctx) {
            _SetError("Pomade_CancelGesture: null context");
            return POMADE_ERROR;
        }
        uint32_t dirty = 0;
        if (!_Impl(ctx)->model.CancelGesture(&dirty)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        if (outDirty) {
            *outDirty = dirty;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_CancelGesture: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetGestureDepth(PomadeModelContext const *ctx)
{
    try {
        return ctx ? _Impl(ctx)->model.GetGestureDepth() : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_Redo(PomadeModelContext *ctx, unsigned int *outDirty)
{
    try {
        if (!ctx) {
            _SetError("Pomade_Redo: null context");
            return POMADE_ERROR;
        }
        uint32_t dirty = 0;
        if (!_Impl(ctx)->model.Redo(&dirty)) {
            _SetError(_Impl(ctx)->model.GetDiagnostic());
            return POMADE_ERROR;
        }
        if (outDirty) {
            *outDirty = dirty;
        }
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_Redo: unknown exception");
        return POMADE_ERROR;
    }
}

int
Pomade_GetRedoDepth(PomadeModelContext const *ctx)
{
    try {
        return ctx ? _Impl(ctx)->model.GetRedoDepth() : 0;
    } catch (...) {
        return 0;
    }
}

int
Pomade_GetUndoLabel(PomadeModelContext const *ctx, int depth, char *out,
                   int cap)
{
    try {
        if (!ctx || !out || cap <= 0) {
            _SetError("Pomade_GetUndoLabel: null argument");
            return POMADE_ERROR;
        }
        std::string label;
        if (!_Impl(ctx)->model.GetUndoLabel(depth, &label)) {
            _SetError("Pomade_GetUndoLabel: no step at that depth");
            return POMADE_ERROR;
        }
        if (int(label.size()) + 1 > cap) {
            _SetError("Pomade_GetUndoLabel: buffer too small");
            return POMADE_ERROR;
        }
        std::memcpy(out, label.c_str(), label.size() + 1);
        return POMADE_OK;
    } catch (std::exception const &e) {
        _SetError(e.what());
        return POMADE_ERROR;
    } catch (...) {
        _SetError("Pomade_GetUndoLabel: unknown exception");
        return POMADE_ERROR;
    }
}

} // extern "C"
