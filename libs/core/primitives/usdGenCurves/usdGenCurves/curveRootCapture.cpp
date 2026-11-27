#include "usdGenCurves/curveRootCapture.h"
#include "usdGenSurface/surfaceInput.h"
#include "usdGenCurves/types/curveSetInput.h"

#include "usdGenSurface/surfaceRootBindings.h"
#include "usdGenSurface/surfaceRootFrames.h"
#include "usdGen/registries/token.h"
#include "usdGen/api/cowArray.h"
#include "usdGenMath/math.h"
#include "usdGen/api/token.h"
#include "usdGenCurves/types/curveTokens.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>


namespace usdGen {
namespace {


bool Fail(std::string const &message, std::string *error)
{
    if (error) *error = message;
    return false;
}

bool Finite(UsdGenVec3f const &value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) &&
        std::isfinite(value[2]);
}

bool Finite(UsdGenVec2f const &value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]);
}

bool Finite(UsdGenMatrix4d const &value)
{
    for (int row = 0; row != 4; ++row)
        for (int column = 0; column != 4; ++column)
            if (!std::isfinite(value[row][column])) return false;
    return true;
}

bool ValidBindingCoordinates(int face, UsdGenVec2f const &uv)
{
    return face >= 0 && Finite(uv) && uv[0] >= 0.0f && uv[1] >= 0.0f;
}

bool ValidSurfaceBinding(UsdGenSurfaceInputView const *surface, int face,
                         UsdGenVec2f const &uv)
{
    if (!ValidBindingCoordinates(face, uv)) return false;
    if (!surface) return uv[0] <= 1.0f && uv[1] <= 1.0f;
    if (size_t(face) >= surface->faceVertexCounts.size()) return false;
    int const vertices = surface->faceVertexCounts[size_t(face)];
    if (vertices < 3) return false;
    if (vertices == 3) return uv[0] + uv[1] <= 1.0f;
    if (vertices == 4) return uv[0] <= 1.0f && uv[1] <= 1.0f;
    return double(uv[0]) < double(vertices - 2) &&
        uv[0] - std::floor(uv[0]) + uv[1] <= 1.0f;
}

bool MatrixAffine(UsdGenMatrix4d const &matrix)
{
    return Finite(matrix) && matrix[0][3] == 0.0 &&
        matrix[1][3] == 0.0 && matrix[2][3] == 0.0 &&
        matrix[3][3] == 1.0;
}

double Dot(UsdGenVec3d const &a, UsdGenVec3d const &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

UsdGenVec3d Cross(UsdGenVec3d const &a, UsdGenVec3d const &b)
{
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

UsdGenVec3d TransformVector(UsdGenVec3d const &value, UsdGenMatrix4d const &matrix)
{
    return {
        value[0] * matrix[0][0] + value[1] * matrix[1][0] + value[2] * matrix[2][0],
        value[0] * matrix[0][1] + value[1] * matrix[1][1] + value[2] * matrix[2][1],
        value[0] * matrix[0][2] + value[1] * matrix[1][2] + value[2] * matrix[2][2]};
}

bool Normalize(UsdGenVec3d const &value, UsdGenVec3d *out)
{
    double const length2 = Dot(value, value);
    if (!out || !std::isfinite(length2) || length2 <= 1.0e-24) return false;
    *out = value / std::sqrt(length2);
    return std::isfinite((*out)[0]) && std::isfinite((*out)[1]) &&
        std::isfinite((*out)[2]);
}

bool TransformPoint(UsdGenVec3d const &value, UsdGenMatrix4d const &matrix,
                    UsdGenVec3d *out)
{
    if (!out) return false;
    *out = TransformVector(value, matrix) +
        UsdGenVec3d(matrix[3][0], matrix[3][1], matrix[3][2]);
    return std::isfinite((*out)[0]) && std::isfinite((*out)[1]) &&
        std::isfinite((*out)[2]);
}

bool Add(size_t a, size_t b, size_t *out)
{
    if (!out || b > std::numeric_limits<size_t>::max() - a) return false;
    *out = a + b;
    return true;
}

} // namespace

bool UsdGenCaptureCurveRoots(
    UsdGenCurveSetInputView const &source,
    UsdGenSurfaceInputView const *surface,
    UsdGenToken rebind,
    UsdGenCurveRootCaptureResult *out,
    std::string *error,
    std::shared_ptr<const UsdGenRestSurfaceBindingCache> *bindingCache)
{
    if (error) error->clear();
    if (!out) return Fail("curve root capture output is null", error);
    if (rebind != UsdGenCurveTokens().never && rebind != UsdGenCurveTokens().onError && rebind != UsdGenCurveTokens().always)
        return Fail("curve root capture rebind policy is invalid", error);
    if (!MatrixAffine(source.worldMatrix))
        return Fail("curve source worldMatrix is non-finite or non-affine", error);

    size_t totalPoints = 0;
    for (int count : source.curveVertexCounts) {
        if (count < 2 || !Add(totalPoints, size_t(count), &totalPoints) ||
            totalPoints > std::numeric_limits<uint32_t>::max())
            return Fail("curve vertex counts are invalid or exceed uint32 cardinality", error);
    }
    size_t const curves = source.curveVertexCounts.size();
    if (curves > std::numeric_limits<uint32_t>::max() ||
        source.points.size() != totalPoints)
        return Fail("curve point cardinality disagrees with curve vertex counts", error);
    if (!source.rest.empty() && source.rest.size() != totalPoints)
        return Fail("curve rest cardinality disagrees with curve vertex counts", error);
    for (auto const &point : source.points)
        if (!Finite(point)) return Fail("curve points contain non-finite values", error);
    for (auto const &point : source.rest)
        if (!Finite(point)) return Fail("curve rest contains non-finite values", error);
    bool const bindingCardinalityMismatch =
        source.skinPrim.size() != source.skinPrimUv.size();
    if (bindingCardinalityMismatch && rebind == UsdGenCurveTokens().never)
        return Fail("curve rootPrim/rootUV cardinalities differ", error);
    if (!source.rootFrame.empty() && source.rootFrame.size() != curves)
        return Fail("curve authored rootFrame cardinality disagrees with curves", error);
    for (auto const &frame : source.rootFrame)
        if (!UsdGenValidateAuthoredRootFrame(frame))
            return Fail("curve authored rootFrame violates the affine orthonormal frame contract", error);

    UsdGenCurveRootCaptureResult candidate;
    auto candidateCache = bindingCache ? *bindingCache
        : std::shared_ptr<const UsdGenRestSurfaceBindingCache>{};
    try {
        candidate.rootPrim.assign(curves, -1);
        candidate.rootUV.assign(curves, UsdGenVec2f(0.0f));
        candidate.frames.assign(curves, UsdGenMatrix4d(0.0));
        candidate.valid.assign(curves, 0);
        candidate.dropMask.assign(curves, 0);
    } catch (...) {
        return Fail("curve root capture result allocation failed", error);
    }

    bool const authoredFrames = !source.rootFrame.empty();
    bool const authoredBindings = !bindingCardinalityMismatch &&
        source.skinPrim.size() == curves &&
        source.skinPrimUv.size() == curves;
    std::vector<uint8_t> bindingValid;
    try {
        bindingValid.assign(curves, 0);
        if (authoredBindings) {
            for (size_t curve = 0; curve != curves; ++curve)
                bindingValid[curve] = ValidSurfaceBinding(
                    surface, source.skinPrim[curve], source.skinPrimUv[curve]);
        }
    } catch (...) {
        return Fail("curve root binding validation allocation failed", error);
    }

    // Explicit surface-free CurveSource inputs may intentionally carry no
    // root channels at all.  Preserve that optional-channel absence for
    // topology-preserving operators; consumers such as Noise must impose
    // their own stronger frame requirement before use.
    if (rebind == UsdGenCurveTokens().never && curves && !surface &&
        source.skinPrim.empty() && source.skinPrimUv.empty() &&
        source.rootFrame.empty()) {
        candidate.rootPrim.clear();
        candidate.rootUV.clear();
        candidate.frames.clear();
        for (size_t curve = 0; curve != curves; ++curve)
            candidate.valid[curve] = 1;
        *out = std::move(candidate);
        return true;
    }

    // A surface-free source is legal only when it already carries complete
    // authored frames (or is empty).  Otherwise a surface is needed either to
    // repair bindings or to construct the absent rest frames.
    bool anyRepair = !authoredBindings;
    for (uint8_t valid : bindingValid) anyRepair = anyRepair || !valid;
    bool const needsBindingRepair = curves &&
        (rebind == UsdGenCurveTokens().always ||
        (rebind == UsdGenCurveTokens().onError && anyRepair));
    if (!surface && curves && (!authoredFrames || needsBindingRepair))
        return Fail("curve root capture requires a rest surface for missing or invalid bindings/frames", error);

    std::vector<UsdGenVec3f> sourceRoots;
    std::vector<size_t> repairIndices;
    if (authoredBindings && rebind != UsdGenCurveTokens().always) {
        for (size_t curve = 0; curve != curves; ++curve) {
            if (!bindingValid[curve]) continue;
            candidate.rootPrim[curve] = source.skinPrim[curve];
            candidate.rootUV[curve] = source.skinPrimUv[curve];
        }
    }
    if (needsBindingRepair) {
        if (!surface) return Fail("curve root capture cannot repair bindings without a rest surface", error);
        try {
            sourceRoots.reserve(curves);
            for (size_t curve = 0, point = 0; curve != curves; ++curve) {
                sourceRoots.push_back((source.rest.empty() ? source.points : source.rest)[point]);
                point += size_t(source.curveVertexCounts[curve]);
            }
            if (rebind == UsdGenCurveTokens().always) {
                repairIndices.resize(curves);
                for (size_t curve = 0; curve != curves; ++curve) {
                    repairIndices[curve] = curve;
                    bindingValid[curve] = 0;
                }
            } else {
                for (size_t curve = 0; curve != curves; ++curve)
                    if (!bindingValid[curve]) {
                        repairIndices.push_back(curve);
                        bindingValid[curve] = 0;
                    }
            }
        } catch (...) {
            return Fail("curve root capture repair staging allocation failed", error);
        }
        UsdGenVec3fArray roots;
        try {
            roots.reserve(repairIndices.size());
            for (size_t index : repairIndices) {
                roots.push_back(sourceRoots[index]);
            }
        } catch (...) {
            return Fail("curve root capture repair input allocation failed", error);
        }
        UsdGenSurfaceRootBindingResult bindings;
        std::string bindingError;
        if (!candidateCache || !candidateCache->Matches(*surface))
            candidateCache = UsdGenRestSurfaceBindingCache::Create(*surface,&bindingError);
        if (!candidateCache || !candidateCache->Bind(roots, source.worldMatrix,
                                                     &bindings, &bindingError))
            return Fail("curve root binding construction failed: " + bindingError, error);
        if (bindings.rootPrim.size() != repairIndices.size() ||
            bindings.rootUV.size() != repairIndices.size() ||
            bindings.valid.size() != repairIndices.size() ||
            bindings.unresolved.size() != repairIndices.size())
            return Fail("curve root binding construction returned inconsistent cardinalities", error);
        for (size_t local = 0; local != repairIndices.size(); ++local) {
            size_t const curve = repairIndices[local];
            if (!bindings.valid[local] || bindings.unresolved[local]) continue;
            candidate.rootPrim[curve] = bindings.rootPrim[local];
            candidate.rootUV[curve] = bindings.rootUV[local];
            bindingValid[curve] = ValidSurfaceBinding(
                surface, candidate.rootPrim[curve], candidate.rootUV[curve]);
            if (!bindingValid[curve])
                return Fail("curve root binding helper returned an invalid binding", error);
            ++candidate.rebound;
        }
    }

    if (authoredFrames) {
        for (size_t curve = 0; curve != curves; ++curve) {
            if (!bindingValid[curve]) continue;
            candidate.frames[curve] = source.rootFrame[curve];
            candidate.valid[curve] = 1;
        }
    } else if (curves) {
        if (!surface) return Fail("curve root capture requires a rest surface to construct frames", error);
        double sourceDeterminant = 0.0;
        UsdGenMatrix4d const sourceInverse = source.worldMatrix.GetInverse(&sourceDeterminant);
        if (!std::isfinite(sourceDeterminant) ||
            std::abs(sourceDeterminant) <= std::numeric_limits<double>::epsilon() ||
            !MatrixAffine(sourceInverse))
            return Fail("curve root capture requires an invertible source worldMatrix for frame conversion", error);
        UsdGenMatrix4d const surfaceToSource = surface->worldMatrix * sourceInverse;
        if (!MatrixAffine(surfaceToSource))
            return Fail("curve root capture produced a non-finite surface-to-source frame transform", error);
        double relativeDeterminant = 0.0;
        UsdGenMatrix4d const sourceToSurface =
            surfaceToSource.GetInverse(&relativeDeterminant);
        if (!std::isfinite(relativeDeterminant) ||
            std::abs(relativeDeterminant) <= std::numeric_limits<double>::epsilon() ||
            !MatrixAffine(sourceToSurface))
            return Fail("curve root capture requires an invertible relative frame transform", error);
        UsdGenMatrix4d normalTransform(0.0);
        for (int row = 0; row != 3; ++row)
            for (int column = 0; column != 3; ++column)
                normalTransform[row][column] = sourceToSurface[column][row];
        std::vector<size_t> frameIndices;
        UsdGenIntArray framePrim;
        UsdGenVec2fArray frameUV;
        try {
            for (size_t curve = 0; curve != curves; ++curve) {
                if (!bindingValid[curve]) continue;
                frameIndices.push_back(curve);
                framePrim.push_back(candidate.rootPrim[curve]);
                frameUV.push_back(candidate.rootUV[curve]);
            }
        } catch (...) {
            return Fail("curve root frame staging allocation failed", error);
        }
        UsdGenSurfaceRootFrameResult frames;
        std::string frameError;
        if (!framePrim.empty() && !UsdGenBuildRestSurfaceRootFrames(
                *surface, framePrim, frameUV, &frames, &frameError))
            return Fail("curve root frame construction failed: " + frameError, error);
        if (frames.frames.size() != frameIndices.size() ||
            frames.valid.size() != frameIndices.size() ||
            frames.dropMask.size() != frameIndices.size())
            return Fail("curve root frame construction returned inconsistent cardinalities", error);
        for (size_t local = 0; local != frameIndices.size(); ++local) {
            size_t const curve = frameIndices[local];
            if (!frames.valid[local] || frames.dropMask[local]) continue;
            // surfaceRootFrames returns axes/origin in surface-local space.
            // Noise and the other point operators consume source-local data.
            // Transform vectors through the relative affine, then rebuild an
            // orthonormal frame so non-uniform object scales do not leak into
            // the operator's basis; the origin receives the full transform.
            UsdGenVec3d tangent = TransformVector(
                {frames.frames[local][0][0], frames.frames[local][0][1], frames.frames[local][0][2]},
                surfaceToSource);
            UsdGenVec3d normal = TransformVector(
                {frames.frames[local][2][0], frames.frames[local][2][1], frames.frames[local][2][2]},
                normalTransform);
            if (!Normalize(normal, &normal)) continue;
            tangent -= normal * Dot(normal, tangent);
            if (!Normalize(tangent, &tangent)) continue;
            UsdGenVec3d binormal;
            if (!Normalize(Cross(normal, tangent), &binormal)) continue;
            UsdGenVec3d origin;
            if (!TransformPoint(
                    {frames.frames[local][3][0], frames.frames[local][3][1], frames.frames[local][3][2]},
                    surfaceToSource, &origin)) continue;
            UsdGenMatrix4d sourceFrame(0.0);
            for (int axis = 0; axis != 3; ++axis) {
                sourceFrame[0][axis] = tangent[axis];
                sourceFrame[1][axis] = binormal[axis];
                sourceFrame[2][axis] = normal[axis];
                sourceFrame[3][axis] = origin[axis];
            }
            sourceFrame[3][3] = 1.0;
            if (!UsdGenValidateAuthoredRootFrame(sourceFrame)) continue;
            candidate.frames[curve] = sourceFrame;
            candidate.valid[curve] = 1;
        }
    }

    for (size_t curve = 0; curve != curves; ++curve) {
        if (candidate.valid[curve]) continue;
        candidate.rootPrim[curve] = -1;
        candidate.rootUV[curve] = UsdGenVec2f(0.0f);
        candidate.frames[curve] = UsdGenMatrix4d(0.0);
        candidate.dropMask[curve] = 1;
        ++candidate.dropped;
    }
    *out = std::move(candidate);
    if (bindingCache) *bindingCache = std::move(candidateCache);
    return true;
}

} // namespace usdGen
