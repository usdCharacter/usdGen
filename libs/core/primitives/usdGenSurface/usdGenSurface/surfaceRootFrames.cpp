#include "usdGen/surfaceRootFrames.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace {

constexpr double kEpsilon = 1.0e-6;

bool Fail(std::string const &message, std::string *error)
{
    if (error) *error = message;
    return false;
}
bool Finite(double value) { return std::isfinite(value); }
bool Finite(GfVec2f const &v) { return Finite(v[0]) && Finite(v[1]); }
bool Finite(GfVec3f const &v) { return Finite(v[0]) && Finite(v[1]) && Finite(v[2]); }
bool Finite(GfVec3d const &v) { return Finite(v[0]) && Finite(v[1]) && Finite(v[2]); }
bool Finite(GfMatrix4d const &m) {
    for (int row = 0; row != 4; ++row)
        for (int column = 0; column != 4; ++column)
            if (!Finite(m[row][column])) return false;
    return true;
}
GfVec3d ToDouble(GfVec3f const &v) { return {v[0], v[1], v[2]}; }
double Dot(GfVec3d const &a, GfVec3d const &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
GfVec3d Cross(GfVec3d const &a, GfVec3d const &b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
bool Normalize(GfVec3d const &input, GfVec3d *out) {
    double const length2 = Dot(input, input);
    if (!Finite(length2) || length2 < kEpsilon * kEpsilon) return false;
    *out = input / std::sqrt(length2);
    return Finite(*out);
}

struct Patch {
    std::vector<uint32_t> corners; // indices into surface faceVertexIndices
    double u = 0.0, v = 0.0;
    bool quad = false;
};

bool MakePatch(int faceCount, uint32_t faceBegin, GfVec2f const &uv,
               Patch *out, std::string *error)
{
    if (!Finite(uv) || uv[0] < 0.0f || uv[1] < 0.0f)
        return Fail("root binding has non-finite or negative skinprimuv", error);
    Patch patch;
    patch.v = uv[1];
    if (faceCount == 3) {
        patch.u = uv[0];
        // Bindings are float-valued; match native CUDA's domain check at
        // rounded boundaries such as (.6f,.4f), whose double sum exceeds 1.
        if (uv[0] + uv[1] > 1.0f)
            return Fail("triangle skinprimuv is outside its barycentric face", error);
        patch.corners = {faceBegin, faceBegin + 1, faceBegin + 2};
    } else if (faceCount == 4) {
        patch.u = uv[0];
        if (patch.u > 1.0f || patch.v > 1.0f)
            return Fail("quad skinprimuv is outside its bilinear face", error);
        patch.quad = true;
        patch.corners = {faceBegin, faceBegin + 1, faceBegin + 2, faceBegin + 3};
    } else {
        // Plan 05 §3.3: the integer part chooses an Imager-compatible fan
        // triangle [0, fan+1, fan+2]; the fractional part is that triangle's
        // barycentric u.  `v` remains the second barycentric coordinate.
        double const encoded = uv[0];
        if (encoded >= double(faceCount - 2))
            return Fail("n-gon skinprimuv fan index is out of range", error);
        int const fan = static_cast<int>(std::floor(encoded));
        patch.u = encoded - double(fan);
        if (fan < 0 || fan >= faceCount - 2 || float(patch.u) + uv[1] > 1.0f)
            return Fail("n-gon skinprimuv does not name a valid fan triangle", error);
        patch.corners = {faceBegin, faceBegin + uint32_t(fan + 1),
                         faceBegin + uint32_t(fan + 2)};
    }
    *out = std::move(patch);
    return true;
}

GfVec3d Interpolate(std::vector<GfVec3d> const &p, Patch const &patch)
{
    if (!patch.quad) {
        double const w = 1.0 - patch.u - patch.v;
        return p[0] * w + p[1] * patch.u + p[2] * patch.v;
    }
    double const a = (1.0 - patch.u) * (1.0 - patch.v);
    double const b = patch.u * (1.0 - patch.v);
    double const c = patch.u * patch.v;
    double const d = (1.0 - patch.u) * patch.v;
    return p[0] * a + p[1] * b + p[2] * c + p[3] * d;
}
GfVec3d DpDu(std::vector<GfVec3d> const &p, Patch const &patch)
{
    if (!patch.quad) return p[1] - p[0];
    return (p[1] - p[0]) * (1.0 - patch.v) + (p[2] - p[3]) * patch.v;
}
GfVec3d GeometricNormal(std::vector<GfVec3d> const &p, Patch const &patch)
{
    GfVec3d n = Cross(p[1] - p[0], p[2] - p[0]);
    if (patch.quad) n += Cross(p[2] - p[0], p[3] - p[0]);
    return n;
}

bool NormalShape(UsdGenSurfaceDesc const &surface, size_t indexCount,
                 std::string *error)
{
    size_t expected = 0;
    switch (surface.restNormalDomain) {
    case UsdGenSurfaceNormalDomain::None:
        if (!surface.restNormals.empty()) return Fail("rest normal domain None has values", error);
        return true;
    case UsdGenSurfaceNormalDomain::Constant: expected = 1; break;
    case UsdGenSurfaceNormalDomain::Uniform: expected = surface.faceVertexCounts.size(); break;
    case UsdGenSurfaceNormalDomain::Vertex: expected = surface.restPoints.size(); break;
    case UsdGenSurfaceNormalDomain::FaceVarying: expected = indexCount; break;
    case UsdGenSurfaceNormalDomain::Invalid:
        return Fail("rest normal domain is invalid", error);
    }
    if (surface.restNormals.size() != expected)
        return Fail("rest normals cardinality does not match its interpolation", error);
    for (GfVec3f const &normal : surface.restNormals)
        if (!Finite(normal)) return Fail("rest normals contain non-finite values", error);
    return true;
}

GfVec3d AuthoredNormal(UsdGenSurfaceDesc const &surface, size_t face,
                        Patch const &patch, std::vector<GfVec3d> const &points,
                        std::vector<uint32_t> const &indices)
{
    switch (surface.restNormalDomain) {
    case UsdGenSurfaceNormalDomain::Constant: return ToDouble(surface.restNormals[0]);
    case UsdGenSurfaceNormalDomain::Uniform: return ToDouble(surface.restNormals[face]);
    case UsdGenSurfaceNormalDomain::Vertex: {
        std::vector<GfVec3d> normals;
        normals.reserve(patch.corners.size());
        for (uint32_t corner : patch.corners)
            normals.push_back(ToDouble(surface.restNormals[indices[corner]]));
        return Interpolate(normals, patch);
    }
    case UsdGenSurfaceNormalDomain::FaceVarying: {
        std::vector<GfVec3d> normals;
        normals.reserve(patch.corners.size());
        for (uint32_t corner : patch.corners) normals.push_back(ToDouble(surface.restNormals[corner]));
        return Interpolate(normals, patch);
    }
    default: return GeometricNormal(points, patch);
    }
}

} // namespace

bool UsdGenValidateAuthoredRootFrame(GfMatrix4d const &m)
{
    if (!Finite(m) || m[0][3] != 0.0 || m[1][3] != 0.0 ||
        m[2][3] != 0.0 || m[3][3] != 1.0) return false;
    // gpu/rootFrames.cu publishes float axes/origin.  A finite double that
    // overflows during this conversion is no more a valid authored frame on
    // CPU than it is on CUDA.
    for (int row = 0; row != 4; ++row)
        for (int column = 0; column != 3; ++column)
            if (!std::isfinite(static_cast<float>(m[row][column]))) return false;
    GfVec3d const t(m[0][0], m[0][1], m[0][2]);
    GfVec3d const b(m[1][0], m[1][1], m[1][2]);
    GfVec3d const n(m[2][0], m[2][1], m[2][2]);
    constexpr double tolerance = 1.0e-8;
    return std::abs(Dot(t, t) - 1.0) <= tolerance &&
           std::abs(Dot(b, b) - 1.0) <= tolerance &&
           std::abs(Dot(n, n) - 1.0) <= tolerance &&
           std::abs(Dot(t, b)) <= tolerance && std::abs(Dot(t, n)) <= tolerance &&
           std::abs(Dot(b, n)) <= tolerance && Dot(Cross(t, b), n) >= 1.0 - tolerance;
}

bool UsdGenBuildRestSurfaceRootFrames(
    UsdGenSurfaceDesc const &surface, VtIntArray const &rootPrim,
    VtVec2fArray const &rootUV, UsdGenSurfaceRootFrameResult *out,
    std::string *error)
{
    if (error) error->clear();
    if (!out) return Fail("root-frame output is null", error);
    if (surface.restFromCurrentPoints || surface.restPoints.empty())
        return Fail("rest root frames require authored/default-time surface rest points", error);
    // Surface points and the frame remain surface-local.  worldMatrix is
    // output/imaging transport only (Plan 05 §4.6), so it is validated but
    // never baked into the local frame.
    if (!Finite(surface.worldMatrix))
        return Fail("surface worldMatrix contains non-finite values", error);
    if (rootPrim.size() != rootUV.size())
        return Fail("rootPrim and rootUV cardinalities differ", error);
    if (rootPrim.size() > std::numeric_limits<uint32_t>::max())
        return Fail("root-frame curve cardinality exceeds uint32", error);
    for (GfVec3f const &point : surface.restPoints)
        if (!Finite(point)) return Fail("surface rest points contain non-finite values", error);

    size_t indexCount = 0;
    for (int count : surface.faceVertexCounts) {
        if (count < 3 || indexCount > std::numeric_limits<size_t>::max() - size_t(count) ||
            indexCount > size_t(std::numeric_limits<uint32_t>::max()) - size_t(count))
            return Fail("surface faceVertexCounts has an invalid face", error);
        indexCount += size_t(count);
    }
    if (surface.faceVertexIndices.size() != indexCount)
        return Fail("surface faceVertexIndices cardinality disagrees with faceVertexCounts", error);
    for (int index : surface.faceVertexIndices)
        if (index < 0 || size_t(index) >= surface.restPoints.size())
            return Fail("surface faceVertexIndices contains an out-of-range vertex", error);
    if (!NormalShape(surface, indexCount, error)) return false;

    UsdGenSurfaceRootFrameResult candidate;
    try {
        size_t const curves = rootPrim.size();
        candidate.frames.assign(curves, GfMatrix4d(0.0));
        candidate.valid.assign(curves, 0);
        candidate.dropMask.assign(curves, 0);
        std::vector<uint32_t> offsets(surface.faceVertexCounts.size() + 1, 0);
        for (size_t face = 0; face != surface.faceVertexCounts.size(); ++face)
            offsets[face + 1] = offsets[face] + uint32_t(surface.faceVertexCounts[face]);
        std::vector<uint32_t> indices(indexCount);
        for (size_t i = 0; i != indexCount; ++i)
            indices[i] = uint32_t(surface.faceVertexIndices[i]);

        for (size_t curve = 0; curve != curves; ++curve) {
            int const faceValue = rootPrim[curve];
            if (faceValue < 0 || size_t(faceValue) >= surface.faceVertexCounts.size())
                return Fail("rootPrim contains a parent-face index out of range", error);
            size_t const face = size_t(faceValue);
            Patch patch;
            if (!MakePatch(surface.faceVertexCounts[face], offsets[face], rootUV[curve], &patch, error))
                return false;
            std::vector<GfVec3d> points;
            points.reserve(patch.corners.size());
            for (uint32_t corner : patch.corners)
                points.push_back(ToDouble(surface.restPoints[indices[corner]]));
            GfVec3d const geometric = GeometricNormal(points, patch);
            GfVec3d const raw = surface.restNormalDomain == UsdGenSurfaceNormalDomain::None
                ? geometric : AuthoredNormal(surface, face, patch, points, indices);
            GfVec3d n;
            GfVec3d tangent = DpDu(points, patch);
            if (!Normalize(raw, &n)) {
                candidate.dropMask[curve] = 1; ++candidate.dropped; continue;
            }
            tangent -= n * Dot(n, tangent);
            if (!Normalize(tangent, &tangent)) {
                uint32_t const edge0 = offsets[face];
                GfVec3d fallback = ToDouble(surface.restPoints[indices[edge0 + 1]]) -
                    ToDouble(surface.restPoints[indices[edge0]]);
                fallback -= n * Dot(n, fallback);
                if (!Normalize(fallback, &tangent)) {
                    candidate.dropMask[curve] = 1; ++candidate.dropped; continue;
                }
            }
            GfVec3d b;
            if (!Normalize(Cross(n, tangent), &b)) {
                candidate.dropMask[curve] = 1; ++candidate.dropped; continue;
            }
            GfVec3d const origin = Interpolate(points, patch);
            if (!Finite(origin)) { candidate.dropMask[curve] = 1; ++candidate.dropped; continue; }
            GfMatrix4d frame(0.0);
            for (int axis = 0; axis != 3; ++axis) {
                frame[0][axis] = tangent[axis]; frame[1][axis] = b[axis]; frame[2][axis] = n[axis];
                frame[3][axis] = origin[axis];
            }
            frame[3][3] = 1.0;
            candidate.frames[curve] = frame;
            candidate.valid[curve] = 1;
        }
    } catch (...) {
        return Fail("root-frame construction allocation failed", error);
    }
    *out = std::move(candidate);
    return true;
}

} // namespace usdGen
