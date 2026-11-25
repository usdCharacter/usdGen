// Ptex map sampling, and the mapping from a coarse mesh face to Ptex face ids
// and face-local (u, v).
//
// usdGen vendors Ptex 2.4.3 (thirdparty/ptex, the private archive
// usdGen_ptex) because the OpenUSD install has no Ptex: PXR_ENABLE_PTEX_SUPPORT
// is off, so a .ptx reaching a Storm material is silently a 1x1 black texture.
// Even with it on, Storm's Ptex accessor needs the mesh-only GetPatchCoord(),
// so Ptex on curves cannot work there. Ptex maps are therefore sampled on the
// CPU at capture time and baked into curve primvars
// (plan/07-look-maps-expressions.md §6).
//
// Ptex stays private to libusdGen: this header does not include Ptexture.h
// and has no USD dependency.
#ifndef USDGEN_MAPS_PTEX_MAP_H
#define USDGEN_MAPS_PTEX_MAP_H

#include "usdGen/export.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace usdGen {

// The UsdGenPtexMap attributes that shape a lookup (02-schema.md, 07 §5.1).
struct UsdGenPtexMapOptions {
    // nearest | bilinear | box | gaussian | bicubic | bspline | catmullrom |
    // mitchell, 1:1 onto PtexFilter::FilterType (bicubic uses sharpness 0).
    std::string filter = "bilinear";
    // Added to the one-texel footprint (the blur argument of PtexFilter::eval).
    float blur = 0.0f;
    int firstChannel = 0;
    // Clamped to the channels the file has from firstChannel on.
    int channelCount = 1;
    // clamp | black | periodic. Informational: a Ptex file carries its own
    // border modes, which the filters apply. Unknown tokens are still rejected.
    std::string borderMode = "clamp";
};

// One opened .ptx file plus the lookup options it was opened with. Immutable
// after Open, so one instance is shared (shared_ptr) across capture workers;
// each worker samples through its own Sampler.
//
// All files go through one process-wide PtexCache, created on first use:
// 32 open files and 256 MB of texel data unless USDGEN_PTEX_MAX_FILES /
// USDGEN_PTEX_CACHE_MB say otherwise (0 MB = unlimited), never premultiplied.
// A texture keeps its file's data referenced while it is alive, so the cache
// can only prune files nobody holds.
class USDGEN_CORE_API UsdGenPtexTexture final {
public:
    // Returns null and sets *error (when non-null) if the file cannot be
    // opened, or the options are invalid: an unknown filter or border token,
    // a negative or non-finite blur, channelCount < 1, or firstChannel outside
    // the file's channels.
    static std::shared_ptr<const UsdGenPtexTexture> Open(
        std::string const &path, UsdGenPtexMapOptions const &options,
        std::string *error);

    ~UsdGenPtexTexture();
    UsdGenPtexTexture(UsdGenPtexTexture const &) = delete;
    UsdGenPtexTexture &operator=(UsdGenPtexTexture const &) = delete;

    int NumFaces() const;
    // Channels stored in the file.
    int NumChannels() const;
    // Channels one Sample() writes: the clamped firstChannel/channelCount window.
    int SampleChannels() const;
    // True for an mt_triangle file (see UsdGenPtexFaceCoordinate).
    bool IsTriangleMesh() const;
    std::string const &Path() const;

    // A per-thread lookup. PtexFilter keeps per-eval scratch, so a Sampler must
    // not be used by two threads at once; make one per worker instead. A
    // Sampler keeps its texture alive.
    class USDGEN_CORE_API Sampler final {
    public:
        ~Sampler();
        Sampler(Sampler const &) = delete;
        Sampler &operator=(Sampler const &) = delete;

        // Writes SampleChannels() floats to `out`. The footprint is one texel
        // of the face (uw1 = 1/res.u, vw2 = 1/res.v, vw1 = uw2 = 0, width 1)
        // widened by the options' blur; u and v are clamped to [0, 1].
        // Returns false, leaving `out` untouched, for a face id outside the
        // file, a non-finite u or v, or a null `out`.
        bool Sample(int faceId, float u, float v, float *out) const;

    private:
        friend class UsdGenPtexTexture;
        struct Impl;
        explicit Sampler(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };

    std::unique_ptr<Sampler> MakeSampler() const;

    // The explicit reload hook: purges the process cache so the next Open
    // re-reads files from disk. A file still held by a live texture or sampler
    // keeps serving its old data and is purged once the last holder is gone.
    static void PurgeCache();

private:
    struct State;
    explicit UsdGenPtexTexture(std::shared_ptr<const State> state);
    std::shared_ptr<const State> state_;
};

// Ptex face ids of a coarse polygon mesh under the quad-mesh convention of
// OpenSubdiv's Far::PtexIndices: a quad occupies one id, any other n-gon n
// consecutive ids (one sub-face per corner). Returns the first id of each
// face; *totalOut (when non-null) receives the number of ids. Faces with
// fewer than three vertices are invalid topology and are not diagnosed here.
// Returns an empty vector and a total of -1 if the ids overflow int.
USDGEN_CORE_API std::vector<int> UsdGenPtexFirstFaceIds(
    int const *faceVertexCounts, size_t faceCount, int *totalOut = nullptr);

// Locates the object-space point p on coarse face `face` in Ptex space.
//
//   points              xyz interleaved, 3 * pointCount floats
//   faceVertexCounts,   the USD mesh topology; faceVertexIndices is the
//   faceVertexIndices   concatenation of the per-face corner lists
//   faceOffsets         start of each face in faceVertexIndices, faceCount + 1
//                       entries
//   firstIds            UsdGenPtexFirstFaceIds (unused when triangleMesh)
//
// Quads: inverse bilinear on the corners with Ptex's convention p(0,0) = p0,
// p(1,0) = p1, p(1,1) = p2, p(0,1) = p3, solved in the face's tangent plane
// (Newell normal), so mildly non-planar quads work; (u, v) is clamped to
// [0, 1].
//
// Other n-gons: the face is split into n sub-quads. Sub-face k spans corner k,
// the midpoint of edge (k, k+1), the vertex centroid and the midpoint of edge
// (k-1, k), in that order: u runs toward the next corner, v toward the
// previous one. The point goes to the sub-face that contains it (the one
// closest to containing it when it lies off the face) and the id is
// firstIds[face] + k.
//
// triangleMesh (an mt_triangle file): valid only when every face is a
// triangle. The id is the face index and p = p0 + u (p1 - p0) + v (p2 - p0),
// clamped to the triangle. That (u, v) <-> corner convention is Ptex's
// documented one but is not yet verified against a painted file (07 §6.3).
// The all-triangle check is O(1): faceOffsets[faceCount] must equal
// 3 * faceCount, which together with valid (>= 3 vertex) faces means every
// face is a triangle.
//
// Returns false for an out-of-range face, inconsistent topology, an index
// outside the points, or degenerate geometry.
USDGEN_CORE_API bool UsdGenPtexFaceCoordinate(
    float const *points, size_t pointCount,
    int const *faceVertexCounts, int const *faceVertexIndices,
    int const *faceOffsets, size_t faceCount, int const *firstIds,
    bool triangleMesh, int face, float px, float py, float pz,
    int *ptexFaceId, float *u, float *v);

// The object-space corners p(0,0), p(1,0), p(1,1), p(0,1) (xyz interleaved,
// 12 floats) of one Ptex face of coarse face `face`: the face itself for a
// quad (subface must be 0), otherwise sub-face `subface` as defined above.
// Evaluating these bilinearly at (u, v) inverts UsdGenPtexFaceCoordinate.
// Returns false for invalid input.
USDGEN_CORE_API bool UsdGenPtexFaceCorners(
    float const *points, size_t pointCount,
    int const *faceVertexCounts, int const *faceVertexIndices,
    int const *faceOffsets, size_t faceCount, int face, int subface,
    float *corners);

}  // namespace usdGen

#endif  // USDGEN_MAPS_PTEX_MAP_H
