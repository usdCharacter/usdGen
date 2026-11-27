// Offline Ptex bake: procedural region/clump maps for a UsdGeomMesh, so
// example scenes can ship a UsdGenPtexMap without a paint package. Texels
// are written against the same Ptex face ids and sub-face corners the
// sampler resolves roots with (usdGen/maps/ptexMap.h), in the mesh's object
// space, as an mt_quad dt_float file with mipmaps and full adjacency.
//
// The prim may also be a face GeomSubset (plan/02-schema.md §2.20): the file
// still covers the whole PARENT mesh (Ptex face ids are parent-mesh ids, rule
// 2), but scattered voronoi seeds land on the subset's faces only, so its
// cells tile the region a subset-bound description grows from.
#include "usdGenSurface/maps/ptexMap.h"

#include "pxr/base/gf/range3d.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/curves.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/pointBased.h"
#include "pxr/usd/usdGeom/subset.h"
#include "pxr/usd/usdGeom/tokens.h"
#include "pxr/usd/usdGeom/xformCache.h"

// Last: <Ptexture.h> brings in <windows.h>.
#include <Ptexture.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

char const *const kUsage =
    "Usage: usdGenBakePtex <stage.usda> <meshPrimPath> <out.ptx> [options]\n"
    "  <meshPrimPath>       a Mesh, or a face GeomSubset of one: the file covers the\n"
    "                       parent mesh, scattered seeds land on the subset's faces\n"
    "  --res N              face resolution 2^N x 2^N, N in [0, 12] (default 5);\n"
    "                       n-gon sub-faces use 2^(N-1)\n"
    "  --pattern P          voronoi | stripes | constant (default voronoi)\n"
    "  --cells N            voronoi seeds scattered on the mesh (default 16)\n"
    "  --seeds-prim PATH    voronoi seeds instead: curve roots of a Curves prim, or the\n"
    "                       points of a Points/Mesh prim (moved into the mesh's space)\n"
    "  --seed S             hash seed for scattering, random values and jitter (default 0)\n"
    "  --values V           index | random (default index): the cell/stripe index, or a\n"
    "                       hash random in [0, 1) per cell/stripe (per channel)\n"
    "  --channels C         1 | 3 (default 1)\n"
    "  --jitter J           wobble the lookup position by J cell sizes (voronoi: the mean\n"
    "                       cell diameter sqrt(area / seeds); stripes: 1 / frequency)\n"
    "  --frequency F        stripes per object-space unit along x (default 1)\n"
    "  --value V            the constant pattern's value (default 0)\n";

struct Options {
    std::string stagePath, meshPath, outPath;
    int resLog2 = 5;
    std::string pattern = "voronoi";
    int cells = 16;
    std::string seedsPrim;
    uint64_t seed = 0;
    bool randomValues = false;
    int channels = 1;
    double jitter = 0.0;
    double frequency = 1.0;
    double value = 0.0;
};

double ParseDouble(std::string const &flag, char const *text)
{
    char *end = nullptr;
    double const value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value))
        throw std::runtime_error(flag + " expects a number, got '" + text + "'");
    return value;
}

long long ParseInt(std::string const &flag, char const *text, long long lo, long long hi)
{
    char *end = nullptr;
    long long const value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || value < lo || value > hi) {
        throw std::runtime_error(flag + " expects an integer in [" + std::to_string(lo) + ", " +
                                 std::to_string(hi) + "], got '" + text + "'");
    }
    return value;
}

Options ParseArgs(int argc, char **argv)
{
    Options options;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string const arg = argv[i];
        if (arg.rfind("--", 0) != 0) {
            positional.push_back(arg);
            continue;
        }
        if (i + 1 >= argc) throw std::runtime_error(arg + " needs a value");
        char const *value = argv[++i];
        if (arg == "--res") options.resLog2 = int(ParseInt(arg, value, 0, 12));
        else if (arg == "--pattern") options.pattern = value;
        else if (arg == "--cells") options.cells = int(ParseInt(arg, value, 1, 1 << 24));
        else if (arg == "--seeds-prim") options.seedsPrim = value;
        else if (arg == "--seed") options.seed = uint64_t(ParseInt(arg, value, 0, INT64_MAX));
        else if (arg == "--channels") options.channels = int(ParseInt(arg, value, 1, 3));
        else if (arg == "--jitter") options.jitter = ParseDouble(arg, value);
        else if (arg == "--frequency") options.frequency = ParseDouble(arg, value);
        else if (arg == "--value") options.value = ParseDouble(arg, value);
        else if (arg == "--values") {
            std::string const mode = value;
            if (mode != "index" && mode != "random")
                throw std::runtime_error("--values expects index or random, got '" + mode + "'");
            options.randomValues = mode == "random";
        } else {
            throw std::runtime_error("unknown option " + arg);
        }
    }
    if (positional.size() != 3) throw std::runtime_error("expected <stage> <meshPrimPath> <out.ptx>");
    options.stagePath = positional[0];
    options.meshPath = positional[1];
    options.outPath = positional[2];
    if (options.pattern != "voronoi" && options.pattern != "stripes" && options.pattern != "constant")
        throw std::runtime_error("--pattern expects voronoi, stripes or constant");
    if (options.channels == 2) throw std::runtime_error("--channels expects 1 or 3");
    if (options.jitter < 0.0) throw std::runtime_error("--jitter must be >= 0");
    if (options.pattern == "stripes" && !(options.frequency > 0.0))
        throw std::runtime_error("--frequency must be > 0");
    return options;
}

// ---------------------------------------------------------------------------
// Hashing and noise (deterministic across hosts)
// ---------------------------------------------------------------------------

uint64_t Mix(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// Uniform in [0, 1).
double HashUnit(uint64_t seed, uint64_t stream, uint64_t index)
{
    return double(Mix(Mix(Mix(seed) ^ stream) ^ index) >> 11) * 0x1.0p-53;
}

// Smooth value noise in [-1, 1] on the unit lattice.
double ValueNoise(uint64_t seed, uint64_t stream, GfVec3d const &p)
{
    double const fx = std::floor(p[0]), fy = std::floor(p[1]), fz = std::floor(p[2]);
    auto const smooth = [](double t) { return t * t * (3.0 - 2.0 * t); };
    double const tx = smooth(p[0] - fx), ty = smooth(p[1] - fy), tz = smooth(p[2] - fz);
    int64_t const ix = int64_t(fx), iy = int64_t(fy), iz = int64_t(fz);
    auto const lattice = [&](int64_t x, int64_t y, int64_t z) {
        uint64_t const h = Mix(Mix(Mix(Mix(seed ^ (stream * 0x2545f4914f6cdd1dull)) ^ uint64_t(x)) ^
                                   uint64_t(y)) ^ uint64_t(z));
        return double(h >> 11) * 0x1.0p-52 - 1.0;
    };
    double result = 0.0;
    for (int corner = 0; corner < 8; ++corner) {
        int const dx = corner & 1, dy = (corner >> 1) & 1, dz = (corner >> 2) & 1;
        double const w = (dx ? tx : 1.0 - tx) * (dy ? ty : 1.0 - ty) * (dz ? tz : 1.0 - tz);
        result += w * lattice(ix + dx, iy + dy, iz + dz);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Nearest seed
// ---------------------------------------------------------------------------

// A static balanced k-d tree; ties go to the lower seed index.
class KdTree {
public:
    explicit KdTree(std::vector<GfVec3d> const &points) : points_(points), order_(points.size()),
                                                          axis_(points.size(), 0)
    {
        for (size_t i = 0; i < order_.size(); ++i) order_[i] = int(i);
        Build(0, int(order_.size()));
    }

    int Nearest(GfVec3d const &p) const
    {
        int best = -1;
        double bestD2 = std::numeric_limits<double>::infinity();
        Search(0, int(order_.size()), p, &best, &bestD2);
        return best;
    }

private:
    void Build(int lo, int hi)
    {
        if (hi - lo <= 1) return;
        GfRange3d bounds;
        for (int i = lo; i < hi; ++i) bounds.UnionWith(points_[size_t(order_[size_t(i)])]);
        GfVec3d const size = bounds.GetSize();
        int const axis = size[0] >= size[1] && size[0] >= size[2] ? 0 : (size[1] >= size[2] ? 1 : 2);
        int const mid = (lo + hi) / 2;
        std::nth_element(order_.begin() + lo, order_.begin() + mid, order_.begin() + hi,
                         [&](int a, int b) {
                             double const pa = points_[size_t(a)][axis], pb = points_[size_t(b)][axis];
                             return pa < pb || (pa == pb && a < b);
                         });
        axis_[size_t(mid)] = int8_t(axis);
        Build(lo, mid);
        Build(mid + 1, hi);
    }

    void Search(int lo, int hi, GfVec3d const &p, int *best, double *bestD2) const
    {
        if (lo >= hi) return;
        int const mid = (lo + hi) / 2;
        int const index = order_[size_t(mid)];
        GfVec3d const &q = points_[size_t(index)];
        double const d2 = (q - p).GetLengthSq();
        if (d2 < *bestD2 || (d2 == *bestD2 && index < *best)) {
            *bestD2 = d2;
            *best = index;
        }
        int const axis = axis_[size_t(mid)];
        double const diff = p[axis] - q[axis];
        bool const leftFirst = diff < 0.0;
        if (leftFirst) Search(lo, mid, p, best, bestD2);
        else Search(mid + 1, hi, p, best, bestD2);
        if (diff * diff <= *bestD2) {
            if (leftFirst) Search(mid + 1, hi, p, best, bestD2);
            else Search(lo, mid, p, best, bestD2);
        }
    }

    std::vector<GfVec3d> const &points_;
    std::vector<int> order_;
    std::vector<int8_t> axis_;
};

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

struct Mesh {
    std::vector<float> points;  // xyz interleaved
    std::vector<int> counts, indices, offsets, firstIds;
    int ptexFaceCount = 0;

    size_t PointCount() const { return points.size() / 3; }
    size_t FaceCount() const { return counts.size(); }
    GfVec3d Point(int i) const
    {
        return GfVec3d(points[3 * size_t(i)], points[3 * size_t(i) + 1], points[3 * size_t(i) + 2]);
    }
};

Mesh ReadMesh(UsdGeomMesh const &usdMesh)
{
    VtVec3fArray points;
    VtIntArray counts, indices;
    UsdTimeCode const time = UsdTimeCode::Default();
    if (!usdMesh.GetPointsAttr().Get(&points, time) ||
        !usdMesh.GetFaceVertexCountsAttr().Get(&counts, time) ||
        !usdMesh.GetFaceVertexIndicesAttr().Get(&indices, time) || points.empty() ||
        counts.empty()) {
        throw std::runtime_error("the mesh needs points, faceVertexCounts and faceVertexIndices");
    }
    Mesh mesh;
    mesh.points.reserve(points.size() * 3);
    for (GfVec3f const &p : points) {
        mesh.points.insert(mesh.points.end(), {p[0], p[1], p[2]});
    }
    mesh.counts.assign(counts.begin(), counts.end());
    mesh.indices.assign(indices.begin(), indices.end());
    mesh.offsets.assign(1, 0);
    for (int count : mesh.counts) {
        if (count < 3) throw std::runtime_error("the mesh has a face with fewer than 3 vertices");
        mesh.offsets.push_back(mesh.offsets.back() + count);
    }
    if (size_t(mesh.offsets.back()) != mesh.indices.size())
        throw std::runtime_error("faceVertexCounts do not sum to the faceVertexIndices size");
    for (int index : mesh.indices) {
        if (index < 0 || size_t(index) >= points.size())
            throw std::runtime_error("faceVertexIndices reference a point that does not exist");
    }
    mesh.firstIds = usdGen::UsdGenPtexFirstFaceIds(mesh.counts.data(), mesh.counts.size(),
                                                   &mesh.ptexFaceCount);
    if (mesh.ptexFaceCount < 0) throw std::runtime_error("too many Ptex faces");
    return mesh;
}

// One Ptex face: a quad, or sub-face `subface` of an n-gon.
struct PtexFace {
    int face = 0;
    int subface = 0;
    Ptex::FaceInfo info;
    float corners[12] = {};
};

// Adjacency exactly as OpenSubdiv's Far::PtexIndices::GetAdjacency derives it
// from the coarse topology: manifold edges only, sub-faces of the same face
// adjacent across their inner edges, and sub-face <-> regular face across a
// half edge.
class Adjacency {
public:
    explicit Adjacency(Mesh const &mesh) : mesh_(mesh)
    {
        for (int f = 0; f < int(mesh.FaceCount()); ++f) {
            int const n = mesh.counts[size_t(f)];
            for (int e = 0; e < n; ++e) uses_[Key(f, e)].push_back({f, e});
        }
    }

    void Resolve(int face, int quadrant, int adjFaces[4], int adjEdges[4]) const
    {
        int const n = mesh_.counts[size_t(face)];
        int const first = mesh_.firstIds[size_t(face)];
        if (n == 4) {
            for (int e = 0; e < 4; ++e) {
                Use const other = Across(face, e);
                if (other.face < 0) {
                    adjFaces[e] = -1;
                    adjEdges[e] = 0;
                } else if (mesh_.counts[size_t(other.face)] == 4) {
                    adjFaces[e] = mesh_.firstIds[size_t(other.face)];
                    adjEdges[e] = other.edge;
                } else {
                    adjFaces[e] = mesh_.firstIds[size_t(other.face)] +
                                  (other.edge + 1) % mesh_.counts[size_t(other.face)];
                    adjEdges[e] = 3;
                }
            }
            return;
        }
        int const next = (quadrant + 1) % n, prev = (quadrant + n - 1) % n;
        adjFaces[1] = first + next;
        adjEdges[1] = 2;
        adjFaces[2] = first + prev;
        adjEdges[2] = 1;
        // Edge 0 is the first half of coarse edge `quadrant`, edge 3 the
        // second half of coarse edge `prev`.
        for (int side : {0, 3}) {
            Use const other = Across(face, side == 0 ? quadrant : prev);
            if (other.face < 0) {
                adjFaces[side] = -1;
                adjEdges[side] = 0;
            } else if (mesh_.counts[size_t(other.face)] == 4) {
                adjFaces[side] = mesh_.firstIds[size_t(other.face)];
                adjEdges[side] = other.edge;
            } else if (side == 0) {
                adjFaces[side] = mesh_.firstIds[size_t(other.face)] +
                                 (other.edge + 1) % mesh_.counts[size_t(other.face)];
                adjEdges[side] = 3;
            } else {
                adjFaces[side] = mesh_.firstIds[size_t(other.face)] + other.edge;
                adjEdges[side] = 0;
            }
        }
    }

private:
    struct Use {
        int face, edge;
    };

    uint64_t Key(int face, int edge) const
    {
        int const n = mesh_.counts[size_t(face)];
        int const *corners = &mesh_.indices[size_t(mesh_.offsets[size_t(face)])];
        uint64_t const a = uint32_t(corners[edge]), b = uint32_t(corners[(edge + 1) % n]);
        return a < b ? (a << 32) | b : (b << 32) | a;
    }

    Use Across(int face, int edge) const
    {
        auto const found = uses_.find(Key(face, edge));
        if (found == uses_.end() || found->second.size() != 2) return {-1, 0};
        Use const &a = found->second[0];
        return a.face == face && a.edge == edge ? found->second[1] : a;
    }

    Mesh const &mesh_;
    std::unordered_map<uint64_t, std::vector<Use>> uses_;
};

std::vector<PtexFace> PtexFaces(Mesh const &mesh, int resLog2)
{
    std::vector<PtexFace> faces(size_t(mesh.ptexFaceCount));
    Adjacency const adjacency(mesh);
    for (int f = 0; f < int(mesh.FaceCount()); ++f) {
        int const n = mesh.counts[size_t(f)];
        int const subfaces = n == 4 ? 1 : n;
        for (int k = 0; k < subfaces; ++k) {
            PtexFace &out = faces[size_t(mesh.firstIds[size_t(f)] + k)];
            out.face = f;
            out.subface = k;
            int adjFaces[4], adjEdges[4];
            adjacency.Resolve(f, k, adjFaces, adjEdges);
            int8_t const faceLog2 = int8_t(n == 4 ? resLog2 : std::max(resLog2 - 1, 0));
            out.info = Ptex::FaceInfo(Ptex::Res(faceLog2, faceLog2), adjFaces, adjEdges, n != 4);
            if (!usdGen::UsdGenPtexFaceCorners(mesh.points.data(), mesh.PointCount(),
                                               mesh.counts.data(), mesh.indices.data(),
                                               mesh.offsets.data(), mesh.FaceCount(), f, k,
                                               out.corners)) {
                throw std::runtime_error("cannot build Ptex face " + std::to_string(f));
            }
        }
    }
    return faces;
}

// Centroid-fan triangles of every face, for area and surface scattering.
struct Surface {
    std::vector<GfVec3d> a, b, c;
    std::vector<double> cumulativeArea;
    double area = 0.0;
};

// `only` (a GeomSubset's faces; empty = every face) limits the triangles, so
// the area and the scattered seeds cover those faces alone.
Surface Triangulate(Mesh const &mesh, std::vector<uint8_t> const &only)
{
    Surface surface;
    for (int f = 0; f < int(mesh.FaceCount()); ++f) {
        if (!only.empty() && !only[size_t(f)]) continue;
        int const n = mesh.counts[size_t(f)];
        int const *corners = &mesh.indices[size_t(mesh.offsets[size_t(f)])];
        GfVec3d centroid(0.0);
        for (int k = 0; k < n; ++k) centroid += mesh.Point(corners[k]);
        centroid /= double(n);
        for (int k = 0; k < n; ++k) {
            GfVec3d const p = mesh.Point(corners[k]), q = mesh.Point(corners[(k + 1) % n]);
            surface.area += 0.5 * GfCross(p - centroid, q - centroid).GetLength();
            surface.a.push_back(centroid);
            surface.b.push_back(p);
            surface.c.push_back(q);
            surface.cumulativeArea.push_back(surface.area);
        }
    }
    return surface;
}

std::vector<GfVec3d> ScatterSeeds(Surface const &surface, int count, uint64_t seed)
{
    if (!(surface.area > 0.0))
        throw std::runtime_error("the mesh (or subset) has no area to scatter seeds on");
    std::vector<GfVec3d> seeds;
    seeds.reserve(size_t(count));
    for (int i = 0; i < count; ++i) {
        double const pick = HashUnit(seed, 1, uint64_t(i)) * surface.area;
        size_t const t = std::min(
            size_t(std::upper_bound(surface.cumulativeArea.begin(), surface.cumulativeArea.end(), pick) -
                   surface.cumulativeArea.begin()),
            surface.cumulativeArea.size() - 1);
        double const r1 = std::sqrt(HashUnit(seed, 2, uint64_t(i)));
        double const r2 = HashUnit(seed, 3, uint64_t(i));
        seeds.push_back(surface.a[t] * (1.0 - r1) + surface.b[t] * (r1 * (1.0 - r2)) +
                        surface.c[t] * (r1 * r2));
    }
    return seeds;
}

std::vector<GfVec3d> ReadSeeds(UsdStageRefPtr const &stage, std::string const &primPath,
                               UsdPrim const &meshPrim)
{
    UsdPrim const prim = stage->GetPrimAtPath(SdfPath(primPath));
    if (!prim) throw std::runtime_error("no seeds prim at " + primPath);
    UsdTimeCode const time = UsdTimeCode::Default();
    VtVec3fArray points;
    std::vector<GfVec3f> local;
    if (UsdGeomCurves const curves{prim}) {
        VtIntArray counts;
        if (!curves.GetPointsAttr().Get(&points, time) ||
            !curves.GetCurveVertexCountsAttr().Get(&counts, time)) {
            throw std::runtime_error(primPath + " has no points/curveVertexCounts");
        }
        size_t offset = 0;
        for (int count : counts) {
            if (count <= 0 || offset + size_t(count) > points.size())
                throw std::runtime_error(primPath + " has inconsistent curveVertexCounts");
            local.push_back(points[offset]);
            offset += size_t(count);
        }
    } else if (UsdGeomPointBased const pointBased{prim}) {
        if (!pointBased.GetPointsAttr().Get(&points, time))
            throw std::runtime_error(primPath + " has no points");
        local.assign(points.begin(), points.end());
    } else {
        throw std::runtime_error(primPath + " is not a Curves, Points or Mesh prim");
    }
    if (local.empty()) throw std::runtime_error(primPath + " provides no seed positions");

    UsdGeomXformCache xforms(time);
    GfMatrix4d const toMesh = xforms.GetLocalToWorldTransform(prim) *
                              xforms.GetLocalToWorldTransform(meshPrim).GetInverse();
    std::vector<GfVec3d> seeds;
    seeds.reserve(local.size());
    for (GfVec3f const &p : local) seeds.push_back(toMesh.Transform(GfVec3d(p)));
    return seeds;
}

// ---------------------------------------------------------------------------
// Texels
// ---------------------------------------------------------------------------

struct Pattern {
    Options const *options = nullptr;
    KdTree const *tree = nullptr;  // voronoi only
    double wobbleScale = 0.0;      // object-space size one jitter unit stands for

    // The region index at object-space position p.
    int64_t Index(GfVec3d p) const
    {
        if (options->jitter > 0.0 && wobbleScale > 0.0) {
            // Lattice at half the scale, so a border wobbles about twice per cell.
            GfVec3d const q = p / (0.5 * wobbleScale);
            double const amplitude = options->jitter * wobbleScale;
            p += GfVec3d(ValueNoise(options->seed, 11, q), ValueNoise(options->seed, 12, q),
                         ValueNoise(options->seed, 13, q)) * amplitude;
        }
        if (tree) return tree->Nearest(p);
        return int64_t(std::floor(p[0] * options->frequency));
    }

    void Value(int64_t index, float *out) const
    {
        for (int c = 0; c < options->channels; ++c) {
            out[c] = options->randomValues
                         ? float(HashUnit(options->seed, 0x5eed0 + uint64_t(c), uint64_t(index)))
                         : float(index);
        }
    }
};

// Fills one Ptex face; `used` collects the region indices it touched.
void Bake(PtexFace const &face, Pattern const &pattern, std::vector<float> *texels,
          std::set<int64_t> *used)
{
    int const resU = face.info.res.u(), resV = face.info.res.v();
    int const channels = pattern.options->channels;
    texels->resize(size_t(resU) * size_t(resV) * size_t(channels));
    float const *c = face.corners;
    bool haveLast = false;
    int64_t last = 0;
    for (int j = 0; j < resV; ++j) {
        double const v = (j + 0.5) / resV;
        for (int i = 0; i < resU; ++i) {
            double const u = (i + 0.5) / resU;
            double const w0 = (1 - u) * (1 - v), w1 = u * (1 - v), w2 = u * v, w3 = (1 - u) * v;
            GfVec3d const p(w0 * c[0] + w1 * c[3] + w2 * c[6] + w3 * c[9],
                            w0 * c[1] + w1 * c[4] + w2 * c[7] + w3 * c[10],
                            w0 * c[2] + w1 * c[5] + w2 * c[8] + w3 * c[11]);
            int64_t const index = pattern.Index(p);
            if (!haveLast || index != last) {
                used->insert(index);
                last = index;
                haveLast = true;
            }
            pattern.Value(index, texels->data() + (size_t(j) * size_t(resU) + size_t(i)) * size_t(channels));
        }
    }
}

}  // namespace

int main(int argc, char **argv) try {
    Options options;
    try {
        options = ParseArgs(argc, argv);
    } catch (std::exception const &e) {
        std::fprintf(stderr, "ERROR: %s\n%s", e.what(), kUsage);
        return 2;
    }
    auto const start = std::chrono::steady_clock::now();

    UsdStageRefPtr const stage = UsdStage::Open(options.stagePath);
    if (!stage) throw std::runtime_error("cannot open stage " + options.stagePath);
    UsdPrim const target = stage->GetPrimAtPath(SdfPath(options.meshPath));
    if (!target) throw std::runtime_error("no prim at " + options.meshPath);
    // A face GeomSubset bakes its parent mesh (§2.20 rules 1-2).
    UsdPrim meshPrim = target;
    VtIntArray subsetIndices;
    bool const isSubset = bool(UsdGeomSubset(target));
    if (isSubset) {
        UsdGeomSubset const subset(target);
        TfToken elementType;
        subset.GetElementTypeAttr().Get(&elementType);
        if (elementType != UsdGeomTokens->face) {
            throw std::runtime_error("GeomSubset " + options.meshPath + " has elementType '" +
                                     elementType.GetString() + "'; bake a \"face\" subset");
        }
        meshPrim = target.GetParent();
        if (!UsdGeomMesh(meshPrim))
            throw std::runtime_error("GeomSubset " + options.meshPath + " is not the child of a Mesh");
        subset.GetIndicesAttr().Get(&subsetIndices);
    }
    UsdGeomMesh const usdMesh(meshPrim);
    if (!usdMesh) throw std::runtime_error("no Mesh or face GeomSubset prim at " + options.meshPath);
    Mesh const mesh = ReadMesh(usdMesh);
    std::vector<uint8_t> inSubset;
    size_t subsetFaceCount = 0;
    if (isSubset) {
        inSubset.assign(mesh.FaceCount(), 0);
        for (int index : subsetIndices) {
            if (index < 0 || size_t(index) >= mesh.FaceCount()) {
                throw std::runtime_error("GeomSubset " + options.meshPath + " names face " +
                                         std::to_string(index) + ", but its parent mesh has " +
                                         std::to_string(mesh.FaceCount()) + " faces");
            }
            if (!inSubset[size_t(index)]) ++subsetFaceCount;
            inSubset[size_t(index)] = 1;
        }
        if (subsetFaceCount == 0 && options.pattern == "voronoi" && options.seedsPrim.empty())
            throw std::runtime_error("GeomSubset " + options.meshPath + " names no faces to scatter seeds on");
    }
    std::vector<PtexFace> const faces = PtexFaces(mesh, options.resLog2);

    Pattern pattern;
    pattern.options = &options;
    std::vector<GfVec3d> seeds;
    std::unique_ptr<KdTree> tree;
    if (options.pattern == "voronoi") {
        Surface const surface = Triangulate(mesh, inSubset);
        seeds = options.seedsPrim.empty() ? ScatterSeeds(surface, options.cells, options.seed)
                                          : ReadSeeds(stage, options.seedsPrim, meshPrim);
        tree = std::make_unique<KdTree>(seeds);
        pattern.tree = tree.get();
        pattern.wobbleScale = std::sqrt(surface.area / double(seeds.size()));
    } else if (options.pattern == "stripes") {
        pattern.wobbleScale = 1.0 / options.frequency;
    }

    Ptex::String error;
    PtexWriter *writer = PtexWriter::open(options.outPath.c_str(), Ptex::mt_quad, Ptex::dt_float,
                                          options.channels, /*alphachan*/ -1,
                                          mesh.ptexFaceCount, error, /*genmipmaps*/ true);
    if (!writer) throw std::runtime_error("cannot create " + options.outPath + ": " + error.c_str());
    writer->writeMeta("usdGen:bakePtex:pattern", options.pattern.c_str());
    writer->writeMeta("usdGen:bakePtex:source", (options.stagePath + " " + options.meshPath).c_str());

    bool ok = true;
    size_t texelCount = 0;
    std::set<int64_t> used;
    if (options.pattern == "constant") {
        float value[3];
        for (int c = 0; c < options.channels; ++c) value[c] = float(options.value);
        for (size_t id = 0; id < faces.size(); ++id) {
            ok = writer->writeConstantFace(int(id), faces[id].info, value) && ok;
            texelCount += size_t(faces[id].info.res.size());
        }
        used.insert(0);
    } else {
        // Bake a batch of faces in parallel, then write it in id order
        // (PtexWriter is single-threaded).
        size_t const threadCount = std::max<size_t>(1, std::thread::hardware_concurrency());
        size_t const batch = threadCount * 16;
        std::vector<std::vector<float>> texels(batch);
        std::vector<std::set<int64_t>> usedPerThread(threadCount);
        for (size_t begin = 0; begin < faces.size() && ok; begin += batch) {
            size_t const end = std::min(faces.size(), begin + batch);
            std::vector<std::thread> workers;
            for (size_t t = 0; t < threadCount; ++t) {
                workers.emplace_back([&, t] {
                    for (size_t id = begin + t; id < end; id += threadCount)
                        Bake(faces[id], pattern, &texels[id - begin], &usedPerThread[t]);
                });
            }
            for (auto &worker : workers) worker.join();
            for (size_t id = begin; id < end; ++id) {
                ok = writer->writeFace(int(id), faces[id].info, texels[id - begin].data()) && ok;
                texelCount += size_t(faces[id].info.res.size());
            }
        }
        for (auto const &set : usedPerThread) used.insert(set.begin(), set.end());
    }
    ok = writer->close(error) && ok;
    writer->release();
    if (!ok) throw std::runtime_error("cannot write " + options.outPath + ": " + error.c_str());

    std::string cells;
    if (options.pattern == "voronoi")
        cells = std::to_string(used.size()) + " of " + std::to_string(seeds.size()) + " cells";
    else if (options.pattern == "stripes")
        cells = std::to_string(used.size()) + " stripes";
    else
        cells = "constant";
    std::string const scope =
        isSubset ? " (subset: " + std::to_string(subsetFaceCount) + " faces)" : "";
    std::printf("usdGenBakePtex: %s: %zu mesh faces%s -> %d Ptex faces, %zu texels, %s (%.2f ms)\n",
                options.outPath.c_str(), mesh.FaceCount(), scope.c_str(), mesh.ptexFaceCount,
                texelCount, cells.c_str(),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count());
    return 0;
} catch (std::exception const &e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
}
