#include "usdGen/maps/ptexMap.h"

// Ptexture.h comes from the private usdGen_ptex archive (PTEX_STATIC,
// PTEX_VENDOR=usdGen); only this translation unit sees it. On ELF/Mach-O its
// declarations are made hidden, so the Ptex vtables and inline code
// instantiated here stay out of libusdGen.so's dynamic table (gate B-1,
// cmake/CheckNoThirdPartyExports.cmake). The system headers it pulls in are
// included first so they keep their own visibility.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ostream>
#if defined(__GNUC__) && !defined(_WIN32)
#  include <pthread.h>
#  include <sys/types.h>
#  if defined(__APPLE__)
#    include <os/lock.h>
#    include <unistd.h>
#  elif !defined(__FreeBSD__)
#    include <alloca.h>
#  endif
#  pragma GCC visibility push(hidden)
#endif
#include <Ptexture.h>
#if defined(__GNUC__) && !defined(_WIN32)
#  pragma GCC visibility pop
#endif

#include "pxr/pxr.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/tf/getenv.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

// ---------------------------------------------------------------------------
// Process cache
// ---------------------------------------------------------------------------

// Reader errors after a successful open (a truncated block, a vanished file)
// reach this handler; the default one writes to std::cerr.
class UsdGenPtexErrorHandler final : public PtexErrorHandler {
public:
    void reportError(char const *error) override { TF_WARN("Ptex: %s", error); }
};

int EnvInt(char const *name, int fallback, int minimum)
{
    std::string const value = TfGetenv(name);
    if (value.empty()) return fallback;
    char *end = nullptr;
    long const parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed < minimum ||
        parsed > std::numeric_limits<int>::max()) {
        TF_WARN("Ignoring %s='%s' (expected an integer >= %d)", name, value.c_str(), minimum);
        return fallback;
    }
    return static_cast<int>(parsed);
}

UsdGenPtexErrorHandler g_ptexErrorHandler;
std::atomic<PtexCache *> g_processCache{nullptr};

// Deliberately never released: textures and samplers may outlive static
// destruction, and a cache torn down under them would be a use-after-free.
// Published once by CAS: a thread that loses the race releases its own cache
// and uses the winner's, so nobody waits.
PtexCache *ProcessCache()
{
    if (PtexCache *cache = g_processCache.load(std::memory_order_acquire)) return cache;
    int const maxFiles = EnvInt("USDGEN_PTEX_MAX_FILES", 32, 1);
    int const cacheMB = EnvInt("USDGEN_PTEX_CACHE_MB", 256, 0);
    PtexCache *fresh = PtexCache::create(maxFiles, size_t(cacheMB) << 20,
                                         /*premultiply*/ false, /*inputHandler*/ nullptr,
                                         &g_ptexErrorHandler);
    PtexCache *winner = nullptr;
    if (g_processCache.compare_exchange_strong(winner, fresh, std::memory_order_acq_rel,
                                               std::memory_order_acquire))
        return fresh;
    fresh->release();
    return winner;
}

struct FilterName {
    char const *token;
    PtexFilter::FilterType type;
};
constexpr FilterName kFilters[] = {
    {"nearest", PtexFilter::f_point},       {"bilinear", PtexFilter::f_bilinear},
    {"box", PtexFilter::f_box},             {"gaussian", PtexFilter::f_gaussian},
    {"bicubic", PtexFilter::f_bicubic},     {"bspline", PtexFilter::f_bspline},
    {"catmullrom", PtexFilter::f_catmullrom}, {"mitchell", PtexFilter::f_mitchell},
};

bool ParseFilter(std::string const &token, PtexFilter::FilterType *type)
{
    for (auto const &entry : kFilters) {
        if (token == entry.token) { *type = entry.type; return true; }
    }
    return false;
}

std::shared_ptr<const UsdGenPtexTexture> Fail(std::string *error, std::string message)
{
    if (error) *error = std::move(message);
    return nullptr;
}

// ---------------------------------------------------------------------------
// Face geometry
// ---------------------------------------------------------------------------

struct Vec3 {
    double x, y, z;
};
Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double Dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 Cross(Vec3 a, Vec3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double Length(Vec3 a) { return std::sqrt(Dot(a, a)); }

struct Vec2 {
    double x, y;
};
Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
Vec2 operator*(Vec2 a, double s) { return {a.x * s, a.y * s}; }
double Dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
double Cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }

// How far (u, v) lies outside the unit square, 0 inside.
double Outside(double u, double v)
{
    return std::max({0.0, -u, u - 1.0, -v, v - 1.0});
}

// A validated view of one coarse face.
struct Face {
    float const *points;
    int const *indices;
    int count;

    Vec3 Corner(int i) const
    {
        float const *p = points + 3 * size_t(indices[i]);
        return {p[0], p[1], p[2]};
    }
    Vec3 Centroid() const
    {
        Vec3 sum{0.0, 0.0, 0.0};
        for (int i = 0; i < count; ++i) sum = sum + Corner(i);
        return sum * (1.0 / count);
    }
    // Corners of sub-face k, in Ptex order (see ptexMap.h).
    void SubfaceCorners(int k, Vec3 const &centroid, Vec3 out[4]) const
    {
        Vec3 const corner = Corner(k);
        out[0] = corner;
        out[1] = (corner + Corner((k + 1) % count)) * 0.5;
        out[2] = centroid;
        out[3] = (corner + Corner((k + count - 1) % count)) * 0.5;
    }
};

bool GetFace(float const *points, size_t pointCount, int const *faceVertexCounts,
             int const *faceVertexIndices, int const *faceOffsets, size_t faceCount,
             int face, Face *out)
{
    if (!points || !faceVertexCounts || !faceVertexIndices || !faceOffsets) return false;
    if (face < 0 || size_t(face) >= faceCount) return false;
    int const count = faceVertexCounts[face];
    int const begin = faceOffsets[face];
    if (count < 3 || begin < 0 || faceOffsets[face + 1] - begin != count) return false;
    int const *indices = faceVertexIndices + begin;
    for (int i = 0; i < count; ++i) {
        if (indices[i] < 0 || size_t(indices[i]) >= pointCount) return false;
    }
    *out = {points, indices, count};
    return true;
}

// Unclamped (u, v) of p on the bilinear patch c[0..3] (Ptex corner order).
// The patch and p are projected onto the plane through c[0] normal to the
// patch's Newell normal and scaled by the patch's larger diagonal, so every
// tolerance below is relative. Of the (up to) two roots the one closest to
// the unit square wins, then two Newton steps absorb the closed form's
// rounding. Returns false for a degenerate patch.
bool InverseBilinear(Vec3 const c[4], Vec3 const &p, double *uOut, double *vOut)
{
    Vec3 normal{0.0, 0.0, 0.0};
    for (int i = 0; i < 4; ++i) {
        Vec3 const a = c[i];
        Vec3 const b = c[(i + 1) % 4];
        normal.x += (a.y - b.y) * (a.z + b.z);
        normal.y += (a.z - b.z) * (a.x + b.x);
        normal.z += (a.x - b.x) * (a.y + b.y);
    }
    double const scale = std::max(Length(c[2] - c[0]), Length(c[3] - c[1]));
    double const normalLength = Length(normal);
    if (!std::isfinite(scale) || !std::isfinite(normalLength) || !(scale > 0.0) ||
        !(normalLength > 1e-12 * scale * scale)) {
        return false;
    }
    normal = normal * (1.0 / normalLength);

    // Any tangent frame works; build it from the axis least aligned with n.
    Vec3 axis{0.0, 0.0, 0.0};
    double const ax = std::fabs(normal.x), ay = std::fabs(normal.y), az = std::fabs(normal.z);
    if (ax <= ay && ax <= az) axis.x = 1.0;
    else if (ay <= az) axis.y = 1.0;
    else axis.z = 1.0;
    Vec3 tangent = Cross(normal, axis);
    tangent = tangent * (1.0 / Length(tangent));
    Vec3 const bitangent = Cross(normal, tangent);

    double const invScale = 1.0 / scale;
    auto project = [&](Vec3 const &x) {
        Vec3 const d = x - c[0];
        return Vec2{Dot(d, tangent) * invScale, Dot(d, bitangent) * invScale};
    };
    // p(u, v) = u e + v f + u v g in the plane, with c[0] at the origin.
    Vec2 const e = project(c[1]);
    Vec2 const f = project(c[3]);
    Vec2 const g = project(c[2]) - e - f;
    Vec2 const h = project(p);
    if (!std::isfinite(h.x) || !std::isfinite(h.y)) return false;

    // Eliminating u gives k2 v^2 + k1 v + k0 = 0.
    double const k2 = Cross(g, f);
    double const k1 = Cross(e, f) + Cross(h, g);
    double const k0 = Cross(h, e);
    double roots[2];
    int rootCount = 0;
    double const discriminant = std::max(0.0, k1 * k1 - 4.0 * k0 * k2);
    double const q = -0.5 * (k1 + std::copysign(std::sqrt(discriminant), k1));
    if (std::fabs(k2) > 1e-12) roots[rootCount++] = q / k2;
    if (q != 0.0) roots[rootCount++] = k0 / q;

    bool found = false;
    double bestU = 0.0, bestV = 0.0, bestOutside = std::numeric_limits<double>::infinity();
    for (int i = 0; i < rootCount; ++i) {
        double const v = roots[i];
        Vec2 const du = e + g * v;
        double const du2 = Dot(du, du);
        if (!std::isfinite(v) || !(du2 > 1e-24)) continue;
        double const u = Dot(h - f * v, du) / du2;
        double const outside = Outside(u, v);
        if (std::isfinite(u) && outside < bestOutside) {
            found = true;
            bestU = u;
            bestV = v;
            bestOutside = outside;
        }
    }
    if (!found) return false;

    for (int iteration = 0; iteration < 2; ++iteration) {
        Vec2 const residual = e * bestU + f * bestV + g * (bestU * bestV) - h;
        Vec2 const ju = e + g * bestV;
        Vec2 const jv = f + g * bestU;
        double const det = Cross(ju, jv);
        if (!(std::fabs(det) > 1e-18)) break;
        double const u = bestU - Cross(residual, jv) / det;
        double const v = bestV - Cross(ju, residual) / det;
        if (!std::isfinite(u) || !std::isfinite(v)) break;
        Vec2 const next = e * u + f * v + g * (u * v) - h;
        if (Dot(next, next) > Dot(residual, residual)) break;
        bestU = u;
        bestV = v;
    }
    *uOut = bestU;
    *vOut = bestV;
    return true;
}

float Clamp01(double value)
{
    return static_cast<float>(std::min(1.0, std::max(0.0, value)));
}

bool TriangleCoordinate(Face const &face, Vec3 const &p, float *uOut, float *vOut)
{
    Vec3 const p0 = face.Corner(0);
    Vec3 const e1 = face.Corner(1) - p0;
    Vec3 const e2 = face.Corner(2) - p0;
    Vec3 const h = p - p0;
    // Least squares in the triangle's plane: the Gram system of (e1, e2).
    double const a = Dot(e1, e1), b = Dot(e1, e2), d = Dot(e2, e2);
    double const det = a * d - b * b;
    if (!std::isfinite(det) || !(det > 1e-12 * a * d)) return false;
    double const r1 = Dot(h, e1), r2 = Dot(h, e2);
    double u = (d * r1 - b * r2) / det;
    double v = (a * r2 - b * r1) / det;
    if (!std::isfinite(u) || !std::isfinite(v)) return false;
    u = std::max(0.0, u);
    v = std::max(0.0, v);
    if (u + v > 1.0) {
        double const sum = u + v;
        u /= sum;
        v /= sum;
    }
    *uOut = static_cast<float>(u);
    *vOut = static_cast<float>(v);
    return true;
}

// Every member that names a Ptex type lives in these internal-linkage
// structs: Ptex is declared hidden on ELF, and GCC warns (-Wattributes) when a
// default-visibility type such as UsdGenPtexTexture::State has a field of a
// hidden type.
struct PtexTextureHandle {
    PtexPtr<PtexTexture> texture;
    PtexFilter::Options filterOptions;
};

struct PtexFilterHandle {
    PtexPtr<PtexFilter> filter;
};

}  // namespace

// ---------------------------------------------------------------------------
// UsdGenPtexTexture
// ---------------------------------------------------------------------------

struct UsdGenPtexTexture::State {
    PtexTextureHandle ptex;
    std::string path;
    float blur = 0.0f;
    int firstChannel = 0;
    int sampleChannels = 1;
    int numFaces = 0;
    int numChannels = 0;
    bool triangleMesh = false;
};

struct UsdGenPtexTexture::Sampler::Impl {
    std::shared_ptr<const UsdGenPtexTexture::State> state;
    PtexFilterHandle ptex;
};

UsdGenPtexTexture::UsdGenPtexTexture(std::shared_ptr<const State> state)
    : state_(std::move(state))
{
}

UsdGenPtexTexture::~UsdGenPtexTexture() = default;

std::shared_ptr<const UsdGenPtexTexture> UsdGenPtexTexture::Open(
    std::string const &path, UsdGenPtexMapOptions const &options, std::string *error)
{
    try {
        PtexFilter::FilterType filterType = PtexFilter::f_bilinear;
        if (!ParseFilter(options.filter, &filterType)) {
            return Fail(error, "unknown Ptex filter '" + options.filter +
                                   "' (expected nearest, bilinear, box, gaussian, bicubic, "
                                   "bspline, catmullrom or mitchell)");
        }
        if (options.borderMode != "clamp" && options.borderMode != "black" &&
            options.borderMode != "periodic") {
            return Fail(error, "unknown Ptex border mode '" + options.borderMode +
                                   "' (expected clamp, black or periodic)");
        }
        if (!std::isfinite(options.blur) || options.blur < 0.0f) {
            return Fail(error, "Ptex blur must be finite and >= 0");
        }
        if (options.channelCount < 1) {
            return Fail(error, "Ptex channelCount must be >= 1");
        }
        if (path.empty()) return Fail(error, "empty Ptex file path");

        PtexCache *cache = ProcessCache();
        if (!cache) return Fail(error, "the Ptex cache could not be created");

        Ptex::String ptexError;
        PtexTexture *texture = cache->get(path.c_str(), ptexError);
        if (!texture && ptexError.empty()) {
            // The cache remembers a failed path and then fails silently.
            // Forget it and retry once, so the caller gets the reason (and a
            // file that has appeared since opens).
            cache->purge(path.c_str());
            texture = cache->get(path.c_str(), ptexError);
        }
        if (!texture) {
            std::string reason = ptexError.c_str();
            while (!reason.empty() && (reason.back() == '\n' || reason.back() == '\r'))
                reason.pop_back();
            if (reason.empty()) reason = "unknown error";
            return Fail(error, "cannot open Ptex file '" + path + "': " + reason);
        }

        auto state = std::make_shared<State>();
        state->ptex.texture.reset(texture);
        Ptex::MeshType const meshType = texture->meshType();
        state->path = path;
        state->numFaces = texture->numFaces();
        state->numChannels = texture->numChannels();
        state->triangleMesh = meshType == Ptex::mt_triangle;
        if (options.firstChannel < 0 || options.firstChannel >= state->numChannels) {
            return Fail(error, "Ptex firstChannel " + std::to_string(options.firstChannel) +
                                   " is outside the " + std::to_string(state->numChannels) +
                                   " channel(s) of '" + path + "'");
        }
        state->firstChannel = options.firstChannel;
        state->sampleChannels =
            std::min(options.channelCount, state->numChannels - options.firstChannel);
        state->blur = options.blur;
        state->ptex.filterOptions = PtexFilter::Options(
            filterType, /*lerp*/ true, /*sharpness*/ 0.0f, /*noedgeblend*/ false);

        // A filter is built here once to prove the file/filter pair works; the
        // samplers each build their own.
        PtexPtr<PtexFilter> probe(PtexFilter::getFilter(texture, state->ptex.filterOptions));
        if (!probe) {
            return Fail(error, "Ptex file '" + path + "' has an unsupported mesh type");
        }
        return std::shared_ptr<const UsdGenPtexTexture>(new UsdGenPtexTexture(std::move(state)));
    } catch (std::exception const &e) {
        return Fail(error, std::string("Ptex open failed: ") + e.what());
    } catch (...) {
        return Fail(error, "Ptex open failed");
    }
}

int UsdGenPtexTexture::NumFaces() const { return state_->numFaces; }
int UsdGenPtexTexture::NumChannels() const { return state_->numChannels; }
int UsdGenPtexTexture::SampleChannels() const { return state_->sampleChannels; }
bool UsdGenPtexTexture::IsTriangleMesh() const { return state_->triangleMesh; }
std::string const &UsdGenPtexTexture::Path() const { return state_->path; }

std::unique_ptr<UsdGenPtexTexture::Sampler> UsdGenPtexTexture::MakeSampler() const
{
    auto impl = std::make_unique<Sampler::Impl>();
    impl->state = state_;
    impl->ptex.filter.reset(
        PtexFilter::getFilter(state_->ptex.texture.get(), state_->ptex.filterOptions));
    if (!impl->ptex.filter) return nullptr;
    return std::unique_ptr<Sampler>(new Sampler(std::move(impl)));
}

void UsdGenPtexTexture::PurgeCache()
{
    if (PtexCache *cache = ProcessCache()) cache->purgeAll();
}

UsdGenPtexTexture::Sampler::Sampler(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

UsdGenPtexTexture::Sampler::~Sampler() = default;

bool UsdGenPtexTexture::Sampler::Sample(int faceId, float u, float v, float *out) const
{
    State const &state = *impl_->state;
    if (!out || faceId < 0 || faceId >= state.numFaces) return false;
    if (!std::isfinite(u) || !std::isfinite(v)) return false;
    u = std::min(1.0f, std::max(0.0f, u));
    v = std::min(1.0f, std::max(0.0f, v));
    Ptex::Res const res = state.ptex.texture->getFaceInfo(faceId).res;
    float const uw = 1.0f / static_cast<float>(res.u());
    float const vw = 1.0f / static_cast<float>(res.v());
    // A filter that cannot read the face data leaves the result unwritten.
    std::fill(out, out + state.sampleChannels, 0.0f);
    impl_->ptex.filter->eval(out, state.firstChannel, state.sampleChannels, faceId, u, v,
                             /*uw1*/ uw, /*vw1*/ 0.0f, /*uw2*/ 0.0f, /*vw2*/ vw,
                             /*width*/ 1.0f, state.blur);
    return true;
}

// ---------------------------------------------------------------------------
// Face ids and coordinates
// ---------------------------------------------------------------------------

std::vector<int> UsdGenPtexFirstFaceIds(int const *faceVertexCounts, size_t faceCount,
                                        int *totalOut)
{
    std::vector<int> firstIds;
    if (faceCount && !faceVertexCounts) {
        if (totalOut) *totalOut = -1;
        return firstIds;
    }
    firstIds.resize(faceCount);
    int64_t next = 0;
    for (size_t face = 0; face < faceCount; ++face) {
        firstIds[face] = static_cast<int>(next);
        int const count = faceVertexCounts[face];
        next += count == 4 ? 1 : std::max(count, 0);
        if (next > std::numeric_limits<int>::max()) {
            if (totalOut) *totalOut = -1;
            return {};
        }
    }
    if (totalOut) *totalOut = static_cast<int>(next);
    return firstIds;
}

bool UsdGenPtexFaceCoordinate(float const *points, size_t pointCount,
                              int const *faceVertexCounts, int const *faceVertexIndices,
                              int const *faceOffsets, size_t faceCount, int const *firstIds,
                              bool triangleMesh, int face, float px, float py, float pz,
                              int *ptexFaceId, float *u, float *v)
{
    if (!ptexFaceId || !u || !v) return false;
    Face coarse;
    if (!GetFace(points, pointCount, faceVertexCounts, faceVertexIndices, faceOffsets,
                 faceCount, face, &coarse)) {
        return false;
    }
    Vec3 const p{px, py, pz};
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) return false;

    if (triangleMesh) {
        if (coarse.count != 3 || int64_t(faceOffsets[faceCount]) != 3 * int64_t(faceCount))
            return false;
        float fu = 0.0f, fv = 0.0f;
        if (!TriangleCoordinate(coarse, p, &fu, &fv)) return false;
        *ptexFaceId = face;
        *u = fu;
        *v = fv;
        return true;
    }

    if (!firstIds || firstIds[face] < 0) return false;
    double pu = 0.0, pv = 0.0;
    if (coarse.count == 4) {
        Vec3 const corners[4] = {coarse.Corner(0), coarse.Corner(1), coarse.Corner(2),
                                 coarse.Corner(3)};
        if (!InverseBilinear(corners, p, &pu, &pv)) return false;
        *ptexFaceId = firstIds[face];
        *u = Clamp01(pu);
        *v = Clamp01(pv);
        return true;
    }

    // Visit the sub-faces nearest corner first; the first that contains p
    // wins, otherwise the one p is least outside of.
    int const n = coarse.count;
    std::pair<double, int> inlineOrder[16];
    std::vector<std::pair<double, int>> heapOrder;
    std::pair<double, int> *order = inlineOrder;
    if (n > 16) {
        heapOrder.resize(size_t(n));
        order = heapOrder.data();
    }
    for (int k = 0; k < n; ++k) {
        Vec3 const d = coarse.Corner(k) - p;
        order[k] = {Dot(d, d), k};
    }
    std::sort(order, order + n);
    Vec3 const centroid = coarse.Centroid();
    constexpr double kInside = 1e-6;
    int bestK = -1;
    double bestU = 0.0, bestV = 0.0, bestOutside = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        int const k = order[i].second;
        Vec3 corners[4];
        coarse.SubfaceCorners(k, centroid, corners);
        double su = 0.0, sv = 0.0;
        if (!InverseBilinear(corners, p, &su, &sv)) continue;
        double const outside = Outside(su, sv);
        if (outside < bestOutside) {
            bestK = k;
            bestU = su;
            bestV = sv;
            bestOutside = outside;
        }
        if (outside <= kInside) break;
    }
    if (bestK < 0) return false;
    *ptexFaceId = firstIds[face] + bestK;
    *u = Clamp01(bestU);
    *v = Clamp01(bestV);
    return true;
}

bool UsdGenPtexFaceCorners(float const *points, size_t pointCount,
                           int const *faceVertexCounts, int const *faceVertexIndices,
                           int const *faceOffsets, size_t faceCount, int face, int subface,
                           float *corners)
{
    if (!corners) return false;
    Face coarse;
    if (!GetFace(points, pointCount, faceVertexCounts, faceVertexIndices, faceOffsets,
                 faceCount, face, &coarse)) {
        return false;
    }
    Vec3 quad[4];
    if (coarse.count == 4) {
        if (subface != 0) return false;
        for (int i = 0; i < 4; ++i) quad[i] = coarse.Corner(i);
    } else {
        if (subface < 0 || subface >= coarse.count) return false;
        coarse.SubfaceCorners(subface, coarse.Centroid(), quad);
    }
    for (int i = 0; i < 4; ++i) {
        corners[3 * i] = static_cast<float>(quad[i].x);
        corners[3 * i + 1] = static_cast<float>(quad[i].y);
        corners[3 * i + 2] = static_cast<float>(quad[i].z);
    }
    return true;
}

}  // namespace usdGen
