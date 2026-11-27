// Ptex maps: Ptex face ids and face-local (u, v) under the quad-mesh
// convention, and the sampler over a .ptx authored here with PtexWriter.
//
// The face-id rule is OpenSubdiv's Far::PtexIndices (a quad is one id, an
// n-gon n ids); the coordinates must invert bilinear interpolation of the
// corners UsdGenPtexFaceCorners reports, which is exactly what the bake tool
// writes texels against.
//
// Usage: testUsdGenPtexMap [file.ptx]. With a path, the test only prints the
// distinct nearest-sampled values of that file per face (a smoke check for
// usdGenBakePtex output) and exits.
#include "usdGenSurface/maps/ptexMap.h"

// The test authors its fixture through the vendored library directly.
#include <Ptexture.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what, int line)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL (line %d): %s\n", line, what.c_str());
    }
}
// Explicit check: the build may define NDEBUG, so assert() proves nothing.
#define CHECK(cond, what) Check(static_cast<bool>(cond), what, __LINE__)

bool Near(double a, double b, double tolerance) { return std::fabs(a - b) <= tolerance; }

struct P3 {
    double x, y, z;
};

constexpr double kPi = 3.14159265358979323846;

// Rotates by 0.7 rad about (1, 1, 1) / sqrt(3) (Rodrigues) and offsets, so no
// test face lies in a coordinate plane. Local z stays the face normal.
P3 Place(double x, double y, double z)
{
    double const c = std::cos(0.7), s = std::sin(0.7);
    double const k = 1.0 / std::sqrt(3.0);
    double const along = (x + y + z) / 3.0 * (1.0 - c);  // k (k . p) (1 - cos)
    double const crossX = k * (z - y), crossY = k * (x - z), crossZ = k * (y - x);
    return {x * c + crossX * s + along + 3.0,
            y * c + crossY * s + along - 1.0,
            z * c + crossZ * s + along + 0.5};
}

struct Mesh {
    std::vector<float> points;
    std::vector<int> counts, indices, offsets, firstIds;
    int total = 0;

    int AddPoint(P3 p)
    {
        points.push_back(float(p.x));
        points.push_back(float(p.y));
        points.push_back(float(p.z));
        return int(points.size() / 3 - 1);
    }
    void AddFace(std::vector<P3> const &corners)
    {
        counts.push_back(int(corners.size()));
        for (auto const &corner : corners) indices.push_back(AddPoint(corner));
    }
    void Finish()
    {
        offsets.assign(1, 0);
        for (int count : counts) offsets.push_back(offsets.back() + count);
        firstIds = UsdGenPtexFirstFaceIds(counts.data(), counts.size(), &total);
    }
    bool Locate(int face, P3 p, int *id, float *u, float *v, bool triangleMesh = false) const
    {
        return UsdGenPtexFaceCoordinate(points.data(), points.size() / 3, counts.data(),
                                        indices.data(), offsets.data(), counts.size(),
                                        firstIds.data(), triangleMesh, face, float(p.x),
                                        float(p.y), float(p.z), id, u, v);
    }
    bool Corners(int face, int subface, float corners[12]) const
    {
        return UsdGenPtexFaceCorners(points.data(), points.size() / 3, counts.data(),
                                     indices.data(), offsets.data(), counts.size(), face,
                                     subface, corners);
    }
    P3 Corner(int face, int i) const
    {
        float const *p = &points[3 * size_t(indices[size_t(offsets[size_t(face)] + i)])];
        return {p[0], p[1], p[2]};
    }
};

P3 Bilinear(float const c[12], double u, double v)
{
    double const w[4] = {(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v};
    P3 p{0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        p.x += w[i] * c[3 * i];
        p.y += w[i] * c[3 * i + 1];
        p.z += w[i] * c[3 * i + 2];
    }
    return p;
}

std::string Uv(float u, float v)
{
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "(%.6f, %.6f)", u, v);
    return buffer;
}

// quad, triangle, pentagon, quad (a trapezoid).
Mesh MixedMesh()
{
    Mesh mesh;
    mesh.AddFace({Place(0, 0, 0), Place(1, 0, 0), Place(1, 1, 0), Place(0, 1, 0)});
    mesh.AddFace({Place(2, 0, 0), Place(3, 0, 0), Place(2.2, 1, 0)});
    std::vector<P3> pentagon;
    for (int k = 0; k < 5; ++k) {
        double const angle = 0.3 + 2.0 * kPi * k / 5.0;
        pentagon.push_back(Place(5 + std::cos(angle), std::sin(angle), 0));
    }
    mesh.AddFace(pentagon);
    mesh.AddFace({Place(0, 2, 0), Place(4, 2, 0), Place(3, 4, 0), Place(1, 4, 0)});
    mesh.Finish();
    return mesh;
}

void CheckFirstFaceIds()
{
    Mesh const mesh = MixedMesh();
    CHECK((mesh.firstIds == std::vector<int>{0, 1, 4, 9}), "first ids of quad/tri/pentagon/quad");
    CHECK(mesh.total == 10, "id total of quad/tri/pentagon/quad");

    int total = -7;
    CHECK(UsdGenPtexFirstFaceIds(nullptr, 0, &total).empty() && total == 0, "empty mesh");
    int const allQuads[] = {4, 4, 4};
    CHECK((UsdGenPtexFirstFaceIds(allQuads, 3, &total) == std::vector<int>{0, 1, 2}) &&
              total == 3, "all-quad mesh is 1:1");
}

void CheckQuadCoordinates()
{
    Mesh const mesh = MixedMesh();
    auto const expect = [&](P3 p, float eu, float ev, char const *what) {
        int id = -1;
        float u = -1, v = -1;
        bool const ok = mesh.Locate(0, p, &id, &u, &v);
        CHECK(ok && id == 0 && Near(u, eu, 1e-5) && Near(v, ev, 1e-5),
              std::string("quad ") + what + " -> " + Uv(u, v));
    };
    expect(Place(0, 0, 0), 0, 0, "p0");
    expect(Place(1, 0, 0), 1, 0, "p1");
    expect(Place(1, 1, 0), 1, 1, "p2");
    expect(Place(0, 1, 0), 0, 1, "p3");
    expect(Place(0.5, 0.5, 0), 0.5f, 0.5f, "centre");
    expect(Place(0.5, 0, 0), 0.5f, 0, "edge 0 midpoint");
    expect(Place(1, 0.5, 0), 1, 0.5f, "edge 1 midpoint");
    expect(Place(0.5, 1, 0), 0.5f, 1, "edge 2 midpoint");
    expect(Place(0, 0.5, 0), 0, 0.5f, "edge 3 midpoint");
    // Off the face: clamped, and a point above the plane projects onto it.
    expect(Place(-0.5, 0.25, 0), 0, 0.25f, "left of the face (clamped)");
    expect(Place(0.25, 0.75, 0.4), 0.25f, 0.75f, "above the face (projected)");

    // The trapezoid round-trips bilinear(u, v).
    float corners[12];
    CHECK(mesh.Corners(3, 0, corners), "trapezoid corners");
    CHECK(!mesh.Corners(3, 1, corners), "a quad has no sub-face 1");
    int worst = 0;
    double worstError = 0;
    for (int i = 0; i <= 10; ++i) {
        for (int j = 0; j <= 10; ++j) {
            double const u0 = i / 10.0, v0 = j / 10.0;
            int id = -1;
            float u = -1, v = -1;
            bool const ok = mesh.Locate(3, Bilinear(corners, u0, v0), &id, &u, &v);
            double const error = std::max(std::fabs(u - u0), std::fabs(v - v0));
            if (!ok || id != 9) ++worst;
            worstError = std::max(worstError, error);
        }
    }
    CHECK(worst == 0, "trapezoid grid resolves to id 9");
    CHECK(worstError < 1e-4, "trapezoid round trip error " + std::to_string(worstError));

    // A general, mildly non-planar quad also round-trips: projection is affine,
    // so the projected surface is the bilinear patch of the projected corners.
    Mesh warped;
    warped.AddFace({Place(0, 0, 0), Place(3, -0.5, 0.1), Place(4, 3, -0.15), Place(-0.5, 2, 0.05)});
    warped.Finish();
    CHECK(warped.Corners(0, 0, corners), "warped corners");
    worstError = 0;
    worst = 0;
    for (int i = 0; i <= 8; ++i) {
        for (int j = 0; j <= 8; ++j) {
            double const u0 = i / 8.0, v0 = j / 8.0;
            int id = -1;
            float u = -1, v = -1;
            if (!warped.Locate(0, Bilinear(corners, u0, v0), &id, &u, &v) || id != 0) ++worst;
            worstError = std::max({worstError, std::fabs(u - u0), std::fabs(v - v0)});
        }
    }
    CHECK(worst == 0 && worstError < 1e-4,
          "non-planar quad round trip error " + std::to_string(worstError));
}

void CheckNgonCoordinates()
{
    Mesh const mesh = MixedMesh();
    P3 const c2 = mesh.Corner(2, 2);
    P3 centroid{0, 0, 0};
    for (int k = 0; k < 5; ++k) {
        P3 const c = mesh.Corner(2, k);
        centroid = {centroid.x + c.x / 5, centroid.y + c.y / 5, centroid.z + c.z / 5};
    }
    P3 const nearCorner2{0.9 * c2.x + 0.1 * centroid.x, 0.9 * c2.y + 0.1 * centroid.y,
                         0.9 * c2.z + 0.1 * centroid.z};
    int id = -1;
    float u = -1, v = -1;
    CHECK(mesh.Locate(2, nearCorner2, &id, &u, &v) && id == 4 + 2,
          "pentagon point near corner 2 -> sub-face id 6, got " + std::to_string(id));
    float corners[12];
    CHECK(mesh.Corners(2, 2, corners), "pentagon sub-face 2 corners");
    P3 const back = Bilinear(corners, u, v);
    CHECK(Near(back.x, nearCorner2.x, 1e-4) && Near(back.y, nearCorner2.y, 1e-4) &&
              Near(back.z, nearCorner2.z, 1e-4),
          "pentagon near-corner point round-trips " + Uv(u, v));
    // The pentagon is regular, so the sub-face is mirror-symmetric about the
    // corner-centroid axis and a point on that axis has u == v.
    CHECK(Near(u, v, 1e-4) && u > 0.05f && u < 0.3f,
          "near-corner point on the diagonal " + Uv(u, v));

    // Sub-face orientation: p(1,0) is the midpoint toward the next corner,
    // p(0,1) the midpoint toward the previous one, p(1,1) the centroid.
    P3 const c3 = mesh.Corner(2, 3), c1 = mesh.Corner(2, 1);
    CHECK(Near(corners[3], (c2.x + c3.x) / 2, 1e-5) && Near(corners[4], (c2.y + c3.y) / 2, 1e-5) &&
              Near(corners[5], (c2.z + c3.z) / 2, 1e-5),
          "sub-face p(1,0) is the midpoint toward the next corner");
    CHECK(Near(corners[9], (c2.x + c1.x) / 2, 1e-5) && Near(corners[10], (c2.y + c1.y) / 2, 1e-5) &&
              Near(corners[11], (c2.z + c1.z) / 2, 1e-5),
          "sub-face p(0,1) is the midpoint toward the previous corner");
    CHECK(Near(corners[6], centroid.x, 1e-5) && Near(corners[7], centroid.y, 1e-5) &&
              Near(corners[8], centroid.z, 1e-5),
          "sub-face p(1,1) is the centroid");

    // Interior points of every sub-face (pentagon and triangle) round-trip.
    struct { int face, count, firstId; } const ngons[] = {{2, 5, 4}, {1, 3, 1}};
    for (auto const &ngon : ngons) {
        int bad = 0;
        double worstError = 0;
        for (int k = 0; k < ngon.count; ++k) {
            CHECK(mesh.Corners(ngon.face, k, corners), "sub-face corners");
            for (int i = 1; i <= 9; ++i) {
                for (int j = 1; j <= 9; ++j) {
                    double const u0 = i / 10.0, v0 = j / 10.0;
                    if (!mesh.Locate(ngon.face, Bilinear(corners, u0, v0), &id, &u, &v) ||
                        id != ngon.firstId + k) {
                        ++bad;
                        continue;
                    }
                    worstError = std::max({worstError, std::fabs(u - u0), std::fabs(v - v0)});
                }
            }
        }
        CHECK(bad == 0, "face " + std::to_string(ngon.face) + ": " + std::to_string(bad) +
                            " interior sub-face points resolved to the wrong id");
        CHECK(worstError < 1e-4, "face " + std::to_string(ngon.face) +
                                     " sub-face round trip error " + std::to_string(worstError));
    }
    CHECK(!mesh.Corners(2, 5, corners) && !mesh.Corners(2, -1, corners), "sub-face out of range");
}

void CheckTriangleMesh()
{
    Mesh mesh;
    mesh.AddFace({Place(0, 0, 0), Place(2, 0, 0), Place(0, 3, 0)});
    mesh.AddFace({Place(2, 0, 0), Place(2, 3, 0), Place(0, 3, 0)});
    mesh.Finish();
    P3 const p0 = mesh.Corner(1, 0), p1 = mesh.Corner(1, 1), p2 = mesh.Corner(1, 2);
    P3 const p{p0.x + 0.2 * (p1.x - p0.x) + 0.3 * (p2.x - p0.x),
               p0.y + 0.2 * (p1.y - p0.y) + 0.3 * (p2.y - p0.y),
               p0.z + 0.2 * (p1.z - p0.z) + 0.3 * (p2.z - p0.z)};
    int id = -1;
    float u = -1, v = -1;
    CHECK(mesh.Locate(1, p, &id, &u, &v, true) && id == 1 && Near(u, 0.2, 1e-5) &&
              Near(v, 0.3, 1e-5),
          "triangle barycentric " + Uv(u, v));
    // Outside the hypotenuse: clamped back onto the triangle.
    P3 const beyond{p0.x + 0.9 * (p1.x - p0.x) + 0.9 * (p2.x - p0.x),
                 p0.y + 0.9 * (p1.y - p0.y) + 0.9 * (p2.y - p0.y),
                 p0.z + 0.9 * (p1.z - p0.z) + 0.9 * (p2.z - p0.z)};
    CHECK(mesh.Locate(1, beyond, &id, &u, &v, true) && Near(u + v, 1.0, 1e-5) && Near(u, 0.5, 1e-5),
          "triangle clamp " + Uv(u, v));
    // The same mesh as a quad-convention file: three sub-faces per triangle.
    CHECK(mesh.Locate(1, p0, &id, &u, &v) && id == 3 && Near(u, 0, 1e-5) && Near(v, 0, 1e-5),
          "triangle corner 0 in quad mode -> sub-face 3");

    Mesh const mixed = MixedMesh();
    CHECK(!mixed.Locate(1, mixed.Corner(1, 0), &id, &u, &v, true),
          "triangle mode refuses a mixed mesh");
}

void CheckInvalidInput()
{
    Mesh mesh = MixedMesh();
    int id = -1;
    float u = 0, v = 0;
    CHECK(!mesh.Locate(-1, Place(0, 0, 0), &id, &u, &v), "face -1");
    CHECK(!mesh.Locate(4, Place(0, 0, 0), &id, &u, &v), "face past the end");
    CHECK(!mesh.Locate(0, P3{NAN, 0, 0}, &id, &u, &v), "non-finite point");
    CHECK(!UsdGenPtexFaceCoordinate(mesh.points.data(), mesh.points.size() / 3,
                                    mesh.counts.data(), mesh.indices.data(),
                                    mesh.offsets.data(), mesh.counts.size(), nullptr, false, 0,
                                    0, 0, 0, &id, &u, &v),
          "quad mode needs first ids");
    CHECK(!UsdGenPtexFaceCoordinate(mesh.points.data(), 3, mesh.counts.data(),
                                    mesh.indices.data(), mesh.offsets.data(),
                                    mesh.counts.size(), mesh.firstIds.data(), false, 0, 0, 0,
                                    0, &id, &u, &v),
          "index outside the points");

    Mesh degenerate;
    degenerate.AddFace({P3{1, 1, 1}, P3{1, 1, 1}, P3{1, 1, 1}, P3{1, 1, 1}});
    degenerate.AddFace({P3{0, 0, 0}, P3{1, 0, 0}, P3{2, 0, 0}, P3{3, 0, 0}});
    degenerate.Finish();
    CHECK(!degenerate.Locate(0, P3{1, 1, 1}, &id, &u, &v), "collapsed quad");
    CHECK(!degenerate.Locate(1, P3{1, 0, 0}, &id, &u, &v), "zero-area quad");
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

constexpr int kRes = 8;

// Face 0: 8x8, value = texel-centre u. Face 1: constant `constant`. They share
// face 0's edge 1 / face 1's edge 3.
bool WriteFixture(std::string const &path, float constant)
{
    Ptex::String error;
    PtexWriter *writer = PtexWriter::open(path.c_str(), Ptex::mt_quad, Ptex::dt_float,
                                          /*nchannels*/ 1, /*alphachan*/ -1, /*nfaces*/ 2, error,
                                          /*genmipmaps*/ true);
    if (!writer) {
        std::printf("cannot create %s: %s\n", path.c_str(), error.c_str());
        return false;
    }
    Ptex::Res const res(int8_t(3), int8_t(3));
    int adjFaces0[4] = {-1, 1, -1, -1}, adjEdges0[4] = {0, 3, 0, 0};
    int adjFaces1[4] = {-1, -1, -1, 0}, adjEdges1[4] = {0, 0, 0, 1};
    std::vector<float> texels(kRes * kRes);
    for (int j = 0; j < kRes; ++j) {
        for (int i = 0; i < kRes; ++i) texels[size_t(j * kRes + i)] = (i + 0.5f) / kRes;
    }
    bool ok = writer->writeFace(0, Ptex::FaceInfo(res, adjFaces0, adjEdges0), texels.data());
    ok = writer->writeConstantFace(1, Ptex::FaceInfo(res, adjFaces1, adjEdges1), &constant) && ok;
    ok = writer->close(error) && ok;
    writer->release();
    if (!ok) std::printf("cannot write %s: %s\n", path.c_str(), error.c_str());
    return ok;
}

UsdGenPtexMapOptions Filter(char const *filter)
{
    UsdGenPtexMapOptions options;
    options.filter = filter;
    return options;
}

void CheckSampling(std::string const &path)
{
    if (!WriteFixture(path, 7.0f)) {
        CHECK(false, "fixture written");
        return;
    }
    std::string error;
    {
        auto nearest = UsdGenPtexTexture::Open(path, Filter("nearest"), &error);
        CHECK(nearest != nullptr, "open nearest: " + error);
        if (!nearest) return;
        CHECK(nearest->NumFaces() == 2, "NumFaces");
        CHECK(nearest->NumChannels() == 1, "NumChannels");
        CHECK(nearest->SampleChannels() == 1, "SampleChannels");
        CHECK(!nearest->IsTriangleMesh(), "quad file");
        CHECK(nearest->Path() == path, "Path");
        auto sampler = nearest->MakeSampler();
        CHECK(sampler != nullptr, "MakeSampler");
        float value = -1;
        CHECK(sampler->Sample(1, 0.3f, 0.7f, &value) && value == 7.0f,
              "nearest face 1 = 7, got " + std::to_string(value));
        CHECK(sampler->Sample(0, 0.30f, 0.5f, &value) && Near(value, 2.5 / kRes, 1e-6),
              "nearest face 0 texel 2, got " + std::to_string(value));
        value = -1;
        CHECK(!sampler->Sample(2, 0.5f, 0.5f, &value) && value == -1, "face 2 is out of range");
        CHECK(!sampler->Sample(-1, 0.5f, 0.5f, &value) && value == -1, "face -1");
        CHECK(!sampler->Sample(0, NAN, 0.5f, &value) && value == -1, "non-finite u");
        CHECK(!sampler->Sample(0, 0.5f, 0.5f, nullptr), "null out");
    }

    auto bilinear = UsdGenPtexTexture::Open(path, Filter("bilinear"), &error);
    CHECK(bilinear != nullptr, "open bilinear: " + error);
    if (!bilinear) return;
    auto sampler = bilinear->MakeSampler();
    float previous = -1;
    bool monotonic = true, linear = true;
    for (int i = 1; i <= 19; ++i) {
        float const u = i * 0.05f;
        float value = -1;
        if (!sampler->Sample(0, u, 0.5f, &value)) { monotonic = false; break; }
        if (!(value > previous)) monotonic = false;
        // Linear texels interpolate exactly between the first and last centre.
        if (u >= 0.5f / kRes && u <= 1 - 0.5f / kRes && !Near(value, u, 1e-5)) linear = false;
        previous = value;
    }
    CHECK(monotonic, "bilinear face 0 is increasing in u");
    CHECK(linear, "bilinear face 0 reproduces u between texel centres");

    // Every filter token opens and samples inside the value range.
    for (char const *token : {"nearest", "bilinear", "box", "gaussian", "bicubic", "bspline",
                              "catmullrom", "mitchell"}) {
        UsdGenPtexMapOptions options = Filter(token);
        options.blur = 0.05f;
        auto texture = UsdGenPtexTexture::Open(path, options, &error);
        CHECK(texture != nullptr, std::string("open with filter ") + token + ": " + error);
        if (!texture) continue;
        float value = -1;
        CHECK(texture->MakeSampler()->Sample(0, 0.5f, 0.5f, &value) && value > 0.3f &&
                  value < 0.7f,
              std::string(token) + " samples face 0 centre, got " + std::to_string(value));
    }

    // Option and file errors.
    auto const expectError = [&](std::string const &file, UsdGenPtexMapOptions const &options,
                                 char const *what) {
        std::string message;
        auto texture = UsdGenPtexTexture::Open(file, options, &message);
        CHECK(!texture && !message.empty(), std::string(what) + " is an error");
        return message;
    };
    std::string const missing = path + ".missing.ptx";
    std::string const first = expectError(missing, Filter("bilinear"), "a missing file");
    // The cache remembers the failure; a second open still explains it.
    std::string const second = expectError(missing, Filter("bilinear"), "a missing file (again)");
    CHECK(first.find("missing") != std::string::npos && second == first,
          "missing-file errors name the file: '" + first + "' / '" + second + "'");
    expectError(path, Filter("trilinear"), "an unknown filter");
    UsdGenPtexMapOptions options;
    options.borderMode = "wrap";
    expectError(path, options, "an unknown border mode");
    options = UsdGenPtexMapOptions();
    options.firstChannel = 1;
    expectError(path, options, "a channel window beyond the file");
    options.firstChannel = -1;
    expectError(path, options, "a negative first channel");
    options = UsdGenPtexMapOptions();
    options.channelCount = 0;
    expectError(path, options, "an empty channel window");
    options = UsdGenPtexMapOptions();
    options.blur = -1;
    expectError(path, options, "a negative blur");
    for (char const *mode : {"clamp", "black", "periodic"}) {
        options = UsdGenPtexMapOptions();
        options.borderMode = mode;
        options.channelCount = 4;
        auto texture = UsdGenPtexTexture::Open(path, options, &error);
        CHECK(texture && texture->SampleChannels() == 1,
              std::string("border mode ") + mode + " opens, channelCount clamps to 1");
    }

    // Concurrent samplers agree bit for bit with a serial pass.
    constexpr int kSamples = 20000;
    auto const lookup = [&](UsdGenPtexTexture::Sampler const &s, int n) {
        uint32_t state = 12345u + uint32_t(n);
        auto next = [&] {
            state = state * 1664525u + 1013904223u;
            return float(state >> 8) / float(1u << 24);
        };
        float value = 0;
        s.Sample(int(next() * 2) % 2, next(), next(), &value);
        return value;
    };
    std::vector<float> reference(kSamples);
    {
        auto serial = bilinear->MakeSampler();
        for (int n = 0; n < kSamples; ++n) reference[size_t(n)] = lookup(*serial, n);
    }
    std::vector<int> mismatches(4, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            auto own = bilinear->MakeSampler();
            for (int pass = 0; pass < 3; ++pass) {
                for (int n = 0; n < kSamples; ++n) {
                    if (lookup(*own, n) != reference[size_t(n)]) ++mismatches[size_t(t)];
                }
            }
        });
    }
    for (auto &thread : threads) thread.join();
    int total = 0;
    for (int m : mismatches) total += m;
    CHECK(total == 0, std::to_string(total) + " concurrent samples differ from the serial pass");

    // Reload: once nobody holds the file, a purge makes the next open re-read it.
    sampler.reset();
    bilinear.reset();
    UsdGenPtexTexture::PurgeCache();
    if (!WriteFixture(path, 9.0f)) {
        CHECK(false, "fixture rewritten");
        return;
    }
    UsdGenPtexTexture::PurgeCache();
    auto reloaded = UsdGenPtexTexture::Open(path, Filter("nearest"), &error);
    float value = -1;
    CHECK(reloaded && reloaded->MakeSampler()->Sample(1, 0.5f, 0.5f, &value) && value == 9.0f,
          "reloaded face 1 = 9, got " + std::to_string(value));
}

// Prints the distinct nearest-sampled texel-centre values of every face.
int DumpFile(std::string const &path)
{
    std::string error;
    auto texture = UsdGenPtexTexture::Open(path, Filter("nearest"), &error);
    if (!texture) {
        std::printf("%s\n", error.c_str());
        return 1;
    }
    UsdGenPtexMapOptions options = Filter("nearest");
    options.channelCount = texture->NumChannels();
    texture = UsdGenPtexTexture::Open(path, options, &error);
    if (!texture) {
        std::printf("%s\n", error.c_str());
        return 1;
    }
    auto sampler = texture->MakeSampler();
    int const channels = texture->SampleChannels();
    std::set<std::vector<float>> all;
    std::printf("%s: %d faces, %d channel(s)\n", path.c_str(), texture->NumFaces(), channels);
    for (int face = 0; face < texture->NumFaces(); ++face) {
        std::set<std::vector<float>> distinct;
        for (int j = 0; j < 16; ++j) {
            for (int i = 0; i < 16; ++i) {
                std::vector<float> value(size_t(channels), 0.0f);
                sampler->Sample(face, (i + 0.5f) / 16, (j + 0.5f) / 16, value.data());
                distinct.insert(value);
                all.insert(value);
            }
        }
        std::printf("  face %2d:", face);
        for (auto const &value : distinct) {
            std::printf(" (");
            for (int c = 0; c < channels; ++c) std::printf(c ? " %g" : "%g", value[size_t(c)]);
            std::printf(")");
        }
        std::printf("\n");
    }
    std::printf("distinct values: %zu\n", all.size());
    return 0;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc > 1) return DumpFile(argv[1]);

    CheckFirstFaceIds();
    CheckQuadCoordinates();
    CheckNgonCoordinates();
    CheckTriangleMesh();
    CheckInvalidInput();

    namespace fs = std::filesystem;
    std::error_code ec;
    auto const stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path const path =
        fs::temp_directory_path(ec) / ("testUsdGenPtexMap_" + std::to_string(stamp) + ".ptx");
    CheckSampling(path.string());
    UsdGenPtexTexture::PurgeCache();
    fs::remove(path, ec);

    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("testUsdGenPtexMap: OK\n");
    return 0;
}
