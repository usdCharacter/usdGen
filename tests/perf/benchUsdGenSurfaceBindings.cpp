// benchUsdGenSurfaceBindings — C3 surface-binding cache throughput.
//
// This benchmark measures the actual UsdGenRestSurfaceBindingCache::Bind
// implementation (k=8 centroid candidates), not GuideInterpolate or an E-4
// guide-interpolation proof.  The gate metric is the warmed median Bind time
// only, with a private TBB task_arena concurrency limit of eight (including
// the calling participant):
//   100k roots <= 25 ms, 1M roots <= 300 ms.
// Create+Bind is reported separately as a cold metric and is not folded into
// those limits.  Timing failures remain informational unless USDGEN_GATE=1.

#include "../utils/pxrInterop.h"
#include "usdGenUsd/lower.h"
#include "usdGen/surfaceRootBindings.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>
#include <tbb/task_arena.h>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;

namespace {

constexpr size_t kGrid = 64;
constexpr size_t kFaceCount = kGrid * kGrid;
constexpr size_t kWarmupRuns = 2;
constexpr size_t kTimedRuns = 5;

struct Fixture {
    UsdGenSurfaceDesc surface;
    usdGen::UsdGenVec3fArray roots;
    bool quads = false;
};

struct CaseResult {
    size_t rootCount = 0;
    double warmMedianMs = 0.0;
    double coldMs = 0.0;
    size_t retainedBytes = 0;
};

bool GateMode()
{
    char const *value = std::getenv("USDGEN_GATE");
    return value && std::string(value) == "1";
}

double Median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values.empty() ? 0.0 : values[values.size() / 2];
}

float NextUnit(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return static_cast<float>(*state >> 8) * (1.0f / 16777216.0f);
}

usdGen::UsdGenVec3f TrianglePoint(usdGen::UsdGenVec3f const &p0, usdGen::UsdGenVec3f const &p1,
                      usdGen::UsdGenVec3f const &p2, float u, float v)
{
    return p0 * (1.0f - u - v) + p1 * u + p2 * v;
}

usdGen::UsdGenVec3f QuadPoint(usdGen::UsdGenVec3f const &p0, usdGen::UsdGenVec3f const &p1,
                  usdGen::UsdGenVec3f const &p2, usdGen::UsdGenVec3f const &p3, float u, float v)
{
    return p0 * ((1.0f - u) * (1.0f - v)) +
           p1 * (u * (1.0f - v)) + p2 * (u * v) +
           p3 * ((1.0f - u) * v);
}

Fixture MakeFixture(bool quads, size_t rootCount)
{
    Fixture fixture;
    fixture.quads = quads;
    fixture.surface.path = quads ? SdfPath("/SurfaceQuads")
                                 : SdfPath("/SurfaceTriangles");
    fixture.surface.id = quads ? 2 : 1;
    size_t const verticesPerFace = quads ? 4 : 3;
    fixture.surface.restPoints = usdGen::UsdGenVec3fArray(kFaceCount * verticesPerFace);
    fixture.surface.faceVertexCounts = VtIntArray(kFaceCount,
                                                   static_cast<int>(verticesPerFace));
    fixture.surface.faceVertexIndices = VtIntArray(kFaceCount * verticesPerFace);

    // Every cell owns a modestly warped patch.  Unique vertices avoid any
    // ambiguity at shared boundaries while retaining a 64x64, 4096-face
    // surface for the spatial index.
    for (size_t face = 0; face != kFaceCount; ++face) {
        size_t const x = face % kGrid;
        size_t const y = face / kGrid;
        float const fx = static_cast<float>(x);
        float const fy = static_cast<float>(y);
        size_t const base = face * verticesPerFace;
        fixture.surface.restPoints[base + 0] = usdGen::UsdGenVec3f(fx, fy, 0.00f);
        fixture.surface.restPoints[base + 1] = usdGen::UsdGenVec3f(fx + 1.0f, fy, 0.06f);
        if (quads) {
            fixture.surface.restPoints[base + 2] =
                usdGen::UsdGenVec3f(fx + 1.0f, fy + 1.0f, 0.14f);
            fixture.surface.restPoints[base + 3] =
                usdGen::UsdGenVec3f(fx, fy + 1.0f, -0.04f);
        } else {
            fixture.surface.restPoints[base + 2] = usdGen::UsdGenVec3f(fx, fy + 1.0f, -0.03f);
        }
        for (size_t corner = 0; corner != verticesPerFace; ++corner)
            fixture.surface.faceVertexIndices[base + corner] =
                static_cast<int>(base + corner);
    }

    fixture.roots = usdGen::UsdGenVec3fArray(rootCount);
    uint32_t state = quads ? 0x5eeda11u : 0xc3a5e7u;
    for (size_t i = 0; i != rootCount; ++i) {
        size_t const face = i % kFaceCount;
        size_t const base = face * verticesPerFace;
        float const first = NextUnit(&state);
        float const second = NextUnit(&state);
        if (quads) {
            float const u = 0.18f + 0.64f * first;
            float const v = 0.18f + 0.64f * second;
            fixture.roots[i] = QuadPoint(fixture.surface.restPoints[base + 0],
                                         fixture.surface.restPoints[base + 1],
                                         fixture.surface.restPoints[base + 2],
                                         fixture.surface.restPoints[base + 3], u, v);
        } else {
            // Keep both barycentric coordinates well inside the triangle.
            float const u = 0.17f + 0.28f * first;
            float const v = 0.17f + 0.28f * second;
            fixture.roots[i] = TrianglePoint(fixture.surface.restPoints[base + 0],
                                              fixture.surface.restPoints[base + 1],
                                              fixture.surface.restPoints[base + 2], u, v);
        }
    }
    return fixture;
}

double Distance2(usdGen::UsdGenVec3f const &a, usdGen::UsdGenVec3f const &b)
{
    double const dx = static_cast<double>(a[0]) - b[0];
    double const dy = static_cast<double>(a[1]) - b[1];
    double const dz = static_cast<double>(a[2]) - b[2];
    return dx * dx + dy * dy + dz * dz;
}

bool Verify(UsdGenSurfaceDesc const &surface, usdGen::UsdGenVec3fArray const &roots,
            UsdGenSurfaceRootBindingResult const &result, std::string *error)
{
    auto fail = [&](char const *message) {
        if (error) *error = message;
        return false;
    };
    if (result.rootPrim.size() != roots.size() || result.rootUV.size() != roots.size() ||
        result.valid.size() != roots.size() || result.unresolved.size() != roots.size() ||
        result.unresolvedCount != 0)
        return fail("binding result cardinality or unresolved count is incorrect");
    std::vector<size_t> faceOffsets(surface.faceVertexCounts.size() + 1, 0);
    for (size_t face = 0; face != surface.faceVertexCounts.size(); ++face) {
        int const count = surface.faceVertexCounts[face];
        if (count != 3 && count != 4) return fail("benchmark produced an unexpected face type");
        faceOffsets[face + 1] = faceOffsets[face] + static_cast<size_t>(count);
    }
    for (size_t i = 0; i != roots.size(); ++i) {
        if (!result.valid[i] || result.unresolved[i] || result.rootPrim[i] < 0 ||
            static_cast<size_t>(result.rootPrim[i]) >= surface.faceVertexCounts.size())
            return fail("binding result contains an invalid or unresolved root");
        usdGen::UsdGenVec2f const uv = result.rootUV[i];
        if (!std::isfinite(uv[0]) || !std::isfinite(uv[1]) || uv[0] < 0.0f ||
            uv[1] < 0.0f)
            return fail("binding result contains a non-finite or negative UV");
        size_t const face = static_cast<size_t>(result.rootPrim[i]);
        size_t const faceOffset = faceOffsets[face];
        int const count = surface.faceVertexCounts[face];
        usdGen::UsdGenVec3f reconstructed;
        if (count == 3) {
            if (uv[0] + uv[1] > 1.0001f) return fail("triangle UV is outside its domain");
            int const a = surface.faceVertexIndices[faceOffset + 0];
            int const b = surface.faceVertexIndices[faceOffset + 1];
            int const c = surface.faceVertexIndices[faceOffset + 2];
            reconstructed = TrianglePoint(surface.restPoints[a], surface.restPoints[b],
                                          surface.restPoints[c], uv[0], uv[1]);
        } else {
            if (uv[0] > 1.0001f || uv[1] > 1.0001f)
                return fail("quad UV is outside its domain");
            int const a = surface.faceVertexIndices[faceOffset + 0];
            int const b = surface.faceVertexIndices[faceOffset + 1];
            int const c = surface.faceVertexIndices[faceOffset + 2];
            int const d = surface.faceVertexIndices[faceOffset + 3];
            reconstructed = QuadPoint(surface.restPoints[a], surface.restPoints[b],
                                      surface.restPoints[c], surface.restPoints[d], uv[0], uv[1]);
        }
        if (Distance2(reconstructed, roots[i]) > 1.0e-6)
            return fail("binding UV does not reconstruct the source root");
    }
    return true;
}

bool RunCase(Fixture const &fixture, size_t rootCount, tbb::task_arena &arena,
             CaseResult *out)
{
    std::string error;
    bool coldOk = true;
    std::shared_ptr<const UsdGenRestSurfaceBindingCache> coldCache;
    UsdGenSurfaceRootBindingResult coldResult;
    auto const coldStart = std::chrono::steady_clock::now();
    arena.execute([&] {
        coldCache = UsdGenRestSurfaceBindingCache::Create(usdGenUsd::Lower(fixture.surface), &error);
        coldOk = coldCache && coldCache->Bind(fixture.roots, usdGen::UsdGenMatrix4d(1.0),
                                               &coldResult, &error);
    });
    auto const coldEnd = std::chrono::steady_clock::now();
    if (!coldOk || !Verify(fixture.surface, fixture.roots, coldResult, &error)) {
        std::fprintf(stderr, "%s cold Create+Bind failed: %s\n",
                     fixture.quads ? "quad" : "triangle", error.c_str());
        return false;
    }

    auto cache = UsdGenRestSurfaceBindingCache::Create(usdGenUsd::Lower(fixture.surface), &error);
    if (!cache) {
        std::fprintf(stderr, "%s warm Create failed: %s\n",
                     fixture.quads ? "quad" : "triangle", error.c_str());
        return false;
    }
    for (size_t i = 0; i != kWarmupRuns; ++i) {
        UsdGenSurfaceRootBindingResult warmup;
        if (!arena.execute([&] {
                return cache->Bind(fixture.roots, usdGen::UsdGenMatrix4d(1.0), &warmup, &error);
            })) {
            std::fprintf(stderr, "warmup Bind failed: %s\n", error.c_str());
            return false;
        }
    }
    std::vector<double> samples;
    samples.reserve(kTimedRuns);
    for (size_t i = 0; i != kTimedRuns; ++i) {
        UsdGenSurfaceRootBindingResult timed;
        auto const begin = std::chrono::steady_clock::now();
        bool const ok = arena.execute([&] {
            return cache->Bind(fixture.roots, usdGen::UsdGenMatrix4d(1.0), &timed, &error);
        });
        auto const end = std::chrono::steady_clock::now();
        if (!ok || !Verify(fixture.surface, fixture.roots, timed, &error)) {
            std::fprintf(stderr, "timed Bind/correctness check failed: %s\n", error.c_str());
            return false;
        }
        samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }

    out->rootCount = rootCount;
    out->warmMedianMs = Median(std::move(samples));
    out->coldMs = std::chrono::duration<double, std::milli>(coldEnd - coldStart).count();
    out->retainedBytes = cache->BytesOwned();
    char const *const shape = fixture.quads ? "bilinear-quads" : "triangles";
    double const limit = rootCount == 100000 ? 25.0 : 300.0;
    bool const pass = out->warmMedianMs <= limit;
    std::printf("%s roots=%zu warm Bind median=%.3f ms (gate %.3f) -> %s%s; "
                "cold Create+Bind=%.3f ms; retained cache=%zu bytes\n",
                shape, rootCount, out->warmMedianMs, limit, pass ? "PASS" : "FAIL",
                pass ? "" : (GateMode() ? " [hard]" : " [info]"), out->coldMs,
                out->retainedBytes);
    return pass || !GateMode();
}

} // namespace

int main()
{
    tbb::task_arena arena(8);
    int failures = 0;
    for (bool quads : {false, true}) {
        CaseResult small, large;
        Fixture smallFixture = MakeFixture(quads, 100000);
        Fixture largeFixture = MakeFixture(quads, 1000000);
        if (!RunCase(smallFixture, 100000, arena, &small)) ++failures;
        if (!RunCase(largeFixture, 1000000, arena, &large)) ++failures;
        if (small.warmMedianMs > 0.0) {
            std::printf("%s warm scaling 100k->1M: %.2fx\n",
                        quads ? "bilinear-quads" : "triangles",
                        large.warmMedianMs / small.warmMedianMs);
        }
    }
    return failures ? 1 : 0;
}
