// CPU-only contract for CudaScatterGrow::ValidateRoots (no device
// needed, so this test passes where the grow legs SKIP): controls,
// topology, finiteness, duplicate stable ids, and per-curve overflow.
// Pins the check order too (index order, Finite before duplicate
// before overflow), so faster validators must keep the same statuses.
#include "usdGen/gpu/scatterGrow.h"

#include <cuda_runtime.h>

#include "tbb/task_arena.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "CHECK failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

static std::shared_ptr<ScatterGrowRoots> Roots(size_t n, uint64_t idBase = 1)
{
    auto r = std::make_shared<ScatterGrowRoots>();
    r->positions.reserve(n);
    r->stableIds.reserve(n);
    r->rootPrim.reserve(n);
    r->rootUV.reserve(n);
    r->rootT.reserve(n);
    r->rootB.reserve(n);
    r->rootN.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        r->positions.push_back(make_float3(float(i), 0.0f, 0.0f));
        r->stableIds.push_back(idBase + i);
        r->rootPrim.push_back(0);
        r->rootUV.push_back(make_float2(0.0f, 0.0f));
        r->rootT.push_back(make_float3(1.0f, 0.0f, 0.0f));
        r->rootB.push_back(make_float3(0.0f, 1.0f, 0.0f));
        r->rootN.push_back(make_float3(0.0f, 0.0f, 1.0f));
    }
    return r;
}

static ScatterGrowControls Controls()
{
    ScatterGrowControls c;
    c.cvCount = 8;
    c.seed = 42;
    c.length = 1.0;
    c.randomLo = 0.8;
    c.randomHi = 1.2;
    return c;
}

// Obviously-correct interleaved reference for the differential fuzz:
// same preamble as ValidateRoots, then the naive per-index loop (linear
// duplicate scan, verbatim overflow math copied from scatterGrow.cu).
// The fuzz below checks status equality on random multi-fault inputs,
// so any precedence or minima-combine bug in the radix version shows.
static uint64_t RefHash64(uint64_t key, uint32_t salt)
{
    uint64_t z = (key ^ (uint64_t(salt) * 0x9E3779B97F4A7C15ull)) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float RefDraw(int seed, uint64_t id, uint32_t salt = 0x47726F77u)
{
    uint64_t key = RefHash64(uint64_t(uint32_t(seed)), salt) ^ id;
    return float(uint32_t(RefHash64(key, salt) >> 32) >> 8) * 0x1.0p-24f;
}
static float3 RefNorm(float3 v)
{
    float const length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (!(length > 1.0e-12f) || !std::isfinite(length))
        return make_float3(0, 1, 0);
    return make_float3(v.x / length, v.y / length, v.z / length);
}
static float3 RefRot(float3 direction, float3 axis, float degrees)
{
    if (degrees == 0.0f) return direction;
    axis = RefNorm(axis);
    float const radians = degrees * (3.14159265358979323846f / 180.0f);
    float const c = std::cos(radians), s = std::sin(radians);
    float const dot = axis.x * direction.x + axis.y * direction.y + axis.z * direction.z;
    float3 const cross = make_float3(
        axis.y * direction.z - axis.z * direction.y,
        axis.z * direction.x - axis.x * direction.z,
        axis.x * direction.y - axis.y * direction.x);
    float const oneMinusC = 1.0f - c;
    return make_float3(
        direction.x * c + cross.x * s + axis.x * dot * oneMinusC,
        direction.y * c + cross.y * s + axis.y * dot * oneMinusC,
        direction.z * c + cross.z * s + axis.z * dot * oneMinusC);
}
static bool RefFinite(float3 v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
static bool RefFinite2(float2 v) { return std::isfinite(v.x) && std::isfinite(v.y); }

static ScatterGrowStatus ReferenceValidate(
    std::shared_ptr<const ScatterGrowRoots> const &r,
    ScatterGrowControls const &c, size_t *total)
{
    if (!r || c.cvCount < 2 || c.cvCount > 64 || !std::isfinite(c.length) ||
        !std::isfinite(c.randomLo) || !std::isfinite(c.randomHi) ||
        !std::isfinite(c.lift) || !std::isfinite(c.fallbackWidth) ||
        c.length < 0 || c.randomLo < 0 || c.randomHi < 0 || c.fallbackWidth < 0 ||
        c.lift < -90.0f || c.lift > 90.0f ||
        c.direction > ScatterGrowDirection::Literal ||
        !std::isfinite(c.azimuth) || c.azimuth < -360.0f || c.azimuth > 360.0f ||
        !std::isfinite(c.azimuthRandom) || c.azimuthRandom < 0.0f ||
        c.azimuthRandom > 1.0f ||
        (c.direction == ScatterGrowDirection::Literal && !RefFinite(c.literalDirection)))
        return ScatterGrowStatus::InvalidArgument;
    size_t n = r->positions.size();
    if (r->stableIds.size() != n || r->rootPrim.size() != n || r->rootUV.size() != n ||
        r->rootT.size() != n || r->rootB.size() != n || r->rootN.size() != n)
        return ScatterGrowStatus::InvalidTopology;
    ScatterGrowRequirements requirements;
    auto requirementStatus = GetScatterGrowRequirements(n, c.cvCount, &requirements);
    if (requirementStatus != ScatterGrowStatus::Ok) return requirementStatus;
    *total = requirements.pointCount;
    for (size_t i = 0; i < n; ++i) {
        if (!RefFinite(r->positions[i]) || !RefFinite2(r->rootUV[i]) ||
            !RefFinite(r->rootT[i]) || !RefFinite(r->rootB[i]) || !RefFinite(r->rootN[i]))
            return ScatterGrowStatus::NonFiniteInput;
        for (size_t j = 0; j < i; ++j)
            if (r->stableIds[j] == r->stableIds[i])
                return ScatterGrowStatus::DuplicateStableId;
        float3 direction = c.direction == ScatterGrowDirection::RootNormal ? r->rootN[i] :
            c.direction == ScatterGrowDirection::RootTangent ? r->rootT[i] : c.literalDirection;
        direction = RefRot(RefNorm(direction), r->rootB[i], c.lift);
        float const azimuth = c.azimuth + c.azimuthRandom * 360.0f *
            (RefDraw(c.seed, r->stableIds[i], 0x4772417Au) - 0.5f);
        direction = RefRot(direction, r->rootN[i], azimuth);
        if (!RefFinite(direction)) return ScatterGrowStatus::NonFiniteInput;
        double const targetDouble = c.length *
            (c.randomLo + double(RefDraw(c.seed, r->stableIds[i])) *
             (c.randomHi - c.randomLo));
        float const target = static_cast<float>(targetDouble);
        if (!std::isfinite(target)) return ScatterGrowStatus::NonFiniteInput;
        float const t = float(c.cvCount - 1) / float(c.cvCount - 1);
        float const distance = target * t;
        float3 const output = make_float3(
            r->positions[i].x + direction.x * distance,
            r->positions[i].y + direction.y * distance,
            r->positions[i].z + direction.z * distance);
        if (!std::isfinite(distance) || !RefFinite(output))
            return ScatterGrowStatus::NonFiniteInput;
    }
    return ScatterGrowStatus::Ok;
}

static bool CheckThreadedResolve();

int main()
{
    size_t total = 0;
    // Valid roots: Ok + exact point total, deterministic across calls.
    {
        auto r = Roots(1000);
        auto c = Controls();
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &total) ==
                  ScatterGrowStatus::Ok &&
              total == 1000 * 8);
        size_t again = 0;
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &again) ==
                  ScatterGrowStatus::Ok &&
              again == total);
    }
    // Empty roots validate with zero points.
    {
        auto r = Roots(0);
        CHECK(CudaScatterGrow::ValidateRoots(r, Controls(), &total) ==
                  ScatterGrowStatus::Ok &&
              total == 0);
    }
    // Duplicate stable ids, including at the tail.
    {
        auto r = Roots(100);
        r->stableIds[99] = r->stableIds[0];
        CHECK(CudaScatterGrow::ValidateRoots(r, Controls(), &total) ==
              ScatterGrowStatus::DuplicateStableId);
    }
    // Fold collision without a true duplicate: distinct ids sharing the
    // radix leg's 32-bit fold must not report DuplicateStableId.
    {
        auto r = Roots(4);
        r->stableIds[1] = (uint64_t(1) << 32) | 0u;  // fold 1, != ids[0]
        size_t tRef = 0, tGot = 0;
        CHECK(ReferenceValidate(r, Controls(), &tRef) ==
                  ScatterGrowStatus::Ok &&
              CudaScatterGrow::ValidateRoots(r, Controls(), &tGot) ==
                  ScatterGrowStatus::Ok &&
              tRef == tGot);
    }
    // Oversized fold group (>64 members): 100 distinct ids on one fold
    // with a true dup pair hidden inside; the group-resolve path must
    // still agree with the linear reference.
    {
        auto r = Roots(200);
        for (size_t i = 0; i < 100; ++i)
            r->stableIds[i] = (uint64_t(i) << 32) | uint64_t(i);  // fold 0
        r->stableIds[50] = r->stableIds[7];
        size_t tRef = 0, tGot = 0;
        CHECK(ReferenceValidate(r, Controls(), &tRef) ==
                  ScatterGrowStatus::DuplicateStableId &&
              CudaScatterGrow::ValidateRoots(r, Controls(), &tGot) ==
                  ScatterGrowStatus::DuplicateStableId);
    }
    // ... same group without the dup: all folds collide, all ids
    // distinct, so the input is valid.
    {
        auto r = Roots(200);
        for (size_t i = 0; i < 200; ++i)
            r->stableIds[i] = (uint64_t(i) << 32) | uint64_t(i);  // fold 0
        size_t tRef = 0, tGot = 0;
        CHECK(ReferenceValidate(r, Controls(), &tRef) ==
                  ScatterGrowStatus::Ok &&
              CudaScatterGrow::ValidateRoots(r, Controls(), &tGot) ==
                  ScatterGrowStatus::Ok &&
              tRef == tGot && tGot == 200 * 8);
    }
    // Check order is index order: an earlier NonFinite beats a later dup.
    {
        auto r = Roots(100);
        r->stableIds[6] = r->stableIds[5];
        r->positions[0] = make_float3(HUGE_VALF, 0.0f, 0.0f);
        CHECK(CudaScatterGrow::ValidateRoots(r, Controls(), &total) ==
              ScatterGrowStatus::NonFiniteInput);
    }
    // ... and an earlier dup beats a later NonFinite.
    {
        auto r = Roots(100);
        r->stableIds[6] = r->stableIds[5];
        r->positions[9] = make_float3(HUGE_VALF, 0.0f, 0.0f);
        CHECK(CudaScatterGrow::ValidateRoots(r, Controls(), &total) ==
              ScatterGrowStatus::DuplicateStableId);
    }
    // Non-finite inputs in every plane.
    {
        auto c = Controls();
        auto r = Roots(10);
        r->rootUV[3] = make_float2(0.0f, NAN);
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &total) ==
              ScatterGrowStatus::NonFiniteInput);
        r = Roots(10);
        r->rootT[3] = make_float3(0.0f, 0.0f, HUGE_VALF);
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &total) ==
              ScatterGrowStatus::NonFiniteInput);
        r = Roots(10);
        r->rootB[3] = make_float3(NAN, 0.0f, 0.0f);
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &total) ==
              ScatterGrowStatus::NonFiniteInput);
        r = Roots(10);
        r->rootN[3] = make_float3(0.0f, HUGE_VALF, 0.0f);
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &total) ==
              ScatterGrowStatus::NonFiniteInput);
    }
    // Topology mismatch.
    {
        auto r = Roots(10);
        r->stableIds.pop_back();
        CHECK(CudaScatterGrow::ValidateRoots(r, Controls(), &total) ==
              ScatterGrowStatus::InvalidTopology);
    }
    // Bad controls, one fault at a time.
    {
        auto r = Roots(4);
        auto c = Controls();
        c.cvCount = 1;
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &total) ==
              ScatterGrowStatus::InvalidArgument);
        c = Controls();
        c.cvCount = 65;
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &total) ==
              ScatterGrowStatus::InvalidArgument);
        c = Controls();
        c.length = -1.0;
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &total) ==
              ScatterGrowStatus::InvalidArgument);
        c = Controls();
        c.lift = 100.0f;
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &total) ==
              ScatterGrowStatus::InvalidArgument);
        c = Controls();
        c.direction = ScatterGrowDirection::Literal;
        c.literalDirection = make_float3(NAN, 0.0f, 0.0f);
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &total) ==
              ScatterGrowStatus::InvalidArgument);
        c = Controls();
        c.azimuthRandom = 2.0f;
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &total) ==
              ScatterGrowStatus::InvalidArgument);
    }
    // Null roots fail closed.
    {
        std::shared_ptr<const ScatterGrowRoots> null;
        CHECK(CudaScatterGrow::ValidateRoots(null, Controls(), &total) ==
              ScatterGrowStatus::InvalidArgument);
    }
    // Overflow at full extension fails; exact cancellation passes.
    {
        auto c = Controls();
        c.length = double(FLT_MAX);
        c.randomLo = 1.0;
        c.randomHi = 1.0;
        auto r = Roots(2);
        r->positions[1] = make_float3(FLT_MAX, 0.0f, 0.0f);
        r->rootN[1] = make_float3(1.0f, 0.0f, 0.0f);
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &total) ==
              ScatterGrowStatus::NonFiniteInput);
        r = Roots(2);
        r->positions[1] = make_float3(FLT_MAX, 0.0f, 0.0f);
        r->rootN[1] = make_float3(-1.0f, 0.0f, 0.0f);
        CHECK(CudaScatterGrow::ValidateRoots(r, c, &total) ==
                  ScatterGrowStatus::Ok &&
              total == 2 * 8);
    }
    // Unswapped random range (BeginFresh swaps; ValidateRoots does not).
    {
        auto c = Controls();
        c.randomLo = 1.2;
        c.randomHi = 0.8;
        CHECK(CudaScatterGrow::ValidateRoots(Roots(16), c, &total) ==
                  ScatterGrowStatus::Ok &&
              total == 16 * 8);
    }
    // Differential fuzz vs the interleaved reference: random multi-fault
    // inputs must agree on status (and on the point total when Ok), so
    // the minima+radix combine keeps exact index-order precedence.
    {
        std::mt19937_64 rng(0x51ab51ab51ab51abull);
        auto pick = [&](std::initializer_list<float> vs) {
            return *(vs.begin() + size_t(rng() % vs.size()));
        };
        int sawOk = 0, sawNonFinite = 0, sawDup = 0;
        for (int trial = 0; trial < 30000; ++trial) {
            size_t const n = size_t(rng() % 13);
            auto r = std::make_shared<ScatterGrowRoots>();
            uint64_t const pool = 1 + size_t(rng() % 5);
            for (size_t i = 0; i < n; ++i) {
                float const px = pick({0.0f, 1.0f, -2.0f, 1e30f, float(FLT_MAX),
                                       std::numeric_limits<float>::infinity(),
                                       std::nanf("")});
                r->positions.push_back(make_float3(px, 0.0f, 0.0f));
                r->stableIds.push_back(rng() % pool);
                r->rootPrim.push_back(0);
                float const uv = pick({0.0f, 0.5f,
                                       std::numeric_limits<float>::infinity()});
                r->rootUV.push_back(make_float2(uv, 0.0f));
                float const frame = pick({0.0f, 1.0f,
                                          std::numeric_limits<float>::quiet_NaN()});
                r->rootT.push_back(make_float3(1.0f, 0.0f, 0.0f));
                r->rootB.push_back(make_float3(0.0f, 1.0f, frame));
                r->rootN.push_back(make_float3(0.0f, 0.0f, 1.0f));
            }
            auto c = Controls();
            c.cvCount = uint32_t(2 + rng() % 8);
            c.seed = int(rng() % 7) - 3;
            c.length = pick({1.0f, 1e30f, float(FLT_MAX)});
            c.randomLo = pick({0.0, 0.8, 1.0});
            c.randomHi = pick({0.8, 1.0, 1.2, 2.0});
            c.lift = pick({0.0f, 45.0f, -90.0f});
            c.azimuth = pick({0.0f, 180.0f, -360.0f});
            c.azimuthRandom = pick({0.0f, 0.5f, 1.0f});
            size_t tRef = 0, tGot = 0;
            ScatterGrowStatus const want =
                ReferenceValidate(r, c, &tRef);
            ScatterGrowStatus const got =
                CudaScatterGrow::ValidateRoots(r, c, &tGot);
            if (want != got || (want == ScatterGrowStatus::Ok && tRef != tGot)) {
                std::fprintf(stderr,
                             "FUZZ mismatch trial %d n=%zu: want %d got %d\n",
                             trial, n, int(want), int(got));
                return 1;
            }
            sawOk += want == ScatterGrowStatus::Ok;
            sawNonFinite += want == ScatterGrowStatus::NonFiniteInput;
            sawDup += want == ScatterGrowStatus::DuplicateStableId;
        }
        CHECK(sawOk > 100 && sawNonFinite > 100 && sawDup > 100);
        // Radix/scan at scale: medium random input with heavy collisions.
        {
            auto r = Roots(2000);
            for (size_t i = 0; i < 2000; ++i)
                r->stableIds[i] = uint64_t(rng() % 500);
            size_t tRef = 0, tGot = 0;
            CHECK(ReferenceValidate(r, Controls(), &tRef) ==
                  CudaScatterGrow::ValidateRoots(r, Controls(), &tGot) &&
                  tRef == tGot);
        }
    }
    // Adversarial magnitudes for the bounded-math fast path: degenerate
    // frames, subnormals, boundary tBound/Pmax values, and literal
    // directions must agree with the reference exactly (unique ids and
    // clean UV isolate the math-leg verdicts).
    {
        std::mt19937_64 rng(0xad9ad9ad9ad9ad9ull);
        auto pick = [&](std::initializer_list<float> vs) {
            return *(vs.begin() + size_t(rng() % vs.size()));
        };
        auto pickd = [&](std::initializer_list<double> vs) {
            return *(vs.begin() + size_t(rng() % vs.size()));
        };
        float const fmax = std::numeric_limits<float>::max();
        int sawOk = 0, sawNonFinite = 0;
        for (int trial = 0; trial < 5000; ++trial) {
            size_t const n = size_t(rng() % 9);
            auto r = std::make_shared<ScatterGrowRoots>();
            for (size_t i = 0; i < n; ++i) {
                float const px = pick({0.0f, -0.0f, 1.0f, fmax / 4.0f,
                                       fmax / 2.0f, fmax, 1e30f, 1e-30f,
                                       std::numeric_limits<float>::denorm_min(),
                                       std::numeric_limits<float>::infinity(),
                                       std::nanf("")});
                float const py = pick({0.0f, -0.0f, 1.0f, -1.0f});
                r->positions.push_back(make_float3(px, py, 0.0f));
                r->stableIds.push_back(i);
                r->rootPrim.push_back(0);
                r->rootUV.push_back(make_float2(0.0f, 0.0f));
                float const f = pick({0.0f, 1.0f, 1e-30f, 1e30f, fmax,
                                      std::numeric_limits<float>::denorm_min()});
                int const which = int(rng() % 4);
                float3 t = make_float3(1.0f, 0.0f, 0.0f);
                float3 b = make_float3(0.0f, 1.0f, 0.0f);
                float3 nn = make_float3(0.0f, 0.0f, 1.0f);
                if (which == 1) t = make_float3(f, 0.0f, 0.0f);
                if (which == 2) b = make_float3(0.0f, f, 0.0f);
                if (which == 3) nn = make_float3(0.0f, 0.0f, f);
                r->rootT.push_back(t);
                r->rootB.push_back(b);
                r->rootN.push_back(nn);
            }
            auto c = Controls();
            c.cvCount = uint32_t(2 + rng() % 63);
            c.length = pickd({1.0, double(fmax) / 1024.0, double(fmax) / 8.0,
                              double(fmax), 1e30});
            c.randomLo = pick({0.0f, 1.0f});
            c.randomHi = pick({1.0f, 2.0f});
            c.lift = pick({0.0f, 45.0f, -90.0f, 90.0f});
            c.azimuth = pick({0.0f, 180.0f, -360.0f, 360.0f});
            c.azimuthRandom = pick({0.0f, 1.0f});
            int const dir = int(rng() % 3);
            c.direction = dir == 0 ? ScatterGrowDirection::RootNormal
                : dir == 1         ? ScatterGrowDirection::RootTangent
                                   : ScatterGrowDirection::Literal;
            float const lf = pick({0.0f, 1.0f, 1e30f, fmax});
            c.literalDirection = make_float3(lf, 0.0f, 0.0f);
            size_t tRef = 0, tGot = 0;
            ScatterGrowStatus const want = ReferenceValidate(r, c, &tRef);
            ScatterGrowStatus const got =
                CudaScatterGrow::ValidateRoots(r, c, &tGot);
            if (want != got || (want == ScatterGrowStatus::Ok && tRef != tGot)) {
                std::fprintf(stderr,
                             "ADVERSARIAL mismatch trial %d n=%zu: want %d got %d\n",
                             trial, n, int(want), int(got));
                return 1;
            }
            sawOk += want == ScatterGrowStatus::Ok;
            sawNonFinite += want == ScatterGrowStatus::NonFiniteInput;
        }
        CHECK(sawOk > 100 && sawNonFinite > 100);
    }
    // Fused positions scan at vector scale: the finite/max fusion must
    // agree with the reference when faults land on 16-word group edges,
    // when max candidates are subnormal or signed zero, at the Pmax
    // fast/slow boundary, and when duplicates precede finite faults
    // (precedence through the poisoned fused prefix).
    {
        float const fmax = std::numeric_limits<float>::max();
        float const denorm = std::numeric_limits<float>::denorm_min();
        float const inf = std::numeric_limits<float>::infinity();
        float const qnan = std::numeric_limits<float>::quiet_NaN();
        float const snan = std::numeric_limits<float>::signaling_NaN();
        auto check = [&](std::shared_ptr<ScatterGrowRoots> const &r,
                         ScatterGrowControls const &c, char const *what) {
            size_t tRef = 0, tGot = 0;
            ScatterGrowStatus const want = ReferenceValidate(r, c, &tRef);
            ScatterGrowStatus const got =
                CudaScatterGrow::ValidateRoots(r, c, &tGot);
            if (want != got ||
                (want == ScatterGrowStatus::Ok && tRef != tGot)) {
                std::fprintf(stderr, "FUSED mismatch (%s): want %d got %d\n",
                             what, int(want), int(got));
                return 1;
            }
            return 0;
        };
        // Single positions faults on group edges: words 15/16/17 split
        // the first two 16-word groups; stride 3 crosses lanes mid-group.
        size_t const n = 100;  // 300 words: 18 groups + 12-word tail
        size_t const words[] = {0, 1, 2, 3, 13, 14, 15, 16, 17, 18,
                                31, 32, 33, 47, 48, 49, 63, 64, 65,
                                143, 144, 145, 191, 192, 193, 287, 288,
                                289, 296, 297, 298, 299};
        float const faults[] = {qnan, snan, inf, -inf};
        for (size_t w : words) {
            for (float fv : faults) {
                size_t const bad = w / 3;
                // No duplicates: the finite fault decides.
                {
                    auto r = Roots(n);
                    reinterpret_cast<float *>(r->positions.data())[w] = fv;
                    if (check(r, Controls(), "edge")) return 1;
                }
                // Duplicate strictly before the fault: dup wins even
                // though the fused max is poisoned past it.
                if (bad > 1) {
                    auto r = Roots(n);
                    reinterpret_cast<float *>(r->positions.data())[w] = fv;
                    r->stableIds[bad - 1] = r->stableIds[0];
                    if (check(r, Controls(), "edge-dup-before")) return 1;
                }
                // Duplicate strictly after the fault: finite wins.
                if (bad + 2 < n) {
                    auto r = Roots(n);
                    reinterpret_cast<float *>(r->positions.data())[w] = fv;
                    r->stableIds[n - 1] = r->stableIds[bad + 1];
                    if (check(r, Controls(), "edge-dup-after")) return 1;
                }
            }
        }
        // Max-candidate sweep: the fused integer max must order
        // subnormals, signed zeros, and huge finites exactly like fmax.
        {
            float const cands[] = {0.0f, -0.0f, denorm, -denorm,
                                   1.0f, -1.0f, 1e30f, fmax / 4.0f,
                                   fmax / 2.0f, fmax};
            for (float mc : cands) {
                auto r = Roots(64);  // 192 words: 12 groups, no tail
                // Plant the candidate pair at a group-edge lane pair;
                // everything else stays small and unique.
                reinterpret_cast<float *>(r->positions.data())[16] = mc;
                reinterpret_cast<float *>(r->positions.data())[17] = -mc;
                if (check(r, Controls(), "maxcand")) return 1;
            }
        }
        // Pmax fast/slow boundary: posMax + 128*tBound against
        // FLT_MAX/4 from both sides, with and without an early dup.
        {
            double const outLim = double(fmax) / 4.0;
            double const tB = 1.0 * 1.2;  // length * max(lo, hi)
            double const edge = outLim - 128.0 * tB;
            float const bounds[] = {float(edge * (1.0 - 1e-6)), float(edge),
                                    float(edge * (1.0 + 1e-6)),
                                    float(edge - 1.0e31),
                                    float(edge + 1.0e31)};
            for (float pb : bounds) {
                auto r = Roots(n);
                r->positions[37] = make_float3(pb, 0.0f, 0.0f);
                if (check(r, Controls(), "pbound")) return 1;
                // Early dup + slow path: math over [0, mLim) must
                // still clear toward DuplicateStableId.
                auto rd = Roots(n);
                rd->positions[37] = make_float3(pb, 0.0f, 0.0f);
                rd->stableIds[50] = rd->stableIds[0];
                if (check(rd, Controls(), "pbound-dup")) return 1;
            }
        }
        // Integer-max claim the fusion rests on: max over sign-cleared
        // bits must equal the fmax chain bit-for-bit on finite lanes
        // (magnitude order is integer order), and any non-finite lane
        // must poison the max toward exponent 0xFF (slow path).
        {
            auto chain = [](uint32_t const *ws, size_t count) {
                float m = 0.0f;
                for (size_t i = 0; i < count; ++i) {
                    float f = 0.0f;
                    uint32_t w = ws[i];
                    std::memcpy(&f, &w, sizeof(float));
                    m = std::fmax(m, std::fabs(f));
                }
                return m;
            };
            auto bits = [](uint32_t const *ws, size_t count) {
                uint32_t mx = 0;
                for (size_t i = 0; i < count; ++i) {
                    uint32_t const mag = ws[i] & 0x7FFFFFFFu;
                    mx = mag > mx ? mag : mx;
                }
                return mx;
            };
            auto equal = [&](uint32_t const *ws, size_t count) {
                float f = 0.0f;
                uint32_t b = bits(ws, count);
                std::memcpy(&f, &b, sizeof(float));
                float c = chain(ws, count);
                uint32_t cb = 0, fb = 0;
                std::memcpy(&cb, &c, sizeof(float));
                std::memcpy(&fb, &f, sizeof(float));
                return cb == fb;
            };
            uint32_t const directed[] = {
                0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
                0x007FFFFFu, 0x807FFFFFu, 0x00800000u, 0x80800000u,
                0x3F800000u, 0xBF800000u, 0x7F7FFFFFu, 0xFF7FFFFFu,
                0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u,
                0x7F800001u, 0xFF800001u};
            CHECK(equal(directed, 12));  // finite lanes: bitwise equal
            // Any non-finite lane poisons the max to exponent 0xFF.
            CHECK((bits(directed, 18) & 0x7F800000u) == 0x7F800000u);
            CHECK((bits(directed + 12, 2) & 0x7F800000u) == 0x7F800000u);
            CHECK((bits(directed + 14, 2) & 0x7F800000u) == 0x7F800000u);
            CHECK((bits(directed + 16, 2) & 0x7F800000u) == 0x7F800000u);
            std::mt19937_64 brng(0xb17ac517ac517abu);
            for (int trial = 0; trial < 20000; ++trial) {
                uint32_t ws[17];
                for (int k = 0; k < 17; ++k) {
                    uint32_t w = uint32_t(brng());
                    // Mix exponents densely near boundaries too.
                    if ((brng() & 7) == 0)
                        w = (w & 0x807FFFFFu) |
                            (uint32_t(brng() % 3 == 0 ? 0x00 : 0xFE) << 23);
                    ws[k] = w;
                }
                bool anyBad = false;
                for (int k = 0; k < 17; ++k)
                    anyBad |= (ws[k] & 0x7F800000u) == 0x7F800000u;
                if (anyBad)
                    CHECK((bits(ws, 17) & 0x7F800000u) == 0x7F800000u);
                else
                    CHECK(equal(ws, 17));
            }
        }
        // Randomized multi-fault sweep at vector scale: faults and
        // dups scattered across groups must keep exact precedence.
        {
            std::mt19937_64 rng(0x5ca1ab1e5ca1ab1eull);
            auto pick = [&](std::initializer_list<float> vs) {
                return *(vs.begin() + size_t(rng() % vs.size()));
            };
            for (int trial = 0; trial < 2000; ++trial) {
                size_t const m = 200;
                auto r = std::make_shared<ScatterGrowRoots>();
                uint64_t const pool = 1 + size_t(rng() % 199);
                for (size_t i = 0; i < m; ++i) {
                    float const px = pick({0.0f, -0.0f, 1.0f, -2.0f,
                                           denorm, -denorm, 1e30f, fmax,
                                           inf, -inf, qnan, snan});
                    float const py = pick({0.0f, 1.0f, -1.0f, inf, qnan});
                    r->positions.push_back(make_float3(px, py, 0.0f));
                    r->stableIds.push_back(rng() % pool);
                    r->rootPrim.push_back(0);
                    float const uv = pick({0.0f, 0.5f, inf, qnan});
                    r->rootUV.push_back(make_float2(uv, 0.0f));
                    float const fz = pick({0.0f, 1.0f, qnan});
                    r->rootT.push_back(make_float3(1.0f, 0.0f, 0.0f));
                    r->rootB.push_back(make_float3(0.0f, 1.0f, fz));
                    r->rootN.push_back(make_float3(0.0f, 0.0f, 1.0f));
                }
                auto c = Controls();
                c.length = pick({1.0f, 1e30f, fmax});
                if (check(r, c, "scale-fuzz")) return 1;
            }
        }
    }
    CHECK(CheckThreadedResolve());
    std::printf("testUsdGenCudaScatterGrowValidate: PASS\n");
    return 0;
}

// Threaded dup-resolve equivalence (r26): past 32768 ids the fold-group
// resolution chunks over workers with a serial boundary pass. Every case
// below runs under 1..8-worker arenas (chunk counts 1..8) with identical
// verdicts. Statuses pin the duplicate minimum exactly through its mLim
// coupling: a wrong-small minimum flips the finite-tie case to Dup, a
// wrong-large minimum flips the overflow-before-dup case to NonFinite.
static uint64_t SwapHalves(uint64_t v)
{
    return (v << 32) | (v >> 32);
}

static bool CheckResolveCase(std::shared_ptr<const ScatterGrowRoots> const &r,
                             ScatterGrowControls const &c,
                             ScatterGrowStatus want, size_t wantTotal)
{
    for (int workers = 1; workers <= 8; ++workers) {
        tbb::task_arena arena(workers);
        ScatterGrowStatus got = ScatterGrowStatus::CudaError;
        size_t total = 0;
        arena.execute([&] {
            got = CudaScatterGrow::ValidateRoots(r, c, &total);
        });
        if (got != want)
            return false;
        if (want == ScatterGrowStatus::Ok && total != wantTotal)
            return false;
    }
    return true;
}

static bool CheckThreadedResolve()
{
    size_t const n = 40000;  // past the 32768 threading threshold
    ScatterGrowControls c = Controls();
    // Unique ids: clean parallel resolve, scans, and fast-path math.
    if (!CheckResolveCase(Roots(n), c, ScatterGrowStatus::Ok, n * 8))
        return false;
    // One dup pair at the extremes: full-range scan, last-index minimum.
    {
        auto r = Roots(n);
        r->stableIds[n - 1] = r->stableIds[0];
        if (!CheckResolveCase(r, c, ScatterGrowStatus::DuplicateStableId, 0))
            return false;
    }
    // All equal: one group spanning every boundary at every chunking,
    // through the oversized-group sort path.
    {
        auto r = Roots(n);
        for (size_t i = 0; i < n; ++i)
            r->stableIds[i] = 7;
        if (!CheckResolveCase(r, c, ScatterGrowStatus::DuplicateStableId, 0))
            return false;
    }
    // Nineteen medium groups (first 2000 unique): minima spread across
    // chunks, several groups straddling boundaries at most chunkings.
    {
        auto r = Roots(n);
        for (size_t g = 1; g < 20; ++g)
            for (size_t i = g * 2000; i < (g + 1) * 2000; ++i)
                r->stableIds[i] = 1000000 + g;
        if (!CheckResolveCase(r, c, ScatterGrowStatus::DuplicateStableId, 0))
            return false;
    }
    // Fold collisions that are NOT duplicates: swapped halves share the
    // lo^hi fold but differ as full ids, so the resolve must accept.
    {
        auto r = Roots(n);
        for (size_t k = 0; k < 1000; ++k) {
            uint64_t const id = 2 * k + 1;
            r->stableIds[2 * k] = id;
            r->stableIds[2 * k + 1] = SwapHalves(id);
        }
        if (!CheckResolveCase(r, c, ScatterGrowStatus::Ok, n * 8))
            return false;
    }
    // Minimum-exactness through mLim: dup at 100 hides the overflow at
    // 200 (DuplicateStableId); dup at 300 exposes it (NonFiniteInput).
    {
        auto cOf = Controls();
        cOf.length = double(FLT_MAX);
        cOf.randomLo = 1.0;
        cOf.randomHi = 1.0;
        auto hidden = Roots(n);
        hidden->stableIds[100] = hidden->stableIds[0];
        hidden->positions[200] = make_float3(FLT_MAX, 0.0f, 0.0f);
        hidden->rootN[200] = make_float3(1.0f, 0.0f, 0.0f);
        if (!CheckResolveCase(hidden, cOf, ScatterGrowStatus::DuplicateStableId,
                              0))
            return false;
        auto exposed = Roots(n);
        exposed->stableIds[300] = exposed->stableIds[0];
        exposed->positions[200] = make_float3(FLT_MAX, 0.0f, 0.0f);
        exposed->rootN[200] = make_float3(1.0f, 0.0f, 0.0f);
        if (!CheckResolveCase(exposed, cOf, ScatterGrowStatus::NonFiniteInput,
                              0))
            return false;
    }
    // Finite-tie: dup and NaN share index 500, finite wins ties. A
    // wrong-small duplicate minimum would flip this to DuplicateStableId.
    {
        auto r = Roots(n);
        r->stableIds[500] = r->stableIds[0];
        r->positions[500] = make_float3(
            std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f);
        if (!CheckResolveCase(r, c, ScatterGrowStatus::NonFiniteInput, 0))
            return false;
    }
    // Skewed shards with a cross-shard minimum: 40% of the ids share
    // shard 0 (fold 0), so the sharded path runs lopsided but below the
    // n/2 fallback line. The dup at 15000 (shard 0) hides the overflow
    // at 20000 (DuplicateStableId); a shard-0 miss would expose it
    // (dBad 35000 scans the overflow -> NonFiniteInput), so the verdict
    // pins the cross-shard minimum and its mLim coupling.
    {
        auto cOf = Controls();
        cOf.length = double(FLT_MAX);
        cOf.randomLo = 1.0;
        cOf.randomHi = 1.0;
        auto r = Roots(n);
        for (size_t i = 0; i < 16000; ++i)
            r->stableIds[i] = (uint64_t(i) << 32) | uint64_t(i);
        r->stableIds[15000] = r->stableIds[5];
        r->stableIds[35000] = r->stableIds[20000];
        r->positions[20000] = make_float3(FLT_MAX, 0.0f, 0.0f);
        r->rootN[20000] = make_float3(1.0f, 0.0f, 0.0f);
        if (!CheckResolveCase(r, cOf, ScatterGrowStatus::DuplicateStableId,
                              0))
            return false;
    }
    // Small multi-occurrence groups (g=5): the <=64 resolve path with
    // repeated matches, in serial and every sharded chunking.
    {
        auto r = Roots(n);
        for (size_t g = 1; g <= 10; ++g)
            for (size_t j = 1; j < 5; ++j)
                r->stableIds[g * 3000 + j] = r->stableIds[g * 3000];
        if (!CheckResolveCase(r, c, ScatterGrowStatus::DuplicateStableId, 0))
            return false;
    }
    // Empty shards: ids confined to 4 of 8 shards, so half the shards
    // take the m<=1 skip while the rest sort; verdict stays Ok.
    {
        auto r = Roots(n);
        for (size_t i = 0; i < n; ++i)
            r->stableIds[i] = uint64_t((i / 4) * 8 + (i % 4) + 1);
        if (!CheckResolveCase(r, c, ScatterGrowStatus::Ok, n * 8))
            return false;
    }
    // Fallback line: 60% of the ids in shard 0 trips the n/2 guard, so
    // the legacy global path must still find the dup pair parked in a
    // thin shard.
    {
        auto r = Roots(n);
        for (size_t i = 0; i < 24000; ++i)
            r->stableIds[i] = (uint64_t(i) << 32) | uint64_t(i);
        r->stableIds[39000] = r->stableIds[25000];
        if (!CheckResolveCase(r, c, ScatterGrowStatus::DuplicateStableId, 0))
            return false;
    }
    // Constant low digit in every shard (r31): folds share their low 16
    // bits within each shard (fold s + 65536*j lands in shard s) while
    // the high digits vary, so every shard's pass 0 is the identity and
    // pass 1 still sorts. Balanced 5K/shard, so the sharded path runs.
    {
        auto r = Roots(n);
        for (size_t i = 0; i < n; ++i)
            r->stableIds[i] = uint64_t((i % 8) + 65536 * (i / 8));
        if (!CheckResolveCase(r, c, ScatterGrowStatus::Ok, n * 8))
            return false;
    }
    // ... with a dup pair parked non-adjacently in shard 0 (indices 0
    // and 16: positions 0 and 2 in shard-input order), so only the
    // pass-1 sort groups them. Skipping pass 1 (or a broken folded
    // vary) reports Ok instead of DuplicateStableId.
    {
        auto r = Roots(n);
        for (size_t i = 0; i < n; ++i)
            r->stableIds[i] = uint64_t((i % 8) + 65536 * (i / 8));
        r->stableIds[16] = r->stableIds[0];
        if (!CheckResolveCase(r, c, ScatterGrowStatus::DuplicateStableId, 0))
            return false;
    }
    return true;
}
