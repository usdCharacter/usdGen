// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenVulkanScatterGrowCpuReject — pin BuildCpu's host rejection
// contract (r48): any non-finite float in any scanned input plane must
// report false with *output untouched, on both the serial (n <= 4096)
// and threaded (n > 4096) drivers, under every direction. Guards the
// pre-scan fusion, which moves these checks per-curve: a removal that
// drops a plane (Normalize3 substitutes instead of rejecting; rootUV is
// only ever copied) must fail here.
#include "usdGen/vulkan/scatterGrowPipeline.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "ScatterGrowCpuReject check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

namespace {
// Deterministic roots with a varied frame per root (mirrors the
// device test's Roots: rootT=(1,0,0), rootB=(0,0,1), rootN=(0,1,0)).
struct Roots {
    uint32_t curves;
    std::vector<float> positions;
    std::vector<uint64_t> stableIds;
    std::vector<int32_t> rootPrim;
    std::vector<float> rootUV;
    std::vector<float> rootT, rootB, rootN;
    Roots(uint32_t c) : curves(c) {
        stableIds.resize(c);
        rootPrim.resize(c);
        rootUV.resize(2 * c);
        rootT.resize(3 * c); rootB.resize(3 * c); rootN.resize(3 * c);
        for (uint32_t i = 0; i < c; ++i) {
            positions.push_back(float(i) * 10.0f + 1.0f);
            positions.push_back(float(i) * 2.0f + 0.5f);
            positions.push_back(0.25f + float(i));
            stableIds[i] = uint64_t(i + 1000);
            rootPrim[i] = int32_t(i * 3 + 1);
            rootUV[2 * i] = float(i) / float(c ? c : 1);
            rootUV[2 * i + 1] = 0.25f;
            rootT[3 * i] = 1.0f; rootT[3 * i + 1] = 0.0f; rootT[3 * i + 2] = 0.0f;
            rootB[3 * i] = 0.0f; rootB[3 * i + 1] = 0.0f; rootB[3 * i + 2] = 1.0f;
            rootN[3 * i] = 0.0f; rootN[3 * i + 1] = 1.0f; rootN[3 * i + 2] = 0.0f;
        }
    }
};

ScatterGrowPipeline::Controls MakeControls(uint32_t cv) {
    ScatterGrowPipeline::Controls controls;
    controls.cvCount = cv; controls.seed = 7; controls.length = 1.5;
    controls.randomLo = 0.5; controls.randomHi = 2.0;
    controls.lift = 0.0f; controls.fallbackWidth = 0.02f;
    controls.direction = ScatterGrowPipeline::Direction::RootNormal;
    return controls;
}

// Sentinel-fill every Outputs plane so "untouched" is bit-checkable.
void FillSentinel(ScatterGrowPipeline::Outputs* o) {
    o->points.assign(5, 1.25f); o->rest.assign(7, 2.5f);
    o->widths.assign(3, 3.75f); o->hairT.assign(11, 5.0f);
    o->offsets.assign(13, 0xA5A5A5A5u); o->ids.assign(17, 0xDEADBEEFDEADBEEFull);
    o->rootPrim.assign(19, -1234567); o->rootUV.assign(23, 6.25f);
    o->rootT.assign(29, 7.5f); o->rootB.assign(31, 8.75f); o->rootN.assign(37, 10.0f);
}

bool SameOutputs(ScatterGrowPipeline::Outputs const& a,
                 ScatterGrowPipeline::Outputs const& b) {
    return a.points == b.points && a.rest == b.rest && a.widths == b.widths &&
        a.hairT == b.hairT && a.offsets == b.offsets && a.ids == b.ids &&
        a.rootPrim == b.rootPrim && a.rootUV == b.rootUV && a.rootT == b.rootT &&
        a.rootB == b.rootB && a.rootN == b.rootN;
}

bool Run(Roots const& r, std::vector<float> const& targets,
         ScatterGrowPipeline::Controls const& controls,
         ScatterGrowPipeline::Outputs* out) {
    return ScatterGrowPipeline::BuildCpu(r.positions, r.stableIds, r.rootPrim,
        r.rootUV, r.rootT, r.rootB, r.rootN, targets, controls, out);
}
} // namespace

int main() {
    uint32_t const n = 64, cv = 8;
    Roots const roots(n);
    auto const controls = MakeControls(cv);
    auto const targets =
        ScatterGrowPipeline::BuildTargets(roots.stableIds, controls);

    // (a) Positive control: valid inputs grow; structural offsets land.
    {
        ScatterGrowPipeline::Outputs out;
        CHECK(Run(roots, targets, controls, &out));
        CHECK(out.offsets.size() == n + 1 && out.offsets[n] == n * cv);
        CHECK(out.points.size() == 3 * size_t(n) * cv);
        CHECK(out.ids.size() == n && out.ids[0] == roots.stableIds[0]);
    }

    // (b) Every scanned plane x {qNaN, +Inf, -Inf} x {first, middle,
    // last curve} x every lane of the curve tuple rejects with output
    // untouched. Planes: positions/T/B/N (stride 3), targets (stride
    // 1), rootUV (stride 2).
    float const poisons[3] = {std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    uint32_t const lanes[3] = {0, n / 2, n - 1};
    auto sweep = [&](std::vector<float> Roots::*plane, uint32_t stride,
                     char const* /*name*/) -> bool {
        for (float v : poisons)
            for (uint32_t c : lanes)
                for (uint32_t k = 0; k < stride; ++k) {
                    Roots bad = roots;
                    (bad.*plane)[size_t(c) * stride + k] = v;
                    ScatterGrowPipeline::Outputs out, before;
                    FillSentinel(&out); before = out;
                    if (Run(bad, targets, controls, &out)) {
                        std::fprintf(stderr, "plane accepted poison %g lane %u curve %u\n",
                            double(v), k, c);
                        return false;
                    }
                    if (!SameOutputs(out, before)) {
                        std::fprintf(stderr, "output touched on poison lane %u curve %u\n",
                            k, c);
                        return false;
                    }
                }
        return true;
    };
    CHECK(sweep(&Roots::positions, 3, "positions"));
    CHECK(sweep(&Roots::rootT, 3, "rootT"));
    CHECK(sweep(&Roots::rootB, 3, "rootB"));
    CHECK(sweep(&Roots::rootN, 3, "rootN"));
    CHECK(sweep(&Roots::rootUV, 2, "rootUV"));
    // Targets live outside Roots (host-precomputed); sweep separately.
    for (float v : poisons)
        for (uint32_t c : lanes) {
            auto badTargets = targets;
            badTargets[c] = v;
            ScatterGrowPipeline::Outputs out, before;
            FillSentinel(&out); before = out;
            CHECK(!Run(roots, badTargets, controls, &out));
            CHECK(SameOutputs(out, before));
        }

    // (c) Checks are direction-independent: the pre-scans ran before any
    // direction select, so poison must reject under Literal+lift+azimuth
    // (exercises both Rodrigues rotations on the poisoned frame) and
    // RootTangent alike. One poison value, middle curve, every lane.
    {
        auto lit = controls;
        lit.direction = ScatterGrowPipeline::Direction::Literal;
        lit.literalDirection[0] = 0.0f; lit.literalDirection[1] = 0.0f;
        lit.literalDirection[2] = 1.0f;
        lit.lift = 30.0f; lit.azimuth = 45.0f; lit.azimuthRandom = 1.0f;
        auto tan = controls;
        tan.direction = ScatterGrowPipeline::Direction::RootTangent;
        tan.lift = -15.0f;
        float const v = std::numeric_limits<float>::quiet_NaN();
        uint32_t const c = n / 2;
        std::vector<float> Roots::*planes[5] = {&Roots::positions, &Roots::rootT,
            &Roots::rootB, &Roots::rootN, &Roots::rootUV};
        uint32_t const strides[5] = {3, 3, 3, 3, 2};
        for (int p = 0; p < 5; ++p)
            for (uint32_t k = 0; k < strides[p]; ++k) {
                Roots bad = roots;
                (bad.*planes[p])[size_t(c) * strides[p] + k] = v;
                ScatterGrowPipeline::Outputs o1, o2, before;
                FillSentinel(&o1); FillSentinel(&o2); before = o1;
                CHECK(!Run(bad, targets, lit, &o1));
                CHECK(SameOutputs(o1, before));
                CHECK(!Run(bad, targets, tan, &o2));
                CHECK(SameOutputs(o2, before));
            }
        auto badTargets = targets;
        badTargets[c] = v;
        ScatterGrowPipeline::Outputs o1, o2, before;
        FillSentinel(&o1); FillSentinel(&o2); before = o1;
        CHECK(!Run(roots, badTargets, lit, &o1));
        CHECK(SameOutputs(o1, before));
        CHECK(!Run(roots, badTargets, tan, &o2));
        CHECK(SameOutputs(o2, before));
    }

    // (d) Malformed spans reject before any allocation: short/long planes.
    {
        Roots bad = roots;
        ScatterGrowPipeline::Outputs out, before;
        FillSentinel(&out); before = out;
        auto shortTargets = targets; shortTargets.pop_back();
        CHECK(!Run(roots, shortTargets, controls, &out));
        CHECK(SameOutputs(out, before));
        bad.rootUV.pop_back();
        CHECK(!Run(bad, targets, controls, &out));
        CHECK(SameOutputs(out, before));
        bad = roots; bad.positions.push_back(0.0f);
        CHECK(!Run(bad, targets, controls, &out));
        CHECK(SameOutputs(out, before));
        auto badCv = controls; badCv.cvCount = 1;
        CHECK(!Run(roots, targets, badCv, &out));
        CHECK(SameOutputs(out, before));
    }

    // (e) Threaded driver (n > 4096): valid grows, poison at either end
    // of the chunk range reports false with output untouched.
    {
        uint32_t const big = 5000;
        Roots const bigRoots(big);
        auto const bigTargets =
            ScatterGrowPipeline::BuildTargets(bigRoots.stableIds, controls);
        ScatterGrowPipeline::Outputs out;
        CHECK(Run(bigRoots, bigTargets, controls, &out));
        CHECK(out.offsets.size() == big + 1 && out.offsets[big] == big * cv);
        Roots bad = bigRoots;
        bad.rootB[3 * size_t(big - 1) + 2] =
            std::numeric_limits<float>::quiet_NaN();
        ScatterGrowPipeline::Outputs rej, before;
        FillSentinel(&rej); before = rej;
        CHECK(!Run(bad, bigTargets, controls, &rej));
        CHECK(SameOutputs(rej, before));
        bad = bigRoots;
        bad.rootUV[0] = std::numeric_limits<float>::infinity();
        FillSentinel(&rej); before = rej;
        CHECK(!Run(bad, bigTargets, controls, &rej));
        CHECK(SameOutputs(rej, before));
    }
    std::printf("ScatterGrowCpuReject: all rejection pins hold\n");
    return 0;
}
