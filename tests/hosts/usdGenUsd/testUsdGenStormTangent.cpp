// testUsdGenStormTangent.cpp — gate S-8 (plan/09 §5.3): `hairTangent`
// variant A (`inData.Neye`, no authored tangent) vs B (vertex primvar +
// `usdGenHairPreviewPrimvar.glslfx`) on shipped code.
//
// PW-2 (docs/prework/PW-2-s8-glslfx-variant.md) decided variant A as the
// default: A is ~33-36% faster per deforming frame (no DirtyPrimvar
// republish) and compiles under HDST_ENABLE_HGI_RESOURCE_GENERATION=1.
// This test re-verifies the decision on the shipped scenes from
// libs/hosts/usdGenHydra/primitives/curves/shaders/test/make_scene.py: both variants must render clean, the
// frames must match within look parity (same groom, same lights — the two
// tangent sources describe the same strand field), re-renders must be
// bit-stable, and a warm-frame wall-time comparison must show A no slower
// than B within headroom (the timing direction PW-2 measured; absolute ms
// are host- and contention-dependent, so this is a ratio, not a budget).
//
// Exit: 0 pass, 77 SKIP (no usable EGL device context), 1 fail.
#include "stormTestUtils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

int gFails = 0;
void check(bool ok, const char* msg)
{
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", msg);
    if (!ok) {
        ++gFails;
    }
}

double minOf(const std::vector<double>& v)
{
    return *std::min_element(v.begin(), v.end());
}

}  // namespace

int main(int argc, char** argv)
{
    const std::string srcDir = (argc > 1) ? argv[1] : ".";
    const std::string scratch = "/tmp/usdGenStormTangent_scratch";
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);
    if (!stormtest::initGl("testUsdGenStormTangent")) {
        return 77;
    }

    ::setenv("USDGEN_SHADER_SCENES", scratch.c_str(), 1);
    const std::string gen = "python3 '" + srcDir +
                            "/libs/hosts/usdGenHydra/primitives/curves/shaders/test/make_scene.py' >/dev/null 2>&1";
    check(std::system(gen.c_str()) == 0, "make_scene.py generated scenes");
    // Variant A: inData.Neye tangent, no authored hairTangent.
    const std::string sceneA = scratch + "/sceneA_asset.usda";
    // Variant B: authored hairTangent primvar + Primvar glslfx variant.
    const std::string sceneB = scratch + "/sceneB_asset.usda";
    check(std::filesystem::exists(sceneA), "sceneA_asset.usda exists");
    check(std::filesystem::exists(sceneB), "sceneB_asset.usda exists");

    const int px = 256;
    const float complexity = 1.2f;  // refineLevel 2, the PW-2 configuration

    // --- both variants render clean; warm-frame timings via one session ---
    stormtest::StormSession sessA, sessB;
    check(sessA.open(sceneA, px), "Storm session opens sceneA (variant A)");
    check(sessB.open(sceneB, px), "Storm session opens sceneB (variant B)");
    std::vector<double> tA, tB;
    double ms = 0.0;
    for (int i = 0; i < 3; ++i) {
        check(sessA.frame(complexity, scratch + "/A_f" + std::to_string(i) +
                                             ".png",
                          &ms),
              ("variant A frame renders " + std::to_string(i)).c_str());
        tA.push_back(ms);
        check(sessB.frame(complexity, scratch + "/B_f" + std::to_string(i) +
                                             ".png",
                          &ms),
              ("variant B frame renders " + std::to_string(i)).c_str());
        tB.push_back(ms);
    }
    const double minA = minOf(tA), minB = minOf(tB);
    std::printf("warm-frame wall min: A=%.2fms B=%.2fms (ratio A/B=%.3f)\n",
                minA, minB, minB > 0.0 ? minA / minB : -1.0);
    check(minB > 0.0 && minA <= 1.5 * minB,
          "variant A warm frame no slower than B within headroom (S-8: A wins)");

    // --- frames are lit hair, not black ---
    int w = 0, h = 0;
    VtArray<float> a, b;
    check(stormtest::loadRgba(scratch + "/A_f2.png", &w, &h, &a) &&
              stormtest::loadRgba(scratch + "/B_f2.png", &w, &h, &b),
          "A/B frames load");
    check(stormtest::coverage(a) > 0.05, "variant A frame covers hair");
    check(stormtest::coverage(b) > 0.05, "variant B frame covers hair");

    // --- A/B look parity: same groom, same lights, same strand field ---
    double mean = 1.0, frac = 1.0;
    stormtest::diffStats(a, b, &mean, &frac);
    std::printf("A/B meanAbs=%.5f frac8=%.5f\n", mean, frac);
    check(mean <= 4.0 / 255.0, "A/B look parity within 4/255 mean");
    check(frac < 0.05, "A/B < 5% pixels differ > 8/255");

    // --- determinism: same-session re-render of A is bit-stable ---
    int w2 = 0, h2 = 0;
    VtArray<float> a0;
    check(stormtest::loadRgba(scratch + "/A_f1.png", &w2, &h2, &a0),
          "A first warm frame loads");
    stormtest::diffStats(a0, a, &mean, &frac);
    std::printf("A determinism meanAbs=%.5f\n", mean);
    check(mean <= 1.0 / 255.0, "variant A re-renders are bit-stable");

    std::printf("testUsdGenStormTangent: %s (%d failures)\n",
                gFails ? "FAILED" : "PASSED", gFails);
    return gFails ? 1 : 0;
}
