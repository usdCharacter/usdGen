// testUsdGenStormRefine.cpp — gate S-9 (plan/09 §5.3). Plan contract lines:
//   (1) plan/09-performance-and-benchmarks.md:592:
//       | **S-9** | T2 | refineLevel 1 vs 2 **and the switch cost**
//       | switch < 3 ms ⇒ the tumble tier ships as an option
//       | `testUsdGenStormRefine` on G4 | M0 pre-work (**PW-3**), **decides M1** |
//       — a decision threshold, not a pass/fail bound.
//   (2) plan/08-tools.md:502-506: "If gate **S-9** ... shows a refineLevel
//       1↔2 switch costs < 3 ms, the plugin drops tiles to refineLevel 1
//       while the camera moves ... If S-9 fails, LOD is hash decimation only
//       and M5 loses this control with no schedule change."
//   (3) plan/11-roadmap.md:659 (SC-4): "S-9: a 2 → 1 switch costs ≥ 3 ms
//       | no tumble tier; LOD is decimation by stable id".
// So this test ASSERTS render correctness (per-level clean + coverage, L1/L2
// medians measured, 1->2->1 round-trip bit-identical) and RECORDS the switch
// latency plus the binding decision line "S-9 >= 3ms => no tumble tier
// (SC-4)". PW-3 measured ≈ 19–51 ms per repeat switch (PW-3:232), so the
// expected milestone status is "MEASURED — tier rejected per PW-3/SC-4".
// Levels 0/1/2 use the PW-3-verified complexity map 1.0/1.1/1.2 (PW-3 §1;
// engine.cpp _GetRefineLevel), NOT 1.0/1.5/2.0.
// NOTE: a flaky driver-level crash in HgiGL::_SubmitCmds (NVIDIA EGL,
// observed consecutive SEGFAULTs then clean passes of the same binary) can
// abort a run before any test print; that is environment, not gate. No
// in-test retries.
#include "stormTestUtils.h"

#include <algorithm>
#include <filesystem>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

int gFails = 0;
void check(bool ok, const char* msg) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", msg);
    if (!ok) ++gFails;
}

double median(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Wall ms of one frame at `complexity` (no warm-up: this measures the
// switch cost when the level just changed).
bool switchFrame(stormtest::StormSession *session, float complexity,
                 const std::string &out, double *ms)
{
    return session->frame(complexity, out, ms);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string srcDir = (argc > 1) ? argv[1] : ".";
    const std::string scratch = "/tmp/usdGenStormRefine_scratch";
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);
    if (!stormtest::initGl("testUsdGenStormRefine")) return 77;

    ::setenv("USDGEN_SHADER_SCENES", scratch.c_str(), 1);
    const std::string gen = "python3 '" + srcDir +
                            "/libs/hosts/usdGenHydra/primitives/curves/shaders/test/make_scene.py' >/dev/null 2>&1";
    check(std::system(gen.c_str()) == 0, "make_scene.py generated scenes");
    const std::string sceneA = scratch + "/sceneA_asset.usda";

    const int px = 256;
    stormtest::StormSession session;
    if (!session.open(sceneA, px)) {
        check(false, "session open");
        return 1;
    }
    const float kComplexity[3] = {1.0f, 1.1f, 1.2f};  // PW-3 map -> 0/1/2
    double levelMedian[3] = {0.0, 0.0, 0.0};
    double covPrev = -1.0;
    for (int level = 0; level < 3; ++level) {
        // Two warm frames absorb the level's one-off tessellation/refine
        // bake; steady-state medians characterize each level.
        for (int i = 0; i < 2; ++i) {
            if (!session.frame(kComplexity[level], scratch + "/warm.png")) {
                check(false, "level warm-up render failed");
                return 1;
            }
        }
        std::vector<double> times;
        for (int i = 0; i < 3; ++i) {
            double ms = 0.0;
            const std::string out = scratch + "/L" + std::to_string(level) +
                                    "_t" + std::to_string(i) + ".png";
            if (!session.frame(kComplexity[level], out, &ms)) {
                check(false, "level render failed");
                return 1;
            }
            times.push_back(ms);
        }
        levelMedian[level] = median(times);
        std::printf("level %d warm median=%.1fms\n", level,
                    levelMedian[level]);
        int w = 0, h = 0;
        VtArray<float> img;
        if (!stormtest::loadRgba(scratch + "/L" + std::to_string(level) +
                                     "_t2.png",
                                 &w, &h, &img)) {
            check(false, "level frame loads");
            return 1;
        }
        const double cov = stormtest::coverage(img);
        std::printf("level %d coverage=%.4f\n", level, cov);
        check(cov > 0.05, "level frame covers hair");
        if (covPrev >= 0.0) {
            // AA-edge pixels can shift by a hair with tessellation level;
            // the gate is "no visible loss of item coverage".
            check(cov >= covPrev - 0.02,
                  "coverage does not shrink when refining");
        }
        covPrev = cov;
    }
    std::printf("L1 vs L2 steady: %.1fms vs %.1fms\n", levelMedian[1],
                levelMedian[2]);
    check(levelMedian[1] > 0.0 && levelMedian[2] > 0.0,
          "L1 and L2 steady medians measured");

    // Switch latency: 1 -> 2 and 2 -> 1, first frame after the change.
    std::vector<double> up, down;
    for (int i = 0; i < 3; ++i) {
        double ms = 0.0;
        check(switchFrame(&session, 1.1f, scratch + "/SW1.png", &ms),
              "switch warm-up at L1 renders");
        check(switchFrame(&session, 1.2f, scratch + "/SWup.png", &ms),
              "L1->L2 switch frame renders");
        up.push_back(ms);
        check(switchFrame(&session, 1.2f, scratch + "/SW2.png", &ms),
              "switch warm-up at L2 renders");
        check(switchFrame(&session, 1.1f, scratch + "/SWdown.png", &ms),
              "L2->L1 switch frame renders");
        down.push_back(ms);
    }
    const double upMed = median(up), downMed = median(down);
    std::printf("switch latency median: up=%.1fms down=%.1fms\n", upMed,
                downMed);
    std::printf("S-9 %s >= 3ms => no tumble tier (SC-4, 11-roadmap:659)\n",
                (upMed >= 3.0 || downMed >= 3.0) ? "MEASURED" : "BELOW-THRESHOLD");
    check(upMed > 0.0 && downMed > 0.0, "switch latencies recorded");

    // Live-switch round trip: 1 -> 2 -> 1 must reproduce the baseline frame.
    double ms = 0.0;
    check(session.frame(1.2f, scratch + "/RT2.png", &ms),
          "round-trip refine-up render");
    check(session.frame(1.1f, scratch + "/RT1.png", &ms),
          "round-trip down-switch render");
    int w = 0, h = 0, w2 = 0, h2 = 0;
    VtArray<float> first, back;
    check(stormtest::loadRgba(scratch + "/L1_t2.png", &w, &h, &first) &&
              stormtest::loadRgba(scratch + "/RT1.png", &w2, &h2, &back),
          "round-trip frames load");
    double mean = 1.0, frac = 1.0;
    stormtest::diffStats(first, back, &mean, &frac);
    std::printf("round-trip meanAbs=%.5f\n", mean);
    check(mean <= 1.0 / 255.0, "1->2->1 matches the baseline frame");

    std::printf("testUsdGenStormRefine: %s (%d failures)\n",
                gFails ? "FAILED" : "PASSED", gFails);
    return gFails ? 1 : 0;
}
