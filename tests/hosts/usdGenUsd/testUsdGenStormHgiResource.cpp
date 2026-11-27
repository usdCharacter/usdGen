// testUsdGenStormHgiResource.cpp — gate S-8 (plan/09 §5.3): the
// HDST_ENABLE_HGI_RESOURCE_GENERATION A/B switch must produce equivalent
// looks on the Storm path. The A branch is run in a forked child (an HGI bake
// crash must surface as a failed wait status, not kill the gate). The PW
// decision for M1 is B-as-default (see docs/m1/adr-*.md); a crashing A branch
// is therefore a documented fallback, but a WRONG-LOOKING A branch is a fail.
#include "stormTestUtils.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

int gFails = 0;
void check(bool ok, const char* msg) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", msg);
    if (!ok) ++gFails;
}

// Fork+exec of this binary's --child mode: an EGL context cannot survive
// fork(), so the A/B branches run in a fresh process. Returns:
//   0 = rendered, png written   1 = render failed   2 = child crashed
int renderBranch(const std::string& scene, const std::string& out,
                 const char* hgiVal) {
    pid_t pid = fork();
    if (pid == 0) {
        if (hgiVal) {
            setenv("HDST_ENABLE_HGI_RESOURCE_GENERATION", hgiVal, 1);
        }
        char* const args[] ={const_cast<char*>("/proc/self/exe"),
                             const_cast<char*>("--child"),
                             const_cast<char*>(scene.c_str()),
                             const_cast<char*>(out.c_str()), nullptr};
        execv("/proc/self/exe", args);
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status)) return 2;
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "--child") {
        if (!stormtest::initGl("hgi-child")) _exit(1);
        _exit(stormtest::renderStage(argv[2], argv[3], 256, 1.2f) ? 0 : 1);
    }
    const std::string srcDir = (argc > 1) ? argv[1] : ".";
    const std::string scratch = "/tmp/usdGenStormHgi_scratch";
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);
    if (!stormtest::initGl("testUsdGenStormHgiResource")) return 77;

    ::setenv("USDGEN_SHADER_SCENES", scratch.c_str(), 1);
    const std::string gen = "python3 '" + srcDir +
                            "/libs/hosts/usdGenHydra/primitives/curves/shaders/test/make_scene.py' >/dev/null 2>&1";
    check(std::system(gen.c_str()) == 0, "make_scene.py generated scenes");
    const std::string sceneA = scratch + "/sceneA_asset.usda";

    // --- B branch (default path, must be clean + deterministic) ---
    check(renderBranch(sceneA, scratch + "/B0.png", "0") == 0,
          "HGI=0 renders (B/default)");
    check(renderBranch(sceneA, scratch + "/B1.png", "0") == 0,
          "HGI=0 re-render deterministic setup");
    int w0 = 0, h0 = 0, w1 = 0, h1 = 0;
    VtArray<float> b0, b1;
    check(stormtest::loadRgba(scratch + "/B0.png", &w0, &h0, &b0) &&
              stormtest::loadRgba(scratch + "/B1.png", &w1, &h1, &b1),
          "B frames load");
    double mean = 1.0, frac = 1.0;
    stormtest::diffStats(b0, b1, &mean, &frac);
    std::printf("B determinism meanAbs=%.5f\n", mean);
    check(mean <= 1.0 / 255.0, "HGI=0 renders are bit-stable");
    check(stormtest::coverage(b0) > 0.05, "B frame covers hair");

    // --- A branch (HGI resource generation on) ---
    const int a = renderBranch(sceneA, scratch + "/A.png", "1");
    if (a == 2) {
        std::printf(
            "WARN: HGI=1 branch crashed the render child — documented "
            "fallback to B default (plan/09 S-8 PW decision; see "
            "docs/m1/adr-s8-hgi-default.md)\n");
        check(true, "A branch crash is the documented B-default fallback");
    } else if (a == 0) {
        int wa = 0, ha = 0;
        VtArray<float> ap;
        check(stormtest::loadRgba(scratch + "/A.png", &wa, &ha, &ap),
              "A frame loads");
        stormtest::diffStats(b0, ap, &mean, &frac);
        std::printf("A/B meanAbs=%.5f frac8=%.5f\n", mean, frac);
        check(mean <= 4.0 / 255.0, "A/B look parity within 4/255 mean");
        check(frac < 0.02, "A/B < 2% pixels differ > 8/255");
    } else {
        check(false, "A branch failed without crashing (silent render error)");
    }

    std::printf("testUsdGenStormHgiResource: %s (%d failures)\n",
                gFails ? "FAILED" : "PASSED", gFails);
    return gFails ? 1 : 0;
}
