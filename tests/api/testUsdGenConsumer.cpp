// testUsdGenConsumer — M0 downstream consumer check (plan §6.2, sol S-5/X-4).
//
// Proves the installed CMake package is usable on its own (12.5): configure +
// build a tiny consumer CMake project (tests/consumer/) that does
// find_package(usdGen) with only the usdGen prefix on CMAKE_PREFIX_PATH (no
// OpenUSD), links usdGen::usdGen, creates the engine, compiles an empty spec
// and runs a commit. This exercises the PATH_VARS-fixed usdGenConfig.cmake
// end to end (sol S-5/X-4).
//
// The install prefix is taken from the USDGEN_INSTALL_PREFIX environment
// variable (the exact name CI must set — the docs lane wires it in the
// install-check job of .github/workflows/usdgen.yml). When it is unset this
// test prints a note and returns 0 (clean skip).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

int Run(const std::string &cmd) {
    std::printf("+ %s\n", cmd.c_str());
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::printf("command failed (rc=%d): %s\n", rc, cmd.c_str());
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    const char *prefix = std::getenv("USDGEN_INSTALL_PREFIX");
    if (!prefix || !*prefix) {
        std::printf(
            "testUsdGenConsumer: USDGEN_INSTALL_PREFIX unset — skipping the "
            "downstream consumer build (set it to the scratch install prefix "
            "after running cmake --install).\n");
        return 0;
    }

    const std::string srcDir = USDGEN_TEST_CONSUMER_SRC;
    const std::string cmake  = USDGEN_TEST_CMAKE_COMMAND;

    // A private scratch build dir so repeated runs never collide.
    const std::string tmpl = "/tmp/usdgen-consumer-XXXXXX";
    char *tmp = mkdtemp(const_cast<char *>(tmpl.data()));
    if (!tmp) {
        std::printf("FAIL: mkdtemp failed\n");
        return 1;
    }
    const std::string buildDir = tmp;

    // The usdGen prefix only: usdGen::usdGen carries no OpenUSD dependency.
    const std::string prefixPath = std::string("\"") + prefix + "\"";

    std::string cfg = "rm -rf '" + buildDir + "' && mkdir -p '" + buildDir + "'";
    if (Run(cfg) != 0) return 1;

    cfg = cmake + " -S '" + srcDir + "' -B '" + buildDir + "'"
          " -DCMAKE_BUILD_TYPE=Release"
          " -DCMAKE_PREFIX_PATH=" + prefixPath;
    if (Run(cfg) != 0) {
        std::printf("FAIL: consumer configure failed\n");
        return 1;
    }

    std::string build = cmake + " --build '" + buildDir + "' -j4";
    if (Run(build) != 0) {
        std::printf("FAIL: consumer build failed\n");
        return 1;
    }

    // Run the consumer: version, engine, empty-spec compile and commit.
    std::string out = buildDir + "/consumer > '" + buildDir + "/out.txt' 2>&1";
    if (Run(out) != 0) {
        std::printf("FAIL: consumer run failed\n");
        return 1;
    }
    out = "cat '" + buildDir + "/out.txt'; grep -q 'consumer: PASS' '" + buildDir + "/out.txt'";
    if (Run(out) != 0) {
        std::printf("FAIL: consumer output missing its PASS line\n");
        return 1;
    }

    std::printf("PASS: downstream consumer built against %s without OpenUSD and ran a commit\n",
                prefix);
    return 0;
}
