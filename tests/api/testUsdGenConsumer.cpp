// testUsdGenConsumer — M0 downstream consumer check (plan §6.2, sol S-5/X-4).
//
// Proves the installed CMake package is actually usable: configure + build a
// tiny consumer CMake project (tests/consumer/) that does
//     find_package(pxr REQUIRED CONFIG)    // first — pxrConfig is single-pass
//     find_package(usdGen REQUIRED)        // usdGenConfig's if(NOT TARGET usd)
//                                          // guard skips the nested pxr re-find
// and links usdGen::usdGen, compiling a main that calls the canonical M0 API
// usdGen::GetVersionString(). This exercises the PATH_VARS-fixed
// usdGenConfig.cmake end to end (sol S-5/X-4).
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
    const std::string usdDir = USDGEN_TEST_USD_INSTALL_DIR;

    // A private scratch build dir so repeated runs never collide.
    const std::string tmpl = "/tmp/usdgen-consumer-XXXXXX";
    char *tmp = mkdtemp(const_cast<char *>(tmpl.data()));
    if (!tmp) {
        std::printf("FAIL: mkdtemp failed\n");
        return 1;
    }
    const std::string buildDir = tmp;

    // CMAKE_PREFIX_PATH must contain BOTH the usdGen install prefix (for
    // find_package(usdGen)) and the OpenUSD install prefix (for the pxr
    // dependency that usdGenConfig pulls in / the consumer re-requests).
    const std::string prefixPath =
        std::string("\"") + prefix + ";" + usdDir + "\"";

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

    // Run the consumer and check it reports the canonical core version.
    std::string out = buildDir + "/consumer > '" + buildDir + "/out.txt' 2>&1";
    if (Run(out) != 0) {
        std::printf("FAIL: consumer run failed\n");
        return 1;
    }
    out = "grep -q 'usdGen' '" + buildDir + "/out.txt'";
    if (Run(out) != 0) {
        std::printf("FAIL: consumer output missing the usdGen version string\n");
        return 1;
    }

    std::printf("PASS: downstream consumer built against %s and called usdGen::GetVersionString()\n",
                prefix);
    return 0;
}
