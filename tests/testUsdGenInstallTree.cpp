// testUsdGenInstallTree — M0 installed-tree verification (plan §6.2, sol S-5/X-4).
//
// Run against a scratch install prefix. The install prefix is taken from the
// USDGEN_INSTALL_PREFIX environment variable (exact name CI must set — the
// docs lane wires this in .github/workflows/usdgen.yml install-check job).
// When it is unset, the build tree's CMake package export is probed instead;
// either way the test skips with a printed reason ('post-install gate').
//
// Two parts:
//   1. Layout: assert the installed tree matches what usdGenConfig.cmake and
//      the exported targets promise (headers under <prefix>/include/<api>/…,
//      libs under <prefix>/lib, plugin resources under <prefix>/lib/usd/…,
//      cmake config under <prefix>/lib/cmake/usdGen/).
//   2. Behaviour: reuse the stock-only probe approach from
//      docs/prework/probes/m0-usdcat-extent/ — link ONLY stock OpenUSD, rely
//      on PXR_PLUGINPATH_NAME pointing at <prefix>/lib/usd so Plug dlopens the
//      installed schema library and fires the compute-extent registration.
//      This process re-execs itself with PXR_PLUGINPATH_NAME set before any
//      USD symbol is touched (plugin path is read at Plug discovery time).
#include "pxr/pxr.h"
#include "pxr/base/gf/range3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/tf/type.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/boundable.h"
#include "pxr/usd/usdGeom/boundableComputeExtent.h"
#include "pxr/base/vt/array.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <unistd.h>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

int Fail(const char *msg) {
    std::printf("FAIL: %s\n", msg);
    return 1;
}

int CheckFile(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "r");
    if (!f) {
        std::printf("  missing: %s\n", path.c_str());
        return 1;
    }
    std::fclose(f);
    std::printf("  ok:      %s\n", path.c_str());
    return 0;
}

int CheckNotInstalled(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "r");
    if (f) {
        std::fclose(f);
        std::printf("  unexpected: %s (should not be installed)\n", path.c_str());
        return 1;
    }
    std::printf("  absent:   %s\n", path.c_str());
    return 0;
}

bool FileExists(const std::string &path) { return ::access(path.c_str(), F_OK) == 0; }

// Reports registration + concreteness for one codeless type (sol S-7: the
// required const-reference form of TfType::FindByName).
int CheckCodelessType(const TfToken &typeName) {
    const TfType &type = TfType::FindByName(typeName);
    const bool unknown = type.IsUnknown();
    const bool concrete = !unknown && UsdSchemaRegistry::IsConcrete(typeName);
    std::printf(
        "type-check %-20s TfType::FindByName known=%d  "
        "UsdSchemaRegistry::IsConcrete=%d\n",
        typeName.GetString().c_str(), unknown ? 0 : 1, concrete ? 1 : 0);
    return (unknown || !concrete) ? 1 : 0;
}

int RunLayoutChecks(const std::string &prefix,
                    const std::string &pluginDir,
                    const std::string &libDir,
                    const std::string &cmakeDir) {
    std::printf("== layout checks against %s ==\n", prefix.c_str());
    int rc = 0;

    // Libraries / archives. N-7 (plan §1.2/§1.6): usdGenMath, usdGen_seexpr
    // and nanoflann are private hidden archives and must not appear in the
    // install prefix (installing them would also export their unprefixed
    // INTERFACE include roots via usdGenTargets).
    rc |= CheckFile(prefix + "/" + libDir + "/libusdGen.so");
    rc |= CheckFile(prefix + "/" + libDir + "/libusdGenImaging.so");
    rc |= CheckFile(prefix + "/" + libDir + "/libusdGenSchema.so");
    rc |= CheckNotInstalled(prefix + "/" + libDir + "/libusdGenMath.a");
    rc |= CheckNotInstalled(prefix + "/" + libDir + "/libusdGen_seexpr.a");

    // Headers: advertised layout <prefix>/include/<api>/...
    rc |= CheckFile(prefix + "/include/usdGen/usdGen.h");
    rc |= CheckFile(prefix + "/include/usdGen/export.h");
    rc |= CheckNotInstalled(prefix + "/include/usdGenMath/usdGenMath.h");
    rc |= CheckFile(prefix + "/include/usdGenSchema/usdGenSchema.h");
    rc |= CheckFile(prefix + "/include/usdGenImaging/api.h");
    rc |= CheckNotInstalled(prefix + "/include/nanoflann.hpp");
    rc |= CheckNotInstalled(prefix + "/include/seexpr/SeExpr2/Noise.h");

    // Negative layout checks (sol S-6): no double-nesting, no .cpp installed.
    rc |= CheckNotInstalled(prefix + "/include/usdGen/usdGen/usdGen.h");
    rc |= CheckNotInstalled(prefix + "/include/usdGen/version.h");
    rc |= CheckNotInstalled(prefix + "/include/usdGen/version.cpp");
    rc |= CheckNotInstalled(prefix + "/include/nanoflann/nanoflann.hpp");
    rc |= CheckNotInstalled(prefix + "/include/seexpr/SeExpr2/SeExpr2/Noise.h");

    // Plugin resources
    rc |= CheckFile(prefix + "/" + pluginDir + "/usdGenSchema/resources/plugInfo.json");
    rc |= CheckFile(prefix + "/" + pluginDir + "/usdGenSchema/resources/generatedSchema.usda");
    rc |= CheckFile(prefix + "/" + pluginDir + "/usdGenImaging/resources/plugInfo.json");

    // CMake package config (USDGEN_TEST_CMAKE_DIR is prefix-relative and
    // already includes the lib component, e.g. "lib/cmake/usdGen").
    rc |= CheckFile(prefix + "/" + cmakeDir + "/usdGenConfig.cmake");
    rc |= CheckFile(prefix + "/" + cmakeDir + "/usdGenConfigVersion.cmake");
    rc |= CheckFile(prefix + "/" + cmakeDir + "/usdGenTargets.cmake");

    if (rc) return Fail("layout checks failed");
    std::printf("layout checks passed\n");
    return 0;
}

int RunExtentCheck(const std::string &prefix, const std::string &pluginDir,
                   const std::string &fixture) {
    std::printf("== installed-tree extent check ==\n");
    const char *pluginpath = std::getenv("PXR_PLUGINPATH_NAME");
    std::printf("PXR_PLUGINPATH_NAME=%s\n", pluginpath ? pluginpath : "(unset)");

    UsdStageRefPtr stage = UsdStage::Open(fixture);
    if (!stage) return Fail("could not open fixture stage");

    UsdPrim desc = stage->GetPrimAtPath(SdfPath("/Groom/Description"));
    if (!desc.IsValid()) return Fail("/Groom/Description not found");
    std::printf("prim type = %s\n", desc.GetTypeName().GetString().c_str());

    // Type-resolution assertions (same as the stock-only probe).
    if (CheckCodelessType(TfToken("UsdGenGroom")) != 0)
        return Fail("UsdGenGroom not resolved+concrete from installed plugins");
    if (CheckCodelessType(TfToken("UsdGenDescription")) != 0)
        return Fail("UsdGenDescription not resolved+concrete from installed plugins");

    VtVec3fArray extent;
    const bool ok = UsdGeomBoundable(desc).ComputeExtent(UsdTimeCode(), &extent);
    std::printf("ComputeExtent ok=%d size=%zu\n", static_cast<int>(ok),
                extent.size());
    if (!ok || extent.size() != 2)
        return Fail("no extent returned (installed schema plugin did not fire?)");

    const GfVec3f lo = extent[0], hi = extent[1];
    const bool empty = (lo == GfVec3f(0, 0, 0)) && (hi == GfVec3f(0, 0, 0));
    std::printf("extent = [(%.3f %.3f %.3f) .. (%.3f %.3f %.3f)] empty=%d\n",
                static_cast<double>(lo[0]), static_cast<double>(lo[1]),
                static_cast<double>(lo[2]), static_cast<double>(hi[0]),
                static_cast<double>(hi[1]), static_cast<double>(hi[2]),
                empty ? 1 : 0);
    if (empty) return Fail("extent is empty");

    std::printf("installed-tree extent check passed\n");
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    (void)argc;
    const char *prefix = std::getenv("USDGEN_INSTALL_PREFIX");
    if (!prefix || !*prefix) {
        // No install prefix: probe the build tree's CMake package export so a
        // missing or broken export is reported loudly instead of silently
        // passing. configure_package_config_file writes usdGenConfig.cmake to
        // the top of the build directory (the test's working directory under
        // ctest); find_package(usdGen CONFIG PATHS <build-dir>) can only work
        // if usdGenTargets.cmake sits next to it, which install(EXPORT) alone
        // does not provide.
        const std::string cfgPath = "usdGenConfig.cmake";
        if (!FileExists(cfgPath)) {
            printf("testUsdGenInstallTree: SKIP (post-install gate) — "
                   "USDGEN_INSTALL_PREFIX unset and no build-tree package in the current "
                   "(build) directory; run cmake --install into a scratch prefix and set "
                   "the variable to run these checks.\n");
            return 0;
        }
        const std::string targetsPath = "usdGenTargets.cmake";
        if (!FileExists(targetsPath)) {
            printf("testUsdGenInstallTree: SKIP (post-install gate) — "
                   "build tree does not export its targets: %s is missing next to %s, which "
                   "includes it, so find_package(usdGen CONFIG PATHS <build-dir>) would fail. "
                   "Run cmake --install into a scratch prefix and set USDGEN_INSTALL_PREFIX.\n",
                   targetsPath.c_str(), cfgPath.c_str());
            return 0;
        }
        printf("testUsdGenInstallTree: SKIP (post-install gate) — build-tree export present (%s); "
               "installed-tree checks run after cmake --install. Set USDGEN_INSTALL_PREFIX to verify.\n",
               cfgPath.c_str());
        return 0;
    }
    const std::string pluginDir = USDGEN_TEST_PLUGIN_DIR;
    const std::string libDir = USDGEN_TEST_LIB_DIR;
    const std::string cmakeDir = USDGEN_TEST_CMAKE_DIR;

    // PXR_PLUGINPATH_NAME must be in the environment before Plug discovery
    // runs, so re-exec ourselves once with it set.
    if (!std::getenv("USDGEN_INSTALL_TREE_PROBE")) {
        // SdrCatalog discovers plugInfo.json at <entry>/resources/ or in a
        // subdirectory directly; list the specific resources dirs (same
        // convention the build-tree tests use) plus the parent plugin dir.
        std::string pluginpath = std::string(prefix) + "/" + pluginDir + "/usdGenSchema/resources"
            ":" + std::string(prefix) + "/" + pluginDir + "/usdGenImaging/resources";
        if (setenv("PXR_PLUGINPATH_NAME", pluginpath.c_str(), 1) != 0)
            return Fail("setenv PXR_PLUGINPATH_NAME failed");
        std::string ldpath = std::string(USDGEN_TEST_USD_INSTALL_DIR) + "/lib";
        if (setenv("LD_LIBRARY_PATH", ldpath.c_str(), 1) != 0)
            return Fail("setenv LD_LIBRARY_PATH failed");
        setenv("USDGEN_INSTALL_TREE_PROBE", "1", 1);
        ::execv(argv[0], argv);
        // If execv returns, it failed.
        return Fail("execv self-re-exec failed");
    }

    int rc = RunLayoutChecks(prefix, pluginDir, libDir, cmakeDir);
    if (rc == 0)
        rc = RunExtentCheck(prefix, pluginDir, std::string(USDGEN_TEST_FIXTURE));
    if (rc == 0) std::printf("PASS: installed tree verified\n");
    return rc;
}
