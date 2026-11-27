// N-5 (plan/09 §5.2, plan/10 §5.6): schema regeneration-drift guard.
// Runs bin/gen_schema.sh (the reference recipe) with the current
// libs/hosts/usdGenUsd/schema/resources stashed aside, then byte-compares the fresh
// products against the checked-in ones. On PASS the worktree is untouched
// (byte-identical); on DRIFT the fresh files stay in the tree for review,
// listing which product moved. This deliberately exercises the real script —
// mirroring its post-processing in here is how the N-2 plugInfo/AutoApply
// drift hid in the first place.

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

#ifndef USDGEN_TEST_SCHEMA_RESOURCES
#error "USDGEN_TEST_SCHEMA_RESOURCES must be defined by CMake"
#endif
#ifndef USDGEN_TEST_SCHEMA_USDA
#define USDGEN_TEST_SCHEMA_USDA "<libs/hosts/usdGenUsd/schema/schema.usda>"
#endif
#ifndef USDGEN_TEST_USD_GEN_SCHEMA_TOOL
#define USDGEN_TEST_USD_GEN_SCHEMA_TOOL ""
#endif

namespace {

// <root>/libs/hosts/usdGenUsd/schema/resources -> <root>
std::string RepoRoot()
{
    std::string r = USDGEN_TEST_SCHEMA_RESOURCES;
    for (int i = 0; i < 5; ++i) {
        const size_t slash = r.rfind('/');
        if (slash == std::string::npos) return "";
        r = r.substr(0, slash);
    }
    return r;
}

}  // namespace

int main()
{
    const std::string root = RepoRoot();
    if (root.empty()) {
        std::printf("FAIL: cannot derive repo root from %s\n",
                    USDGEN_TEST_SCHEMA_RESOURCES);
        return 1;
    }

    // CI builds OpenUSD with --no-python, so usdGenSchema (a Python tool) is
    // not in the prefix. SKIP_RETURN_CODE 77 keeps N-5 as a real drift gate
    // wherever the tool exists.
    const char *tool = USDGEN_TEST_USD_GEN_SCHEMA_TOOL;
    if (!tool || !tool[0] || access(tool, X_OK) != 0) {
        std::printf("SKIP: usdGenSchema not found at '%s' "
                    "(OpenUSD python tools unavailable)\n",
                    tool ? tool : "");
        return 77;
    }

    std::string probe;
    probe += "set -eu; ";
    probe += "res=" + std::string(USDGEN_TEST_SCHEMA_RESOURCES) + "; ";
    probe += "tmp=$(mktemp -d); ";
    probe += "cp $res/generatedSchema.usda $res/plugInfo.json $tmp/; ";
    probe += "if ! bash " + root + "/bin/gen_schema.sh >/tmp/usdGenSchemaUpToDate.gen.log 2>&1; "
             "then echo GEN_SCRIPT_FAILED; rm -rf $tmp; exit 2; fi; ";
    probe += "rc=0; ";
    probe += "if cmp -s $tmp/generatedSchema.usda $res/generatedSchema.usda; "
             "then echo GEN_SCHEMA_IDENTICAL; else echo GEN_SCHEMA_DRIFT; "
             "cp $tmp/generatedSchema.usda $res/; rc=1; fi; ";
    probe += "if cmp -s $tmp/plugInfo.json $res/plugInfo.json; "
             "then echo PLUG_INFO_IDENTICAL; else echo PLUG_INFO_DRIFT; "
             "cp $tmp/plugInfo.json $res/; rc=1; fi; ";
    probe += "rm -rf $tmp; exit $rc";

    std::array<char, 256> buf{};
    std::string seen;
    int rc = -1;
    if (FILE *pipe = popen(probe.c_str(), "r")) {
        while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
            seen += buf.data();
        }
        rc = pclose(pipe);
    } else {
        std::printf("FAIL: could not run regeneration probe\n");
        return 1;
    }

    const bool scriptOk = seen.find("GEN_SCRIPT_FAILED") == std::string::npos;
    const bool schemaOk = seen.find("GEN_SCHEMA_IDENTICAL") != std::string::npos;
    const bool plugOk = seen.find("PLUG_INFO_IDENTICAL") != std::string::npos;

    if (scriptOk && schemaOk && plugOk && rc == 0) {
        std::printf("PASS: libs/hosts/usdGenUsd/schema/resources is byte-identical to "
                    "a fresh bin/gen_schema.sh run (N-5)\n");
        return 0;
    }
    if (!scriptOk) {
        std::printf("SKIP: bin/gen_schema.sh could not run "
                    "(usdGenSchema/python toolchain unavailable)\n");
        if (!seen.empty()) std::printf("--- probe output ---\n%s", seen.c_str());
        return 77;
    }
    std::printf("FAIL: N-5 schema resources out of date with "
                "libs/hosts/usdGenUsd/schema/schema.usda — the fresh output has been "
                "left in place; review the diff, commit or revert\n");
    std::printf("  generatedSchema.usda: %s\n  plugInfo.json: %s\n  rc=%d\n",
                scriptOk ? (schemaOk ? "identical" : "DRIFT") : "untested",
                scriptOk ? (plugOk ? "identical" : "DRIFT") : "untested", rc);
    if (!seen.empty()) std::printf("--- probe output ---\n%s", seen.c_str());
    return 1;
}
