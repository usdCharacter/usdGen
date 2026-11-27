// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenEngineCreate (12.0): links usdGen only.
//   get:      Get() without Create builds the engine from usdGenExtensions.txt
//             beside usdGen; Create after it returns the same status.
//   manifest: Create twice with a named manifest (comment, blank and CRLF
//             lines, backends in the reverse of the default order); the
//             backends' handles follow the manifest's order; the report sink
//             is installed.
//   missing:  a named manifest that does not exist fails with its path and
//             leaves a bare engine.

#include "utils/testToken.h"
#include "usdGen/api/engine.h"
#include "usdGen/api/status.h"
#include "usdGen/api/ext/backendDef.h"
#include "usdGen/api/ext/opRegistry.h"
#include "usdGen/api/ext/typeRegistry.h"

#include <cstdio>
#include <cstring>
#include <string>

using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (ok) return;
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
}

int g_reports = 0;
void CountReport(void *user, UsdGenSeverity, char const *) noexcept
{
    ++*static_cast<int *>(user);
}

void CheckPrimitives(std::string const &mode)
{
    UsdGenEngine const &engine = UsdGenEngine::Get();
    Check(engine.FindOp(TestToken("UsdGenScatter")) == UsdGenOpId(0), mode + ": Scatter is op 0");
    Check(engine.FindOp(TestToken("UsdGenNoise")) != UsdGenOpId::Invalid, mode + ": Noise registered");
    Check(engine.FindOp(TestToken("noSuchOp")) == UsdGenOpId::Invalid, mode + ": unknown op");
    Check(engine.FindType(TestToken("UsdGenCurveSet")) == UsdGenTypeId(0), mode + ": CurveSet is type 0");
    Check(engine.FindType(TestToken("noSuchType")) == UsdGenTypeId::Invalid, mode + ": unknown type");
    Check(UsdGenOpRegistry::Get().ParamEvaluator().create != nullptr, mode + ": usdGenExpr loaded");
    Check(!UsdGenOpRegistry::Get().IsFrozen(), mode + ": Create does not freeze");
}

uint32_t BackendIndex(char const *name)
{
    UsdGenBackendId const id = UsdGenFindBackend(name);
    return id == UsdGenBackendId::Invalid ? 0u : uint32_t(id);
}

void RunGet()
{
    UsdGenEngine &first = UsdGenEngine::Get();
    Check(&first == &UsdGenEngine::Get(), "get: one engine");
    UsdGenStatus const status = UsdGenEngine::Create();
    Check(status.ok(), "get: default manifest loads: " + std::string(status.message()));
    CheckPrimitives("get");
    uint32_t const cuda = BackendIndex("cuda");
    Check(cuda != 0, "get: usdGenCuda registers its backend");
#if defined(USDGEN_TEST_VULKAN_BACKEND)
    uint32_t const vulkan = BackendIndex("vulkan");
    Check(vulkan != 0 && cuda < vulkan, "get: default order loads usdGenCuda before usdGenVulkan");
#endif
}

void RunManifest()
{
    UsdGenEngineDesc desc;
    desc.manifest = USDGEN_TEST_MANIFEST;
    desc.report = &CountReport;
    desc.reportUser = &g_reports;
    UsdGenStatus const status = UsdGenEngine::Create(desc);
    Check(status.ok(), "manifest: loads: " + std::string(status.message()));
    UsdGenEngineDesc missing;
    missing.manifest = USDGEN_TEST_MANIFEST ".missing";
    UsdGenStatus const again = UsdGenEngine::Create(missing);
    Check(again.ok(), "manifest: a second Create returns the first status");
    CheckPrimitives("manifest");
    uint32_t const cuda = BackendIndex("cuda");
#if defined(USDGEN_TEST_VULKAN_BACKEND)
    uint32_t const vulkan = BackendIndex("vulkan");
    Check(vulkan == 1 && cuda == 2, "manifest: backends follow the manifest's order (vulkan " +
                                        std::to_string(vulkan) + ", cuda " + std::to_string(cuda) + ")");
#else
    Check(cuda == 1, "manifest: usdGenCuda is the first backend");
#endif
    int const before = g_reports;
    UsdGenReport(UsdGenSeverity::Status, "testUsdGenEngineCreate: %s", "sink");
    Check(g_reports == before + 1, "manifest: the report sink is installed");
    UsdGenSetReportSink(nullptr, nullptr);
}

void RunMissing()
{
    UsdGenEngineDesc desc;
    desc.manifest = USDGEN_TEST_MANIFEST ".missing";
    UsdGenStatus const status = UsdGenEngine::Create(desc);
    Check(!status.ok() && status.code() == UsdGenStatusCode::Failed, "missing: Create fails");
    Check(status.message().find(".missing") != std::string::npos,
          "missing: the message names the manifest: " + std::string(status.message()));
    Check(!UsdGenEngine::Create().ok(), "missing: a second Create returns the failure");
    Check(UsdGenOpRegistry::Get().Count() == 0, "missing: no ops");
    Check(UsdGenTypeRegistry::Get().Count() == 0, "missing: no types");
    Check(UsdGenEngine::Get().FindOp(TestToken("UsdGenScatter")) == UsdGenOpId::Invalid,
          "missing: no Scatter");
}

}  // namespace

int main(int argc, char **argv)
{
    std::string const mode = argc > 1 ? argv[1] : "get";
    if (mode == "get") RunGet();
    else if (mode == "manifest") RunManifest();
    else if (mode == "missing") RunMissing();
    else {
        std::printf("usage: testUsdGenEngineCreate [get|manifest|missing]\n");
        return 2;
    }
    if (g_failures) {
        std::printf("testUsdGenEngineCreate %s: %d failure(s)\n", mode.c_str(), g_failures);
        return 1;
    }
    std::printf("testUsdGenEngineCreate %s: OK\n", mode.c_str());
    return 0;
}
