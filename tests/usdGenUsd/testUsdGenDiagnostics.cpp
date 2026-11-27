// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenDiagnostics (step 11.8): the engine's env, debug, trace and
// report doors. Env parsing matches TfGetenvBool/TfGetenvInt; TF_DEBUG (set
// by ctest) is read with TfDebug's syntax and SyncDebugCodes agrees; trace
// scopes and counters reach an installed sink; reports reach Tf through the
// sink usdGenUsd installs, and a custom sink sees severity and full text.
// Tf's int read throws on text that is no number; the engine's 11.7 caller
// caught that as the fallback, which the engine's own parser keeps.

#include "usdGen/api/debug.h"
#include "usdGen/api/env.h"
#include "usdGen/api/status.h"
#include "usdGen/api/trace.h"
#include "usdGenUsd/debugCodes.h"

#include "pxr/base/tf/errorMark.h"
#include "pxr/base/tf/getenv.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, char const *what)
{
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_failures;
}

void SetEnv(char const *name, char const *value)
{
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    if (*value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

// The engine's 11.7 calls: Tf, with a conversion failure as the fallback.
bool TfBool(char const *name, bool fallback)
{
    try {
        return PXR_NS::TfGetenvBool(name, fallback);
    } catch (...) {
        return fallback;
    }
}
int TfInt(char const *name, int fallback)
{
    try {
        return PXR_NS::TfGetenvInt(name, fallback);
    } catch (...) {
        return fallback;
    }
}

std::vector<std::string> g_trace;
void TraceBegin(void *, char const *name) noexcept { g_trace.push_back(std::string("+") + name); }
void TraceEnd(void *, char const *name) noexcept { g_trace.push_back(std::string("-") + name); }
void TraceCounter(void *, char const *name, double delta) noexcept
{
    g_trace.push_back(std::string("#") + name + "=" + std::to_string(int(delta)));
}

void TracedFunction() { USDGEN_TRACE_FUNCTION(); }

UsdGenSeverity g_severity = UsdGenSeverity::Status;
std::string g_message;
void Capture(void *, UsdGenSeverity s, char const *message) noexcept
{
    g_severity = s;
    g_message = message;
}

}  // namespace

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // TF_DEBUG="USDGEN_C* USDGEN_SCH* -USDGEN_SCHEDULE USDGEN_FUR" (ctest).
    Check(UsdGenDebugEnabled(UsdGenDebugCode::Commit), "TF_DEBUG: a '*' prefix turns Commit on");
    Check(!UsdGenDebugEnabled(UsdGenDebugCode::Schedule), "TF_DEBUG: a later '-' turns Schedule off");
    Check(!UsdGenDebugEnabled(UsdGenDebugCode::Ingress), "TF_DEBUG: Ingress not named stays off");
    Check(UsdGenDebugEnabled(UsdGenDebugCode::Fur), "TF_DEBUG: an exact name turns Fur on");
    usdGenUsd::SyncDebugCodes();
    Check(UsdGenDebugEnabled(UsdGenDebugCode::Commit) && !UsdGenDebugEnabled(UsdGenDebugCode::Schedule) &&
              !UsdGenDebugEnabled(UsdGenDebugCode::Ingress) && UsdGenDebugEnabled(UsdGenDebugCode::Fur),
          "SyncDebugCodes: TfDebug read TF_DEBUG the same way");
    UsdGenSetDebugEnabled(UsdGenDebugCode::Ingress, true);
    UsdGenSetDebugEnabled(UsdGenDebugCode::Fur, false);
    Check(UsdGenDebugEnabled(UsdGenDebugCode::Ingress) && !UsdGenDebugEnabled(UsdGenDebugCode::Fur) &&
              UsdGenDebugEnabled(UsdGenDebugCode::Commit),
          "UsdGenSetDebugEnabled flips one code");

    char const *const kName = "USDGEN_TEST_DIAGNOSTICS_ENV";
    char const *const kValues[] = {"", "1", "0", "true", "TRUE", "Yes", "on", "ON", "off", "no", "2", "truex",
                                   " 1", "42", "-7", "abc", "12abc", "yes please", "99999999999", "-", "+3"};
    bool envOk = true;
    for (char const *v : kValues) {
        SetEnv(kName, v);
        for (bool fb : {false, true})
            if (UsdGenGetEnvBool(kName, fb) != TfBool(kName, fb)) {
                std::printf("  bool mismatch for '%s' (fallback %d)\n", v, int(fb));
                envOk = false;
            }
        if (UsdGenGetEnvInt(kName, 5) != TfInt(kName, 5)) {
            std::printf("  int mismatch for '%s'\n", v);
            envOk = false;
        }
        char const *const raw = UsdGenGetEnv(kName);
        if ((*v != 0) != (raw != nullptr && *raw != 0)) envOk = false;
    }
    SetEnv(kName, "");
    Check(envOk, "UsdGenGetEnvBool/Int match TfGetenvBool/Int (failure: fallback)");
    Check(UsdGenGetEnv(kName) == nullptr || *UsdGenGetEnv(kName) == 0, "unset variable reads as null");

    Check(UsdGenGetTraceSink() == nullptr, "no trace sink without a Hydra host");
    {
        USDGEN_TRACE_SCOPE("not recorded");
    }
    UsdGenTraceSink const sink{&TraceBegin, &TraceEnd, &TraceCounter, nullptr};
    UsdGenSetTraceSink(&sink);
    {
        USDGEN_TRACE_SCOPE("outer");
        USDGEN_TRACE_COUNTER("count", 3);
        TracedFunction();
    }
    UsdGenSetTraceSink(nullptr);
    {
        USDGEN_TRACE_SCOPE("after");
    }
    std::vector<std::string> const expected = {"+outer", "#count=3", "+TracedFunction", "-TracedFunction",
                                               "-outer"};
    Check(g_trace == expected, "trace scopes, counters and __func__ names reach the sink in order");

    {
        PXR_NS::TfErrorMark mark;
        UsdGenReport(UsdGenSeverity::CodingError, "diagnostics test coding error %d", 7);
        Check(!mark.IsClean(), "usdGenUsd's sink turns a coding error into a Tf error");
        mark.Clear();
    }
    UsdGenSetReportSink(&Capture, nullptr);
    std::string const big(3000, 'x');
    UsdGenReport(UsdGenSeverity::Warning, "w %s %d", big.c_str(), 9);
    Check(g_severity == UsdGenSeverity::Warning && g_message == "w " + big + " 9",
          "a custom sink sees the severity and the whole formatted message");
    UsdGenSetReportSink(nullptr, nullptr);

    std::printf("%s\n", g_failures ? "FAILED" : "PASSED");
    return g_failures ? 1 : 0;
}
