// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Engine reports (UsdGenReport) become Tf diagnostics in every process that
// loads usdGenUsd: installed at load, removed at unload. The install is one
// atomic store into usdGen (loaded first, a link dependency), no registry.
#include "usdGen/api/status.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/diagnostic.h"

#include <cstdio>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenUsd {
namespace {

void ReportToTf(void *, usdGen::UsdGenSeverity s, char const *message) noexcept
{
    try {
        switch (s) {
        case usdGen::UsdGenSeverity::Status: TF_STATUS("%s", message); break;
        case usdGen::UsdGenSeverity::Warning: TF_WARN("%s", message); break;
        case usdGen::UsdGenSeverity::Error: TF_RUNTIME_ERROR("%s", message); break;
        case usdGen::UsdGenSeverity::CodingError: TF_CODING_ERROR("%s", message); break;
        }
    } catch (...) {
        std::fputs(message, stderr);
        std::fputc('\n', stderr);
    }
}

struct ReportSinkInstall {
    ReportSinkInstall() noexcept { usdGen::UsdGenSetReportSink(&ReportToTf, nullptr); }
    ~ReportSinkInstall() { usdGen::UsdGenSetReportSink(nullptr, nullptr); }
};
ReportSinkInstall const gReportSinkInstall;

}  // namespace
}  // namespace usdGenUsd
