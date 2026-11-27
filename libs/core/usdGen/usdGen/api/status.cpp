// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/status.h"

#include "pxr/base/tf/diagnostic.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace {
std::atomic<UsdGenReportFn> gReportFn{nullptr};
std::atomic<void *> gReportUser{nullptr};

void Forward(UsdGenSeverity s, char const *message) noexcept
{
    if (UsdGenReportFn const fn = gReportFn.load(std::memory_order_acquire)) {
        fn(gReportUser.load(std::memory_order_relaxed), s, message);
        return;
    }
    try {
        switch (s) {
        case UsdGenSeverity::Status: TF_STATUS("%s", message); break;
        case UsdGenSeverity::Warning: TF_WARN("%s", message); break;
        case UsdGenSeverity::Error: TF_RUNTIME_ERROR("%s", message); break;
        case UsdGenSeverity::CodingError: TF_CODING_ERROR("%s", message); break;
        }
    } catch (...) {
        std::fputs(message, stderr);
        std::fputc('\n', stderr);
    }
}
}  // namespace

void UsdGenSetReportSink(UsdGenReportFn fn, void *user) noexcept
{
    gReportUser.store(user, std::memory_order_relaxed);
    gReportFn.store(fn, std::memory_order_release);
}

void UsdGenReport(UsdGenSeverity s, char const *fmt, ...) noexcept
{
    char small[1024];
    va_list ap;
    va_start(ap, fmt);
    va_list again;
    va_copy(again, ap);
    int const n = std::vsnprintf(small, sizeof(small), fmt, ap);
    va_end(ap);
    if (n < 0) {
        va_end(again);
        Forward(s, fmt);
        return;
    }
    if (size_t(n) < sizeof(small)) {
        va_end(again);
        Forward(s, small);
        return;
    }
    try {
        std::string big(size_t(n) + 1, '\0');
        std::vsnprintf(&big[0], big.size(), fmt, again);
        va_end(again);
        Forward(s, big.c_str());
    } catch (...) {
        va_end(again);
        Forward(s, small);   // truncated
    }
}

}  // namespace usdGen
