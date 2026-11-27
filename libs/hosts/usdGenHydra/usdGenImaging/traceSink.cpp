// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Engine trace scopes (usdGen/api/trace.h) become TraceCollector events in
// every process that loads usdGenImaging: installed at load, removed at
// unload. Off while the collector is disabled (one atomic load per scope).
#include "usdGen/api/trace.h"

#include "pxr/pxr.h"
#include "pxr/base/trace/collector.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {
namespace {

void Begin(void *, char const *name) noexcept
{
    if (!TraceCollector::IsEnabled()) return;
    try {
        TraceCollector::GetInstance().BeginEvent(TraceDynamicKey(name));
    } catch (...) {
    }
}

void End(void *, char const *name) noexcept
{
    if (!TraceCollector::IsEnabled()) return;
    try {
        TraceCollector::GetInstance().EndEvent(TraceDynamicKey(name));
    } catch (...) {
    }
}

void Counter(void *, char const *name, double delta) noexcept
{
    if (!TraceCollector::IsEnabled()) return;
    try {
        TraceCollector::GetInstance().RecordCounterDelta(TraceDynamicKey(name), delta);
    } catch (...) {
    }
}

constexpr usdGen::UsdGenTraceSink kTraceSink{&Begin, &End, &Counter, nullptr};

struct TraceSinkInstall {
    TraceSinkInstall() noexcept { usdGen::UsdGenSetTraceSink(&kTraceSink); }
    ~TraceSinkInstall()
    {
        if (usdGen::UsdGenGetTraceSink() == &kTraceSink) usdGen::UsdGenSetTraceSink(nullptr);
    }
};
TraceSinkInstall const gTraceSinkInstall;

}  // namespace
}  // namespace usdGenImaging
