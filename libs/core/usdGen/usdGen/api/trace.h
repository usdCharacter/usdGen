// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Trace scopes and counters through a host-installed sink (usdGenImaging
// forwards to Tf's TraceCollector). No sink: a scope costs one acquire load.
// Scope names stay literals, so bin/trace_playback.ps1 filters still match;
// USDGEN_TRACE_FUNCTION names the scope __func__.
#pragma once
#include "usdGen/api/export.h"

namespace usdGen {

/// Every name is a string with static storage duration.
struct UsdGenTraceSink {
    void (*begin)(void *user, char const *name) noexcept;
    void (*end)(void *user, char const *name) noexcept;
    void (*counter)(void *user, char const *name, double delta) noexcept;
    void *user;
};
/// Release-store; null turns tracing off. The sink outlives every scope that
/// loaded it.
USDGEN_API void UsdGenSetTraceSink(UsdGenTraceSink const *sink) noexcept;
/// Acquire load.
USDGEN_API UsdGenTraceSink const *UsdGenGetTraceSink() noexcept;

class UsdGenTraceScope {
public:
    explicit UsdGenTraceScope(char const *name) noexcept : _sink(UsdGenGetTraceSink()), _name(name)
    {
        if (_sink) _sink->begin(_sink->user, name);
    }
    ~UsdGenTraceScope()
    {
        if (_sink) _sink->end(_sink->user, _name);
    }
    UsdGenTraceScope(UsdGenTraceScope const &) = delete;
    UsdGenTraceScope &operator=(UsdGenTraceScope const &) = delete;

private:
    UsdGenTraceSink const *_sink;
    char const *_name;
};

}  // namespace usdGen

#define USDGEN_TRACE_CAT2(a, b) a##b
#define USDGEN_TRACE_CAT(a, b) USDGEN_TRACE_CAT2(a, b)
#define USDGEN_TRACE_SCOPE(name) ::usdGen::UsdGenTraceScope USDGEN_TRACE_CAT(usdGenTrace_, __LINE__)(name)
#define USDGEN_TRACE_FUNCTION() USDGEN_TRACE_SCOPE(__func__)
#define USDGEN_TRACE_COUNTER(name, delta) \
    do { \
        if (::usdGen::UsdGenTraceSink const *usdGenTraceSink_ = ::usdGen::UsdGenGetTraceSink()) \
            usdGenTraceSink_->counter(usdGenTraceSink_->user, name, double(delta)); \
    } while (0)
