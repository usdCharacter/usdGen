// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The engine: one per process. Create loads the extensions, runs their
// usdGenRegister and builds the process tables; registries freeze at the
// first compile.
#pragma once
#include "usdGen/api/export.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/status.h"
#include "usdGen/api/token.h"

namespace usdGen {

struct UsdGenEngineDesc {
    /// Extension manifest; null: USDGEN_EXTENSIONS (path list, ';' on
    /// Windows, ':' elsewhere), else <usdGen's directory>/usdGenExtensions.txt.
    /// One file name per line in load order, relative to the manifest's
    /// directory; blank lines and '#' lines are skipped. A missing default
    /// manifest loads nothing; a missing named one is an error.
    char const *manifest = nullptr;
    UsdGenReportFn report = nullptr;   // optional: installed before loading
    void *reportUser = nullptr;
};

class USDGEN_API UsdGenEngine {
public:
    /// Loads extensions in manifest order, runs each usdGenRegister, and
    /// creates the cache-domain table (the Ptex cache comes with
    /// usdGenSurface's usdGenRegister). Idempotent; publication by CAS. A
    /// later call returns the first call's status and ignores its manifest.
    static UsdGenStatus Create(UsdGenEngineDesc const &desc = {});
    /// Acquire load. Calls Create({}) first if nobody has.
    static UsdGenEngine &Get();
    UsdGenOpId FindOp(UsdGenToken name) const noexcept;        // compile phase
    UsdGenTypeId FindType(UsdGenToken name) const noexcept;

    UsdGenEngine(UsdGenEngine const &) = delete;
    UsdGenEngine &operator=(UsdGenEngine const &) = delete;

private:
    UsdGenEngine() = default;
    UsdGenStatus _status;
};

}  // namespace usdGen
