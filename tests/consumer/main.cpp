// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen consumer probe: against the installed api/ headers and usdGen only,
// create the engine (loads the installed extensions), compile an empty spec
// and run a commit.
#include "usdGen/api/engine.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/session.h"
#include "usdGen/api/usdGen.h"

#include <cstdio>
#include <cstring>
#include <string>

int main() {
    using namespace usdGen;
    char const *version = GetVersionString();
    std::printf("%s\n", version ? version : "(null)");
    if (!version || !std::strstr(version, "usdGen")) {
        std::printf("FAIL: version string\n");
        return 1;
    }

    UsdGenStatus const created = UsdGenEngine::Create();
    if (!created.ok()) {
        std::printf("FAIL: UsdGenEngine::Create: %.*s\n",
                    static_cast<int>(created.message().size()),
                    created.message().data());
        return 1;
    }
    UsdGenEngine::Get();

    int rc = 0;
    {
        UsdGenSession session;
        session.SetGraphSpec(UsdGenGraphSpec{});
        UsdGenSession::SnapshotPtr const snapshot =
            session.CommitSnapshot(0.0, UsdGenCommitReason::NoticeBatchEnd);
        if (!snapshot) {
            std::printf("FAIL: commit published no snapshot\n");
            rc = 1;
        } else {
            for (std::string const &e : snapshot->diagnostics.errors)
                std::printf("  error: %s\n", e.c_str());
            std::printf("commit: generation %lld, %d nodes, %zu errors\n",
                        snapshot->generation
                            ? static_cast<long long>(snapshot->generation->id) : -1LL,
                        snapshot->graphInfo.NodeCount(),
                        snapshot->diagnostics.errors.size());
            // An empty spec has no operator: the compiler must say so.
            if (!snapshot->generation && snapshot->diagnostics.errors.empty()) {
                std::printf("FAIL: commit neither published nor diagnosed\n");
                rc = 1;
            }
        }
        session.Drain();
    }
    if (rc == 0) std::printf("consumer: PASS\n");
    return rc;
}
