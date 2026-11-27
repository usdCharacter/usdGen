// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenEngineBare (step 6.5): links usdGen only and loads no extension
// (USDGEN_EXTENSIONS holds only a separator). The engine builds and runs
// without any primitive: an op that names no primitive registers, a spec of
// it compiles, a Session commit publishes a generation with no error and no
// publication, a spec naming an op no library registered fails with a
// diagnostic naming its type, and the first compile freezes the registries.
// An empty GraphSpec keeps the compiler's "no operator nodes" contract.

#include "usdGen/api/graphSpec.h"
#include "usdGen/compile/compiler.h"
#include "usdGen/compile/graph.h"
#include "usdGen/exec/session.h"
#include "usdGen/registries/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/token.h"
#include "usdGen/registries/typeRegistry.h"

#include "pxr/base/tf/token.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (ok) return;
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
}

std::string Joined(std::vector<std::string> const &messages)
{
    std::string all;
    for (std::string const &m : messages) all += (all.empty() ? "" : " | ") + m;
    return all;
}

void RegisterNothing(UsdGenRegistrar &) {}

/// An op that names no primitive: no input, no channel, nothing to evaluate.
class BareOp final : public UsdGenOp
{
public:
    explicit BareOp(UsdGenOpRecord const &record) noexcept : UsdGenOp(record) {}
    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override { return {1, 2}; }
    bool Capture(UsdGenCaptureContext const &, UsdGenOpInputs const &, UsdGenCapture *,
                 UsdGenDiagnostics *) override { return true; }
    void Evaluate(UsdGenEvalContext const &, UsdGenCapture const &,
                  UsdGenChunkView const &) const override {}
};

std::unique_ptr<UsdGenOp> CreateBareOp(UsdGenOpRecord const &record, void *)
{
    return std::make_unique<BareOp>(record);
}

void RegisterBareOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "BareTestOp";
    d.displayName = "BareTestOp";
    d.create = &CreateBareOp;
    d.geometryInputArity = 0;
    r.RegisterOp(d);
}

UsdGenGraphSpec OneNodeSpec(char const *type)
{
    UsdGenGraphSpec spec;
    UsdGenOpNode node;
    node.key = UsdGenNodeKeyOfText("/Groom/Op");
    UsdGenCompileTokenScope scope;   // test data, not a cook intern
    node.type = UsdGenInternToken(type);
    spec.nodes.push_back(node);
    spec.terminal = node.key;
    spec.description = UsdGenNodeKeyOfText("/Groom");
    return spec;
}

}  // namespace

int main()
{
#if defined(_WIN32)
    _putenv_s("USDGEN_EXTENSIONS", ";");   // an empty value would unset it
#else
    setenv("USDGEN_EXTENSIONS", ":", 1);
#endif
    std::string error = "stale";
    Check(UsdGenEngineEnsureInitialized(&error) && error.empty(), "bare engine initializes: " + error);
    Check(UsdGenOpRegistry::Get().Count() == 0, "no extension, no ops");
    Check(UsdGenTypeRegistry::Get().Count() == 0, "no extension, no types");
    Check(UsdGenOpRegistry::Get().ParamEvaluator().create == nullptr, "no extension, no evaluator");
    Check(!UsdGenOpRegistry::Get().IsFrozen(), "not frozen before the first compile");

    // An op that names no primitive registers before the first compile.
    Check(UsdGenRegisterStatic(&RegisterBareOp), "a primitive-free op registers");
    Check(UsdGenOpRegistry::Get().Count() == 1, "one op registered");

    // A spec of it compiles; the first compile freezes the registries.
    {
        UsdGenGraph graph;
        UsdGenCompileResult const result = UsdGenCompiler().Compile(OneNodeSpec("BareTestOp"), &graph);
        Check(result.ok && result.errors.empty(), "a primitive-free spec compiles: " + Joined(result.errors));
        Check(graph.NodeCount() == 1, "one compiled node");
    }
    Check(UsdGenOpRegistry::Get().IsFrozen(), "the first compile freezes the registries");
    Check(!UsdGenRegisterStatic(&RegisterNothing), "no registration after the freeze");

    // An empty GraphSpec keeps the compiler's contract: no operator, no graph.
    {
        UsdGenGraph graph;
        UsdGenCompileResult const result = UsdGenCompiler().Compile(UsdGenGraphSpec(), &graph);
        Check(!result.ok && Joined(result.errors).find("no operator nodes") != std::string::npos,
              "an empty spec reports no operator nodes: " + Joined(result.errors));
    }

    // A Session commit publishes a generation with no error and, with no
    // publisher registered, no publication.
    {
        UsdGenSession session;
        session.SetGraphSpec(OneNodeSpec("BareTestOp"));
        UsdGenSession::SnapshotPtr const snapshot =
            session.CommitSnapshot(0.0, UsdGenCommitReason::SetTime);
        Check(snapshot != nullptr, "commit returns a snapshot");
        if (snapshot) {
            Check(snapshot->diagnostics.errors.empty(),
                  "the commit has no error: " + Joined(snapshot->diagnostics.errors));
            Check(snapshot->generation != nullptr, "the commit publishes a generation");
            if (snapshot->generation)
                Check(!snapshot->generation->publication && !snapshot->generation->device,
                      "the generation carries no publication");
        }
        session.Drain();
    }

    // A spec naming an op no library registered fails, naming the type.
    {
        UsdGenGraphSpec const spec = OneNodeSpec("UsdGenNoise");
        UsdGenGraph graph;
        UsdGenCompileResult const result = UsdGenCompiler().Compile(spec, &graph);
        Check(!result.ok, "a spec naming UsdGenNoise does not compile");
        Check(Joined(result.errors).find("UsdGenNoise") != std::string::npos,
              "the diagnostic names the type: " + Joined(result.errors));
    }

    if (g_failures) {
        std::printf("testUsdGenEngineBare: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("testUsdGenEngineBare: OK\n");
    return 0;
}
