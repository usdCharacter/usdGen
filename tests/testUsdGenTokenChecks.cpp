// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenTokenChecks: the runtime half of the step-4 token rule. A token
// interned while a cook runs (here: from an op's Evaluate on the workers)
// fails UsdGenScheduler::Run with "usdGen: a token was interned during the
// cook"; with USDGEN_TOKEN_CHECKS=0 (the "off" run) it is a warning and the
// run succeeds. An intern inside UsdGenCompileTokenScope never fails a run.

#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/token.h"
#include "usdGen/scheduler.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

enum class Intern { None, Exec, CompileScope };
std::atomic<Intern> g_intern{Intern::None};

/// Passes points through; interns from Evaluate when asked.
class InternProbe final : public UsdGenOp
{
public:
    explicit InternProbe(UsdGenOpRecord const &r) : UsdGenOp(r) {}
    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override { return {3, 1}; }
    bool Capture(UsdGenCaptureContext const &, UsdGenCurveBuffer const &, UsdGenCapture *,
                 UsdGenDiagnostics *) override { return true; }
    void Evaluate(UsdGenEvalContext const &, UsdGenCapture const &,
                  UsdGenChunkView *) const override
    {
        switch (g_intern.load(std::memory_order_relaxed)) {
        case Intern::Exec:
            (void)UsdGenInternToken("x");
            break;
        case Intern::CompileScope: {
            UsdGenCompileTokenScope scope;
            (void)UsdGenInternToken("x");
            break;
        }
        case Intern::None: break;
        }
    }
};

void RegisterProbe(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenTokenCheckProbe";
    d.displayName = "TokenCheckProbe";
    d.create = [](UsdGenOpRecord const &rec, void *) -> std::unique_ptr<UsdGenOp> {
        return std::make_unique<InternProbe>(rec);
    };
    r.RegisterOp(d);
}

UsdGenGraphDesc MakeDesc()
{
    constexpr int N = 8;
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/probe");
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.restPoints = VtVec3fArray((N + 1) * (N + 1));
    s.uv = VtVec2fArray((N + 1) * (N + 1));
    for (int j = 0; j <= N; ++j)
        for (int i = 0; i <= N; ++i) {
            int const k = j * (N + 1) + i;
            s.restPoints[k] = GfVec3f(i * 0.1f, j * 0.1f, 0.0f);
            s.uv[k] = GfVec2f(float(i) / N, float(j) / N);
        }
    s.faceVertexCounts = VtIntArray(N * N, 4);
    s.faceVertexIndices = VtIntArray(N * N * 4);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            int const a = j * (N + 1) + i, o = (j * N + i) * 4;
            s.faceVertexIndices[o + 0] = a;
            s.faceVertexIndices[o + 1] = a + 1;
            s.faceVertexIndices[o + 2] = a + N + 2;
            s.faceVertexIndices[o + 3] = a + N + 1;
        }
    d.surfaces.push_back(std::move(s));
    auto add = [&](char const *name, char const *type, char const *input) {
        UsdGenNodeDesc n;
        n.path = SdfPath(std::string("/groom/") + name);
        n.type = TfToken(type);
        if (*input) n.inputs.push_back(SdfPath(std::string("/groom/") + input));
        else n.surfaces.push_back(SdfPath("/groom/surface"));
        d.nodes.push_back(std::move(n));
    };
    add("scatter", "UsdGenScatter", "");
    add("grow", "UsdGenGrow", "scatter");
    add("probe", "UsdGenTokenCheckProbe", "grow");
    return d;
}

bool HasText(std::vector<std::string> const &messages, char const *text)
{
    for (std::string const &m : messages)
        if (m.find(text) != std::string::npos) return true;
    return false;
}

}  // namespace

int main(int argc, char **argv)
{
    bool const off = argc > 1 && std::strcmp(argv[1], "off") == 0;
    Check(UsdGenRegisterStatic(&RegisterProbe), "probe registers before the first compile");
    Check(UsdGenTokenChecksEnabled() == !off,
          off ? "USDGEN_TOKEN_CHECKS=0 turns checks off" : "token checks are on by default");

    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const compiled = compiler.Compile(MakeDesc(), &graph);
    Check(compiled.ok, "graph compiles");
    if (!compiled.ok) return 1;
    Check(UsdGenOpRegistry::Get().IsFrozen(), "the first compile froze the registries");

    UsdGenScheduler scheduler(4);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    uint64_t gen = 0;
    auto run = [&](Intern intern) {
        g_intern.store(intern);
        for (int i = 0; i < graph.NodeCount(); ++i)
            for (auto &word : graph.Node(i).chunkDirty) word = UsdGenDirtyParameter;
        UsdGenRunResult result = scheduler.Run(graph, ctx, ++gen);
        g_intern.store(Intern::None);
        return result;
    };
    char const *const kMessage = "a token was interned during the cook";

    UsdGenRunResult const clean = run(Intern::None);
    Check(!clean.diagnostics.HasErrors() && clean.terminalOutput &&
              clean.terminalOutput->totalCurves > 0,
          "a run that interns nothing succeeds");

    uint64_t const execBefore = UsdGenExecInternCount();
    UsdGenRunResult const exec = run(Intern::Exec);
    Check(UsdGenExecInternCount() > execBefore, "Evaluate's intern counts as an exec intern");
    if (off) {
        Check(!exec.diagnostics.HasErrors(), "checks off: the run still succeeds");
        Check(HasText(exec.diagnostics.warnings, kMessage), "checks off: the run warns");
    } else {
        Check(exec.diagnostics.HasErrors() && HasText(exec.diagnostics.errors, kMessage),
              "an intern during Run fails it with the diagnostic");
    }

    UsdGenRunResult const scoped = run(Intern::CompileScope);
    Check(!scoped.diagnostics.HasErrors() && !HasText(scoped.diagnostics.warnings, kMessage),
          "an intern inside UsdGenCompileTokenScope does not fail the run");

    UsdGenRunResult const after = run(Intern::None);
    Check(!after.diagnostics.HasErrors(), "the next clean run succeeds again");

    std::printf("testUsdGenTokenChecks%s: %s\n", off ? " (off)" : "",
                g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
