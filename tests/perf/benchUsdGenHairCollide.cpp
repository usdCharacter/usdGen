// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// benchUsdGenHairCollide — UsdGenCollide 1M-hair collision benchmark.
//
// Fixture: a CurveSource groom of --hairs straight strands over an NX x NY
// quad-grid collider, cooked through one UsdGenCollide node (offset 0.5,
// pushAmount 1, iterations 1). Legs:
//   static   Compile once, then ForceFullDirty + Run per rep (steady-state
//            collide Evaluate; the capture is reused).
//   hair     per-frame deforming groom (curve points move; collider still):
//            Recompile + Run per frame, capture reused.
//   collider per-frame deforming collider (points move, surfaceGeneration
//            bumps): Recompile + Run per frame, collide re-captures.
//   stiff    static leg with resolveType=stiff.
//   probe    static Runs with the collide phase probe on, single-threaded,
//            so the broadphase / narrowphase / response split is wall time
//            and adds up to the probe wall. Informational only — the
//            headline totals come from the unprobed legs (the probe's clock
//            reads perturb the pass slightly).
//
// Hair-hair (self) collision is not benchmarked: no self-collision operator
// exists in-tree (UsdGenCollide is hair-surface only), so there is nothing
// to run it through.
//
// Usage: benchUsdGenHairCollide [--hairs N] [--cvs M] [--grid G] [--reps R]
//   [--threads T] [--stage NAME] [--quick]
// --quick sets 2k hairs, a 10x10 collider, 3 reps (the ctest shape).
// Output: one "result <stage>_ms <median> (best ... worst ...)" line plus a
// "checksum <stage> <fnv1a>" line per leg (median of R, fixture built once
// outside every timing window). Exit is nonzero on hard failure only
// (compile/run errors, NaN output, nondeterministic repeat, an idle leg
// that evaluated zero chunks); timings are informational.
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/opRegistry.h"
#include "usdGen/ops/collide.h"
#include "usdGen/scheduler.h"
#include "usdGen/types.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

struct Options {
    int hairs = 1000000;
    int cvs = 5;
    int grid = 100;  // grid x grid quads -> 2x triangles
    int reps = 9;
    int threads = 0;  // 0 = product default (calibration / USDGEN_THREAD_LIMIT)
    std::string stage;  // empty = all
    bool quick = false;
};

double Median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double NowMs()
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Deterministic [0,1) hash; the whole fixture derives from it (no RNG).
double Hash01(uint64_t k)
{
    k += 0x9e3779b97f4a7c15ULL;
    k = (k ^ (k >> 30)) * 0xbf58476d1ce4e5b9ULL;
    k = (k ^ (k >> 27)) * 0x94d049bb133111ebULL;
    return double((k ^ (k >> 31)) >> 11) / double(1ull << 53);
}

struct Fnv {
    uint64_t h = 14695981039346656037ull;
    void Add(void const *p, size_t n)
    {
        auto const *b = static_cast<unsigned char const *>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= uint64_t(b[i]);
            h *= 1099511628211ull;
        }
    }
    void AddSize(size_t n) { Add(&n, sizeof(n)); }
};

uint64_t ChecksumTerminal(UsdGenGraph const &graph)
{
    UsdGenCurveBuffer const &t = graph.Output();
    Fnv fnv;
    fnv.AddSize(t.px.size());
    if (!t.px.empty()) fnv.Add(t.px.cdata(), t.px.size() * sizeof(float));
    if (!t.py.empty()) fnv.Add(t.py.cdata(), t.py.size() * sizeof(float));
    if (!t.pz.empty()) fnv.Add(t.pz.cdata(), t.pz.size() * sizeof(float));
    return fnv.h;
}

bool OutputFinite(UsdGenGraph const &graph)
{
    UsdGenCurveBuffer const &t = graph.Output();
    for (float v : t.px) if (!std::isfinite(v)) return false;
    for (float v : t.py) if (!std::isfinite(v)) return false;
    for (float v : t.pz) if (!std::isfinite(v)) return false;
    return true;
}

void Report(char const *stage, std::vector<double> const &ms,
            uint64_t checksum, char const *extra = "")
{
    double const med = Median(ms);
    double const best = *std::min_element(ms.begin(), ms.end());
    double const worst = *std::max_element(ms.begin(), ms.end());
    std::printf("result %s_ms %.3f (best %.3f worst %.3f)%s%s\n", stage,
                med, best, worst, extra[0] ? " " : "", extra);
    std::printf("checksum %s %016llx\n", stage,
                (unsigned long long)checksum);
    std::fflush(stdout);
}

bool WantStage(Options const &opts, char const *stage)
{
    return opts.stage.empty() || opts.stage == stage;
}

// Force every chunk of the collide node fully value-dirty (steady-state
// worst case: the capture is reused, every chunk re-evaluates).
void ForceCollideDirty(UsdGenGraph &graph, UsdGenGraph::NodeId collide)
{
    for (uint8_t &byte : graph.Node(collide).chunkDirty)
        byte = UsdGenDirtyParameter;
}

uint64_t CollideChunksEvaluated(UsdGenRunResult const &run,
                                UsdGenGraph::NodeId collide)
{
    for (UsdGenNodeRunStats const &st : run.nodeStats)
        if (st.id == collide) return st.chunksEvaluated;
    return 0;
}

void CollideCaptureEvalMs(UsdGenRunResult const &run,
                          UsdGenGraph::NodeId collide,
                          double *captureMs, double *evalMs)
{
    *captureMs = 0.0;
    *evalMs = 0.0;
    for (UsdGenNodeRunStats const &st : run.nodeStats) {
        if (st.id != collide) continue;
        *captureMs = st.captureMs;
        *evalMs = st.evalMs;
    }
}

// NX x NY grid collider over [0,100]^2 with gentle deterministic bumps so
// queries span interior/edge/vertex regions. Points at frame f; f=0 is rest.
UsdGenSurfaceDesc MakeCollider(int n, int frame)
{
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/bench/collider");
    s.id = 0;
    double const bump = frame == 0 ? 0.0 : 0.35 * std::sin(0.7 * frame);
    s.points = VtVec3fArray(size_t(n + 1) * size_t(n + 1));
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            float const x = 100.0f * float(i) / float(n);
            float const y = 100.0f * float(j) / float(n);
            float z = 2.0f * float(std::sin(0.11 * x) * std::cos(0.13 * y));
            if (frame != 0) {
                z += float(bump * std::sin(0.05 * x + 0.3 * frame) *
                           std::cos(0.06 * y - 0.2 * frame));
            }
            s.points[size_t(j) * size_t(n + 1) + size_t(i)] =
                GfVec3f(x, y, z);
        }
    }
    s.restPoints = s.points;
    s.faceVertexCounts = VtIntArray(size_t(n) * size_t(n), 4);
    s.faceVertexIndices = VtIntArray(size_t(n) * size_t(n) * 4);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            size_t const q = (size_t(j) * size_t(n) + size_t(i)) * 4;
            int const v00 = j * (n + 1) + i;
            s.faceVertexIndices[q + 0] = v00;
            s.faceVertexIndices[q + 1] = v00 + 1;
            s.faceVertexIndices[q + 2] = v00 + n + 2;
            s.faceVertexIndices[q + 3] = v00 + n + 1;
        }
    }
    s.surfaceGeneration = frame == 0 ? 1 : uint64_t(1000 + frame);
    return s;
}

// Straight strands over the collider span: roots jittered on a lattice, root
// height scattered around the surface (about half the CVs penetrate at
// offset 0.5), strands tilted deterministically. Frame f waves the groom;
// f=0 is rest.
UsdGenCurveSetDesc MakeGroom(int hairs, int cvs, int frame)
{
    UsdGenCurveSetDesc set;
    set.path = SdfPath("/bench/hair");
    set.role = UsdGenRole::Curves;
    set.curveRole = TfToken("hair");
    set.curveVertexCounts.assign(size_t(hairs), cvs);
    set.points.resize(size_t(hairs) * size_t(cvs));
    int const side = int(std::ceil(std::sqrt(double(hairs))));
    for (int c = 0; c < hairs; ++c) {
        int const gx = c % side, gy = c / side;
        double const jx = Hash01(uint64_t(c) * 4);
        double const jy = Hash01(uint64_t(c) * 4 + 1);
        double const jz = Hash01(uint64_t(c) * 4 + 2);
        double const jt = Hash01(uint64_t(c) * 4 + 3);
        float const rx = 100.0f * float(gx + jx) / float(side);
        float const ry = 100.0f * float(gy + jy) / float(side);
        float rz = 2.0f * float(std::sin(0.11 * rx) * std::cos(0.13 * ry)) +
            float(-0.3 + 1.0 * jz);
        if (frame != 0)
            rz += float(0.2 * std::sin(0.5 * frame + 6.28 * jt));
        float const tiltX = float(0.3 * (jt - 0.5));
        for (int i = 0; i < cvs; ++i) {
            set.points[size_t(c) * size_t(cvs) + size_t(i)] =
                GfVec3f(rx + tiltX * float(i) * 0.3f,
                        ry + float(i) * 0.3f, rz + 0.05f * float(i));
        }
    }
    set.rest = set.points;
    set.widths.assign(size_t(hairs) * size_t(cvs), 0.1f);
    set.curveId.resize(size_t(hairs));
    for (int c = 0; c < hairs; ++c) set.curveId[size_t(c)] = uint64_t(c);
    set.skinPrim.assign(size_t(hairs), 0);
    set.skinPrimUv.assign(size_t(hairs), GfVec2f(0.2f, 0.3f));
    set.rootFrame.assign(size_t(hairs), GfMatrix4d(1.0));
    return set;
}

UsdGenGraphDesc MakeDesc(Options const &opts, bool stiff)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/bench");
    desc.defaultWidth = 0.01f;
    desc.curveSets.push_back(MakeGroom(opts.hairs, opts.cvs, 0));
    desc.surfaces.push_back(MakeCollider(opts.grid, 0));
    UsdGenNodeDesc source;
    source.path = SdfPath("/bench/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {SdfPath("/bench/hair")};
    source.params = {{TfToken("rebind"), VtValue(TfToken("never")), false},
                     {TfToken("useRest"), VtValue(true), false}};
    desc.nodes.push_back(source);
    UsdGenNodeDesc collide;
    collide.path = SdfPath("/bench/collide");
    collide.type = TfToken("UsdGenCollide");
    collide.seed = 7;
    collide.inputs = {SdfPath("/bench/source")};
    collide.params = {{TfToken("offset"), VtValue(0.5f), false},
                      {TfToken("pushAmount"), VtValue(1.0f), false},
                      {TfToken("iterations"), VtValue(1), false}};
    if (stiff)
        collide.params.push_back(
            {TfToken("resolveType"), VtValue(TfToken("stiff")), false});
    collide.surfaces = {SdfPath("/bench/collider")};
    desc.nodes.push_back(collide);
    desc.terminal = SdfPath("/bench/collide");
    return desc;
}

}  // namespace

int main(int argc, char **argv)
{
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](char const *flag) -> char const * {
            if (i + 1 >= argc) {
                std::printf("benchUsdGenHairCollide: %s needs a value\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--hairs") opts.hairs = std::atoi(need("--hairs"));
        else if (arg == "--cvs") opts.cvs = std::atoi(need("--cvs"));
        else if (arg == "--grid") opts.grid = std::atoi(need("--grid"));
        else if (arg == "--reps") opts.reps = std::atoi(need("--reps"));
        else if (arg == "--threads") opts.threads = std::atoi(need("--threads"));
        else if (arg == "--stage") opts.stage = need("--stage");
        else if (arg == "--quick") opts.quick = true;
        else if (arg == "--help" || arg == "-h") {
            std::printf("usage: benchUsdGenHairCollide [--hairs N] [--cvs M] "
                        "[--grid G] [--reps R] [--threads T] [--stage NAME] "
                        "[--quick]\n");
            return 0;
        } else {
            std::printf("benchUsdGenHairCollide: unknown arg %s\n", arg.c_str());
            return 2;
        }
    }
    if (opts.quick) {
        opts.hairs = 2000;
        opts.grid = 10;
        opts.reps = 3;
    }
    if (opts.hairs <= 0 || opts.cvs <= 0 || opts.grid <= 0 || opts.reps <= 0) {
        std::printf("benchUsdGenHairCollide: hairs/cvs/grid/reps must be > 0\n");
        return 2;
    }
    usdGenRegisterM1Operators();
    std::printf("config hairs=%d cvs=%d grid=%d tris=%d reps=%d threads=%d\n",
                opts.hairs, opts.cvs, opts.grid, 2 * opts.grid * opts.grid,
                opts.reps, opts.threads);
    std::fflush(stdout);

    UsdGenCompiler compiler;
    UsdGenScheduler scheduler(opts.threads);
    UsdGenEvalContext ctx;
    uint64_t gen = 0;

    // One graph per resolveType; each leg warms up, checks determinism, then
    // times. Returns false on hard failure.
    auto runStaticLeg = [&](bool stiff, char const *name) -> bool {
        UsdGenGraphDesc desc = MakeDesc(opts, stiff);
        UsdGenGraph graph;
        if (!compiler.Compile(desc, &graph).ok) {
            std::printf("%s: compile failed\n", name);
            return false;
        }
        ctx.desc = &graph.Desc();
        UsdGenGraph::NodeId const collide =
            graph.NodeIdForPath(SdfPath("/bench/collide"));
        if (scheduler.Run(graph, ctx, ++gen).diagnostics.HasErrors()) {
            std::printf("%s: warmup run failed\n", name);
            return false;
        }
        uint64_t const repeat0 = ChecksumTerminal(graph);
        ForceCollideDirty(graph, collide);
        scheduler.Run(graph, ctx, ++gen);
        if (ChecksumTerminal(graph) != repeat0) {
            std::printf("%s: nondeterministic repeat FAIL\n", name);
            return false;
        }
        std::vector<double> tWall, tCapture, tEval;
        uint64_t checksum = 0;
        for (int i = 0; i < opts.reps; ++i) {
            ForceCollideDirty(graph, collide);
            double const t0 = NowMs();
            UsdGenRunResult const run = scheduler.Run(graph, ctx, ++gen);
            tWall.push_back(NowMs() - t0);
            if (run.diagnostics.HasErrors()) {
                std::printf("%s: rep %d failed\n", name, i);
                return false;
            }
            if (CollideChunksEvaluated(run, collide) == 0) {
                std::printf("%s: rep %d evaluated zero chunks\n", name, i);
                return false;
            }
            double capMs = 0.0, evalMs = 0.0;
            CollideCaptureEvalMs(run, collide, &capMs, &evalMs);
            tCapture.push_back(capMs);
            tEval.push_back(evalMs);
            checksum = ChecksumTerminal(graph);
            if (checksum != repeat0) {
                std::printf("%s: rep %d checksum moved on static input\n",
                            name, i);
                return false;
            }
        }
        if (!OutputFinite(graph)) {
            std::printf("%s: non-finite output\n", name);
            return false;
        }
        char extra[256];
        std::snprintf(extra, sizeof(extra),
                      "collide_capture=%.3f collide_eval=%.3f",
                      Median(tCapture), Median(tEval));
        Report(name, tWall, checksum, extra);
        return true;
    };

    // Deforming-input leg: per frame the desc moves (hair points or collider
    // points + generation), then Recompile + Run. The collide capture must be
    // reused across hair frames and must move across collider frames.
    auto runDeformLeg = [&](bool moveCollider, char const *name) -> bool {
        UsdGenGraphDesc desc = MakeDesc(opts, false);
        UsdGenGraph graph;
        if (!compiler.Compile(desc, &graph).ok) {
            std::printf("%s: compile failed\n", name);
            return false;
        }
        UsdGenGraph::NodeId const collide =
            graph.NodeIdForPath(SdfPath("/bench/collide"));
        ctx.desc = &graph.Desc();
        if (scheduler.Run(graph, ctx, ++gen).diagnostics.HasErrors()) {
            std::printf("%s: warmup run failed\n", name);
            return false;
        }
        UsdGenEpoch const epoch0 = graph.Node(collide).captureEpoch;
        std::vector<double> tWall, tCapture, tEval;
        uint64_t prev = ChecksumTerminal(graph), checksum = prev;
        bool epochMoved = false;
        for (int i = 1; i <= opts.reps; ++i) {
            if (moveCollider)
                desc.surfaces[0] = MakeCollider(opts.grid, i);
            else
                desc.curveSets[0] = MakeGroom(opts.hairs, opts.cvs, i);
            double const t0 = NowMs();
            UsdGenCompileResult const rr = compiler.Recompile(desc, &graph);
            ctx.desc = &graph.Desc();
            UsdGenRunResult const run = scheduler.Run(graph, ctx, ++gen);
            tWall.push_back(NowMs() - t0);
            if (!rr.ok || run.diagnostics.HasErrors()) {
                std::printf("%s: frame %d failed\n", name, i);
                return false;
            }
            if (CollideChunksEvaluated(run, collide) == 0) {
                std::printf("%s: frame %d evaluated zero chunks\n", name, i);
                return false;
            }
            epochMoved = epochMoved ||
                graph.Node(collide).captureEpoch != epoch0;
            double capMs = 0.0, evalMs = 0.0;
            CollideCaptureEvalMs(run, collide, &capMs, &evalMs);
            tCapture.push_back(capMs);
            tEval.push_back(evalMs);
            checksum = ChecksumTerminal(graph);
            if (checksum == prev) {
                std::printf("%s: frame %d checksum identical on moved input\n",
                            name, i);
                return false;
            }
            prev = checksum;
        }
        if (moveCollider && !epochMoved) {
            std::printf("%s: capture never re-ran on a moving collider\n", name);
            return false;
        }
        // Note: the hair leg re-captures too (an upstream re-capture bumps
        // the topologyVersion the collide digest mixes in), so per-frame
        // capture cost is real on both deform legs; the extra field reports
        // whether the capture moved.
        if (!OutputFinite(graph)) {
            std::printf("%s: non-finite output\n", name);
            return false;
        }
        char extra[256];
        std::snprintf(extra, sizeof(extra),
                      "collide_capture=%.3f collide_eval=%.3f recaptured=%s",
                      Median(tCapture), Median(tEval),
                      epochMoved ? "yes" : "no");
        Report(name, tWall, checksum, extra);
        return true;
    };

    // Phase-probe leg: static Runs with the probe on, on a dedicated
    // single-threaded scheduler, so the phase sums are wall time. Totals
    // are informational (the probe perturbs); the split shows where time
    // goes. The probe must not change a single output bit.
    auto runProbeLeg = [&]() -> bool {
        char const *name = "probe";
        UsdGenGraphDesc desc = MakeDesc(opts, false);
        UsdGenGraph graph;
        if (!compiler.Compile(desc, &graph).ok) {
            std::printf("%s: compile failed\n", name);
            return false;
        }
        UsdGenScheduler probeScheduler(1);
        UsdGenEvalContext probeCtx;
        probeCtx.desc = &graph.Desc();
        uint64_t probeGen = 0;
        UsdGenGraph::NodeId const collide =
            graph.NodeIdForPath(SdfPath("/bench/collide"));
        if (probeScheduler.Run(graph, probeCtx, ++probeGen)
                .diagnostics.HasErrors()) {
            std::printf("%s: warmup run failed\n", name);
            return false;
        }
        uint64_t const expected = ChecksumTerminal(graph);
        std::vector<double> tWall, tBroad, tNarrow, tResp;
        for (int i = 0; i < 3; ++i) {
            ForceCollideDirty(graph, collide);
            UsdGenCollideOp::ResetPhaseProbe();
            UsdGenCollideOp::SetPhaseProbe(true);
            double const t0 = NowMs();
            UsdGenRunResult const run =
                probeScheduler.Run(graph, probeCtx, ++probeGen);
            double const wall = NowMs() - t0;
            UsdGenCollideOp::SetPhaseProbe(false);
            UsdGenCollidePhaseStats const ph =
                UsdGenCollideOp::ReadPhaseProbe();
            if (run.diagnostics.HasErrors()) {
                std::printf("%s: rep %d failed\n", name, i);
                return false;
            }
            if (ChecksumTerminal(graph) != expected) {
                std::printf("%s: probe perturbed the output bits\n", name);
                return false;
            }
            tWall.push_back(wall);
            tBroad.push_back(ph.broadMs);
            tNarrow.push_back(ph.narrowMs);
            tResp.push_back(ph.respMs);
            if (i == 0)
                std::printf("probe_counts queries=%llu tritests=%llu "
                            "nodevisits=%llu\n",
                            (unsigned long long)ph.queries,
                            (unsigned long long)ph.triTests,
                            (unsigned long long)ph.nodeVisits);
        }
        char extra[256];
        std::snprintf(extra, sizeof(extra),
                      "broad=%.3f narrow=%.3f resp=%.3f (1-thread wall)",
                      Median(tBroad), Median(tNarrow), Median(tResp));
        Report(name, tWall, expected, extra);
        return true;
    };

    bool ok = true;
    if (WantStage(opts, "static") && !runStaticLeg(false, "static")) ok = false;
    if (ok && WantStage(opts, "hair") && !runDeformLeg(false, "hair")) ok = false;
    if (ok && WantStage(opts, "collider") && !runDeformLeg(true, "collider"))
        ok = false;
    if (ok && WantStage(opts, "stiff") && !runStaticLeg(true, "stiff")) ok = false;
    if (ok && WantStage(opts, "probe") && !runProbeLeg()) ok = false;
    return ok ? 0 : 1;
}
