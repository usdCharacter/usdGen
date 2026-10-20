// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGenCookStats — offline validation utility. Cooks every UsdGenDescription
// of a stage on the CPU reference lane (stage-sourced graph descriptions, the
// same path the T0 tests use) and prints, per description: strand and CV
// counts, CV-count range, clump count, wall time, validation/compile/run
// diagnostics. With --bake <out.usda>, also writes each cooked result as a
// plain linear BasisCurves prim (points, widths, uniform clumpId), so a groom
// can be inspected without the usdGen plugins.
//
//   usdGenCookStats scene.usda [--only <substring>] [--bake out.usda] [--threads N] [--hash]
// --hash prints the reference groomer's output hash (FNV-1a over points, counts,
// lengths, clump ids in generation order) for comparison with its build log.
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGenImaging/usdGenGraphDescBuilder.h"
#include "usdGenImaging/usdGenGraphDescBuilderStage.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include "pxr/base/vt/array.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/basisCurves.h"
#include "pxr/usd/usdGeom/metrics.h"
#include "pxr/usd/usdGeom/primvarsAPI.h"
#include "pxr/usd/usdGeom/tokens.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

UsdGenPlane const *FindPlane(std::vector<UsdGenPlane> const &planes, char const *name)
{
    for (UsdGenPlane const &p : planes)
        if (p.name == TfToken(name)) return &p;
    return nullptr;
}

/// FNV-1a-64 with the reference groomer's offset basis.
uint64_t Fnv(uint64_t h, void const *data, size_t bytes)
{
    unsigned char const *p = static_cast<unsigned char const *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

/// The reference build's output hash over the cooked strands in generation
/// order (strandIndex): points, point counts, strand lengths (restLength x
/// lengthScale, or the polyline length when the plane is absent) and clump ids.
uint64_t ReferenceHash(UsdGenCurveBuffer const &b, std::vector<int> const &counts,
                       double lengthScale)
{
    uint32_t const R = b.totalCurves;
    std::vector<uint32_t> first(R + 1, 0);
    for (uint32_t c = 0; c < R; ++c) first[c + 1] = first[c] + uint32_t(counts[c]);
    std::vector<uint32_t> order(R);
    for (uint32_t c = 0; c < R; ++c) order[c] = c;
    UsdGenPlane const *index = FindPlane(b.extraCurve, "strandIndex");
    if (index && index->type == TfToken("int") && index->i.size() == R)
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t c) {
            return index->i[a] < index->i[c];
        });
    UsdGenPlane const *rest = FindPlane(b.extraCurve, "restLength");
    UsdGenPlane const *clump = FindPlane(b.extraCurve, "clumpId_0");
    std::vector<float> points;
    std::vector<int32_t> n, ids;
    std::vector<float> lengths;
    for (uint32_t c : order) {
        float L = 0.0f;
        for (uint32_t k = first[c]; k < first[c + 1]; ++k) {
            points.push_back(b.px[k]); points.push_back(b.py[k]); points.push_back(b.pz[k]);
            if (k > first[c]) {
                float const dx = b.px[k] - b.px[k - 1], dy = b.py[k] - b.py[k - 1],
                            dz = b.pz[k] - b.pz[k - 1];
                L += std::sqrt(dx * dx + dy * dy + dz * dz);
            }
        }
        n.push_back(counts[c]);
        if (rest && rest->type == TfToken("float") && rest->f.size() == R)
            L = float(double(rest->f[c]) * lengthScale);
        lengths.push_back(L);
        ids.push_back(clump && clump->type == TfToken("int") && clump->i.size() == R ? clump->i[c] : -1);
    }
    uint64_t h = 0x14650fb0739d0383ull;
    h = Fnv(h, points.data(), points.size() * 4);
    h = Fnv(h, n.data(), n.size() * 4);
    h = Fnv(h, lengths.data(), lengths.size() * 4);
    h = Fnv(h, ids.data(), ids.size() * 4);
    return h;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: usdGenCookStats scene.usda [--only <substring>] [--bake out.usda] [--threads N]\n");
        return 2;
    }
    std::string scene = argv[1], only, bake;
    int threads = 8;
    bool hash = false, hydra = false, dumpSets = false;
    for (int i = 2; i < argc; ++i) {
        std::string const a = argv[i];
        if (a == "--hash") hash = true;
        if (a == "--hydra") hydra = true;
        if (a == "--dump-sets") dumpSets = true;
    }
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string const flag = argv[i];
        if (flag == "--hash" || flag == "--hydra" || flag == "--dump-sets") { --i; continue; }
        if (flag == "--only") only = argv[i + 1];
        else if (flag == "--bake") bake = argv[i + 1];
        else if (flag == "--threads") threads = std::max(1, std::atoi(argv[i + 1]));
    }
    UsdStageRefPtr stage = UsdStage::Open(scene);
    if (!stage) {
        std::fprintf(stderr, "cannot open %s\n", scene.c_str());
        return 2;
    }
    usdGenRegisterM1Operators();

    UsdStageRefPtr out;
    if (!bake.empty()) {
        out = UsdStage::CreateNew(bake);
        if (!out) {
            std::fprintf(stderr, "cannot create %s\n", bake.c_str());
            return 2;
        }
        UsdGeomSetStageUpAxis(out, UsdGeomGetStageUpAxis(stage));
        UsdGeomSetStageMetersPerUnit(out, UsdGeomGetStageMetersPerUnit(stage));
    }

    // --hydra builds every description through the production scene-index
    // path (the one usdview uses) instead of the stage-sourced builder.
    UsdImagingSceneIndices indices;
    if (hydra) {
        UsdImagingCreateSceneIndicesInfo info;
        info.stage = stage;
        indices = UsdImagingCreateSceneIndices(info);
        indices.stageSceneIndex->SetTime(UsdTimeCode::Default());
    }
    int failures = 0;
    size_t totalStrands = 0, totalCvs = 0;
    for (UsdPrim const &prim : stage->Traverse()) {
        if (prim.GetTypeName() != TfToken("UsdGenDescription")) continue;
        std::string const path = prim.GetPath().GetString();
        if (!only.empty() && path.find(only) == std::string::npos) continue;
        auto const t0 = std::chrono::steady_clock::now();
        UsdGenGraphDesc desc = hydra
            ? usdGenImaging::BuildGraphDescFromHydra(*indices.finalSceneIndex, prim.GetPath())
            : usdGenImaging::BuildGraphDescFromStage(stage, prim.GetPath());
        for (auto const &e : desc.validationErrors) std::printf("  validation: %s\n", e.c_str());
        if (dumpSets) {
            for (auto const &cs : desc.curveSets) {
                std::printf("  curveSet %s role=%s curves=%zu planes=%zu", cs.path.GetText(),
                            cs.role == UsdGenRole::Reference ? "reference" : "curves",
                            cs.curveVertexCounts.size(), cs.authoredPlanes.size());
                size_t shown = 0;
                for (auto const &pl : cs.authoredPlanes)
                    if (shown++ < 6) std::printf(" %s", pl.name.GetText());
                std::printf("\n");
            }
        }
        UsdGenCompiler compiler;
        UsdGenGraph graph;
        UsdGenCompileResult const compiled = compiler.Compile(desc, &graph);
        if (!compiled.ok) {
            ++failures;
            std::printf("%s: COMPILE FAILED\n", path.c_str());
            for (auto const &e : compiled.errors) std::printf("  %s\n", e.c_str());
            continue;
        }
        UsdGenScheduler scheduler = UsdGenScheduler(threads);
        UsdGenEvalContext ctx;
        ctx.desc = &graph.Desc();
        UsdGenRunResult const run = scheduler.Run(graph, ctx, 1);
        double const ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        for (auto const &w : run.diagnostics.warnings) std::printf("  warning: %s\n", w.c_str());
        if (run.diagnostics.HasErrors()) {
            ++failures;
            std::printf("%s: RUN FAILED\n", path.c_str());
            for (auto const &e : run.diagnostics.errors) std::printf("  %s\n", e.c_str());
            continue;
        }
        UsdGenCurveBuffer const &b = graph.Output();
        uint32_t minCv = 0, maxCv = 0;
        std::vector<int> counts(b.totalCurves, 0);
        for (uint32_t c = 0; c < b.totalCurves; ++c) {
            int const n = b.cvOffsets.empty()
                ? int(b.totalCurves ? b.totalCvs / b.totalCurves : 0)
                : b.cvOffsets[c + 1] - b.cvOffsets[c];
            counts[c] = n;
            minCv = c == 0 ? uint32_t(n) : std::min(minCv, uint32_t(n));
            maxCv = std::max(maxCv, uint32_t(n));
        }
        size_t clumps = 0;
        UsdGenPlane const *clump = FindPlane(b.extraCurve, "clumpId_0");
        if (clump && clump->type == TfToken("int")) {
            std::set<int> ids;
            for (int v : clump->i) if (v >= 0) ids.insert(v);
            clumps = ids.size();
        }
        std::printf("%s: strands=%u cvs=%u cvPerStrand=[%u,%u] clumps=%zu ms=%.1f nodes=%zu\n",
                    path.c_str(), b.totalCurves, b.totalCvs, minCv, maxCv, clumps, ms,
                    desc.nodes.size());
        if (hash)
            std::printf("  referenceHash=%016llx (restLength x 1)\n",
                        (unsigned long long)ReferenceHash(b, counts, 1.0));
        totalStrands += b.totalCurves;
        totalCvs += b.totalCvs;

        if (out && b.totalCurves) {
            std::string name = path;
            std::replace(name.begin(), name.end(), '/', '_');
            UsdGeomBasisCurves curves = UsdGeomBasisCurves::Define(out, SdfPath("/Baked/" + name.substr(1)));
            VtVec3fArray points(b.totalCvs);
            for (uint32_t i = 0; i < b.totalCvs; ++i) points[i] = GfVec3f(b.px[i], b.py[i], b.pz[i]);
            curves.CreateTypeAttr(VtValue(UsdGeomTokens->linear));
            curves.CreatePointsAttr(VtValue(points));
            curves.CreateCurveVertexCountsAttr(VtValue(VtIntArray(counts.begin(), counts.end())));
            if (b.width.size() == b.totalCvs) {
                curves.CreateWidthsAttr(VtValue(b.width));
                curves.SetWidthsInterpolation(UsdGeomTokens->vertex);
            } else {
                curves.CreateWidthsAttr(VtValue(VtFloatArray(1, desc.defaultWidth)));
                curves.SetWidthsInterpolation(UsdGeomTokens->constant);
            }
            if (clump && clump->type == TfToken("int") && clump->i.size() == b.totalCurves)
                UsdGeomPrimvarsAPI(curves.GetPrim()).CreatePrimvar(
                    TfToken("clumpId"), SdfValueTypeNames->IntArray, UsdGeomTokens->uniform)
                    .Set(clump->i);
        }
    }
    std::printf("total: strands=%zu cvs=%zu failures=%d\n", totalStrands, totalCvs, failures);
    if (out) out->GetRootLayer()->Save();
    return failures ? 1 : 0;
}
