// The guide-interpolation and Ptex clump examples, cooked end to end.
//
// examples/guide-interpolate-plane.usda
//   * voronoi variant: usdGen:region is geoSampler("guideCurves", "$index"),
//     so every strand's heaviest guide is the guide rooted nearest it;
//   * ptexMap variant: usdGen:region is ptex("regionMap"), so every guide a
//     strand blends lies in the strand's own map region (crossover 0);
//   * smooth variant: strands blend several guides.
// examples/clump-ptex-plane.usda
//   * clumpId_0 follows the Ptex clump map, clumpId_1 nests inside it, and
//     clumped tips converge;
// Dirty propagation, through the groom scene index:
//   * editing the guide curves recooks the published strands;
//   * retargeting the expression's input:guideCurves recooks them;
//   * retargeting a UsdGenPtexMap's file recooks the clumps.
#include "utils/pxrInterop.h"
#include "usdGen/compiler.h"
#include "usdGenTestUtils/hydraPublishers.h"
#include "usdGenUsd/lower.h"
#include "nodeSets.h"
#include "usdGen/graph.h"
#include "usdGen/maps/ptexMap.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGenUsd/walk/usdGenGraphDescBuilderStage.h"

#include "pxr/imaging/hd/dataSource.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/relationship.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/variantSets.h"
#include "pxr/usd/usdGeom/basisCurves.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what.c_str()); }
    else std::printf("ok: %s\n", what.c_str());
}

constexpr char const *kGuideScene = USDGEN_TEST_SOURCE_DIR "/examples/guide-interpolate-plane.usda";
constexpr char const *kClumpScene = USDGEN_TEST_SOURCE_DIR "/examples/clump-ptex-plane.usda";
constexpr char const *kGuideDescription = "/World/Groom/Hair";
constexpr char const *kClumpDescription = "/World/Groom/Fur";

struct Cooked {
    bool ok = false;
    UsdGenGraphDesc desc;
    UsdGenCurveBuffer out;
    uint32_t cvs = 0;
};

Cooked Cook(UsdStageRefPtr const &stage, char const *description, std::string const &label)
{
    Cooked result;
    result.desc = usdGenImaging::BuildGraphDescFromStage(stage, SdfPath(description));
    for (auto const &error : result.desc.validationErrors)
        std::printf("  validation: %s\n", error.c_str());
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const compiled = usdGenUsd::Compile(compiler, result.desc, &graph);
    if (!compiled.ok) {
        Check(false, label + ": compile");
        for (auto const &error : compiled.errors) std::printf("  %s\n", error.c_str());
        return result;
    }
    UsdGenScheduler scheduler(4);
    UsdGenEvalContext context;
    UsdGenRunResult const run = scheduler.Run(graph, context, 1);
    for (auto const &warning : run.diagnostics.warnings) std::printf("  warning: %s\n", warning.c_str());
    if (run.diagnostics.HasErrors()) {
        Check(false, label + ": cook");
        for (auto const &error : run.diagnostics.errors) std::printf("  %s\n", error.c_str());
        return result;
    }
    result.out = OutputBuffer(graph);
    result.ok = result.out.totalCurves > 0 && result.out.cvOffsets.empty() &&
        result.out.totalCvs % result.out.totalCurves == 0;
    result.cvs = result.ok ? result.out.totalCvs / result.out.totalCurves : 0;
    Check(result.ok, label + ": cooks to a uniform strand set (" +
          std::to_string(result.out.totalCurves) + " strands)");
    return result;
}

UsdGenPlane const *Plane(UsdGenCurveBuffer const &b, char const *name)
{
    for (auto const &plane : b.extraCurve)
        if (plane.name == name) return &plane;
    return nullptr;
}

usdGen::UsdGenVec3f Cv(UsdGenCurveBuffer const &b, size_t cv) { return usdGen::UsdGenVec3f(b.px[cv], b.py[cv], b.pz[cv]); }

/// Guide roots (first CVs) of the sampled geometry.
std::vector<usdGen::UsdGenVec3f> GuideRoots(UsdGenGeometryDesc const &g)
{
    std::vector<usdGen::UsdGenVec3f> roots;
    size_t first = 0;
    for (int n : g.counts) { roots.push_back(g.points[first]); first += size_t(n); }
    return roots;
}

size_t Nearest(std::vector<usdGen::UsdGenVec3f> const &points, usdGen::UsdGenVec3f const &q)
{
    size_t best = 0;
    double bestDistance = 1e30;
    for (size_t i = 0; i < points.size(); ++i) {
        double const d = (points[i] - q).GetLengthSq();
        if (d < bestDistance) { bestDistance = d; best = i; }
    }
    return best;
}

void SetVariant(UsdStageRefPtr const &stage, char const *description, char const *variant)
{
    UsdVariantSet set = stage->GetPrimAtPath(SdfPath(description)).GetVariantSet("region");
    Check(set.SetVariantSelection(variant), std::string("select region variant ") + variant);
}

// --- guide example -------------------------------------------------------------

void CheckGuideVoronoi(UsdStageRefPtr const &stage)
{
    SetVariant(stage, kGuideDescription, "voronoi");
    Cooked c = Cook(stage, kGuideDescription, "guide example, voronoi regions");
    if (!c.ok) return;
    Check(c.desc.geometries.size() == 1 &&
          c.desc.geometries[0].path == SdfPath("/World/Guides") &&
          c.desc.geometries[0].kind == UsdGenGeometryKind::Curves,
          "input:guideCurves resolves to the guide curves");
    Check(c.cvs == 12, "strands carry usdGen:cvCount CVs");
    UsdGenPlane const *index = Plane(c.out, "guideIndex");
    UsdGenPlane const *weight = Plane(c.out, "guideWeight");
    Check(index && index->type == TfToken("int") && index->arity == 3 &&
          index->i.size() == c.out.totalCurves * 3, "guideIndex is published per strand");
    Check(weight && weight->arity == 3 && weight->f.size() == c.out.totalCurves * 3,
          "guideWeight is published per strand");
    if (!index || !weight || c.desc.geometries.empty()) return;

    std::vector<usdGen::UsdGenVec3f> const roots = GuideRoots(c.desc.geometries[0]);
    size_t ownCell = 0, followsLean = 0;
    for (size_t s = 0; s < c.out.totalCurves; ++s) {
        usdGen::UsdGenVec3f const root = c.out.rest[s * c.cvs];
        size_t const cell = Nearest(roots, root);
        if (index->i[s * 3] == int(cell)) ++ownCell;
        // The strand leans the way its cell's guide leans.
        size_t first = 0;
        for (size_t g = 0; g < cell; ++g) first += size_t(c.desc.geometries[0].counts[g]);
        size_t const last = first + size_t(c.desc.geometries[0].counts[cell]) - 1;
        usdGen::UsdGenVec3f guideLean = c.desc.geometries[0].points[last] - c.desc.geometries[0].points[first];
        usdGen::UsdGenVec3f strandLean = Cv(c.out, (s + 1) * c.cvs - 1) - Cv(c.out, s * c.cvs);
        guideLean[1] = strandLean[1] = 0.0f;
        if (usdGen::UsdGenDot(guideLean, strandLean) > 0.0f) ++followsLean;
    }
    double const n = double(c.out.totalCurves);
    std::printf("  heaviest guide is the voronoi cell's: %.1f%%, leans with it: %.1f%%\n",
                100.0 * ownCell / n, 100.0 * followsLean / n);
    Check(ownCell == c.out.totalCurves, "every strand's heaviest guide is its voronoi cell's guide");
    Check(followsLean > 0.95 * n, "strands lean with their cell's guide");
}

void CheckGuidePtex(UsdStageRefPtr const &stage)
{
    SetVariant(stage, kGuideDescription, "ptexMap");
    Cooked c = Cook(stage, kGuideDescription, "guide example, ptex region map");
    if (!c.ok) return;
    UsdGenPlane const *index = Plane(c.out, "guideIndex");
    UsdGenPlane const *weight = Plane(c.out, "guideWeight");
    if (!index || !weight) { Check(false, "guide planes are published"); return; }

    // Independent region lookup: the same map, read at every strand root.
    UsdGenMapDesc const *map = nullptr;
    for (auto const &m : c.desc.maps) if (m.path.GetName() == "regionMap") map = &m;
    Check(map && !map->resolvedAssetPath.empty(), "the region map resolves to a file");
    if (!map) return;
    UsdGenPtexMapOptions options;
    options.filter = "nearest";
    std::string error;
    auto texture = UsdGenPtexTexture::Open(map->resolvedAssetPath, options, &error);
    Check(bool(texture), "the baked region map opens: " + error);
    if (!texture) return;
    auto sampler = texture->MakeSampler();
    UsdGenSurfaceDesc const *skin = nullptr;
    for (auto const &s : c.desc.surfaces) if (!s.faceVertexCounts.empty()) skin = &s;
    if (!skin) { Check(false, "the skin surface is in the description"); return; }
    std::vector<int> offsets(skin->faceVertexCounts.size() + 1, 0);
    for (size_t f = 0; f < skin->faceVertexCounts.size(); ++f)
        offsets[f + 1] = offsets[f] + skin->faceVertexCounts[f];
    std::vector<int> const firstIds =
        UsdGenPtexFirstFaceIds(skin->faceVertexCounts.cdata(), skin->faceVertexCounts.size());
    auto regionAt = [&](size_t strand) {
        usdGen::UsdGenVec3f const p = c.out.rest[strand * c.cvs];
        int id = -1; float u = 0, v = 0, value = -1;
        UsdGenPtexFaceCoordinate(reinterpret_cast<float const *>(skin->restPoints.cdata()),
                                 skin->restPoints.size(), skin->faceVertexCounts.cdata(),
                                 skin->faceVertexIndices.cdata(), offsets.data(),
                                 skin->faceVertexCounts.size(), firstIds.data(), false,
                                 c.out.rootPrim[strand], p[0], p[1], p[2], &id, &u, &v);
        sampler->Sample(id, u, v, &value);
        return int(std::lround(value));
    };
    std::vector<int> strandRegion(c.out.totalCurves);
    for (size_t s = 0; s < c.out.totalCurves; ++s) strandRegion[s] = regionAt(s);
    // A guide's region is the region of the strand rooted nearest it.
    std::vector<usdGen::UsdGenVec3f> const roots = GuideRoots(c.desc.geometries[0]);
    std::vector<usdGen::UsdGenVec3f> strandRoots(c.out.totalCurves);
    for (size_t s = 0; s < c.out.totalCurves; ++s) strandRoots[s] = c.out.rest[s * c.cvs];
    std::vector<int> guideRegion(roots.size());
    for (size_t g = 0; g < roots.size(); ++g)
        guideRegion[g] = strandRegion[Nearest(strandRoots, roots[g])];
    std::set<int> regions(strandRegion.begin(), strandRegion.end());
    size_t crossings = 0, blended = 0;
    for (size_t s = 0; s < c.out.totalCurves; ++s) {
        size_t contributing = 0;
        for (int k = 0; k < 3; ++k) {
            int const g = index->i[s * 3 + k];
            if (g < 0 || weight->f[s * 3 + k] <= 0.0f) continue;
            ++contributing;
            if (guideRegion[size_t(g)] != strandRegion[s]) ++crossings;
        }
        if (contributing > 1) ++blended;
    }
    std::printf("  %zu map regions, %zu strands blend several guides, %zu cross-region weights\n",
                regions.size(), blended, crossings);
    Check(regions.size() >= 3, "the map splits the skin into several regions");
    Check(crossings == 0, "no strand takes weight from a guide of another map region");
    Check(blended > c.out.totalCurves / 2, "inside a region, strands interpolate between guides");
}

void CheckGuideSmooth(UsdStageRefPtr const &stage)
{
    SetVariant(stage, kGuideDescription, "smooth");
    Cooked c = Cook(stage, kGuideDescription, "guide example, no regions");
    if (!c.ok) return;
    UsdGenPlane const *weight = Plane(c.out, "guideWeight");
    if (!weight) { Check(false, "guideWeight is published"); return; }
    size_t blended = 0;
    for (size_t s = 0; s < c.out.totalCurves; ++s)
        if (weight->f[s * 3 + 1] > 0.0f) ++blended;
    Check(blended > c.out.totalCurves * 3 / 4, "without regions most strands blend several guides");
}

// --- clump example ---------------------------------------------------------------

void CheckClump(UsdStageRefPtr const &stage)
{
    Cooked c = Cook(stage, kClumpDescription, "ptex clump example");
    if (!c.ok) return;
    UsdGenPlane const *level0 = Plane(c.out, "clumpId_0");
    UsdGenPlane const *level1 = Plane(c.out, "clumpId_1");
    Check(level0 && level0->type == TfToken("int") && level0->i.size() == c.out.totalCurves,
          "clumpId_0 is published per strand");
    Check(level1 && level1->i.size() == c.out.totalCurves, "clumpId_1 is published per strand");
    if (!level0 || !level1) return;
    for (int level = 0; level != 2; ++level) {
        std::string const suffix = std::to_string(level);
        UsdGenPlane const *center = Plane(c.out, ("clumpCenter_" + suffix).c_str());
        UsdGenPlane const *centerId = Plane(c.out, ("clumpCenterId_" + suffix).c_str());
        UsdGenPlane const *weight = nullptr;
        for (UsdGenPlane const &plane : c.out.extraCv)
            if (plane.name == TfToken("clumpWeight_" + suffix)) weight = &plane;
        Check(center && center->type == TfToken("float") && center->arity == 3 &&
              center->f.size() == size_t(c.out.totalCurves) * 3 &&
              centerId && centerId->type == TfToken("int") && centerId->arity == 2 &&
              centerId->i.size() == size_t(c.out.totalCurves) * 2 &&
              weight && weight->interpolation == TfToken("vertex") &&
              weight->f.size() == c.out.totalCvs,
              "Clump publishes a complete native motion quartet");
    }

    std::map<int, std::vector<size_t>> clumps;
    for (size_t s = 0; s < c.out.totalCurves; ++s)
        if (level0->i[s] >= 0) clumps[level0->i[s]].push_back(s);
    std::printf("  %zu map clumps over %u strands\n", clumps.size(), c.out.totalCurves);
    Check(clumps.size() >= 40 && clumps.size() <= 120, "the clump map's cells are the level-0 clumps");

    size_t converging = 0, measured = 0;
    for (auto const &clump : clumps) {
        if (clump.second.size() < 6) continue;
        usdGen::UsdGenVec3d rootCentre(0.0), tipCentre(0.0);
        for (size_t s : clump.second) {
            rootCentre += usdGen::UsdGenVec3d(Cv(c.out, s * c.cvs));
            tipCentre += usdGen::UsdGenVec3d(Cv(c.out, (s + 1) * c.cvs - 1));
        }
        rootCentre /= double(clump.second.size());
        tipCentre /= double(clump.second.size());
        double rootSpread = 0.0, tipSpread = 0.0;
        for (size_t s : clump.second) {
            usdGen::UsdGenVec3d r = usdGen::UsdGenVec3d(Cv(c.out, s * c.cvs)) - rootCentre;
            usdGen::UsdGenVec3d t = usdGen::UsdGenVec3d(Cv(c.out, (s + 1) * c.cvs - 1)) - tipCentre;
            r[1] = t[1] = 0.0;
            rootSpread += r.GetLength();
            tipSpread += t.GetLength();
        }
        ++measured;
        if (tipSpread < rootSpread) ++converging;
    }
    std::printf("  %zu of %zu clumps converge at the tips\n", converging, measured);
    Check(measured > 0 && converging > measured * 8 / 10, "clumped tips converge");

    // Level 1 nests in level 0.
    std::map<int, std::set<int>> parents;
    for (size_t s = 0; s < c.out.totalCurves; ++s)
        if (level1->i[s] >= 0) parents[level1->i[s]].insert(level0->i[s]);
    size_t nested = 0;
    for (auto const &entry : parents) nested += entry.second.size() == 1;
    Check(!parents.empty() && nested == parents.size(), "every sub-clump lies inside one map clump");
    Check(parents.size() > clumps.size(), "the second level splits the map clumps");

    // The envelope: mask 0 is the unclumped input.
    UsdAttribute mask = stage->GetAttributeAtPath(SdfPath("/World/Groom/Fur/Ops/clump.usdGen:mask"));
    Check(mask && mask.Set(0.0f), "author usdGen:mask = 0 on the clump");
    Cooked off = Cook(stage, kClumpDescription, "clump muted by its mask");
    if (off.ok && off.out.totalCvs == c.out.totalCvs) {
        for (UsdGenPlane const &plane : off.out.extraCv)
            if (plane.name == TfToken("clumpWeight_0") ||
                plane.name == TfToken("clumpWeight_1"))
                Check(std::all_of(plane.f.begin(), plane.f.end(),
                                  [](float w) { return w == 0.0f; }),
                      "mask zero publishes zero effective cohesion");
        size_t moved = 0;
        for (size_t i = 0; i < c.out.totalCvs; ++i)
            if (Cv(off.out, i) != Cv(c.out, i)) ++moved;
        Check(moved > c.out.totalCvs / 4, "mask 0 leaves the input unclumped");
    }
    mask.Set(1.0f);
    mask.Set(0.5f);
    Cooked half = Cook(stage, kClumpDescription, "clump with half mask");
    if (half.ok && half.out.totalCvs == c.out.totalCvs) {
        auto weightPlane = [](UsdGenCurveBuffer const &buffer, char const *name)
            -> UsdGenPlane const * {
            for (UsdGenPlane const &plane : buffer.extraCv)
                if (plane.name == TfToken(name)) return &plane;
            return nullptr;
        };
        auto const *full0 = weightPlane(c.out, "clumpWeight_0");
        auto const *full1 = weightPlane(c.out, "clumpWeight_1");
        auto const *half0 = weightPlane(half.out, "clumpWeight_0");
        auto const *half1 = weightPlane(half.out, "clumpWeight_1");
        bool matchesGeometryMask = full0 && full1 && half0 && half1;
        if (matchesGeometryMask)
            for (size_t cv = 0; cv < c.out.totalCvs; ++cv) {
                float const fullCohesion = 1.0f -
                    (1.0f - full0->f[cv]) * (1.0f - full1->f[cv]);
                float const halfCohesion = 1.0f -
                    (1.0f - half0->f[cv]) * (1.0f - half1->f[cv]);
                if (std::fabs(halfCohesion - 0.5f * fullCohesion) > 1e-5f) {
                    matchesGeometryMask = false;
                    break;
                }
            }
        Check(matchesGeometryMask,
              "two-level native cohesion applies a partial mask once");
    }
    mask.Set(1.0f);
    // Stray attenuation is part of the native motion weight, not merely the
    // positional Clump output. Every strand becomes a stray at rate one.
    SdfPath const clumpPath("/World/Groom/Fur/Ops/clump");
    UsdAttribute strayRate = stage->GetAttributeAtPath(
        clumpPath.AppendProperty(TfToken("usdGen:clump:stray:rate")));
    UsdAttribute strayAmount = stage->GetAttributeAtPath(
        clumpPath.AppendProperty(TfToken("usdGen:clump:stray:amount")));
    UsdAttribute strayFalloff = stage->GetAttributeAtPath(
        clumpPath.AppendProperty(TfToken("usdGen:clump:stray:falloff")));
    float oldRate = 0.0f, oldAmount = 0.0f, oldFalloff = 0.0f;
    bool const haveStray = strayRate && strayAmount && strayFalloff &&
        strayRate.Get(&oldRate) && strayAmount.Get(&oldAmount) &&
        strayFalloff.Get(&oldFalloff);
    Check(haveStray, "clump example authors stray controls");
    if (haveStray) {
        strayRate.Set(1.0f); strayAmount.Set(1.0f); strayFalloff.Set(0.0f);
        Cooked stray = Cook(stage, kClumpDescription, "all strands stray");
        if (stray.ok) {
            int checked = 0;
            for (UsdGenPlane const &plane : stray.out.extraCv) {
                if (plane.name != TfToken("clumpWeight_0") &&
                    plane.name != TfToken("clumpWeight_1")) continue;
                ++checked;
                Check(std::all_of(plane.f.begin(), plane.f.end(),
                                  [](float w) { return w == 0.0f; }),
                      "fully stray strands publish zero effective cohesion");
            }
            Check(checked == 2, "stray test observes both native clump levels");
        }
        strayRate.Set(oldRate); strayAmount.Set(oldAmount);
        strayFalloff.Set(oldFalloff);
    }
}

// --- dirty propagation through the scene index --------------------------------------

struct Scene {
    UsdStageRefPtr stage;
    UsdImagingSceneIndices indices;
    HdSceneIndexBaseRefPtr groom;
    UsdGenGroomSceneIndex *owner = nullptr;
};

Scene MakeScene(char const *file)
{
    Scene out;
    out.stage = UsdStage::Open(file, UsdStage::LoadAll);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = out.stage;
    out.indices = UsdImagingCreateSceneIndices(info);
    out.groom = UsdGenGroomSceneIndex::New(out.indices.finalSceneIndex);
    out.owner = dynamic_cast<UsdGenGroomSceneIndex *>(out.groom.operator->());
    return out;
}

/// Published points of every tile, concatenated in tile order.
std::vector<usdGen::UsdGenVec3f> PublishedPoints(Scene const &scene, char const *description)
{
    std::vector<usdGen::UsdGenVec3f> points;
    SdfPath const render = SdfPath(description).AppendChild(TfToken("__usdGenRender"));
    SdfPathVector tiles = scene.groom->GetChildPrimPaths(render);
    std::sort(tiles.begin(), tiles.end());
    for (SdfPath const &tile : tiles) {
        HdSceneIndexPrim prim = scene.groom->GetPrim(tile);
        auto sampled = HdSampledDataSource::Cast(HdContainerDataSource::Get(
            prim.dataSource, HdDataSourceLocator(TfToken("primvars"), TfToken("points"),
                                                 TfToken("primvarValue"))));
        if (!sampled) continue;
        VtValue const value = sampled->GetValue(0.0f);
        if (!value.IsHolding<VtVec3fArray>()) continue;
        for (usdGen::UsdGenVec3f const &p : ToEngine(value.UncheckedGet<VtVec3fArray>())) points.push_back(p);
    }
    return points;
}

bool Differs(std::vector<usdGen::UsdGenVec3f> const &a, std::vector<usdGen::UsdGenVec3f> const &b)
{
    if (a.size() != b.size()) return true;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return true;
    return false;
}

float MaxHeight(std::vector<usdGen::UsdGenVec3f> const &points)
{
    float h = 0.0f;
    for (usdGen::UsdGenVec3f const &p : points) h = std::max(h, p[1]);
    return h;
}

void CheckGuideDirtyPropagation()
{
    Scene scene = MakeScene(kGuideScene);
    Check(scene.owner != nullptr, "the guide example opens in a groom scene index");
    if (!scene.owner) return;
    scene.owner->Synchronize();
    std::vector<usdGen::UsdGenVec3f> const before = PublishedPoints(scene, kGuideDescription);
    Check(!before.empty(), "the guide example publishes strands through Hydra");
    if (before.empty()) return;

    // 1. Lengthen every guide: its prim is only named by usdGen:guides and by
    //    the expression's input:guideCurves.
    UsdGeomBasisCurves guides(scene.stage->GetPrimAtPath(SdfPath("/World/Guides")));
    VtIntArray counts;
    VtVec3fArray points;
    guides.GetCurveVertexCountsAttr().Get(&counts);
    guides.GetPointsAttr().Get(&points);
    size_t first = 0;
    for (int n : counts) {
        for (int k = 1; k < n; ++k)
            points[first + k] = points[first] + (points[first + k] - points[first]) * 1.5f;
        first += size_t(n);
    }
    Check(guides.GetPointsAttr().Set(points), "lengthen the guide curves on the stage");
    scene.indices.stageSceneIndex->ApplyPendingUpdates();
    scene.owner->Synchronize();
    std::vector<usdGen::UsdGenVec3f> const longer = PublishedPoints(scene, kGuideDescription);
    Check(Differs(before, longer), "a guide edit recooks the published strands");
    Check(MaxHeight(longer) > MaxHeight(before) * 1.2f, "the strands grow with their guides");

    // 2. Retarget input:guideCurves to a sparser set: the voronoi regions move.
    UsdGeomBasisCurves sparse = UsdGeomBasisCurves::Define(scene.stage, SdfPath("/World/SparseRoots"));
    VtIntArray sparseCounts{1, 1, 1, 1};
    VtVec3fArray sparsePoints{GfVec3f(-0.5f, 0, -0.5f), GfVec3f(0.5f, 0, -0.5f),
                              GfVec3f(-0.5f, 0, 0.5f), GfVec3f(0.5f, 0, 0.5f)};
    sparse.GetCurveVertexCountsAttr().Set(sparseCounts);
    sparse.GetPointsAttr().Set(sparsePoints);
    sparse.GetTypeAttr().Set(TfToken("linear"));
    UsdRelationship input = scene.stage->GetPrimAtPath(
        SdfPath("/World/Groom/Hair/Expressions/voronoiRegion")).GetRelationship(TfToken("input:guideCurves"));
    Check(input && input.SetTargets({SdfPath("/World/SparseRoots")}),
          "retarget input:guideCurves on the expression prim");
    scene.indices.stageSceneIndex->ApplyPendingUpdates();
    scene.owner->Synchronize();
    std::vector<usdGen::UsdGenVec3f> const retargeted = PublishedPoints(scene, kGuideDescription);
    Check(Differs(longer, retargeted), "retargeting an expression input recooks the strands");

    // 3. Edit the new target: the dependency follows the relationship.
    sparsePoints[0] = GfVec3f(0.9f, 0, 0.9f);
    sparse.GetPointsAttr().Set(sparsePoints);
    scene.indices.stageSceneIndex->ApplyPendingUpdates();
    scene.owner->Synchronize();
    std::vector<usdGen::UsdGenVec3f> const moved = PublishedPoints(scene, kGuideDescription);
    Check(Differs(retargeted, moved), "editing the newly targeted geometry recooks the strands");
}

void CheckClumpMapDirtyPropagation()
{
    Scene scene = MakeScene(kClumpScene);
    if (!scene.owner) { Check(false, "the clump example opens in a groom scene index"); return; }
    scene.owner->Synchronize();
    std::vector<usdGen::UsdGenVec3f> const before = PublishedPoints(scene, kClumpDescription);
    Check(!before.empty(), "the clump example publishes strands through Hydra");
    UsdAttribute file = scene.stage->GetAttributeAtPath(
        SdfPath("/World/Groom/Fur/Maps/clumpRegions.usdGen:map:file"));
    Check(file && file.Set(SdfAssetPath("./maps/clump_tightness.ptx")),
          "point the clump region map at another file");
    scene.indices.stageSceneIndex->ApplyPendingUpdates();
    scene.owner->Synchronize();
    std::vector<usdGen::UsdGenVec3f> const after = PublishedPoints(scene, kClumpDescription);
    Check(Differs(before, after), "a map prim edit recooks the clumps");
}

} // namespace

int main()
{
    usdGenTest::EnsureHydraPublishers();
    {
        UsdStageRefPtr const guideStage = UsdStage::Open(kGuideScene);
        if (!guideStage) { std::printf("FAIL: cannot open %s\n", kGuideScene); return 1; }
        CheckGuideVoronoi(guideStage);
        CheckGuidePtex(guideStage);
        CheckGuideSmooth(guideStage);
    }
    {
        UsdStageRefPtr const clumpStage = UsdStage::Open(kClumpScene);
        if (!clumpStage) { std::printf("FAIL: cannot open %s\n", kClumpScene); return 1; }
        CheckClump(clumpStage);
    }
    CheckGuideDirtyPropagation();
    CheckClumpMapDirtyPropagation();
    UsdGenGroomSceneIndex::DrainRetired();
    std::printf("testUsdGenGroomExamples: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
