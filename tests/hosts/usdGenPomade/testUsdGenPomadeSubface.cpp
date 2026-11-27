// Exact sub-face region guide generation: closed graph regions can be much
// smaller than one scalp quad, so neither K3's face centroid nor a face-list
// root sampler is allowed to choose the guide support.

#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadeScalp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, char const *what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    } else {
        std::printf("ok:   %s\n", what);
    }
}

struct OneQuad {
    std::vector<float> points = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<int> counts = {4};
    std::vector<int> indices = {0, 1, 2, 3};
};

usdGenPomade::PomadeHit Hit(usdGenPomade::PomadeScalpMesh const &mesh, float x,
                          float z)
{
    using namespace usdGenPomade;
    PomadeHit hit;
    hit.hit = PomadeFacePosition(mesh, 0, x, z, &hit.px, &hit.py, &hit.pz);
    hit.faceId = hit.hit ? 0 : -1;
    hit.ptexFaceId = hit.faceId;
    hit.u = x;
    hit.v = z;
    hit.ny = 1.0f;
    return hit;
}

bool InPolygon(float x, float z, std::vector<float> const &polygon)
{
    bool inside = false;
    size_t const count = polygon.size() / 2;
    for (size_t i = 0, j = count - 1; i < count; j = i++) {
        float const xi = polygon[i * 2], zi = polygon[i * 2 + 1];
        float const xj = polygon[j * 2], zj = polygon[j * 2 + 1];
        float const ex = xi - xj, ez = zi - zj;
        float const len2 = ex * ex + ez * ez;
        if (len2 > 0.0f) {
            float t = ((x - xj) * ex + (z - zj) * ez) / len2;
            t = std::min(std::max(t, 0.0f), 1.0f);
            float const dx = xj + t * ex - x;
            float const dz = zj + t * ez - z;
            if (dx * dx + dz * dz <= 1e-10f) {
                return true;
            }
        }
        if ((zi > z) != (zj > z)) {
            float const crossing = xj + (z - zj) * (xi - xj) / (zi - zj);
            if (x <= crossing) {
                inside = !inside;
            }
        }
    }
    return inside;
}

bool AddLoop(usdGenPomade::PomadeModel *model,
             usdGenPomade::PomadeScalpMesh const &mesh,
             std::vector<float> const &polygon)
{
    std::vector<int> nodes;
    for (size_t i = 0; i < polygon.size(); i += 2) {
        int const node = model->GraphAddNode(Hit(mesh, polygon[i],
                                                  polygon[i + 1]));
        if (node < 0) {
            return false;
        }
        nodes.push_back(node);
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (model->GraphConnect(nodes[i], nodes[(i + 1) % nodes.size()]) < 0) {
            return false;
        }
    }
    return true;
}

std::vector<usdGenPomade::PomadeGuideRoot> RootsForTube(
    usdGenPomade::PomadeModel const &model, int tubeId)
{
    std::vector<usdGenPomade::PomadeGuideRoot> out;
    auto const &guides = model.GetGuides();
    auto const &roots = model.GetRoots();
    if (guides.tubeIds.size() != roots.size()) {
        return out;
    }
    for (size_t i = 0; i < roots.size(); ++i) {
        if (guides.tubeIds[i] == tubeId) {
            out.push_back(roots[i]);
        }
    }
    return out;
}

std::vector<float> GuidePointsForTube(usdGenPomade::PomadeModel const &model,
                                      int tubeId)
{
    std::vector<float> out;
    auto const &guides = model.GetGuides();
    size_t cvOffset = 0;
    for (size_t i = 0; i < guides.counts.size(); ++i) {
        size_t const span = size_t(guides.counts[i]) * 3;
        if (cvOffset * 3 + span > guides.points.size() ||
            i >= guides.tubeIds.size()) {
            return {};
        }
        if (guides.tubeIds[i] == tubeId) {
            out.insert(out.end(), guides.points.begin() + cvOffset * 3,
                       guides.points.begin() + cvOffset * 3 + span);
        }
        cvOffset += size_t(guides.counts[i]);
    }
    return out;
}

bool SameRoots(std::vector<usdGenPomade::PomadeGuideRoot> const &a,
               std::vector<usdGenPomade::PomadeGuideRoot> const &b,
               size_t count)
{
    if (a.size() < count || b.size() < count) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (a[i].faceId != b[i].faceId || a[i].u != b[i].u ||
            a[i].v != b[i].v || a[i].px != b[i].px ||
            a[i].py != b[i].py || a[i].pz != b[i].pz) {
            return false;
        }
    }
    return true;
}

bool DistinctPositions(std::vector<usdGenPomade::PomadeGuideRoot> const &roots)
{
    for (size_t i = 0; i < roots.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            float const dx = roots[i].px - roots[j].px;
            float const dy = roots[i].py - roots[j].py;
            float const dz = roots[i].pz - roots[j].pz;
            if (dx * dx + dy * dy + dz * dz < 1e-12f) {
                return false;
            }
        }
    }
    return true;
}

bool FinalRootsInSupport(usdGenPomade::PomadeModel const &model, int tubeA,
                         std::vector<float> const &polyA, int tubeB,
                         std::vector<float> const &polyB)
{
    auto const &guides = model.GetGuides();
    auto const &roots = model.GetRoots();
    if (guides.guideCount <= 0 || guides.counts.size() != roots.size() ||
        guides.tubeIds.size() != roots.size() || guides.points.empty()) {
        return false;
    }
    size_t cvOffset = 0;
    bool sawA = false, sawB = false;
    for (size_t i = 0; i < roots.size(); ++i) {
        int const count = guides.counts[i];
        if (count <= 0 || (cvOffset + size_t(count)) * 3 >
                              guides.points.size()) {
            return false;
        }
        float const x = guides.points[cvOffset * 3];
        float const z = guides.points[cvOffset * 3 + 2];
        // Exact-region fill anchors the final, resampled guide to its scalp
        // root. Check the final product rather than only the root cache.
        if (x != roots[i].px || z != roots[i].pz) {
            return false;
        }
        if (guides.tubeIds[i] == tubeA) {
            sawA = true;
            if (!InPolygon(x, z, polyA)) {
                return false;
            }
        } else if (guides.tubeIds[i] == tubeB) {
            sawB = true;
            if (!InPolygon(x, z, polyB)) {
                return false;
            }
        } else {
            return false;
        }
        cvOffset += size_t(count);
    }
    return sawA && (tubeB < 0 || sawB) &&
           cvOffset * 3 == guides.points.size();
}

bool FinalRootsInPartitionedSupport(
    usdGenPomade::PomadeModel const &model, std::vector<int> const &leftTubes,
    std::vector<float> const &left, int rightTube,
    std::vector<float> const &right)
{
    auto const &guides = model.GetGuides();
    auto const &roots = model.GetRoots();
    if (guides.counts.size() != roots.size() ||
        guides.tubeIds.size() != roots.size()) {
        return false;
    }
    size_t cvOffset = 0;
    for (size_t i = 0; i < roots.size(); ++i) {
        int const count = guides.counts[i];
        if (count <= 0 || (cvOffset + size_t(count)) * 3 >
                              guides.points.size()) {
            return false;
        }
        float const x = guides.points[cvOffset * 3];
        float const z = guides.points[cvOffset * 3 + 2];
        if (x != roots[i].px || z != roots[i].pz) {
            return false;
        }
        bool const isLeft = std::find(leftTubes.begin(), leftTubes.end(),
                                      guides.tubeIds[i]) != leftTubes.end();
        if (isLeft ? !InPolygon(x, z, left)
                   : guides.tubeIds[i] != rightTube ||
                         !InPolygon(x, z, right)) {
            return false;
        }
        cvOffset += size_t(count);
    }
    return cvOffset * 3 == guides.points.size();
}

void CheckTwoSubfaceRegionsAndFreeze()
{
    using namespace usdGenPomade;
    OneQuad const quad;
    std::vector<float> const left = {0.08f, 0.08f, 0.30f, 0.08f,
                                     0.30f, 0.30f, 0.08f, 0.30f};
    std::vector<float> const right = {0.70f, 0.70f, 0.92f, 0.70f,
                                      0.92f, 0.92f, 0.70f, 0.92f};
    PomadeModel model;
    Check(model.BindScalp(quad.points, quad.counts, quad.indices),
          "subface: one quad binds");
    PomadeScalpMesh const &scalp = *model.GetScalp();
    Check(AddLoop(&model, scalp, left) && AddLoop(&model, scalp, right) &&
              model.GetGraph().RegionCount() == 2 && model.Rasterise(),
          "subface: two disjoint small regions rasterise");
    Check(model.RegionAtSurface(0, 0.5f, 0.5f) == -1,
          "subface: neither region contains the coarse-face centroid");
    Check(model.BuildTubeFromRegion(0, 4, 8, 2.0f) &&
              model.BuildTubeFromRegion(1, 4, 8, 2.0f),
          "subface: auto tubes build without a centroid-owned face");
    int const tubeA = model.TubeForRegion(0);
    int const tubeB = model.TubeForRegion(1);
    PomadeModel::FillParams fill;
    fill.density = 2000.0f;  // tiny 0.0484 patches, yet many root samples
    fill.cvCount = 5;
    fill.seed = 37;
    Check(tubeA >= 0 && tubeB >= 0 && tubeA != tubeB &&
              model.SetTubeFillParams(tubeA, fill) &&
              model.SetTubeFillParams(tubeB, fill) && model.RefillGuides(1.0f),
          "subface: high-density exact-region fill runs");
    Check(FinalRootsInSupport(model, tubeA, left, tubeB, right),
          "subface: final guide roots stay in their distinct polygons");

    // The commit/hydrate worker derives the same guides from snapshots. The
    // pure exact-region generator must match the live tube-0 block bitwise.
    PomadeScalpGraph graph;
    PomadeRegionLoops loops;
    std::string err;
    PomadeModel::GraphSnapshot const graphSnapshot = model.SnapshotGraph();
    bool const restored =
        graph.Restore(*model.GetScalp(), graphSnapshot.nodes,
                      graphSnapshot.edges, graphSnapshot.linked, &err) &&
        PomadeFlattenLoops(graph, &loops, &err);
    PomadeGuideSet const committed =
        restored ? PomadeGenerateGuidesOnRegionForTube(
                       model.Snapshot(), *model.GetScalp(), loops, 0, tubeA)
                 : PomadeGuideSet();
    Check(tubeA == 0 && restored &&
              committed.points == GuidePointsForTube(model, tubeA),
          "subface: live and snapshot-generated guide blocks agree");

    std::vector<PomadeGuideRoot> const initial = RootsForTube(model, tubeA);
    model.SetFreezeRoots(true);
    fill.density = 4000.0f;
    Check(model.SetTubeFillParams(tubeA, fill) && model.RefillGuides(1.0f),
          "subface: frozen high-density growth runs");
    std::vector<PomadeGuideRoot> const grown = RootsForTube(model, tubeA);
    Check(grown.size() > initial.size() &&
              SameRoots(initial, grown, initial.size()) &&
              DistinctPositions(grown),
          "subface: frozen growth preserves its prefix without duplicates");
    fill.density = 1000.0f;
    Check(model.SetTubeFillParams(tubeA, fill) && model.RefillGuides(1.0f),
          "subface: frozen shrink runs");
    std::vector<PomadeGuideRoot> const shrunk = RootsForTube(model, tubeA);
    Check(!shrunk.empty() && shrunk.size() < grown.size() &&
              SameRoots(initial, shrunk, shrunk.size()) &&
              DistinctPositions(shrunk),
          "subface: frozen shrink keeps the deterministic prefix");
    fill.density = 2000.0f;
    Check(model.SetTubeFillParams(tubeA, fill) && model.RefillGuides(1.0f),
          "subface: frozen density restoration runs");
    Check(SameRoots(initial, RootsForTube(model, tubeA), initial.size()),
          "subface: frozen refill restores the original deterministic prefix");

    model.SetFreezeRoots(false);
    std::vector<int> children;
    Check(model.SubdivideTube(tubeA, 2, "kmeans", 9, &children) &&
              children.size() == 2 && model.RefillGuides(1.0f),
          "subface: first descendant partition refills");
    auto countGuides = [&](int tube) {
        int count = 0;
        for (int id : model.GetGuides().tubeIds) {
            count += id == tube ? 1 : 0;
        }
        return count;
    };
    int const firstChildCount = countGuides(children[0]);
    int const secondChildCount = countGuides(children[1]);
    bool noParent = countGuides(tubeA) == 0;
    model.SetFreezeRoots(true);
    std::vector<PomadeGuideRoot> const childFrozenRoots = model.GetRoots();
    std::vector<float> const childFrozenPoints = model.GetGuides().points;
    int const childFrozenCount = model.GetGuides().guideCount;
    Check(model.RefillGuides(1.0f) &&
              model.GetGuides().guideCount == childFrozenCount &&
              model.GetRoots().size() == childFrozenRoots.size() &&
              SameRoots(childFrozenRoots, model.GetRoots(),
                        childFrozenRoots.size()) &&
              model.GetGuides().points == childFrozenPoints &&
              FinalRootsInPartitionedSupport(model, children, left, tubeB,
                                             right),
          "subface: frozen children refill without count inflation");
    model.SetFreezeRoots(false);
    std::vector<int> grandChildren;
    Check(noParent && firstChildCount > 0 && secondChildCount > 0 &&
              model.SubdivideTube(children[0], 2, "kmeans", 13,
                                  &grandChildren) &&
              grandChildren.size() == 2 && model.RefillGuides(1.0f),
          "subface: nested descendants replace their parents");
    bool noAncestorGuides = countGuides(tubeA) == 0 &&
                           countGuides(children[0]) == 0 &&
                           countGuides(children[1]) > 0 &&
                           countGuides(grandChildren[0]) > 0 &&
                           countGuides(grandChildren[1]) > 0;
    int owned = 0;
    bool inside = true;
    for (size_t i = 0; i < model.GetRoots().size(); ++i) {
        int const id = model.GetGuides().tubeIds[i];
        if (id != tubeB) {
            ++owned;
            inside = inside && InPolygon(model.GetRoots()[i].px,
                                         model.GetRoots()[i].pz, left);
        }
    }
    Check(noAncestorGuides && inside && owned > 0 &&
              owned < int(initial.size()) * 2,
          "subface: descendants own one non-multiplied partition of roots");
}

void CheckConcaveRegion()
{
    using namespace usdGenPomade;
    OneQuad const quad;
    // An L-shaped region: its bounding box includes the missing top-right
    // square, which catches any implementation that samples only the box.
    std::vector<float> const concave = {0.06f, 0.06f, 0.46f, 0.06f,
                                        0.46f, 0.22f, 0.22f, 0.22f,
                                        0.22f, 0.46f, 0.06f, 0.46f};
    PomadeModel model;
    Check(model.BindScalp(quad.points, quad.counts, quad.indices),
          "concave: one quad binds");
    Check(AddLoop(&model, *model.GetScalp(), concave) && model.Rasterise() &&
              model.BuildTubeFromRegion(0, 4, 8, 2.0f),
          "concave: auto tube builds from a concave subface region");
    int const tube = model.TubeForRegion(0);
    PomadeModel::FillParams fill;
    fill.density = 1500.0f;
    fill.cvCount = 5;
    fill.seed = 91;
    Check(tube >= 0 && model.SetTubeFillParams(tube, fill) &&
              model.RefillGuides(1.0f),
          "concave: high-density fill runs");
    Check(FinalRootsInSupport(model, tube, concave, -1, {}),
          "concave: every final guide root is inside the L-shaped polygon");
}

}  // namespace

int main()
{
    CheckTwoSubfaceRegionsAndFreeze();
    CheckConcaveRegion();
    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
