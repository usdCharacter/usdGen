// testUsdGenPomadeGraph — T0: K1/K2/K3 CPU twins, G1 graph ops, the model
// graph layer and the P2 C ABI (plan/17 P2 exit).
//
// No Hydra, no stage, no files: the fixture is a 4x4 quad grid in code.
// Proven here:
//   * K1: raycast hits/misses/degenerate input, closest point, face
//     coordinate round-trip, LBVH shape (2F-1 nodes, full order), refit;
//   * K2: the trace rides the surface with pinned ends and bounded samples;
//   * G1: weld merges ids and re-links edges (duplicates collapse),
//     welded graphs are watertight, unweld/reweld round-trips, connect
//     splits a region, delete merges, split/snap/mirror behave, strokes
//     close into regions and weld their ends, linked regions share one
//     interpolation id;
//   * K3: per-face rasterise (4+4 claimed, 8 uncovered), the intersection
//     map on overlapping loops, per-texel classification matching per-face
//     ids on uniform faces, the resolution rule (uniform -> 1x1);
//   * the model: graph edits bump the version AND the map version while
//     tube edits bump only the version (no bake on sculpt drags);
//   * the C ABI drives the same graph (plus its error paths);
//   * face GeomSubset scalps (plan/02 §2.20): the LBVH, raycast, closest
//     point, K3 raster, region picking and ABI placement skip the faces
//     a subset leaves out while face ids stay parent-mesh ids, and an
//     empty or out-of-range subset is an error;
//   * TN-6 parity (CUDA builds with a device only): K1/K2/K3 device lanes
//     match their CPU twins (K1 over a face subset too).

#include "usdGenPomade/pomadeApi.h"
#include "usdGenPomade/pomadeGraph.h"
#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadeRegion.h"
#include "usdGenPomade/pomadeScalp.h"

#ifdef USDGEN_POMADE_HAS_CUDA
#include "usdGenPomade/pomadeKernels.h"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
    std::fflush(stdout);
}

bool Near(float a, float b, float eps = 1e-4f)
{
    return std::fabs(a - b) <= eps;
}

// The 4x4 quad grid: face f = ix*4 + iz covers x in [x0+ix, x0+ix+1],
// z in [iz, iz+1]; position (x, z) is (u = z-iz, v = x-x0-ix).
struct Grid {
    usdGenPomade::PomadeScalpMesh mesh;
    usdGenPomade::PomadeScalpBvh bvh;
    float x0 = 0.0f;
};

Grid MakeGrid(float x0 = 0.0f)
{
    Grid grid;
    grid.x0 = x0;
    for (int ix = 0; ix <= 4; ++ix) {
        for (int iz = 0; iz <= 4; ++iz) {
            grid.mesh.points.push_back(x0 + float(ix));
            grid.mesh.points.push_back(0.0f);
            grid.mesh.points.push_back(float(iz));
        }
    }
    auto pid = [](int ix, int iz) { return ix * 5 + iz; };
    for (int ix = 0; ix < 4; ++ix) {
        for (int iz = 0; iz < 4; ++iz) {
            grid.mesh.faceVertexCounts.push_back(4);
            grid.mesh.faceVertexIndices.push_back(pid(ix, iz));
            grid.mesh.faceVertexIndices.push_back(pid(ix, iz + 1));
            grid.mesh.faceVertexIndices.push_back(pid(ix + 1, iz + 1));
            grid.mesh.faceVertexIndices.push_back(pid(ix + 1, iz));
        }
    }
    std::string err;
    if (!usdGenPomade::PomadeScalpFinalize(&grid.mesh, &err)) {
        std::printf("FAIL: fixture finalize: %s\n", err.c_str());
        ++g_failures;
        return grid;
    }
    if (!usdGenPomade::PomadeScalpBvhBuild(grid.mesh, &grid.bvh, &err)) {
        std::printf("FAIL: fixture bvh: %s\n", err.c_str());
        ++g_failures;
    }
    return grid;
}

// The same grid bound through a face GeomSubset (plan/02 §2.20): every face
// but 5 (x and z in [1, 2]), so a region drawn around the hole covers a face
// the subset leaves out. Authored unsorted and with a repeat, as a union of
// two subsets would produce.
std::vector<int> const kSubsetFaces = {15, 0, 1, 2, 3, 4, 6, 7,
                                       8, 9, 10, 11, 12, 13, 14, 0};
int const kHoleFace = 5;

Grid MakeSubsetGrid()
{
    Grid grid = MakeGrid();
    grid.mesh.activeFaces = kSubsetFaces;
    std::string err;
    if (!usdGenPomade::PomadeScalpFinalize(&grid.mesh, &err) ||
        !usdGenPomade::PomadeScalpBvhBuild(grid.mesh, &grid.bvh, &err)) {
        std::printf("FAIL: subset fixture: %s\n", err.c_str());
        ++g_failures;
    }
    return grid;
}

usdGenPomade::PomadeHit Locate(Grid const &grid, float x, float z)
{
    using namespace usdGenPomade;
    int ix = std::min(std::max(int(std::floor(x - grid.x0)), 0), 3);
    int iz = std::min(std::max(int(std::floor(z)), 0), 3);
    PomadeHit hit;
    hit.hit = true;
    hit.faceId = ix * 4 + iz;
    hit.u = z - float(iz);
    hit.v = x - grid.x0 - float(ix);
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    if (!PomadeFacePosition(grid.mesh, hit.faceId, hit.u, hit.v, &px, &py,
                           &pz)) {
        hit.hit = false;
        return hit;
    }
    hit.px = px;
    hit.py = py;
    hit.pz = pz;
    hit.nx = 0.0f;
    hit.ny = 1.0f;
    hit.nz = 0.0f;
    return hit;
}

// Two adjoining rects by place+connect (the stroke flow, explicit): A is
// x in [x0, x0+2], z in [1, 3]; B is x in [x0+2, x0+4], z in [1, 3].
// Returns the 6 node ids (A corners, then B's two new corners).
std::vector<int> BuildAdjoining(Grid &grid, usdGenPomade::PomadeScalpGraph *graph)
{
    int a0 = graph->AddNode(Locate(grid, grid.x0 + 0.0f, 1.0f));
    int a1 = graph->AddNode(Locate(grid, grid.x0 + 2.0f, 1.0f));
    int a2 = graph->AddNode(Locate(grid, grid.x0 + 2.0f, 3.0f));
    int a3 = graph->AddNode(Locate(grid, grid.x0 + 0.0f, 3.0f));
    graph->Connect(grid.mesh, a0, a1);
    graph->Connect(grid.mesh, a1, a2);
    graph->Connect(grid.mesh, a2, a3);
    graph->Connect(grid.mesh, a3, a0);
    int b1 = graph->AddNode(Locate(grid, grid.x0 + 4.0f, 1.0f));
    int b2 = graph->AddNode(Locate(grid, grid.x0 + 4.0f, 3.0f));
    graph->Connect(grid.mesh, a1, b1);
    graph->Connect(grid.mesh, b1, b2);
    graph->Connect(grid.mesh, b2, a2);
    // (a2 -> a1) closes over A's edge: Connect refuses the duplicate.
    graph->Connect(grid.mesh, a2, a1);
    return {a0, a1, a2, a3, b1, b2};
}

void RayTo(float const origin[3], float tx, float ty, float tz,
           float outDir[3])
{
    outDir[0] = tx - origin[0];
    outDir[1] = ty - origin[1];
    outDir[2] = tz - origin[2];
    float const length = std::sqrt(outDir[0] * outDir[0] +
                                   outDir[1] * outDir[1] +
                                   outDir[2] * outDir[2]);
    if (length > 0.0f) {
        outDir[0] /= length;
        outDir[1] /= length;
        outDir[2] /= length;
    }
}

void CheckK1(Grid &grid)
{
    using namespace usdGenPomade;
    Check(grid.bvh.valid && grid.bvh.nodes.size() == 2 * 16 - 1,
          "K1: the LBVH holds 2F-1 nodes");
    std::vector<char> seen(16, 0);
    for (int f : grid.bvh.order) {
        if (f >= 0 && f < 16) {
            seen[size_t(f)] = 1;
        }
    }
    bool full = true;
    for (char c : seen) {
        full = full && c;
    }
    Check(full && grid.bvh.order.size() == 16,
          "K1: the Morton order covers every face");
    float const origin[3] = {1.5f, 5.0f, 0.5f};
    float const down[3] = {0.0f, -1.0f, 0.0f};
    PomadeHit const hit = PomadeRaycastCpu(grid.mesh, grid.bvh, origin, down);
    Check(hit.hit && hit.faceId == 4 && Near(hit.u, 0.5f) &&
              Near(hit.v, 0.5f) && Near(hit.px, 1.5f) && Near(hit.py, 0.0f) &&
              Near(hit.pz, 0.5f) && Near(hit.ny, 1.0f),
          "K1: the raycast hits the face under an interior point");

    // A real StageView pixel can land exactly on a quad fan diagonal.  The
    // pre-fix float Moller-Trumbore tests rejected both triangles for this
    // captured Qt ray (one shared barycentric was -4.7e-8, the other -4.4e-8).
    Grid const seam = MakeGrid(-1.0f);
    float const seamOrigin[3] = {1.992840022f, 13.90000005f, 1.992939308f};
    float const seamDir[3] = {-0.071235519f, -0.994912051f, -0.071242625f};
    PomadeHit const seamHit = PomadeRaycastCpu(seam.mesh, seam.bvh, seamOrigin,
                                              seamDir);
    Check(seamHit.hit && seamHit.faceId == 4 && Near(seamHit.px, 0.9976f,
                                                       2e-4f) &&
              Near(seamHit.pz, 0.9976f, 2e-4f),
          "K1: the captured StageView fan-diagonal ray cannot crack a quad");
    bool seamSweep = true;
    for (int step = -4; step <= 4; ++step) {
        float const offset = float(step) * 2e-5f;
        float sweepDir[3] = {0.0f, 0.0f, 0.0f};
        RayTo(seamOrigin, 0.9976f + offset, 0.0f, 0.9976f - offset,
              sweepDir);
        seamSweep = seamSweep && PomadeRaycastCpu(seam.mesh, seam.bvh,
                                                  seamOrigin, sweepDir).hit;
    }
    Check(seamSweep, "K1: a sweep across both sides of a fan diagonal stays watertight");
    float outsideDir[3] = {0.0f, 0.0f, 0.0f};
    RayTo(seamOrigin, -1.0002f, 0.0f, 0.5f, outsideDir);
    Check(!PomadeRaycastCpu(seam.mesh, seam.bvh, seamOrigin, outsideDir).hit,
          "K1: the watertight diagonal rule does not grow the outer silhouette");
    float const nonUnitDir[3] = {seamDir[0] * 7.0f, seamDir[1] * 7.0f,
                                  seamDir[2] * 7.0f};
    float const awayDir[3] = {-seamDir[0], -seamDir[1], -seamDir[2]};
    float const belowOrigin[3] = {seamHit.px, -1.0f, seamHit.pz};
    float const upThroughSurface[3] = {0.0f, 1.0f, 0.0f};
    Check(PomadeRaycastCpu(seam.mesh, seam.bvh, seamOrigin, nonUnitDir).hit &&
              !PomadeRaycastCpu(seam.mesh, seam.bvh, seamOrigin, awayDir).hit &&
              PomadeRaycastCpu(seam.mesh, seam.bvh, belowOrigin,
                              upThroughSurface).hit,
          "K1: seam handling retains nonunit normalization and two-sided hits");
    float const up[3] = {0.0f, 1.0f, 0.0f};
    Check(!PomadeRaycastCpu(grid.mesh, grid.bvh, origin, up).hit,
          "K1: an upward ray misses");
    float const zero[3] = {0.0f, 0.0f, 0.0f};
    Check(!PomadeRaycastCpu(grid.mesh, grid.bvh, origin, zero).hit,
          "K1: a degenerate direction misses, never NaNs");
    float const above[3] = {2.5f, 1.0f, 2.5f};
    PomadeHit const near = PomadeClosestPointCpu(grid.mesh, above);
    Check(near.hit && near.faceId == 2 * 4 + 2 && Near(near.px, 2.5f) &&
              Near(near.py, 0.0f) && Near(near.pz, 2.5f),
          "K1: the closest point projects onto the grid");
    // A far-field point in a vertex Voronoi region (the Ericson vertex-C
    // regression: the closest face must win, not the first fan triangle).
    float const corner[3] = {-3.0f, 2.0f, -3.0f};
    PomadeHit const vc = PomadeClosestPointCpu(grid.mesh, corner);
    Check(vc.hit && vc.faceId == 0 && Near(vc.px, 0.0f) && Near(vc.py, 0.0f) &&
              Near(vc.pz, 0.0f),
          "K1: far-field closest points land on the corner vertex");
    // Face coordinate round-trip on every face.
    bool roundTrip = true;
    for (int f = 0; f < 16 && roundTrip; ++f) {
        for (auto uv : {std::make_pair(0.25f, 0.25f),
                        std::make_pair(0.7f, 0.3f)}) {
            float px = 0.0f, py = 0.0f, pz = 0.0f;
            float u = 0.0f, v = 0.0f;
            int ptex = -1;
            roundTrip =
                PomadeFacePosition(grid.mesh, f, uv.first, uv.second, &px,
                                  &py, &pz) &&
                PomadeFaceCoordinate(grid.mesh, f, px, py, pz, &u, &v,
                                    &ptex) &&
                Near(u, uv.first, 1e-3f) && Near(v, uv.second, 1e-3f) &&
                ptex == f;
            if (!roundTrip) {
                break;
            }
        }
    }
    Check(roundTrip, "K1: face coordinates round-trip on every quad");
    // Refit after a deform keeps the BVH usable.
    grid.mesh.points[1] = 0.5f;
    Check(PomadeScalpBvhRefit(grid.mesh, &grid.bvh),
          "K1: refit accepts a deformed scalp");
    PomadeHit const hit2 = PomadeRaycastCpu(grid.mesh, grid.bvh, origin, down);
    Check(hit2.hit && hit2.faceId == 4, "K1: hits survive a refit");
    grid.mesh.points[1] = 0.0f;
    Check(PomadeScalpBvhRefit(grid.mesh, &grid.bvh), "K1: refit restores rest");
}

void CheckK2(Grid &grid)
{
    using namespace usdGenPomade;
    float const a[3] = {0.5f, 0.0f, 1.5f};
    float const b[3] = {3.5f, 0.0f, 1.5f};
    std::vector<float> const poly = PomadeTraceEdgeCpu(grid.mesh, a, b);
    bool onSurface = poly.size() >= 6;
    for (size_t i = 0; onSurface && i < poly.size() / 3; ++i) {
        onSurface = Near(poly[i * 3 + 1], 0.0f, 1e-5f);
    }
    Check(onSurface, "K2: the trace rides the surface");
    Check(poly.size() == 4 * 3, "K2: 3 units at h=1 trace 4 samples");
    Check(poly[0] == a[0] && poly[1] == a[1] && poly[2] == a[2] &&
              poly[9] == b[0] && poly[10] == b[1] && poly[11] == b[2],
          "K2: ends are pinned bit-exactly (shared borders join)");
}

void CheckGraphOps(Grid &grid)
{
    using namespace usdGenPomade;
    PomadeScalpGraph graph;
    std::vector<int> ids = BuildAdjoining(grid, &graph);
    Check(graph.NodeCount() == 6 && graph.EdgeCount() == 7 &&
              graph.RegionCount() == 2,
          "G1: two adjoining rects are 6 nodes, 7 edges, 2 regions");
    Check(graph.IsWatertight(), "G1: the welded graph is watertight");
    // The shared boundary appears in both loops (the §2.2 encoding).
    auto const &regions = graph.Regions();
    std::vector<int> shared;
    for (int nid : regions[0].loop) {
        if (std::find(regions[1].loop.begin(), regions[1].loop.end(), nid) !=
            regions[1].loop.end()) {
            shared.push_back(nid);
        }
    }
    Check(shared.size() == 2, "G1: the shared edge is in both loops");
    // Weld merges ids and re-links: a parallel edge collapses onto the
    // shared one, and the regions survive.
    int c0 = graph.AddNode(Locate(grid, grid.x0 + 2.0f, 1.0f));
    int c1 = graph.AddNode(Locate(grid, grid.x0 + 2.0f, 3.0f));
    int ce = graph.Connect(grid.mesh, c0, c1);
    Check(ce >= 0 && graph.EdgeCount() == 8, "G1: a parallel edge adds one");
    Check(graph.Weld(grid.mesh, ids[1], c0), "G1: weld takes the first end");
    Check(graph.Weld(grid.mesh, ids[2], c1), "G1: weld takes the second end");
    Check(graph.NodeCount() == 6 && graph.EdgeCount() == 7 &&
              graph.RegionCount() == 2 && graph.IsWatertight(),
          "G1: welding collapses the duplicate pair, regions survive");
    // Unweld/reweld round-trips: the shared corner splits per region, the
    // HUD sees the coincident set, and re-welding restores the partition.
    std::vector<int> copies = graph.Unweld(grid.mesh, ids[1]);
    Check(copies.size() == 1, "G1: a twice-shared node unwelds into two");
    Check(!graph.CoincidentSets(0.01f).empty(),
          "G1: coincident unwelded nodes are reported for the HUD");
    Check(graph.Weld(grid.mesh, ids[1], copies[0]),
          "G1: reweld merges the copies");
    Check(graph.NodeCount() == 6 && graph.RegionCount() == 2 &&
              graph.IsWatertight() && graph.CoincidentSets(0.01f).empty(),
          "G1: unweld/reweld round-trips to the watertight graph");
    // Connect across a region splits it; deleting the edge merges it back.
    PomadeScalpGraph single;
    int s0 = single.AddNode(Locate(grid, 0.0f, 1.0f));
    int s1 = single.AddNode(Locate(grid, 4.0f, 1.0f));
    int s2 = single.AddNode(Locate(grid, 4.0f, 3.0f));
    int s3 = single.AddNode(Locate(grid, 0.0f, 3.0f));
    single.Connect(grid.mesh, s0, s1);
    single.Connect(grid.mesh, s1, s2);
    single.Connect(grid.mesh, s2, s3);
    single.Connect(grid.mesh, s3, s0);
    Check(single.RegionCount() == 1, "G1: one rect is one region");
    int eTop = -1;
    for (auto const &e : single.Edges()) {
        if (e.alive &&
            ((e.a == s0 && e.b == s1) || (e.a == s1 && e.b == s0))) {
            eTop = e.id;
        }
    }
    int eBottom = -1;
    for (auto const &e : single.Edges()) {
        if (e.alive &&
            ((e.a == s2 && e.b == s3) || (e.a == s3 && e.b == s2))) {
            eBottom = e.id;
        }
    }
    int m0 = single.SplitEdge(grid.mesh, eTop, Locate(grid, 2.0f, 1.0f));
    int m1 = single.SplitEdge(grid.mesh, eBottom, Locate(grid, 2.0f, 3.0f));
    Check(m0 >= 0 && m1 >= 0, "G1: splitting edges adds mid nodes");
    int across = single.Connect(grid.mesh, m0, m1);
    Check(across >= 0 && single.RegionCount() == 2,
          "G1: connecting across splits the region in two");
    Check(single.DeleteEdge(grid.mesh, across) && single.RegionCount() == 1,
          "G1: deleting the edge merges the regions back");
    // Delete a valence-2 node re-links the loop; snap finds nodes/edges.
    Check(single.DeleteNode(grid.mesh, m0) && single.RegionCount() == 1,
          "G1: deleting a valence-2 node keeps the region");
    float const q[3] = {0.02f, 0.0f, 1.02f};
    Check(single.SnapNode(q, 0.1f) == s0, "G1: snap finds the corner node");
    Check(single.SnapNode(q, 0.001f) < 0, "G1: snap respects the radius");
    float const mid[3] = {2.0f, 0.0f, 1.0f};
    Check(single.SnapEdge(mid, 0.1f) >= 0, "G1: snap finds the split edge");
    // Link: two regions share one interpolation id; tubes stay distinct.
    PomadeScalpGraph linked;
    BuildAdjoining(grid, &linked);
    Check(linked.InterpId(0) == 0 && linked.InterpId(1) == 1,
          "G1: unlinked regions interpolate apart");
    Check(linked.LinkRegions(0, 1), "G1: regions link");
    Check(linked.InterpId(0) == linked.InterpId(1),
          "G1: linked regions share one interpolation id");
    Check(linked.Regions()[0].color[0] != linked.Regions()[1].color[0] ||
              linked.Regions()[0].color[1] != linked.Regions()[1].color[1],
          "G1: linked regions keep distinct colours (tubes stay distinct)");
    Check(linked.UnlinkRegions(0, 1) && linked.InterpId(1) == 1,
          "G1: unlink restores distinct ids");
    // Mirror-X twins nodes across x = 0 on a symmetric scalp.
    Grid sym = MakeGrid(-2.0f);
    PomadeScalpGraph mgraph;
    int n0 = mgraph.AddNode(Locate(sym, 1.0f, 1.0f));
    int n1 = mgraph.AddNode(Locate(sym, 1.0f, 2.0f));
    mgraph.Connect(sym.mesh, n0, n1);
    auto mapping = mgraph.MirrorX(sym.mesh);
    Check(mapping.size() == 2 && mgraph.NodeCount() == 4 &&
              mgraph.EdgeCount() == 2,
          "G1: mirror-X twins the nodes and the edge");
    bool mirrored = true;
    for (auto const &pr : mapping) {
        PomadeGraphNode const *a = mgraph.FindNode(pr.first);
        PomadeGraphNode const *b = mgraph.FindNode(pr.second);
        mirrored = mirrored && a && b && Near(b->p[0], -a->p[0], 1e-3f) &&
                   Near(b->p[2], a->p[2], 1e-3f);
    }
    Check(mirrored, "G1: twins sit at mirrored positions");
    // Strokes: a closed rect is one region; adjoining ends weld.
    PomadeScalpGraph strokes;
    auto rect = [&](float x0, float x1) {
        std::vector<PomadeHit> samples;
        float const corners[4][2] = {
            {x0, 1.0f}, {x1, 1.0f}, {x1, 3.0f}, {x0, 3.0f}};
        for (int k = 0; k < 4; ++k) {
            for (int i = 0; i < 5; ++i) {
                float const t = float(i) / 5.0f;
                samples.push_back(Locate(
                    grid, corners[k][0] +
                              (corners[(k + 1) % 4][0] - corners[k][0]) * t,
                    corners[k][1] +
                        (corners[(k + 1) % 4][1] - corners[k][1]) * t));
            }
        }
        samples.push_back(Locate(grid, x0, 1.0f));
        return samples;
    };
    PomadeStrokeResult rA =
        PomadeStrokeToChain(grid.mesh, &strokes, rect(0.0f, 2.0f), 0.1f, 0.05f);
    Check(rA.closedLoop && strokes.RegionCount() == 1,
          "G1: a closing stroke becomes a region immediately");
    PomadeStrokeResult rB =
        PomadeStrokeToChain(grid.mesh, &strokes, rect(2.0f, 4.0f), 0.1f, 0.05f);
    Check(rB.closedLoop && rB.weldedStart && rB.weldedEnd &&
              strokes.RegionCount() == 2 && strokes.IsWatertight(),
          "G1: the adjoining stroke welds both ends, watertight");
    Check(strokes.NodeCount() == 6 && strokes.EdgeCount() == 7,
          "G1: the adjoining stroke shares the boundary (6 nodes, 7 edges)");
    bool oneShared = false;
    if (strokes.RegionCount() == 2) {
        auto const &l0 = strokes.Regions()[0].loop;
        auto const &l1 = strokes.Regions()[1].loop;
        int shared = 0;
        for (int n : l0) {
            for (int m : l1) {
                shared += (n == m) ? 1 : 0;
            }
        }
        oneShared = (shared == 2);
    }
    Check(oneShared,
          "G1: the two regions share exactly two boundary nodes");
}

void CheckK3(Grid &grid)
{
    using namespace usdGenPomade;
    PomadeScalpGraph graph;
    BuildAdjoining(grid, &graph);
    PomadeRegionMaps maps;
    std::string err;
    Check(PomadeRasteriseRegionsCpu(grid.mesh, graph, &maps, &err),
          "K3: rasterise runs: " + err);
    // A claims ix in {0, 1} x iz in {1, 2} (faces 1, 2, 5, 6, 9, 10... in
    // f = ix*4 + iz: (0,1)=1, (0,2)=2, (1,1)=5, (1,2)=6); B the mirror.
    auto expect = [&](int ix, int iz) {
        if (iz < 1 || iz > 2) {
            return -1;
        }
        if (ix <= 1) {
            return 0;
        }
        return 1;
    };
    bool perFace = maps.faceRegion.size() == 16;
    for (int ix = 0; perFace && ix < 4; ++ix) {
        for (int iz = 0; perFace && iz < 4; ++iz) {
            perFace = maps.faceRegion[size_t(ix * 4 + iz)] ==
                      expect(ix, iz);
        }
    }
    Check(perFace, "K3: per-face ids match the two-rect partition");
    Check(maps.uncoveredCount == 8 && maps.intersectedCount == 0,
          "K3: 8 uncovered faces, no intersections");
    // Linked regions bake one id: the primvar folds the link.
    graph.LinkRegions(0, 1);
    Check(PomadeRasteriseRegionsCpu(grid.mesh, graph, &maps, &err),
          "K3: rasterise runs linked: " + err);
    bool linked = true;
    for (int id : maps.faceRegion) {
        linked = linked && (id < 0 || id == 0);
    }
    Check(linked && maps.intersectedCount == 0,
          "K3: linked regions share one baked id, never intersect");
    graph.UnlinkRegions(0, 1);
    // Overlapping loops intersect: B shifted over A claims faces twice.
    PomadeScalpGraph overlap;
    int o0 = overlap.AddNode(Locate(grid, 0.0f, 0.0f));
    int o1 = overlap.AddNode(Locate(grid, 2.0f, 0.0f));
    int o2 = overlap.AddNode(Locate(grid, 2.0f, 2.0f));
    int o3 = overlap.AddNode(Locate(grid, 0.0f, 2.0f));
    overlap.Connect(grid.mesh, o0, o1);
    overlap.Connect(grid.mesh, o1, o2);
    overlap.Connect(grid.mesh, o2, o3);
    overlap.Connect(grid.mesh, o3, o0);
    int p0 = overlap.AddNode(Locate(grid, 1.0f, 1.0f));
    int p1 = overlap.AddNode(Locate(grid, 3.0f, 1.0f));
    int p2 = overlap.AddNode(Locate(grid, 3.0f, 3.0f));
    int p3 = overlap.AddNode(Locate(grid, 1.0f, 3.0f));
    overlap.Connect(grid.mesh, p0, p1);
    overlap.Connect(grid.mesh, p1, p2);
    overlap.Connect(grid.mesh, p2, p3);
    overlap.Connect(grid.mesh, p3, p0);
    Check(overlap.RegionCount() == 2, "K3: overlapping loops are 2 regions");
    Check(PomadeRasteriseRegionsCpu(grid.mesh, overlap, &maps, &err),
          "K3: rasterise runs overlapped: " + err);
    Check(maps.intersectedCount > 0,
          "K3: the root-intersection map flags double-claimed faces");
    // Per-texel classification agrees with per-face ids on uniform faces.
    PomadeRegionLoops loops;
    Check(PomadeFlattenLoops(graph, &loops, &err),
          "K3: loops flatten: " + err);
    float cA[3] = {0.5f, 0.0f, 1.5f};
    float cB[3] = {3.5f, 0.0f, 1.5f};
    float cO[3] = {0.5f, 0.0f, 0.5f};
    Check(PomadeClassifyPointCpu(loops, cA) == 0 &&
              PomadeClassifyPointCpu(loops, cB) == 1 &&
              PomadeClassifyPointCpu(loops, cO) == -1,
          "K3: texel queries match the per-face partition");
    PomadeRasteriseRegionsCpu(grid.mesh, graph, &maps, &err);
    std::vector<int> const res =
        PomadeFaceResLog2(grid.mesh, maps, loops);
    Check(res[size_t(0 * 4 + 0)] == 6,
          "K3: a boundary-adjacent face keeps 64x64 detail");
    std::vector<int> const forced =
        PomadeFaceResLog2(grid.mesh, maps, loops, 2);
    bool all2 = forced.size() == 16;
    for (int r : forced) {
        all2 = all2 && r == 2;
    }
    Check(all2, "K3: the artist res override forces every face");

    // The adjoining two-by-two regions above have no face that is clear of
    // every boundary: their shared and outer boundaries touch every face.
    // A full-grid loop gives this four-by-four fixture a truly interior face
    // and proves the conservative boundary rule still keeps 1x1 collapse.
    PomadeScalpGraph fullGrid;
    int const f0 = fullGrid.AddNode(Locate(grid, 0.0f, 0.0f));
    int const f1 = fullGrid.AddNode(Locate(grid, 4.0f, 0.0f));
    int const f2 = fullGrid.AddNode(Locate(grid, 4.0f, 4.0f));
    int const f3 = fullGrid.AddNode(Locate(grid, 0.0f, 4.0f));
    fullGrid.Connect(grid.mesh, f0, f1);
    fullGrid.Connect(grid.mesh, f1, f2);
    fullGrid.Connect(grid.mesh, f2, f3);
    fullGrid.Connect(grid.mesh, f3, f0);
    PomadeRegionLoops fullLoops;
    PomadeRegionMaps fullMaps;
    bool const fullOk = PomadeFlattenLoops(fullGrid, &fullLoops, &err) &&
                        PomadeRasteriseRegionsCpu(grid.mesh, fullGrid,
                                                   &fullMaps, &err);
    std::vector<int> const fullRes =
        PomadeFaceResLog2(grid.mesh, fullMaps, fullLoops);
    Check(fullOk && fullRes.size() == 16 &&
              fullRes[size_t(1 * 4 + 1)] == 0,
          "K3: a truly interior face collapses to 1x1");
}

void CheckModel(Grid &grid)
{
    using namespace usdGenPomade;
    PomadeModel model;
    Check(model.BindScalp(grid.mesh.points, grid.mesh.faceVertexCounts,
                          grid.mesh.faceVertexIndices),
          "model: the scalp binds");
    Check(model.HasScalp(), "model: HasScalp reports the binding");
    uint64_t const vBind = model.GetVersion();
    uint64_t const mBind = model.GetMapVersion();
    Check(model.BuildTestTube(), "model: the test tube builds");
    Check(model.GetVersion() > vBind && model.GetMapVersion() == mBind,
          "model: tube edits bump the version, never the map version");
    int a0 = model.GraphAddNode(Locate(grid, 0.0f, 1.0f));
    Check(a0 >= 0 && model.GetMapVersion() > mBind,
          "model: graph edits bump the map version (bake key)");
    int a1 = model.GraphAddNode(Locate(grid, 2.0f, 1.0f));
    int a2 = model.GraphAddNode(Locate(grid, 2.0f, 3.0f));
    int a3 = model.GraphAddNode(Locate(grid, 0.0f, 3.0f));
    model.GraphConnect(a0, a1);
    model.GraphConnect(a1, a2);
    model.GraphConnect(a2, a3);
    model.GraphConnect(a3, a0);
    Check(model.Rasterise(), "model: K3 runs at gesture end");
    Check(model.GetRegionMaps().uncoveredCount == 12,
          "model: one rect leaves 12 faces uncovered");
    PomadeModel::GraphSnapshot snap = model.SnapshotGraph();
    Check(snap.nodes.size() == 4 && snap.edges.size() == 4 &&
              snap.regionLoops.size() == 1 && snap.faceRegions.size() == 16,
          "model: the snapshot carries nodes, edges, loops, primvar");
    // Restore round-trips the graph bit-exactly (the hydrate write path).
    PomadeModel other;
    other.BindScalp(grid.mesh.points, grid.mesh.faceVertexCounts,
                    grid.mesh.faceVertexIndices);
    Check(other.RestoreGraph(snap), "model: RestoreGraph accepts it");
    PomadeModel::GraphSnapshot check = other.SnapshotGraph();
    Check(check.regionLoops == snap.regionLoops &&
              check.faceRegions == snap.faceRegions,
          "model: restore round-trips loops + primvar exactly");
    // Mirror-X twins placements across x = 0.
    Grid sym = MakeGrid(-2.0f);
    PomadeModel mirror;
    mirror.BindScalp(sym.mesh.points, sym.mesh.faceVertexCounts,
                     sym.mesh.faceVertexIndices);
    mirror.SetMirrorX(true);
    int m0 = mirror.GraphAddNode(Locate(sym, 1.0f, 1.0f));
    Check(m0 >= 0 && mirror.SnapshotGraph().nodes.size() == 2,
          "model: mirror-X twins every placed node");
}

void CheckCApi(Grid &grid)
{
    PomadeModelContext *ctx = nullptr;
    Check(Pomade_Create(&ctx) == POMADE_OK && ctx, "C ABI: model creates");
    Check(Pomade_BindScalp(nullptr, nullptr, 0, nullptr, 0, nullptr, 0) ==
              POMADE_ERROR,
          "C ABI: bind rejects nulls");
    std::vector<float> const &pts = grid.mesh.points;
    std::vector<int> const &counts = grid.mesh.faceVertexCounts;
    std::vector<int> const &indices = grid.mesh.faceVertexIndices;
    Check(Pomade_BindScalp(ctx, pts.data(), int(pts.size()), counts.data(),
                          int(counts.size()), indices.data(),
                          int(indices.size())) == POMADE_OK,
          "C ABI: scalp binds");
    Check(Pomade_HasScalp(ctx) == 1, "C ABI: HasScalp reports it");
    // K1 through the ABI.
    float const origin[3] = {1.5f, 5.0f, 0.5f};
    float const down[3] = {0.0f, -1.0f, 0.0f};
    int hit = 0, face = -1;
    float uv[2] = {0.0f, 0.0f};
    float p[3] = {0.0f, 0.0f, 0.0f};
    float n[3] = {0.0f, 0.0f, 0.0f};
    Check(Pomade_Raycast(ctx, origin, down, &hit, &face, uv, p, n) ==
                  POMADE_OK &&
              hit == 1 && face == 4,
          "C ABI: raycast hits face 4");
    // One rect by place + connect.
    int ids[4] = {-1, -1, -1, -1};
    float const corners[4][2] = {{0.0f, 1.0f},
                                 {2.0f, 1.0f},
                                 {2.0f, 3.0f},
                                 {0.0f, 3.0f}};
    bool placed = true;
    for (int i = 0; i < 4; ++i) {
        usdGenPomade::PomadeHit h =
            Locate(grid, corners[i][0], corners[i][1]);
        placed = placed &&
                 Pomade_GraphAddNode(ctx, h.faceId, h.u, h.v, &ids[i]) ==
                     POMADE_OK;
    }
    Check(placed, "C ABI: four nodes place");
    int edge = -1;
    bool connected = true;
    for (int i = 0; i < 4; ++i) {
        connected = connected &&
                    Pomade_GraphConnect(ctx, ids[i], ids[(i + 1) % 4],
                                       &edge) == POMADE_OK;
    }
    Check(connected, "C ABI: the rect connects");
    Check(Pomade_GraphConnect(ctx, ids[0], ids[1], &edge) == POMADE_ERROR,
          "C ABI: duplicate connects fail honestly");
    Check(Pomade_GraphWeld(ctx, ids[0], 9999) == POMADE_ERROR,
          "C ABI: welds of dead nodes fail honestly");
    int nodes = 0, edges = 0, regions = 0;
    Check(Pomade_GetGraphCounts(ctx, &nodes, &edges, &regions) == POMADE_OK &&
              nodes == 4 && edges == 4 && regions == 1,
          "C ABI: census reads 4/4/1");
    Check(Pomade_Rasterise(ctx) == POMADE_OK, "C ABI: rasterise runs");
    int faceRegions[16] = {0};
    int got = 0;
    Check(Pomade_ReadFaceRegions(ctx, faceRegions, 16, &got) == POMADE_OK &&
              got == 16 && faceRegions[1] == 0 && faceRegions[0] == -1,
          "C ABI: the live map reads back per face");
    Check(Pomade_ReadFaceRegions(ctx, faceRegions, 4, &got) == POMADE_ERROR,
          "C ABI: short outputs fail honestly");
    // Link needs two regions: split this one first (tested above in G1, so
    // here only the ABI error path matters).
    Check(Pomade_GraphLinkRegions(ctx, 0, 7) == POMADE_ERROR,
          "C ABI: links of absent regions fail honestly");
    Check(Pomade_GraphSnapNode(ctx, p, 10.0f) >= 0,
          "C ABI: snap finds a node");
    Check(Pomade_GraphSnapNode(nullptr, p, 10.0f) == -1,
          "C ABI: snap of a null context is -1");
    unsigned long long const mv = Pomade_GetMapVersion(ctx);
    Check(mv > 0, "C ABI: the map version advances on graph edits");
    Check(Pomade_Destroy(ctx) == POMADE_OK, "C ABI: model destroys");
    Check(Pomade_Destroy(nullptr) == POMADE_OK, "C ABI: null destroy is a noop");
}

// A loop around the subset's hole: corners on faces 0, 12, 15 and 3, every
// traced edge over active faces, face 5 strictly inside. `mesh` traces it.
std::vector<int> AddRingRegion(Grid const &grid,
                               usdGenPomade::PomadeScalpMesh const &mesh,
                               usdGenPomade::PomadeScalpGraph *graph)
{
    int const r0 = graph->AddNode(Locate(grid, 0.5f, 0.5f));
    int const r1 = graph->AddNode(Locate(grid, 3.5f, 0.5f));
    int const r2 = graph->AddNode(Locate(grid, 3.5f, 3.5f));
    int const r3 = graph->AddNode(Locate(grid, 0.5f, 3.5f));
    graph->Connect(mesh, r0, r1);
    graph->Connect(mesh, r1, r2);
    graph->Connect(mesh, r2, r3);
    graph->Connect(mesh, r3, r0);
    return {r0, r1, r2, r3};
}

// plan/02 §2.20: a face GeomSubset scalp. Face ids stay parent-mesh ids and
// every per-face array stays parent-sized; only the surface queries (LBVH
// raycast, closest point, K3, region picking, ABI placement) skip the faces
// the subset leaves out, and an invalid subset is an error.
void CheckSubset(Grid const &whole)
{
    using namespace usdGenPomade;
    Grid sub = MakeSubsetGrid();
    std::vector<int> wantActive;
    for (int f = 0; f < 16; ++f) {
        if (f != kHoleFace) {
            wantActive.push_back(f);
        }
    }
    Check(sub.mesh.finalized && sub.mesh.activeFaces == wantActive,
          "subset: finalize sorts and de-dupes the parent face ids");
    Check(sub.mesh.faceVertexCounts.size() == 16 &&
              sub.mesh.faceActive.size() == 16 &&
              sub.mesh.faceAreas.size() == 16 &&
              sub.mesh.faceNeighbours.size() == 16,
          "subset: per-face arrays stay parent-sized (no renumbering)");
    bool flags = true;
    for (int f = 0; f < 16; ++f) {
        flags = flags && PomadeScalpFaceActive(sub.mesh, f) == (f != kHoleFace);
    }
    Check(flags && !PomadeScalpFaceActive(sub.mesh, 16) &&
              !PomadeScalpFaceActive(sub.mesh, -1),
          "subset: exactly the subset's faces are active");
    Check(sub.bvh.valid && sub.bvh.order.size() == 15 &&
              sub.bvh.nodes.size() == 2 * 15 - 1 &&
              std::find(sub.bvh.order.begin(), sub.bvh.order.end(),
                        kHoleFace) == sub.bvh.order.end(),
          "subset: the LBVH holds the subset's faces only");

    // K1: rays and closest points respect the subset.
    float const down[3] = {0.0f, -1.0f, 0.0f};
    float const overHole[3] = {1.5f, 5.0f, 1.5f};
    float const overActive[3] = {1.5f, 5.0f, 2.5f};
    Check(!PomadeRaycastCpu(sub.mesh, sub.bvh, overHole, down).hit &&
              PomadeRaycastCpu(whole.mesh, whole.bvh, overHole, down).faceId ==
                  kHoleFace,
          "subset: a ray at a face outside the subset misses (the whole "
          "mesh hits it)");
    PomadeHit const in = PomadeRaycastCpu(sub.mesh, sub.bvh, overActive, down);
    Check(in.hit && in.faceId == 6 && in.ptexFaceId == 6 &&
              Near(in.u, 0.5f) && Near(in.v, 0.5f),
          "subset: a ray at a subset face hits it by its parent id");
    float const aboveHole[3] = {1.2f, 1.0f, 1.5f};
    PomadeHit const snapped = PomadeClosestPointCpu(sub.mesh, aboveHole);
    Check(snapped.hit && snapped.faceId == 1 && Near(snapped.px, 1.0f) &&
              Near(snapped.pz, 1.5f),
          "subset: the closest point lands on a subset face, by parent id");
    PomadeHit const anyFace =
        PomadeClosestPointCpu(sub.mesh, aboveHole, /*activeOnly*/ false);
    Check(anyFace.hit && anyFace.faceId == kHoleFace &&
              Near(anyFace.px, 1.2f) && Near(anyFace.pz, 1.5f),
          "subset: activeOnly=false still sees every parent face");
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    Check(PomadeFacePosition(sub.mesh, kHoleFace, 0.5f, 0.5f, &px, &py, &pz) &&
              Near(px, 1.5f) && Near(pz, 1.5f),
          "subset: a left-out face still addresses parent geometry");
    Check(PomadeScalpBvhRefit(sub.mesh, &sub.bvh) &&
              PomadeRaycastCpu(sub.mesh, sub.bvh, overActive, down).faceId == 6 &&
              !PomadeRaycastCpu(sub.mesh, sub.bvh, overHole, down).hit,
          "subset: a refit keeps the subset-only tree");

    // Invalid subsets are errors, never a silent clamp or a whole bind.
    for (std::vector<int> const &bad : {std::vector<int>{3, 16},
                                        std::vector<int>{-1, 2}}) {
        PomadeScalpMesh mesh = whole.mesh;
        mesh.activeFaces = bad;
        std::string err;
        bool const refused = !PomadeScalpFinalize(&mesh, &err) &&
                             !mesh.finalized &&
                             err.find("out of range") != std::string::npos;
        Check(refused,
              "subset: an out-of-range face id is refused (" + err + ")");
    }

    // K3: left-out faces are never claimed and never counted uncovered.
    PomadeRegionMaps maps;
    std::string err;
    PomadeScalpGraph empty;
    Check(PomadeRasteriseRegionsCpu(sub.mesh, empty, &maps, &err) &&
              maps.uncoveredCount == 15 && maps.faceRegion.size() == 16,
          "subset K3: with no region, only the 15 subset faces are uncovered");
    PomadeScalpGraph ring;
    AddRingRegion(whole, sub.mesh, &ring);
    Check(ring.RegionCount() == 1, "subset K3: the ring is one region");
    PomadeRegionMaps wholeMaps;
    Check(PomadeRasteriseRegionsCpu(sub.mesh, ring, &maps, &err) &&
              PomadeRasteriseRegionsCpu(whole.mesh, ring, &wholeMaps, &err),
          "subset K3: the ring rasterises on both bindings: " + err);
    bool claimed = maps.faceRegion.size() == 16;
    for (int f = 0; claimed && f < 16; ++f) {
        claimed = f == kHoleFace
            ? maps.faceRegion[size_t(f)] == -1 && !maps.covered[size_t(f)]
            : maps.faceRegion[size_t(f)] == 0;
    }
    Check(claimed && maps.uncoveredCount == 0 && maps.intersectedCount == 0,
          "subset K3: the ring claims every subset face and never the hole");
    Check(wholeMaps.faceRegion[size_t(kHoleFace)] == 0,
          "subset K3: the whole mesh claims the same face (the mask is the "
          "only difference)");
    PomadeRegionLoops loops;
    PomadeFlattenLoops(ring, &loops, &err);
    std::vector<int> const res = PomadeFaceResLog2(sub.mesh, maps, loops, 3);
    Check(res.size() == 16 && res[size_t(kHoleFace)] == 0 && res[6] == 3,
          "subset K3: a left-out face bakes 1x1 even under an override");

    // The model: RegionAtSurface and the face-map census.
    PomadeModel model;
    Check(model.BindScalp(whole.mesh.points, whole.mesh.faceVertexCounts,
                          whole.mesh.faceVertexIndices, kSubsetFaces),
          "subset model: the scalp binds through a face subset");
    std::shared_ptr<PomadeScalpMesh const> const bound = model.GetScalp();
    Check(bound && bound->activeFaces == wantActive,
          "subset model: the bound scalp keeps the parent mesh + subset");
    PomadeModel rejected;
    Check(!rejected.BindScalp(whole.mesh.points, whole.mesh.faceVertexCounts,
                              whole.mesh.faceVertexIndices, {0, 99}) &&
              !rejected.HasScalp() &&
              std::string(rejected.GetDiagnostic()).find("out of range") !=
                  std::string::npos,
          "subset model: an out-of-range subset refuses the bind");
    int const n0 = model.GraphAddNode(Locate(whole, 0.5f, 0.5f));
    int const n1 = model.GraphAddNode(Locate(whole, 3.5f, 0.5f));
    int const n2 = model.GraphAddNode(Locate(whole, 3.5f, 3.5f));
    int const n3 = model.GraphAddNode(Locate(whole, 0.5f, 3.5f));
    model.GraphConnect(n0, n1);
    model.GraphConnect(n1, n2);
    model.GraphConnect(n2, n3);
    model.GraphConnect(n3, n0);
    Check(model.Rasterise() && model.GetRegionMaps().uncoveredCount == 0 &&
              model.SnapshotGraph().faceRegions.size() == 16 &&
              model.SnapshotGraph().faceRegions[size_t(kHoleFace)] == -1,
          "subset model: the live map is parent-sized, the hole reads -1");
    Check(model.RegionAtSurface(6, 0.5f, 0.5f) == 0 &&
              model.RegionAtSurface(kHoleFace, 0.5f, 0.5f) == -1,
          "subset model: region picking never answers for a left-out face");

    // The C ABI.
    PomadeModelContext *ctx = nullptr;
    Check(Pomade_Create(&ctx) == POMADE_OK && ctx, "subset C ABI: model creates");
    std::vector<float> const &pts = whole.mesh.points;
    std::vector<int> const &counts = whole.mesh.faceVertexCounts;
    std::vector<int> const &indices = whole.mesh.faceVertexIndices;
    Check(Pomade_BindScalpSubset(ctx, pts.data(), int(pts.size()),
                                counts.data(), int(counts.size()),
                                indices.data(), int(indices.size()), nullptr,
                                0) == POMADE_ERROR &&
              std::string(Pomade_GetLastError()).find("names no faces") !=
                  std::string::npos,
          "subset C ABI: an empty subset is an error, not the whole mesh");
    int const outOfRange[2] = {3, 16};
    Check(Pomade_BindScalpSubset(ctx, pts.data(), int(pts.size()),
                                counts.data(), int(counts.size()),
                                indices.data(), int(indices.size()),
                                outOfRange, 2) == POMADE_ERROR &&
              Pomade_HasScalp(ctx) == 0,
          "subset C ABI: an out-of-range subset is an error");
    Check(Pomade_BindScalpSubset(ctx, pts.data(), int(pts.size()),
                                counts.data(), int(counts.size()),
                                indices.data(), int(indices.size()),
                                kSubsetFaces.data(),
                                int(kSubsetFaces.size())) == POMADE_OK &&
              Pomade_HasScalp(ctx) == 1,
          "subset C ABI: the subset binds");
    int active[16] = {0};
    int got = 0;
    bool activeOk =
        Pomade_ReadScalpFaceActive(ctx, active, 16, &got) == POMADE_OK &&
        got == 16;
    for (int f = 0; activeOk && f < 16; ++f) {
        activeOk = active[f] == (f == kHoleFace ? 0 : 1);
    }
    Check(activeOk && Pomade_ReadScalpFaceActive(ctx, active, 4, &got) ==
                          POMADE_ERROR,
          "subset C ABI: the per-face active flags read back");
    int hit = 1, face = -1;
    float uv[2] = {0.0f, 0.0f};
    float p[3] = {0.0f, 0.0f, 0.0f};
    float nrm[3] = {0.0f, 0.0f, 0.0f};
    Check(Pomade_Raycast(ctx, overHole, down, &hit, &face, uv, p, nrm) ==
                  POMADE_OK &&
              hit == 0,
          "subset C ABI: the raycast misses a left-out face");
    int nodeId = -1;
    Check(Pomade_GraphAddNode(ctx, kHoleFace, 0.5f, 0.5f, &nodeId) ==
                  POMADE_ERROR &&
              Pomade_GraphAddNode(ctx, 6, 0.5f, 0.5f, &nodeId) == POMADE_OK,
          "subset C ABI: nodes place on subset faces only");
    Check(Pomade_Destroy(ctx) == POMADE_OK, "subset C ABI: model destroys");
}

#ifdef USDGEN_POMADE_HAS_CUDA
bool HaveCudaDevice()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void CheckCudaParity(Grid &grid)
{
    using namespace usdGenPomade;
    if (!HaveCudaDevice()) {
        std::printf("skip: no CUDA device for the TN-6 parity checks\n");
        return;
    }
    // Layout: the device BVH node must match the host bytes exactly.
    bool layout =
        sizeof(PomadeDeviceBvhNode) == sizeof(PomadeScalpBvhNode) &&
        offsetof(PomadeDeviceBvhNode, minX) ==
            offsetof(PomadeScalpBvhNode, minX) &&
        offsetof(PomadeDeviceBvhNode, left) ==
            offsetof(PomadeScalpBvhNode, left) &&
        offsetof(PomadeDeviceBvhNode, faceBegin) ==
            offsetof(PomadeScalpBvhNode, faceBegin);
    Check(layout, "TN-6: the device BVH node matches the host layout");
    if (!layout) {
        return;
    }
    char err[256] = {0};
    // Upload the mesh once for all three kernels.
    float *dPts = nullptr;
    int *dCounts = nullptr, *dIndices = nullptr, *dOffsets = nullptr;
    float *dNormals = nullptr;
    PomadeDeviceBvhNode *dNodes = nullptr;
    int *dOrder = nullptr;
    size_t const nPts = grid.mesh.points.size();
    size_t const nCounts = grid.mesh.faceVertexCounts.size();
    size_t const nIndices = grid.mesh.faceVertexIndices.size();
    size_t const nNodes = grid.bvh.nodes.size();
    cudaMalloc(&dPts, nPts * sizeof(float));
    cudaMalloc(&dCounts, nCounts * sizeof(int));
    cudaMalloc(&dIndices, nIndices * sizeof(int));
    cudaMalloc(&dOffsets, (nCounts + 1) * sizeof(int));
    cudaMalloc(&dNormals, nCounts * 3 * sizeof(float));
    cudaMalloc(&dNodes, nNodes * sizeof(PomadeDeviceBvhNode));
    cudaMalloc(&dOrder, nCounts * sizeof(int));
    cudaMemcpy(dPts, grid.mesh.points.data(), nPts * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dCounts, grid.mesh.faceVertexCounts.data(),
               nCounts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dIndices, grid.mesh.faceVertexIndices.data(),
               nIndices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dOffsets, grid.mesh.faceOffsets.data(),
               (nCounts + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dNormals, grid.mesh.faceNormals.data(),
               nCounts * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dNodes, grid.bvh.nodes.data(),
               nNodes * sizeof(PomadeDeviceBvhNode), cudaMemcpyHostToDevice);
    cudaMemcpy(dOrder, grid.bvh.order.data(), nCounts * sizeof(int),
               cudaMemcpyHostToDevice);
    // K1: ordinary hits/misses plus the captured StageView fan-diagonal
    // sample, a nearby seam sample and an outer-silhouette miss.
    int const rayCount = 9;
    float origins[rayCount * 3] = {
        1.5f, 5.0f, 0.5f, 0.5f, 5.0f, 3.5f,
        3.5f, 5.0f, 2.5f, 2.0f, 5.0f, 2.0f,
        1.5f, 5.0f, 0.5f, 1.5f, 5.0f, 0.5f,
        1.992840022f, 13.90000005f, 1.992939308f,
        1.992840022f, 13.90000005f, 1.992939308f,
        1.992840022f, 13.90000005f, 1.992939308f};
    float dirs[rayCount * 3] = {
        0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f,
        0.0f, 1.0f,  0.0f, 0.0f, 0.0f,  0.0f,
        -0.071235519f, -0.994912051f, -0.071242625f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    RayTo(origins + 7 * 3, 0.99762f, 0.0f, 0.99758f,
          dirs + 7 * 3);
    RayTo(origins + 8 * 3, -0.0002f, 0.0f, -0.0002f,
          dirs + 8 * 3);
    float *dOrg = nullptr, *dDir = nullptr;
    PomadeDeviceHit *dHits = nullptr;
    cudaMalloc(&dOrg, sizeof(origins));
    cudaMalloc(&dDir, sizeof(dirs));
    cudaMalloc(&dHits, rayCount * sizeof(PomadeDeviceHit));
    cudaMemcpy(dOrg, origins, sizeof(origins), cudaMemcpyHostToDevice);
    cudaMemcpy(dDir, dirs, sizeof(dirs), cudaMemcpyHostToDevice);
    bool launched = PomadeLaunchRaycastBatch(
        dPts, dCounts, dIndices, dOffsets, dNormals, dNodes, dOrder,
        int(nCounts), int(nNodes), grid.bvh.root, dOrg, dDir, dHits, rayCount,
        nullptr, err, sizeof(err));
    Check(launched, "TN-6: K1 launches");
    cudaDeviceSynchronize();
    PomadeDeviceHit hits[rayCount];
    cudaMemcpy(hits, dHits, sizeof(hits), cudaMemcpyDeviceToHost);
    bool k1 = launched;
    for (int r = 0; r < rayCount && k1; ++r) {
        PomadeHit const cpu = PomadeRaycastCpu(grid.mesh, grid.bvh,
                                             origins + r * 3, dirs + r * 3);
        bool const cpuHit = cpu.hit;
        bool const gpuHit = hits[r].faceId >= 0;
        k1 = cpuHit == gpuHit;
        if (cpuHit && gpuHit) {
            k1 = k1 && hits[r].faceId == cpu.faceId &&
                 Near(hits[r].t, cpu.t, 1e-5f) &&
                 Near(hits[r].px, cpu.px, 1e-4f) &&
                 Near(hits[r].py, cpu.py, 1e-4f) &&
                 Near(hits[r].pz, cpu.pz, 1e-4f);
        }
    }
    Check(k1, "TN-6: K1 device raycast matches the CPU twin");
    // The same rays over a face-subset LBVH (plan/02 §2.20): the device
    // walks the uploaded subset-only tree, so it misses the hole exactly
    // where the CPU twin does. The mesh buffers are shared (same grid).
    Grid const sub = MakeSubsetGrid();
    float const holeOrigin[3] = {1.5f, 5.0f, 1.5f};
    std::memcpy(origins + 5 * 3, holeOrigin, sizeof(holeOrigin));
    dirs[5 * 3 + 0] = 0.0f;
    dirs[5 * 3 + 1] = -1.0f;
    dirs[5 * 3 + 2] = 0.0f;
    size_t const nSubNodes = sub.bvh.nodes.size();
    size_t const nSubOrder = sub.bvh.order.size();
    PomadeDeviceBvhNode *dSubNodes = nullptr;
    int *dSubOrder = nullptr;
    cudaMalloc(&dSubNodes, nSubNodes * sizeof(PomadeDeviceBvhNode));
    cudaMalloc(&dSubOrder, nSubOrder * sizeof(int));
    cudaMemcpy(dSubNodes, sub.bvh.nodes.data(),
               nSubNodes * sizeof(PomadeDeviceBvhNode), cudaMemcpyHostToDevice);
    cudaMemcpy(dSubOrder, sub.bvh.order.data(), nSubOrder * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dOrg, origins, sizeof(origins), cudaMemcpyHostToDevice);
    cudaMemcpy(dDir, dirs, sizeof(dirs), cudaMemcpyHostToDevice);
    bool const subLaunched = PomadeLaunchRaycastBatch(
        dPts, dCounts, dIndices, dOffsets, dNormals, dSubNodes, dSubOrder,
        int(nCounts), int(nSubNodes), sub.bvh.root, dOrg, dDir, dHits,
        rayCount, nullptr, err, sizeof(err));
    cudaDeviceSynchronize();
    PomadeDeviceHit subHits[rayCount];
    cudaMemcpy(subHits, dHits, sizeof(subHits), cudaMemcpyDeviceToHost);
    bool k1Subset = subLaunched && subHits[5].faceId < 0;
    for (int r = 0; r < rayCount && k1Subset; ++r) {
        PomadeHit const cpu = PomadeRaycastCpu(sub.mesh, sub.bvh,
                                             origins + r * 3, dirs + r * 3);
        k1Subset = cpu.hit == (subHits[r].faceId >= 0) &&
                   (!cpu.hit || (subHits[r].faceId == cpu.faceId &&
                                 Near(subHits[r].t, cpu.t, 1e-5f)));
    }
    Check(k1Subset,
          "TN-6: K1 device raycast over a face subset matches the CPU twin "
          "and misses the left-out face");
    cudaFree(dSubNodes);
    cudaFree(dSubOrder);
    cudaFree(dOrg);
    cudaFree(dDir);
    cudaFree(dHits);
    // K2: the same trace on both lanes.
    float const a[3] = {0.5f, 0.0f, 1.5f};
    float const b[3] = {3.5f, 0.0f, 1.5f};
    std::vector<float> const cpuPoly = PomadeTraceEdgeCpu(grid.mesh, a, b);
    int const nSamples = int(cpuPoly.size() / 3);
    float *dPos = nullptr;
    int *dFaces = nullptr;
    cudaMalloc(&dPos, cpuPoly.size() * sizeof(float));
    cudaMalloc(&dFaces, size_t(nSamples) * sizeof(int));
    Check(PomadeLaunchEdgeTrace(dPts, dCounts, dIndices, dOffsets,
                               int(nCounts), a, b, dPos, dFaces, nSamples,
                               nullptr, err, sizeof(err)),
          "TN-6: K2 launches");
    cudaDeviceSynchronize();
    std::vector<float> gpuPoly(cpuPoly.size(), 0.0f);
    cudaMemcpy(gpuPoly.data(), dPos, cpuPoly.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    bool k2 = gpuPoly.size() == cpuPoly.size();
    for (size_t i = 0; k2 && i < cpuPoly.size(); ++i) {
        k2 = Near(gpuPoly[i], cpuPoly[i], 1e-4f);
    }
    Check(k2, "TN-6: K2 device trace matches the CPU twin");
    cudaFree(dPos);
    cudaFree(dFaces);
    // K3: 48 query points over two regions, exact int equality.
    PomadeScalpGraph graph;
    BuildAdjoining(grid, &graph);
    PomadeRegionLoops loops;
    std::string lerr;
    PomadeFlattenLoops(graph, &loops, &lerr);
    std::vector<float> queries;
    for (int ix = 0; ix < 4; ++ix) {
        for (int iz = 0; iz < 4; ++iz) {
            for (auto off : {0.25f, 0.5f, 0.75f}) {
                queries.push_back(float(ix) + off);
                queries.push_back(0.0f);
                queries.push_back(float(iz) + off);
            }
        }
    }
    int const nQ = int(queries.size() / 3);
    float *dQ = nullptr;
    float *dLoop = nullptr, *dN = nullptr, *dP = nullptr, *dU = nullptr,
          *dV = nullptr;
    int *dBegin = nullptr, *dCount = nullptr, *dInterp = nullptr,
        *dOut = nullptr;
    cudaMalloc(&dQ, queries.size() * sizeof(float));
    cudaMalloc(&dLoop, loops.points.size() * sizeof(float));
    cudaMalloc(&dBegin, loops.loopBegin.size() * sizeof(int));
    cudaMalloc(&dCount, loops.loopCount.size() * sizeof(int));
    cudaMalloc(&dN, loops.planeN.size() * sizeof(float));
    cudaMalloc(&dP, loops.planeP.size() * sizeof(float));
    cudaMalloc(&dU, loops.basisU.size() * sizeof(float));
    cudaMalloc(&dV, loops.basisV.size() * sizeof(float));
    cudaMalloc(&dInterp, loops.interpIds.size() * sizeof(int));
    cudaMalloc(&dOut, size_t(nQ) * sizeof(int));
    cudaMemcpy(dQ, queries.data(), queries.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dLoop, loops.points.data(),
               loops.points.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dBegin, loops.loopBegin.data(),
               loops.loopBegin.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dCount, loops.loopCount.data(),
               loops.loopCount.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dN, loops.planeN.data(), loops.planeN.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dP, loops.planeP.data(), loops.planeP.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dU, loops.basisU.data(), loops.basisU.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dV, loops.basisV.data(), loops.basisV.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dInterp, loops.interpIds.data(),
               loops.interpIds.size() * sizeof(int), cudaMemcpyHostToDevice);
    Check(PomadeLaunchClassifyPoints(
              dQ, nQ, dLoop, dBegin, dCount, dN, dP, dU, dV, dInterp,
              int(loops.loopCount.size()), dOut, nullptr, err, sizeof(err)),
          "TN-6: K3 launches");
    cudaDeviceSynchronize();
    std::vector<int> gpuIds(size_t(nQ), -2);
    cudaMemcpy(gpuIds.data(), dOut, size_t(nQ) * sizeof(int),
               cudaMemcpyDeviceToHost);
    bool k3 = true;
    for (int i = 0; i < nQ && k3; ++i) {
        int const cpu = PomadeClassifyPointCpu(loops, &queries[size_t(i) * 3]);
        k3 = gpuIds[size_t(i)] == cpu;
    }
    Check(k3, "TN-6: K3 device classification matches bit-exactly");
    cudaFree(dQ);
    cudaFree(dLoop);
    cudaFree(dBegin);
    cudaFree(dCount);
    cudaFree(dN);
    cudaFree(dP);
    cudaFree(dU);
    cudaFree(dV);
    cudaFree(dInterp);
    cudaFree(dOut);
    cudaFree(dPts);
    cudaFree(dCounts);
    cudaFree(dIndices);
    cudaFree(dOffsets);
    cudaFree(dNormals);
    cudaFree(dNodes);
    cudaFree(dOrder);
}
#else
void CheckCudaParity(Grid &)
{
    std::printf("skip: GPU-less build, no TN-6 parity checks\n");
}
#endif

}  // namespace

int
main()
{
    Grid grid = MakeGrid();
    CheckK1(grid);
    CheckK2(grid);
    CheckGraphOps(grid);
    CheckK3(grid);
    CheckModel(grid);
    CheckCApi(grid);
    CheckSubset(grid);
    CheckCudaParity(grid);
    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
