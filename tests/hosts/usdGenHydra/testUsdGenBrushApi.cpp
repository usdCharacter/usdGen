// testUsdGenBrushApi (T1, no GPU) — the attribute-brush C ABI
// (usdGenImaging/usdGenBrushApi.h):
//
//   * the mesh handle picks (uniform-grid index == brute force on a plane),
//     expands footprints onto neighbouring faces and suggests a resolution;
//   * the stroke handle applies set / add / erase with the inner (hardness)
//     radius, tracks touched faces, and Commit reproduces the working grid;
//   * smooth visibly softens a hard 1/0 edge that only persists as 4 corners
//     per quad (the cross-face corner smoother);
//   * the overlay entry points reach an attached preview scene index.
#include "utils/pxrInterop.h"
#include "usdGenImaging/usdGenBrushApi.h"
#include "usdGenImaging/attributePreviewSceneIndex.h"

#include "pxr/imaging/hd/primvarSchema.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

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
}

// n x n unit quads on the y = 0 plane: face (i, j) spans x in [i, i+1],
// z in [j, j+1]; u runs along +x, v along +z.
struct Plane {
    int n = 0;
    std::vector<double> points;
    std::vector<int> indices;
    int Face(int i, int j) const { return j * n + i; }
};

Plane MakePlane(int n)
{
    Plane p;
    p.n = n;
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) {
            p.points.push_back(double(i));
            p.points.push_back(0.0);
            p.points.push_back(double(j));
        }
    auto vid = [&](int i, int j) { return j * (n + 1) + i; };
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            p.indices.push_back(vid(i, j));
            p.indices.push_back(vid(i + 1, j));
            p.indices.push_back(vid(i + 1, j + 1));
            p.indices.push_back(vid(i, j + 1));
        }
    return p;
}

std::vector<float> Corners(void *stroke, int faces, int channels = 1)
{
    std::vector<float> out(size_t(faces) * 4 * size_t(channels));
    UsdGenBrush_StrokeWorkingCorners(stroke, out.data());
    return out;
}

class DirtyRecorder final : public HdSceneIndexObserver {
public:
    std::vector<std::pair<SdfPath, HdDataSourceLocatorSet>> dirtied;
    void PrimsAdded(HdSceneIndexBase const &, AddedPrimEntries const &) override {}
    void PrimsRemoved(HdSceneIndexBase const &, RemovedPrimEntries const &) override {}
    void PrimsDirtied(HdSceneIndexBase const &, DirtiedPrimEntries const &entries) override
    {
        for (auto const &entry : entries) dirtied.emplace_back(entry.primPath, entry.dirtyLocators);
    }
    void PrimsRenamed(HdSceneIndexBase const &, RenamedPrimEntries const &) override {}
    size_t CountFor(SdfPath const &path, HdDataSourceLocator const &locator) const
    {
        size_t n = 0;
        // Exact: the set names this very locator (a structural dirty must
        // not count as a values-only one, and vice versa).
        for (auto const &entry : dirtied) {
            if (entry.first != path) continue;
            for (auto const &l : entry.second)
                if (l == locator) ++n;
        }
        return n;
    }
};

VtVec3fArray DisplayColors(HdSceneIndexBase &index, SdfPath const &path, TfToken *interp)
{
    HdSceneIndexPrim const prim = index.GetPrim(path);
    HdPrimvarSchema const pv(HdContainerDataSource::Cast(HdContainerDataSource::Get(
        prim.dataSource, HdDataSourceLocator(TfToken("primvars"), TfToken("displayColor")))));
    *interp = TfToken();
    if (pv.IsDefined() && pv.GetInterpolation())
        *interp = pv.GetInterpolation()->GetTypedValue(0.0f);
    if (!pv.IsDefined() || !pv.GetPrimvarValue()) return VtVec3fArray();
    VtValue const value = pv.GetPrimvarValue()->GetValue(0.0f);
    return value.IsHolding<VtVec3fArray>() ? value.UncheckedGet<VtVec3fArray>() : VtVec3fArray();
}

}  // namespace

int main()
{
    Check(UsdGenBrush_ApiVersion() == 1, "the brush ABI speaks version 1");

    // ---- mesh: pick, footprint, resolution ---------------------------------
    Plane const plane = MakePlane(4);
    int const F = 16;
    void *mesh = UsdGenBrush_MeshCreate(plane.points.data(), int(plane.points.size() / 3),
                                        plane.indices.data(), F);
    Check(mesh != nullptr && UsdGenBrush_MeshFaceCount(mesh) == F, "the mesh handle builds");
    {
        int const bad[4] = {0, 1, 2, 99};
        Check(UsdGenBrush_MeshCreate(plane.points.data(), int(plane.points.size() / 3), bad, 1) ==
                  nullptr,
              "an out-of-range face-vertex index is rejected");
    }
    {
        double const o[3] = {2.25, 5.0, 1.75};
        double const d[3] = {0.0, -1.0, 0.0};
        int face = -1;
        float u = -1, v = -1;
        double p[3] = {0, 0, 0};
        int const hit = UsdGenBrush_MeshPick(mesh, o, d, &face, &u, &v, p);
        Check(hit == 1 && face == plane.Face(2, 1) && std::fabs(u - 0.25f) < 1e-5f &&
                  std::fabs(v - 0.75f) < 1e-5f && std::fabs(p[1]) < 1e-9,
              "a downward ray picks face (2,1) at (0.25, 0.75)");
        double const up[3] = {0.0, 1.0, 0.0};
        Check(UsdGenBrush_MeshPick(mesh, o, up, &face, &u, &v, p) == 0,
              "a ray pointing away misses");
        double const below[3] = {1.5, -3.0, 3.5};
        Check(UsdGenBrush_MeshPick(mesh, below, up, &face, &u, &v, p) == 1 &&
                  face == plane.Face(1, 3),
              "the back side picks too (both windings)");
        // Exhaustive agreement with the analytic answer over a sample grid.
        bool all = true;
        for (int k = 0; k < 64; ++k) {
            double const x = 0.03 + (k % 8) * 0.49, z = 0.07 + (k / 8) * 0.49;
            double const oo[3] = {x, 2.0, z};
            if (UsdGenBrush_MeshPick(mesh, oo, d, &face, &u, &v, p) != 1 ||
                face != plane.Face(int(x), int(z)))
                all = false;
        }
        Check(all, "grid-indexed picks agree with the plane layout everywhere");
    }
    {
        int faces[64];
        float us[64], vs[64], rs[64];
        int const n = UsdGenBrush_MeshFootprint(mesh, plane.Face(1, 1), 0.95f, 0.5f, nullptr,
                                                0.2f, faces, us, vs, rs, 64);
        bool hasRight = false;
        for (int i = 1; i < n; ++i)
            if (faces[i] == plane.Face(2, 1) && us[i] < 1e-5f && std::fabs(vs[i] - 0.5f) < 1e-5f)
                hasRight = true;
        Check(n >= 2 && faces[0] == plane.Face(1, 1) && std::fabs(us[0] - 0.95f) < 1e-6f &&
                  hasRight && std::fabs(rs[0] - 0.2f) < 1e-6f,
              "a dab near the +u edge spills onto the right neighbour, primary first");
        int const inside = UsdGenBrush_MeshFootprint(mesh, plane.Face(1, 1), 0.5f, 0.5f, nullptr,
                                                    0.1f, faces, us, vs, rs, 64);
        Check(inside == 1, "a dab in the middle of a face stays on it");
        int one[1] = {-1};
        int const total = UsdGenBrush_MeshFootprint(mesh, plane.Face(1, 1), 0.95f, 0.5f, nullptr,
                                                   0.2f, one, nullptr, nullptr, nullptr, 1);
        Check(total == n && one[0] == plane.Face(1, 1),
              "a truncated footprint returns the total count, primary written first");
    }
    Check(std::fabs(UsdGenBrush_MeshFaceEdgeLen(mesh, 0) - 1.0f) < 1e-6f,
          "the longest edge of a unit quad is 1");
    {
        char info[256] = {0};
        int const res = UsdGenBrush_MeshSuggestResolution(mesh, 4000000, info, 256);
        Check(res == 32 && std::string(info).find("32 px/face") == 0,
              "a small mesh gets 32 px/face (" + std::string(info) + ")");
        int const tight = UsdGenBrush_MeshSuggestResolution(mesh, 16 * 8 * 8, info, 256);
        Check(tight == 8, "the texel budget lowers the resolution");
        int const floor4 = UsdGenBrush_MeshSuggestResolution(mesh, 1, info, 256);
        Check(floor4 == 4, "the resolution floors at 4");
    }

    // ---- stroke: set / hardness / erase / add / commit ----------------------
    {
        void *stroke = UsdGenBrush_StrokeCreate(mesh, F, 8, 1, nullptr, 0.0f);
        Check(stroke != nullptr, "a stroke over the mesh builds");
        int const f = plane.Face(1, 1);
        // Hardness 0: corners sit at 0.707 of a 0.75 radius -> nearly zero.
        int stamps = UsdGenBrush_StrokeDab(stroke, f, 0.5f, 0.5f, 0.75f, 0.0f, 1.0f, 1.0f, 0,
                                           USDGEN_BRUSH_MODE_SET, USDGEN_BRUSH_FALLOFF_SMOOTH, 0,
                                           0.5f);
        std::vector<float> c = Corners(stroke, F);
        Check(stamps >= 1 && c[size_t(f) * 4] < 0.05f, "a soft dab barely reaches the corners");
        UsdGenBrush_StrokeAbort(stroke);
        Check(UsdGenBrush_StrokeDabCount(stroke) == 0 && Corners(stroke, F)[size_t(f) * 4] == 0.0f,
              "abort restores the base");
        // Hardness 1: the whole disk is at full weight.
        UsdGenBrush_StrokeDab(stroke, f, 0.5f, 0.5f, 0.75f, 1.0f, 1.0f, 1.0f, 0,
                              USDGEN_BRUSH_MODE_SET, USDGEN_BRUSH_FALLOFF_SMOOTH, 0, 0.5f);
        c = Corners(stroke, F);
        bool full = true;
        for (int i = 0; i < 4; ++i) full = full && std::fabs(c[size_t(f) * 4 + i] - 1.0f) < 1e-6f;
        Check(full, "a hard dab sets every corner the radius reaches");
        // Neighbour faces share those vertices: the footprint painted them too.
        Check(std::fabs(c[size_t(plane.Face(2, 1)) * 4 + 0] - 1.0f) < 1e-6f,
              "the footprint painted the neighbour's shared corner");
        std::vector<int> touched(64, -1);
        int const nTouched = UsdGenBrush_StrokeTakeTouched(stroke, touched.data(), 64);
        Check(nTouched >= 5 && touched[0] < touched[1],
              "touched faces come back ascending (" + std::to_string(nTouched) + ")");
        Check(UsdGenBrush_StrokeTakeTouched(stroke, touched.data(), 64) == 0,
              "a take clears the touched set");
        // Erase blends toward the default (0.5 here via a second stroke).
        void *erase = UsdGenBrush_StrokeCreate(mesh, F, 8, 1, nullptr, 0.5f);
        UsdGenBrush_StrokeDab(erase, f, 0.5f, 0.5f, 0.75f, 1.0f, 1.0f, 1.0f, 0,
                              USDGEN_BRUSH_MODE_SET, USDGEN_BRUSH_FALLOFF_SMOOTH, 0, 0.5f);
        UsdGenBrush_StrokeDab(erase, f, 0.5f, 0.5f, 0.75f, 1.0f, 1.0f, 7.0f, 0,
                              USDGEN_BRUSH_MODE_ERASE, USDGEN_BRUSH_FALLOFF_SMOOTH, 0, 0.5f);
        Check(std::fabs(Corners(erase, F)[size_t(f) * 4] - 0.5f) < 1e-6f,
              "erase returns painted corners to the map default, ignoring value");
        UsdGenBrush_StrokeDestroy(erase);
        // Add + a move trail, then commit == working.
        UsdGenBrush_StrokeDab(stroke, plane.Face(0, 0), 0.2f, 0.2f, 0.8f, 0.5f, 0.5f, 0.25f, 0,
                              USDGEN_BRUSH_MODE_ADD, USDGEN_BRUSH_FALLOFF_LINEAR, 0, 0.5f);
        UsdGenBrush_StrokeDab(stroke, plane.Face(0, 0), 0.8f, 0.9f, 0.8f, 0.5f, 0.5f, 0.25f, 0,
                              USDGEN_BRUSH_MODE_ADD, USDGEN_BRUSH_FALLOFF_LINEAR, 1, 0.25f);
        Check(UsdGenBrush_StrokeDabCount(stroke) > 3, "a move records an interpolated trail");
        std::vector<float> working = Corners(stroke, F);
        std::vector<float> committed(working.size());
        UsdGenBrush_StrokeCommitCorners(stroke, committed.data());
        Check(working == committed, "commit reproduces the working corners bit for bit");
        Check(UsdGenBrush_StrokeDab(stroke, 99, 0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 0, 0, 0, 0,
                                    0.5f) < 0 &&
                  UsdGenBrush_StrokeDab(stroke, 0, 0.5f, 0.5f, 0.5f, 1.5f, 1.0f, 1.0f, 0, 0, 0, 0,
                                        0.5f) < 0,
              "a bad face or hardness is rejected");
        std::vector<float> rgb(size_t(F) * 12);
        Check(UsdGenBrush_StrokePreviewColors(stroke, 0, USDGEN_BRUSH_COLORMAP_GRAY, 0.0f, 1.0f,
                                              rgb.data()) == F * 4 &&
                  std::fabs(rgb[size_t(f) * 12] - 1.0f) < 1e-6f,
              "gray preview colours follow the working corners");
        UsdGenBrush_StrokeDestroy(stroke);
    }

    // ---- smooth softens a hard corner-only edge -----------------------------
    {
        // Rows j <= 1 at 1.0, rows j >= 2 at 0.0: a hard edge along z = 2.
        std::vector<float> base(size_t(F) * 4);
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i)
                for (int k = 0; k < 4; ++k) base[size_t(plane.Face(i, j)) * 4 + k] = j <= 1 ? 1.0f : 0.0f;
        void *stroke = UsdGenBrush_StrokeCreate(mesh, F, 16, 1, base.data(), 1.0f);
        // One stroke along the edge, on the upper rows' v = 1 border.
        UsdGenBrush_StrokeDab(stroke, plane.Face(0, 1), 0.5f, 1.0f, 1.0f, 0.5f, 1.0f, 0.0f, 0,
                              USDGEN_BRUSH_MODE_SMOOTH, USDGEN_BRUSH_FALLOFF_SMOOTH, 0, 0.5f);
        for (int i = 1; i < 4; ++i)
            UsdGenBrush_StrokeDab(stroke, plane.Face(i, 1), 0.5f, 1.0f, 1.0f, 0.5f, 1.0f, 0.0f, 0,
                                  USDGEN_BRUSH_MODE_SMOOTH, USDGEN_BRUSH_FALLOFF_SMOOTH, 0, 0.5f);
        std::vector<float> c = Corners(stroke, F);
        bool moved = true;
        float minMove = 1.0f;
        for (int i = 1; i < 3; ++i) {
            float const upper = c[size_t(plane.Face(i, 1)) * 4 + 3];  // (0,1) corner on z = 2
            float const lower = c[size_t(plane.Face(i, 2)) * 4 + 0];  // (0,0) corner on z = 2
            minMove = std::min(minMove, std::min(1.0f - upper, lower));
            moved = moved && (1.0f - upper) > 0.15f && lower > 0.15f;
        }
        Check(moved, "one smooth stroke moves both sides of the edge by > 0.15 (min " +
                         std::to_string(minMove) + ")");
        std::vector<float> committed(c.size());
        UsdGenBrush_StrokeCommitCorners(stroke, committed.data());
        Check(committed == c, "smooth commits exactly what the working grid shows");
        UsdGenBrush_StrokeDestroy(stroke);
    }

    // ---- overlapping set stamps keep their falloff --------------------------
    {
        // A 12 x 12 plane; one straight stroke along the z = 6 vertex row,
        // world radius 3 faces, strength 1, smooth falloff, hardness 0: every
        // texel under the trail is hit by many overlapping stamps. Before the
        // max-weight model each stamp re-applied Set and the band saturated
        // to 1 (a hard disc); now the corner value depends only on the
        // strongest stamp, i.e. on the distance from the centreline.
        Plane const wide = MakePlane(12);
        int const WF = 144;
        void *wideMesh = UsdGenBrush_MeshCreate(wide.points.data(), int(wide.points.size() / 3),
                                                wide.indices.data(), WF);
        void *stroke = UsdGenBrush_StrokeCreate(wideMesh, WF, 8, 1, nullptr, 0.0f);
        for (int i = 2; i < 10; ++i) {
            UsdGenBrush_StrokeDab(stroke, wide.Face(i, 5), 0.0f, 1.0f, 3.0f, 0.0f, 1.0f, 1.0f, 0,
                                  USDGEN_BRUSH_MODE_SET, USDGEN_BRUSH_FALLOFF_SMOOTH, 0, 0.5f);
            UsdGenBrush_StrokeDab(stroke, wide.Face(i, 5), 1.0f, 1.0f, 3.0f, 0.0f, 1.0f, 1.0f, 0,
                                  USDGEN_BRUSH_MODE_SET, USDGEN_BRUSH_FALLOFF_SMOOTH, 1, 0.25f);
        }
        std::vector<float> c = Corners(stroke, WF);
        // Corner 0 of face (6, j) sits at x = 6, z = j: distance j - 6.
        float level[4];
        for (int d = 0; d < 4; ++d) level[d] = c[size_t(wide.Face(6, 6 + d)) * 4];
        std::printf("falloff band (distance 0..3 from the stroke): %.4f %.4f %.4f %.4f\n",
                    level[0], level[1], level[2], level[3]);
        bool monotonic = level[0] > level[1] && level[1] > level[2] && level[2] > level[3];
        Check(std::fabs(level[0] - 1.0f) < 1e-6f && monotonic && level[1] < 0.9f &&
                  level[2] > 0.05f && level[3] == 0.0f,
              "a strength-1 stroke keeps its falloff across the band (4 distinct levels)");
        // The same distance on the other side mirrors it.
        Check(std::fabs(c[size_t(wide.Face(6, 5 - 1)) * 4] - level[2]) < 1e-5f,
              "the band is symmetric about the centreline");
        std::vector<float> committed(c.size());
        UsdGenBrush_StrokeCommitCorners(stroke, committed.data());
        Check(committed == c, "the overlapping stroke commits exactly the working grid");
        // Erase in the same stroke is its own segment and still falls off.
        UsdGenBrush_StrokeDab(stroke, wide.Face(6, 5), 0.5f, 1.0f, 2.0f, 0.5f, 1.0f, 0.0f, 0,
                              USDGEN_BRUSH_MODE_ERASE, USDGEN_BRUSH_FALLOFF_LINEAR, 0, 0.5f);
        std::vector<float> e = Corners(stroke, WF);
        float const erasedCentre = e[size_t(wide.Face(6, 6)) * 4];   // (6, 6): 0.5 away
        float const erasedRing = e[size_t(wide.Face(6, 7)) * 4];     // (6, 7): 1.12 away
        Check(erasedCentre == 0.0f && erasedRing > 0.0f && erasedRing < level[1],
              "an erase segment after the set clears the inner radius and falls off (" +
                  std::to_string(erasedRing) + ")");
        UsdGenBrush_StrokeCommitCorners(stroke, committed.data());
        Check(committed == e, "set + erase segments commit exactly the working grid");
        UsdGenBrush_StrokeDestroy(stroke);
        UsdGenBrush_MeshDestroy(wideMesh);
    }

    // ---- face mask: a GeomSubset's faces are the only paintable ones --------
    {
        // Columns i <= 1 of the 4 x 4 plane are the subset.
        std::vector<int> mask;
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i <= 1; ++i) mask.push_back(plane.Face(i, j));
        mask.push_back(plane.Face(0, 0));  // duplicates are harmless
        int const P = int(plane.points.size() / 3);
        void *masked = UsdGenBrush_MeshCreateMasked(plane.points.data(), P, plane.indices.data(),
                                                    F, mask.data(), int(mask.size()));
        Check(masked != nullptr && UsdGenBrush_MeshFaceCount(masked) == F,
              "a masked mesh keeps every parent face");
        Check(UsdGenBrush_MeshFaceInMask(masked, plane.Face(1, 2)) == 1 &&
                  UsdGenBrush_MeshFaceInMask(masked, plane.Face(2, 2)) == 0 &&
                  UsdGenBrush_MeshFaceInMask(mesh, plane.Face(2, 2)) == 1 &&
                  UsdGenBrush_MeshFaceInMask(masked, F) < 0,
              "MeshFaceInMask reports the mask (an unmasked mesh allows all)");
        int const outside[1] = {F};
        Check(UsdGenBrush_MeshCreateMasked(plane.points.data(), P, plane.indices.data(), F,
                                           outside, 1) == nullptr &&
                  UsdGenBrush_MeshCreateMasked(plane.points.data(), P, plane.indices.data(), F,
                                               mask.data(), -1) == nullptr,
              "an out-of-range mask id or a negative count is rejected");
        void *none = UsdGenBrush_MeshCreateMasked(plane.points.data(), P, plane.indices.data(),
                                                  F, mask.data(), 0);
        Check(none != nullptr && UsdGenBrush_MeshFaceInMask(none, 0) == 0,
              "an empty mask masks every face out");
        UsdGenBrush_MeshDestroy(none);

        double const d[3] = {0.0, -1.0, 0.0};
        int face = -1;
        float u = 0, v = 0;
        double p[3];
        double const inside[3] = {1.5, 2.0, 1.5};
        double const off[3] = {2.25, 2.0, 1.75};
        Check(UsdGenBrush_MeshPick(masked, inside, d, &face, &u, &v, p) == 1 &&
                  face == plane.Face(1, 1),
              "a pick inside the mask hits");
        Check(UsdGenBrush_MeshPick(masked, off, d, &face, &u, &v, p) == 0,
              "a pick on a masked face is a miss");
        {
            // Two stacked planes: the upper one (y = 1) is masked out and
            // must still occlude the paintable lower one.
            Plane two = MakePlane(2);
            int const half = int(two.points.size() / 3);
            std::vector<double> pts = two.points;
            for (int k = 0; k < half; ++k) {
                pts.push_back(two.points[size_t(k) * 3]);
                pts.push_back(1.0);
                pts.push_back(two.points[size_t(k) * 3 + 2]);
            }
            std::vector<int> idx = two.indices;
            for (int k : two.indices) idx.push_back(k + half);
            int const lower[4] = {0, 1, 2, 3};
            void *stack = UsdGenBrush_MeshCreateMasked(pts.data(), 2 * half, idx.data(), 8,
                                                       lower, 4);
            double const above[3] = {0.5, 5.0, 0.5};
            double const below[3] = {0.5, -5.0, 0.5};
            double const up[3] = {0.0, 1.0, 0.0};
            Check(stack && UsdGenBrush_MeshPick(stack, above, d, &face, &u, &v, p) == 0,
                  "a masked face in front occludes (miss, not a paint-through)");
            Check(stack && UsdGenBrush_MeshPick(stack, below, up, &face, &u, &v, p) == 1 &&
                      face == 0,
                  "the unoccluded side of the subset still picks");
            UsdGenBrush_MeshDestroy(stack);
        }

        int faces[64];
        float us[64], vs[64], rs[64];
        int const n = UsdGenBrush_MeshFootprint(masked, plane.Face(1, 1), 0.95f, 0.5f, nullptr,
                                                0.6f, faces, us, vs, rs, 64);
        bool clean = n >= 2 && faces[0] == plane.Face(1, 1);
        bool spilled = false;
        for (int i = 1; i < n; ++i) {
            clean = clean && UsdGenBrush_MeshFaceInMask(masked, faces[i]) == 1;
            spilled = spilled || faces[i] == plane.Face(1, 2);
        }
        Check(clean && spilled,
              "the footprint spills onto subset neighbours only (" + std::to_string(n) +
                  " faces)");

        // A hard set dab right on the subset border: masked faces stay at the
        // base, and only subset faces come back touched.
        void *stroke = UsdGenBrush_StrokeCreate(masked, F, 8, 1, nullptr, 0.0f);
        Check(UsdGenBrush_StrokeDab(stroke, plane.Face(2, 1), 0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f,
                                    0, USDGEN_BRUSH_MODE_SET, USDGEN_BRUSH_FALLOFF_SMOOTH, 0,
                                    0.5f) == -11 &&
                  UsdGenBrush_StrokeDabCount(stroke) == 0,
              "a dab on a masked face is rejected (-11), nothing recorded");
        UsdGenBrush_StrokeDab(stroke, plane.Face(1, 1), 1.0f, 0.5f, 1.5f, 1.0f, 1.0f, 1.0f, 0,
                              USDGEN_BRUSH_MODE_SET, USDGEN_BRUSH_FALLOFF_SMOOTH, 0, 0.5f);
        UsdGenBrush_StrokeDab(stroke, plane.Face(1, 2), 1.0f, 0.9f, 1.5f, 1.0f, 1.0f, 1.0f, 0,
                              USDGEN_BRUSH_MODE_SET, USDGEN_BRUSH_FALLOFF_SMOOTH, 1, 0.5f);
        std::vector<float> c = Corners(stroke, F);
        bool outsideZero = true, insidePainted = false;
        for (int f = 0; f < F; ++f)
            for (int k = 0; k < 4; ++k) {
                if (UsdGenBrush_MeshFaceInMask(masked, f) == 0)
                    outsideZero = outsideZero && c[size_t(f) * 4 + size_t(k)] == 0.0f;
                else
                    insidePainted = insidePainted || c[size_t(f) * 4 + size_t(k)] == 1.0f;
            }
        Check(outsideZero && insidePainted,
              "a border set dab paints subset corners and never a masked face");
        std::vector<int> touched(64, -1);
        int const nTouched = UsdGenBrush_StrokeTakeTouched(stroke, touched.data(), 64);
        bool touchedInside = nTouched > 0;
        for (int i = 0; i < nTouched; ++i)
            touchedInside = touchedInside && UsdGenBrush_MeshFaceInMask(masked, touched[size_t(i)]) == 1;
        Check(touchedInside, "only subset faces come back touched");
        std::vector<float> committed(c.size());
        UsdGenBrush_StrokeCommitCorners(stroke, committed.data());
        Check(committed == c, "a masked stroke commits exactly the working grid");
        UsdGenBrush_StrokeDestroy(stroke);

        // Smooth along the subset border: subset = 1, outside = 0. The masked
        // smoother averages paintable faces only, so the border corners keep
        // 1 and the outside keeps 0; the unmasked one pulls both across.
        std::vector<float> base(size_t(F) * 4);
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i)
                for (int k = 0; k < 4; ++k) base[size_t(plane.Face(i, j)) * 4 + k] = i <= 1 ? 1.0f : 0.0f;
        auto smoothBorder = [&](void *m) {
            void *s = UsdGenBrush_StrokeCreate(m, F, 8, 1, base.data(), 1.0f);
            for (int j = 0; j < 4; ++j)
                UsdGenBrush_StrokeDab(s, plane.Face(1, j), 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.0f, 0,
                                      USDGEN_BRUSH_MODE_SMOOTH, USDGEN_BRUSH_FALLOFF_SMOOTH, 0,
                                      0.5f);
            std::vector<float> out = Corners(s, F);
            std::vector<float> again(out.size());
            UsdGenBrush_StrokeCommitCorners(s, again.data());
            UsdGenBrush_StrokeDestroy(s);
            return std::make_pair(out, again == out);
        };
        auto const maskedSmooth = smoothBorder(masked);
        auto const openSmooth = smoothBorder(mesh);
        float const borderMasked = maskedSmooth.first[size_t(plane.Face(1, 1)) * 4 + 1];
        float const borderOpen = openSmooth.first[size_t(plane.Face(1, 1)) * 4 + 1];
        bool outsideKept = true;
        for (int j = 0; j < 4; ++j)
            for (int i = 2; i < 4; ++i)
                for (int k = 0; k < 4; ++k)
                    outsideKept = outsideKept &&
                                  maskedSmooth.first[size_t(plane.Face(i, j)) * 4 + size_t(k)] == 0.0f;
        Check(outsideKept && borderMasked == 1.0f && borderOpen < 0.9f && maskedSmooth.second,
              "masked smooth never writes outside faces nor pulls their paint in (border " +
                  std::to_string(borderMasked) + " masked, " + std::to_string(borderOpen) +
                  " open)");
        UsdGenBrush_MeshDestroy(masked);
    }

    // ---- per-move cost on a 100k-face mesh ---------------------------------
    {
        Plane const big = MakePlane(317);  // 100,489 faces
        int const bigF = big.n * big.n;
        auto t0 = std::chrono::steady_clock::now();
        void *bigMesh = UsdGenBrush_MeshCreate(big.points.data(), int(big.points.size() / 3),
                                               big.indices.data(), bigF);
        auto t1 = std::chrono::steady_clock::now();
        char info[256] = {0};
        int const res = UsdGenBrush_MeshSuggestResolution(bigMesh, 4000000, info, 256);
        void *stroke = UsdGenBrush_StrokeCreate(bigMesh, bigF, res, 1, nullptr, 1.0f);
        auto t2 = std::chrono::steady_clock::now();
        double const o[3] = {150.3, 10.0, 150.6};
        double const d[3] = {0.0, -1.0, 0.0};
        int face = -1;
        float u = 0, v = 0;
        double p[3];
        std::vector<float> corners(size_t(bigF) * 4), rgb(size_t(bigF) * 12);
        std::vector<int> touched(4096);
        int const moves = 50;
        auto t3 = std::chrono::steady_clock::now();
        for (int k = 0; k < moves; ++k) {
            double const oo[3] = {o[0] + 0.1 * k, o[1], o[2]};
            UsdGenBrush_MeshPick(bigMesh, oo, d, &face, &u, &v, p);
            UsdGenBrush_StrokeDab(stroke, face, u, v, 2.0f, 0.5f, 0.5f, 0.0f, 0,
                                  USDGEN_BRUSH_MODE_SET, USDGEN_BRUSH_FALLOFF_SMOOTH, 1, 0.5f);
            UsdGenBrush_StrokeTakeTouched(stroke, touched.data(), 4096);
            UsdGenBrush_StrokeWorkingCorners(stroke, corners.data());
            UsdGenBrush_StrokePreviewColors(stroke, 0, 0, 0.0f, 1.0f, rgb.data());
        }
        auto t4 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::printf("perf: 100k faces (%s): mesh %.1f ms, stroke create %.1f ms, "
                    "per move (pick+dab+corners+colours) %.2f ms\n",
                    info, ms(t0, t1), ms(t1, t2), ms(t3, t4) / moves);
        Check(bigMesh && stroke && face >= 0, "the 100k-face mesh picks and paints");
        UsdGenBrush_StrokeDestroy(stroke);
        UsdGenBrush_MeshDestroy(bigMesh);
    }

    // ---- the overlay entry points reach an attached index -------------------
    {
        UsdStageRefPtr stage = UsdStage::CreateInMemory("brush-api-preview");
        stage->GetRootLayer()->ImportFromString(R"USDA(#usda 1.0
def Mesh "Scalp"
{
    int[] faceVertexCounts = [4, 4]
    int[] faceVertexIndices = [0, 1, 4, 3, 1, 2, 5, 4]
    point3f[] points = [(-1,0,-1), (0,0,-1), (1,0,-1), (-1,0,1), (0,0,1), (1,0,1)]
}
)USDA");
        UsdImagingCreateSceneIndicesInfo sceneInfo;
        sceneInfo.stage = stage;
        UsdImagingSceneIndices indices = UsdImagingCreateSceneIndices(sceneInfo);
        int const before = UsdGenBrush_PreviewIndexCount();
        auto preview = usdGenImaging::UsdGenAttributePreviewSceneIndex::New(
            indices.finalSceneIndex, true);
        Check(UsdGenBrush_PreviewIndexCount() == before + 1,
              "an attached preview index registers globally");
        DirtyRecorder recorder;
        preview->AddObserver(TfCreateWeakPtr(&recorder));
        SdfPath const scalp("/Scalp");
        HdDataSourceLocator const structural(TfToken("primvars"), TfToken("displayColor"));
        HdDataSourceLocator const values(TfToken("primvars"), TfToken("displayColor"),
                                         TfToken("primvarValue"));
        std::vector<float> rgb(8 * 3, 0.0f);
        for (int i = 0; i < 4; ++i) rgb[size_t(4 + i) * 3] = 1.0f;  // face 1 red
        Check(UsdGenBrush_PreviewSet("/Scalp", rgb.data(), 8) == 1, "PreviewSet accepts");
        TfToken interp;
        VtVec3fArray colors = DisplayColors(*preview, scalp, &interp);
        Check(colors.size() == 8 && interp == TfToken("faceVarying") && colors[4][0] == 1.0f &&
                  colors[0][0] == 0.0f,
              "the overlay serves faceVarying colours");
        Check(recorder.CountFor(scalp, structural) == 1 && recorder.CountFor(scalp, values) == 0,
              "the first Set dirties primvars/displayColor");
        recorder.dirtied.clear();
        rgb[0] = 0.5f;
        UsdGenBrush_PreviewSet("/Scalp", rgb.data(), 8);
        Check(recorder.dirtied.size() == 1 && recorder.CountFor(scalp, values) == 1 &&
                  recorder.CountFor(scalp, structural) == 0,
              "a re-Set dirties primvarValue only");
        Check(DisplayColors(*preview, scalp, &interp)[0][0] == 0.5f, "the re-Set lands");
        UsdGenBrush_PreviewSet("/Scalp", rgb.data(), 4);
        Check(DisplayColors(*preview, scalp, &interp).empty(),
              "a mis-sized overlay is withheld from the mesh");
        recorder.dirtied.clear();
        Check(UsdGenBrush_PreviewClear("/Scalp") == 1 && UsdGenBrush_PreviewClear("/Scalp") == 0,
              "PreviewClear removes once");
        Check(DisplayColors(*preview, scalp, &interp).empty() &&
                  recorder.CountFor(scalp, structural) == 1,
              "clearing restores upstream with a structural dirty");
        UsdGenBrush_PreviewSet("/Scalp", rgb.data(), 8);
        recorder.dirtied.clear();
        Check(UsdGenBrush_PreviewClearAll() == 1 && UsdGenBrush_PreviewClearAll() == 0,
              "PreviewClearAll removes every overlay once");
        Check(DisplayColors(*preview, scalp, &interp).empty() &&
                  recorder.CountFor(scalp, structural) == 1,
              "and restores upstream with a structural dirty");
        preview->RemoveObserver(TfCreateWeakPtr(&recorder));
        preview.Reset();
        Check(UsdGenBrush_PreviewIndexCount() == before, "a destroyed index detaches");
    }

    UsdGenBrush_MeshDestroy(mesh);
    std::printf("testUsdGenBrushApi: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
