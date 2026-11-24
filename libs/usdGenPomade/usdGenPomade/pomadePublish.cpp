// usdGenPomade — per-level publication staging implementation (plan/18 §2.2).
#include "usdGenPomade/pomadePublish.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace usdGenPomade {

namespace {

// -- hashing ---------------------------------------------------------------
//
// FNV-1a over raw bytes. Only equality matters: two equal hashes mean the
// staged slice is reused, so the mix has to see every input the tessellation
// reads and nothing it does not.

constexpr uint64_t kHashSeed = 0xCBF29CE484222325ull;

uint64_t _HashBytes(uint64_t h, void const *data, size_t bytes)
{
    unsigned char const *p = static_cast<unsigned char const *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        h ^= uint64_t(p[i]);
        h *= 0x100000001B3ull;
    }
    return h;
}

uint64_t _HashInt(uint64_t h, int v)
{
    return _HashBytes(h, &v, sizeof(v));
}

uint64_t _HashFloat(uint64_t h, float v)
{
    // Normalise the two zeroes so a -0 never restages a tube.
    if (v == 0.0f) {
        v = 0.0f;
    }
    return _HashBytes(h, &v, sizeof(v));
}

uint64_t _HashFloats(uint64_t h, std::vector<float> const &v)
{
    h = _HashInt(h, int(v.size()));
    for (float f : v) {
        h = _HashFloat(h, f);
    }
    return h;
}

// -- small geometry helpers ------------------------------------------------

// Each polygon edge sweeps a longitudinal strip. Share its normals along
// the length, including across sparse control rings, but never with the
// neighbouring strip: those boundaries are the selectable CV rails.
// Face-varying normals retain the shared mesh/pick topology and describe
// the actual taper, bend and twist instead of a radial cylinder normal.
void _FillStripNormals(VtVec3fArray const &points, int base, int rings,
                       int rv, int faceOffset, VtVec3fArray *normals)
{
    std::vector<GfVec3f> faces(size_t((rings - 1) * rv));
    auto unit = [](GfVec3f n) {
        float const length = n.GetLength();
        return length > 1e-12f ? n / length : GfVec3f(0.0f);
    };
    for (int r = 0; r < rings - 1; ++r) {
        for (int s = 0; s < rv; ++s) {
            int const next = (s + 1) % rv;
            GfVec3f const &a = points[size_t(base + r * rv + s)];
            GfVec3f const &b = points[size_t(base + r * rv + next)];
            GfVec3f const &c = points[size_t(base + (r + 1) * rv + next)];
            GfVec3f const &d = points[size_t(base + (r + 1) * rv + s)];
            faces[size_t(r * rv + s)] =
                unit(GfCross(b - a, d - a) + GfCross(c - b, d - b));
        }
    }
    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < rv; ++s) {
            GfVec3f n(0.0f);
            if (r > 0) n += faces[size_t((r - 1) * rv + s)];
            if (r + 1 < rings) n += faces[size_t(r * rv + s)];
            n = unit(n);
            if (n.GetLengthSq() < 1e-12f) {
                // Collapsed or folded strips must still publish finite,
                // nonzero normals; prefer either incident nonzero face.
                n = faces[size_t(std::min(r, rings - 2) * rv + s)];
                if (n.GetLengthSq() < 1e-12f && r > 0)
                    n = faces[size_t((r - 1) * rv + s)];
                if (n.GetLengthSq() < 1e-12f) n = GfVec3f(0, 1, 0);
            }
            if (r + 1 < rings) {
                size_t const q = size_t(faceOffset + r * rv + s) * 4;
                (*normals)[q] = (*normals)[q + 1] = n;
            }
            if (r > 0) {
                size_t const q = size_t(faceOffset + (r - 1) * rv + s) * 4;
                (*normals)[q + 2] = (*normals)[q + 3] = n;
            }
        }
    }
}

void _Extend(GfVec3f *mn, GfVec3f *mx, GfVec3f const &p)
{
    for (int a = 0; a < 3; ++a) {
        (*mn)[a] = std::min((*mn)[a], p[a]);
        (*mx)[a] = std::max((*mx)[a], p[a]);
    }
}

void _Bounds(VtVec3fArray const &points, GfVec3f *mn, GfVec3f *mx)
{
    *mn = GfVec3f(0.0f);
    *mx = GfVec3f(0.0f);
    for (size_t i = 0; i < points.size(); ++i) {
        if (i == 0) {
            *mn = *mx = points[0];
        } else {
            _Extend(mn, mx, points[i]);
        }
    }
}

GfVec3f _Rgb(PomadeRgb const &c)
{
    return GfVec3f(c.r, c.g, c.b);
}

// Mean distance from the root ring's centroid to its verts: the tube's own
// scale. Every overlay width is expressed in it, so a groom authored in
// centimetres and one authored in metres both read correctly. (Screen-space
// widths need the camera, which arrives with the V2 controller; until then
// the overlays scale with the geometry, not with the zoom.)
float _RootRadius(std::vector<float> const &positions, int pointOffset,
                  int ringVerts)
{
    if (ringVerts <= 0) {
        return 0.5f;
    }
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (int s = 0; s < ringVerts; ++s) {
        size_t const o = size_t(pointOffset + s) * 3;
        cx += positions[o + 0];
        cy += positions[o + 1];
        cz += positions[o + 2];
    }
    cx /= ringVerts;
    cy /= ringVerts;
    cz /= ringVerts;
    double sum = 0.0;
    for (int s = 0; s < ringVerts; ++s) {
        size_t const o = size_t(pointOffset + s) * 3;
        double const dx = positions[o + 0] - cx;
        double const dy = positions[o + 1] - cy;
        double const dz = positions[o + 2] - cz;
        sum += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    float const r = float(sum / ringVerts);
    return r > 1e-6f ? r : 0.5f;
}

GfVec3f _Point(std::vector<float> const &xyz, int vertex)
{
    size_t const o = size_t(vertex) * 3;
    return GfVec3f(xyz[o + 0], xyz[o + 1], xyz[o + 2]);
}

// The overlay widths of one level. They depend on the focus level, the
// display scale and each tube's own radius, and on nothing else — so a
// focus change or a camera move re-fills four small arrays and dirties
// four leaves, never a point buffer.
void _FillWidths(PomadeStagedLevel *staged)
{
    bool const focused = staged->focused;
    float const scale = staged->displayScale;
    staged->centerCurveWidth.resize(staged->tubes.size());
    staged->ringCurveWidth.clear();
    staged->centerCVWidth.clear();
    staged->ringCVWidth.clear();
    for (size_t t = 0; t < staged->tubes.size(); ++t) {
        PomadeTubeSlice const &slice = staged->tubes[t];
        float const r = slice.radius;
        staged->centerCurveWidth[t] = PomadeOverlayWidth(
            focused ? PomadeOverlayPixels::kCenterCurveFocused
                    : PomadeOverlayPixels::kCenterCurveOther,
            scale, r * (focused ? 0.16f : 0.06f));
        float const cvWidth = PomadeOverlayWidth(
            focused ? PomadeOverlayPixels::kCenterCVFocused
                    : PomadeOverlayPixels::kCenterCVOther,
            scale, r * (focused ? 0.34f : 0.20f));
        for (int c = 0; c < slice.centerCount; ++c) {
            staged->centerCVWidth.push_back(cvWidth);
        }
        float const ringWidth = PomadeOverlayWidth(
            PomadeOverlayPixels::kRingCurve, scale, r * 0.05f);
        float const ringCVWidth = PomadeOverlayWidth(
            PomadeOverlayPixels::kRingCV, scale,
            r * (focused ? 0.22f : 0.14f));
        for (int k = 0; k < slice.ringCurveCount; ++k) {
            staged->ringCurveWidth.push_back(ringWidth);
            for (int s = 0; s < slice.ringVerts; ++s) {
                staged->ringCVWidth.push_back(ringCVWidth);
            }
        }
    }
}

// The ring overlay of one level, built from the same staged surface grid the
// mesh uses: a ring curve IS the grid row its section sits on, so the rings
// an artist drags can never drift from the surface they bound. Kept apart
// from _FillLevelArrays because ring visibility (plan/18 §2.4a) changes with
// the selection and the active sub-mode, and neither may restage anything
// else.
void _FillRingArrays(std::vector<float> const &positions,
                     PomadeStagedLevel *staged)
{
    int ringCurveTotal = 0;
    int ringPointTotal = 0;
    int ringCVTotal = 0;
    for (PomadeTubeSlice const &s : staged->tubes) {
        ringCurveTotal += s.ringCurveCount;
        ringPointTotal += s.ringCurveCount * (s.ringVerts + 1);
        ringCVTotal += s.ringCurveCount * s.ringVerts;
    }
    staged->ringPoints.resize(size_t(ringPointTotal));
    staged->ringIndices.resize(size_t(ringPointTotal));
    staged->ringVertexCounts.resize(size_t(ringCurveTotal));
    staged->ringCurveColor.resize(size_t(ringCurveTotal));
    staged->ringCurveTubeId.resize(size_t(ringCurveTotal));
    staged->ringCurveSection.resize(size_t(ringCurveTotal));
    staged->ringCurveSelected.assign(size_t(ringCurveTotal), 0);
    staged->ringCVPoints.resize(size_t(ringCVTotal));
    staged->ringCVColor.resize(size_t(ringCVTotal));
    staged->ringCVTubeId.resize(size_t(ringCVTotal));
    staged->ringCVSection.resize(size_t(ringCVTotal));

    int ringPointCursor = 0;
    int ringCVCursor = 0;
    int ringCurveCursor = 0;
    for (PomadeTubeSlice &slice : staged->tubes) {
        int const rv = slice.ringVerts;
        slice.ringCurveOffset = ringCurveCursor;
        for (int k = 0; k < slice.ringCurveCount; ++k) {
            int const row =
                std::min(k * slice.ringStride, slice.ringCount - 1);
            size_t const curve = size_t(ringCurveCursor++);
            staged->ringVertexCounts[curve] = rv + 1;
            staged->ringCurveColor[curve] = slice.clumpColor;
            staged->ringCurveTubeId[curve] = slice.tubeId;
            staged->ringCurveSection[curve] = k;
            for (int s = 0; s <= rv; ++s) {
                int const vertex = slice.pointOffset + row * rv + (s % rv);
                staged->ringPoints[size_t(ringPointCursor)] =
                    _Point(positions, vertex);
                staged->ringIndices[size_t(ringPointCursor)] = ringPointCursor;
                ++ringPointCursor;
            }
            for (int s = 0; s < rv; ++s) {
                int const vertex = slice.pointOffset + row * rv + s;
                staged->ringCVPoints[size_t(ringCVCursor)] =
                    _Point(positions, vertex);
                staged->ringCVColor[size_t(ringCVCursor)] = slice.clumpColor;
                staged->ringCVTubeId[size_t(ringCVCursor)] = slice.tubeId;
                staged->ringCVSection[size_t(ringCVCursor)] = k;
                ++ringCVCursor;
            }
        }
    }
    _Bounds(staged->ringPoints, &staged->ringMin, &staged->ringMax);
}

// Two nearby rails bracket every authored longitudinal corner. Four straight
// spans fill the broad side between them without bowing the cross-section.
// Add a holding row at each end even at the lowest display density.
// Pin every end vertex: boundary interpolation alone can shrink a curved
// root polygon away from its surface attachment.
void _FillSubdivisionCage(PomadeStagedLevel *level)
{
    auto &cage = level->subdivision;
    constexpr float hold = 0.05f;
    constexpr float radial[] = {0.0f, hold, 0.275f, 0.5f, 0.725f, 1.0f - hold};
    constexpr int samples = int(sizeof(radial) / sizeof(radial[0]));
    size_t endVertices = 0;
    for (auto const &tube : level->tubes)
        endVertices += size_t(2 * samples * tube.ringVerts);
    cage.points.reserve(level->points.size() * samples + endVertices);
    size_t const faces = level->faceVertexCounts.size() * samples + endVertices;
    cage.faceVertexCounts.reserve(faces);
    cage.faceVertexIndices.reserve(faces * 4);
    cage.sourceFaces.reserve(faces);
    cage.cornerIndices.reserve(endVertices);
    cage.cornerSharpnesses.reserve(endVertices);
    for (auto const &tube : level->tubes) {
        int const rv = tube.ringVerts;
        int const columns = samples * rv;
        int const rows = tube.ringCount + 2;
        int const base = int(cage.points.size());
        for (int r = 0; r < rows; ++r) {
            int row = std::max(0, std::min(r - 1, tube.ringCount - 1));
            float blend = 0.0f;
            if (r == 1) {
                blend = hold;
            } else if (r == rows - 2) {
                row = tube.ringCount - 2;
                blend = 1.0f - hold;
            }
            auto point = [&](int s) {
                int const at = tube.pointOffset + row * rv + s;
                GfVec3f const p = level->points[size_t(at)];
                return blend == 0.0f ? p :
                    p * (1.0f - blend) + level->points[size_t(at + rv)] * blend;
            };
            for (int s = 0; s < rv; ++s) {
                GfVec3f const a = point(s), b = point((s + 1) % rv);
                for (float f : radial) {
                    if (r == 0 || r == rows - 1) {
                        cage.cornerIndices.push_back(int(cage.points.size()));
                        cage.cornerSharpnesses.push_back(10.0f); // infinite in OpenSubdiv
                    }
                    cage.points.push_back(a * (1.0f - f) + b * f);
                }
            }
        }
        for (int r = 0; r < rows - 1; ++r) {
            int const sourceRow = std::max(0, std::min(r - 1, tube.ringCount - 2));
            for (int c = 0; c < columns; ++c) {
                int const next = (c + 1) % columns;
                cage.faceVertexCounts.push_back(4);
                cage.faceVertexIndices.push_back(base + r * columns + c);
                cage.faceVertexIndices.push_back(base + r * columns + next);
                cage.faceVertexIndices.push_back(base + (r + 1) * columns + next);
                cage.faceVertexIndices.push_back(base + (r + 1) * columns + c);
                cage.sourceFaces.push_back(tube.faceOffset + sourceRow * rv + c / samples);
            }
        }
    }
}

// Build every Hydra array of one level from the host mirror. `reuse` is the
// previous snapshot when nothing about this level's geometry changed, in
// which case the arrays are shared (VtArray is refcounted) and only the
// widths are re-derived.
void _FillLevelArrays(std::vector<float> const &positions,
                      std::vector<PomadeTubeDesc const *> const &descs,
                      PomadeStagedLevel const *reuse, bool reuseRings,
                      PomadeStagedLevel *staged)
{
    if (reuse && reuse->tubes.size() == staged->tubes.size()) {
        for (size_t t = 0; t < staged->tubes.size(); ++t) {
            staged->tubes[t].radius = reuse->tubes[t].radius;
        }
        staged->points = reuse->points;
        staged->subdivision = reuse->subdivision;
        staged->faceVertexCounts = reuse->faceVertexCounts;
        staged->faceVertexIndices = reuse->faceVertexIndices;
        staged->faceTubeId = reuse->faceTubeId;
        staged->faceClumpColor = reuse->faceClumpColor;
        staged->faceSelected = reuse->faceSelected;
        staged->extentMin = reuse->extentMin;
        staged->extentMax = reuse->extentMax;
        staged->centerPoints = reuse->centerPoints;
        staged->centerVertexCounts = reuse->centerVertexCounts;
        staged->centerIndices = reuse->centerIndices;
        staged->centerCurveColor = reuse->centerCurveColor;
        staged->centerCurveTubeId = reuse->centerCurveTubeId;
        staged->centerCurveSelected = reuse->centerCurveSelected;
        staged->centerCVColor = reuse->centerCVColor;
        staged->centerCVTubeId = reuse->centerCVTubeId;
        staged->centerCVIndex = reuse->centerCVIndex;
        staged->centerMin = reuse->centerMin;
        staged->centerMax = reuse->centerMax;
        if (reuseRings) {
            for (size_t t = 0; t < staged->tubes.size(); ++t) {
                staged->tubes[t].ringCurveOffset =
                    reuse->tubes[t].ringCurveOffset;
            }
            staged->ringPoints = reuse->ringPoints;
            staged->ringVertexCounts = reuse->ringVertexCounts;
            staged->ringIndices = reuse->ringIndices;
            staged->ringCurveColor = reuse->ringCurveColor;
            staged->ringCurveTubeId = reuse->ringCurveTubeId;
            staged->ringCurveSection = reuse->ringCurveSection;
            staged->ringCurveSelected = reuse->ringCurveSelected;
            staged->ringCVPoints = reuse->ringCVPoints;
            staged->ringCVColor = reuse->ringCVColor;
            staged->ringCVTubeId = reuse->ringCVTubeId;
            staged->ringCVSection = reuse->ringCVSection;
            staged->ringMin = reuse->ringMin;
            staged->ringMax = reuse->ringMax;
        } else {
            _FillRingArrays(positions, staged);
        }
        _FillWidths(staged);
        return;
    }

    PomadeTubeSlice const &last = staged->tubes.back();
    int const pointTotal = last.pointOffset + last.pointCount;
    int const faceTotal = last.faceOffset + last.faceCount;
    int const centerTotal = last.centerOffset + last.centerCount;

    staged->points.resize(size_t(pointTotal));
    for (int v = 0; v < pointTotal; ++v) {
        staged->points[size_t(v)] = _Point(positions, v);
    }
    _Bounds(staged->points, &staged->extentMin, &staged->extentMax);

    staged->faceVertexCounts.assign(size_t(faceTotal), 4);
    staged->faceVertexIndices.resize(size_t(faceTotal) * 4);
    staged->faceTubeId.resize(size_t(faceTotal));
    staged->faceClumpColor.resize(size_t(faceTotal));
    staged->faceSelected.assign(size_t(faceTotal), 0);

    staged->centerPoints.resize(size_t(centerTotal));
    staged->centerVertexCounts.resize(staged->tubes.size());
    staged->centerIndices.resize(size_t(centerTotal));
    staged->centerCurveColor.resize(staged->tubes.size());
    staged->centerCurveTubeId.resize(staged->tubes.size());
    staged->centerCurveSelected.assign(staged->tubes.size(), 0);
    staged->centerCVColor.resize(size_t(centerTotal));
    staged->centerCVTubeId.resize(size_t(centerTotal));
    staged->centerCVIndex.resize(size_t(centerTotal));

    for (size_t t = 0; t < staged->tubes.size(); ++t) {
        PomadeTubeSlice &slice = staged->tubes[t];
        PomadeTubeDesc const &desc = *descs[t];
        slice.radius =
            _RootRadius(positions, slice.pointOffset, slice.ringVerts);

        // Quad strip between adjacent grid rows, CCW from outside — the
        // spelling both model tessellation paths use.
        int const rv = slice.ringVerts;
        for (int r = 0; r < slice.ringCount - 1; ++r) {
            for (int s = 0; s < rv; ++s) {
                int const q = slice.faceOffset + r * rv + s;
                int const s1 = (s + 1) % rv;
                int const base = slice.pointOffset;
                staged->faceVertexIndices[size_t(q) * 4 + 0] =
                    base + r * rv + s;
                staged->faceVertexIndices[size_t(q) * 4 + 1] =
                    base + r * rv + s1;
                staged->faceVertexIndices[size_t(q) * 4 + 2] =
                    base + (r + 1) * rv + s1;
                staged->faceVertexIndices[size_t(q) * 4 + 3] =
                    base + (r + 1) * rv + s;
                staged->faceTubeId[size_t(q)] = slice.tubeId;
                staged->faceClumpColor[size_t(q)] = slice.clumpColor;
            }
        }

        staged->centerVertexCounts[t] = slice.centerCount;
        staged->centerCurveColor[t] = slice.clumpColor;
        staged->centerCurveTubeId[t] = slice.tubeId;
        std::vector<PomadeFrame> frames;
        std::string err;
        bool const haveFrames = PomadeTubeFramesCpu(desc, &frames, &err);
        for (int c = 0; c < slice.centerCount; ++c) {
            size_t const o = size_t(slice.centerOffset + c);
            float x = desc.centerX[size_t(c)];
            float y = desc.centerY[size_t(c)];
            float z = desc.centerZ[size_t(c)];
            // The core line and its dots must sit inside the visible tube.
            // A malformed imported section keeps the legacy raw cage point;
            // the model pick path uses the same fallback.
            if (haveFrames)
                PomadeCenterHandlePointCpu(desc, frames, c, &x, &y, &z, &err);
            staged->centerPoints[o] = GfVec3f(x, y, z);
            staged->centerIndices[o] = int(o);
            staged->centerCVColor[o] = slice.clumpColor;
            staged->centerCVTubeId[o] = slice.tubeId;
            staged->centerCVIndex[o] = c;
        }

    }
    _FillSubdivisionCage(staged);
    _Bounds(staged->subdivision.points, &staged->extentMin, &staged->extentMax);
    _Bounds(staged->centerPoints, &staged->centerMin, &staged->centerMax);
    _FillRingArrays(positions, staged);
    _FillWidths(staged);
}

// -- selection (plan/18 §2.3, §2.4a) ---------------------------------------
//
// The selection is model state; publication turns it into three things:
// the `selected` uniform on the level mesh and on the curve overlays
// (0 none / 1 selected / 2 hover, which is what pomadeTube.glslfx reads),
// and white / yellow CV dots. Nothing here re-tessellates: a click changes
// small arrays and one hash, and the index dirties one leaf per prim.

GfVec3f const kSelectedColor(1.0f, 1.0f, 1.0f);
GfVec3f const kHoverColor(1.0f, 0.85f, 0.10f);

// One tube's share of the selection, keyed the way the staged arrays are
// walked so the fill is a lookup per face / curve / CV.
struct _TubeSelection {
    int tube = 0;  // 0 none, 1 selected, 2 hover
    std::map<int, int> centerCVs;                  // cv -> state
    std::map<std::pair<int, int>, int> sectionCVs;  // (ring, slot) -> state
    std::map<int, int> rings;                      // ring -> state
};
using _SelectionByTube = std::map<int, _TubeSelection>;

void _NoteItem(_SelectionByTube *out, PomadeSelectionItem const &item,
               int state, std::map<int, int> const &tubeLevels)
{
    auto noteOwner = [&]() -> _TubeSelection & {
        _TubeSelection &selection = (*out)[item.id];
        // A center, ring, or section CV is an edit handle owned by this
        // tube.  Its local white/yellow feedback remains below, while the
        // mesh gets the same durable rim/lift cue as a body selection.
        // State 2 (hover) wins over a prior selected handle on the owner.
        selection.tube = std::max(selection.tube, state);
        return selection;
    };
    switch (item.kind) {
    case PomadePick_TubeVert:
        (*out)[item.id].tube = state;
        break;
    case PomadePick_Level:
        // A level selection IS every tube at that level (plan/18 §2.3).
        for (auto const &kv : tubeLevels) {
            if (kv.second == item.id) {
                (*out)[kv.first].tube = state;
            }
        }
        break;
    case PomadePick_CenterCV:
        noteOwner().centerCVs[item.subId] = state;
        break;
    case PomadePick_SectionCV:
        noteOwner().sectionCVs[std::make_pair(item.subId, item.subSubId)] =
            state;
        break;
    case PomadePick_SectionRing:
        noteOwner().rings[item.subId] = state;
        break;
    default:
        break;  // graph and guide selection is drawn by their own prims
    }
}

_SelectionByTube _ResolveSelection(std::vector<PomadeSelectionItem> const &items,
                                   PomadeSelectionItem const &hover,
                                   std::map<int, int> const &tubeLevels)
{
    _SelectionByTube out;
    for (PomadeSelectionItem const &item : items) {
        _NoteItem(&out, item, 1, tubeLevels);
    }
    if (hover.kind) {
        _NoteItem(&out, hover, 2, tubeLevels);  // hover wins over selected
    }
    return out;
}

uint64_t _HashTubeSelection(uint64_t h, _TubeSelection const &sel)
{
    h = _HashInt(h, sel.tube);
    h = _HashInt(h, int(sel.centerCVs.size()));
    for (auto const &kv : sel.centerCVs) {
        h = _HashInt(h, kv.first);
        h = _HashInt(h, kv.second);
    }
    h = _HashInt(h, int(sel.sectionCVs.size()));
    for (auto const &kv : sel.sectionCVs) {
        h = _HashInt(h, kv.first.first);
        h = _HashInt(h, kv.first.second);
        h = _HashInt(h, kv.second);
    }
    h = _HashInt(h, int(sel.rings.size()));
    for (auto const &kv : sel.rings) {
        h = _HashInt(h, kv.first);
        h = _HashInt(h, kv.second);
    }
    return h;
}

// Write the selection into one level's arrays. Called only when the
// level's selection hash changed, so the reuse path keeps sharing its
// VtArrays with the previous snapshot in the common case.
void _FillSelection(_SelectionByTube const &selection,
                    PomadeStagedLevel *staged)
{
    _TubeSelection const kNone;
    staged->faceSelected.assign(staged->faceVertexCounts.size(), 0);
    staged->centerCurveSelected.assign(staged->tubes.size(), 0);
    staged->ringCurveSelected.assign(staged->ringCurveSection.size(), 0);
    int ringCVCursor = 0;
    for (size_t t = 0; t < staged->tubes.size(); ++t) {
        PomadeTubeSlice const &slice = staged->tubes[t];
        auto const hit = selection.find(slice.tubeId);
        _TubeSelection const &sel = hit == selection.end() ? kNone
                                                           : hit->second;
        for (int f = 0; f < slice.faceCount; ++f) {
            staged->faceSelected[size_t(slice.faceOffset + f)] = sel.tube;
        }
        staged->centerCurveSelected[t] = sel.tube;
        for (int c = 0; c < slice.centerCount; ++c) {
            size_t const o = size_t(slice.centerOffset + c);
            auto const cv = sel.centerCVs.find(c);
            int const state = cv == sel.centerCVs.end() ? 0 : cv->second;
            staged->centerCVColor[o] = state == 2   ? kHoverColor
                                       : state == 1 ? kSelectedColor
                                                    : slice.clumpColor;
        }
        for (int k = 0; k < slice.ringCurveCount; ++k) {
            size_t const curve = size_t(slice.ringCurveOffset + k);
            auto const ring = sel.rings.find(k);
            int const ringState = ring == sel.rings.end() ? 0 : ring->second;
            staged->ringCurveSelected[curve] = ringState;
            for (int s = 0; s < slice.ringVerts; ++s) {
                auto const cv = sel.sectionCVs.find(std::make_pair(k, s));
                int state = cv == sel.sectionCVs.end() ? 0 : cv->second;
                if (!state) {
                    state = ringState;  // a selected ring lights its CVs
                }
                staged->ringCVColor[size_t(ringCVCursor)] =
                    state == 2   ? kHoverColor
                    : state == 1 ? kSelectedColor
                                 : slice.clumpColor;
                ++ringCVCursor;
            }
        }
    }
}

// The K9/K10 guide preview as the committed Guides spelling: cubic bspline,
// pinned wrap, per-vertex widths and hairT, per-curve clump colour. One
// level's curves at a time: the caller filters the merged preview by the
// level each curve's tube sits at, so every curve keeps its own tube's
// colour and id.
void _FillGuides(std::vector<float> const &points,
                 std::vector<int> const &counts,
                 std::vector<GfVec3f> const &colors,
                 std::vector<int> const &tubeIds, PomadeStagedLevel *staged)
{
    size_t const cvTotal = points.size() / 3;
    staged->guidePoints.resize(cvTotal);
    staged->guideIndices.resize(cvTotal);
    staged->guideWidths.assign(
        cvTotal, PomadeOverlayWidth(PomadeOverlayPixels::kGuide,
                                   staged->displayScale, 0.015f));
    staged->guideHairT.resize(cvTotal);
    for (size_t i = 0; i < cvTotal; ++i) {
        staged->guidePoints[i] =
            GfVec3f(points[i * 3 + 0], points[i * 3 + 1], points[i * 3 + 2]);
        staged->guideIndices[i] = int(i);
    }
    staged->guideVertexCounts.resize(counts.size());
    staged->guideCurveColor.assign(colors.begin(), colors.end());
    staged->guideCurveTubeId.assign(tubeIds.begin(), tubeIds.end());
    size_t cursor = 0;
    for (size_t g = 0; g < counts.size(); ++g) {
        int const c = counts[g];
        staged->guideVertexCounts[g] = c;
        for (int i = 0; i < c && cursor < cvTotal; ++i, ++cursor) {
            staged->guideHairT[cursor] =
                c > 1 ? float(i) / float(c - 1) : 0.0f;
        }
    }
    staged->guideCount = int(counts.size());
    _Bounds(staged->guidePoints, &staged->guideMin, &staged->guideMax);
}

}  // namespace

float
PomadeOverlayWidth(float pixels, float displayScale, float fallbackWidth)
{
    if (displayScale > 0.0f && pixels > 0.0f) {
        return pixels * displayScale;
    }
    return fallbackWidth;
}

uint64_t
PomadeTubeContentHash(PomadeTubeDesc const &desc, int displaySegments,
                     PomadeTubeShape const &shape, bool isPrimaryTube)
{
    uint64_t h = kHashSeed;
    h = _HashInt(h, desc.tubeId);
    h = _HashInt(h, desc.level);
    h = _HashInt(h, desc.regionId);
    h = _HashInt(h, desc.childIndex);
    h = _HashInt(h, desc.ringVerts);
    h = _HashInt(h, displaySegments);
    h = _HashFloats(h, desc.centerX);
    h = _HashFloats(h, desc.centerY);
    h = _HashFloats(h, desc.centerZ);
    h = _HashFloats(h, desc.rootSurfaceOffsets);
    h = _HashInt(h, int(desc.sections.size()));
    for (PomadeTubeSection const &s : desc.sections) {
        h = _HashFloat(h, s.t);
        h = _HashFloat(h, s.scale);
        h = _HashFloat(h, s.twist);
        h = _HashFloats(h, s.u);
        h = _HashFloats(h, s.v);
    }
    if (isPrimaryTube) {
        // Tube 0 keeps the legacy cylinder display until a section op runs,
        // and that path reads radius/length/rings off the shape, which the
        // desc does not carry.
        h = _HashInt(h, shape.rings);
        h = _HashInt(h, shape.ringVerts);
        h = _HashFloat(h, shape.radius);
        h = _HashFloat(h, shape.length);
    }
    return h;
}

void
PomadeStageTestTubeMesh(PomadeTubeShape const &shape, PomadeStagedTubeMesh *out)
{
    if (!out) {
        return;
    }
    PomadeTubeShape used = shape;
    used.rings = std::max(shape.rings, 2);
    used.ringVerts = std::max(shape.ringVerts, 3);
    int const rings = used.rings;
    int const rv = used.ringVerts;
    std::vector<float> cx(size_t(rings), 0.0f);
    std::vector<float> cy(size_t(rings), 0.0f);
    std::vector<float> cz(size_t(rings), 0.0f);
    for (int r = 0; r < rings; ++r) {
        cy[size_t(r)] = used.length * float(r) / float(rings - 1);
    }
    int const vertexCount = PomadeTubeVertexCount(used);
    int const quadCount = PomadeTubeQuadCount(used);
    PomadeStagedTubeMesh staged;
    staged.points.resize(size_t(vertexCount));
    staged.normals.resize(size_t(quadCount) * 4);
    for (int v = 0; v < vertexCount; ++v) {
        PomadeTessellatedVertex const tv =
            PomadeTessellateVertex(used, v, cx.data(), cy.data(), cz.data());
        staged.points[size_t(v)] = GfVec3f(tv.px, tv.py, tv.pz);
    }
    staged.faceVertexCounts.assign(size_t(quadCount), 4);
    _FillStripNormals(staged.points, 0, rings, rv, 0, &staged.normals);
    staged.faceVertexIndices.resize(size_t(quadCount) * 4);
    for (int r = 0; r < rings - 1; ++r) {
        for (int s = 0; s < rv; ++s) {
            int const q = r * rv + s;
            int const s1 = (s + 1) % rv;
            staged.faceVertexIndices[size_t(q) * 4 + 0] = r * rv + s;
            staged.faceVertexIndices[size_t(q) * 4 + 1] = r * rv + s1;
            staged.faceVertexIndices[size_t(q) * 4 + 2] = (r + 1) * rv + s1;
            staged.faceVertexIndices[size_t(q) * 4 + 3] = (r + 1) * rv + s;
        }
    }
    _Bounds(staged.points, &staged.extentMin, &staged.extentMax);
    staged.version = 0;
    *out = std::move(staged);
}

PomadePublisher::PomadePublisher() = default;
PomadePublisher::~PomadePublisher() = default;

void
PomadePublisher::Clear()
{
    _current = PomadeStagedModel();
}

bool
PomadePublisher::Stage(PomadeModel const &model)
{
    std::vector<PomadeModel::TubeView> const views = model.SnapshotTubes();
    if (views.empty()) {
        return false;
    }
    int const segments = std::max(model.GetDisplaySegments(), 1);
    int const ringDisplay = model.GetRingDisplay();
    float const displayScale = model.GetDisplayScale();
    bool amplifiedTiles = false;
    bool guidesVisible = true;
    model.ResolveHairDisplay(&amplifiedTiles, &guidesVisible);
    PomadeTubeShape const shape = model.GetShape();
    PomadeModel::HostTubeMesh const &host = model.GetHostMesh();
    int const focusLevel = model.GetFocusLevel();
    // Hierarchy Levels can show a mixed-depth active cut: expanding one
    // branch replaces only that branch's parent with its children.  The
    // model predicate is deliberately a no-op until that presentation is
    // enabled, preserving every existing caller's numeric-level batches.
    bool const activeCut = model.GetActiveCutEnabled();

    // Group by level; sort by tube id so the layout is a pure function of
    // the model and two publishes of the same model agree byte for byte.
    std::map<int, std::vector<PomadeTubeDesc>> byLevel;
    std::map<int, int> tubeLevels;
    // Keep the unfiltered hierarchy for guide ownership. A cut removes an
    // edit mesh from its numeric batch; it must never erase that leaf's
    // generated hairs from Fill/Output.
    std::map<int, PomadeTubeDesc> allDescs;
    for (PomadeModel::TubeView const &view : views) {
        allDescs[view.desc.tubeId] = view.desc;
        if (!model.IsTubeVisibleInActiveCut(view.desc.tubeId)) {
            continue;
        }
        int const level = std::max(view.desc.level, 1);
        byLevel[level].push_back(view.desc);
        tubeLevels[view.desc.tubeId] = level;
    }
    // One read of the selection per publish (plan/18 §2.3). A level-kind
    // item needs the tube levels above, which is why it resolves here and
    // not in the model.
    _SelectionByTube const selection = _ResolveSelection(
        model.SelectionItems(0), model.SelectionHover(), tubeLevels);

    PomadeStagedModel next;
    next.amplifiedTilesVisible = amplifiedTiles;
    int restaged = 0;

    for (auto &levelEntry : byLevel) {
        int const level = levelEntry.first;
        std::vector<PomadeTubeDesc> &descs = levelEntry.second;
        std::sort(descs.begin(), descs.end(),
                  [](PomadeTubeDesc const &a, PomadeTubeDesc const &b) {
                      return a.tubeId < b.tubeId;
                  });

        PomadeStagedLevel staged;
        staged.level = level;
        PomadeModel::LevelDisplay const display = model.GetLevelDisplay(level);
        staged.visible = display.visible;
        staged.xray = display.xray;
        staged.xrayOpacity = display.xray ? display.xrayOpacity : 0.0f;
        staged.centers = display.centers;
        staged.centerCVDots = display.centerCVDots;
        staged.ringCVDots = display.ringCVDots;
        staged.centersOnly = display.centersOnly;
        staged.guidesVisible = guidesVisible && display.guides;
        // A branch cut has no single meaningful numeric focus level: every
        // frontier member is directly editable, even when its siblings live
        // at L1/L2/L3 in the same frame.
        staged.focused = activeCut || level == focusLevel;
        staged.displayScale = displayScale;

        // -- layout ---------------------------------------------------------
        // Sizes come from the desc, so the whole layout is known before a
        // single vertex is tessellated. That is what lets an unchanged tube
        // keep the bytes it already has.
        std::vector<PomadeTubeDesc const *> kept;
        uint64_t topoHash = kHashSeed;
        uint64_t pointsHash = kHashSeed;
        uint64_t uniformHash = kHashSeed;
        uint64_t ringHash = kHashSeed;
        int pointCursor = 0;
        int faceCursor = 0;
        int centerCursor = 0;
        int ringCurveCursor = 0;
        for (PomadeTubeDesc const &desc : descs) {
            bool const primary = desc.tubeId == 0;
            PomadeTubeSlice slice;
            slice.tubeId = desc.tubeId;
            slice.level = level;
            slice.regionId = desc.regionId;
            slice.childIndex = desc.childIndex;
            slice.contentHash =
                PomadeTubeContentHash(desc, segments, shape, primary);
            if (primary) {
                slice.ringVerts = std::max(shape.ringVerts, 3);
                int const verts = int(host.positions.size() / 3);
                if (verts < slice.ringVerts) {
                    continue;
                }
                slice.ringCount = verts / slice.ringVerts;
                // The legacy test tube keeps its unsampled host grid until
                // a section edit switches it to K5.
                slice.ringStride = slice.ringCount ==
                    PomadeTessellatedRingCount(desc, segments) ? segments : 1;
            } else {
                slice.ringVerts = desc.ringVerts;
                slice.ringCount = PomadeTessellatedRingCount(desc, segments);
                slice.ringStride = segments;
            }
            slice.centerCount = int(desc.centerX.size());
            if (slice.ringCount < 2 || slice.ringVerts < 3 ||
                slice.centerCount < 2) {
                continue;
            }
            slice.pointCount = slice.ringCount * slice.ringVerts;
            slice.faceCount = (slice.ringCount - 1) * slice.ringVerts;
            slice.ringCurveCount =
                (slice.ringCount - 1) / slice.ringStride + 1;
            // plan/18 §2.4a: the cross-section rings and their CV dots draw
            // in Tube mode's Ring/Section sub-modes, and on selected tubes
            // everywhere else. Zeroing the curve count here is what makes
            // that one decision: the whole ring layout (array sizes, curve
            // offsets, fill loops, bounds) follows from it.
            if (ringDisplay == PomadeModel::Rings_Off) {
                slice.ringCurveCount = 0;
            } else if (ringDisplay != PomadeModel::Rings_All) {
                auto const sel = selection.find(desc.tubeId);
                bool const on =
                    sel != selection.end() &&
                    (sel->second.tube != 0 || !sel->second.rings.empty() ||
                     !sel->second.sectionCVs.empty());
                if (!on) {
                    slice.ringCurveCount = 0;
                }
            }
            slice.pointOffset = pointCursor;
            slice.faceOffset = faceCursor;
            slice.centerOffset = centerCursor;
            slice.ringCurveOffset = ringCurveCursor;
            slice.clumpColor = _Rgb(
                PomadeClumpColor(desc.regionId, desc.level, desc.childIndex));
            pointCursor += slice.pointCount;
            faceCursor += slice.faceCount;
            centerCursor += slice.centerCount;
            ringCurveCursor += slice.ringCurveCount;

            topoHash = _HashInt(topoHash, slice.tubeId);
            topoHash = _HashInt(topoHash, slice.ringCount);
            topoHash = _HashInt(topoHash, slice.ringVerts);
            topoHash = _HashInt(topoHash, slice.ringStride);
            topoHash = _HashInt(topoHash, slice.centerCount);
            pointsHash = _HashBytes(pointsHash, &slice.contentHash,
                                    sizeof(slice.contentHash));
            uniformHash = _HashInt(uniformHash, slice.tubeId);
            uniformHash = _HashFloat(uniformHash, slice.clumpColor[0]);
            uniformHash = _HashFloat(uniformHash, slice.clumpColor[1]);
            uniformHash = _HashFloat(uniformHash, slice.clumpColor[2]);
            ringHash = _HashInt(ringHash, slice.tubeId);
            ringHash = _HashInt(ringHash, slice.ringCurveCount);
            ringHash = _HashInt(ringHash, slice.ringVerts);
            staged.tubes.push_back(slice);
            kept.push_back(&desc);
        }
        if (staged.tubes.empty()) {
            continue;
        }
        staged.topologyHash = topoHash;
        staged.pointsHash = pointsHash;
        staged.uniformHash = uniformHash;
        staged.ringHash = ringHash;
        {
            // Only the tubes this level actually holds contribute, so
            // selecting a child never dirties its parent's level.
            uint64_t selHash = kHashSeed;
            for (PomadeTubeSlice const &s : staged.tubes) {
                auto const hit = selection.find(s.tubeId);
                if (hit == selection.end()) {
                    continue;
                }
                selHash = _HashInt(selHash, s.tubeId);
                selHash = _HashTubeSelection(selHash, hit->second);
            }
            staged.selectionHash = selHash;
        }

        // -- host mirror: re-tessellate only what changed -------------------
        PomadeStagedLevel const *prev = nullptr;
        {
            auto it = _current.levels.find(level);
            if (it != _current.levels.end()) {
                prev = &it->second;
            }
        }
        bool const layoutKept =
            prev != nullptr && prev->topologyHash == staged.topologyHash;
        LevelMirror &mirror = _mirrors[level];
        size_t const floatCount = size_t(pointCursor) * 3;
        bool rebuiltMirror = false;
        if (!layoutKept || mirror.positions.size() < floatCount) {
            mirror.positions.assign(floatCount, 0.0f);
            mirror.normals.assign(floatCount, 0.0f);
            rebuiltMirror = true;
        }
        std::map<int, uint64_t> unchanged;
        if (layoutKept && !rebuiltMirror) {
            for (PomadeTubeSlice const &s : prev->tubes) {
                unchanged[s.tubeId] = s.contentHash;
            }
        }
        int levelRestaged = 0;
        std::vector<float> scratchPos;
        std::vector<float> scratchNrm;
        std::vector<float> scratchT;
        std::vector<PomadeFrame> frames;
        for (size_t i = 0; i < staged.tubes.size(); ++i) {
            PomadeTubeSlice const &slice = staged.tubes[i];
            PomadeTubeDesc const &desc = *kept[i];
            auto const hit = unchanged.find(slice.tubeId);
            if (hit != unchanged.end() && hit->second == slice.contentHash) {
                continue;  // the bytes in the mirror are still correct
            }
            float *outPos =
                mirror.positions.data() + size_t(slice.pointOffset) * 3;
            float *outNrm =
                mirror.normals.data() + size_t(slice.pointOffset) * 3;
            size_t const sliceFloats = size_t(slice.pointCount) * 3;
            if (desc.tubeId == 0) {
                // Device -> pinned host -> the level mirror. The pinned
                // block is what Hydra's arrays are filled from, never the
                // model's own storage (plan/17 §4.3 phase A).
                if (host.positions.size() < sliceFloats ||
                    host.normals.size() < sliceFloats) {
                    continue;
                }
                float *pinnedPos = _pinnedPositions.Ensure(sliceFloats);
                float *pinnedNrm = _pinnedNormals.Ensure(sliceFloats);
                if (pinnedPos && pinnedNrm &&
                    model.CopyDeviceToHost(pinnedPos, pinnedNrm,
                                           sliceFloats)) {
                    std::memcpy(outPos, pinnedPos,
                                sliceFloats * sizeof(float));
                    std::memcpy(outNrm, pinnedNrm,
                                sliceFloats * sizeof(float));
                } else {
                    std::memcpy(outPos, host.positions.data(),
                                sliceFloats * sizeof(float));
                    std::memcpy(outNrm, host.normals.data(),
                                sliceFloats * sizeof(float));
                }
            } else {
                std::string err;
                if (!PomadeTubeFramesCpu(desc, &frames, &err)) {
                    continue;
                }
                if (!PomadeTessellateCpu(desc, frames, segments, &scratchPos,
                                        &scratchNrm, &scratchT, &err)) {
                    continue;
                }
                if (scratchPos.size() < sliceFloats ||
                    scratchNrm.size() < sliceFloats) {
                    continue;
                }
                std::memcpy(outPos, scratchPos.data(),
                            sliceFloats * sizeof(float));
                std::memcpy(outNrm, scratchNrm.data(),
                            sliceFloats * sizeof(float));
            }
            ++levelRestaged;
        }
        restaged += levelRestaged;

        bool const geometryKept =
            layoutKept && !rebuiltMirror && levelRestaged == 0;
        bool const ringsKept =
            geometryKept && prev->ringHash == staged.ringHash;
        _FillLevelArrays(mirror.positions, kept,
                         geometryKept ? prev : nullptr, ringsKept, &staged);
        // The selection arrays are rewritten only when the selection that
        // touches this level changed; otherwise the reuse path above has
        // already shared the previous snapshot's arrays, and a drag with a
        // stable selection copies no bytes at all.
        if (!geometryKept || !ringsKept ||
            prev->selectionHash != staged.selectionHash) {
            _FillSelection(selection, &staged);
        }
        next.levels[level] = std::move(staged);
    }

    if (next.levels.empty()) {
        return false;
    }

    // -- guides ------------------------------------------------------------
    // The live refill merges every producing tube's set with per-guide
    // tube attribution; each level stages its own tubes' curves in the
    // owning tube's clump colour (plan/18 §2.2 guides/L<n>). A curve
    // whose tube is no longer staged (dropped after the last refill)
    // falls back to level 1, where the pre-split set always lived.
    {
        PomadeModel::GuidePreview const preview = model.GetGuidePreview();
        if (preview.guideCount > 0 && !preview.points.empty() &&
            !preview.counts.empty() &&
            preview.tubeIds.size() == preview.counts.size()) {
            std::map<int, int> tubeLevel;  // tubeId -> level
            for (auto const &lv : next.levels) {
                for (PomadeTubeSlice const &s : lv.second.tubes) {
                    tubeLevel[s.tubeId] = lv.first;
                }
            }
            int fallbackLevel = 1;
            if (next.levels.find(1) == next.levels.end()) {
                fallbackLevel = next.levels.begin()->first;
            }
            std::vector<size_t> offsets(preview.counts.size() + 1, 0);
            for (size_t g = 0; g < preview.counts.size(); ++g) {
                offsets[g + 1] = offsets[g] + size_t(preview.counts[g]);
            }
            struct _GuideOwner {
                size_t index = 0;
                int tubeId = -1;
                GfVec3f color = GfVec3f(0.6f);
            };
            std::map<int, std::vector<_GuideOwner>> byLevel;
            for (size_t g = 0; g < preview.counts.size(); ++g) {
                int const ownerId = preview.tubeIds[g];
                int batchId = ownerId;
                auto owner = allDescs.find(ownerId);
                // An active cut stages the full groom under the nearest
                // visible ancestor, but retains the producing leaf's id and
                // colour. Outside a cut this loop exits on the owner.
                while (activeCut &&
                       !model.IsTubeVisibleInActiveCut(batchId)) {
                    auto const current = allDescs.find(batchId);
                    if (current == allDescs.end() ||
                        current->second.parentTubeId < 0) {
                        batchId = -1;
                        break;
                    }
                    batchId = current->second.parentTubeId;
                }
                auto const batch = tubeLevel.find(batchId);
                int const level = batch == tubeLevel.end() ? fallbackLevel
                                                            : batch->second;
                GfVec3f const color = owner == allDescs.end()
                    ? next.levels[level].tubes.front().clumpColor
                    : _Rgb(PomadeClumpColor(owner->second.regionId,
                                            owner->second.level,
                                            owner->second.childIndex));
                byLevel[level].push_back(_GuideOwner{g, ownerId, color});
            }
            for (auto const &lv : byLevel) {
                auto it = next.levels.find(lv.first);
                if (it == next.levels.end()) {
                    continue;
                }
                std::vector<float> points;
                std::vector<int> counts;
                std::vector<GfVec3f> colors;
                std::vector<int> tubeIds;
                for (_GuideOwner const &guide : lv.second) {
                    colors.push_back(guide.color);
                    tubeIds.push_back(guide.tubeId);
                    counts.push_back(preview.counts[guide.index]);
                    size_t const begin = offsets[guide.index] * 3;
                    size_t end = offsets[guide.index + 1] * 3;
                    if (end > preview.points.size()) {
                        end = preview.points.size();
                    }
                    points.insert(points.end(),
                                  preview.points.begin() + ptrdiff_t(begin),
                                  preview.points.begin() + ptrdiff_t(end));
                }
                _FillGuides(points, counts, colors, tubeIds, &it->second);
            }
        }
    }

    next.restagedTubeCount = restaged;
    _current = std::move(next);
    return true;
}

} // namespace usdGenPomade
