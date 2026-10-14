// testUsdGenTileReuse — per-array tile carry for deform-only cooks (T1).
//
// A rebuilt tile re-gathers its points but carries its previous non-point
// arrays (counts, widths, hairT, hairId, st, displayColor, extra planes)
// when the cook wrote only point planes (deforms alone) and the partition,
// baseline, and colors are unchanged. Asserted at session level:
//
//   1. Two deform cooks (moved surface between them): points differ, every
//      non-point array is IsIdentical, and LastReport() marks only points
//      (plus rebaked furTau) dirty.
//   2. The carry is content, not storage aliasing of convenience: a width
//      edit re-gathers widths (report widthsDirty), a look edit re-gathers
//      displayColor only (widths still carried), and a segments edit
//      re-gathers everything through a topology change.
//
// Tier T1 (registered in CMake); engine-only body (gate B-1).

#include "usdGen/opRegistry.h"
#include "usdGen/session.h"
#include "usdGen/types.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;

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

UsdGenGraphDesc MakeDeformDesc()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/groom");
    desc.terminal = SdfPath("/groom/deform");
    desc.time = 0.0;
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.id = 0;
    int const NX = 15, NY = 15;
    s.restPoints = VtVec3fArray((NX + 1) * (NY + 1));
    s.uv = VtVec2fArray((NX + 1) * (NY + 1));
    for (int j = 0; j <= NY; ++j)
        for (int i = 0; i <= NX; ++i) {
            int const k = j * (NX + 1) + i;
            float const x = float(i) * 0.1f, y = float(j) * 0.1f;
            s.restPoints[k] = GfVec3f(x, y, 2.0f * std::sin(x * 0.3f) * std::cos(y * 0.3f));
            s.uv[k] = GfVec2f(float(i) / NX, float(j) / NY);
        }
    s.points = s.restPoints;
    s.faceVertexCounts = VtIntArray(NX * NY, 4);
    s.faceVertexIndices = VtIntArray(NX * NY * 4);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            int const a = j * (NX + 1) + i, o = (j * NX + i) * 4;
            s.faceVertexIndices[o] = a;
            s.faceVertexIndices[o + 1] = a + 1;
            s.faceVertexIndices[o + 2] = a + NX + 2;
            s.faceVertexIndices[o + 3] = a + NX + 1;
        }
    desc.surfaces.push_back(s);
    auto addNode = [&](std::string const &name, TfToken type, std::string const &input, int seed) {
        UsdGenNodeDesc n;
        n.path = SdfPath("/groom/" + name);
        n.type = type;
        n.enabled = true;
        n.seed = seed;
        if (!input.empty()) n.inputs.push_back(SdfPath("/groom/" + input));
        if (type == TfToken("UsdGenScatter") || type == TfToken("UsdGenDeform"))
            n.surfaces.push_back(SdfPath("/groom/surface"));
        desc.nodes.push_back(std::move(n));
    };
    addNode("scatter", TfToken("UsdGenScatter"), "", 42);
    addNode("grow", TfToken("UsdGenGrow"), "scatter", 43);
    addNode("width", TfToken("UsdGenWidth"), "grow", 44);
    addNode("deform", TfToken("UsdGenDeform"), "width", 45);
    auto setp = [&](std::string const &name, TfToken p, VtValue v) {
        for (auto &n : desc.nodes)
            if (n.path == SdfPath("/groom/" + name))
                n.params.push_back(UsdGenParamValue{p, v, false});
    };
    setp("scatter", TfToken("density"), VtValue(25.0));
    setp("grow", TfToken("segments"), VtValue(4));
    setp("grow", TfToken("length"), VtValue(1.0));
    setp("width", TfToken("width"), VtValue(0.02));
    setp("deform", TfToken("rbfSamples"), VtValue(100));
    setp("deform", TfToken("lockRoots"), VtValue(true));
    return desc;
}

UsdGenTilePublication const *FindTile(UsdGenGeneration const &gen, UsdGenTileId tile)
{
    for (UsdGenTilePublication const &t : gen.tiles)
        if (t.tile == tile) return &t;
    return nullptr;
}

bool HasPrimvar(std::vector<TfToken> const &names, char const *name)
{
    for (TfToken const &t : names)
        if (t == TfToken(name)) return true;
    return false;
}

template <typename T>
bool SameBytes(VtArray<T> const &a, VtArray<T> const &b)
{
    if (a.size() != b.size()) return false;
    size_t const n = a.size() * sizeof(T);
    return n == 0 || std::memcmp(a.data(), b.data(), n) == 0;
}

}  // namespace

int main()
{
    usdGenRegisterM1Operators();

    UsdGenSession session(8);
    UsdGenGraphDesc desc = MakeDeformDesc();

    // ---- 1: first cook ----------------------------------------------------
    session.SetGraphDesc(desc);
    UsdGenGenerationConstPtr g1 = session.Commit(1.0, UsdGenCommitReason::NoticeBatchEnd);
    Check(g1 != nullptr && !g1->tiles.empty(), "first deform cook publishes tiles");

    // ---- 2: moved surface -> deform-only cook ------------------------------
    UsdGenGraphDesc desc2 = desc;
    for (GfVec3f &p : desc2.surfaces[0].points) p[2] += 0.5f;
    session.SetGraphDesc(desc2);
    UsdGenGenerationConstPtr g2 = session.Commit(2.0, UsdGenCommitReason::NoticeBatchEnd);
    Check(g2 != nullptr && g2->tiles.size() == g1->tiles.size(),
          "second deform cook publishes the same tile set");
    bool anyPointsMoved = false;
    bool allCarried = true;
    bool colorsCarried = true;
    bool tauFresh = true;
    for (UsdGenTilePublication const &b : g1->tiles) {
        UsdGenTilePublication const *a = FindTile(*g2, b.tile);
        if (!a) { allCarried = false; continue; }
        if (!SameBytes(b.points, a->points)) anyPointsMoved = true;
        allCarried = allCarried &&
            a->curveVertexCounts.IsIdentical(b.curveVertexCounts) &&
            a->widths.IsIdentical(b.widths) &&
            a->hairT.IsIdentical(b.hairT) &&
            a->hairId.IsIdentical(b.hairId) &&
            a->st.IsIdentical(b.st);
        colorsCarried = colorsCarried && a->displayColor.IsIdentical(b.displayColor);
        auto tau = [&](UsdGenTilePublication const &t, char const *name) -> UsdGenPlane const * {
            for (UsdGenPlane const &p : t.extraUniform)
                if (p.name == TfToken(name)) return &p;
            return nullptr;
        };
        UsdGenPlane const *p0 = tau(b, "furTauP"), *p1 = tau(*a, "furTauP");
        UsdGenPlane const *n0 = tau(b, "furTauN"), *n1 = tau(*a, "furTauN");
        tauFresh = tauFresh && p0 && p1 && n0 && n1 &&
            !p1->f.IsIdentical(p0->f) && !n1->f.IsIdentical(n0->f);
    }
    Check(anyPointsMoved, "moved surface displaces points (non-vacuous deform)");
    Check(allCarried, "deform-only cook carries counts/widths/hairT/hairId/st storage");
    Check(colorsCarried, "deform-only cook carries displayColor storage (stable look)");
    Check(tauFresh, "occlusion rebakes furTau planes on a deform cook (no over-carry)");
    bool reportPrecise = true;
    for (UsdGenTileDirty const &td : session.LastReport().tiles) {
        reportPrecise = reportPrecise && td.pointsDirty && !td.widthsDirty &&
            !HasPrimvar(td.dirtyPrimvars, "hairT") &&
            !HasPrimvar(td.dirtyPrimvars, "hairId") &&
            !HasPrimvar(td.dirtyPrimvars, "st") &&
            !HasPrimvar(td.dirtyPrimvars, "displayColor") &&
            HasPrimvar(td.dirtyPrimvars, "furTauP") &&
            HasPrimvar(td.dirtyPrimvars, "furTauN");
    }
    Check(reportPrecise, "LastReport() marks points+furTau dirty, carried primvars clean");

    // ---- 3: width edit -> a non-deform ran, widths re-gather ----------------
    UsdGenGraphDesc desc3 = desc2;
    for (auto &n : desc3.nodes)
        if (n.path == SdfPath("/groom/width"))
            for (auto &p : n.params)
                if (p.name == TfToken("width")) p.value = VtValue(0.05);
    session.SetGraphDesc(desc3);
    UsdGenGenerationConstPtr g3 = session.Commit(3.0, UsdGenCommitReason::NoticeBatchEnd);
    bool widthsFresh = g3 && g3->tiles.size() == g2->tiles.size();
    bool widthsValuesMoved = false;
    for (UsdGenTilePublication const &b : g2->tiles) {
        UsdGenTilePublication const *a = g3 ? FindTile(*g3, b.tile) : nullptr;
        if (!a) { widthsFresh = false; continue; }
        widthsFresh = widthsFresh && !a->widths.IsIdentical(b.widths);
        if (!SameBytes(b.widths, a->widths)) widthsValuesMoved = true;
    }
    Check(widthsFresh, "width edit re-gathers widths (gate closes on non-deform writes)");
    Check(widthsValuesMoved, "re-gathered widths carry the edited value");
    bool widthsReported = false;
    for (UsdGenTileDirty const &td : session.LastReport().tiles)
        widthsReported = widthsReported || td.widthsDirty;
    Check(widthsReported, "LastReport() marks widths dirty after a width edit");

    // ---- 4: look edit -> colors re-gather, arrays still carried -------------
    UsdGenGraphDesc desc4 = desc3;
    desc4.look.rootColor = GfVec3f(0.9f, 0.1f, 0.1f);
    session.SetGraphDesc(desc4);
    UsdGenGenerationConstPtr g4 = session.Commit(4.0, UsdGenCommitReason::NoticeBatchEnd);
    bool colorsFresh = g4 && g4->tiles.size() == g3->tiles.size();
    bool colorsValuesMoved = false;
    bool arraysStillCarried = g4 && g4->tiles.size() == g3->tiles.size();
    for (UsdGenTilePublication const &b : g3->tiles) {
        UsdGenTilePublication const *a = g4 ? FindTile(*g4, b.tile) : nullptr;
        if (!a) { colorsFresh = false; arraysStillCarried = false; continue; }
        colorsFresh = colorsFresh && !a->displayColor.IsIdentical(b.displayColor);
        if (!SameBytes(b.displayColor, a->displayColor)) colorsValuesMoved = true;
        arraysStillCarried = arraysStillCarried && a->widths.IsIdentical(b.widths) &&
            a->hairT.IsIdentical(b.hairT);
    }
    Check(colorsFresh, "look edit re-gathers displayColor (colors gate closes on recolour)");
    Check(colorsValuesMoved, "re-gathered displayColor carries the edited root color");
    Check(arraysStillCarried, "look edit still carries widths/hairT (arrays gate ignores recolour)");

    // ---- 5: segments edit -> topology change re-gathers everything ----------
    UsdGenGraphDesc desc5 = desc4;
    for (auto &n : desc5.nodes)
        if (n.path == SdfPath("/groom/grow"))
            for (auto &p : n.params)
                if (p.name == TfToken("segments")) p.value = VtValue(6);
    session.SetGraphDesc(desc5);
    UsdGenGenerationConstPtr g5 = session.Commit(5.0, UsdGenCommitReason::NoticeBatchEnd);
    bool topologyFresh = g5 && !g5->tiles.empty();
    for (UsdGenTilePublication const &b : g4->tiles) {
        UsdGenTilePublication const *a = g5 ? FindTile(*g5, b.tile) : nullptr;
        if (!a) continue;
        topologyFresh = topologyFresh && !a->curveVertexCounts.IsIdentical(b.curveVertexCounts) &&
            !a->widths.IsIdentical(b.widths);
    }
    Check(topologyFresh, "segments edit re-gathers counts/widths (gate closes on repartition)");

    std::printf(g_failures ? "testUsdGenTileReuse: FAILED (%d)\n"
                           : "testUsdGenTileReuse: PASS\n",
                g_failures);
    return g_failures ? 1 : 0;
}
