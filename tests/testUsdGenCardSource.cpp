// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenCardSource: card grids (ops/cardGeometry.h) and the
// UsdGenCardSource generator.
//
//   (1) card grid geometry against hand-computed values: rows/cols with and
//       without smoothing, the Surface side reference, width / ridge / sink /
//       twist / length scale, tube perimeter, the card-surface lookup with
//       flipU, depth and longitudinal bias, render-mesh counts;
//   (2) end-to-end cooks over a flat scalp: strand ids, generation order,
//       strand seeds, root snap, tip placement, layer / uvId / tile filters,
//       amount and maxStrands subsampling, reference-order independence,
//       determinism.
//
// Harness mode (validation of an imported scene, not run by ctest):
//   testUsdGenCardSource --usda <file.usda> [--dump <out.txt>]
// reads a flat .usda layer (authored by the groom importer card script plus a Mesh),
// cooks every UsdGenCardSource prim in it against the layer's Mesh, prints
// "set <prim> strands=<n>" per operator, and the card render-mesh size of all
// cards ("mesh vertices=<v> indices=<i>"). --dump writes one line per strand:
// <prim> <curveId hex> <strandIndex> <cardIndex> <root xyz> <tip xyz>.
#include "usdGen/authoredPlaneConvert.h"
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/ops/cardGeometry.h"
#include "usdGen/ops/strandUtil.h"
#include "usdGen/scheduler.h"

#include "pxr/usd/sdf/attributeSpec.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/sdf/primSpec.h"
#include "pxr/usd/sdf/relationshipSpec.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace usdGen;

static int failures = 0;
static void Check(bool v, std::string const &s)
{
    if (!v) { ++failures; std::printf("FAIL: %s\n", s.c_str()); }
    else std::printf("ok: %s\n", s.c_str());
}
static bool Near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) <= tol; }
static bool Near(GfVec3f const &a, GfVec3f const &b, float tol = 1e-5f)
{
    return (a - b).GetLength() <= tol;
}
static std::string Str(GfVec3f const &v)
{
    char buf[96];
    std::snprintf(buf, sizeof(buf), "(%.6f, %.6f, %.6f)", v[0], v[1], v[2]);
    return buf;
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Flat scalp on y = 0, x in [-0.2, 0.4], z in [-0.1, 0.1], quads wound for
/// +Y normals, uv = (x, z).
static UsdGenSurfaceDesc FlatScalp()
{
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/cards/scalp");
    s.id = 0;
    int const nx = 12, nz = 4;
    for (int iz = 0; iz <= nz; ++iz)
        for (int ix = 0; ix <= nx; ++ix) {
            float const x = -0.2f + 0.05f * float(ix), z = -0.1f + 0.05f * float(iz);
            s.restPoints.push_back(GfVec3f(x, 0.0f, z));
            s.uv.push_back(GfVec2f(x, z));
        }
    auto at = [&](int ix, int iz) { return iz * (nx + 1) + ix; };
    for (int iz = 0; iz < nz; ++iz)
        for (int ix = 0; ix < nx; ++ix) {
            s.faceVertexCounts.push_back(4);
            for (int v : {at(ix, iz), at(ix, iz + 1), at(ix + 1, iz + 1), at(ix + 1, iz)})
                s.faceVertexIndices.push_back(v);
        }
    s.points = s.restPoints;
    s.surfaceGeneration = 1;
    return s;
}

static card::Card StraightCard(uint64_t id = 1000)
{
    card::Card c;
    c.points = {GfVec3f(0, 0, 0), GfVec3f(0.1f, 0, 0), GfVec3f(0.2f, 0, 0)};
    c.id = id;
    c.uvId = 3;
    c.tile[0] = c.tile[1] = c.tile[2] = 1;
    c.width = 0.02f;
    c.orientation = card::kSurface;
    // The root (0,0,0) is corner 0 of quad 28 (ix 4, iz 2) -> fan triangle 56.
    c.rootTri = 56;
    c.rootBary = GfVec3f(1.0f, 0.0f, 0.0f);
    return c;
}

// Clamped-end uniform Catmull-Rom x4 of 0, 0.1, 0.2 along x (hand-evaluated):
// segment 0: 0.5 (0.1 t + 0.2 t^2 - 0.1 t^3); segment 1: 0.5 (0.2 + 0.2 t + 0.1 t^2 - 0.1 t^3).
static float const kSmoothX[9] = {0.0f, 0.01796875f, 0.04375f, 0.07265625f, 0.1f,
                                  0.12734375f, 0.15625f, 0.18203125f, 0.2f};

struct Geometry
{
    UsdGenSurfaceDesc surface = FlatScalp();
    strand::TriangleBvh bvh;
    std::vector<card::Card> cards;
    card::Context ctx;
    Geometry()
    {
        strand::BuildSurfaceBvh(surface, true, &bvh);
        ctx.scalp = &bvh;
        ctx.bboxCentre = GfVec3f(0.1f, 0.0f, 0.0f);
        ctx.cards = &cards;
    }
    std::vector<card::Span> Build(card::Card c, bool smooth = true)
    {
        c.Sanitize();
        cards = {c};
        ctx.smooth = smooth;
        return card::BuildCard(0, ctx);
    }
};

// ---------------------------------------------------------------------------
// (1) card grids
// ---------------------------------------------------------------------------

static void TestGeometry()
{
    Geometry g;
    {
        auto spans = g.Build(StraightCard());
        Check(spans.size() == 1, "ribbon builds one span");
        card::Span const &s = spans[0];
        Check(s.rows == 9 && s.cols == 2, "smoothed ribbon of 3 points: 4n-3 = 9 rows, 2 columns");
        bool uniform = true, widths = true, normals = true, edges = true;
        for (uint32_t i = 0; i < s.rows; ++i) {
            float const x = kSmoothX[i];
            uniform = uniform && Near(s.rowParam[i], x / 0.2f);
            widths = widths && Near(s.rowWidth[i], 0.01f);
            // Surface side = normalize(first segment x N0) = +Z; column 0 at -S.
            edges = edges && Near(s.V[i * 2], GfVec3f(x, 0, -0.005f)) &&
                Near(s.V[i * 2 + 1], GfVec3f(x, 0, 0.005f));
            normals = normals && Near(s.N[i * 2], GfVec3f(0, 1, 0));
        }
        Check(uniform, "Catmull-Rom rows (clamped ends), row parameter = normalised arc length");
        Check(widths, "row width = width * root profile (0.02 * 0.5)");
        Check(edges, "edges at C -/+ S * half width, S = +Z for a card along +X on a +Y scalp");
        Check(normals, "card normal B = S x T = +Y");
        Check(s.flipU && !s.tube && Near(s.length, 0.2f), "ribbon span: flipU, middle-column length");

        GfVec3f p, n;
        card::CardSurface(s, 0.8f, 0.5f, 0.1f, &p, &n);
        // flipU: u 0.8 -> 0.2 -> z = -0.005 * 0.8 + 0.005 * 0.2; depth 0.1 row widths along +Y.
        Check(Near(p, GfVec3f(0.1f, 0.001f, -0.003f)) && Near(n, GfVec3f(0, 1, 0)),
              "cardSurface(0.8, 0.5, 0.1) = (0.1, 0.001, -0.003) " + Str(p));
        card::Span biased = s;
        biased.vBias = 0.25f;
        card::CardSurface(biased, 0.5f, 0.5f, 0.0f, &p, &n);
        Check(Near(p, GfVec3f(0.15f, 0, 0)), "longitudinal bias 0.25 un-biases v 0.5 -> 0.75 " + Str(p));
        card::CardSurface(s, 0.5f, -1.0f, 0.0f, &p, &n);
        Check(Near(p, GfVec3f(0, 0, 0)), "cardSurface clamps v below the root");
    }
    {
        auto spans = g.Build(StraightCard(), false);
        Check(spans.size() == 1 && spans[0].rows == 3 && spans[0].cols == 2,
              "unsmoothed ribbon keeps the spine rows");
    }
    {
        card::Card c = StraightCard();
        c.widthSegments = 2;
        c.centerRidge = 0.5f;
        auto spans = g.Build(c);
        Check(spans[0].cols == 3 && Near(spans[0].V[4 * 3 + 1], GfVec3f(0.1f, 0.0025f, 0.0f)),
              "centre ridge raises the middle column by ridge * half width");
        c.ridgeRootFlatten = 1.0f;
        spans = g.Build(c);
        Check(Near(spans[0].V[1][1], 0.0f) && Near(spans[0].V[8 * 3 + 1][1], 0.0025f),
              "ridgeRootFlatten fades the ridge in from the root");
    }
    {
        card::Card c = StraightCard();
        c.sink = 0.001f;
        auto spans = g.Build(c);
        card::Span const &s = spans[0];
        // F = min(L, max(2 width, L/4)) = 0.05: full sink along N0 = +Y at the root, none past F.
        Check(Near(s.V[0][1], -0.001f) && Near(s.V[1][1], -0.001f) && Near(s.V[3 * 2][1], 0.0f) &&
              Near(s.V[0][0], 0.0f),
              "sink pushes the root row 1 mm under the scalp and fades out by F");
        for (int row : {1, 2}) {
            float const w = 1.0f - strand::SmoothStep01(kSmoothX[row] / 0.05f);
            Check(Near(s.V[size_t(row) * 2][1], -0.001f * w),
                  "sink falloff is 1 - smoothstep(s / F), row " + std::to_string(row));
        }
    }
    {
        card::Card c = StraightCard();
        c.twistRoot = 90.0f;
        auto spans = g.Build(c);
        // S' = rot(+Z, T = +X, 90 deg) = -Y: column 0 (at -S') goes to +Y.
        Check(Near(spans[0].V[0], GfVec3f(0, 0.005f, 0)) && Near(spans[0].V[1], GfVec3f(0, -0.005f, 0)),
              "twist root 90 deg stands the root row up (" + Str(spans[0].V[0]) + ")");
        Check(Near(spans[0].V[8 * 2], GfVec3f(0.2f, 0, -0.005f)), "twist profile returns to 0 at the tip");
    }
    {
        card::Card c = StraightCard();
        c.lengthMax = 0.5f;
        auto spans = g.Build(c);
        Check(Near(spans[0].V[8 * 2], GfVec3f(0.1f, 0, -0.005f)), "length.max 0.5 halves the card");
        c.lengthMin = 0.5f;
        c.lengthMax = 1.0f;
        c.lengthRandomize = 1.0f;
        spans = g.Build(c);
        float const scale = 0.5f + strand::Hash01(1000) * 0.5f;
        Check(Near(spans[0].V[8 * 2][0], 0.2f * scale),
              "random length = lo + U01(splitmix64(id)) (hi - lo)");
        c.twinId = 5;   // a mirror twin: the seed is min(id, twin)
        spans = g.Build(c);
        float const twinScale = 0.5f + strand::Hash01(5) * 0.5f;
        Check(Near(spans[0].V[8 * 2][0], 0.2f * twinScale), "mirror twins share the length seed");
    }
    {
        card::Card c = StraightCard();
        c.cardType = card::kTube;
        auto spans = g.Build(c);
        Check(spans.size() == 1 && spans[0].tube && !spans[0].flipU && spans[0].cols == 13,
              "tube: 3 sides minimum x 4 when smoothing + seam = 13 columns");
        float const perimeter = 12.0f * 2.0f * 0.005f * std::sin(3.14159265f / 12.0f);
        Check(Near(spans[0].rowWidth[3], perimeter), "tube row width = ring perimeter");
        Check(Near(spans[0].V[0], spans[0].V[12]), "tube ring seam duplicated");
        c.points = {GfVec3f(0, 0, 0), GfVec3f(0.1f, 0, 0)};
        spans = g.Build(c);
        Check(spans[0].rows == 5, "a 2-point tube is still subdivided (4n-3)");
    }
    {
        card::Card c = StraightCard();
        c.cardType = card::kCross;
        c.tile[1] = 4;
        c.crossWidth = 0.5f;
        auto spans = g.Build(c);
        Check(spans.size() == 2 && spans[1].strip == 1 && spans[1].tile == 4,
              "cross: two strips, the second on tile2");
        // Strip 1 is turned 90 deg about T: S' = -Y, half width * crossWidth.
        Check(Near(spans[1].V[4 * 2], GfVec3f(0.1f, 0.0025f, 0)) && Near(spans[1].rowWidth[4], 0.005f),
              "cross strip 1 stands upright at crossWidth");
    }
    {
        // Mesh counts: ribbon (9 x 2), cross widthSegments 2 on 4 points (2 x 13 x 3),
        // tube on 2 points (5 x 13).
        std::vector<card::Span> all;
        auto add = [&](card::Card c) { auto s = g.Build(c); all.insert(all.end(), s.begin(), s.end()); };
        add(StraightCard());
        card::Card cross = StraightCard();
        cross.cardType = card::kCross;
        cross.widthSegments = 2;
        cross.points.push_back(GfVec3f(0.3f, 0, 0));
        add(cross);
        card::Card tube = StraightCard();
        tube.cardType = card::kTube;
        tube.points = {GfVec3f(0, 0, 0), GfVec3f(0.1f, 0, 0)};
        add(tube);
        uint64_t v = 0, x = 0;
        card::MeshCounts(all, &v, &x);
        Check(v == 18 + 78 + 65 && x == 48 + 288 + 288,
              "render-mesh counts " + std::to_string(v) + "/" + std::to_string(x));
    }
    {
        // Free orientation on a card along +X: side from the chord, same answer.
        card::Card c = StraightCard();
        c.orientation = card::kFree;
        auto spans = g.Build(c);
        Check(Near(spans[0].V[0], GfVec3f(0, 0, -0.005f)), "Free side = chord x N0");
        c.orientation = card::kConform;
        spans = g.Build(c);
        Check(Near(spans[0].V[0], GfVec3f(0, 0, -0.005f)) && Near(spans[0].N[0], GfVec3f(0, 1, 0)),
              "Conform on a flat scalp keeps the card flat");
        c.orientation = card::kMeshNormal;
        c.points = {GfVec3f(0, 0, 0), GfVec3f(0.1f, 0.05f, 0), GfVec3f(0.15f, 0.12f, 0)};
        spans = g.Build(c);
        bool fixedSide = true;
        // N0 = +Y, chord in the XY plane: S0 = chord x N0 = +Z, and every row keeps it.
        for (uint32_t i = 0; i < spans[0].rows; ++i) {
            GfVec3f const side = spans[0].V[i * 2 + 1] - spans[0].V[i * 2];
            fixedSide = fixedSide && Near(strand::Normalize(side, GfVec3f(0.0f)), GfVec3f(0, 0, 1), 1e-4f);
        }
        if (!fixedSide) std::printf("  side row 0 %s\n", Str(spans[0].V[1] - spans[0].V[0]).c_str());
        Check(fixedSide, "Mesh Normal keeps one sideways direction on a bending card");
    }
    {
        // Mirror twins about x = 0.1 (the bbox centre): the secondary (larger id) takes the
        // twin's root frame mirrored, so its grid is the mirror image of the primary's, its
        // normal sign flips back to the mirrored normal, and its U is not flipped.
        card::Card a = StraightCard(10), b = StraightCard(20);
        a.points = {GfVec3f(0, 0, 0), GfVec3f(-0.1f, 0, 0), GfVec3f(-0.2f, 0, 0)};
        b.points = {GfVec3f(0.2f, 0, 0), GfVec3f(0.3f, 0, 0), GfVec3f(0.4f, 0, 0)};
        a.twinId = 20;
        b.twinId = 10;
        b.rootTri = -1;   // the secondary reads the twin's binding
        a.twistRoot = b.twistRoot = 30.0f;
        a.Sanitize();
        b.Sanitize();
        g.cards = {a, b};
        g.ctx.smooth = true;
        auto sa = card::BuildCard(0, g.ctx), sb = card::BuildCard(1, g.ctx);
        bool mirrored = sa.size() == 1 && sb.size() == 1;
        for (size_t v = 0; mirrored && v < sa[0].V.size(); ++v) {
            GfVec3f const m(0.2f - sa[0].V[v][0], sa[0].V[v][1], sa[0].V[v][2]);
            GfVec3f const n(-sa[0].N[v][0], sa[0].N[v][1], sa[0].N[v][2]);
            mirrored = Near(sb[0].V[v], m) && Near(sb[0].N[v], n);
        }
        Check(mirrored, "mirror-secondary grid and normals are the twin's mirror image (twist 30 deg)");
        Check(sa[0].flipU && !sb[0].flipU, "the mirror-secondary strip is not U-flipped");
    }
    {
        card::Card c = StraightCard();
        c.erased = true;
        Check(g.Build(c).empty(), "erased cards build nothing");
        c = StraightCard();
        c.distributionMask = 0.4f;
        Check(g.Build(c).empty(), "a card masked below 0.5 builds nothing");
        c = StraightCard();
        c.uvId = -1;
        Check(g.Build(c).empty(), "a card without a UV set builds nothing");
        c = StraightCard();
        c.cardType = card::kBraid;
        Check(g.Build(c).empty(), "braid cards build no grid");
    }
}

// ---------------------------------------------------------------------------
// (2) the generator
// ---------------------------------------------------------------------------

static UsdGenAuthoredPlaneDesc IntPlane(char const *name, std::vector<int> const &v)
{
    UsdGenAuthoredPlaneDesc p;
    p.name = TfToken(name);
    p.type = UsdGenAuthoredPlaneType::Int32;
    p.domain = UsdGenAuthoredPlaneDomain::Primitive;
    p.arity = 1;
    p.intValues = VtIntArray(v.begin(), v.end());
    return p;
}
static UsdGenAuthoredPlaneDesc FloatPlane(char const *name, std::vector<float> const &v)
{
    UsdGenAuthoredPlaneDesc p;
    p.name = TfToken(name);
    p.type = UsdGenAuthoredPlaneType::Float32;
    p.domain = UsdGenAuthoredPlaneDomain::Primitive;
    p.arity = 1;
    p.floatValues = VtFloatArray(v.begin(), v.end());
    return p;
}

struct SpineSpec
{
    std::vector<GfVec3f> points;
    int id, layer, uvId, tile, tile2, type;
    int erased = 0;
    int rootTri = -1;   // fan triangle whose corner 0 is the root (bary (1, 0, 0))
};

static UsdGenCurveSetDesc SpineSet(std::vector<SpineSpec> const &cards)
{
    UsdGenCurveSetDesc set;
    set.path = SdfPath("/cards/data/cards");
    set.role = UsdGenRole::Reference;
    set.type = TfToken("linear");
    set.wrap = TfToken("nonperiodic");
    std::vector<int> idLo, idHi, layer, uvId, tile, tile2, type, erased, segs;
    std::vector<float> width, vBias;
    std::vector<int> rootTri;
    UsdGenAuthoredPlaneDesc bary = FloatPlane("rootBary", {});
    bary.arity = 3;
    for (SpineSpec const &c : cards) {
        set.curveVertexCounts.push_back(int(c.points.size()));
        for (GfVec3f const &p : c.points) set.points.push_back(p);
        idLo.push_back(c.id); idHi.push_back(0); layer.push_back(c.layer); uvId.push_back(c.uvId);
        tile.push_back(c.tile); tile2.push_back(c.tile2); type.push_back(c.type);
        erased.push_back(c.erased); segs.push_back(1); width.push_back(0.02f); vBias.push_back(0.5f);
        rootTri.push_back(c.rootTri);
        for (float b : {1.0f, 0.0f, 0.0f}) bary.floatValues.push_back(b);
    }
    set.rest = set.points;
    for (auto p : {IntPlane("cardIdLo", idLo), IntPlane("cardIdHi", idHi), IntPlane("layer", layer),
                   IntPlane("uvId", uvId), IntPlane("tile", tile), IntPlane("tile2", tile2),
                   IntPlane("cardType", type), IntPlane("erased", erased),
                   IntPlane("widthSegments", segs), FloatPlane("width", width),
                   FloatPlane("vBias", vBias), IntPlane("rootTri", rootTri), bary})
        set.authoredPlanes.push_back(p);
    return set;
}

struct LibStrand { int uvId, tile, index, count; float x; };

static UsdGenCurveSetDesc LibrarySet(std::vector<LibStrand> const &strands)
{
    UsdGenCurveSetDesc set;
    set.path = SdfPath("/cards/data/cardStrands_0000000000000001");
    set.role = UsdGenRole::Reference;
    set.type = TfToken("linear");
    set.wrap = TfToken("nonperiodic");
    std::vector<int> uv, tile, index, count;
    for (LibStrand const &s : strands) {
        set.curveVertexCounts.push_back(5);
        for (int k = 0; k < 5; ++k) set.points.push_back(GfVec3f(s.x, -0.5f + 0.25f * float(k), 0.1f));
        uv.push_back(s.uvId); tile.push_back(s.tile); index.push_back(s.index); count.push_back(s.count);
    }
    set.rest = set.points;
    for (auto p : {IntPlane("uvId", uv), IntPlane("tile", tile), IntPlane("index", index),
                   IntPlane("tileStrandCount", count)})
        set.authoredPlanes.push_back(p);
    return set;
}

struct Fixture
{
    UsdGenGraphDesc desc;
    Fixture()
    {
        desc.description = SdfPath("/cards");
        desc.terminal = SdfPath("/cards/source");
        desc.surfaces = {FlatScalp()};
        std::vector<GfVec3f> const along = {GfVec3f(0, 0, 0), GfVec3f(0.1f, 0, 0), GfVec3f(0.2f, 0, 0)};
        std::vector<GfVec3f> const cross = {GfVec3f(0, 0, 0.05f), GfVec3f(0.1f, 0, 0.05f),
                                            GfVec3f(0.2f, 0, 0.05f)};
        // Spine order puts the layer-1 card first: emission order is by layer.
        desc.curveSets = {
            SpineSet({{cross, 2000, 1, 3, 2, 2, card::kCross, 0, 80},
                      {along, 1000, 0, 3, 1, 1, card::kRibbon, 0, 56},
                      {along, 3000, 0, 3, 5, 5, card::kRibbon},        // no library entry
                      {along, 4000, 0, 3, 1, 1, card::kRibbon, 1},     // erased
                      {along, 5000, 0, -1, 1, 1, card::kRibbon}}),     // no UV set
            LibrarySet({{3, 1, 0, 3, -0.25f}, {3, 1, 1, 3, 0.0f}, {3, 1, 2, 3, 0.3f},
                        {3, 2, 0, 3, 0.1f}, {3, 2, 2, 3, -0.1f}})};
        UsdGenNodeDesc node;
        node.path = desc.terminal;
        node.type = TfToken("UsdGenCardSource");
        node.surfaces = {desc.surfaces[0].path};
        node.curves = {desc.curveSets[0].path, desc.curveSets[1].path};
        node.params = {{TfToken("cvCount"), VtValue(6), false},
                       {TfToken("cardSource:rootFlatten"), VtValue(0.5f), false}};
        desc.nodes = {node};
    }
    void Set(char const *name, VtValue const &v)
    {
        auto &params = desc.nodes[0].params;
        for (auto &p : params)
            if (p.name == TfToken(name)) { p.value = v; return; }
        params.push_back({TfToken(name), v, false});
    }
};

static bool Cook(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *out, std::string *errors = nullptr)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult compiled = compiler.Compile(desc, &graph);
    if (!compiled.ok) {
        for (auto const &e : compiled.errors) {
            std::printf("  compile: %s\n", e.c_str());
            if (errors) *errors += e;
        }
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    UsdGenRunResult run = scheduler.Run(graph, ctx, 1);
    if (run.diagnostics.HasErrors()) {
        for (auto const &e : run.diagnostics.errors) {
            std::printf("  run: %s\n", e.c_str());
            if (errors) *errors += e;
        }
        return false;
    }
    *out = graph.Output();
    return true;
}

static UsdGenPlane const *Plane(UsdGenCurveBuffer const &b, char const *name)
{
    for (UsdGenPlane const &p : b.extraCurve)
        if (p.name == TfToken(name)) return &p;
    return nullptr;
}

static uint64_t SM(uint64_t z) { return strand::SplitMix64(z); }
static uint64_t Id(uint64_t card, uint64_t j, int strip)
{
    return SM(SM(card) ^ j ^ (uint64_t(strip + 1) << 40)) | 1ull;
}

static void TestGenerator()
{
    Fixture f;
    UsdGenCurveBuffer out;
    bool const ok = Cook(f.desc, &out);
    Check(ok, "card source cooks");
    if (!ok) return;
    // card 1000 (layer 0, tile 1): 3 strands; cross 2000 (layer 1, tile 2): 2 strands x 2 strips.
    Check(out.totalCurves == 7 && out.totalCvs == 42 && out.cvOffsets.empty(),
          "7 strands of 6 CVs (" + std::to_string(out.totalCurves) + ")");
    UsdGenPlane const *index = Plane(out, "strandIndex"), *seed = Plane(out, "strandSeed"),
                      *cardIndex = Plane(out, "cardIndex");
    Check(index && seed && cardIndex && index->type == TfToken("int"),
          "strandIndex / strandSeed / cardIndex planes");
    if (!index || !seed || !cardIndex || out.totalCurves != 7) return;
    std::vector<uint64_t> const expected = {Id(1000, 0, 0), Id(1000, 1, 0), Id(1000, 2, 0),
                                            Id(2000, 0, 0), Id(2000, 2, 0), Id(2000, 0, 1),
                                            Id(2000, 2, 1)};
    bool order = true, seeds = true, cardsOk = true, hairT = true;
    std::set<int> indices;
    for (uint32_t c = 0; c < out.totalCurves; ++c) {
        int const k = index->i[c];
        indices.insert(k);
        order = order && k >= 0 && k < 7 && out.curveId[c] == expected[size_t(k)];
        seeds = seeds && seed->i[c] == int(SM(out.curveId[c]) & 0x7fffffffull);
        cardsOk = cardsOk && cardIndex->i[c] == (k < 3 ? 1 : 0);
        for (int i = 0; i < 6; ++i) hairT = hairT && Near(out.hairT[c * 6 + i], float(i) / 5.0f);
    }
    Check(order && indices.size() == 7,
          "ids = splitmix64(splitmix64(cardId) ^ j ^ (strip+1) << 40) | 1, generation order by layer");
    Check(seeds, "strandSeed = splitmix64(id) & 0x7fffffff");
    Check(cardsOk, "cardIndex names the spine curve");
    Check(hairT, "hairT = i / (n - 1)");

    // Strand j = 2 of card 1000: tile x 0.3 -> -S side: z = -0.3 * 2 * hw = -0.003;
    // depth 0.1 * (1 - 0.5) row widths (0.01) -> +0.0005 along +Y, snapped to y = 0 at the root.
    for (uint32_t c = 0; c < out.totalCurves; ++c) {
        if (index->i[c] != 2) continue;
        GfVec3f const root(out.px[c * 6], out.py[c * 6], out.pz[c * 6]);
        GfVec3f const tip(out.px[c * 6 + 5], out.py[c * 6 + 5], out.pz[c * 6 + 5]);
        Check(Near(root, GfVec3f(0, 0, -0.003f)), "root snapped onto the scalp " + Str(root));
        Check(Near(tip, GfVec3f(0.2f, 0.0005f, -0.003f)), "tip on the card, pushed by depth " + Str(tip));
        GfVec3f const mid(out.px[c * 6 + 3], out.py[c * 6 + 3], out.pz[c * 6 + 3]);
        Check(Near(mid[1], 0.0005f) && mid[0] > 0.11f && mid[0] < 0.13f,
              "body keeps the card depth after the root snap " + Str(mid));
        // x = 0 is the edge between faces 15 and 16 (row iz = 1 of 12 quads).
        Check(out.rootPrim[c] == 15 || out.rootPrim[c] == 16,
              "rootPrim is the scalp face under the root (" + std::to_string(out.rootPrim[c]) + ")");
        Check(Near(out.rootN[c], GfVec3f(0, 1, 0)) && Near(GfDot(out.rootT[c], out.rootN[c]), 0.0f) &&
              Near(out.rootB[c], GfCross(out.rootN[c], out.rootT[c])),
              "root frame: N = face normal, T _|_ N, B = N x T");
        Check(Near(out.rootUV[c][0], 0.0f) && Near(out.rootUV[c][1], -0.003f), "rootUV at the snapped root");
        Check(Near(out.rest[c * 6 + 5], tip), "rest = generated points");
    }
    // Cross strip 1 stands upright: its strands leave the scalp.
    float maxY = 0.0f;
    for (uint32_t c = 0; c < out.totalCurves; ++c)
        if (index->i[c] >= 5) maxY = std::max(maxY, out.py[c * 6 + 5]);
    Check(maxY > 0.0f, "cross strip-1 strands stand off the scalp");

    UsdGenCurveBuffer again;
    Cook(f.desc, &again);
    Check(again.px == out.px && again.py == out.py && again.pz == out.pz && again.curveId == out.curveId,
          "two cooks are identical");
    Fixture swapped;
    std::swap(swapped.desc.nodes[0].curves[0], swapped.desc.nodes[0].curves[1]);
    UsdGenCurveBuffer s2;
    Check(Cook(swapped.desc, &s2) && s2.curveId == out.curveId && s2.px == out.px,
          "the spine and library sets are told apart by planes, not order");

    auto count = [](Fixture &fx) {
        UsdGenCurveBuffer b;
        return Cook(fx.desc, &b) ? int(b.totalCurves) : -1;
    };
    { Fixture g; g.Set("cardSource:layer", VtValue(1)); Check(count(g) == 4, "layer filter 1 -> cross only"); }
    { Fixture g; g.Set("cardSource:tile", VtValue(1)); Check(count(g) == 3, "tile filter 1 -> ribbon only"); }
    { Fixture g; g.Set("cardSource:uvId", VtValue(2)); Check(count(g) == 0, "uvId filter 2 -> nothing"); }
    { Fixture g; g.Set("cardSource:braid", VtValue(true)); Check(count(g) == 0, "braid source: no braid cards"); }
    {
        // A braid card on tile 1 (entry of 3 strands): n = min(max(3, round(3 * 3 * amount)),
        // max(3, maxStrands / braidCards)) strands, ids splitmix64(splitmix64(id) ^ k ^ 1 << 40) | 1,
        // roots spread on the R2 sequence around the spine root and snapped to the scalp.
        Fixture g;
        g.Set("cardSource:braid", VtValue(true));
        g.desc.curveSets[0] = SpineSet({{{GfVec3f(0, 0, 0), GfVec3f(0.1f, 0, 0), GfVec3f(0.2f, 0, 0)},
                                         6000, 0, 3, 1, 1, card::kBraid, 0, 56},
                                        {{GfVec3f(0, 0, 0), GfVec3f(0.1f, 0, 0)}, 7000, 0, 3, 1, 1,
                                         card::kRibbon, 0, 56}});
        UsdGenCurveBuffer b;
        bool const cooked = Cook(g.desc, &b);
        Check(cooked && b.totalCurves == 9, "braid source grows 3 x 3 strands on the braid card only (" +
              std::to_string(b.totalCurves) + ")");
        std::set<uint64_t> ids(b.curveId.begin(), b.curveId.end()), want;
        for (uint64_t q = 0; q < 9; ++q) want.insert(Id(6000, q, 0));
        bool onScalp = true, spread = false;
        for (uint32_t c = 0; c < b.totalCurves; ++c) {
            onScalp = onScalp && std::fabs(b.py[c * 6]) < 1e-6f;
            spread = spread || std::fabs(b.pz[c * 6]) > 0.005f;
        }
        Check(ids == want, "braid strand ids");
        Check(onScalp && spread, "braid roots spread around the card root, on the scalp");
        g.Set("cardSource:amount", VtValue(0.2f));
        Check(count(g) == 3, "braid strand count never drops below 3");
    }
    {
        auto keep = [&](uint64_t id, double frac) {
            return !(double(SM(id ^ 0x5a17c4ull) >> 11) * (1.0 / 9007199254740992.0) >= frac);
        };
        int expectHalf = 0, expectCap = 0;
        for (uint64_t id : expected) {
            expectHalf += keep(id, 0.5) ? 1 : 0;
            expectCap += keep(id, 2.0 / 9.0) ? 1 : 0;   // total entry strands 3 + 2 x 3
        }
        Fixture g; g.Set("cardSource:amount", VtValue(0.5f));
        Check(count(g) == expectHalf, "amount 0.5 keeps ids with U(splitmix64(id ^ 0x5a17c4)) < 0.5 (" +
              std::to_string(expectHalf) + ")");
        Fixture h; h.Set("cardSource:maxStrands", VtValue(2));
        Check(count(h) == expectCap, "maxStrands 2 of 9 keeps the 2/9 fraction (" +
              std::to_string(expectCap) + ")");
    }
    {
        Fixture g; g.Set("cvCount", VtValue(1));
        UsdGenCurveBuffer b;
        std::string errors;
        Check(!Cook(g.desc, &b, &errors), "cvCount 1 is rejected");
    }
}

// ---------------------------------------------------------------------------
// Harness: cook the UsdGenCardSource prims of a flat .usda layer.
// ---------------------------------------------------------------------------

static VtValue Default(SdfLayerHandle const &layer, SdfPath const &prim, std::string const &name,
                       TfToken *interpolation = nullptr)
{
    SdfAttributeSpecHandle const a = layer->GetAttributeAtPath(prim.AppendProperty(TfToken(name)));
    if (!a) return VtValue();
    if (interpolation) {
        VtValue const i = a->GetInfo(TfToken("interpolation"));
        *interpolation = i.IsHolding<TfToken>() ? i.UncheckedGet<TfToken>() : TfToken();
    }
    return a->GetDefaultValue();
}

template <class T>
static T Get(SdfLayerHandle const &layer, SdfPath const &prim, std::string const &name)
{
    VtValue const v = Default(layer, prim, name);
    return v.IsHolding<T>() ? v.UncheckedGet<T>() : T();
}

static bool ReadCurveSet(SdfLayerHandle const &layer, SdfPath const &path, UsdGenCurveSetDesc *out)
{
    SdfPrimSpecHandle const prim = layer->GetPrimAtPath(path);
    if (!prim) return false;
    out->path = path;
    out->role = UsdGenRole::Reference;
    out->type = TfToken("linear");
    out->wrap = TfToken("nonperiodic");
    out->curveVertexCounts = Get<VtIntArray>(layer, path, "curveVertexCounts");
    out->points = Get<VtVec3fArray>(layer, path, "points");
    out->rest = out->points;
    size_t total = 0;
    for (int c : out->curveVertexCounts) total += size_t(std::max(c, 0));
    std::string const prefix = "primvars:" + UsdGenAuthoredPlanePrefix();
    for (SdfAttributeSpecHandle const &a : prim->GetAttributes()) {
        std::string const name = a->GetName();
        if (name.compare(0, prefix.size(), prefix) != 0 || name.find(":indices") != std::string::npos)
            continue;
        TfToken interpolation;
        VtValue const value = Default(layer, path, name, &interpolation);
        UsdGenAuthoredPlaneDesc plane;
        if (ConvertAuthoredPlane(TfToken(name.substr(prefix.size())), value, interpolation,
                                 out->curveVertexCounts.size(), total, &plane))
            out->authoredPlanes.push_back(std::move(plane));
    }
    return true;
}

static bool ReadMesh(SdfLayerHandle const &layer, SdfPath const &path, UsdGenSurfaceDesc *out)
{
    out->path = path;
    out->faceVertexCounts = Get<VtIntArray>(layer, path, "faceVertexCounts");
    out->faceVertexIndices = Get<VtIntArray>(layer, path, "faceVertexIndices");
    out->restPoints = Get<VtVec3fArray>(layer, path, "points");
    out->points = out->restPoints;
    TfToken interpolation;
    VtValue normals = Default(layer, path, "primvars:normals", &interpolation);
    if (normals.IsEmpty()) {
        normals = Default(layer, path, "normals", &interpolation);
        if (interpolation.IsEmpty()) interpolation = TfToken("vertex");
    }
    if (normals.IsHolding<VtVec3fArray>()) {
        out->restNormals = normals.UncheckedGet<VtVec3fArray>();
        out->restNormalDomain = interpolation == TfToken("faceVarying")
            ? UsdGenSurfaceNormalDomain::FaceVarying : UsdGenSurfaceNormalDomain::Vertex;
    }
    VtValue const st = Default(layer, path, "primvars:st");
    if (st.IsHolding<VtVec2fArray>()) out->uv = st.UncheckedGet<VtVec2fArray>();
    out->surfaceGeneration = 1;
    return !out->faceVertexCounts.empty() && !out->restPoints.empty();
}

static SdfPath FirstTarget(SdfLayerHandle const &layer, SdfPath const &prim, char const *rel)
{
    SdfRelationshipSpecHandle const r = layer->GetRelationshipAtPath(prim.AppendProperty(TfToken(rel)));
    if (!r) return SdfPath();
    auto const items = r->GetTargetPathList().GetAppliedItems();
    return items.empty() ? SdfPath() : items.front();
}

/// The reference buffer the compiler builds for a curve set (points + uniform planes).
static UsdGenCurveBuffer BufferFromSet(UsdGenCurveSetDesc const &set)
{
    UsdGenCurveBuffer b;
    b.totalCurves = uint32_t(set.curveVertexCounts.size());
    b.totalCvs = uint32_t(set.points.size());
    b.cvOffsets = VtIntArray(set.curveVertexCounts.size() + 1, 0);
    for (size_t c = 0; c < set.curveVertexCounts.size(); ++c)
        b.cvOffsets[c + 1] = b.cvOffsets[c] + set.curveVertexCounts[c];
    for (GfVec3f const &p : set.points) {
        b.px.push_back(p[0]); b.py.push_back(p[1]); b.pz.push_back(p[2]);
    }
    for (UsdGenAuthoredPlaneDesc const &a : set.authoredPlanes) {
        if (a.domain != UsdGenAuthoredPlaneDomain::Primitive) continue;
        UsdGenPlane p;
        p.name = a.name;
        p.interpolation = TfToken("uniform");
        p.arity = a.arity;
        if (a.type == UsdGenAuthoredPlaneType::Int32) { p.type = TfToken("int"); p.i = a.intValues; }
        else { p.type = TfToken("float"); p.f = a.floatValues; }
        b.extraCurve.push_back(p);
    }
    return b;
}

static int Harness(std::string const &usda, std::string const &dumpPath)
{
    SdfLayerRefPtr const layer = SdfLayer::FindOrOpen(usda);
    if (!layer) { std::printf("cannot open %s\n", usda.c_str()); return 2; }
    std::vector<SdfPath> sources;
    SdfPath meshPath;
    layer->Traverse(SdfPath::AbsoluteRootPath(), [&](SdfPath const &p) {
        if (!p.IsPrimPath()) return;
        SdfPrimSpecHandle const prim = layer->GetPrimAtPath(p);
        if (!prim) return;
        if (prim->GetTypeName() == TfToken("UsdGenCardSource")) sources.push_back(p);
        if (prim->GetTypeName() == TfToken("Mesh") && meshPath.IsEmpty()) meshPath = p;
    });
    std::sort(sources.begin(), sources.end());
    UsdGenSurfaceDesc surface;
    if (meshPath.IsEmpty() || !ReadMesh(layer, meshPath, &surface)) {
        std::printf("no Mesh in %s\n", usda.c_str());
        return 2;
    }
    FILE *dump = dumpPath.empty() ? nullptr : std::fopen(dumpPath.c_str(), "w");
    bool meshDone = false;
    for (SdfPath const &src : sources) {
        UsdGenGraphDesc desc;
        desc.description = src.GetParentPath();
        desc.terminal = src;
        desc.surfaces = {surface};
        UsdGenCurveSetDesc spine, lib;
        SdfPath const cardsPath = FirstTarget(layer, src, "usdGen:cards");
        SdfPath const libPath = FirstTarget(layer, src, "usdGen:cardStrands");
        if (!ReadCurveSet(layer, cardsPath, &spine) || !ReadCurveSet(layer, libPath, &lib)) {
            std::printf("set %s: missing card inputs\n", src.GetText());
            continue;
        }
        desc.curveSets = {spine, lib};
        UsdGenNodeDesc node;
        node.path = src;
        node.type = TfToken("UsdGenCardSource");
        node.surfaces = {surface.path};
        node.curves = {cardsPath, libPath};
        SdfPrimSpecHandle const prim = layer->GetPrimAtPath(src);
        for (SdfAttributeSpecHandle const &a : prim->GetAttributes()) {
            std::string const name = a->GetName();
            if (name.compare(0, 7, "usdGen:") != 0) continue;
            node.params.push_back({TfToken(name.substr(7)), a->GetDefaultValue(), false});
        }
        desc.nodes = {node};

        if (!meshDone) {
            // Render-mesh size of every card: smoothed grids, every non-erased
            // card (the viewport rule; card visibility is not transported).
            meshDone = true;
            UsdGenCurveBuffer const b = BufferFromSet(spine);
            std::vector<card::Card> cards;
            std::string error;
            if (!card::DecodeCards(b, &cards, &error)) {
                std::printf("cannot decode cards: %s\n", error.c_str());
                continue;
            }
            strand::TriangleBvh bvh;
            strand::BuildSurfaceBvh(surface, true, &bvh);
            card::Context ctx;
            ctx.scalp = &bvh;
            ctx.cards = &cards;
            ctx.smooth = true;
            std::vector<card::Span> spans;
            size_t built = 0;
            for (uint32_t c = 0; c < cards.size(); ++c) {
                if (cards[c].erased) continue;
                auto s = card::BuildCard(c, ctx, false);
                built += s.empty() ? 0 : 1;
                spans.insert(spans.end(), std::make_move_iterator(s.begin()), std::make_move_iterator(s.end()));
            }
            uint64_t vertices = 0, indices = 0;
            card::MeshCounts(spans, &vertices, &indices);
            std::printf("mesh cards=%zu spans=%zu vertices=%" PRIu64 " indices=%" PRIu64 "\n", built,
                        spans.size(), vertices, indices);
        }
        auto const t0 = std::chrono::steady_clock::now();
        UsdGenCurveBuffer out;
        if (!Cook(desc, &out)) { std::printf("set %s: cook failed\n", src.GetText()); continue; }
        double const ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("set %s strands=%u cvs=%u (%.0f ms)\n", src.GetText(), out.totalCurves,
                    out.totalCurves ? out.totalCvs / out.totalCurves : 0u, ms);
        if (dump) {
            UsdGenPlane const *index = Plane(out, "strandIndex"), *cardIndex = Plane(out, "cardIndex");
            uint32_t const n = out.totalCurves ? out.totalCvs / out.totalCurves : 0;
            for (uint32_t c = 0; c < out.totalCurves; ++c) {
                size_t const r = size_t(c) * n, t = r + n - 1;
                std::fprintf(dump, "%s %016" PRIx64 " %d %d %.9g %.9g %.9g %.9g %.9g %.9g\n",
                             src.GetName().c_str(), out.curveId[c], index ? index->i[c] : -1,
                             cardIndex ? cardIndex->i[c] : -1, out.px[r], out.py[r], out.pz[r],
                             out.px[t], out.py[t], out.pz[t]);
            }
        }
    }
    if (dump) std::fclose(dump);
    return 0;
}

int main(int argc, char **argv)
{
    std::string usda, dumpPath;
    for (int i = 1; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--usda")) usda = argv[++i];
        else if (!std::strcmp(argv[i], "--dump")) dumpPath = argv[++i];
    }
    usdGenRegisterM1Operators();
    if (!usda.empty()) return Harness(usda, dumpPath);
    TestGeometry();
    TestGenerator();
    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
