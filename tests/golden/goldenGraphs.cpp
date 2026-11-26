// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// In-code golden graphs. Every array is a closed-form function of its
// indices so the descriptors are identical on every run and platform.
#include "golden/goldenGraphs.h"
#include "usdGenUsd/lower.h"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec4d.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/value.h"

#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace usdGenGolden {
namespace {

constexpr int kSkinFaces = 8;          // skin: 8x8 quads over [0,1]^2 in XZ
constexpr int kHairSide = 8;           // one strand over each skin face
constexpr int kHairCvs = 8;
constexpr int kGuideSide = 3;
constexpr int kGuideCvs = 6;

SdfPath P(char const *leaf) { return SdfPath(std::string("/g/") + leaf); }

UsdGenParamValue Param(char const *name, VtValue const &value)
{
    return UsdGenParamValue{TfToken(name), value, false};
}

float Bump(float x, float z) { return 0.05f * std::sin(3.0f * x) * std::cos(2.0f * z); }

/// A Y-up grid mesh (normal +Y) with a gentle bump; \`deform\` adds a
/// time-free wave to the current points only.
UsdGenSurfaceDesc Skin(char const *leaf, int faces, float size, bool deform)
{
    UsdGenSurfaceDesc s;
    s.path = P(leaf);
    s.id = 0;
    float const h = size / float(faces);
    for (int j = 0; j <= faces; ++j)
        for (int i = 0; i <= faces; ++i) {
            float const x = h * float(i), z = h * float(j);
            s.restPoints.push_back(GfVec3f(x, Bump(x, z), z));
            s.uv.push_back(GfVec2f(float(i) / float(faces), float(j) / float(faces)));
        }
    auto at = [&](int i, int j) { return j * (faces + 1) + i; };
    for (int j = 0; j < faces; ++j)
        for (int i = 0; i < faces; ++i) {
            s.faceVertexCounts.push_back(4);
            for (int v : {at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j)})
                s.faceVertexIndices.push_back(v);
        }
    s.points = s.restPoints;
    if (deform)
        for (size_t k = 0; k < s.points.size(); ++k) {
            GfVec3f &p = s.points[k];
            p += GfVec3f(0.04f * std::sin(2.0f * p[2]), 0.12f * p[0] * p[2],
                         0.03f * std::cos(3.0f * p[0]));
        }
    s.surfaceGeneration = deform ? 2 : 1;
    return s;
}

UsdGenAuthoredPlaneDesc IntPlane(char const *name, std::vector<int> const &v,
                                 UsdGenAuthoredPlaneDomain domain =
                                     UsdGenAuthoredPlaneDomain::Primitive,
                                 uint8_t arity = 1)
{
    UsdGenAuthoredPlaneDesc p;
    p.name = TfToken(name);
    p.type = UsdGenAuthoredPlaneType::Int32;
    p.domain = domain;
    p.arity = arity;
    p.intValues = VtIntArray(v.begin(), v.end());
    return p;
}

UsdGenAuthoredPlaneDesc FloatPlane(char const *name, std::vector<float> const &v,
                                   UsdGenAuthoredPlaneDomain domain =
                                       UsdGenAuthoredPlaneDomain::Primitive,
                                   uint8_t arity = 1)
{
    UsdGenAuthoredPlaneDesc p;
    p.name = TfToken(name);
    p.type = UsdGenAuthoredPlaneType::Float32;
    p.domain = domain;
    p.arity = arity;
    p.floatValues = VtFloatArray(v.begin(), v.end());
    return p;
}

GfVec3f GuideRoot(int g)
{
    return GfVec3f(0.2f + 0.3f * float(g % kGuideSide), 0.0f, 0.2f + 0.3f * float(g / kGuideSide));
}

/// 8x8 strands, one per skin face, each with its own lean and droop (some
/// dip below the skin), plus the planes the whole-strand stylers read.
UsdGenCurveSetDesc Hair(bool withPlanes)
{
    UsdGenCurveSetDesc set;
    set.path = P("hair");
    set.role = UsdGenLane::Chunked;
    set.curveRole = TfToken("hair");
    int const n = kHairSide * kHairSide;
    set.curveVertexCounts.assign(size_t(n), kHairCvs);
    std::vector<int> primary, mirror, seeds, index;
    std::vector<float> restLength, knot;
    for (int c = 0; c < n; ++c) {
        float const x = (float(c % kHairSide) + 0.5f) / float(kHairSide);
        float const z = (float(c / kHairSide) + 0.5f) / float(kHairSide);
        GfVec3f const root(x, Bump(x, z), z);
        float const a = 0.7f * float(c), droop = 0.2f + 0.04f * float(c % 31);
        float length = 0.0f;
        GfVec3f prev = root;
        for (int i = 0; i < kHairCvs; ++i) {
            float const t = float(i) / float(kHairCvs - 1);
            GfVec3f const p = root + GfVec3f(0.25f * t * std::cos(a),
                                             0.6f * t - droop * t * t,
                                             0.25f * t * std::sin(a));
            length += float((p - prev).GetLength());
            prev = p;
            set.points.push_back(p);
            knot.push_back(0.5f + 0.5f * std::sin(3.0f * t + float(c)));
        }
        set.curveId.push_back(uint64_t(1000 + 17 * c));
        set.skinPrim.push_back(c);
        set.skinPrimUv.push_back(GfVec2f(0.5f, 0.5f));
        GfMatrix4d frame(1.0);
        frame.SetRow(0, GfVec4d(1, 0, 0, 0));
        frame.SetRow(1, GfVec4d(0, 0, -1, 0));
        frame.SetRow(2, GfVec4d(0, 1, 0, 0));
        frame.SetRow(3, GfVec4d(root[0], root[1], root[2], 1));
        set.rootFrame.push_back(frame);
        // nearest guide on the 3x3 lattice
        int best = 0;
        float bestD = 1e30f;
        for (int g = 0; g < kGuideSide * kGuideSide; ++g) {
            float const d = float((GuideRoot(g) - GfVec3f(x, 0.0f, z)).GetLengthSq());
            if (d < bestD) { bestD = d; best = g; }
        }
        primary.push_back(best);
        mirror.push_back(c % 2 ? -1 : 1);
        seeds.push_back(7 * c + 3);
        index.push_back(c);
        restLength.push_back(length);
    }
    set.rest = set.points;
    set.widths.assign(set.points.size(), 0.01f);
    if (withPlanes) {
        set.authoredPlanes.push_back(IntPlane("primaryGuide", primary));
        set.authoredPlanes.push_back(IntPlane("nearestGuide", primary));
        set.authoredPlanes.push_back(IntPlane("mirrorSign", mirror));
        set.authoredPlanes.push_back(IntPlane("strandSeed", seeds));
        set.authoredPlanes.push_back(IntPlane("strandIndex", index));
        set.authoredPlanes.push_back(FloatPlane("restLength", restLength));
        set.authoredPlanes.push_back(
            FloatPlane("braidKnotMask", knot, UsdGenAuthoredPlaneDomain::Point));
    }
    set.curveGeneration = 1;
    return set;
}

/// 3x3 leaning guides (reference role).
UsdGenCurveSetDesc Guides(char const *leaf, float scale)
{
    UsdGenCurveSetDesc set;
    set.path = P(leaf);
    set.role = UsdGenLane::Reference;
    set.curveRole = TfToken("guide");
    int const n = kGuideSide * kGuideSide;
    set.curveVertexCounts.assign(size_t(n), kGuideCvs);
    for (int g = 0; g < n; ++g) {
        GfVec3f const root = GuideRoot(g);
        float const a = 1.3f * float(g);
        for (int i = 0; i < kGuideCvs; ++i) {
            float const t = float(i) / float(kGuideCvs - 1);
            set.points.push_back(root + scale * GfVec3f(0.15f * t * std::cos(a), 0.8f * t,
                                                        0.15f * t * std::sin(a)));
        }
        set.curveId.push_back(uint64_t(50 + g));
        set.skinPrim.push_back(g);
        set.skinPrimUv.push_back(GfVec2f(0.5f, 0.5f));
        set.rootFrame.push_back(GfMatrix4d(1.0));
    }
    set.rest = set.points;
    set.widths.assign(set.points.size(), 0.02f);
    set.curveGeneration = 1;
    return set;
}

/// Two-CV root stubs for GuideInterpolate, one per hair strand.
UsdGenCurveSetDesc Roots()
{
    UsdGenCurveSetDesc hair = Hair(false);
    UsdGenCurveSetDesc set;
    set.path = P("roots");
    set.role = UsdGenLane::Chunked;
    set.curveRole = TfToken("hair");
    size_t const n = hair.curveVertexCounts.size();
    set.curveVertexCounts.assign(n, 2);
    for (size_t c = 0; c < n; ++c) {
        GfVec3f const r = hair.points[c * kHairCvs];
        set.points.push_back(r);
        set.points.push_back(r + GfVec3f(0.0f, 0.01f, 0.0f));
    }
    set.rest = set.points;
    set.widths.assign(set.points.size(), 0.02f);
    set.curveId = hair.curveId;
    set.skinPrim = hair.skinPrim;
    set.skinPrimUv = hair.skinPrimUv;
    set.rootFrame = hair.rootFrame;
    set.curveGeneration = 1;
    return set;
}

UsdGenNodeDesc Node(char const *leaf, char const *type, int seed)
{
    UsdGenNodeDesc node;
    node.path = P(leaf);
    node.type = TfToken(type);
    node.seed = seed;
    return node;
}

UsdGenNodeDesc CurveSource(char const *leaf, SdfPath const &curves)
{
    UsdGenNodeDesc node = Node(leaf, "UsdGenCurveSource", 1);
    node.curves = {curves};
    node.params = {Param("rebind", VtValue(TfToken("never"))),
                   Param("useRest", VtValue(true))};
    return node;
}

UsdGenGraphDesc Base()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/g");
    desc.defaultWidth = 0.01f;
    desc.tileTarget = 4;
    desc.time = 0.0;
    desc.executionBackend = kUsdGenBackendCpu;
    return desc;
}

/// CurveSource over the hair, then \`op\` as terminal.
UsdGenGraphDesc OnHair(UsdGenNodeDesc op, bool withPlanes = true)
{
    UsdGenGraphDesc desc = Base();
    desc.curveSets.push_back(Hair(withPlanes));
    desc.nodes.push_back(CurveSource("source", P("hair")));
    op.inputs = {P("source")};
    desc.nodes.push_back(op);
    desc.terminal = op.path;
    return desc;
}

/// Scatter on the skin, Grow, then \`op\` as terminal (\`op\` empty: Grow).
UsdGenGraphDesc OnGrown(UsdGenNodeDesc const *op, bool deformSkin, int skinFaces = kSkinFaces,
                        float skinSize = 1.0f, double density = 100.0)
{
    UsdGenGraphDesc desc = Base();
    desc.surfaces.push_back(Skin("skin", skinFaces, skinSize, deformSkin));
    UsdGenNodeDesc scatter = Node("scatter", "UsdGenScatter", 42);
    scatter.surfaces = {P("skin")};
    scatter.params = {Param("density", VtValue(float(density)))};
    UsdGenNodeDesc grow = Node("grow", "UsdGenGrow", 43);
    grow.inputs = {P("scatter")};
    grow.params = {Param("segments", VtValue(6)), Param("length", VtValue(0.5f))};
    desc.nodes = {scatter, grow};
    desc.terminal = P("grow");
    if (op) {
        UsdGenNodeDesc node = *op;
        node.inputs = {P("grow")};
        desc.nodes.push_back(node);
        desc.terminal = node.path;
    }
    return desc;
}

/// A closed axis-aligned box (outward winding) the hair passes through.
UsdGenSurfaceDesc Box(char const *leaf, GfVec3f const &lo, GfVec3f const &hi)
{
    UsdGenSurfaceDesc s;
    s.path = P(leaf);
    for (int k = 0; k < 8; ++k)
        s.restPoints.push_back(GfVec3f(k & 1 ? hi[0] : lo[0], k & 2 ? hi[1] : lo[1],
                                       k & 4 ? hi[2] : lo[2]));
    s.points = s.restPoints;
    int const quads[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4},
                             {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
    for (auto const &q : quads) {
        s.faceVertexCounts.push_back(4);
        for (int v : q) s.faceVertexIndices.push_back(v);
    }
    s.surfaceGeneration = 1;
    return s;
}

VtValue Ramp(std::vector<GfVec2f> const &knots) { return VtValue(VtVec2fArray(knots.begin(), knots.end())); }

// ---- card fixture (mirrors testUsdGenCardSource) ----------------------------

UsdGenCurveSetDesc CardSpines()
{
    UsdGenCurveSetDesc set;
    set.path = P("cards");
    set.role = UsdGenLane::Reference;
    set.type = TfToken("linear");
    set.wrap = TfToken("nonperiodic");
    struct Spec { float x, z; int id, layer, uvId, tile, type; };
    Spec const cards[] = {{0.2f, 0.3f, 2000, 1, 3, 2, 1}, {0.5f, 0.5f, 1000, 0, 3, 1, 0},
                          {0.7f, 0.2f, 3000, 0, 3, 2, 0}, {0.4f, 0.8f, 4000, 0, 3, 1, 1}};
    std::vector<int> idLo, idHi, layer, uvId, tile, tile2, type, erased, segs, rootTri;
    std::vector<float> width, vBias, bary;
    for (Spec const &c : cards) {
        set.curveVertexCounts.push_back(3);
        for (int k = 0; k < 3; ++k)
            set.points.push_back(GfVec3f(c.x + 0.1f * float(k), Bump(c.x, c.z), c.z));
        idLo.push_back(c.id); idHi.push_back(0); layer.push_back(c.layer);
        uvId.push_back(c.uvId); tile.push_back(c.tile); tile2.push_back(c.tile);
        type.push_back(c.type); erased.push_back(0); segs.push_back(1);
        width.push_back(0.02f); vBias.push_back(0.5f); rootTri.push_back(-1);
        bary.insert(bary.end(), {1.0f, 0.0f, 0.0f});
    }
    set.rest = set.points;
    for (auto const &p : {IntPlane("cardIdLo", idLo), IntPlane("cardIdHi", idHi),
                          IntPlane("layer", layer), IntPlane("uvId", uvId),
                          IntPlane("tile", tile), IntPlane("tile2", tile2),
                          IntPlane("cardType", type), IntPlane("erased", erased),
                          IntPlane("widthSegments", segs), FloatPlane("width", width),
                          FloatPlane("vBias", vBias), IntPlane("rootTri", rootTri),
                          FloatPlane("rootBary", bary, UsdGenAuthoredPlaneDomain::Primitive, 3)})
        set.authoredPlanes.push_back(p);
    set.curveGeneration = 1;
    return set;
}

UsdGenCurveSetDesc CardLibrary()
{
    UsdGenCurveSetDesc set;
    set.path = P("cardStrands_0000000000000001");
    set.role = UsdGenLane::Reference;
    set.type = TfToken("linear");
    set.wrap = TfToken("nonperiodic");
    struct Lib { int uvId, tile, index, count; float x; };
    Lib const strands[] = {{3, 1, 0, 3, -0.25f}, {3, 1, 1, 3, 0.0f}, {3, 1, 2, 3, 0.3f},
                           {3, 2, 0, 3, 0.1f}, {3, 2, 2, 3, -0.1f}};
    std::vector<int> uv, tile, index, count;
    for (Lib const &s : strands) {
        set.curveVertexCounts.push_back(5);
        for (int k = 0; k < 5; ++k)
            set.points.push_back(GfVec3f(s.x, -0.5f + 0.25f * float(k), 0.1f));
        uv.push_back(s.uvId); tile.push_back(s.tile); index.push_back(s.index);
        count.push_back(s.count);
    }
    set.rest = set.points;
    for (auto const &p : {IntPlane("uvId", uv), IntPlane("tile", tile),
                          IntPlane("index", index), IntPlane("tileStrandCount", count)})
        set.authoredPlanes.push_back(p);
    set.curveGeneration = 1;
    return set;
}

// ---- per-op fixtures ---------------------------------------------------------

using Tuning = std::vector<UsdGenParamValue>;

void Tune(UsdGenNodeDesc *node, bool tuned, Tuning const &params)
{
    if (tuned) node->params.insert(node->params.end(), params.begin(), params.end());
}

UsdGenGraphDesc MakeOp(std::string const &type, bool tuned)
{
    using V = VtValue;
    UsdGenNodeDesc op = Node("op", type.c_str(), 7);
    if (type == "UsdGenScatter") {
        UsdGenGraphDesc desc = Base();
        desc.surfaces.push_back(Skin("skin", kSkinFaces, 1.0f, false));
        op.surfaces = {P("skin")};
        Tune(&op, tuned, {Param("density", V(220.0f)), Param("flip", V(true))});
        desc.nodes = {op};
        desc.terminal = op.path;
        return desc;
    }
    if (type == "UsdGenGrow") {
        UsdGenGraphDesc desc = OnGrown(nullptr, false);
        desc.nodes[1].path = OpPath();
        desc.terminal = OpPath();
        if (!tuned) desc.nodes[1].params.clear();
        else
            desc.nodes[1].params.insert(desc.nodes[1].params.end(),
                {Param("lengthRandom", V(GfVec2f(0.6f, 1.3f))), Param("lift", V(0.2f)),
                 Param("direction", V(TfToken("vector"))),
                 Param("directionVector", V(GfVec3f(0.3f, 1.0f, 0.1f)))});
        return desc;
    }
    if (type == "UsdGenCurveSource") {
        UsdGenGraphDesc desc = Base();
        desc.curveSets.push_back(Hair(true));
        UsdGenNodeDesc source = CurveSource("op", P("hair"));
        Tune(&source, tuned, {Param("resampleTo", V(5))});
        desc.nodes = {source};
        desc.terminal = source.path;
        return desc;
    }
    if (type == "UsdGenReferenceSource") {
        UsdGenGraphDesc desc = Base();
        desc.curveSets.push_back(Guides("guides", 1.0f));
        op.references = {P("guides")};
        desc.nodes = {op};
        desc.terminal = op.path;
        return desc;
    }
    if (type == "UsdGenCardSource") {
        UsdGenGraphDesc desc = Base();
        desc.surfaces.push_back(Skin("skin", kSkinFaces, 1.0f, false));
        desc.curveSets = {CardSpines(), CardLibrary()};
        op.surfaces = {P("skin")};
        op.curves = {desc.curveSets[0].path, desc.curveSets[1].path};
        op.params = {Param("cvCount", V(6))};
        Tune(&op, tuned, {Param("cardSource:rootFlatten", V(0.5f))});
        desc.nodes = {op};
        desc.terminal = op.path;
        return desc;
    }
    if (type == "UsdGenDeform") {
        op.surfaces = {P("skin")};
        op.params = {Param("rbfSamples", V(40))};
        Tune(&op, tuned, {Param("lockRoots", V(false))});
        return OnGrown(&op, true);
    }
    if (type == "UsdGenGuideInterpolate") {
        UsdGenGraphDesc desc = Base();
        desc.curveSets = {Roots(), Guides("guides", 1.0f)};
        desc.nodes.push_back(CurveSource("source", P("roots")));
        op.inputs = {P("source")};
        op.references = {P("guides")};
        op.params = {Param("cvCount", V(6))};
        desc.nodes.push_back(op);
        desc.terminal = op.path;
        return desc;
    }
    if (type == "UsdGenWidthBlend") {
        UsdGenGraphDesc desc = Base();
        desc.curveSets.push_back(Hair(false));
        desc.nodes.push_back(CurveSource("source", P("hair")));
        UsdGenNodeDesc left = Node("left", "UsdGenWidth", 2), right = Node("right", "UsdGenWidth", 3);
        left.inputs = right.inputs = {P("source")};
        left.params = {Param("width", V(0.02f))};
        right.params = {Param("width", V(0.06f))};
        op.inputs = {P("left"), P("right")};
        Tune(&op, tuned, {Param("widthBlend:weight", V(0.3f))});
        desc.nodes.push_back(left);
        desc.nodes.push_back(right);
        desc.nodes.push_back(op);
        desc.terminal = op.path;
        return desc;
    }
    if (type == "UsdGenCollide") {
        UsdGenGraphDesc desc = Base();
        desc.surfaces.push_back(Skin("skin", kSkinFaces, 1.0f, false));
        desc.surfaces.push_back(Box("box", GfVec3f(0.3f, 0.15f, 0.3f), GfVec3f(0.7f, 0.45f, 0.7f)));
        desc.curveSets.push_back(Hair(true));
        desc.nodes.push_back(CurveSource("source", P("hair")));
        op.inputs = {P("source")};
        op.surfaces = {P("skin"), P("box")};
        Tune(&op, tuned, {Param("offset", V(0.02f)), Param("pushAmount", V(1.0f)),
                          Param("iterations", V(3))});
        desc.nodes.push_back(op);
        desc.terminal = op.path;
        return desc;
    }
    // CurveSource-fed stylers.
    UsdGenGraphDesc desc;
    if (type == "UsdGenPart") {
        UsdGenCurveSetDesc parting;
        parting.path = P("parting");
        parting.role = UsdGenLane::Reference;
        parting.curveRole = TfToken("guide");
        parting.curveVertexCounts = {3};
        parting.points = {GfVec3f(0.5f, 0, -0.1f), GfVec3f(0.45f, 0, 0.5f), GfVec3f(0.55f, 0, 1.1f)};
        parting.rest = parting.points;
        op.references = {parting.path};
        Tune(&op, tuned, {Param("part:radius", V(0.15f)), Param("part:strength", V(0.8f))});
        desc = OnHair(op);
        desc.curveSets.push_back(parting);
        return desc;
    }
    if (type == "UsdGenFreeze") {
        UsdGenCurveSetDesc snap = Hair(false);
        snap.path = P("snap");
        snap.role = UsdGenLane::Reference;
        for (GfVec3f &p : snap.points) p += GfVec3f(0.0f, 0.0f, 0.1f * p[1]);
        snap.rest = snap.points;
        op.references = {snap.path};
        Tune(&op, tuned, {Param("frozen:tier", V(TfToken("sublayer")))});
        desc = OnHair(op);
        desc.curveSets.push_back(snap);
        return desc;
    }
    if (type == "UsdGenTieClump") {
        op.references = {P("guides")};
        op.params = {Param("tie:positions", V(VtVec3fArray{GfVec3f(0.25f, 0.4f, 0.25f),
                                                           GfVec3f(0.75f, 0.3f, 0.6f)})),
                     Param("tie:entryCounts", V(VtIntArray{2, 1})),
                     Param("tie:guideIndices", V(VtIntArray{0, 1, 5})),
                     Param("tie:vertexIndices", V(VtIntArray{3, 3, 2}))};
        desc = OnHair(op);
        desc.curveSets.push_back(Guides("guides", 1.0f));
        return desc;
    }
    if (type == "UsdGenBraid" || type == "UsdGenTrim") {
        if (type == "UsdGenBraid") op.curves = {P("guides")};
        else op.references = {P("guides")};
        if (type == "UsdGenTrim") Tune(&op, tuned, {Param("trim:angle", V(30.0f))});
        desc = OnHair(op);
        desc.curveSets.push_back(Guides("guides", 1.0f));
        return desc;
    }
    if (type == "UsdGenExprOp") {
        op.params = {Param("expr:source", V(std::string("[$t * 0.2, 0, -$t * 0.1]")))};
        return OnHair(op);
    }
    if (type == "UsdGenSculptLayer") {
        op.params = {Param("sculpt:curveIds", V(VtArray<uint64_t>{1000, 1017, 1153})),
                     Param("sculpt:cvOffsets", V(VtIntArray{0, 8, 16, 24})),
                     Param("sculpt:deltas", V([] {
                         VtVec3fArray d;
                         for (int i = 0; i < 24; ++i)
                             d.push_back(GfVec3f(0.01f * float(i % 8), 0.0f, -0.005f * float(i % 8)));
                         return d;
                     }()))};
        Tune(&op, tuned, {Param("sculpt:weight", V(0.7f))});
        return OnHair(op);
    }
    VtValue const rise = Ramp({GfVec2f(0.0f, 0.2f), GfVec2f(1.0f, 1.0f)});
    VtValue const dip = Ramp({GfVec2f(0.0f, 1.0f), GfVec2f(0.5f, 0.3f), GfVec2f(1.0f, 0.8f)});
    std::map<std::string, Tuning> const tuning = {
        {"UsdGenNoise", {Param("noise:magnitude", V(0.08f)), Param("noise:frequency", V(4.0f)),
                         Param("noise:octaves", V(3)), Param("noise:magnitude:knots", rise)}},
        {"UsdGenLength", {Param("length:mode", V(TfToken("randomRange"))),
                          Param("length:min", V(0.4f)), Param("length:max", V(0.9f))}},
        {"UsdGenWidth", {Param("width", V(0.03f)), Param("width:knots", dip)}},
        {"UsdGenClump", {Param("clump:levels", V(2)), Param("clump:profile:knots", rise)}},
        {"UsdGenCurl", {Param("radius", V(0.04f)), Param("frequency", V(3.0f)),
                        Param("radius:knots", rise)}},
        {"UsdGenBend", {Param("angle", V(25.0f)), Param("angle:knots", dip)}},
        {"UsdGenWave", {Param("amplitudeU", V(0.05f)), Param("amplitudeN", V(0.03f)),
                        Param("frequencyU", V(3.0f))}},
        {"UsdGenDirection", {Param("amount", V(0.6f)), Param("tiltU", V(0.3f)),
                             Param("aroundN", V(0.2f)), Param("direction:knots", rise)}},
        {"UsdGenSmooth", {Param("iterations", V(4)), Param("strength", V(0.7f))}},
        {"UsdGenResample", {Param("cvCount", V(12))}},
        {"UsdGenScale", {Param("scale", V(1.4f)), Param("scale:knots", dip)}},
        {"UsdGenStraighten", {Param("tangentStraightness", V(0.5f))}},
        {"UsdGenDisplace", {Param("displace:amount", V(0.05f))}},
        {"UsdGenWind", {Param("constStrength", V(0.4f)), Param("gustStrength", V(0.2f)),
                        Param("stiffness", V(0.3f)), Param("stiffness:knots", rise)}},
        {"UsdGenTwist", {Param("twist:tip", V(0.8f)), Param("twist:mid", V(0.4f)),
                         Param("twist:noise", V(0.2f))}},
        {"UsdGenFrizz", {Param("frizz:amplitude", V(0.1f)), Param("frizz:wavesX", V(3.0f)),
                         Param("frizz:wavesZ", V(2.0f)), Param("frizz:noise", V(0.3f))}},
    };
    auto const it = tuning.find(type);
    if (it == tuning.end()) return UsdGenGraphDesc();
    Tune(&op, tuned, it->second);
    return OnHair(op);
}

char const *const kTypes[] = {
    "UsdGenBend", "UsdGenBraid", "UsdGenCardSource", "UsdGenClump", "UsdGenCollide",
    "UsdGenCurl", "UsdGenCurveSource", "UsdGenDeform", "UsdGenDirection", "UsdGenDisplace",
    "UsdGenExprOp", "UsdGenFreeze", "UsdGenFrizz", "UsdGenGrow", "UsdGenGuideInterpolate",
    "UsdGenLength", "UsdGenNoise", "UsdGenPart", "UsdGenReferenceSource", "UsdGenResample",
    "UsdGenScale", "UsdGenScatter", "UsdGenSculptLayer", "UsdGenSmooth", "UsdGenStraighten",
    "UsdGenTieClump", "UsdGenTrim", "UsdGenTwist", "UsdGenWave", "UsdGenWidth",
    "UsdGenWidthBlend", "UsdGenWind"};

}  // namespace

SdfPath const &OpPath()
{
    static SdfPath const path("/g/op");
    return path;
}

UsdGenGraphDesc MakeOpGraph(TfToken const &type, bool tuned)
{
    return MakeOp(type.GetString(), tuned);
}

std::vector<TfToken> FixtureTypes()
{
    std::vector<TfToken> types;
    for (char const *t : kTypes) types.push_back(TfToken(t));
    return types;
}

std::vector<GoldenGraph> MakeGoldenGraphs()
{
    std::vector<GoldenGraph> graphs;
    for (char const *t : kTypes) {
        graphs.push_back({std::string("op/") + t + "/defaults", MakeOp(t, false)});
        graphs.push_back({std::string("op/") + t + "/tuned", MakeOp(t, true)});
    }
    // Scatter on the Catmull-Clark limit surface (limit-surface root UVs).
    {
        UsdGenGraphDesc desc = MakeOp("UsdGenScatter", true);
        desc.surfaces[0].subdivisionScheme = TfToken("catmullClark");
        desc.nodes[0].params.push_back(Param("subdivisionLevel", VtValue(1)));
        graphs.push_back({"op/UsdGenScatter/limit", desc});
    }
    // CurveSource re-binding its roots to a rest surface.
    {
        UsdGenGraphDesc desc = MakeOp("UsdGenCurveSource", false);
        desc.surfaces.push_back(Skin("skin", kSkinFaces, 1.0f, false));
        desc.nodes[0].surfaces = {P("skin")};
        desc.nodes[0].params = {Param("rebind", VtValue(TfToken("always"))),
                                Param("useRest", VtValue(true))};
        graphs.push_back({"op/UsdGenCurveSource/rebind", desc});
    }
    // The M1 chain on a larger skin: several chunks and tiles.
    {
        UsdGenNodeDesc noise = Node("noise", "UsdGenNoise", 44);
        noise.params = {Param("noise:magnitude", VtValue(0.05f)),
                        Param("noise:frequency", VtValue(3.0f)),
                        Param("noise:octaves", VtValue(2))};
        UsdGenGraphDesc desc = OnGrown(&noise, false, 24, 4.0f, 400.0);
        UsdGenNodeDesc length = Node("length", "UsdGenLength", 45);
        length.inputs = {P("noise")};
        length.params = {Param("length:mode", VtValue(TfToken("scale"))),
                         Param("length:value", VtValue(1.2f))};
        UsdGenNodeDesc width = Node("width", "UsdGenWidth", 46);
        width.inputs = {P("length")};
        width.params = {Param("width", VtValue(0.02f))};
        desc.nodes.push_back(length);
        desc.nodes.push_back(width);
        desc.terminal = P("width");
        desc.tileTarget = 16;
        graphs.push_back({"chain/m1", desc});
    }
    // Two clumps (descriptor-order level numbering), clump-aware wind, cull.
    {
        UsdGenGraphDesc desc = OnHair(Node("clumpA", "UsdGenClump", 11), false);
        UsdGenNodeDesc clumpB = Node("clumpB", "UsdGenClump", 12);
        clumpB.inputs = {P("clumpA")};
        clumpB.params = {Param("clump:levels", VtValue(2))};
        UsdGenNodeDesc wind = Node("wind", "UsdGenWind", 13);
        wind.inputs = {P("clumpB")};
        wind.params = {Param("constStrength", VtValue(0.3f))};
        UsdGenNodeDesc length = Node("length", "UsdGenLength", 14);
        length.inputs = {P("wind")};
        length.params = {Param("length:mode", VtValue(TfToken("cull"))),
                         Param("cullThreshold", VtValue(0.3f))};
        UsdGenNodeDesc width = Node("width", "UsdGenWidth", 15);
        width.inputs = {P("length")};
        desc.nodes.insert(desc.nodes.end(), {clumpB, wind, length, width});
        desc.terminal = P("width");
        graphs.push_back({"chain/clumps", desc});
    }
    // Connected parameters at each domain (expression lane).
    {
        UsdGenNodeDesc noise = Node("noise", "UsdGenNoise", 31);
        UsdGenGraphDesc desc = OnHair(noise, false);
        UsdGenNodeDesc length = Node("length", "UsdGenLength", 32);
        length.inputs = {P("noise")};
        UsdGenNodeDesc width = Node("width", "UsdGenWidth", 33);
        width.inputs = {P("length")};
        desc.nodes.push_back(length);
        desc.nodes.push_back(width);
        desc.terminal = P("width");
        auto bind = [&](char const *node, char const *leaf, char const *source,
                        char const *dest, expr::Domain domain, float literal) {
            SdfPath const exprPath = P(leaf);
            expr::ValueShape const shape{expr::ScalarType::Float32, 1, 1, 1, 1, false};
            desc.expressions.push_back(
                {exprPath, source, {{TfToken("result"), TfToken("float"), shape}}});
            for (UsdGenNodeDesc &n : desc.nodes) {
                if (n.path != P(node)) continue;
                UsdGenExpressionBinding b;
                b.expression = exprPath;
                b.output = TfToken("result");
                b.nativeType = TfToken("float");
                b.destination = TfToken(dest);
                b.destinationShape = shape;
                b.domain = domain;
                b.literal = VtValue(literal);
                n.expressionBindings.push_back(b);
            }
        };
        bind("noise", "exprMagnitude", "0.02 + 0.06 * $u", "noise:magnitude",
             expr::Domain::Primitive, 0.05f);
        bind("length", "exprLength", "$value * 0.9", "length:value", expr::Domain::Groom, 1.0f);
        bind("width", "exprWidth", "$value * (1.0 - 0.8 * $t)", "width", expr::Domain::Point,
             0.03f);
        bind("width", "exprMask", "0.5 + 0.5 * $v", "mask", expr::Domain::Primitive, 1.0f);
        graphs.push_back({"chain/expressions", desc});
    }
    // Guides through interpolation, then the whole-strand stylers.
    {
        UsdGenGraphDesc desc = MakeOp("UsdGenGuideInterpolate", true);
        char const *chain[] = {"UsdGenCurl", "UsdGenSmooth", "UsdGenScale", "UsdGenResample"};
        SdfPath input = OpPath();
        int seed = 20;
        for (char const *t : chain) {
            UsdGenNodeDesc node = Node(t + 6, t, seed++);
            node.inputs = {input};
            input = node.path;
            desc.nodes.push_back(node);
        }
        desc.terminal = input;
        graphs.push_back({"chain/guided", desc});
    }
    return graphs;
}

}  // namespace usdGenGolden
