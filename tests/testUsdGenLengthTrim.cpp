// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenLengthTrim: the imported-groom length/trim/smooth/resample modes
// on the CPU lane, cooked through UsdGenCompiler + UsdGenScheduler over
// hand-built strand sets (CurveSource of straight strands -> operator):
//   Length randomRange: default-max rule, random scale range and clamps,
//     shareByClump, followClumpMean, root-sampled length maps (product),
//     the restLength plane, the mask envelope, determinism;
//   Trim: cut at the first blade crossing, blade angle/invert, per-curve
//     overrides, stored vs surface normals, strands that miss are untouched;
//   Smooth box: hand-computed Jacobi box pass, fixed endpoints;
//   Resample spacing: per-strand counts, ragged layout, arc-length
//     uniformity, hairT, carried planes, a ragged downstream sweep.
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// Reference random stream, restated here so the operator's hashing is
// checked against an independent copy: splitmix64 + 53-bit U01.
static uint64_t SplitMix(uint64_t z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float RefU01(int32_t seed)
{
    uint64_t const z = SplitMix(uint64_t(int64_t(seed)));
    return float(double(z >> 11) * (1.0 / 9007199254740992.0));
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

using Curve = std::vector<GfVec3f>;

static Curve Straight(GfVec3f const &root, int cvs, float step)
{
    Curve c;
    for (int i = 0; i < cvs; ++i) c.push_back(root + GfVec3f(0.0f, float(i) * step, 0.0f));
    return c;
}

static UsdGenCurveSetDesc HairSet(SdfPath const &path, std::vector<Curve> const &curves,
                                  UsdGenRole role = UsdGenRole::Curves)
{
    UsdGenCurveSetDesc set;
    set.path = path;
    set.role = role;
    set.curveRole = TfToken(role == UsdGenRole::Curves ? "hair" : "guide");
    for (Curve const &c : curves) {
        set.curveVertexCounts.push_back(int(c.size()));
        for (GfVec3f const &p : c) set.points.push_back(p);
    }
    set.rest = set.points;
    size_t const n = curves.size();
    if (role == UsdGenRole::Curves) {
        set.widths.assign(set.points.size(), 0.01f);
        set.curveId.resize(n);
        for (size_t c = 0; c < n; ++c) set.curveId[c] = uint64_t(c);
        set.skinPrim.resize(n);
        for (size_t c = 0; c < n; ++c) set.skinPrim[c] = int(c);
        set.skinPrimUv.assign(n, GfVec2f(0.2f, 0.3f));
        set.rootFrame.assign(n, GfMatrix4d(1.0));
    }
    return set;
}

static void AddIntPlane(UsdGenCurveSetDesc &set, char const *name, std::vector<int> const &v)
{
    UsdGenAuthoredPlaneDesc plane;
    plane.name = TfToken(name);
    plane.type = UsdGenAuthoredPlaneType::Int32;
    plane.domain = UsdGenAuthoredPlaneDomain::Primitive;
    plane.arity = 1;
    plane.intValues = VtIntArray(v.begin(), v.end());
    set.authoredPlanes.push_back(plane);
}

static void AddFloatPlane(UsdGenCurveSetDesc &set, char const *name, std::vector<float> const &v,
                          UsdGenAuthoredPlaneDomain domain, uint8_t arity = 1)
{
    UsdGenAuthoredPlaneDesc plane;
    plane.name = TfToken(name);
    plane.type = UsdGenAuthoredPlaneType::Float32;
    plane.domain = domain;
    plane.arity = arity;
    plane.floatValues = VtFloatArray(v.begin(), v.end());
    set.authoredPlanes.push_back(plane);
}

static UsdGenNodeDesc SourceNode()
{
    UsdGenNodeDesc node;
    node.path = SdfPath("/groom/source");
    node.type = TfToken("UsdGenCurveSource");
    node.curves = {SdfPath("/groom/hair")};
    node.params = {{TfToken("rebind"), VtValue(TfToken("never")), false},
                   {TfToken("useRest"), VtValue(true), false}};
    return node;
}

static UsdGenNodeDesc OpNode(char const *path, char const *type, SdfPath const &input)
{
    UsdGenNodeDesc node;
    node.path = SdfPath(path);
    node.type = TfToken(type);
    node.seed = 3;
    node.inputs = {input};
    return node;
}

static void Param(UsdGenNodeDesc &node, char const *name, VtValue const &value)
{
    node.params.push_back({TfToken(name), value, false});
}

static UsdGenGraphDesc BaseDesc(UsdGenCurveSetDesc const &hair)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/groom");
    desc.defaultWidth = 0.01f;
    desc.curveSets.push_back(hair);
    desc.nodes.push_back(SourceNode());
    return desc;
}

static bool Cook(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *out,
                 std::vector<std::string> *warnings = nullptr)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult compiled = compiler.Compile(desc, &graph);
    if (!compiled.ok) {
        for (auto const &e : compiled.errors) std::printf("  compile: %s\n", e.c_str());
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    UsdGenRunResult run = scheduler.Run(graph, ctx, 1);
    if (run.diagnostics.HasErrors()) {
        for (auto const &e : run.diagnostics.errors) std::printf("  run: %s\n", e.c_str());
        return false;
    }
    if (warnings) *warnings = run.diagnostics.warnings;
    *out = graph.Output();
    return true;
}

struct Spans
{
    std::vector<uint32_t> first;
    explicit Spans(UsdGenCurveBuffer const &b)
    {
        first.assign(size_t(b.totalCurves) + 1, 0);
        for (uint32_t c = 0; c < b.totalCurves; ++c) {
            uint32_t const n = b.cvOffsets.empty() ? b.totalCvs / std::max(1u, b.totalCurves)
                                                   : uint32_t(b.cvOffsets[c + 1] - b.cvOffsets[c]);
            first[c + 1] = first[c] + n;
        }
    }
    uint32_t N(uint32_t c) const { return first[c + 1] - first[c]; }
};

static GfVec3f Pt(UsdGenCurveBuffer const &b, Spans const &s, uint32_t c, uint32_t i)
{
    size_t const k = s.first[c] + i;
    return GfVec3f(b.px[k], b.py[k], b.pz[k]);
}

static float StrandLength(UsdGenCurveBuffer const &b, Spans const &s, uint32_t c)
{
    float L = 0.0f;
    for (uint32_t i = 1; i < s.N(c); ++i) L += (Pt(b, s, c, i) - Pt(b, s, c, i - 1)).GetLength();
    return L;
}

static UsdGenPlane const *CurvePlane(UsdGenCurveBuffer const &b, char const *name)
{
    for (UsdGenPlane const &p : b.extraCurve)
        if (p.name == TfToken(name)) return &p;
    return nullptr;
}

static UsdGenPlane const *CvPlane(UsdGenCurveBuffer const &b, char const *name)
{
    for (UsdGenPlane const &p : b.extraCv)
        if (p.name == TfToken(name)) return &p;
    return nullptr;
}

static bool SamePoints(UsdGenCurveBuffer const &a, UsdGenCurveBuffer const &b)
{
    return a.totalCvs == b.totalCvs && a.px.size() == b.px.size() &&
           std::memcmp(a.px.cdata(), b.px.cdata(), a.px.size() * sizeof(float)) == 0 &&
           std::memcmp(a.py.cdata(), b.py.cdata(), a.py.size() * sizeof(float)) == 0 &&
           std::memcmp(a.pz.cdata(), b.pz.cdata(), a.pz.size() * sizeof(float)) == 0;
}

// Three straight strands along +Y (5 CVs, length 1) at x = 0, 1, 2.
static UsdGenCurveSetDesc ThreeStrands()
{
    return HairSet(SdfPath("/groom/hair"), {Straight(GfVec3f(0, 0, 0), 5, 0.25f),
                                            Straight(GfVec3f(1, 0, 0), 5, 0.25f),
                                            Straight(GfVec3f(2, 0, 0), 5, 0.25f)});
}

static UsdGenGraphDesc LengthDesc(UsdGenCurveSetDesc const &hair,
                                  std::vector<std::pair<char const *, VtValue>> const &params)
{
    UsdGenGraphDesc desc = BaseDesc(hair);
    UsdGenNodeDesc length = OpNode("/groom/length", "UsdGenLength", SdfPath("/groom/source"));
    Param(length, "length:mode", VtValue(TfToken("randomRange")));
    for (auto const &p : params) Param(length, p.first, p.second);
    desc.nodes.push_back(length);
    desc.terminal = SdfPath("/groom/length");
    return desc;
}

// ---------------------------------------------------------------------------
// Length randomRange
// ---------------------------------------------------------------------------

static void TestLengthDefaultMax()
{
    UsdGenCurveBuffer out;
    bool ok = Cook(LengthDesc(ThreeStrands(), {{"length:min", VtValue(0.5f)},
                                                     {"length:max", VtValue(1.5f)}}), &out);
    Check(ok, "Length randomRange cooks");
    if (!ok) return;
    Spans const s(out);
    bool exact = true;
    for (uint32_t c = 0; c < 3; ++c)
        for (uint32_t j = 0; j < 5; ++j) {
            GfVec3f const p = Pt(out, s, c, j);
            exact = exact && Near(p[0], float(c)) && Near(p[1], 0.375f * float(j)) && Near(p[2], 0.0f);
        }
    Check(exact, "randomize 0 scales every strand by max (1.5), straight past the tip");
    UsdGenPlane const *rest = CurvePlane(out, "restLength");
    bool restOk = rest && rest->type == TfToken("float") && rest->f.size() == 3;
    for (size_t c = 0; restOk && c < 3; ++c) restOk = Near(rest->f[c], 1.0f);
    Check(restOk, "restLength plane carries the pre-scale length (1.0)");

    // Clamps: min/max are clamped to [0.01, 2] (randomize 0 -> 2), and a
    // reversed pair is swapped (max wins after the swap: 1.5).
    ok = Cook(LengthDesc(ThreeStrands(), {{"length:min", VtValue(-3.0f)},
                                          {"length:max", VtValue(5.0f)}}), &out);
    Check(ok && Near(Pt(out, Spans(out), 1, 4)[1], 2.0f), "max is clamped to 2");
    ok = Cook(LengthDesc(ThreeStrands(), {{"length:min", VtValue(1.5f)},
                                          {"length:max", VtValue(0.5f)}}), &out);
    Check(ok && Near(Pt(out, Spans(out), 1, 4)[1], 1.5f), "a reversed min/max pair is swapped");
    // A factor mode declares no restLength plane (the plane belongs to a
    // node configured for randomRange).
    {
        UsdGenGraphDesc scale = BaseDesc(ThreeStrands());
        UsdGenNodeDesc length = OpNode("/groom/length", "UsdGenLength", SdfPath("/groom/source"));
        Param(length, "length:mode", VtValue(TfToken("scale")));
        Param(length, "length:value", VtValue(2.0f));
        scale.nodes.push_back(length);
        scale.terminal = length.path;
        ok = Cook(scale, &out);
        Check(ok && !CurvePlane(out, "restLength") && Near(Pt(out, Spans(out), 1, 4)[1], 2.0f),
              "scale mode is unchanged and emits no restLength");
    }
    ok = Cook(LengthDesc(ThreeStrands(), {{"length:min", VtValue(0.01f)},
                                          {"length:max", VtValue(0.5f)}}), &out);
    Check(ok && Near(Pt(out, Spans(out), 0, 4)[1], 0.5f) && Near(Pt(out, Spans(out), 0, 2)[1], 0.25f),
          "shortening resamples uniformly over the kept part");
}

static void TestLengthRandomRange()
{
    UsdGenCurveSetDesc hair = ThreeStrands();
    std::vector<int> const seeds{11, 22, -33};
    AddIntPlane(hair, "strandSeed", seeds);
    UsdGenCurveBuffer out, again;
    auto params = [](float randomize) {
        return std::vector<std::pair<char const *, VtValue>>{
            {"length:min", VtValue(0.5f)}, {"length:max", VtValue(1.5f)},
            {"length:randomize", VtValue(randomize)}};
    };
    bool ok = Cook(LengthDesc(hair, params(1.0f)), &out);
    Check(ok, "randomize 1 cooks");
    if (!ok) return;
    Spans s(out);
    bool match = true, inRange = true;
    for (uint32_t c = 0; c < 3; ++c) {
        float const scale = 0.5f + RefU01(seeds[c]) * (1.5f - 0.5f);
        float const tip = Pt(out, s, c, 4)[1];
        match = match && Near(tip, scale, 1e-5f) && Near(Pt(out, s, c, 2)[1], 0.5f * scale);
        inRange = inRange && tip >= 0.5f && tip <= 1.5f;
    }
    Check(match, "randomize 1: scale = lo + U01(splitmix64(int64(int32 strandSeed))) (hi - lo)");
    Check(inRange, "random lengths stay inside [min, max]");
    Check(!Near(Pt(out, s, 0, 4)[1], Pt(out, s, 1, 4)[1], 1e-4f),
          "different strand seeds draw different lengths");

    ok = Cook(LengthDesc(hair, params(0.25f)), &out);
    s = Spans(out);
    match = ok;
    for (uint32_t c = 0; ok && c < 3; ++c) {
        float const scale = (0.5f + RefU01(seeds[c]) * 1.0f) * 0.25f + 0.75f * 1.5f;
        match = match && Near(Pt(out, s, c, 4)[1], scale, 1e-5f);
    }
    Check(match, "randomize r blends: (lo + u (hi - lo)) r + (1 - r) hi");

    ok = Cook(LengthDesc(hair, params(1.0f)), &again);
    Check(ok && Cook(LengthDesc(hair, params(1.0f)), &out) && SamePoints(out, again),
          "randomRange is deterministic across cooks");

    // The usdGen:mask envelope: 0 passes the input through bitwise.
    auto masked = params(1.0f);
    masked.push_back({"mask", VtValue(0.0f)});
    ok = Cook(LengthDesc(hair, masked), &out);
    s = Spans(out);
    Check(ok && Near(Pt(out, s, 0, 4)[1], 1.0f, 0.0f) && Near(Pt(out, s, 2, 4)[1], 1.0f, 0.0f),
          "usdGen:mask = 0 leaves the strands untouched");
}

static void TestLengthShareByClump()
{
    UsdGenCurveSetDesc hair = HairSet(SdfPath("/groom/hair"),
        {Straight(GfVec3f(0, 0, 0), 5, 0.25f), Straight(GfVec3f(1, 0, 0), 5, 0.5f),
         Straight(GfVec3f(2, 0, 0), 5, 0.25f), Straight(GfVec3f(3, 0, 0), 5, 0.125f)});
    AddIntPlane(hair, "strandSeed", {1, 2, 3, 4});
    AddIntPlane(hair, "clumpId_0", {7, 7, -1, 7});
    std::vector<std::pair<char const *, VtValue>> params{
        {"length:min", VtValue(0.5f)}, {"length:max", VtValue(1.5f)},
        {"length:randomize", VtValue(1.0f)}, {"length:shareByClump", VtValue(true)}};
    UsdGenCurveBuffer out;
    bool ok = Cook(LengthDesc(hair, params), &out);
    Check(ok, "shareByClump cooks");
    if (!ok) return;
    Spans s(out);
    float const shared = 0.5f + RefU01(7);
    float const L[4] = {1.0f, 2.0f, 1.0f, 0.5f};
    Check(Near(StrandLength(out, s, 0) / L[0], shared, 1e-5f) &&
          Near(StrandLength(out, s, 1) / L[1], shared, 1e-5f) &&
          Near(StrandLength(out, s, 3) / L[3], shared, 1e-5f),
          "clump members share one scale drawn from the clump id");
    Check(Near(StrandLength(out, s, 2), 0.5f + RefU01(3), 1e-5f),
          "an unclumped strand keeps its own strand seed");
    params[3].second = VtValue(false);
    ok = Cook(LengthDesc(hair, params), &out);
    s = Spans(out);
    Check(ok && Near(StrandLength(out, s, 0), 0.5f + RefU01(1), 1e-5f) &&
          !Near(StrandLength(out, s, 0) / L[0], StrandLength(out, s, 1) / L[1], 1e-4f),
          "without shareByClump each member draws from its own seed");
}

static void TestLengthFollowClumpMean()
{
    // A: x=0, length 1; B: x=1, length 2; one clump. Mean: x=0.5, y=0.375k,
    // length 1.5. scale 2 -> target 3: mean points (0.5, 0.75j) (straight
    // past its tip), plus each member's offset from the mean.
    UsdGenCurveSetDesc hair = HairSet(SdfPath("/groom/hair"),
        {Straight(GfVec3f(0, 0, 0), 5, 0.25f), Straight(GfVec3f(1, 0, 0), 5, 0.5f)});
    AddIntPlane(hair, "clumpId_0", {0, 0});
    std::vector<std::pair<char const *, VtValue>> params{
        {"length:min", VtValue(2.0f)}, {"length:max", VtValue(2.0f)},
        {"length:followClumpMean", VtValue(true)}};
    UsdGenCurveBuffer out;
    bool ok = Cook(LengthDesc(hair, params), &out);
    Check(ok, "followClumpMean cooks");
    if (!ok) return;
    Spans s(out);
    bool exact = true;
    for (uint32_t j = 0; j < 5; ++j) {
        exact = exact && Near(Pt(out, s, 0, j)[0], 0.0f) && Near(Pt(out, s, 0, j)[1], 0.625f * j) &&
                Near(Pt(out, s, 1, j)[0], 1.0f) && Near(Pt(out, s, 1, j)[1], 0.875f * j);
    }
    Check(exact, "members follow the clump mean stretched to the clump length, keeping offsets");
    UsdGenPlane const *rest = CurvePlane(out, "restLength");
    Check(rest && rest->f.size() == 2 && Near(rest->f[0], 1.5f) && Near(rest->f[1], 1.5f),
          "restLength is the clump-mean reference length");
    params[2].second = VtValue(false);
    ok = Cook(LengthDesc(hair, params), &out);
    s = Spans(out);
    Check(ok && Near(Pt(out, s, 0, 4)[1], 2.0f) && Near(Pt(out, s, 1, 4)[1], 4.0f),
          "without followClumpMean each strand scales its own length");
}

static UsdGenSurfaceDesc TwoQuadScalp()
{
    // Two disjoint quads in the y = 0 plane: face 0 spans x in [-0.5, 0.5],
    // face 1 spans x in [0.5, 1.5] (z in [-0.5, 0.5]).
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/scalp");
    surface.faceVertexCounts = VtIntArray{4, 4};
    surface.faceVertexIndices = VtIntArray{0, 1, 2, 3, 4, 5, 6, 7};
    surface.points = VtVec3fArray{
        GfVec3f(-0.5f, 0, -0.5f), GfVec3f(-0.5f, 0, 0.5f), GfVec3f(0.5f, 0, 0.5f), GfVec3f(0.5f, 0, -0.5f),
        GfVec3f(0.5f, 0, -0.5f), GfVec3f(0.5f, 0, 0.5f), GfVec3f(1.5f, 0, 0.5f), GfVec3f(1.5f, 0, -0.5f)};
    surface.restPoints = surface.points;
    surface.surfaceGeneration = 1;
    return surface;
}

static UsdGenMapDesc PaintMap(char const *path, std::vector<float> const &values)
{
    UsdGenMapDesc map;
    map.path = SdfPath(path);
    map.type = TfToken("UsdGenPaintMap");
    map.paintSurface = SdfPath("/scalp");
    map.paintPrimvar = TfToken("usdGen:paint:length");
    map.paintInterpolation = TfToken("faceVarying");
    map.paintValues = VtFloatArray(values.begin(), values.end());
    return map;
}

static void TestLengthMask()
{
    UsdGenCurveSetDesc hair = HairSet(SdfPath("/groom/hair"),
        {Straight(GfVec3f(0, 0, 0), 5, 0.25f), Straight(GfVec3f(1, 0, 0), 5, 0.25f)});
    UsdGenGraphDesc desc = LengthDesc(hair, {{"length:min", VtValue(0.5f)},
                                             {"length:max", VtValue(1.5f)}});
    desc.surfaces.push_back(TwoQuadScalp());
    desc.maps.push_back(PaintMap("/groom/maskA", {0.5f, 0.5f, 0.5f, 0.5f, 1, 1, 1, 1}));
    desc.maps.push_back(PaintMap("/groom/maskB", std::vector<float>(8, 0.8f)));
    UsdGenNodeDesc &length = desc.nodes.back();
    length.surfaces = {SdfPath("/scalp")};
    for (char const *path : {"/groom/maskA", "/groom/maskB"}) {
        length.maps.push_back(SdfPath(path));
        length.mapBindings.push_back({SdfPath(path), TfToken("usdGen:lengthMap")});
    }
    UsdGenCurveBuffer out;
    bool const ok = Cook(desc, &out);
    Check(ok, "Length with usdGen:lengthMap cooks");
    if (!ok) return;
    Spans const s(out);
    // scale = max * maskA * maskB at the root: 1.5*0.5*0.8 and 1.5*1*0.8.
    Check(Near(Pt(out, s, 0, 4)[1], 0.6f, 1e-5f) && Near(Pt(out, s, 0, 2)[1], 0.3f, 1e-5f),
          "the length map product scales the strand (root on face 0: 0.4)");
    Check(Near(Pt(out, s, 1, 4)[1], 1.2f, 1e-5f), "root on face 1: 0.8 -> scale 1.2");
    UsdGenPlane const *rest = CurvePlane(out, "restLength");
    Check(rest && rest->f.size() == 2 && Near(rest->f[0], 1.0f) && Near(rest->f[1], 1.0f),
          "masked strands still report their pre-scale length");
}

// ---------------------------------------------------------------------------
// Trim
// ---------------------------------------------------------------------------

struct TrimSetup
{
    std::vector<Curve> curves;
    std::vector<float> normals;            // per point xyz, empty: no plane
    std::vector<float> angle, distance;    // per-curve overrides, empty: none
    std::vector<int> invert;
};

static UsdGenGraphDesc TrimDesc(TrimSetup const &t,
                                std::vector<std::pair<char const *, VtValue>> const &params,
                                bool withSurface = false)
{
    UsdGenGraphDesc desc = BaseDesc(ThreeStrands());
    UsdGenCurveSetDesc blades = HairSet(SdfPath("/groom/trimCurves"), t.curves, UsdGenRole::Reference);
    if (!t.normals.empty())
        AddFloatPlane(blades, "normal", t.normals, UsdGenAuthoredPlaneDomain::Point, 3);
    if (!t.angle.empty())
        AddFloatPlane(blades, "trimAngle", t.angle, UsdGenAuthoredPlaneDomain::Primitive);
    if (!t.distance.empty())
        AddFloatPlane(blades, "trimDistance", t.distance, UsdGenAuthoredPlaneDomain::Primitive);
    if (!t.invert.empty()) AddIntPlane(blades, "trimInvert", t.invert);
    desc.curveSets.push_back(blades);
    UsdGenNodeDesc trim = OpNode("/groom/trim", "UsdGenTrim", SdfPath("/groom/source"));
    trim.references = {blades.path};
    for (auto const &p : params) Param(trim, p.first, p.second);
    if (withSurface) {
        // A wall in the z = -0.1 plane facing +Z.
        UsdGenSurfaceDesc wall;
        wall.path = SdfPath("/wall");
        wall.faceVertexCounts = VtIntArray{4};
        wall.faceVertexIndices = VtIntArray{0, 1, 2, 3};
        wall.points = VtVec3fArray{GfVec3f(-1, -1, -0.1f), GfVec3f(3, -1, -0.1f),
                                   GfVec3f(3, 2, -0.1f), GfVec3f(-1, 2, -0.1f)};
        wall.restPoints = wall.points;
        wall.surfaceGeneration = 1;
        desc.surfaces.push_back(wall);
        trim.surfaces = {wall.path};
    }
    desc.nodes.push_back(trim);
    desc.terminal = SdfPath("/groom/trim");
    return desc;
}

static std::vector<float> PlusZ(size_t points)
{
    std::vector<float> n;
    for (size_t i = 0; i < points; ++i) { n.push_back(0); n.push_back(0); n.push_back(1); }
    return n;
}

// The blade line at height y, z = -0.05, from x0 to x1 (3 points).
static Curve BladeLine(float y, float x0, float x1)
{
    return {GfVec3f(x0, y, -0.05f), GfVec3f(0.5f * (x0 + x1), y, -0.05f), GfVec3f(x1, y, -0.05f)};
}

// Strand c cut at arc length s: 5 CVs at y = s j / 4 along x = c.
static bool CutAt(UsdGenCurveBuffer const &out, uint32_t c, float s, float tol = 1e-4f)
{
    Spans const sp(out);
    bool ok = sp.N(c) == 5;
    for (uint32_t j = 0; ok && j < 5; ++j) {
        GfVec3f const p = Pt(out, sp, c, j);
        ok = Near(p[0], float(c), 1e-6f) && Near(p[1], s * float(j) / 4.0f, tol) && Near(p[2], 0.0f, 1e-6f);
    }
    if (!ok) {
        std::printf("  strand %u:", c);
        for (uint32_t j = 0; j < sp.N(c); ++j) {
            GfVec3f const p = Pt(out, sp, c, j);
            std::printf(" (%g %g %g)", p[0], p[1], p[2]);
        }
        std::printf("\n");
    }
    return ok;
}

static void TestTrim()
{
    // Registry contract.
    std::unique_ptr<UsdGenOp> op = UsdGenOpRegistry::Get().Create(TfToken("UsdGenTrim"));
    Check(op && op->ReferenceInputs().size() == 1 &&
          op->ReferenceInputs()[0] == TfToken("trim:curves") &&
          op->TopologyEffect() == UsdGenTopoFx::None, "UsdGenTrim declares trim:curves");

    UsdGenCurveBuffer input, out;
    {
        UsdGenGraphDesc d = BaseDesc(ThreeStrands());
        d.terminal = SdfPath("/groom/source");
        Check(Cook(d, &input), "the untrimmed source cooks");
    }

    // One blade at y = 0.6 over strands 0 and 1, standing along +Z.
    TrimSetup t;
    t.curves = {BladeLine(0.6f, -0.5f, 1.5f)};
    t.normals = PlusZ(3);
    bool ok = Cook(TrimDesc(t, {}), &out);
    Check(ok, "Trim cooks");
    if (!ok) return;
    Check(CutAt(out, 0, 0.6f) && CutAt(out, 1, 0.6f),
          "strands are cut where they cross the blade and resampled to 5 CVs");
    Spans const sp(out), si(input);
    bool untouched = sp.N(2) == 5;
    for (uint32_t j = 0; untouched && j < 5; ++j)
        untouched = Pt(out, sp, 2, j) == Pt(input, si, 2, j);
    Check(untouched, "a strand that misses every blade is untouched (bitwise)");

    // Tilt: D = cos(45) N' + sin(45) B, B = T x N' = -Y -> the blade leans
    // down; the strand (z = 0) crosses it at y = 0.6 - 0.05 = 0.55. Invert
    // leans it up: 0.65.
    ok = Cook(TrimDesc(t, {{"trim:angle", VtValue(45.0f)}}), &out);
    Check(ok && CutAt(out, 0, 0.55f), "trim:angle tilts the blade about the curve tangent");
    ok = Cook(TrimDesc(t, {{"trim:angle", VtValue(45.0f)}, {"trim:invert", VtValue(true)}}), &out);
    Check(ok && CutAt(out, 0, 0.65f), "trim:invert flips the tilt");
    // clamp(angle, 0, 89): a negative angle stands the blade upright (an
    // unclamped -30 would lean it up and cut at 0.6 + 0.05 tan(30) = 0.629).
    ok = Cook(TrimDesc(t, {{"trim:angle", VtValue(-30.0f)}}), &out);
    Check(ok && CutAt(out, 0, 0.6f), "the blade angle is clamped to [0, 89] degrees");

    // First crossing: a second (later) curve at y = 0.3 over strand 1 only.
    TrimSetup two = t;
    two.curves.push_back(BladeLine(0.3f, 0.5f, 1.5f));
    two.normals = PlusZ(6);
    ok = Cook(TrimDesc(two, {}), &out);
    Check(ok && CutAt(out, 0, 0.6f) && CutAt(out, 1, 0.3f),
          "a strand crossing two blades is cut at the first crossing from the root");

    // Per-curve overrides: curve 0 angle 45 inverted (0.65), curve 1 as is.
    TrimSetup over = two;
    over.angle = {45.0f, 0.0f};
    over.invert = {1, 0};
    ok = Cook(TrimDesc(over, {}), &out);
    Check(ok && CutAt(out, 0, 0.65f) && CutAt(out, 1, 0.3f),
          "trimAngle / trimInvert planes override the operator per curve");
    over.distance = {0.02f, 0.1f};
    ok = Cook(TrimDesc(over, {}), &out);
    Spans const so(out);
    Check(ok && Near(Pt(out, so, 0, 4)[1], 1.0f, 0.0f) && CutAt(out, 1, 0.3f),
          "a short trimDistance blade (z in [-0.05, -0.03]) misses the strand");

    // Normals from the bound surface: the wall at z = -0.1 faces +Z and lies
    // within normalSearch (0.08) of the blade curve; with a tiny search the
    // fallback +Y blade stands in the z = -0.05 plane and misses.
    TrimSetup bare;
    bare.curves = t.curves;
    ok = Cook(TrimDesc(bare, {}, true), &out);
    Check(ok && CutAt(out, 0, 0.6f) && CutAt(out, 1, 0.6f),
          "without stored normals the closest surface normal stands the blade");
    ok = Cook(TrimDesc(bare, {{"trim:normalSearch", VtValue(0.01f)}}, true), &out);
    Spans const sb(out);
    Check(ok && Near(Pt(out, sb, 0, 4)[1], 1.0f, 0.0f) && Near(Pt(out, sb, 1, 4)[1], 1.0f, 0.0f),
          "out of normalSearch the +Y fallback blade never reaches the strands");

    UsdGenCurveBuffer again;
    Check(Cook(TrimDesc(two, {}), &out) && Cook(TrimDesc(two, {}), &again) && SamePoints(out, again),
          "Trim is deterministic across cooks");

    // Length -> Trim: the blade cuts the length-scaled strands.
    UsdGenGraphDesc chain = TrimDesc(t, {});
    UsdGenNodeDesc length = OpNode("/groom/length", "UsdGenLength", SdfPath("/groom/source"));
    Param(length, "length:mode", VtValue(TfToken("randomRange")));
    Param(length, "length:max", VtValue(0.5f));
    chain.nodes.insert(chain.nodes.end() - 1, length);
    chain.nodes.back().inputs = {SdfPath("/groom/length")};
    ok = Cook(chain, &out);
    Spans const sc(out);
    Check(ok && Near(Pt(out, sc, 0, 4)[1], 0.5f) && CurvePlane(out, "restLength"),
          "strands shortened below the blade are not trimmed; restLength flows through");
}

// ---------------------------------------------------------------------------
// Smooth box
// ---------------------------------------------------------------------------

static void TestSmoothBox()
{
    // x zig-zags 0,1,0,1,0,1 while y = k.
    Curve zig;
    for (int k = 0; k < 6; ++k) zig.push_back(GfVec3f(float(k % 2), float(k), 0.0f));
    auto desc = [&](int window, float strength) {
        UsdGenGraphDesc d = BaseDesc(HairSet(SdfPath("/groom/hair"), {zig}));
        UsdGenNodeDesc smooth = OpNode("/groom/smooth", "UsdGenSmooth", SdfPath("/groom/source"));
        Param(smooth, "mode", VtValue(TfToken("box")));
        Param(smooth, "window", VtValue(window));
        Param(smooth, "strength", VtValue(strength));
        d.nodes.push_back(smooth);
        d.terminal = smooth.path;
        return d;
    };
    UsdGenCurveBuffer out;
    bool ok = Cook(desc(2, 0.5f), &out);
    Check(ok, "Smooth box cooks");
    if (!ok) return;
    Spans const s(out);
    // P[k] = 0.5 O[k] + 0.5 mean(O[k-2 .. k+2]) (clamped window), ends fixed.
    float const ex[6] = {0.0f, 0.75f, 0.2f, 0.8f, 0.25f, 1.0f};
    float const ey[6] = {0.0f, 1.25f, 2.0f, 3.0f, 3.75f, 5.0f};
    bool exact = true;
    for (uint32_t k = 0; k < 6; ++k)
        exact = exact && Near(Pt(out, s, 0, k)[0], ex[k]) && Near(Pt(out, s, 0, k)[1], ey[k]);
    Check(exact, "box window 2: one Jacobi pass of (1-s) O + s mean, both endpoints fixed");
    ok = Cook(desc(1, 1.0f), &out);
    Spans const s1(out);
    // window 1, strength 1: P[k] = mean(O[k-1..k+1]); x: 1/3, 2/3, 1/3, 2/3.
    Check(ok && Near(Pt(out, s1, 0, 1)[0], 1.0f / 3.0f) && Near(Pt(out, s1, 0, 2)[0], 2.0f / 3.0f) &&
          Near(Pt(out, s1, 0, 4)[0], 2.0f / 3.0f) && Near(Pt(out, s1, 0, 5)[0], 1.0f) &&
          Near(Pt(out, s1, 0, 0)[0], 0.0f), "box window 1 at full strength");
    ok = Cook(desc(2, -0.5f), &out);
    Spans const s2(out);
    Check(ok && Near(Pt(out, s2, 0, 1)[0], 1.0f, 0.0f), "box strength is clamped to [0, 1]");
}

// ---------------------------------------------------------------------------
// Resample spacing
// ---------------------------------------------------------------------------

static void TestResampleSpacing()
{
    std::vector<Curve> curves{
        Straight(GfVec3f(0, 0, 0), 5, 0.25f),                         // L = 1
        Straight(GfVec3f(1, 0, 0), 4, 0.2f),                          // L = 0.6
        Straight(GfVec3f(2, 0, 0), 2, 0.05f),                         // L = 0.05
        {GfVec3f(3, 0, 0), GfVec3f(3, 1, 0), GfVec3f(4, 1, 0)}};      // L = 2, a corner
    UsdGenCurveSetDesc hair = HairSet(SdfPath("/groom/hair"), curves);
    AddIntPlane(hair, "strandSeed", {5, 6, 7, 8});
    std::vector<float> ramp;
    for (Curve const &c : curves)
        for (size_t i = 0; i < c.size(); ++i) ramp.push_back(float(i) / float(c.size() - 1));
    AddFloatPlane(hair, "tParam", ramp, UsdGenAuthoredPlaneDomain::Point);
    UsdGenGraphDesc desc = BaseDesc(hair);
    UsdGenNodeDesc resample = OpNode("/groom/resample", "UsdGenResample", SdfPath("/groom/source"));
    Param(resample, "spacing", VtValue(0.125f));
    desc.nodes.push_back(resample);
    desc.terminal = resample.path;
    UsdGenCurveBuffer out;
    bool ok = Cook(desc, &out);
    Check(ok, "Resample spacing cooks");
    if (!ok) return;
    Spans const s(out);
    // clamp(ceil(L / 0.125) + 1, 2, 4096): 9, 6 (4.8 -> 5), 2 (0.4 -> 1 -> 2), 17.
    uint32_t const want[4] = {9, 6, 2, 17};
    bool counts = out.totalCurves == 4 && out.totalCvs == 34 && out.cvOffsets.size() == 5;
    for (uint32_t c = 0; counts && c < 4; ++c) counts = s.N(c) == want[c];
    Check(counts, "per-strand counts follow ceil(L / spacing) + 1 (ragged output)");
    if (!counts) return;
    bool uniform = true;
    for (uint32_t c = 0; c < 4; ++c) {
        float const L = c == 0 ? 1.0f : c == 1 ? 0.6f : c == 2 ? 0.05f : 2.0f;
        float const seg = L / float(want[c] - 1);
        for (uint32_t j = 1; j < want[c]; ++j)
            uniform = uniform && Near((Pt(out, s, c, j) - Pt(out, s, c, j - 1)).GetLength(), seg, 2e-5f);
        uniform = uniform && Pt(out, s, c, 0) == curves[c].front() &&
                  Near((Pt(out, s, c, want[c] - 1) - curves[c].back()).GetLength(), 0.0f, 1e-5f);
    }
    Check(uniform, "samples are uniform in arc length, root and tip exact");
    Check(Near(Pt(out, s, 3, 8)[0], 3.0f) && Near(Pt(out, s, 3, 8)[1], 1.0f) &&
          Near(Pt(out, s, 3, 12)[0], 3.5f), "the corner strand is sampled along its arc");
    bool hairT = out.hairT.size() == out.totalCvs;
    for (uint32_t j = 0; hairT && j < 17; ++j) hairT = Near(out.hairT[s.first[3] + j], float(j) / 16.0f);
    Check(hairT, "hairT runs 0 -> 1 over every ragged strand");
    UsdGenPlane const *seed = CurvePlane(out, "strandSeed");
    UsdGenPlane const *tParam = CvPlane(out, "tParam");
    Check(seed && seed->i.size() == 4 && seed->i[3] == 8, "uniform planes are carried");
    Check(tParam && tParam->f.size() == 34 && Near(tParam->f[s.first[3] + 16], 1.0f) &&
          Near(tParam->f[s.first[0] + 4], 0.5f), "vertex planes follow the new CV counts");
    Check(out.width.size() == 34 && out.rest.size() == 34, "width and rest follow the new counts");

    // A ragged downstream sweep: Smooth box over the resampled strands.
    UsdGenGraphDesc chain = desc;
    UsdGenNodeDesc smooth = OpNode("/groom/smooth", "UsdGenSmooth", resample.path);
    Param(smooth, "mode", VtValue(TfToken("box")));
    chain.nodes.push_back(smooth);
    chain.terminal = smooth.path;
    UsdGenCurveBuffer smoothed;
    ok = Cook(chain, &smoothed);
    Spans const ss(smoothed);
    Check(ok && smoothed.totalCvs == 34 && ss.N(3) == 17 &&
          Pt(smoothed, ss, 3, 16) == Pt(out, s, 3, 16) && Pt(smoothed, ss, 3, 8)[0] > 3.0f,
          "a downstream Smooth sweeps the ragged layout (corner rounded, tip fixed)");

    UsdGenCurveBuffer again;
    Check(Cook(desc, &again) && SamePoints(out, again), "spacing resample is deterministic");
}

int main()
{
    usdGenRegisterM1Operators();
    TestLengthDefaultMax();
    TestLengthRandomRange();
    TestLengthShareByClump();
    TestLengthFollowClumpMean();
    TestLengthMask();
    TestTrim();
    TestSmoothBox();
    TestResampleSpacing();
    std::printf(failures ? "FAILED (%d)\n" : "testUsdGenLengthTrim passed\n", failures);
    return failures ? 1 : 0;
}
