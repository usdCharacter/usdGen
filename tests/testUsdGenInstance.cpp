// UsdGenInstance instancer contract: bake + Hydra assembly.
//
// The only operator that emits no curves (02 §2.11, 06 §4.3; v1, M6).
// Drives usdGenImaging::UsdGenInstancer directly over hand-built evaluated
// curve buffers (no engine, no scene index):
//   (1) inst_/Prototypes path spellings;
//   (2) fail-closed params (archives/unknown primitive, camera/unknown
//       orient, empty/mismatched weights, bad width/length) and fail-closed
//       inputs (no curves, ragged/planar mismatch, missing curveId/frames,
//       bad displayColor, empty/duplicate prototype targets);
//   (3) counts + prototype partition coverage;
//   (4) orient surfaceFrame (translations, identity rotation, card scales,
//       normalOffset) and curveTangent (tangent-following frame, rootB
//       fallback, tangent-parallel-to-normal fallback);
//   (5) weights incl. determinism;
//   (6) twist/scale incl. scaleRandom range and twist handedness;
//   (7) width knots + interpolation (linear/constant/unknown-fallback);
//   (8) spheres (identity rotation, uniform width scale, length/knots
//       ignored, frameless world bake);
//   (9) variationPrimvars (displayColor per-curve/per-CV, float/int/constant
//       extra planes, unresolved names, dedupe);
//  (10) Hydra data sources (topology round-trip, instance primvars,
//       primOrigin, purpose/visibility, exactly-one instancedBy path,
//       MoonRay wire types: quath rotations, packed varyings, color role);
//  (11) notice locators (Translations, never topology, for value edits).
//  (12) inline rotation quat matches ExtractRotationQuat bit for bit.
//  (13) threaded ranges: two big bakes agree bitwise on every output,
//       incl. the per-CV gather and the arity-2 pack at 40K curves.
#include "usdGenImaging/usdGenInstancer.h"

#include "usdGen/curveBuffer.h"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/quatd.h"
#include "pxr/base/gf/quath.h"
#include "pxr/base/gf/quatf.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/instancedBySchema.h"
#include "pxr/imaging/hd/instancerTopologySchema.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/imaging/hd/xformSchema.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGenImaging;

static int failures = 0;
static void Check(bool v, char const *s)
{
    if (!v) { ++failures; std::printf("FAIL: %s\n", s); }
    else std::printf("ok: %s\n", s);
}

constexpr float kTol = 1e-4f;
static bool Near(float a, float b, float tol = kTol)
{
    return std::fabs(a - b) <= tol;
}
static bool NearVec(GfVec3f const &a, GfVec3f const &b, float tol = kTol)
{
    return Near(a[0], b[0], tol) && Near(a[1], b[1], tol) && Near(a[2], b[2], tol);
}

// Leaf sample value at Default() time, or empty VtValue.
static VtValue LeafValue(HdDataSourceBaseHandle ds)
{
    if (HdSampledDataSourceHandle s = HdSampledDataSource::Cast(ds))
        return s->GetValue(0.0);
    return VtValue();
}
static bool LeafIsToken(VtValue const &v, char const *text)
{
    if (v.IsHolding<TfToken>()) return v.UncheckedGet<TfToken>() == TfToken(text);
    if (v.IsHolding<std::string>()) return v.UncheckedGet<std::string>() == text;
    return false;
}
static HdContainerDataSourceHandle Child(HdContainerDataSourceHandle const &c,
                                         TfToken const &name)
{
    if (!c) return nullptr;
    return HdContainerDataSource::Cast(c->Get(name));
}
static HdContainerDataSourceHandle Child(HdContainerDataSourceHandle const &c,
                                         char const *name)
{
    return Child(c, TfToken(name));
}

// Straight strands along +Y from the given roots with identity root frames,
// so rootT=(1,0,0), rootB=(0,1,0), rootN=(0,0,1) and the root segment
// tangent equals rootB. curveId[c] = c.
static UsdGenCurveBuffer StraightStrands(std::vector<GfVec3f> const &roots,
                                         int cvs, float step)
{
    UsdGenCurveBuffer curves;
    uint32_t const n = uint32_t(roots.size());
    curves.totalCurves = n;
    curves.totalCvs = n * uint32_t(cvs);
    curves.px.resize(curves.totalCvs);
    curves.py.resize(curves.totalCvs);
    curves.pz.resize(curves.totalCvs);
    for (uint32_t c = 0; c != n; ++c)
        for (int i = 0; i < cvs; ++i) {
            uint32_t const o = c * uint32_t(cvs) + uint32_t(i);
            curves.px[o] = roots[c][0];
            curves.py[o] = roots[c][1] + float(i) * step;
            curves.pz[o] = roots[c][2];
        }
    curves.curveId.resize(n);
    for (uint32_t c = 0; c != n; ++c) curves.curveId[c] = uint64_t(c);
    curves.rootT.assign(n, GfVec3f(1, 0, 0));
    curves.rootB.assign(n, GfVec3f(0, 1, 0));
    curves.rootN.assign(n, GfVec3f(0, 0, 1));
    return curves;
}

static UsdGenInstanceParams CardsParams()
{
    UsdGenInstanceParams params;
    params.prototypes = {SdfPath("/groom/Prototypes/cardA"),
                         SdfPath("/groom/Prototypes/cardB")};
    return params;
}

static bool BakeOk(UsdGenInstanceParams const &params,
                   UsdGenInstanceCurves const &input,
                   UsdGenInstanceResult *result)
{
    std::string error;
    bool const ok = UsdGenInstancer::Bake(
        params, input,
        SdfPath("/groom/__usdGenRender/inst_op"), result, &error);
    if (!ok) std::printf("  bake error: %s\n", error.c_str());
    return ok;
}

static bool SameBake(UsdGenInstanceResult const &a,
                     UsdGenInstanceResult const &b)
{
    if (!(a.translations == b.translations && a.rotations == b.rotations &&
          a.scales == b.scales && a.prototypeIndex == b.prototypeIndex &&
          a.instanceIndices.size() == b.instanceIndices.size()))
        return false;
    for (size_t i = 0; i != a.instanceIndices.size(); ++i)
        if (a.instanceIndices[i] != b.instanceIndices[i])
            return false;
    return true;
}

// --- (1) paths ----------------------------------------------------------

static void CheckPaths()
{
    SdfPath const inst = UsdGenInstancer::InstancerPath(
        SdfPath("/Bird/Groom/cards"), "op02_cards");
    Check(inst == SdfPath("/Bird/Groom/cards/__usdGenRender/inst_op02_cards"),
          "instancer path is <description>/__usdGenRender/inst_<op>");
    SdfPath const proto = UsdGenInstancer::PrototypePath(inst, "cardA");
    Check(proto == SdfPath("/Bird/Groom/cards/__usdGenRender/"
                            "inst_op02_cards/Prototypes/cardA"),
          "prototype path is <instancer>/Prototypes/<name>");
    Check(UsdGenInstancer::InstancerPath(SdfPath("/g"), "").IsEmpty(),
          "empty op name fails closed to an empty path");
    Check(UsdGenInstancer::PrototypePath(inst, "").IsEmpty(),
          "empty prototype name fails closed to an empty path");
}

// --- (2) fail-closed ------------------------------------------------------

static void CheckFailClosed()
{
    UsdGenInstanceParams const base = CardsParams();
    // Params alone.
    Check(UsdGenInstancer::Validate(base, 2), "cards validate");
    UsdGenInstanceParams spheres = base;
    spheres.primitive = TfToken("spheres");
    Check(UsdGenInstancer::Validate(spheres, 2), "spheres validate");
    for (char const *orient : {"surfaceFrame", "curveTangent", "world"}) {
        UsdGenInstanceParams p = base;
        p.orient = TfToken(orient);
        Check(UsdGenInstancer::Validate(p, 2), "orient validates");
    }
    std::string error;
    UsdGenInstanceParams archives = base;
    archives.primitive = TfToken("archives");
    Check(!UsdGenInstancer::Validate(archives, 2, &error) &&
          error.find("archives") != std::string::npos,
          "primitive=archives fails closed");
    UsdGenInstanceParams bogus = base;
    bogus.primitive = TfToken("tubes");
    Check(!UsdGenInstancer::Validate(bogus, 2), "unknown primitive fails closed");
    UsdGenInstanceParams camera = base;
    camera.orient = TfToken("camera");
    Check(!UsdGenInstancer::Validate(camera, 2, &error) &&
          error.find("camera") != std::string::npos,
          "orient=camera fails closed");
    UsdGenInstanceParams orientBogus = base;
    orientBogus.orient = TfToken("view");
    Check(!UsdGenInstancer::Validate(orientBogus, 2),
          "unknown orient fails closed");
    Check(!UsdGenInstancer::Validate(base, 0), "zero prototypes fail closed");
    UsdGenInstanceParams shortWeights = base;
    shortWeights.weights = VtFloatArray{0.5f};
    Check(!UsdGenInstancer::Validate(shortWeights, 2),
          "weights/prototype size mismatch fails closed");
    UsdGenInstanceParams negWeights = base;
    negWeights.weights = VtFloatArray{0.5f, -0.5f};
    Check(!UsdGenInstancer::Validate(negWeights, 2),
          "negative weights fail closed");
    UsdGenInstanceParams zeroWeights = base;
    zeroWeights.weights = VtFloatArray{0.0f, 0.0f};
    Check(!UsdGenInstancer::Validate(zeroWeights, 2),
          "all-zero weights fail closed");
    UsdGenInstanceParams negWidth = base;
    negWidth.width = -0.01f;
    Check(!UsdGenInstancer::Validate(negWidth, 2), "negative width fails closed");
    UsdGenInstanceParams negLength = base;
    negLength.length = -0.01f;
    Check(!UsdGenInstancer::Validate(negLength, 2),
          "negative length fails closed");

    // Inputs.
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    UsdGenCurveBuffer curves = StraightStrands(roots, 4, 0.25f);
    UsdGenInstanceCurves input;
    input.curves = &curves;
    UsdGenInstanceResult result;
    Check(!BakeOk(base, UsdGenInstanceCurves(), &result),
          "null curves fail closed");
    UsdGenCurveBuffer empty;
    UsdGenInstanceCurves emptyInput;
    emptyInput.curves = &empty;
    Check(!BakeOk(base, emptyInput, &result), "zero curves fail closed");
    UsdGenCurveBuffer clipped = curves;
    clipped.px.resize(clipped.px.size() - 1);
    UsdGenInstanceCurves clippedInput;
    clippedInput.curves = &clipped;
    Check(!BakeOk(base, clippedInput, &result),
          "point-plane/CV-count mismatch fails closed");
    UsdGenCurveBuffer noIds = curves;
    noIds.curveId.clear();
    UsdGenInstanceCurves noIdsInput;
    noIdsInput.curves = &noIds;
    Check(!BakeOk(base, noIdsInput, &result), "missing curveId fails closed");
    UsdGenCurveBuffer noFrames = curves;
    noFrames.rootT.clear();
    noFrames.rootN.clear();
    noFrames.rootB.clear();
    UsdGenInstanceCurves noFramesInput;
    noFramesInput.curves = &noFrames;
    Check(!BakeOk(base, noFramesInput, &result),
          "missing root frames fail surfaceFrame closed");
    UsdGenInstanceCurves badColor = input;
    badColor.displayColor.resize(3);  // neither per-curve (2) nor per-CV (8)
    Check(!BakeOk(base, badColor, &result),
          "mis-sized displayColor fails closed");
    UsdGenInstanceParams emptyTarget = base;
    emptyTarget.prototypes = {SdfPath::EmptyPath(), SdfPath("/g/cardB")};
    Check(!BakeOk(emptyTarget, input, &result),
          "empty prototype target fails closed");
    UsdGenInstanceParams dupLeaf = base;
    dupLeaf.prototypes = {SdfPath("/a/Prototypes/card"),
                          SdfPath("/b/Prototypes/card")};
    Check(!BakeOk(dupLeaf, input, &result),
          "duplicate re-rooted prototype leaves fail closed");
    UsdGenCurveBuffer ragged = curves;
    ragged.cvOffsets = VtIntArray{0, 3, 8, 5};  // bad size + decreasing
    UsdGenInstanceCurves raggedInput;
    raggedInput.curves = &ragged;
    Check(!BakeOk(base, raggedInput, &result),
          "invalid ragged offsets fail closed");
    Check(BakeOk(base, input, &result), "valid cards bake");
}

// --- (3) counts + partition -------------------------------------------------

static void CheckCountsAndPartition()
{
    std::vector<GfVec3f> roots;
    for (int i = 0; i < 8; ++i) roots.push_back(GfVec3f(float(i), 0, 0));
    UsdGenCurveBuffer curves = StraightStrands(roots, 4, 0.25f);
    UsdGenInstanceCurves input;
    input.curves = &curves;
    UsdGenInstanceParams params = CardsParams();
    params.prototypes.push_back(SdfPath("/groom/Prototypes/quill"));
    params.weights = VtFloatArray{0.6f, 0.3f, 0.1f};
    UsdGenInstanceResult result;
    Check(BakeOk(params, input, &result), "weighted bake cooks");
    Check(result.translations.size() == 8 && result.rotations.size() == 8 &&
          result.scales.size() == 8 && result.prototypeIndex.size() == 8,
          "one instance-rate entry per curve");
    Check(result.prototypePaths.size() == 3 && result.instanceIndices.size() == 3,
          "one topology entry per prototype");
    Check(result.mask.empty(), "mask stays empty (all true)");
    size_t covered = 0;
    std::set<int> seen;
    bool consistent = true;
    for (size_t p = 0; p != 3; ++p) {
        covered += result.instanceIndices[p].size();
        for (int id : result.instanceIndices[p]) {
            seen.insert(id);
            consistent = consistent && id >= 0 && id < 8 &&
                result.prototypeIndex[size_t(id)] == int(p);
        }
    }
    Check(covered == 8 && seen.size() == 8,
          "instanceIndices partition every instance exactly once");
    Check(consistent, "prototypeIndex agrees with the partition");
    Check(result.prototypePaths[0] ==
              SdfPath("/groom/__usdGenRender/inst_op/Prototypes/cardA") &&
          result.prototypePaths[2] ==
              SdfPath("/groom/__usdGenRender/inst_op/Prototypes/quill"),
          "prototypes re-root as instancer namespace children");
}

// --- (4) orient ---------------------------------------------------------------

static void CheckSurfaceFrame()
{
    std::vector<GfVec3f> const roots = {GfVec3f(1, 2, 3), GfVec3f(4, 5, 6)};
    UsdGenCurveBuffer curves = StraightStrands(roots, 4, 0.25f);
    UsdGenInstanceCurves input;
    input.curves = &curves;
    UsdGenInstanceResult result;
    Check(BakeOk(CardsParams(), input, &result), "surfaceFrame bakes");
    bool placed = NearVec(result.translations[0], roots[0]) &&
        NearVec(result.translations[1], roots[1]);
    Check(placed, "surfaceFrame translates instances to strand roots");
    bool ident = true;
    for (size_t i = 0; i != 2; ++i) {
        GfQuatf const q = result.rotations[i];
        ident = ident && Near(q.GetReal(), 1.0f) &&
            NearVec(q.GetImaginary(), GfVec3f(0, 0, 0)) &&
            Near(q.GetLength(), 1.0f);
        ident = ident && NearVec(result.scales[i], GfVec3f(0.02f, 0.09f, 1.0f));
    }
    Check(ident, "axis frames give identity rotation + (width,length,1) scale");

    UsdGenInstanceParams offset = CardsParams();
    offset.normalOffset = 0.02f;
    Check(BakeOk(offset, input, &result), "normalOffset bakes");
    Check(NearVec(result.translations[0], roots[0] + GfVec3f(0, 0, 0.02f)),
          "normalOffset pushes along rootN");

    UsdGenInstanceParams world = CardsParams();
    world.orient = TfToken("world");
    UsdGenCurveBuffer noFrames = curves;
    noFrames.rootT.clear();
    noFrames.rootN.clear();
    noFrames.rootB.clear();
    UsdGenInstanceCurves noFramesInput;
    noFramesInput.curves = &noFrames;
    Check(BakeOk(world, noFramesInput, &result),
          "world bakes without root frames");
    Check(Near(result.rotations[0].GetReal(), 1.0f),
          "world orient is the identity rotation");
}

static void CheckCurveTangent()
{
    // (a) Strands along +Y: tangent == rootB, so curveTangent matches
    // surfaceFrame exactly.
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    UsdGenCurveBuffer curves = StraightStrands(roots, 4, 0.25f);
    UsdGenInstanceCurves input;
    input.curves = &curves;
    UsdGenInstanceParams tangent = CardsParams();
    tangent.orient = TfToken("curveTangent");
    UsdGenInstanceResult a, b;
    Check(BakeOk(tangent, input, &a), "curveTangent bakes");
    Check(BakeOk(CardsParams(), input, &b), "surfaceFrame reference bakes");
    bool same = a.translations.size() == b.translations.size();
    for (size_t i = 0; same && i != b.translations.size(); ++i) {
        same = NearVec(a.translations[i], b.translations[i]) &&
            Near(a.rotations[i].GetReal(), b.rotations[i].GetReal()) &&
            NearVec(a.rotations[i].GetImaginary(),
                    b.rotations[i].GetImaginary()) &&
            NearVec(a.scales[i], b.scales[i]);
    }
    Check(same, "curveTangent matches surfaceFrame when tangent == rootB");

    // (b) Strand along +X: local Y (length axis) must map to the tangent.
    UsdGenCurveBuffer sideways;
    sideways.totalCurves = 1;
    sideways.totalCvs = 2;
    sideways.px = VtFloatArray{0.0f, 1.0f};
    sideways.py = VtFloatArray{0.0f, 0.0f};
    sideways.pz = VtFloatArray{0.0f, 0.0f};
    sideways.curveId = VtArray<uint64_t>{7u};
    sideways.rootT.assign(1, GfVec3f(1, 0, 0));
    sideways.rootB.assign(1, GfVec3f(0, 1, 0));
    sideways.rootN.assign(1, GfVec3f(0, 0, 1));
    UsdGenInstanceCurves sidewaysInput;
    sidewaysInput.curves = &sideways;
    UsdGenInstanceResult s;
    Check(BakeOk(tangent, sidewaysInput, &s), "sideways tangent bakes");
    GfQuatf const q = s.rotations[0];
    Check(NearVec(q.Transform(GfVec3f(0, 1, 0)), GfVec3f(1, 0, 0)),
          "curveTangent maps local Y onto the strand tangent");
    Check(NearVec(q.Transform(GfVec3f(0, 0, 1)), GfVec3f(0, 0, 1)),
          "curveTangent keeps the surface normal as local Z");
    Check(Near(q.GetLength(), 1.0f), "tangent rotation is a unit quaternion");

    // (c) Single-CV span: legal topology, falls back to rootB.
    UsdGenCurveBuffer single = sideways;
    single.totalCvs = 1;
    single.px = VtFloatArray{2.0f};
    single.py = VtFloatArray{3.0f};
    single.pz = VtFloatArray{4.0f};
    UsdGenInstanceCurves singleInput;
    singleInput.curves = &single;
    UsdGenInstanceResult one;
    Check(BakeOk(tangent, singleInput, &one), "single-CV tangent bakes");
    Check(Near(one.rotations[0].GetReal(), 1.0f) &&
          NearVec(one.translations[0], GfVec3f(2, 3, 4)),
          "single-CV tangent falls back to the rootB frame");

    // (d) Strand along +Z (parallel to rootN): the rootT fallback keeps the
    // frame well-defined and local Y still maps to the tangent.
    UsdGenCurveBuffer vertical = sideways;
    vertical.px = VtFloatArray{0.0f, 0.0f};
    vertical.py = VtFloatArray{0.0f, 0.0f};
    vertical.pz = VtFloatArray{0.0f, 1.0f};
    UsdGenInstanceCurves verticalInput;
    verticalInput.curves = &vertical;
    UsdGenInstanceResult v;
    Check(BakeOk(tangent, verticalInput, &v), "normal-parallel tangent bakes");
    Check(NearVec(v.rotations[0].Transform(GfVec3f(0, 1, 0)), GfVec3f(0, 0, 1)),
          "normal-parallel tangent still maps local Y onto the strand");
}

// --- (5) weights ------------------------------------------------------------------

static void CheckWeights()
{
    std::vector<GfVec3f> roots;
    for (int i = 0; i < 8; ++i) roots.push_back(GfVec3f(float(i), 0, 0));
    UsdGenCurveBuffer curves = StraightStrands(roots, 4, 0.25f);
    UsdGenInstanceCurves input;
    input.curves = &curves;

    UsdGenInstanceParams one = CardsParams();
    one.prototypes = {SdfPath("/groom/Prototypes/only")};
    UsdGenInstanceResult r1;
    Check(BakeOk(one, input, &r1), "single prototype bakes");
    bool allZero = r1.instanceIndices.size() == 1;
    for (size_t i = 0; allZero && i != 8; ++i)
        allZero = r1.prototypeIndex[i] == 0 &&
            r1.instanceIndices[0][i] == int(i);
    Check(allZero, "single prototype owns every instance in order");

    UsdGenInstanceParams skewed = CardsParams();
    skewed.prototypes.push_back(SdfPath("/groom/Prototypes/quill"));
    skewed.weights = VtFloatArray{1.0f, 0.0f, 0.0f};
    UsdGenInstanceResult r2;
    Check(BakeOk(skewed, input, &r2), "skewed weights bake");
    bool first = true;
    for (size_t i = 0; i != 8; ++i) first = first && r2.prototypeIndex[i] == 0;
    Check(first && r2.instanceIndices[1].empty() && r2.instanceIndices[2].empty(),
          "[1,0,0] weights bind every instance to prototype 0");

    UsdGenInstanceParams varied = CardsParams();
    varied.weights = VtFloatArray{0.6f, 0.4f};
    UsdGenInstanceResult r3, r4;
    Check(BakeOk(varied, input, &r3) && BakeOk(varied, input, &r4),
          "weighted bake repeats");
    bool stable = true;
    for (size_t i = 0; i != 8; ++i)
        stable = stable && r3.prototypeIndex[i] == r4.prototypeIndex[i] &&
            NearVec(r3.translations[i], r4.translations[i]) &&
            NearVec(r3.scales[i], r4.scales[i]);
    Check(stable, "hashed prototype/scale choices are deterministic");

    UsdGenInstanceResult r5;
    Check(BakeOk(CardsParams(), input, &r5), "empty weights bake");
    size_t covered = r5.instanceIndices[0].size() + r5.instanceIndices[1].size();
    Check(covered == 8, "empty weights partition uniformly over prototypes");
}

// --- (6) twist/scale -----------------------------------------------------------------

static void CheckTwistAndScale()
{
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    UsdGenCurveBuffer curves = StraightStrands(roots, 4, 0.25f);
    UsdGenInstanceCurves input;
    input.curves = &curves;

    UsdGenInstanceParams twisted = CardsParams();
    twisted.twist = 90.0f;
    UsdGenInstanceResult t;
    Check(BakeOk(twisted, input, &t), "twist=90 bakes");
    // Local-space R_y(+90): X -> -Z, Z -> +X.
    Check(NearVec(t.rotations[0].Transform(GfVec3f(1, 0, 0)), GfVec3f(0, 0, -1)),
          "twist=90 sends local X to -Z");
    Check(NearVec(t.rotations[0].Transform(GfVec3f(0, 0, 1)), GfVec3f(1, 0, 0)),
          "twist=90 sends local Z to +X");
    Check(NearVec(t.rotations[0].Transform(GfVec3f(0, 1, 0)), GfVec3f(0, 1, 0)),
          "twist holds the length axis fixed");

    UsdGenInstanceParams grown = CardsParams();
    grown.scale = 2.0f;
    UsdGenInstanceResult g;
    Check(BakeOk(grown, input, &g), "scale=2 bakes");
    Check(NearVec(g.scales[0], GfVec3f(0.04f, 0.18f, 2.0f)),
          "scale multiplies the (width,length,1) card scale");

    UsdGenInstanceParams varied = CardsParams();
    varied.scaleRandom = GfVec2f(0.5f, 1.5f);
    UsdGenInstanceResult v, swapped;
    Check(BakeOk(varied, input, &v), "scaleRandom bakes");
    bool inRange = true;
    for (size_t i = 0; i != 2; ++i) {
        float const s = v.scales[i][1] / 0.09f;
        inRange = inRange && s >= 0.5f - kTol && s <= 1.5f + kTol &&
            Near(v.scales[i][0], 0.02f * s) && Near(v.scales[i][2], s);
    }
    Check(inRange, "scaleRandom draws a uniform per-instance multiplier");
    UsdGenInstanceParams flip = varied;
    flip.scaleRandom = GfVec2f(1.5f, 0.5f);
    Check(BakeOk(flip, input, &swapped), "swapped scaleRandom bakes");
    Check(NearVec(v.scales[0], swapped.scales[0]) &&
          NearVec(v.scales[1], swapped.scales[1]),
          "scaleRandom is order-insensitive");

    UsdGenInstanceParams jitter = CardsParams();
    jitter.twistRandom = 30.0f;
    UsdGenInstanceResult j, j2;
    Check(BakeOk(jitter, input, &j) && BakeOk(jitter, input, &j2),
          "twistRandom bakes repeatably");
    Check(Near(j.rotations[0].GetReal(), j2.rotations[0].GetReal()) &&
          Near(j.rotations[0].GetLength(), 1.0f),
          "twistRandom draws are deterministic unit quaternions");
    UsdGenInstanceResult flat;
    Check(BakeOk(CardsParams(), input, &flat), "zero twist reference bakes");
    Check(Near(flat.rotations[0].GetReal(), flat.rotations[1].GetReal()),
          "zero twistRandom gives identical per-instance rotations");
}

// --- (7) width knots ------------------------------------------------------------------

static void CheckWidthKnots()
{
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0)};
    UsdGenCurveBuffer curves = StraightStrands(roots, 4, 0.25f);
    UsdGenInstanceCurves input;
    input.curves = &curves;

    UsdGenInstanceParams linear = CardsParams();
    linear.widthKnots = VtVec2fArray{GfVec2f(0, 1), GfVec2f(1, 0.2f)};
    linear.widthInterpolation = TfToken("linear");
    UsdGenInstanceResult l;
    Check(BakeOk(linear, input, &l), "linear width knots bake");
    Check(Near(l.scales[0][0], 0.02f * 0.6f),
          "linear knots scale card width by the mid-strand value");

    UsdGenInstanceParams held = linear;
    held.widthInterpolation = TfToken("constant");
    UsdGenInstanceResult h;
    Check(BakeOk(held, input, &h), "constant width knots bake");
    Check(Near(h.scales[0][0], 0.02f * 1.0f),
          "constant knots hold the previous knot value");

    UsdGenInstanceParams fallback = linear;
    fallback.widthInterpolation = TfToken("smoothstep");
    UsdGenInstanceResult f;
    Check(BakeOk(fallback, input, &f), "unknown interpolation still bakes");
    Check(Near(f.scales[0][0], 0.02f * 0.6f),
          "unknown interpolation falls back to linear");
}

// --- (8) spheres ------------------------------------------------------------------------

static void CheckSpheres()
{
    std::vector<GfVec3f> const roots = {GfVec3f(1, 2, 3)};
    UsdGenCurveBuffer curves = StraightStrands(roots, 4, 0.25f);
    UsdGenInstanceCurves input;
    input.curves = &curves;
    UsdGenInstanceParams spheres = CardsParams();
    spheres.primitive = TfToken("spheres");
    spheres.prototypes = {SdfPath("/groom/Prototypes/ball")};
    spheres.twist = 45.0f;
    spheres.length = 5.0f;
    spheres.widthKnots = VtVec2fArray{GfVec2f(0, 1), GfVec2f(1, 0.1f)};
    UsdGenInstanceResult s;
    Check(BakeOk(spheres, input, &s), "spheres bake");
    Check(Near(s.rotations[0].GetReal(), 1.0f) &&
          NearVec(s.rotations[0].GetImaginary(), GfVec3f(0, 0, 0)),
          "spheres use identity rotation (twist is a no-op)");
    Check(NearVec(s.scales[0], GfVec3f(0.02f)),
          "spheres scale uniformly by the width diameter");
    UsdGenInstanceParams plain = spheres;
    plain.length = 0.09f;
    plain.widthKnots.clear();
    plain.twist = 0.0f;
    UsdGenInstanceResult p;
    Check(BakeOk(plain, input, &p), "plain spheres bake");
    Check(NearVec(p.scales[0], s.scales[0]),
          "spheres ignore length and width knots");

    UsdGenInstanceParams free = plain;
    free.orient = TfToken("world");
    UsdGenCurveBuffer noFrames = curves;
    noFrames.rootT.clear();
    noFrames.rootN.clear();
    noFrames.rootB.clear();
    UsdGenInstanceCurves noFramesInput;
    noFramesInput.curves = &noFrames;
    UsdGenInstanceResult w;
    Check(BakeOk(free, noFramesInput, &w),
          "spheres+world bake with no root frames at all");
}

// --- (9) variation primvars ------------------------------------------------------------------

static void CheckVariationPrimvars()
{
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    UsdGenCurveBuffer curves = StraightStrands(roots, 4, 0.25f);
    UsdGenPlane age;
    age.name = TfToken("usdGen:featherAge");
    age.interpolation = TfToken("uniform");
    age.type = TfToken("float");
    age.arity = 1;
    age.f = VtFloatArray{0.25f, 0.75f};
    curves.extraCurve.push_back(age);
    UsdGenPlane zone;
    zone.name = TfToken("usdGen:zone");
    zone.interpolation = TfToken("constant");
    zone.type = TfToken("int");
    zone.arity = 1;
    zone.i = VtIntArray{3};
    curves.extraCurve.push_back(zone);
    UsdGenInstanceCurves input;
    input.curves = &curves;
    input.displayColor =
        VtVec3fArray{GfVec3f(1, 0, 0), GfVec3f(0, 0, 1)};

    UsdGenInstanceParams params = CardsParams();
    params.variationPrimvars = VtArray<TfToken>{
        TfToken("displayColor"), TfToken("usdGen:featherAge"),
        TfToken("usdGen:zone"), TfToken("usdGen:missing")};
    UsdGenInstanceResult result;
    Check(BakeOk(params, input, &result), "variation primvars bake");
    Check(result.varyings.size() == 3, "known varyings publish, all three");
    bool color = false, ageOk = false, zoneOk = false;
    for (UsdGenPlane const &plane : result.varyings) {
        if (plane.name == TfToken("displayColor")) {
            color = plane.type == TfToken("float") && plane.arity == 3 &&
                plane.f.size() == 6 && Near(plane.f[0], 1.0f) &&
                Near(plane.f[5], 1.0f);
        } else if (plane.name == TfToken("usdGen:featherAge")) {
            ageOk = plane.type == TfToken("float") && plane.f.size() == 2 &&
                Near(plane.f[0], 0.25f) && Near(plane.f[1], 0.75f);
        } else if (plane.name == TfToken("usdGen:zone")) {
            zoneOk = plane.type == TfToken("int") && plane.i.size() == 2 &&
                plane.i[0] == 3 && plane.i[1] == 3;
        }
    }
    Check(color, "displayColor publishes per-instance RGB");
    Check(ageOk, "float extra planes publish per-instance values");
    Check(zoneOk, "constant extra planes broadcast to every instance");
    Check(result.unresolvedPrimvars.size() == 1 &&
          result.unresolvedPrimvars[0] == TfToken("usdGen:missing"),
          "names with no baked source publish nothing and are reported");

    // Per-CV displayColor samples each curve's root CV (CV 0 and CV 4).
    UsdGenInstanceCurves vertexInput = input;
    vertexInput.displayColor = VtVec3fArray{
        GfVec3f(1, 0, 0), GfVec3f(0, 1, 0), GfVec3f(0, 1, 0), GfVec3f(0, 1, 0),
        GfVec3f(0, 0, 1), GfVec3f(0, 1, 0), GfVec3f(0, 1, 0), GfVec3f(0, 1, 0)};
    UsdGenInstanceResult vertex;
    Check(BakeOk(params, vertexInput, &vertex), "per-CV displayColor bakes");
    bool rootsOnly = false;
    for (UsdGenPlane const &plane : vertex.varyings) {
        if (plane.name == TfToken("displayColor"))
            rootsOnly = plane.f.size() == 6 && Near(plane.f[0], 1.0f) &&
                Near(plane.f[5], 1.0f);
    }
    Check(rootsOnly, "per-CV displayColor is sampled at the root CV");

    // Defaults: empty authored list means {"displayColor"}; with no colors
    // baked that name is reported, not invented.
    UsdGenInstanceCurves bare;
    bare.curves = &curves;
    UsdGenInstanceResult bareResult;
    Check(BakeOk(CardsParams(), bare, &bareResult), "colorless bake cooks");
    Check(bareResult.varyings.empty() &&
          bareResult.unresolvedPrimvars.size() == 1 &&
          bareResult.unresolvedPrimvars[0] == TfToken("displayColor"),
          "default displayColor with no source is reported unresolved");

    UsdGenInstanceParams dupes = CardsParams();
    dupes.variationPrimvars = VtArray<TfToken>{
        TfToken("displayColor"), TfToken("displayColor")};
    UsdGenInstanceResult duped;
    Check(BakeOk(dupes, input, &duped), "duplicate variation names bake");
    Check(duped.varyings.size() == 1, "duplicate names publish once");
}

// --- (10) Hydra data sources -------------------------------------------------------------------

static void CheckDataSources()
{
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    UsdGenCurveBuffer curves = StraightStrands(roots, 4, 0.25f);
    UsdGenInstanceCurves input;
    input.curves = &curves;
    input.displayColor =
        VtVec3fArray{GfVec3f(1, 0, 0), GfVec3f(0, 0, 1)};
    UsdGenInstanceResult result;
    Check(BakeOk(CardsParams(), input, &result), "data-source bake cooks");
    result.purpose = TfToken("render");
    HdContainerDataSourceHandle c = UsdGenInstancer::BuildInstancerDataSource(
        result, SdfPath("/groom"));
    Check(c != nullptr, "instancer data source builds");

    HdInstancerTopologySchema topo =
        HdInstancerTopologySchema::GetFromParent(c);
    Check(topo.GetPrototypes() != nullptr, "topology carries prototypes");
    VtValue protoValue = topo.GetPrototypes()->GetValue(0.0);
    bool protoOk = protoValue.IsHolding<VtArray<SdfPath>>() &&
        protoValue.UncheckedGet<VtArray<SdfPath>>().size() == 2 &&
        protoValue.UncheckedGet<VtArray<SdfPath>>()[0] == result.prototypePaths[0];
    Check(protoOk, "prototypes round-trip the re-rooted paths");
    HdIntArrayVectorSchema indices = topo.GetInstanceIndices();
    bool idxOk = indices.GetNumElements() == 2;
    for (size_t p = 0; idxOk && p != 2; ++p) {
        VtValue v = indices.GetElement(p)->GetValue(0.0);
        idxOk = v.IsHolding<VtIntArray>() &&
            v.UncheckedGet<VtIntArray>() == result.instanceIndices[p];
    }
    Check(idxOk, "instanceIndices round-trip one VtIntArray per prototype");
    Check(!topo.GetMask(), "empty mask stays absent");

    HdContainerDataSourceHandle primvars = Child(c, "primvars");
    Check(primvars != nullptr, "instancer carries primvars");
    bool xforms = true;
    for (TfToken const &name : {HdInstancerTokens->instanceTranslations,
                                HdInstancerTokens->instanceRotations,
                                HdInstancerTokens->instanceScales}) {
        HdContainerDataSourceHandle pv =
            HdContainerDataSource::Cast(primvars->Get(name));
        xforms = xforms && pv &&
            LeafIsToken(LeafValue(pv->Get(TfToken("interpolation"))), "instance");
    }
    Check(xforms, "Translations/Rotations/Scales are instance-interpolated");
    VtValue tValue = LeafValue(
        Child(primvars, HdInstancerTokens->instanceTranslations)
            ->Get(TfToken("primvarValue")));
    Check(tValue.IsHolding<VtVec3fArray>() &&
          tValue.UncheckedGet<VtVec3fArray>() == result.translations,
          "instanceTranslations round-trip the baked roots");
    HdContainerDataSourceHandle colorPv =
        HdContainerDataSource::Cast(primvars->Get(TfToken("displayColor")));
    Check(colorPv &&
          LeafIsToken(LeafValue(colorPv->Get(TfToken("interpolation"))),
                      "instance"),
          "displayColor publishes at instance interpolation");

    // MoonRay wire format (docs/moonray-fur.md): USD maps PointInstancer
    // `orientations` (quath[]) straight through to hydra:instanceRotations
    // (dataSourcePointInstancer.cpp), and hdMoonray only accepts
    // VtQuathArray there -- a VtQuatfArray mis-syncs every instance.
    // Storm accepts both, so the data source publishes quath while Bake
    // keeps full float precision.
    VtValue rValue = LeafValue(
        Child(primvars, HdInstancerTokens->instanceRotations)
            ->Get(TfToken("primvarValue")));
    bool quathOk = rValue.IsHolding<VtQuathArray>() &&
        rValue.UncheckedGet<VtQuathArray>().size() == result.rotations.size();
    for (size_t i = 0; quathOk && i != result.rotations.size(); ++i) {
        GfQuath const q = rValue.UncheckedGet<VtQuathArray>()[i];
        GfQuatf const b = result.rotations[i];
        GfVec3h const qi = q.GetImaginary();
        GfVec3f const bi = b.GetImaginary();
        quathOk = Near(float(q.GetReal()), b.GetReal(), 2e-3f) &&
            Near(float(qi[0]), bi[0], 2e-3f) &&
            Near(float(qi[1]), bi[1], 2e-3f) &&
            Near(float(qi[2]), bi[2], 2e-3f);
    }
    Check(quathOk, "instanceRotations publish as VtQuathArray");
    // The inline quath conversion matches the GfQuath(GfQuatf) oracle bit
    // for bit.
    bool quathExact = rValue.IsHolding<VtQuathArray>() &&
        rValue.UncheckedGet<VtQuathArray>().size() == result.rotations.size();
    for (size_t i = 0; quathExact && i != result.rotations.size(); ++i) {
        quathExact = rValue.UncheckedGet<VtQuathArray>()[i] ==
            GfQuath(result.rotations[i]);
    }
    Check(quathExact, "instanceRotations match GfQuath(GfQuatf) exactly");
    // The vector conversion matches the scalar oracle bit for bit on
    // adversarial lanes: +-0, subnormal-producing magnitudes, float
    // subnormals, max-half, overflow (->inf), infinities and NaN payloads
    // (an exp-255 group redoes its lanes through the scalar spelling).
    // Runs twice: once clean (vector groups) and once with NaN/Inf lanes
    // (mixed vector/scalar-redo groups), plus once under FPCR.FZ to pin
    // FPCR-independence.
    {
        float const lanes[] = {
            0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.3333333f, 0.1f,
            1e-6f, -1e-6f, 6e-5f, 5.96e-8f, 3e-8f, 1e-9f, 1e-40f,
            65504.0f, -65504.0f, 65505.0f, 1e6f,
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
            -std::numeric_limits<float>::quiet_NaN(),
        };
        auto bitsOf = [](float v) {
            uint32_t b;
            std::memcpy(&b, &v, sizeof(b));
            return b;
        };
        auto floatOf = [](uint32_t b) {
            float v;
            std::memcpy(&v, &b, sizeof(v));
            return v;
        };
        uint32_t const nanPayloads[] = {
            0x7fc00001u, 0x7f800001u, 0x7fffffffu, 0xffc12345u, 0x7fbfffffu,
        };
        for (int pass = 0; pass < 3; ++pass) {
            UsdGenInstanceResult adv = result;
            size_t const nq = 64;
            adv.rotations.resize(nq);
            adv.translations.resize(nq);
            adv.scales.resize(nq);
            for (size_t q = 0; q < nq; ++q) {
                float c[4];
                for (int k = 0; k < 4; ++k) {
                    size_t const pick = (q * 4 + size_t(k)) %
                        (sizeof(lanes) / sizeof(lanes[0]));
                    c[k] = lanes[pick];
                }
                // Every eighth quat carries raw NaN payloads instead.
                if (pass == 1 && (q % 8) == 7) {
                    for (int k = 0; k < 4; ++k)
                        c[k] = floatOf(nanPayloads[(q + size_t(k)) % 5]);
                }
                adv.rotations[q] = GfQuatf(c[0], c[1], c[2], c[3]);
            }
            // The clean pass must stay vector-routed: drop the NaN/Inf
            // lanes the table above mixes in.
            if (pass != 1) {
                for (size_t q = 0; q < nq; ++q) {
                    GfQuatf qq = adv.rotations[q];
                    float cc[4] = {qq.GetReal(), qq.GetImaginary()[0],
                                   qq.GetImaginary()[1], qq.GetImaginary()[2]};
                    for (int k = 0; k < 4; ++k) {
                        uint32_t const b = bitsOf(cc[k]);
                        if (((b >> 23) & 0xff) >= 143)
                            cc[k] = 0.25f;
                    }
                    adv.rotations[q] = GfQuatf(cc[0], cc[1], cc[2], cc[3]);
                }
            }
#if defined(__aarch64__)
            uint64_t savedFpcr = 0, fzFpcr = 0;
            if (pass == 2) {
                __asm__ volatile("mrs %0, fpcr" : "=r"(savedFpcr));
                fzFpcr = savedFpcr | uint64_t(0x03000000);
                __asm__ volatile("msr fpcr, %0" :: "r"(fzFpcr));
            }
#endif
            HdContainerDataSourceHandle ac =
                UsdGenInstancer::BuildInstancerDataSource(adv,
                                                          SdfPath("/groom"));
#if defined(__aarch64__)
            if (pass == 2)
                __asm__ volatile("msr fpcr, %0" :: "r"(savedFpcr));
#endif
            bool advOk = ac != nullptr;
            VtValue av;
            if (advOk) {
                HdContainerDataSourceHandle apv = Child(ac, "primvars");
                HdContainerDataSourceHandle arot = apv
                    ? HdContainerDataSource::Cast(
                          apv->Get(HdInstancerTokens->instanceRotations))
                    : nullptr;
                av = arot ? LeafValue(arot->Get(TfToken("primvarValue")))
                          : VtValue();
                advOk = av.IsHolding<VtQuathArray>() &&
                    av.UncheckedGet<VtQuathArray>().size() == nq;
            }
            for (size_t q = 0; advOk && q < nq; ++q) {
                GfQuath const got = av.UncheckedGet<VtQuathArray>()[q];
                GfQuatf const src = adv.rotations[q];
                GfQuath const want = GfQuath(
                    GfHalf(src.GetReal()), GfVec3h(src.GetImaginary()));
                // Bitwise compare (NaN lanes never compare ==).
                advOk = std::memcmp(&got, &want, sizeof(got)) == 0;
            }
            Check(advOk, pass == 0
                              ? "quath vector path matches scalar bit for bit"
                              : pass == 1
                              ? "quath scalar fallback matches bit for bit"
                              : "quath matches under FPCR.FZ");
        }
    }

    // hdMoonray ignores primvar elementSize, so float varyings publish
    // packed (arity 3 -> VtVec3fArray) instead of flat float arrays; the
    // color role matches USD's own color3f primvar convention.
    VtValue colorValue = LeafValue(colorPv->Get(TfToken("primvarValue")));
    Check(colorValue.IsHolding<VtVec3fArray>() &&
          colorValue.UncheckedGet<VtVec3fArray>() == input.displayColor,
          "displayColor publishes packed VtVec3fArray");
    Check(LeafIsToken(LeafValue(colorPv->Get(TfToken("role"))), "color"),
          "displayColor carries the color role");
    Check(colorPv->Get(TfToken("elementSize")) == nullptr,
          "packed varyings carry no elementSize");

    // Packing across arities and types: float2 packs, scalar int rides
    // through flat (MoonRay reads scalar ints; only Storm honors
    // elementSize on the rest).
    UsdGenCurveBuffer packed = StraightStrands(roots, 4, 0.25f);
    UsdGenPlane pair;
    pair.name = TfToken("usdGen:pair");
    pair.interpolation = TfToken("uniform");
    pair.type = TfToken("float");
    pair.arity = 2;
    pair.f = VtFloatArray{0.0f, 1.0f, 2.0f, 3.0f};
    packed.extraCurve.push_back(pair);
    UsdGenPlane zone;
    zone.name = TfToken("usdGen:zone");
    zone.interpolation = TfToken("uniform");
    zone.type = TfToken("int");
    zone.arity = 1;
    zone.i = VtIntArray{3, 4};
    packed.extraCurve.push_back(zone);
    UsdGenInstanceCurves packedInput;
    packedInput.curves = &packed;
    packedInput.displayColor = input.displayColor;
    UsdGenInstanceParams packedParams = CardsParams();
    packedParams.variationPrimvars = VtArray<TfToken>{
        TfToken("displayColor"), TfToken("usdGen:pair"), TfToken("usdGen:zone")};
    UsdGenInstanceResult packedResult;
    Check(BakeOk(packedParams, packedInput, &packedResult),
          "packing bake cooks");
    HdContainerDataSourceHandle pc = UsdGenInstancer::BuildInstancerDataSource(
        packedResult, SdfPath("/groom"));
    HdContainerDataSourceHandle ppv = Child(pc, "primvars");
    HdContainerDataSourceHandle pairPv =
        ppv ? HdContainerDataSource::Cast(ppv->Get(TfToken("usdGen:pair")))
            : nullptr;
    bool pairOk = false;
    if (pairPv) {
        VtValue pairValue = LeafValue(pairPv->Get(TfToken("primvarValue")));
        pairOk = pairValue.IsHolding<VtVec2fArray>() &&
            pairValue.UncheckedGet<VtVec2fArray>().size() == 2 &&
            pairValue.UncheckedGet<VtVec2fArray>()[1] == GfVec2f(2.0f, 3.0f);
    }
    Check(pairOk, "float2 varyings publish packed VtVec2fArray");
    Check(pairPv && pairPv->Get(TfToken("role")) == nullptr,
          "non-color varyings carry no role");
    HdContainerDataSourceHandle zonePv =
        ppv ? HdContainerDataSource::Cast(ppv->Get(TfToken("usdGen:zone")))
            : nullptr;
    bool zoneDsOk = false;
    if (zonePv) {
        VtValue zoneValue = LeafValue(zonePv->Get(TfToken("primvarValue")));
        zoneDsOk = zoneValue.IsHolding<VtIntArray>() &&
            zoneValue.UncheckedGet<VtIntArray>() == VtIntArray{3, 4};
    }
    Check(zoneDsOk, "int varyings publish VtIntArray");

    HdXformSchema xform = HdXformSchema::GetFromParent(c);
    VtValue matrix = xform.GetMatrix()->GetValue(0.0);
    Check(matrix.IsHolding<GfMatrix4d>() &&
          matrix.UncheckedGet<GfMatrix4d>() == GfMatrix4d(1.0),
          "instancer xform is identity");
    VtValue reset = xform.GetResetXformStack()->GetValue(0.0);
    Check(reset.IsHolding<bool>() && reset.UncheckedGet<bool>(),
          "instancer sets resetXformStack");
    VtValue origin = LeafValue(Child(c, "primOrigin")->Get(TfToken("scenePath")));
    Check(origin.IsHolding<SdfPath>() &&
          origin.UncheckedGet<SdfPath>() == SdfPath("/groom"),
          "primOrigin/scenePath carries the description");
    HdVisibilitySchema visibility = HdVisibilitySchema::GetFromParent(c);
    VtValue visible = visibility.GetVisibility()->GetValue(0.0);
    Check(visible.IsHolding<bool>() && visible.UncheckedGet<bool>(),
          "default visibility is visible");
    Check(LeafIsToken(LeafValue(Child(c, "purpose")->Get(TfToken("purpose"))),
                      "render"),
          "authored purpose passes through");

    // Unauthored purpose omits the container (tiles behave the same: an
    // absent container resolves to the geometry render tag).
    UsdGenInstanceResult bare = result;
    bare.purpose = TfToken();
    bare.visibility = TfToken("invisible");
    HdContainerDataSourceHandle c2 = UsdGenInstancer::BuildInstancerDataSource(
        bare, SdfPath());
    Check(c2->Get(TfToken("purpose")) == nullptr,
          "empty purpose omits the purpose container");
    VtValue hidden = HdVisibilitySchema::GetFromParent(c2)
                         .GetVisibility()
                         ->GetValue(0.0);
    Check(hidden.IsHolding<bool>() && !hidden.UncheckedGet<bool>(),
          "visibility=invisible writes false");
    Check(c2->Get(TfToken("primOrigin")) == nullptr,
          "empty primOrigin omits the container");

    HdContainerDataSourceHandle by = UsdGenInstancer::BuildInstancedByDataSource(
        SdfPath("/groom/__usdGenRender/inst_op"),
        SdfPath("/groom/__usdGenRender/inst_op/Prototypes/cardA"));
    HdInstancedBySchema bound(by);
    VtValue boundPaths = bound.GetPaths()->GetValue(0.0);
    Check(boundPaths.IsHolding<VtArray<SdfPath>>() &&
          boundPaths.UncheckedGet<VtArray<SdfPath>>().size() == 1 &&
          boundPaths.UncheckedGet<VtArray<SdfPath>>()[0] ==
              SdfPath("/groom/__usdGenRender/inst_op"),
          "instancedBy/paths holds exactly one path");
    VtValue boundRoots = bound.GetPrototypeRoots()->GetValue(0.0);
    Check(boundRoots.IsHolding<VtArray<SdfPath>>() &&
          boundRoots.UncheckedGet<VtArray<SdfPath>>()[0] ==
              SdfPath("/groom/__usdGenRender/inst_op/Prototypes/cardA"),
          "instancedBy/prototypeRoots names the re-rooted prototype");
}

// --- (11) notices -------------------------------------------------------------------------------------

static void CheckNotices()
{
    UsdGenInstancer::InstanceNotices topo =
        UsdGenInstancer::NoticesFor(true, false);
    std::vector<HdDataSourceLocator> locs = topo.all();
    Check(locs.size() == 1 &&
          locs[0] == HdDataSourceLocator(TfToken("instancerTopology")),
          "count/binding edits dirty instancerTopology");
    UsdGenInstancer::InstanceNotices value =
        UsdGenInstancer::NoticesFor(false, true);
    locs = value.all();
    Check(locs.size() == 1 &&
          locs[0] == HdDataSourceLocator(TfToken("primvars"),
                                         HdInstancerTokens->instanceTranslations,
                                         TfToken("primvarValue")),
          "transform edits dirty Translations, never the topology");
    UsdGenInstancer::InstanceNotices tint = UsdGenInstancer::NoticesFor(
        false, false, {TfToken("displayColor")});
    locs = tint.all();
    Check(locs.size() == 1 &&
          locs[0] == HdDataSourceLocator(TfToken("primvars"),
                                         TfToken("displayColor"),
                                         TfToken("primvarValue")),
          "varying edits dirty that primvarValue alone");
    Check(UsdGenInstancer::NoticesFor(false, false).all().empty(),
          "quiet bake emits no locators");
}

// --- (12) inline rotation quat -----------------------------------------
// Bake's inline frame-to-quat conversion must match
// GfMatrix4d::ExtractRotationQuat bit for bit: differential check over the
// 24 axis-aligned frames (derivation is exact there) plus seeded random
// orthonormal frames, comparing every baked rotation exactly. The set
// covers both quat branches (identity takes the trace path, the 180-degree
// frames take the diagonal path).

static uint64_t QuatTestState = 0x123456789abcdefull;
static uint64_t QuatTestNext()
{
    uint64_t z = (QuatTestState += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static double QuatTestUnit()
{
    return double(QuatTestNext() >> 11) * 0x1.0p-53;
}

static void CheckInlineQuat()
{
    std::vector<GfVec3f> tFrames, bFrames, nFrames;
    // All 24 axis-aligned orthonormal frames: y over the six signed axes,
    // z over the four orthogonal to y, x = y cross z.
    GfVec3d const axes[6] = {GfVec3d(1, 0, 0), GfVec3d(-1, 0, 0),
                             GfVec3d(0, 1, 0), GfVec3d(0, -1, 0),
                             GfVec3d(0, 0, 1), GfVec3d(0, 0, -1)};
    for (int yi = 0; yi < 6; ++yi)
        for (int zi = 0; zi < 6; ++zi) {
            if (zi / 2 == yi / 2) continue;  // parallel to y
            GfVec3d const y = axes[yi], z = axes[zi];
            GfVec3d const x = GfCross(y, z);
            tFrames.push_back(GfVec3f(x));
            bFrames.push_back(GfVec3f(y));
            nFrames.push_back(GfVec3f(z));
        }
    // Seeded random orthonormal frames via double Gram-Schmidt.
    for (int k = 0; k < 2000; ++k) {
        GfVec3d a(QuatTestUnit() * 2.0 - 1.0, QuatTestUnit() * 2.0 - 1.0,
                  QuatTestUnit() * 2.0 - 1.0);
        GfVec3d b(QuatTestUnit() * 2.0 - 1.0, QuatTestUnit() * 2.0 - 1.0,
                  QuatTestUnit() * 2.0 - 1.0);
        if (a.GetLength() < 0.2 || b.GetLength() < 0.2) {
            --k;
            continue;
        }
        GfVec3d const y = a / a.GetLength();
        GfVec3d zb = b - y * GfDot(b, y);
        if (zb.GetLength() < 0.2) {
            --k;
            continue;
        }
        GfVec3d const z = zb / zb.GetLength();
        GfVec3d const x = GfCross(y, z);
        tFrames.push_back(GfVec3f(x));
        bFrames.push_back(GfVec3f(y));
        nFrames.push_back(GfVec3f(z));
    }
    uint32_t const n = uint32_t(tFrames.size());
    UsdGenCurveBuffer curves;
    curves.totalCurves = n;
    curves.totalCvs = n;
    curves.px.assign(n, 0.0f);
    curves.py.assign(n, 0.0f);
    curves.pz.assign(n, 0.0f);
    curves.curveId.resize(n);
    for (uint32_t c = 0; c != n; ++c) curves.curveId[c] = uint64_t(c);
    curves.rootT.assign(n, GfVec3f(0, 0, 0));
    curves.rootB.assign(n, GfVec3f(0, 0, 0));
    curves.rootN.assign(n, GfVec3f(0, 0, 0));
    for (uint32_t c = 0; c != n; ++c) {
        curves.rootT[c] = tFrames[c];
        curves.rootB[c] = bFrames[c];
        curves.rootN[c] = nFrames[c];
    }
    UsdGenInstanceCurves input;
    input.curves = &curves;
    UsdGenInstanceResult result;
    Check(BakeOk(CardsParams(), input, &result),
          "quat differential bake succeeds");
    if (result.rotations.size() != n) {
        Check(false, "quat differential has one rotation per curve");
        return;
    }
    size_t exact = 0, elseBranch = 0, traceBranch = 0;
    for (uint32_t c = 0; c != n; ++c) {
        // Mirror of Bake's surfaceFrame derivation, then the USD routine.
        GfVec3d yAxis = GfVec3d(curves.rootB[c]) /
            GfVec3d(curves.rootB[c]).GetLength();
        GfVec3d z = GfVec3d(curves.rootN[c]) -
            yAxis * GfDot(GfVec3d(curves.rootN[c]), yAxis);
        GfVec3d zAxis = z / z.GetLength();
        GfVec3d xAxis = GfCross(yAxis, zAxis);
        GfMatrix4d basis(1.0);
        basis.SetRow(0, GfVec4d(xAxis[0], xAxis[1], xAxis[2], 0.0));
        basis.SetRow(1, GfVec4d(yAxis[0], yAxis[1], yAxis[2], 0.0));
        basis.SetRow(2, GfVec4d(zAxis[0], zAxis[1], zAxis[2], 0.0));
        GfQuatf const expect = GfQuatf(basis.ExtractRotationQuat());
        if (result.rotations[c] == expect) ++exact;
        // Same branch condition as the quat routine: trace vs the
        // largest diagonal.
        double const d[3] = {xAxis[0], yAxis[1], zAxis[2]};
        int bi = 0;
        if (d[0] > d[1]) bi = (d[0] > d[2] ? 0 : 2);
        else bi = (d[1] > d[2] ? 1 : 2);
        if (d[0] + d[1] + d[2] > d[bi]) ++traceBranch;
        else ++elseBranch;
    }
    Check(exact == n, "inline quat matches ExtractRotationQuat exactly");
    Check(traceBranch > 0 && elseBranch > 0,
          "quat differential covers both branches");
}

static void CheckThreadedRanges()
{
    // 40K strands sit far above the 32768-curve threading threshold, so
    // two bakes (thread scheduling varies run to run) must agree bitwise
    // on every threaded output: the four lanes plus the per-prototype
    // index partition. Cards+twist exercises the frame/trig path;
    // spheres exercises the early-continue path.
    std::vector<GfVec3f> roots;
    roots.reserve(40000);
    for (int i = 0; i != 40000; ++i)
        roots.push_back(GfVec3f(float(i % 200), float(i / 200),
                                float((i * 7) % 13)));
    UsdGenCurveBuffer curves = StraightStrands(roots, 4, 0.25f);
    UsdGenInstanceCurves input;
    input.curves = &curves;

    UsdGenInstanceParams cards = CardsParams();
    cards.twist = 15.0f;
    cards.twistRandom = 30.0f;
    UsdGenInstanceResult c1, c2;
    Check(BakeOk(cards, input, &c1) && BakeOk(cards, input, &c2),
          "threaded cards bakes repeatably");
    Check(c1.translations.size() == 40000 && SameBake(c1, c2),
          "threaded cards ranges are bitwise identical");

    UsdGenInstanceParams spheres = CardsParams();
    spheres.primitive = TfToken("spheres");
    UsdGenInstanceResult s1, s2;
    Check(BakeOk(spheres, input, &s1) && BakeOk(spheres, input, &s2),
          "threaded spheres bakes repeatably");
    Check(s1.translations.size() == 40000 && SameBake(s1, s2),
          "threaded spheres ranges are bitwise identical");

    // The same 40K curves drive the threaded per-CV gather (n > 32768)
    // and the chunked arity-2 pack (320KB > 256KB): the gathered varyings
    // must equal an independently sampled expectation bitwise, and the
    // published VtVec2fArray must equal the source floats bitwise, across
    // two bakes (thread scheduling varies run to run).
    UsdGenCurveBuffer pairCurves = curves;
    UsdGenPlane pair;
    pair.name = TfToken("usdGen:pair");
    pair.interpolation = TfToken("uniform");
    pair.type = TfToken("float");
    pair.arity = 2;
    pair.f.resize(80000);
    for (size_t i = 0; i != pair.f.size(); ++i)
        pair.f[i] = float(int(i % 1024) - 512) / 512.0f;
    pairCurves.extraCurve.push_back(pair);
    UsdGenInstanceCurves vertexInput;
    vertexInput.curves = &pairCurves;
    vertexInput.displayColor.resize(pairCurves.totalCvs);
    for (uint32_t i = 0; i != pairCurves.totalCvs; ++i)
        vertexInput.displayColor[i] =
            GfVec3f(float(i % 251) / 251.0f, float((i * 3) % 127) / 127.0f,
                    float((i * 7) % 89) / 89.0f);
    UsdGenInstanceParams vary = CardsParams();
    vary.twist = 15.0f;
    vary.twistRandom = 30.0f;
    vary.variationPrimvars = VtArray<TfToken>{TfToken("displayColor"),
                                              TfToken("usdGen:pair")};
    UsdGenInstanceResult v1, v2;
    Check(BakeOk(vary, vertexInput, &v1) &&
              BakeOk(vary, vertexInput, &v2),
          "threaded varyings bakes repeatably");
    // Independent expectation: uniform topology, so curve c's root CV is
    // CV 4c; the pair plane copies through verbatim.
    VtFloatArray wantColor;
    wantColor.resize(120000);
    for (uint32_t c = 0; c != 40000; ++c) {
        GfVec3f const rgb = vertexInput.displayColor[c * 4];
        wantColor[3 * c + 0] = rgb[0];
        wantColor[3 * c + 1] = rgb[1];
        wantColor[3 * c + 2] = rgb[2];
    }
    auto findPlane = [](UsdGenInstanceResult const &r,
                        TfToken const &name) -> UsdGenPlane const * {
        for (UsdGenPlane const &p : r.varyings)
            if (p.name == name)
                return &p;
        return nullptr;
    };
    bool gatherOk = false, pairBakeOk = false;
    if (UsdGenPlane const *dc = findPlane(v1, TfToken("displayColor")))
        gatherOk = dc->f == wantColor;
    if (UsdGenPlane const *pp = findPlane(v1, TfToken("usdGen:pair")))
        pairBakeOk = pp->f == pair.f;
    Check(gatherOk, "threaded per-CV gather matches root-CV sampling");
    Check(pairBakeOk, "threaded bake carries the arity-2 plane verbatim");
    bool varyRepeat = false;
    if (UsdGenPlane const *dc = findPlane(v2, TfToken("displayColor")))
        varyRepeat = dc->f == wantColor;
    if (UsdGenPlane const *pp = findPlane(v2, TfToken("usdGen:pair")))
        varyRepeat = varyRepeat && pp->f == pair.f;
    Check(varyRepeat && SameBake(v1, v2),
          "threaded varyings repeat bitwise across bakes");
    HdContainerDataSourceHandle vc = UsdGenInstancer::BuildInstancerDataSource(
        v1, SdfPath("/groom"));
    HdContainerDataSourceHandle vpv = Child(vc, "primvars");
    HdContainerDataSourceHandle vpair =
        vpv ? HdContainerDataSource::Cast(vpv->Get(TfToken("usdGen:pair")))
            : nullptr;
    bool packOk = false;
    if (vpair) {
        VtValue pairValue = LeafValue(vpair->Get(TfToken("primvarValue")));
        if (pairValue.IsHolding<VtVec2fArray>()) {
            VtVec2fArray const &a =
                pairValue.UncheckedGet<VtVec2fArray>();
            packOk = a.size() == 40000 &&
                std::memcmp(a.cdata(), pair.f.cdata(),
                            a.size() * sizeof(GfVec2f)) == 0;
        }
    }
    Check(packOk, "chunked arity-2 pack matches the source floats bitwise");
}

int main()
{
    CheckPaths();
    CheckFailClosed();
    CheckCountsAndPartition();
    CheckSurfaceFrame();
    CheckCurveTangent();
    CheckWeights();
    CheckTwistAndScale();
    CheckWidthKnots();
    CheckSpheres();
    CheckVariationPrimvars();
    CheckDataSources();
    CheckNotices();
    CheckInlineQuat();
    CheckThreadedRanges();
    std::printf("testUsdGenInstance: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
