// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGen/compiler.h"
#include "usdGenUsd/lower.h"
#include "usdGen/curveLoader.h"
#include "usdGen/curveRootCapture.h"
#include "usdGen/op.h"
#include "usdGen/vulkan/executionPlan.h"

#include <cmath>
#include <cstdio>
#include <memory>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while (false)

namespace {

UsdGenGraphDesc Fixture(bool surface = true, TfToken rebind = TfToken("never"))
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/Groom");
    d.executionBackend = UsdGenExecutionBackend::Vulkan;
    d.defaultWidth = .125f;

    if (surface) {
        UsdGenSurfaceDesc scalp;
        scalp.path = SdfPath("/Scalp");
        scalp.restPoints = {{0,0,0}, {1,0,0}, {0,1,0}};
        scalp.points = scalp.restPoints;
        scalp.faceVertexCounts = {3};
        scalp.faceVertexIndices = {0,1,2};
        scalp.uv = {{0,0}, {1,0}, {0,1}};
        d.surfaces.push_back(scalp);
    }

    UsdGenCurveSetDesc hair;
    hair.path = SdfPath("/Groom/C3");
    hair.role = UsdGenLane::Chunked;
    hair.curveRole = TfToken("hair");
    hair.type = TfToken("cubic"); hair.basis = TfToken("bspline"); hair.wrap = TfToken("pinned");
    hair.curveVertexCounts = {3, 3, 3};
    hair.points = {{.2f,.2f,0}, {.2f,.2f,1}, {.2f,.2f,2},
                   {.7f,.1f,0}, {.7f,.1f,1}, {.7f,.1f,2},
                   {.1f,.7f,0}, {.1f,.7f,1}, {.1f,.7f,2}};
    hair.rest = hair.points;
    hair.widths = VtFloatArray(9, .02f);
    hair.curveId = {42, 7, 19}; // canonical output is 7, 19, 42
    hair.skinPrim = {0, 0, 0};
    hair.skinPrimUv = {{.7f,.1f}, {.1f,.7f}, {.2f,.2f}};
    d.curveSets.push_back(hair);

    UsdGenNodeDesc source;
    source.path = SdfPath("/Groom/Ops/source"); source.type = TfToken("UsdGenCurveSource");
    source.curves = {hair.path};
    if (surface) source.surfaces = {SdfPath("/Scalp")};
    source.params = {{TfToken("useRest"), VtValue(true), false},
                     {TfToken("idSource"), VtValue(TfToken("primvar")), false},
                     {TfToken("resampleTo"), VtValue(0), false},
                     {TfToken("rebind"), VtValue(rebind), false}};
    UsdGenNodeDesc width;
    width.path = SdfPath("/Groom/Ops/width"); width.type = TfToken("UsdGenWidth");
    width.inputs = {source.path}; width.params = {{TfToken("width"), VtValue(.5f), false}};
    d.nodes = {source, width}; d.terminal = width.path;
    return d;
}

UsdGenGraphDesc RaggedRootedFixture()
{
    auto d = Fixture();
    auto &hair = d.curveSets[0];
    hair.curveVertexCounts = {3, 2, 4};
    hair.points.clear(); hair.rest.clear();
    for (int c = 0; c != 3; ++c)
        for (int i = 0; i != hair.curveVertexCounts[c]; ++i) {
            hair.points.push_back(GfVec3f(float(10 * c + i), float(c), 1.0f + i));
            hair.rest.push_back(GfVec3f(float(100 + 10 * c + i), float(10 + c), 2.0f + i));
        }
    hair.widths = VtFloatArray(9, .02f);
    hair.curveId = {30, 10, 20}; // middle curve drops; survivors sort as old 2, old 0
    hair.skinPrim = {0, 99, 0};
    hair.skinPrimUv = {{.3f,.7f}, {.5f,.5f}, {.8f,.2f}};

    UsdGenAuthoredPlaneDesc pointF;
    pointF.name = TfToken("pointFloat"); pointF.type = UsdGenAuthoredPlaneType::Float32;
    pointF.domain = UsdGenAuthoredPlaneDomain::Point; pointF.arity = 2;
    for (int i = 0; i != 9; ++i) { pointF.floatValues.push_back(1000.f + i); pointF.floatValues.push_back(2000.f + i); }
    UsdGenAuthoredPlaneDesc pointI;
    pointI.name = TfToken("pointInt"); pointI.type = UsdGenAuthoredPlaneType::Int32;
    pointI.domain = UsdGenAuthoredPlaneDomain::Point; pointI.arity = 1;
    for (int i = 0; i != 9; ++i) pointI.intValues.push_back(3000 + i);
    UsdGenAuthoredPlaneDesc primF;
    primF.name = TfToken("primFloat"); primF.type = UsdGenAuthoredPlaneType::Float32;
    primF.domain = UsdGenAuthoredPlaneDomain::Primitive; primF.arity = 1;
    primF.floatValues = {30.f, 10.f, 20.f};
    UsdGenAuthoredPlaneDesc primI;
    primI.name = TfToken("primInt"); primI.type = UsdGenAuthoredPlaneType::Int32;
    primI.domain = UsdGenAuthoredPlaneDomain::Primitive; primI.arity = 2;
    primI.intValues = {301,302, 101,102, 201,202};
    UsdGenAuthoredPlaneDesc groomF;
    groomF.name = TfToken("groomFloat"); groomF.type = UsdGenAuthoredPlaneType::Float32;
    groomF.domain = UsdGenAuthoredPlaneDomain::Groom; groomF.arity = 1;
    groomF.floatValues = {7.5f};
    UsdGenAuthoredPlaneDesc groomI;
    groomI.name = TfToken("groomInt"); groomI.type = UsdGenAuthoredPlaneType::Int32;
    groomI.domain = UsdGenAuthoredPlaneDomain::Groom; groomI.arity = 2;
    groomI.intValues = {81, 82};
    hair.authoredPlanes = {pointF, pointI, primF, primI, groomF, groomI};
    return d;
}

bool Load(UsdGenGraphDesc const& d, UsdGenCurveBuffer* out, UsdGenDiagnostics* diag)
{
    usdGen::UsdGenGraphSpec const lowered = usdGenUsd::Lower(d);
    UsdGenParamView params{&lowered, &lowered.nodes[0]};
    UsdGenCaptureContext ctx; ctx.desc = &lowered; ctx.params = &params;
    return UsdGenCurveLoader::Load(ctx, UsdGenCurveSetInputs(lowered)[0], out, diag);
}

bool Capture(UsdGenCurveSetDesc const& c, UsdGenSurfaceDesc const* s, char const* policy,
             UsdGenCurveRootCaptureResult* out)
{
    std::string error;
    return UsdGenCaptureCurveRoots(usdGenUsd::Lower(c), (s ? usdGenUsd::Addr(UsdGenSurfaceInputView(usdGenUsd::Lower(*s))) : nullptr), TfToken(policy), out, &error);
}

bool Near(double a, double b, double tolerance = 2.0e-6)
{
    return std::abs(a - b) <= tolerance;
}

bool NearVec(GfVec3d const& value, GfVec3d const& expected,
             double tolerance = 2.0e-6)
{
    return Near(value[0], expected[0], tolerance) &&
        Near(value[1], expected[1], tolerance) &&
        Near(value[2], expected[2], tolerance);
}

GfVec3d FrameAxis(GfMatrix4d const& frame, int row)
{
    return {frame[row][0], frame[row][1], frame[row][2]};
}

} // namespace

int main()
{
    auto d = Fixture();
    UsdGenDiagnostics diagnostics;
    auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(d), &diagnostics);
    CHECK(handle && !diagnostics.HasErrors());
    auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
    CHECK(plan && plan->HasRootBindings() && plan->Source().key == usdGenUsd::KeyOf(SdfPath("/Groom/C3")));
    CHECK(plan->SourceControls().rebind == TfToken("never"));
    CHECK(plan->Descriptor().get() != static_cast<void const *>(&d) && VtArray<uint64_t>(plan->Source().curveId.begin(), plan->Source().curveId.end()) == VtArray<uint64_t>({42,7,19}));

    UsdGenCurveBuffer loaded;
    CHECK(Load(d, &loaded, &diagnostics));
    CHECK(loaded.totalCurves == 3 && loaded.totalCvs == 9);
    CHECK(loaded.curveId == VtArray<uint64_t>({7,19,42}));
    CHECK(loaded.rootPrim == VtIntArray({0,0,0}));
    CHECK(loaded.rootUV == VtVec2fArray({GfVec2f(.1f,.7f), GfVec2f(.2f,.2f), GfVec2f(.7f,.1f)}));
    CHECK(loaded.rootT.size() == 3 && loaded.rootB.size() == 3 && loaded.rootN.size() == 3);
    CHECK(loaded.hairT[0] == 0.f && loaded.hairT[2] == 1.f && loaded.width[0] == .02f);

    // Plan and loader own immutable snapshots, and loading never edits the input descriptor.
    auto before = d;
    d.curveSets[0].curveId[0] = 999;
    d.curveSets[0].points[0] = GfVec3f(9,9,9);
    CHECK(plan->Source().curveId[0] == 42 && plan->Source().points[0] == GfVec3f(.2f,.2f,0));
    CHECK(before.curveSets[0].curveId[0] == 42 && loaded.curveId[0] == 7);

    // Authored frames are copied exactly; malformed matrices and cardinalities fail closed.
    auto explicitFrames = Fixture();
    explicitFrames.curveSets[0].rootFrame = VtMatrix4dArray(3, GfMatrix4d(1.0));
    UsdGenCurveRootCaptureResult roots;
    CHECK(Capture(explicitFrames.curveSets[0], &explicitFrames.surfaces[0], "never", &roots));
    CHECK(roots.valid == VtArray<uint8_t>({1,1,1}) && roots.frames[0] == GfMatrix4d(1.0));
    auto malformed = explicitFrames;
    malformed.curveSets[0].rootFrame[1][0][0] = -1.0;
    CHECK(!Capture(malformed.curveSets[0], &malformed.surfaces[0], "never", &roots));
    auto shortBindings = explicitFrames;
    shortBindings.curveSets[0].skinPrimUv.pop_back();
    CHECK(!Capture(shortBindings.curveSets[0], &shortBindings.surfaces[0], "never", &roots));

    // never records unresolved roots as explicit drops; onError repairs only those roots;
    // always recomputes even an otherwise valid authored binding.
    auto invalid = Fixture();
    invalid.curveSets[0].skinPrim[1] = 99;
    CHECK(Capture(invalid.curveSets[0], &invalid.surfaces[0], "never", &roots));
    CHECK(roots.dropped == 1 && roots.valid == VtArray<uint8_t>({1,0,1}) && roots.rootPrim[1] == -1);
    CHECK(Capture(invalid.curveSets[0], &invalid.surfaces[0], "onError", &roots));
    CHECK(roots.rebound == 1 && roots.valid == VtArray<uint8_t>({1,1,1}) && roots.rootPrim[1] == 0);
    auto changed = invalid;
    changed.curveSets[0].skinPrimUv[0] = GfVec2f(.9f,.05f);
    CHECK(Capture(changed.curveSets[0], &changed.surfaces[0], "always", &roots));
    CHECK(roots.rebound == 3 && roots.rootUV == VtVec2fArray({GfVec2f(.2f,.2f),
          GfVec2f(.7f,.1f), GfVec2f(.1f,.7f)}));

    // Derived frames deliberately stay in CurveSource-local coordinates.
    // Equal, finite invertible affine transforms therefore cancel exactly:
    // translation, rotation, non-uniform scale, shear, and a reflection must
    // leave the plane's local X/Y/Z frame and UV origin unchanged.  Clear
    // rootFrame so this exercises construction rather than authored frames.
    for (int variant = 0; variant != 5; ++variant) {
        auto affine = Fixture();
        affine.curveSets[0].rootFrame.clear();
        GfMatrix4d matrix(1.0);
        if (variant == 0) { matrix[3][0] = 3.0; matrix[3][1] = -4.0; matrix[3][2] = 2.0; }
        if (variant == 1) { matrix[0][0] = 0.0; matrix[0][1] = 1.0;
                            matrix[1][0] = -1.0; matrix[1][1] = 0.0; }
        if (variant == 2) { matrix[0][0] = 2.0; matrix[1][1] = 3.0; matrix[2][2] = .5; }
        if (variant == 3) { matrix[0][1] = .25; matrix[2][0] = -.5; }
        if (variant == 4) matrix[0][0] = -1.0;
        affine.curveSets[0].worldMatrix = matrix;
        affine.surfaces[0].worldMatrix = matrix;
        auto const sourcePoints = affine.curveSets[0].points;
        auto const sourceRest = affine.curveSets[0].rest;
        CHECK(Capture(affine.curveSets[0], &affine.surfaces[0], "never", &roots));
        CHECK(roots.valid == VtArray<uint8_t>({1,1,1}));
        for (size_t i = 0; i != roots.frames.size(); ++i) {
            CHECK(NearVec(FrameAxis(roots.frames[i], 0), {1,0,0}) &&
                  NearVec(FrameAxis(roots.frames[i], 1), {0,1,0}) &&
                  NearVec(FrameAxis(roots.frames[i], 2), {0,0,1}));
        }
        // The capture result is source order: the independent barycentric
        // origins are (u,v,0), not world-baked coordinates.
        CHECK(NearVec(FrameAxis(roots.frames[0], 3), {.7,.1,0}) &&
              NearVec(FrameAxis(roots.frames[1], 3), {.1,.7,0}) &&
              NearVec(FrameAxis(roots.frames[2], 3), {.2,.2,0}));
        UsdGenCurveBuffer affineOut;
        CHECK(Load(affine, &affineOut, &diagnostics));
        CHECK(affineOut.rootT == VtVec3fArray(3, GfVec3f(1,0,0)) &&
              affineOut.rootB == VtVec3fArray(3, GfVec3f(0,1,0)) &&
              affineOut.rootN == VtVec3fArray(3, GfVec3f(0,0,1)));
        // The shared loader transports both source channels verbatim; object
        // transforms affect only root-frame conversion, never C3 points/rest.
        CHECK(affine.curveSets[0].points == sourcePoints &&
              affine.curveSets[0].rest == sourceRest &&
              affineOut.px[0] == .7f && affineOut.py[0] == .1f &&
              affineOut.pz[0] == 0.f && affineOut.rest[0] == GfVec3f(.7f,.1f,0));
    }

    // With unequal transforms, calculate the expected row-vector relative
    // frame directly.  The surface is rotated +90 degrees around Z then
    // translated; in source space X becomes +Y, its normal remains +Z, and
    // B = N x T is -X.  `never` retains valid authored bindings.
    auto relative = Fixture();
    relative.curveSets[0].rootFrame.clear();
    relative.surfaces[0].worldMatrix[0][0] = 0.0;
    relative.surfaces[0].worldMatrix[0][1] = 1.0;
    relative.surfaces[0].worldMatrix[1][0] = -1.0;
    relative.surfaces[0].worldMatrix[1][1] = 0.0;
    relative.surfaces[0].worldMatrix[3][0] = 5.0;
    relative.surfaces[0].worldMatrix[3][1] = -2.0;
    relative.surfaces[0].worldMatrix[3][2] = 7.0;
    CHECK(Capture(relative.curveSets[0], &relative.surfaces[0], "never", &roots));
    CHECK(NearVec(FrameAxis(roots.frames[0], 0), {0,1,0}) &&
          NearVec(FrameAxis(roots.frames[0], 1), {-1,0,0}) &&
          NearVec(FrameAxis(roots.frames[0], 2), {0,0,1}) &&
          NearVec(FrameAxis(roots.frames[0], 3), {4.9,-1.3,7}));
    UsdGenCurveBuffer relativeOut;
    CHECK(Load(relative, &relativeOut, &diagnostics));
    // Loader output is canonical id order: old curve 1 is output curve 0.
    CHECK(Near(relativeOut.rootT[0][0], 0) && Near(relativeOut.rootT[0][1], 1) &&
          Near(relativeOut.rootB[0][0], -1) && Near(relativeOut.rootN[0][2], 1) &&
          relativeOut.px[0] == .7f && relativeOut.py[0] == .1f);

    // onError rebinding uses the source worldMatrix to find the surface.  Put
    // old curve 1's root at the independently known surface-world position
    // for local (.2,.2,0), then invalidate only that authored binding.
    auto repaired = relative;
    repaired.curveSets[0].skinPrim[1] = 99;
    repaired.curveSets[0].points[3] = GfVec3f(4.8f,-1.8f,7.0f);
    repaired.curveSets[0].rest[3] = repaired.curveSets[0].points[3];
    CHECK(Capture(repaired.curveSets[0], &repaired.surfaces[0], "onError", &roots));
    CHECK(roots.rebound == 1 && roots.valid == VtArray<uint8_t>({1,1,1}) &&
          roots.rootPrim[1] == 0 && Near(roots.rootUV[1][0], .2f) &&
          Near(roots.rootUV[1][1], .2f) &&
          NearVec(FrameAxis(roots.frames[1], 0), {0,1,0}) &&
          NearVec(FrameAxis(roots.frames[1], 1), {-1,0,0}) &&
          NearVec(FrameAxis(roots.frames[1], 2), {0,0,1}) &&
          NearVec(FrameAxis(roots.frames[1], 3), {4.8,-1.8,7}));

    // Surface-free legacy path preserves the optional root channels and still loads/compiles.
    auto empty = Fixture(false);
    empty.curveSets[0].skinPrim.clear(); empty.curveSets[0].skinPrimUv.clear();
    UsdGenCurveBuffer emptyOut;
    CHECK(Load(empty, &emptyOut, &diagnostics));
    CHECK(emptyOut.totalCurves == 3 && emptyOut.rootPrim.empty() && emptyOut.rootUV.empty() &&
          emptyOut.rootT.empty() && emptyOut.hairT.size() == 9);
    CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(empty)));

    // Independent rooted-ragged source oracle: invalid middle root is dropped
    // before canonical stable-ID sorting (old curve 2/id20, then old 0/id30).
    auto ragged = RaggedRootedFixture();
    auto raggedBefore = ragged;
    CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(ragged)));
    UsdGenCurveBuffer raggedOut;
    CHECK(Load(ragged, &raggedOut, &diagnostics));
    CHECK(raggedOut.totalCurves == 2 && raggedOut.totalCvs == 7 &&
          raggedOut.cvOffsets == VtIntArray({0,4,7}) &&
          raggedOut.curveId == VtArray<uint64_t>({20,30}));
    CHECK(raggedOut.px == VtFloatArray({20,21,22,23,0,1,2}) &&
          raggedOut.py == VtFloatArray({2,2,2,2,0,0,0}) &&
          raggedOut.pz == VtFloatArray({1,2,3,4,1,2,3}));
    CHECK(raggedOut.rest == VtVec3fArray({GfVec3f(120,12,2), GfVec3f(121,12,3),
          GfVec3f(122,12,4), GfVec3f(123,12,5), GfVec3f(100,10,2),
          GfVec3f(101,10,3), GfVec3f(102,10,4)}) &&
          raggedOut.hairT == VtFloatArray({0, 1.f/3.f, 2.f/3.f, 1, 0, .5f, 1}));
    CHECK(raggedOut.rootPrim == VtIntArray({0,0}) &&
          raggedOut.rootUV == VtVec2fArray({GfVec2f(.8f,.2f), GfVec2f(.3f,.7f)}));
    CHECK(raggedOut.extraCv.size() == 2 && raggedOut.extraCurve.size() == 4);
    auto const &pf = raggedOut.extraCv[0].name == TfToken("pointFloat") ? raggedOut.extraCv[0] : raggedOut.extraCv[1];
    auto const &pi = raggedOut.extraCv[0].name == TfToken("pointInt") ? raggedOut.extraCv[0] : raggedOut.extraCv[1];
    CHECK(pf.name == TfToken("pointFloat") && pf.type == TfToken("float") && pf.arity == 2 &&
          pf.f == VtFloatArray({1005,2005,1006,2006,1007,2007,1008,2008,1000,2000,1001,2001,1002,2002}) &&
          pi.name == TfToken("pointInt") && pi.i == VtIntArray({3005,3006,3007,3008,3000,3001,3002}));
    auto const curvePlane = [&](TfToken name) -> UsdGenPlane const & {
        for (auto const &plane : raggedOut.extraCurve) if (plane.name == name) return plane;
        return raggedOut.extraCurve.front(); // unreachable for this validated fixture
    };
    auto const &prf = curvePlane(TfToken("primFloat"));
    auto const &pri = curvePlane(TfToken("primInt"));
    auto const &grf = curvePlane(TfToken("groomFloat"));
    auto const &gri = curvePlane(TfToken("groomInt"));
    CHECK(prf.f == VtFloatArray({20,30}) && pri.i == VtIntArray({201,202,301,302}) &&
          grf.f == VtFloatArray({7.5f}) && gri.i == VtIntArray({81,82}));
    ragged.curveSets[0].authoredPlanes.back().intValues[0] = 999;
    CHECK(raggedBefore.curveSets[0].authoredPlanes.back().intValues[0] == 81 &&
          raggedOut.extraCurve[1].i == VtIntArray({81,82}));

    // All malformed roots drop cleanly while constant groom planes remain.
    auto allDropped = RaggedRootedFixture();
    allDropped.curveSets[0].skinPrim = VtIntArray({99,98,97});
    UsdGenCurveBuffer droppedOut;
    auto const groomIndex = [](std::vector<UsdGenPlane> const &planes, TfToken name) {
        for (size_t i = 0; i != planes.size(); ++i) if (planes[i].name == name) return i;
        return planes.size();
    };
    CHECK(Load(allDropped, &droppedOut, &diagnostics) && droppedOut.totalCurves == 0 &&
          droppedOut.totalCvs == 0 &&
          (droppedOut.cvOffsets.empty() || droppedOut.cvOffsets == VtIntArray({0})) &&
          droppedOut.extraCurve.size() == 4 &&
          groomIndex(droppedOut.extraCurve, TfToken("groomFloat")) < droppedOut.extraCurve.size() &&
          droppedOut.extraCurve[groomIndex(droppedOut.extraCurve, TfToken("groomFloat"))].f == VtFloatArray({7.5f}));
    std::puts("Vulkan source capture contract tests passed");
    return 0;
}
