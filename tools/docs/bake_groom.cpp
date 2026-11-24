// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Bake the stage's usdGen operator graph to ordinary USD BasisCurves for
// offline renderers. The compiler and scheduler are the same CPU cook tested
// by testUsdGenGroomExamples; this tool never synthesizes strand geometry.
#include "usdGen/compiler.h"
#include "usdGen/curveBuffer.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGenImaging/usdGenGraphDescBuilderStage.h"

#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/basisCurves.h"
#include "pxr/usd/usdGeom/primvarsAPI.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/base/gf/vec2i.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/gf/vec4i.h"
#include "pxr/base/gf/vec4f.h"
#include "pxr/base/vt/types.h"

#include <cstdio>
#include <stdexcept>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

void ExportPlane(UsdGeomPrimvarsAPI const &api,
                 usdGen::UsdGenPlane const &plane)
{
    auto const write = [&](SdfValueTypeName const &type, auto const &values) {
        UsdGeomPrimvar const primvar = api.CreatePrimvar(
            plane.name, type, plane.interpolation);
        if (!primvar || !primvar.Set(values))
            throw std::runtime_error("cannot export cooked primvar '" +
                                     plane.name.GetString() + "'");
    };
    size_t const count = plane.type == TfToken("float")
        ? plane.f.size() / plane.arity : plane.i.size() / plane.arity;
    if (plane.type == TfToken("float")) {
        switch (plane.arity) {
        case 1: write(SdfValueTypeNames->FloatArray, plane.f); return;
        case 2: {
            VtVec2fArray values(count);
            for (size_t n = 0; n < count; ++n)
                values[n] = GfVec2f(plane.f[n * 2], plane.f[n * 2 + 1]);
            write(SdfValueTypeNames->Float2Array, values); return;
        }
        case 3: {
            VtVec3fArray values(count);
            for (size_t n = 0; n < count; ++n)
                values[n] = GfVec3f(plane.f[n * 3], plane.f[n * 3 + 1],
                                    plane.f[n * 3 + 2]);
            write(SdfValueTypeNames->Float3Array, values); return;
        }
        case 4: {
            VtVec4fArray values(count);
            for (size_t n = 0; n < count; ++n)
                values[n] = GfVec4f(plane.f[n * 4], plane.f[n * 4 + 1],
                                    plane.f[n * 4 + 2], plane.f[n * 4 + 3]);
            write(SdfValueTypeNames->Float4Array, values); return;
        }
        }
    } else if (plane.type == TfToken("int")) {
        switch (plane.arity) {
        case 1: write(SdfValueTypeNames->IntArray, plane.i); return;
        case 2: {
            VtVec2iArray values(count);
            for (size_t n = 0; n < count; ++n)
                values[n] = GfVec2i(plane.i[n * 2], plane.i[n * 2 + 1]);
            write(SdfValueTypeNames->Int2Array, values); return;
        }
        case 3: {
            VtVec3iArray values(count);
            for (size_t n = 0; n < count; ++n)
                values[n] = GfVec3i(plane.i[n * 3], plane.i[n * 3 + 1],
                                    plane.i[n * 3 + 2]);
            write(SdfValueTypeNames->Int3Array, values); return;
        }
        case 4: {
            VtVec4iArray values(count);
            for (size_t n = 0; n < count; ++n)
                values[n] = GfVec4i(plane.i[n * 4], plane.i[n * 4 + 1],
                                    plane.i[n * 4 + 2], plane.i[n * 4 + 3]);
            write(SdfValueTypeNames->Int4Array, values); return;
        }
        }
    }
    throw std::runtime_error("unsupported cooked primvar type/arity for '" +
                             plane.name.GetString() + "': " +
                             plane.type.GetString() + "/" +
                             std::to_string(plane.arity));
}

} // namespace

int main(int argc, char **argv) try {
    if (argc < 4 || argc > 6) {
        std::fprintf(stderr, "usage: usdGenBakeGroom input.usd description-path output.usd [material-path|-] [time-code]\n");
        return 2;
    }
    usdGen::usdGenRegisterM1Operators();
    UsdStageRefPtr stage = UsdStage::Open(argv[1], UsdStage::LoadAll);
    if (!stage) throw std::runtime_error("cannot open input stage");
    SdfPath const description(argv[2]);
    UsdPrim descPrim = stage->GetPrimAtPath(description);
    if (!descPrim || descPrim.GetTypeName() != TfToken("UsdGenDescription"))
        throw std::runtime_error("description path is not a UsdGenDescription");

    usdGenImaging::UsdGenGraphDescBuildOptions options;
    if (argc == 6) options.time = std::stod(argv[5]);
    usdGen::UsdGenGraphDesc const desc =
        usdGenImaging::BuildGraphDescFromStage(stage, description, options);
    for (auto const &warning : desc.validationErrors)
        std::fprintf(stderr, "validation: %s\n", warning.c_str());
    usdGen::UsdGenCompiler compiler;
    usdGen::UsdGenGraph graph;
    auto const compiled = compiler.Compile(desc, &graph);
    if (!compiled.ok) {
        for (auto const &error : compiled.errors) std::fprintf(stderr, "compile: %s\n", error.c_str());
        return 1;
    }
    usdGen::UsdGenScheduler scheduler(4);
    usdGen::UsdGenEvalContext context;
    // The graph descriptor carries the requested sample time, but the CPU
    // scheduler takes its evaluation clock from the context passed to Run.
    // Keep both clocks aligned for time-varying operators and expressions.
    context.time = options.time;
    auto const run = scheduler.Run(graph, context, 1);
    for (auto const &warning : run.diagnostics.warnings)
        std::fprintf(stderr, "cook warning: %s\n", warning.c_str());
    if (run.diagnostics.HasErrors()) {
        for (auto const &error : run.diagnostics.errors) std::fprintf(stderr, "cook: %s\n", error.c_str());
        return 1;
    }
    auto const &buf = graph.Output();
    if (!buf.totalCurves || !buf.totalCvs || buf.px.size() != buf.totalCvs ||
        buf.py.size() != buf.totalCvs || buf.pz.size() != buf.totalCvs)
        throw std::runtime_error("cook returned no complete curve set");
    std::string planeError;
    if (!usdGen::detail::_UsdGenValidateExtraPlanes(buf, &planeError))
        throw std::runtime_error("cook returned invalid named planes: " + planeError);

    VtVec3fArray points(buf.totalCvs);
    for (size_t i = 0; i < points.size(); ++i)
        points[i] = GfVec3f(buf.px[i], buf.py[i], buf.pz[i]);
    VtIntArray counts(buf.totalCurves);
    if (buf.cvOffsets.empty()) {
        if (buf.totalCvs % buf.totalCurves)
            throw std::runtime_error("nonintegral uniform curve topology");
        for (size_t i = 0; i < counts.size(); ++i)
            counts[i] = int(buf.totalCvs / buf.totalCurves);
    } else {
        if (buf.cvOffsets.size() != size_t(buf.totalCurves) + 1)
            throw std::runtime_error("invalid ragged curve topology");
        for (size_t i = 0; i < counts.size(); ++i)
            counts[i] = buf.cvOffsets[i + 1] - buf.cvOffsets[i];
    }
    for (int n : counts) if (n < 2) throw std::runtime_error("curve has fewer than two CVs");

    GfVec3f root(0.64f, 0.43f, 0.24f), tip(0.85f, 0.65f, 0.37f);
    descPrim.GetAttribute(TfToken("usdGen:look:rootColor")).Get(&root);
    descPrim.GetAttribute(TfToken("usdGen:look:tipColor")).Get(&tip);

    // Make the cooked geometry independently renderable. Deactivating the
    // description also removes its synthetic Hydra material and live tiles.
    descPrim.SetActive(false);
    SdfPath const bakedPath = description.GetParentPath().AppendChild(
        TfToken("Baked_" + description.GetName()));
    UsdGeomBasisCurves curves = UsdGeomBasisCurves::Define(stage, bakedPath);
    curves.GetPrim().CreateAttribute(TfToken("usdGen:bakedTime"),
        SdfValueTypeNames->Double, true).Set(options.time);
    curves.CreatePointsAttr().Set(points);
    curves.CreateCurveVertexCountsAttr().Set(counts);
    curves.CreateTypeAttr().Set(UsdGeomTokens->cubic);
    curves.CreateBasisAttr().Set(UsdGeomTokens->bspline);
    curves.CreateWrapAttr().Set(UsdGeomTokens->pinned);
    if (buf.width.size() == buf.totalCvs) {
        curves.CreateWidthsAttr().Set(buf.width);
        curves.SetWidthsInterpolation(UsdGeomTokens->vertex);
    } else if (buf.width.size() == 1) {
        curves.CreateWidthsAttr().Set(buf.width);
        curves.SetWidthsInterpolation(UsdGeomTokens->constant);
    } else {
        throw std::runtime_error("cook returned no valid width plane");
    }
    SdfPath materialPath;
    if (argc >= 5 && std::string(argv[4]) != "-") {
        materialPath = SdfPath(argv[4]);
        if (!stage->GetPrimAtPath(materialPath))
            throw std::runtime_error("material path does not exist");
    } else {
        // A portable fallback for stages whose look is described only by
        // usdGen:look attributes. The shader reads the baked CV colour.
        materialPath = SdfPath("/World/BakedFurMaterial");
        if (!stage->GetPrimAtPath(materialPath)) {
            UsdPrim material = stage->DefinePrim(materialPath, TfToken("Material"));
            SdfPath const shaderPath = materialPath.AppendChild(TfToken("Surface"));
            SdfPath const readerPath = materialPath.AppendChild(TfToken("Color"));
            UsdPrim surface = stage->DefinePrim(shaderPath, TfToken("Shader"));
            UsdPrim reader = stage->DefinePrim(readerPath, TfToken("Shader"));
            material.CreateAttribute(TfToken("outputs:surface"), SdfValueTypeNames->Token)
                .AddConnection(shaderPath.AppendProperty(TfToken("outputs:surface")));
            surface.CreateAttribute(TfToken("info:id"), SdfValueTypeNames->Token)
                .Set(TfToken("UsdPreviewSurface"));
            surface.CreateAttribute(TfToken("inputs:roughness"), SdfValueTypeNames->Float)
                .Set(0.68f);
            surface.CreateAttribute(TfToken("inputs:diffuseColor"), SdfValueTypeNames->Color3f)
                .AddConnection(readerPath.AppendProperty(TfToken("outputs:result")));
            surface.CreateAttribute(TfToken("outputs:surface"), SdfValueTypeNames->Token);
            reader.CreateAttribute(TfToken("info:id"), SdfValueTypeNames->Token)
                .Set(TfToken("UsdPrimvarReader_float3"));
            reader.CreateAttribute(TfToken("inputs:varname"), SdfValueTypeNames->Token)
                .Set(TfToken("displayColor"));
            reader.CreateAttribute(TfToken("inputs:fallback"), SdfValueTypeNames->Color3f)
                .Set(root);
            reader.CreateAttribute(TfToken("outputs:result"), SdfValueTypeNames->Color3f);
        }
    }
    curves.GetPrim().CreateRelationship(TfToken("material:binding")).SetTargets({materialPath});
    UsdGeomPrimvarsAPI const primvars(curves);
    // Preserve C3 identity and rest bindings alongside the styled geometry.
    // A later CurveSource/Wind cook must see the original stable IDs and rest
    // coordinates, not derive new IDs or treat the moving baked points as rest.
    if (buf.rest.size() == buf.totalCvs) {
        UsdGeomPrimvar const rest = primvars.CreatePrimvar(
            TfToken("rest"), SdfValueTypeNames->Point3fArray, UsdGeomTokens->vertex);
        if (!rest || !rest.Set(buf.rest))
            throw std::runtime_error("cannot export baked rest positions");
    }
    if (buf.curveId.size() == buf.totalCurves) {
        UsdGeomPrimvar const ids = primvars.CreatePrimvar(
            TfToken("usdGen:curveId"), SdfValueTypeNames->UInt64Array,
            UsdGeomTokens->uniform);
        if (!ids || !ids.Set(buf.curveId))
            throw std::runtime_error("cannot export baked curve IDs");
    }
    if (buf.rootPrim.size() == buf.totalCurves) {
        UsdGeomPrimvar const skinPrim = primvars.CreatePrimvar(
            TfToken("skinprim"), SdfValueTypeNames->IntArray, UsdGeomTokens->uniform);
        if (!skinPrim || !skinPrim.Set(buf.rootPrim))
            throw std::runtime_error("cannot export baked root face indices");
    }
    if (buf.rootUV.size() == buf.totalCurves) {
        UsdGeomPrimvar const skinUv = primvars.CreatePrimvar(
            TfToken("skinprimuv"), SdfValueTypeNames->TexCoord2fArray,
            UsdGeomTokens->uniform);
        if (!skinUv || !skinUv.Set(buf.rootUV))
            throw std::runtime_error("cannot export baked root UVs");
    }
    if (buf.rootT.size() == buf.totalCurves &&
        buf.rootB.size() == buf.totalCurves &&
        buf.rootN.size() == buf.totalCurves) {
        VtMatrix4dArray frames(buf.totalCurves);
        size_t first = 0;
        for (size_t curve = 0; curve < frames.size(); ++curve) {
            GfMatrix4d frame(1.0);
            frame.SetRow3(0, GfVec3d(buf.rootT[curve]));
            frame.SetRow3(1, GfVec3d(buf.rootB[curve]));
            frame.SetRow3(2, GfVec3d(buf.rootN[curve]));
            frame.SetRow3(3, GfVec3d(points[first]));
            frames[curve] = frame;
            first += static_cast<size_t>(counts[curve]);
        }
        UsdGeomPrimvar const rootFrame = primvars.CreatePrimvar(
            TfToken("usdGen:rootFrame"), SdfValueTypeNames->Matrix4dArray,
            UsdGeomTokens->uniform);
        if (!rootFrame || !rootFrame.Set(frames))
            throw std::runtime_error("cannot export baked root frames");
    }
    bool hasCookedDisplayColor = false;
    for (auto const &plane : buf.extraCurve) {
        ExportPlane(primvars, plane);
        hasCookedDisplayColor |= plane.name == TfToken("displayColor");
    }
    for (auto const &plane : buf.extraCv) {
        ExportPlane(primvars, plane);
        hasCookedDisplayColor |= plane.name == TfToken("displayColor");
    }
    // Renderer-neutral colour survives the procedural material handoff. A
    // two-colour root/tip ramp communicates strand shape in preview shaders.
    if (!hasCookedDisplayColor) {
        VtVec3fArray colors(buf.totalCvs);
        size_t first = 0;
        for (int n : counts) {
            for (int i = 0; i < n; ++i) {
                float const t = n > 1 ? float(i) / float(n - 1) : 0.f;
                colors[first + i] = root * (1.f - t) + tip * t;
            }
            first += n;
        }
        UsdGeomPrimvar const displayColor = primvars.CreatePrimvar(
            TfToken("displayColor"), SdfValueTypeNames->Color3fArray, UsdGeomTokens->vertex);
        if (!displayColor || !displayColor.Set(colors))
            throw std::runtime_error("cannot export baked displayColor");
    }
    if (!stage->Export(argv[3])) throw std::runtime_error("cannot export baked stage");
    std::printf("Baked %u real usdGen curves / %u CVs to %s\n",
                buf.totalCurves, buf.totalCvs, argv[3]);
    return 0;
} catch (std::exception const &e) {
    std::fprintf(stderr, "usdGenBakeGroom: %s\n", e.what());
    return 1;
}
