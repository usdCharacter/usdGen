// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Sample the authored quad Catmull-Clark Shield with OpenSubdiv Far patches.
// This docs proof path is independent of Collide's Bfr evaluator and proxy.
#include "usdGenSurface/limitSurface.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/imageable.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

int main(int argc, char **argv) try {
    if (argc != 6) {
        throw std::runtime_error(
            "usage: usdGenCollideFarOracle scene.usd /Mesh output.bin rate time");
    }
    auto const stage = UsdStage::Open(argv[1]);
    if (!stage) throw std::runtime_error("cannot open scene");
    UsdGeomMesh const mesh(stage->GetPrimAtPath(SdfPath(argv[2])));
    if (!mesh) throw std::runtime_error("target is not a Mesh");
    int const rate = std::stoi(argv[4]);
    if (rate < 4 || rate > 128 || (rate & (rate - 1)))
        throw std::runtime_error("rate must be a power of two in [4,128]");
    double const time = std::stod(argv[5]);
    usdGen::UsdGenSurfaceDesc s;
    mesh.GetSubdivisionSchemeAttr().Get(&s.subdivisionScheme);
    mesh.GetFaceVertexCountsAttr().Get(&s.faceVertexCounts);
    mesh.GetFaceVertexIndicesAttr().Get(&s.faceVertexIndices);
    mesh.GetPointsAttr().Get(&s.restPoints, UsdTimeCode(time));
    mesh.GetHoleIndicesAttr().Get(&s.holeIndices);
    mesh.GetInterpolateBoundaryAttr().Get(&s.interpolateBoundary);
    mesh.GetCreaseIndicesAttr().Get(&s.creaseIndices);
    mesh.GetCreaseLengthsAttr().Get(&s.creaseLengths);
    mesh.GetCreaseSharpnessesAttr().Get(&s.creaseSharpnesses);
    mesh.GetCornerIndicesAttr().Get(&s.cornerIndices);
    mesh.GetCornerSharpnessesAttr().Get(&s.cornerSharpnesses);
    if (s.subdivisionScheme != TfToken("catmullClark") || !s.holeIndices.empty())
        throw std::runtime_error("hero Far oracle requires unholed Catmull-Clark");
    for (int count : s.faceVertexCounts)
        if (count != 4) throw std::runtime_error("hero Far oracle requires quads");
    usdGen::UsdGenLimitSurface surface;
    std::string error;
    if (!surface.Build(s, 6, &error))
        throw std::runtime_error("Far patch build failed: " + error);
    auto const transform = UsdGeomImageable(mesh).ComputeLocalToWorldTransform(
        UsdTimeCode(time));
    std::ofstream out(argv[3], std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output");
    char const magic[8] = {'F','A','R','C','C','0','0','1'};
    uint32_t const faces = uint32_t(s.faceVertexCounts.size());
    uint32_t const step = uint32_t(rate);
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<char const*>(&faces), sizeof(faces));
    out.write(reinterpret_cast<char const*>(&step), sizeof(step));
    for (uint32_t face = 0; face < faces; ++face) {
        for (int row = 0; row <= rate; ++row) {
            for (int column = 0; column <= rate; ++column) {
                GfVec3f p, du, dv;
                if (!surface.Evaluate(int(face), float(column) / rate,
                                  float(row) / rate, &p, &du, &dv))
                    throw std::runtime_error("Far patch evaluation failed");
                GfVec3d const world = transform.Transform(GfVec3d(p));
                double const xyz[3] = {world[0], world[1], world[2]};
                out.write(reinterpret_cast<char const*>(xyz), sizeof(xyz));
            }
        }
    }
    if (!out) throw std::runtime_error("incomplete Far sample output");
    return 0;
} catch (std::exception const& error) {
    std::fprintf(stderr, "Far oracle: %s\n", error.what());
    return 1;
}
