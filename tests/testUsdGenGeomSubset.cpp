// testUsdGenGeomSubset — a GeomSubset wherever usdGen needs a mesh
// (02-schema.md §2.20, ADR R15), end to end from an authored stage.
//
//   * usdGen:surface on a face GeomSubset: both builders produce one
//     self-contained surface desc (the parent Mesh's geometry, so face ids
//     stay parent ids, plus sorted, unique parent faces) and no stub;
//   * Scatter roots only on the subset's faces, with exactly the roots the
//     whole mesh gives those faces (widening a subset re-rolls nothing);
//   * subsets of one mesh union (rule 4); the mesh itself in the union
//     selects every face; an empty subset scatters nothing;
//   * a non-face subset, or an index outside the parent, fails closed;
//   * usdGen:paint:surface and an expression input:<name> accept a subset:
//     paint reads the parent's primvar with the outside faces at
//     usdGen:map:default, and geoSampler() visits only the subset's faces;
//   * the Hydra builder agrees with the stage builder on all of it.
//
// Requires the plugin path at runtime (adapters publish the Hydra values):
// run under ctest, which supplies it.

#include "usdGenImaging/usdGenGraphDescBuilder.h"
#include "usdGenImaging/usdGenGraphDescBuilderStage.h"
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"

#include "pxr/pxr.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;
using namespace usdGenImaging;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what.c_str()); }
}

// A 3 x 3 grid of unit quads in the XY plane; face f = 3 * row + column.
// primvars:paint gives every corner of face f the value (f + 1) / 10, inside
// usdGen:map:clamp's (0, 1) fallback.
std::string Stage()
{
    std::string points, indices, counts, paint;
    for (int y = 0; y <= 3; ++y)
        for (int x = 0; x <= 3; ++x)
            points += (points.empty() ? "" : ", ") + std::string("(") +
                      std::to_string(x) + ", " + std::to_string(y) + ", 0)";
    for (int f = 0; f < 9; ++f) {
        int const x = f % 3, y = f / 3;
        int const corners[4] = {y * 4 + x, y * 4 + x + 1, (y + 1) * 4 + x + 1, (y + 1) * 4 + x};
        counts += (f ? ", " : "") + std::string("4");
        for (int c : corners) {
            indices += (indices.empty() ? "" : ", ") + std::to_string(c);
            paint += (paint.empty() ? "" : ", ") + std::to_string(0.1 * (f + 1));
        }
    }
    auto description = [](char const *name, char const *surface, char const *extra = "") {
        return std::string("        def UsdGenDescription \"") + name + "\" {\n"
               "            rel usdGen:surface = " + surface + "\n"
               "            def Scope \"Ops\" {\n" + extra +
               "                def UsdGenScatter \"scatter\" {\n"
               "                    int usdGen:seed = 5\n"
               "                    float usdGen:density = 40\n"
               "                }\n"
               "            }\n"
               "        }\n";
    };
    // Width reads an expression per strand (ptex() reads at the root) or per
    // CV, replacing the literal.
    auto width = [](char const *expression, char const *evaluation) {
        return std::string(
               "                def UsdGenWidth \"width\" {\n"
               "                    float usdGen:width = 1 (\n"
               "                        customData = { dictionary usdGen = { string evaluation = \"") +
               evaluation + "\" } }\n"
               "                    )\n"
               "                    float usdGen:width.connect = <" + expression + ".outputs:result>\n"
               "                    bool usdGen:replace = true\n"
               "                }\n"
               "                def UsdGenGrow \"grow\" {\n"
               "                    int usdGen:segments = 3\n"
               "                    float usdGen:length = 0.5\n"
               "                }\n";
    };
    std::string s = "#usda 1.0\n"
        "def Xform \"World\" {\n"
        "    def Mesh \"Skin\" (prepend apiSchemas = [\"UsdGenRestAPI\"]) {\n"
        "        point3f[] points = [" + points + "]\n"
        "        int[] faceVertexCounts = [" + counts + "]\n"
        "        int[] faceVertexIndices = [" + indices + "]\n"
        "        float[] primvars:paint = [" + paint + "] (interpolation = \"faceVarying\")\n"
        "        def GeomSubset \"crown\" {\n"
        "            uniform token elementType = \"face\"\n"
        "            uniform token familyName = \"usdGenSurfaces\"\n"
        "            int[] indices = [4, 1, 1]\n"
        "        }\n"
        "        def GeomSubset \"temple\" {\n"
        "            uniform token elementType = \"face\"\n"
        "            int[] indices = [7]\n"
        "        }\n"
        "        def GeomSubset \"bald\" {\n"
        "            uniform token elementType = \"face\"\n"
        "            int[] indices = []\n"
        "        }\n"
        "        def GeomSubset \"verts\" {\n"
        "            uniform token elementType = \"point\"\n"
        "            int[] indices = [0]\n"
        "        }\n"
        "        def GeomSubset \"wild\" {\n"
        "            uniform token elementType = \"face\"\n"
        "            int[] indices = [2, 99]\n"
        "        }\n"
        "    }\n"
        "    def Xform \"Grooms\" {\n";
    s += description("Whole", "</World/Skin>");
    s += description("Crown", "</World/Skin/crown>");
    s += description("Union", "[</World/Skin/crown>, </World/Skin/temple>]");
    s += description("WithMesh", "[</World/Skin/crown>, </World/Skin>]");
    s += description("Bald", "</World/Skin/bald>");
    s += description("Verts", "</World/Skin/verts>");
    s += description("Wild", "</World/Skin/wild>");
    // Paint on a subset: faces 1 and 4 keep their paint, every other face
    // reads usdGen:map:default = 0.25.
    std::string painted = description("Painted", "</World/Skin/crown>",
        width("/World/Grooms/Painted/Expressions/paintWidth", "primitive").c_str());
    painted.insert(painted.rfind("        }\n"),
        "            def Scope \"Maps\" {\n"
        "                def UsdGenPaintMap \"paint\" {\n"
        "                    rel usdGen:paint:surface = </World/Skin/crown>\n"
        "                    token usdGen:paint:primvar = \"paint\"\n"
        "                    float usdGen:map:default = 0.25\n"
        "                }\n"
        "            }\n"
        "            def Scope \"Expressions\" {\n"
        "                def UsdGenExpression \"paintWidth\" {\n"
        "                    string usdGen:expr:source = \"ptex(\\\"paint\\\")\"\n"
        "                    rel input:paint = </World/Grooms/Painted/Maps/paint>\n"
        "                    custom float outputs:result\n"
        "                }\n"
        "            }\n");
    s += painted;
    // geoSampler over the temple subset: the nearest face is always 7.
    std::string sampled = description("Sampled", "</World/Skin>",
        width("/World/Grooms/Sampled/Expressions/templeFace", "point").c_str());
    sampled.insert(sampled.rfind("        }\n"),
        "            def Scope \"Expressions\" {\n"
        "                def UsdGenExpression \"templeFace\" {\n"
        "                    string usdGen:expr:source = \"geoSampler(\\\"skin\\\", \\\"$primIndex\\\")\"\n"
        "                    rel input:skin = </World/Skin/temple>\n"
        "                    custom float outputs:result\n"
        "                }\n"
        "            }\n");
    s += sampled;
    s += "    }\n}\n";
    return s;
}

struct Built {
    UsdGenGraphDesc stage, hydra;
};

Built Build(UsdStageRefPtr const &stage, HdSceneIndexBaseRefPtr const &index,
            char const *name)
{
    SdfPath const path = SdfPath("/World/Grooms").AppendChild(TfToken(name));
    UsdGenGraphDescBuildOptions options;
    options.time = 0.0;
    return {BuildGraphDescFromStage(stage, path, options),
            BuildGraphDescFromHydra(*index, path, options)};
}

bool Cook(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *out, std::string *error,
          std::vector<std::string> *warnings = nullptr)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    auto const compiled = compiler.Compile(desc, &graph);
    if (warnings) *warnings = compiled.warnings;
    if (!compiled.ok) {
        *error = compiled.errors.empty() ? "compile failed" : compiled.errors.front();
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    auto const run = scheduler.Run(graph, context, 1);
    if (run.diagnostics.HasErrors()) {
        *error = run.diagnostics.errors.empty() ? "run failed" : run.diagnostics.errors.front();
        return false;
    }
    if (warnings)
        warnings->insert(warnings->end(), run.diagnostics.warnings.begin(),
                         run.diagnostics.warnings.end());
    *out = graph.Output();
    return true;
}

bool Mentions(std::vector<std::string> const &errors, char const *text)
{
    for (auto const &e : errors)
        if (e.find(text) != std::string::npos) return true;
    return false;
}

// The builders must agree on every subset field, and on failing closed.
void CheckParity(Built const &b, std::string const &name)
{
    Check(b.stage.surfaces.size() == b.hydra.surfaces.size(), name + ": surface count parity");
    for (size_t i = 0; i < std::min(b.stage.surfaces.size(), b.hydra.surfaces.size()); ++i) {
        auto const &s = b.stage.surfaces[i];
        auto const &h = b.hydra.surfaces[i];
        std::string const ctx = name + " surface " + s.path.GetString();
        Check(s.path == h.path, ctx + ": path parity");
        Check(s.isSubset == h.isSubset, ctx + ": isSubset parity");
        Check(s.subsetFaces == h.subsetFaces, ctx + ": subsetFaces parity");
        Check(s.faceVertexCounts == h.faceVertexCounts && s.faceVertexIndices == h.faceVertexIndices,
              ctx + ": topology parity");
        Check(s.restPoints == h.restPoints, ctx + ": rest parity");
        Check(s.worldMatrix == h.worldMatrix, ctx + ": matrix parity");
    }
    Check(b.stage.geometries.size() == b.hydra.geometries.size(), name + ": geometry count parity");
    for (size_t i = 0; i < std::min(b.stage.geometries.size(), b.hydra.geometries.size()); ++i) {
        auto const &s = b.stage.geometries[i];
        auto const &h = b.hydra.geometries[i];
        Check(s.path == h.path && s.isSubset == h.isSubset && s.subsetFaces == h.subsetFaces &&
                  s.counts == h.counts && s.generation == h.generation,
              name + ": geometry " + s.path.GetString() + " parity");
    }
    Check(b.stage.maps.size() == b.hydra.maps.size(), name + ": map count parity");
    for (size_t i = 0; i < std::min(b.stage.maps.size(), b.hydra.maps.size()); ++i)
        Check(b.stage.maps[i].paintSurface == b.hydra.maps[i].paintSurface &&
                  b.stage.maps[i].paintValues == b.hydra.maps[i].paintValues,
              name + ": paint map " + b.stage.maps[i].path.GetString() + " parity");
    Check(b.stage.validationErrors.empty() == b.hydra.validationErrors.empty(),
          name + ": both builders fail closed, or neither");
}

// curve id -> (root face, root position) for a cooked buffer.
std::map<uint64_t, std::pair<int, GfVec3f>> Roots(UsdGenCurveBuffer const &b)
{
    std::map<uint64_t, std::pair<int, GfVec3f>> roots;
    size_t const n = b.totalCurves;
    size_t const perCurve = n ? b.totalCvs / n : 0;
    for (size_t c = 0; c < n; ++c) {
        size_t const root = b.cvOffsets.empty() ? c * perCurve : size_t(b.cvOffsets[c]);
        roots[b.curveId[c]] = {b.rootPrim[c], GfVec3f(b.px[root], b.py[root], b.pz[root])};
    }
    return roots;
}

}  // namespace

int main()
{
    usdGenRegisterM1Operators();
    SdfLayerRefPtr layer = SdfLayer::CreateAnonymous(".usda");
    if (!layer->ImportFromString(Stage())) {
        std::printf("FAIL: the fixture stage does not parse\n");
        return 1;
    }
    UsdStageRefPtr const stage = UsdStage::Open(layer);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    UsdImagingSceneIndices const indices = UsdImagingCreateSceneIndices(info);
    if (!indices.finalSceneIndex) {
        std::printf("FAIL: no final scene index\n");
        return 1;
    }
    HdSceneIndexBaseRefPtr const index = indices.finalSceneIndex;
    std::string error;

    // ---- the whole mesh: the reference roots ---------------------------
    Built const whole = Build(stage, index, "Whole");
    CheckParity(whole, "Whole");
    UsdGenCurveBuffer wholeRoots;
    Check(Cook(whole.stage, &wholeRoots, &error), "the whole mesh cooks: " + error);
    auto const reference = Roots(wholeRoots);
    Check(reference.size() > 100, "the whole mesh scatters roots on every face");

    auto restrictedTo = [&](std::set<int> const &faces) {
        std::map<uint64_t, std::pair<int, GfVec3f>> kept;
        for (auto const &root : reference)
            if (faces.count(root.second.first)) kept.insert(root);
        return kept;
    };
    auto sameRoots = [](std::map<uint64_t, std::pair<int, GfVec3f>> const &a,
                        std::map<uint64_t, std::pair<int, GfVec3f>> const &b) {
        if (a.size() != b.size()) return false;
        for (auto const &root : a) {
            auto const it = b.find(root.first);
            if (it == b.end() || it->second.first != root.second.first ||
                (it->second.second - root.second.second).GetLength() > 1e-6f)
                return false;
        }
        return true;
    };

    // ---- one subset ----------------------------------------------------
    Built const crown = Build(stage, index, "Crown");
    CheckParity(crown, "Crown");
    Check(crown.stage.validationErrors.empty(), "a face subset binds without diagnostics");
    Check(crown.stage.surfaces.size() == 1,
          "a subset target is one self-contained surface, with no parent stub");
    if (!crown.stage.surfaces.empty()) {
        UsdGenSurfaceDesc const &s = crown.stage.surfaces[0];
        Check(s.path == SdfPath("/World/Skin/crown") && s.isSubset,
              "the surface is the subset target");
        Check(UsdGenSurfaceMeshPath(s) == SdfPath("/World/Skin"), "its mesh is the parent");
        Check(s.subsetFaces == VtIntArray({1, 4}), "authored [4, 1, 1] becomes sorted, unique [1, 4]");
        Check(!whole.stage.surfaces.empty() &&
                  s.faceVertexCounts == whole.stage.surfaces[0].faceVertexCounts &&
                  s.restPoints == whole.stage.surfaces[0].restPoints &&
                  s.worldMatrix == whole.stage.surfaces[0].worldMatrix,
              "it carries the parent mesh's full geometry");
        Check(s.surfaceGeneration != whole.stage.surfaces[0].surfaceGeneration,
              "its identity differs from the whole mesh's");
    }
    UsdGenCurveBuffer crownRoots;
    Check(Cook(crown.stage, &crownRoots, &error), "a subset surface cooks: " + error);
    Check(sameRoots(Roots(crownRoots), restrictedTo({1, 4})),
          "a subset scatters exactly the whole mesh's roots on its faces, ids and all");

    // ---- union (rule 4) ------------------------------------------------
    Built const unioned = Build(stage, index, "Union");
    CheckParity(unioned, "Union");
    Check(unioned.stage.surfaces.size() == 1 && unioned.stage.surfaces[0].isSubset &&
              unioned.stage.surfaces[0].subsetFaces == VtIntArray({1, 4, 7}),
          "two subsets of one mesh union into the bound surface");
    Check(!unioned.stage.nodes.empty() && unioned.stage.nodes[0].surfaces.size() == 1,
          "the unioned member leaves the node's surface list");
    UsdGenCurveBuffer unionRoots;
    Check(Cook(unioned.stage, &unionRoots, &error), "a union cooks: " + error);
    Check(sameRoots(Roots(unionRoots), restrictedTo({1, 4, 7})),
          "a union scatters the roots of every member's faces once");

    Built const withMesh = Build(stage, index, "WithMesh");
    CheckParity(withMesh, "WithMesh");
    Check(withMesh.stage.surfaces.size() == 1 &&
              withMesh.stage.surfaces[0].subsetFaces.size() == 9,
          "a union with the mesh itself selects every face");
    UsdGenCurveBuffer withMeshRoots;
    Check(Cook(withMesh.stage, &withMeshRoots, &error), "a union with the mesh cooks: " + error);
    Check(sameRoots(Roots(withMeshRoots), reference), "and scatters exactly the whole mesh");

    // ---- an empty subset -----------------------------------------------
    Built const bald = Build(stage, index, "Bald");
    CheckParity(bald, "Bald");
    Check(bald.stage.surfaces.size() == 1 && bald.stage.surfaces[0].isSubset &&
              bald.stage.surfaces[0].subsetFaces.empty(),
          "an empty subset is a restricted surface with no face");
    UsdGenCurveBuffer baldRoots;
    Check(Cook(bald.stage, &baldRoots, &error), "an empty subset cooks: " + error);
    Check(baldRoots.totalCurves == 0, "an empty subset scatters nothing, not the whole mesh");

    // ---- failing closed (rules 1 and 7) --------------------------------
    Built const verts = Build(stage, index, "Verts");
    CheckParity(verts, "Verts");
    Check(Mentions(verts.stage.validationErrors, "elementType 'point'") &&
              !verts.hydra.validationErrors.empty(),
          "a point subset is a hard diagnostic in both builders");
    UsdGenCurveBuffer none;
    Check(!Cook(verts.stage, &none, &error), "a point-subset description publishes nothing");

    Built const wild = Build(stage, index, "Wild");
    CheckParity(wild, "Wild");
    Check(Mentions(wild.stage.validationErrors, "names face 99") &&
              Mentions(wild.hydra.validationErrors, "names face 99"),
          "an index outside the parent is a hard diagnostic naming it");
    Check(!Cook(wild.stage, &none, &error), "an out-of-range subset publishes nothing");

    // ---- paint on a subset ---------------------------------------------
    Built const painted = Build(stage, index, "Painted");
    CheckParity(painted, "Painted");
    Check(painted.stage.validationErrors.empty(),
          "a PaintMap on a subset captures without diagnostics" +
              (painted.stage.validationErrors.empty() ? "" : ": " + painted.stage.validationErrors[0]));
    if (painted.stage.maps.size() == 1) {
        UsdGenMapDesc const &map = painted.stage.maps[0];
        Check(map.paintSurface == SdfPath("/World/Skin"), "the paint is the parent mesh's primvar");
        bool masked = map.paintValues.size() == 36;
        for (size_t corner = 0; masked && corner < 36; ++corner) {
            int const face = int(corner / 4);
            float const expected = face == 1 || face == 4 ? float(0.1 * (face + 1)) : 0.25f;
            masked = std::abs(map.paintValues[corner] - expected) < 1e-6f;
        }
        Check(masked, "faces outside the subset read usdGen:map:default");
    } else {
        Check(false, "the Painted description captures one map");
    }
    UsdGenCurveBuffer paintedCurves;
    std::vector<std::string> paintWarnings;
    Check(Cook(painted.stage, &paintedCurves, &error, &paintWarnings),
          "paint on a subset root surface cooks: " + error);
    Check(paintWarnings.empty(), "a subset paint read warns about nothing" +
          (paintWarnings.empty() ? "" : ": " + paintWarnings[0]));
    {
        bool reads = paintedCurves.totalCurves > 0 && !paintedCurves.width.empty();
        std::string first;
        size_t const perCurve = paintedCurves.totalCurves
            ? paintedCurves.totalCvs / paintedCurves.totalCurves : 0;
        for (size_t c = 0; reads && c < paintedCurves.totalCurves; ++c) {
            int const face = paintedCurves.rootPrim[c];
            size_t const root = paintedCurves.cvOffsets.empty()
                ? c * perCurve : size_t(paintedCurves.cvOffsets[c]);
            reads = (face == 1 || face == 4) &&
                    std::abs(paintedCurves.width[root] - float(0.1 * (face + 1))) < 1e-4f;
            if (!reads)
                first = ": strand " + std::to_string(c) + " on face " + std::to_string(face) +
                        " has width " + std::to_string(paintedCurves.width[root]);
        }
        Check(reads, "strands on the subset read their face's paint through ptex()" + first);
    }

    // ---- geoSampler over a subset --------------------------------------
    Built const sampled = Build(stage, index, "Sampled");
    CheckParity(sampled, "Sampled");
    Check(sampled.stage.validationErrors.empty(), "a subset expression input captures cleanly");
    Check(sampled.stage.geometries.size() == 1 &&
              sampled.stage.geometries[0].path == SdfPath("/World/Skin/temple") &&
              sampled.stage.geometries[0].isSubset &&
              sampled.stage.geometries[0].subsetFaces == VtIntArray({7}) &&
              sampled.stage.geometries[0].counts.size() == 9,
          "a subset input is the parent mesh restricted to its faces");
    UsdGenCurveBuffer sampledCurves;
    Check(Cook(sampled.stage, &sampledCurves, &error), "a subset geoSampler cooks: " + error);
    {
        bool nearestIsTemple = sampledCurves.totalCurves > 0 && !sampledCurves.width.empty();
        for (float w : sampledCurves.width)
            nearestIsTemple = nearestIsTemple && std::abs(w - 7.0f) < 1e-4f;
        Check(nearestIsTemple, "every strand's nearest subset face is the temple's parent face 7");
    }

    std::printf("geomSubset: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
