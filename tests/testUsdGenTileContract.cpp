// testUsdGenTileContract — gate SI-1 / C2 (plan 06 §4.1, plan 05 §"SI-1",
// docs/m1/interfaces.md §7): the tile publication contract.
//
// Tier T0: engine + the two pure statics of UsdGenTilePublisher. No stage, no
// plugin load, no Hydra engine (the engine never sees a stage, S8).
//
// A 32768-curve groom (guides curve set -> UsdGenGrow terminal) is committed
// through the real session; every assertion is made on the PUBLISHED
// UsdGenTilePublication values and on the data sources the publisher builds
// from them (the identical payload Hydra's GetPrim would expose):
//   * SI-1: points.size() == sum(curveVertexCounts) per tile,
//           uniform arrays sized curveCount, vertex arrays sized total,
//   * tile count == min(target, chunks) and paths under
//     <description>/__usdGenRender/tile_NNNN (4-digit),
//   * HdType basisCurves; the C2 child-name set; cubic/pinned never authored,
//   * hand-authored per-tile fields (xform, purpose, visibility, material,
//     primOrigin) match the graph desc post-flattening,
//   * commit determinism (IsIdentical between identical commits),
//   * generation id strictly monotonic across commits,
//   * NoticesFor(UsdGenTileDirty) emits EXACTLY the §4.1 locator mapping.
//
// Timing-soft (plan 09): nothing here is budget-gated.

#include "usdGen/curveBuffer.h"
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/opRegistry.h"
#include "usdGen/session.h"
#include "usdGen/types.h"

#include "usdGenImaging/usdGenTilePublisher.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/dataSourceTypeDefs.h"
#include "pxr/imaging/hd/dataSourceLocator.h"
#include "pxr/imaging/hd/basisCurvesSchema.h"
#include "pxr/imaging/hd/basisCurvesTopologySchema.h"
#include "pxr/imaging/hd/materialBindingsSchema.h"
#include "pxr/imaging/hd/materialNetworkSchema.h"
#include "pxr/imaging/hd/materialNodeSchema.h"
#include "pxr/imaging/hd/materialSchema.h"
#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/usd/usd/timeCode.h"
#include "pxr/imaging/hd/tokens.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <memory>
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
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

class GuideIndexWriter final : public UsdGenOp
{
public:
    TfToken Type() const override { return TfToken("UsdGenTestGuideIndexWriter"); }
    TfSpan<const TfToken> TopologyParameters() const override { return {}; }
    TfSpan<const TfToken> ValueParameters() const override { return {}; }
    TfSpan<const TfToken> OutputPrimvars() const override { return _outputs; }
    bool Bind(UsdGenParamView const&, UsdGenDiagnostics*) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const&) const override { return {19, 1}; }
    bool Capture(UsdGenCaptureContext const&, UsdGenCurveBuffer const&,
                 UsdGenCapture*, UsdGenDiagnostics*) override { return true; }
    void Evaluate(UsdGenEvalContext const&, UsdGenCapture const&,
                  UsdGenChunkView* view) const override {
        if (!view->outI || !view->outI[0]) return;
        for (uint32_t curve = 0; curve != view->curveCount; ++curve)
            for (uint32_t component = 0; component != 3; ++component)
                view->outI[0][curve * 3 + component] = int(component);
    }
private:
    std::array<TfToken, 1> const _outputs{{TfToken("guideIndex")}};
};

const int kCurveCount = 32768;   // >= 32768 per plan 05's SI-1 fixture size
const int kVertsPerCurve = 4;

// ---------------------------------------------------------------------------
// Synthetic desc: a surface-scatter generator feeding one UsdGenGrow, the
// canonical M1 groom shape. A generator is required for tiles to exist at
// all: the scheduler only (re)partitions the tile set when a generator's
// capture yields curves (scheduler.cpp "Generators establish/refresh the
// topology"), and the M1 reference lane does not materialize curveSets into
// buffers (scheduler.cpp header: 03 §1.5). ~32 940 roots come out of a
// 18.15x18.15 mesh at the default density 100, >= the 32 768 SI-1 fixture
// size; guides ride along as a declared Reference curveSet (R23).

// ---------------------------------------------------------------------------
UsdGenGraphDesc MakeGroomDesc(SdfPath const &descPath, float width = 0.02f)
{
    UsdGenGraphDesc d;
    d.description = descPath;
    d.terminal = descPath.AppendChild(TfToken("guideIndexWriter"));
    d.tileTarget = 64;
    d.curveBasis = TfToken("bspline");
    d.xformMatrix = GfMatrix4d(1.0);   // stage-side: post-flattening desc matrix
    d.purpose = TfToken("render");
    d.visibility = TfToken("invisible");
    d.materialPath = descPath.AppendChild(TfToken("look")).AppendChild(TfToken("material"));
    d.time = 0.0;

    UsdGenCurveSetDesc guides;
    guides.path = descPath.AppendChild(TfToken("guides"));
    guides.role = UsdGenRole::Reference;
    guides.curveRole = TfToken("guide");
    guides.curveVertexCounts = VtIntArray(kCurveCount, kVertsPerCurve);
    guides.points.resize(size_t(kCurveCount) * kVertsPerCurve);
    guides.curveId.resize(size_t(kCurveCount));
    for (int c = 0; c < kCurveCount; ++c) {
        // Deterministic lattice roots; z grows along the curve.
        float const u = float(c % 256) * 0.01f;
        float const v = float(c / 256) * 0.01f;
        for (int j = 0; j < kVertsPerCurve; ++j) {
            guides.points[size_t(c) * kVertsPerCurve + j] =
                GfVec3f(u, v, 0.25f * j);
        }
        guides.curveId[c] = uint64_t(c) + 1;
    }
    guides.widths = VtFloatArray(size_t(kCurveCount) * kVertsPerCurve, 1.0f);
    d.curveSets.push_back(std::move(guides));

    // Generator surface: a flat 4x4-corner grid scaled so rest area is
    // 18.15^2 = 329.42; at the default density 100 scatter emits ~32 940
    // roots (>= kCurveCount), spread over 9 faces.
    UsdGenSurfaceDesc surf;
    surf.path = descPath.AppendChild(TfToken("surface"));
    surf.id = 0;
    surf.faceVertexCounts = VtIntArray(9, 4);
    surf.faceVertexIndices = VtIntArray{0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6,
                                        4, 5, 9, 8, 5, 6, 10, 9, 6, 7, 11, 10,
                                        8, 9, 13, 12, 9, 10, 14, 13, 10, 11, 15, 14};
    surf.restPoints.resize(16);
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            surf.restPoints[size_t(j) * 4 + i] =
                GfVec3f(6.05f * i, 6.05f * j, 0.0f);
    d.surfaces.push_back(std::move(surf));

    UsdGenNodeDesc scatter;
    scatter.path = descPath.AppendChild(TfToken("scatter"));
    scatter.type = TfToken("UsdGenScatter");
    scatter.enabled = true;
    scatter.seed = 11;
    scatter.surfaces.push_back(surf.path);
    scatter.params.push_back(UsdGenParamValue{TfToken("flip"), VtValue(false), false});
    d.nodes.push_back(std::move(scatter));

    UsdGenNodeDesc grow;
    grow.path = descPath.AppendChild(TfToken("grow"));
    grow.type = TfToken("UsdGenGrow");
    grow.enabled = true;
    grow.seed = 7;
    grow.inputs = { descPath.AppendChild(TfToken("scatter")) };   // scatter's roots
    grow.params.push_back(UsdGenParamValue{TfToken("segments"), VtValue(8), false});
    grow.params.push_back(UsdGenParamValue{TfToken("width"), VtValue(width), false});
    d.nodes.push_back(std::move(grow));
    UsdGenNodeDesc guideIndex;
    guideIndex.path = d.terminal;
    guideIndex.type = TfToken("UsdGenTestGuideIndexWriter");
    guideIndex.inputs = {descPath.AppendChild(TfToken("grow"))};
    d.nodes.push_back(std::move(guideIndex));
    return d;
}

std::string TilePathRegexCheck(SdfPath const &descPath, UsdGenTileId tile)
{
    // Publisher's own statics are the contract's source of truth; assert the
    // documented shape too (plan 06 §4.1): description/__usdGenRender/tile_NNNN.
    SdfPath expected = UsdGenTilePublisher::TilePath(descPath, tile);
    std::string s = expected.GetText();
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s/%s/tile_%04u", descPath.GetText(),
                  UsdGenTilePublisher::RenderNamespace().GetText(),
                  unsigned(tile));
    bool prefix = (s == buf);
    return prefix ? std::string() : ("path is " + s + ", expected " + buf);
}

std::set<std::string> ChildNames(HdContainerDataSourceHandle const &c)
{
    std::set<std::string> names;
    if (!c) return names;
    for (TfToken const &t : c->GetNames()) names.insert(t.GetString());
    return names;
}

// 26.08 has no HdDataSourceIsIdentical: compare leaf VtValues recursively.
bool ValuesIdentical(HdDataSourceBaseHandle const &a, HdDataSourceBaseHandle const &b)
{
    HdContainerDataSourceHandle ca = HdContainerDataSource::Cast(a);
    HdContainerDataSourceHandle cb = HdContainerDataSource::Cast(b);
    if (ca || cb) {
        if (!ca || !cb) return false;
        if (ca->GetNames() != cb->GetNames()) return false;
        for (TfToken const &t : ca->GetNames())
            if (!ValuesIdentical(ca->Get(t), cb->Get(t))) return false;
        return true;
    }
    HdSampledDataSourceHandle sa = HdSampledDataSource::Cast(a);
    HdSampledDataSourceHandle sb = HdSampledDataSource::Cast(b);
    if (sa && sb) return sa->GetValue(0.0) == sb->GetValue(0.0);
    return bool(sa) == bool(sb) && !sa;
}

// Leaf sample value at Default() time, or empty VtValue.
VtValue LeafValue(HdDataSourceBaseHandle ds)
{
    if (HdSampledDataSourceHandle s = HdSampledDataSource::Cast(ds))
        return s->GetValue(0.0);
    return VtValue();
}

// Token-valued leaf, tolerant of TfToken vs std::string storage.
bool LeafIsToken(VtValue const &v, char const *text)
{
    if (v.IsHolding<TfToken>()) return v.UncheckedGet<TfToken>() == TfToken(text);
    if (v.IsHolding<std::string>()) return v.UncheckedGet<std::string>() == text;
    return false;
}

// Int-valued leaf, tolerant of int/int64 storage.
bool LeafIsInt(VtValue const &v, int64_t expect)
{
    if (v.IsHolding<int>()) return int64_t(v.Get<int>()) == expect;
    if (v.IsHolding<int64_t>()) return v.Get<int64_t>() == expect;
    return false;
}

size_t SumCounts(VtIntArray const &counts)
{
    size_t n = 0;
    for (int i : counts) n += size_t(i);
    return n;
}

void CheckPartitionBoundaries()
{
    UsdGenGraphDesc desc = MakeGroomDesc(SdfPath("/partition"));
    desc.tileTarget = 32;
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult result = compiler.Compile(desc, &graph);
    Check(result.ok, "boundary partition fixture compiles");
    if (!result.ok) return;
    for (int nChunks : {33, 34, 65}) {
        Check(graph.Repartition(nChunks * kUsdGenDefaultChunkSize, 1),
              "boundary repartition reports layout");
        auto tiles = graph.Tiles();
        Check(tiles.size() == 32, "small partition clamps to 32 tiles");
        uint32_t covered = 0;
        for (size_t i = 0; i < tiles.size(); ++i) {
            Check(tiles[i].firstChunk == covered && tiles[i].chunkCount >= 1 &&
                  tiles[i].chunkCount <= 3, "tile boundary has no phantom chunk");
            covered += tiles[i].chunkCount;
        }
        Check(covered == static_cast<uint32_t>(nChunks),
              "tile boundaries cover every chunk exactly once");
        if (nChunks == 33) {
            Check(tiles[0].chunkCount == 2 && tiles[1].chunkCount == 1 &&
                  tiles[31].chunkCount == 1, "33 chunks use [2,31x1]");
        } else if (nChunks == 34) {
            Check(tiles[0].chunkCount == 2 && tiles[1].chunkCount == 2 &&
                  tiles[2].chunkCount == 1 && tiles[31].chunkCount == 1,
                  "34 chunks use [2,2,30x1]");
        } else {
            bool golden = tiles[16].chunkCount == 2;
            for (int i = 0; i < 16; ++i) golden &= tiles[i].chunkCount == 3;
            for (int i = 17; i < 32; ++i) golden &= tiles[i].chunkCount == 1;
            Check(golden, "65 chunks use [16x3,2,15x1]");
        }
        for (int node = 0; node < graph.NodeCount(); ++node) {
            auto chunks = graph.Chunks(static_cast<UsdGenGraph::NodeId>(node));
            for (size_t c = 0; c < chunks.size(); ++c) {
                auto const &chunk = chunks[c];
                Check(chunk.tile < tiles.size() &&
                      c >= tiles[chunk.tile].firstChunk &&
                      c < tiles[chunk.tile].firstChunk + tiles[chunk.tile].chunkCount,
                      "chunk tile ID agrees with tile interval");
            }
        }
    }
    desc.tileTarget = 64;
    UsdGenGraph canonical;
    result = compiler.Compile(desc, &canonical);
    Check(result.ok && canonical.Repartition(100000, 4),
          "canonical 100k repartition reports layout");
    Check(canonical.Tiles().size() == 49,
          "100k/512/tileTarget64 remains 49 tiles");
}

}  // namespace

int main()
{
    usdGenRegisterM1Operators();
    Check(UsdGenOpRegistry::Get().Register(TfToken("UsdGenTestGuideIndexWriter"),
              [] { return std::make_unique<GuideIndexWriter>(); }),
          "registers typed extra-plane publication writer");
    CheckPartitionBoundaries();

    SdfPath const descPath("/groom");
    UsdGenSession session;
    session.SetGraphDesc(MakeGroomDesc(descPath));
    UsdGenGenerationConstPtr gen =
        session.Commit(0.0, UsdGenCommitReason::LiveOverride);
    Check(bool(gen), "first commit publishes a generation");
    if (!gen) {
        std::printf("testUsdGenTileContract: FAILED (no generation)\n");
        return 1;
    }

    // ---- SI-1 / C2 per-tile value contract ------------------------------
    size_t totalCurves = 0, totalPoints = 0;
    bool si1 = true, uniformSizes = true, vertexSizes = true, typedExtras = true, bases = true,
         refines = true, extents = true, paths = true, xforms = true,
         inherit = true, idArrays = true;
    size_t countMismatch = 0;
    for (auto const &pub : gen->tiles) {
        size_t const curves = size_t(pub.curveVertexCounts.size());
        size_t const pts = SumCounts(pub.curveVertexCounts);
        totalCurves += curves;
        totalPoints += pub.points.size();
        if (pub.points.size() != pts) { si1 = false; ++countMismatch; }
        if (pub.hairT.size() != pts) vertexSizes = false;
        if (pub.hairId.size() != curves) uniformSizes = false;
        if (pub.hairId.empty()) idArrays = false;
        if (pub.st.size() != curves && !pub.st.empty()) uniformSizes = false;
        if (!pub.widths.empty() &&
            pub.widths.size() != curves && pub.widths.size() != pts)
            si1 = false;
        if (pub.displayColor.size() != curves && pub.displayColor.size() != pts)
            si1 = false;
        for (auto const &p : pub.extraUniform) {
            size_t elements = p.interpolation == TfToken("constant") ? 1 :
                p.interpolation == TfToken("vertex") ? pts : curves;
            size_t values = elements * p.arity;
            if (p.type == TfToken("int")) {
                if (p.i.size() != values || !p.f.empty())
                    (p.interpolation == TfToken("vertex") ? vertexSizes : uniformSizes) = false;
            } else if (p.type == TfToken("float")) {
                if (p.f.size() != values || !p.i.empty())
                    (p.interpolation == TfToken("vertex") ? vertexSizes : uniformSizes) = false;
            } else {
                uniformSizes = false;
            }
            if (p.name == TfToken("guideIndex"))
                typedExtras = typedExtras && p.interpolation == TfToken("uniform") &&
                    p.type == TfToken("int") && p.arity == 3 &&
                    p.i.size() == curves * 3 && p.f.empty();
        }
        typedExtras = typedExtras && std::any_of(pub.extraUniform.begin(),
            pub.extraUniform.end(), [](UsdGenPlane const& p) {
                return p.name == TfToken("guideIndex");
            });
        if (pub.basis != "bspline") bases = false;
        if (pub.refineLevel != 2) refines = false;
        if (!(pub.extentMin[0] <= pub.extentMax[0] &&
              pub.extentMin[1] <= pub.extentMax[1] &&
              pub.extentMin[2] <= pub.extentMax[2])) extents = false;
        GfVec3d mn(+1e30, +1e30, +1e30), mx(-1e30, -1e30, -1e30);
        for (auto const &p : pub.points) {
            mn[0] = std::min(mn[0], double(p[0])); mn[1] = std::min(mn[1], double(p[1]));
            mn[2] = std::min(mn[2], double(p[2]));
            mx[0] = std::max(mx[0], double(p[0])); mx[1] = std::max(mx[1], double(p[1]));
            mx[2] = std::max(mx[2], double(p[2]));
        }
        if ((mn - pub.extentMin).GetLength() > 1e-4 ||
            (mx - pub.extentMax).GetLength() > 1e-4) extents = false;
        if (!TilePathRegexCheck(descPath, pub.tile).empty()) paths = false;
        if (!(pub.xformMatrix == GfMatrix4d(1.0))) xforms = false;
        if (pub.purpose != TfToken("render") || pub.visibility != TfToken("invisible") ||
            pub.materialPath != descPath.AppendChild(TfToken("look")).AppendChild(TfToken("material")))
            inherit = false;
        if (pub.primOrigin != descPath) inherit = false;   // pickTarget == description
    }
    Check(!gen->tiles.empty(), "published at least one tile");
    Check(totalCurves >= size_t(kCurveCount),
          "tile curve set covers >= 32768 curves (got " + std::to_string(totalCurves) + ")");
    Check(gen->tiles.size() <= size_t(64),
          "tile count <= tileTarget 64 (got " + std::to_string(gen->tiles.size()) + ")");
    Check(si1, "SI-1: points.size() == sum(curveVertexCounts) on every tile"
               " (mismatches: " + std::to_string(countMismatch) + ")");
    Check(uniformSizes, "uniform primvars sized curveCount on every tile");
    Check(idArrays, "hairId uniform float array present on every tile (C2)");
    Check(vertexSizes, "vertex primvars sized points on every tile");
    Check(typedExtras, "typed extra-plane publication preserves int arity-3 metadata and payload");
    Check(bases, "curve basis == usdGen:curve:basis (bspline) on every tile");
    Check(refines, "published refineLevel == 2 on every tile (the FALLBACK "
                   "underlay; an upstream displayStyle opinion wins, C2)");
    Check(extents, "extent/min|max == min|max over tile points (exact per frame)");
    Check(paths, "tile primPaths == <desc>/__usdGenRender/tile_NNNN (4-digit)");
    Check(xforms, "identity xform matrix (rest-only graph)");
    Check(inherit, "purpose/visibility/material/primOrigin inherited by hand (C2)");

    // ---- Publisher-built data sources: HdType + C2 child-name set -------
    if (!gen->tiles.empty()) {
        UsdGenTilePublication const &pub = gen->tiles[0];
        // T0 scope: BuildTileDataSource is the pure data-source assembler,
        // so the observable surface here is the container tree the scene
        // index would hand Hydra. The prim-type token (basisCurves) is a
        // property of HdSceneIndexPrim, asserted in the T1 imaging test.
        HdContainerDataSourceHandle c = UsdGenTilePublisher::BuildTileDataSource(pub);
        Check(bool(c), "BuildTileDataSource returns a container");
        if (c) {
            std::set<std::string> names = ChildNames(c);
            // Contract container is `basisCurves` (06 §4.1); the publisher
            // also carries displayStyle/purpose/visibility/materialBindings/
            // primOrigin/__dependencies/generation alongside it.
            for (char const *req : { "basisCurves", "primvars", "extent", "xform",
                                     "displayStyle", "purpose", "visibility",
                                     "materialBindings", "primOrigin" }) {
                Check(names.count(req) != 0,
                      std::string("C2 child '") + req + "' published");
            }
            if (HdContainerDataSourceHandle basisCurves =
                    HdContainerDataSource::Cast(c->Get(TfToken("basisCurves")))) {
                // C2 (docs/freezes/C2.md:20-23): Hydra reads curveVertexCounts
                // AND type/basis/wrap from inside basisCurves/topology
                // (hd/basisCurvesTopologySchema.h:35-42). Authoring any of
                // them as a direct child of `basisCurves` is invisible to
                // Hydra, which then falls back to type = linear / basis =
                // bezier / wrap = nonperiodic
                // (hd/sceneIndexAdapterSceneDelegate.cpp:865-911) and the
                // tile can never render as a smooth cubic curve.
                std::set<std::string> bcn = ChildNames(basisCurves);
                Check(bcn.count("topology") != 0,
                      "basisCurves/topology container published (Hd nesting, C2)");
                for (char const *forbidden : { "type", "basis", "wrap" }) {
                    Check(bcn.count(forbidden) == 0,
                          std::string("'") + forbidden + "' is NOT a direct "
                          "child of basisCurves (Hydra never reads it there)");
                }
                HdBasisCurvesTopologySchema topo =
                    HdBasisCurvesSchema::GetFromParent(c).GetTopology();
                Check(topo.IsDefined(),
                      "HdBasisCurvesSchema::GetTopology() resolves");
                Check(topo.GetType() &&
                          topo.GetType()->GetTypedValue(0) == HdTokens->cubic,
                      "basisCurves/topology/type == cubic (contract constant, C2)");
                Check(topo.GetWrap() &&
                          topo.GetWrap()->GetTypedValue(0) == HdTokens->pinned,
                      "basisCurves/topology/wrap == pinned (contract constant, C2)");
                Check(topo.GetBasis() &&
                          topo.GetBasis()->GetTypedValue(0) == HdTokens->bspline,
                      "basisCurves/topology/basis == usdGen:curve:basis (bspline)");
                Check(bool(topo.GetCurveVertexCounts()),
                      "basisCurves/topology/curveVertexCounts published");
                Check(!topo.GetCurveIndices(),
                      "basisCurves/topology/curveIndices NEVER published (C2)");
            } else {
                Check(false, "basisCurves container published");
            }
            if (HdContainerDataSourceHandle primvars = HdContainerDataSource::Cast(
                    c->Get(TfToken("primvars")))) {
                std::set<std::string> pv = ChildNames(primvars);
                Check(pv.count("usdGenId") == 0,
                      "usdGenId is NOT published in M1 (S30 reserved)");
                Check(pv.count("normals") == 0,
                      "normals NEVER published (forces Storm ribbon key, C2)");
                // Always emitted: points/widths/hairT/hairId plus blocked
                // velocities/accelerations sentinels. st/displayColor/
                // bakeColor/extra planes ride only when the engine emits
                // them (look bake / bound data).
                for (char const *req : { "points", "widths", "hairT", "hairId",
                                         "velocities", "accelerations" }) {
                    Check(pv.count(req) != 0,
                          std::string("primvar '") + req + "' published");
                }
            }
            if (HdContainerDataSourceHandle style = HdContainerDataSource::Cast(
                    c->Get(TfToken("displayStyle")))) {
                // C2: the publisher's refineLevel is the UNDERLAY only.
                // UsdGenGroomSceneIndex::GetPrim overlays the owning
                // Description's upstream displayStyle on top of it, so the
                // usdview complexity slider (pushed as
                // SetRefineLevelFallback -> every stage prim) wins 0..3
                // exactly as it does for a native BasisCurves. What the
                // publisher alone emits is the no-opinion fallback.
                Check(LeafIsInt(LeafValue(style->Get(TfToken("refineLevel"))), 2),
                      "displayStyle/refineLevel == 2 when no host opinion "
                      "exists (published fallback, C2)");
            }
            // The app-facing literal "allPurpose" is normalized to Hydra's
            // empty-token default child, whose value is a binding schema.
            HdMaterialBindingsSchema bindings =
                HdMaterialBindingsSchema::GetFromParent(c);
            auto materialPath = bindings.GetMaterialBinding().GetPath();
            Check(bindings.IsDefined() && materialPath &&
                      materialPath->GetTypedValue(0) == pub.materialPath,
                  "materialBindings default empty-token child carries path");
            {
                // Previously untested branch: with NO authored material the
                // tile used to carry no materialBindings at all, so Storm
                // fell back to flat displayColor shading and the hair lobes
                // never ran. It now binds the synthetic default material in
                // the 06 §4.4 material_storm slot.
                UsdGenTilePublication unbound = pub;
                unbound.materialPath = SdfPath();
                unbound.materialPurpose = TfToken();
                HdContainerDataSourceHandle cu =
                    UsdGenTilePublisher::BuildTileDataSource(unbound);
                HdMaterialBindingsSchema unboundBindings =
                    HdMaterialBindingsSchema::GetFromParent(cu);
                auto unboundPath = unboundBindings.GetMaterialBinding().GetPath();
                SdfPath const expected =
                    UsdGenTilePublisher::MaterialPath(descPath);
                Check(unboundBindings.IsDefined() && unboundPath &&
                          unboundPath->GetTypedValue(0) == expected,
                      "unbound tile binds <desc>/__usdGenRender/material_storm");
                Check(UsdGenTilePublisher::DefaultMaterialPath(unbound.primPath)
                          == expected,
                      "DefaultMaterialPath(tile) == MaterialPath(description)");
                Check(bool(UsdGenTilePublisher::BuildDefaultMaterialDataSource()),
                      "synthetic default material data source builds");
            }
            if (HdContainerDataSourceHandle primvars = HdContainerDataSource::Cast(
                    c->Get(TfToken("primvars")))) {
                HdContainerDataSourceHandle msw = HdContainerDataSource::Cast(
                    primvars->Get(TfToken("minScreenSpaceWidths")));
                Check(bool(msw) &&
                          LeafIsToken(LeafValue(msw->Get(TfToken("interpolation"))),
                                      "constant"),
                      "primvars/minScreenSpaceWidths is constant (C2:35)");
            }
            HdVisibilitySchema visibility = HdVisibilitySchema::GetFromParent(c);
            Check(visibility.IsDefined() && visibility.GetVisibility() &&
                      visibility.GetVisibility()->GetTypedValue(0) ==
                          (pub.visibility != TfToken("invisible")),
                  "visibility/visibility is Hydra bool, not an app token");
            UsdGenTilePublication emptyPurpose = pub;
            emptyPurpose.materialPurpose = TfToken();
            HdMaterialBindingsSchema emptyBindings =
                HdMaterialBindingsSchema::GetFromParent(
                    UsdGenTilePublisher::BuildTileDataSource(emptyPurpose));
            auto emptyPath = emptyBindings.GetMaterialBinding().GetPath();
            Check(emptyBindings.IsDefined() && emptyPath &&
                      emptyPath->GetTypedValue(0) == pub.materialPath,
                  "empty app material purpose uses Hydra default binding");
            // Purpose: omitted when the publication carries none (unauthored
            // description purpose) so Hydra resolves the geometry render tag;
            // publishing purpose="default" would match NO tag and Storm would
            // never sync the tile (2026-09-12: 49 tiles, itemsDrawn == 1).
            // Authored values pass through verbatim (they ARE render tags).
            if (pub.purpose.IsEmpty()) {
                Check(ChildNames(c).count("purpose") == 0,
                      "no purpose container when publication purpose is empty "
                      "(-> geometry render tag)");
            }
            {
                UsdGenTilePublication authored = pub;
                authored.purpose = TfToken("render");
                HdContainerDataSourceHandle ca =
                    UsdGenTilePublisher::BuildTileDataSource(authored);
                HdContainerDataSourceHandle pa = HdContainerDataSource::Cast(
                    ca ? ca->Get(TfToken("purpose")) : nullptr);
                Check(bool(pa) &&
                          LeafIsToken(LeafValue(pa->Get(TfToken("purpose"))),
                                      "render"),
                      "authored purpose passes through verbatim");
                UsdGenTilePublication bare = pub;
                bare.purpose = TfToken();
                HdContainerDataSourceHandle cb =
                    UsdGenTilePublisher::BuildTileDataSource(bare);
                Check(cb && ChildNames(cb).count("purpose") == 0,
                      "no purpose container when publication purpose is empty "
                      "(-> geometry render tag)");
            }
        }
    }

    // ---- Determinism: identical inputs -> identical published values -----
    session.SetGraphDesc(MakeGroomDesc(descPath));
    UsdGenGenerationConstPtr gen2 =
        session.Commit(0.0, UsdGenCommitReason::LiveOverride);
    Check(bool(gen2), "second commit publishes");
    if (gen2) {
        Check(gen2->id > gen->id,
              "generation id strictly increases (" + std::to_string(gen->id) +
                  " -> " + std::to_string(gen2->id) + ")");
        bool identical = gen2->tiles.size() == gen->tiles.size();
        bool dsIdentical = true;
        if (identical) {
            for (size_t i = 0; i < gen->tiles.size() && identical; ++i) {
                UsdGenTilePublication const &a = gen->tiles[i];
                UsdGenTilePublication const &b = gen2->tiles[i];
                identical = a.tile == b.tile && a.primPath == b.primPath &&
                    a.points == b.points && a.curveVertexCounts == b.curveVertexCounts;
                HdContainerDataSourceHandle da =
                    UsdGenTilePublisher::BuildTileDataSource(a);
                HdContainerDataSourceHandle db =
                    UsdGenTilePublisher::BuildTileDataSource(b);
                if (da && db && !ValuesIdentical(HdDataSourceBaseHandle(da),
                                                 HdDataSourceBaseHandle(db)))
                    dsIdentical = false;
            }
        }
        Check(identical, "re-commit of identical desc publishes identical tile payloads");
        Check(dsIdentical, "re-commit data sources compare value-identical");
    }

    if (gen2 && !gen2->tiles.empty() && !gen->tiles.empty()) {
        // Additive overload (06 §4.1): the container can carry a prim-level
        // `generation` int source with the snapshot id.
        HdContainerDataSourceHandle ca =
            UsdGenTilePublisher::BuildTileDataSource(gen->tiles[0]);
        HdContainerDataSourceHandle cb =
            UsdGenTilePublisher::BuildTileDataSource(gen2->tiles[0], int64_t(gen2->id));
        if (ca && cb) {
            VtValue sv = LeafValue(cb->Get(TfToken("generation")));
            bool stampOk = (sv.IsHolding<int>() &&
                            int64_t(sv.UncheckedGet<int>()) == int64_t(gen2->id)) ||
                           (sv.IsHolding<int64_t>() &&
                            sv.UncheckedGet<int64_t>() == int64_t(gen2->id));
            Check(stampOk, "stamped generation source carries the snapshot id");
        }
    }

    // ---- Path shapes (SI-1): nested, never flat-with-slashes -------------
    {
        Check(UsdGenTilePublisher::RenderNamespace() == TfToken("__usdGenRender"),
              "render namespace token is __usdGenRender (single source of truth)");
        SdfPath tp = UsdGenTilePublisher::TilePath(descPath, UsdGenTileId(7));
        Check(tp == descPath.AppendChild(TfToken("__usdGenRender"))
                        .AppendChild(TfToken("tile_0007")),
              std::string("TilePath nested, zero-padded: ") + tp.GetText());
        Check(tp.GetString().find("tile_0007") != std::string::npos &&
                  tp.GetPathElementCount() == 3,
              "tile path is 3 elements deep (desc/__usdGenRender/tile_%04u)");
        SdfPath gp = UsdGenTilePublisher::GuidePath(descPath, TfToken("guideA"));
        Check(gp == descPath.AppendChild(TfToken("__usdGenRender"))
                        .AppendChild(TfToken("guides"))
                        .AppendChild(TfToken("guideA")),
              std::string("GuidePath nested: ") + gp.GetText());
    }

    // ---- Dirty -> notice mapping (06 §5.1 channel discipline) -----------
    // NoticesFor is a pure static of the publisher, so the mapping is graded
    // unconditionally: the §5.1 bare-leaf precondition (no caching filter
    // between usdGen and the render index) only constrains a live engine
    // chain, never these unit assertions. A host-enabled caching scene
    // index is reported, not graded around.
    if (char const *cach =
            std::getenv("USDIMAGINGGL_ENGINE_ENABLE_CACHING_SCENE_INDEX");
        cach && cach[0] != '\0' && std::string(cach) != "0" &&
            std::string(cach) != "false" && std::string(cach) != "off") {
        std::printf("[info] caching scene index enabled (%s); mapping still "
                    "graded (pure publisher statics)\n", cach);
    }
    {
        auto has = [](std::vector<HdDataSourceLocator> const &v,
                      std::vector<TfToken> const &path) {
            for (HdDataSourceLocator const &loc : v) {
                if (loc.GetElementCount() != path.size())
                    continue;
                bool eq = true;
                for (size_t i = 0; i < path.size(); ++i)
                    eq = eq && loc.GetElement(i) == path[i];
                if (eq)
                    return true;
            }
            return false;
        };
        const TfToken pv("primvars"), val("primvarValue"), pts("points"),
            wdt("widths"), ext("extent"), mn("min"), mx("max");

        // §5.1 row 1: points -> bare leaf + extent/min + extent/max.
        UsdGenTileDirty pointsOnly;
        pointsOnly.primPath = descPath;
        pointsOnly.pointsDirty = true;
        auto nPts = UsdGenTilePublisher::NoticesFor(pointsOnly).all();
        Check(has(nPts, { pv, pts, val }) && has(nPts, { ext, mn }) &&
                  has(nPts, { ext, mx }),
              "points dirty -> bare primvars/points/primvarValue + extent min/max");
        Check(!has(nPts, { pv, wdt, val }),
              "points-only notice carries no widths locator");

        // §5.1 row 2: a width-writing node is routed by the engine as an
        // ADDITIONAL dirtyPrimvars entry; the publisher must surface the
        // widths leaf alongside points, never as a container sentinel.
        UsdGenTileDirty pointsPlusWidths;
        pointsPlusWidths.primPath = descPath;
        pointsPlusWidths.pointsDirty = true;
        pointsPlusWidths.dirtyPrimvars = { wdt };
        auto nW = UsdGenTilePublisher::NoticesFor(pointsPlusWidths).all();
        Check(has(nW, { pv, pts, val }) && has(nW, { pv, wdt, val }),
              "width-writing-node dirty -> points AND widths leaves");

        // A widths-only delta must NOT invalidate points (whole-prim scatter).
        UsdGenTileDirty widthsOnly;
        widthsOnly.primPath = descPath;
        widthsOnly.widthsDirty = true;
        widthsOnly.dirtyPrimvars = { TfToken("st") };
        auto nw = UsdGenTilePublisher::NoticesFor(widthsOnly).all();
        Check(!has(nw, { pv, pts, val }) && !has(nw, { ext, mn }) &&
                  !has(nw, { ext, mx }),
              "widths-only notice carries no points or extent locator");
        Check(has(nw, { pv, TfToken("st"), val }),
              "dirtyPrimvars surfaces a primvars/st/primvarValue dirty locator");

        // §5.1 row 4: a primvar appearing for the first time is announced as
        // primvars/<name> ONCE — never a /primvarValue dirty (that would not
        // make Storm build the container). An added tile carries its whole
        // new-primvar set in newPrimvars.
        UsdGenTileDirty added;
        added.primPath = descPath;
        added.added = true;
        added.newPrimvars = { TfToken("st"), TfToken("displayColor") };
        auto nAdd = UsdGenTilePublisher::NoticesFor(added);
        bool allBare = !nAdd.newPrimvarLocators.empty();
        for (HdDataSourceLocator const &loc : nAdd.newPrimvarLocators) {
            allBare = allBare && loc.GetElementCount() == 2 &&
                loc.GetElement(0) == pv;
        }
        Check(allBare,
              "added tile announces each new primvar once, bare, no leaf dirty");
        auto nAddAll = nAdd.all();
        Check(has(nAddAll, { pv, TfToken("st") }) &&
                  has(nAddAll, { pv, TfToken("displayColor") }),
              "added-tile notice contains primvars/st and primvars/displayColor");

        // xform-only delta: the tile's own xform/matrix and nothing else
        // (§5.1: Hydra dirtiness is per prim, never hierarchical).
        UsdGenTileDirty xformOnly;
        xformOnly.primPath = descPath;
        xformOnly.xformDirty = true;
        auto nx = UsdGenTilePublisher::NoticesFor(xformOnly).all();
        Check(has(nx, { TfToken("xform"), TfToken("matrix") }) &&
                  !has(nx, { pv, pts, val }) && !has(nx, { pv, wdt, val }),
              "xform-only delta is xform-only (no republish, S4)");
    }

    // Synthetic scalp-shadow material: the Storm-only UsdGenScalpShadow
    // node in the glslfx context plus the universal UsdPreviewSurface
    // fallback (docs/moonray-fur.md) -- without the fallback hdMoonray
    // looks for an RDL DSO named UsdGenScalpShadow.
    {
        UsdGenLookDesc look;
        look.rootColor = GfVec3f(0.10f, 0.05f, 0.02f);
        HdContainerDataSourceHandle const ds =
            UsdGenTilePublisher::BuildScalpShadowMaterialDataSource(look);
        HdMaterialSchema const mat = HdMaterialSchema::GetFromParent(ds);
        HdMaterialNodeSchema const shaded = mat
            .GetMaterialNetwork(TfToken("glslfx"))
            .GetNodes().Get(TfToken("surface"));
        Check(shaded.IsDefined() && shaded.GetNodeIdentifier() &&
                  shaded.GetNodeIdentifier()->GetTypedValue(0) ==
                      UsdGenTilePublisher::ScalpShadowIdentifier(),
              "scalp-shadow material shades UsdGenScalpShadow in glslfx");
        HdMaterialNodeSchema const fallback = mat
            .GetMaterialNetwork(HdMaterialSchemaTokens->universalRenderContext)
            .GetNodes().Get(TfToken("surface"));
        Check(fallback.IsDefined() && fallback.GetNodeIdentifier() &&
                  fallback.GetNodeIdentifier()->GetTypedValue(0) ==
                      TfToken("UsdPreviewSurface"),
              "scalp-shadow material falls back to UsdPreviewSurface");
    }

    // Bulk-vs-serial tile differential: the same scatter->grow geometry
    // published with the guideIndex terminal (uniform extra plane: serial
    // loop) and with the bare grow terminal (uniform chunks, covered
    // planes, constant color: bulk path) must carry element-identical
    // core arrays on every tile.
    {
        UsdGenGraphDesc bulkDesc = MakeGroomDesc(descPath);
        bulkDesc.nodes.pop_back();  // drop the guideIndex writer
        bulkDesc.terminal = descPath.AppendChild(TfToken("grow"));
        UsdGenSession bulkSession;
        bulkSession.SetGraphDesc(std::move(bulkDesc));
        UsdGenGenerationConstPtr bulkGen =
            bulkSession.Commit(0.0, UsdGenCommitReason::LiveOverride);
        Check(bool(bulkGen), "bulk-terminal commit publishes a generation");
        Check(bulkGen && bulkGen->tiles.size() == gen->tiles.size(),
              "bulk and serial commits publish the same tile count (bulk " +
                  std::to_string(bulkGen ? bulkGen->tiles.size() : 0) +
                  ", serial " + std::to_string(gen->tiles.size()) + ")");
        bool coresEqual = bulkGen &&
            bulkGen->tiles.size() == gen->tiles.size();
        if (coresEqual) {
            for (size_t i = 0; i < gen->tiles.size(); ++i) {
                UsdGenTilePublication const &a = gen->tiles[i];
                UsdGenTilePublication const &b = bulkGen->tiles[i];
                coresEqual = coresEqual && a.tile == b.tile &&
                    a.primPath == b.primPath &&
                    a.curveVertexCounts == b.curveVertexCounts &&
                    a.points == b.points && a.widths == b.widths &&
                    a.hairT == b.hairT && a.hairId == b.hairId &&
                    a.st == b.st && a.displayColor == b.displayColor &&
                    a.extentMin == b.extentMin && a.extentMax == b.extentMax;
                if (!coresEqual)
                    break;
            }
        }
        Check(coresEqual,
              "bulk and serial tile core arrays are element-identical");
    }

    return g_failures ? 1 : 0;
}
