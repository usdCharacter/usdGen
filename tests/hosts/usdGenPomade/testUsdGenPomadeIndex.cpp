// testUsdGenPomadeIndex — T1: the Pomade scene index publishes the active model.
//
// Plan/18 §2.6: the registry join and the per-level publication are proven
// here, before any Python touches them. No stage, no renderer: the input is
// an empty HdRetainedSceneIndex, so this is T1 with no plugin-path
// environment (the pomade library links directly).
//
// Proven here:
//   * the index attaches to PomadeRegistry on construction and detaches on
//     destruction; it owns no model;
//   * with no active model and USDGENPOMADE_TEST_TUBE=1 it publishes the
//     static test tube, and the first Pomade_Activate removes it
//     (PrimsRemoved) for good — deactivating does not bring it back;
//   * an activated model publishes /__usdGenPomade/tubes/L1 plus its center
//     curves, CV dots, rings and ring CVs, with uniform primvars sized to
//     the real face/curve count (not the P3 single-element `constant`);
//   * Pomade_SubdivideTube adds the whole L2 family, and the L1 prims are
//     dirtied for topology while L2 arrives as PrimsAdded;
//   * a single-tube move re-tessellates exactly one tube and dirties
//     EXACTLY the point/extent leaves of the level it moved in — no
//     topology, no uniforms, no other level;
//   * an x-ray change dirties exactly the xray primvar of that level's
//     mesh, a visibility change exactly the visibility leaf of every prim
//     in the level, and a focus change exactly the overlay widths;
//   * the clump palette is the same table the scalp tint reads;
//   * input prims pass through GetPrim/GetChildPrimPaths untouched, and
//     input notices forward;
//   * V1: a selection change dirties EXACTLY the `selected` primvar of the
//     level mesh and its curves, and exactly displayColor on the CV dots,
//     re-tessellating nothing; a hover reads 2 where a selection reads 1;
//     a level selection lights every tube at that level and no other; and
//     the gizmo / brushRing prims appear when the model holds a record,
//     dirty only their own leaves when it moves, and are removed when it
//     is cleared;
//   * V8: every scalp face a region claims is painted the clumpColor of
//     the tube rooted in that region, the tint mesh is lifted off the
//     scalp it copies so the two cannot z-fight, and the overlay widths
//     are the section 2.4a pixel targets times the model's display scale
//     (with the pre-V8 radius-relative widths when no camera has set
//     one).

#include "usdGenPomade/imaging/pomadeSceneIndex.h"
#include "usdGenPomade/pomadeApi.h"
#include "usdGenPomade/pomadeApiStage.h"
#include "usdGenPomade/pomadeGizmo.h"
#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadePublish.h"
#include "usdGenPomade/pomadeSelection.h"
#include "usdGenPomade/pomadeTransport.h"
#include "usdGenPomade/pomadeRegistry.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"
#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/dataSourceLocator.h"
#include "pxr/imaging/hd/dataSourceTypeDefs.h"
#include "pxr/imaging/hd/materialBindingSchema.h"
#include "pxr/imaging/hd/materialBindingsSchema.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/retainedSceneIndex.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"

#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

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
    // Unbuffered: a crash in a later block must not swallow the line that
    // says how far the run got.
    std::fflush(stdout);
}

bool HasChild(HdSceneIndexBase const &index, SdfPath const &parent,
              SdfPath const &child)
{
    for (SdfPath const &candidate : index.GetChildPrimPaths(parent)) {
        if (candidate == child) {
            return true;
        }
    }
    return false;
}

HdSampledDataSourceHandle SampledAt(HdContainerDataSourceHandle const &root,
                                    HdDataSourceLocator const &locator)
{
    return HdSampledDataSource::Cast(
        HdContainerDataSource::Get(root, locator));
}

HdSampledDataSourceHandle Primvar(HdSceneIndexBase const &index,
                                  SdfPath const &path, char const *name)
{
    return SampledAt(index.GetPrim(path).dataSource,
                     HdDataSourceLocator(TfToken("primvars"), TfToken(name),
                                         TfToken("primvarValue")));
}

// The material a prim binds for allPurpose, or an empty path.
SdfPath MaterialBindingOf(HdSceneIndexBase const &index, SdfPath const &path)
{
    HdSampledDataSourceHandle ds = SampledAt(
        index.GetPrim(path).dataSource,
        HdDataSourceLocator(HdMaterialBindingsSchemaTokens->materialBindings,
                            HdMaterialBindingsSchemaTokens->allPurpose,
                            HdMaterialBindingSchemaTokens->path));
    if (!ds) {
        return SdfPath();
    }
    VtValue const v = ds->GetValue(0.0f);
    return v.IsHolding<SdfPath>() ? v.UncheckedGet<SdfPath>() : SdfPath();
}

template <class T>
bool ArraySize(HdSceneIndexBase const &index, SdfPath const &path,
               char const *name, size_t expected, T *out = nullptr)
{
    HdSampledDataSourceHandle ds = Primvar(index, path, name);
    if (!ds) {
        return false;
    }
    VtValue v = ds->GetValue(0.0f);
    if (!v.IsHolding<T>()) {
        return false;
    }
    T const &array = v.UncheckedGet<T>();
    if (out) {
        *out = array;
    }
    return array.size() == expected;
}

template <class T>
bool ArrayNonEmpty(HdSceneIndexBase const &index, SdfPath const &path,
                   char const *name)
{
    HdSampledDataSourceHandle ds = Primvar(index, path, name);
    if (!ds) {
        return false;
    }
    VtValue const v = ds->GetValue(0.0f);
    return v.IsHolding<T>() && !v.UncheckedGet<T>().empty();
}

class RecordingObserver : public HdSceneIndexObserver {
public:
    void PrimsAdded(HdSceneIndexBase const &,
                    AddedPrimEntries const &entries) override
    {
        added.insert(added.end(), entries.begin(), entries.end());
    }
    void PrimsRemoved(HdSceneIndexBase const &,
                      RemovedPrimEntries const &entries) override
    {
        removed.insert(removed.end(), entries.begin(), entries.end());
    }
    void PrimsDirtied(HdSceneIndexBase const &,
                      DirtiedPrimEntries const &entries) override
    {
        dirtied.insert(dirtied.end(), entries.begin(), entries.end());
    }
    void PrimsRenamed(HdSceneIndexBase const &,
                      RenamedPrimEntries const &entries) override
    {
        renamed.insert(renamed.end(), entries.begin(), entries.end());
    }
    void Clear()
    {
        added.clear();
        removed.clear();
        dirtied.clear();
        renamed.clear();
    }
    bool WasAdded(SdfPath const &path) const
    {
        for (auto const &e : added) {
            if (e.primPath == path) {
                return true;
            }
        }
        return false;
    }
    bool WasRemoved(SdfPath const &path) const
    {
        for (auto const &e : removed) {
            if (e.primPath == path) {
                return true;
            }
        }
        return false;
    }
    // The locator set sent for `path`, or an empty set when it was not
    // dirtied. Used for exact-equality asserts, so a co-dirty fails loudly.
    HdDataSourceLocatorSet DirtiedFor(SdfPath const &path) const
    {
        HdDataSourceLocatorSet set;
        for (auto const &e : dirtied) {
            if (e.primPath == path) {
                set.insert(e.dirtyLocators);
            }
        }
        return set;
    }
    bool WasDirtied(SdfPath const &path) const
    {
        for (auto const &e : dirtied) {
            if (e.primPath == path) {
                return true;
            }
        }
        return false;
    }
    AddedPrimEntries added;
    RemovedPrimEntries removed;
    DirtiedPrimEntries dirtied;
    RenamedPrimEntries renamed;
};

size_t LocatorSetSize(HdDataSourceLocatorSet const &set)
{
    size_t n = 0;
    for (HdDataSourceLocator const &loc : set) {
        (void)loc;
        ++n;
    }
    return n;
}

std::string ToString(HdDataSourceLocatorSet const &set)
{
    std::string out = "{";
    bool first = true;
    for (HdDataSourceLocator const &loc : set) {
        if (!first) {
            out += ", ";
        }
        first = false;
        out += loc.GetString();
    }
    return out + "}";
}

HdDataSourceLocator Pv(char const *name)
{
    return HdDataSourceLocator(TfToken("primvars"), TfToken(name),
                               TfToken("primvarValue"));
}

HdDataSourceLocator const kExtentMin(TfToken("extent"), TfToken("min"));
HdDataSourceLocator const kExtentMax(TfToken("extent"), TfToken("max"));
HdDataSourceLocator const kVisibility(TfToken("visibility"),
                                      TfToken("visibility"));

struct SubdivisionPoint {
    GfVec3f p = GfVec3f(0.0f);
    void Clear() { p = GfVec3f(0.0f); }
    void AddWithWeight(SubdivisionPoint const &other, float weight)
    { p += other.p * weight; }
};

void CheckSubdivision(usdGenPomade::PomadeSubdivisionCage const &cage,
                      int bevelVertex, int columns, bool straightProfile = false)
{
    bool straightSpans = true;
    for (size_t row = 0; row < cage.points.size(); row += size_t(columns)) {
        for (int side = 0; side < columns; side += 6) {
            GfVec3f const a = cage.points[row + size_t(side)];
            GfVec3f const edge = cage.points[row + size_t((side + 6) % columns)] - a;
            float const lengthSq = edge.GetLengthSq();
            float previousFraction = 0.0f;
            for (int span = 1; span < 6; ++span) {
                GfVec3f const offset = cage.points[row + size_t(side + span)] - a;
                float const fraction = GfDot(offset, edge) / lengthSq;
                straightSpans &= (offset - edge * fraction).GetLength() < 1e-6f &&
                                 fraction > previousFraction && fraction < 1.0f;
                previousFraction = fraction;
            }
        }
    }
    Check(straightSpans,
          "all cross-section spans stay ordered on straight sides between authored CV rails");
    namespace Far = OpenSubdiv::Far;
    namespace Sdc = OpenSubdiv::Sdc;
    Far::TopologyDescriptor desc;
    desc.numVertices = int(cage.points.size());
    desc.numFaces = int(cage.faceVertexCounts.size());
    desc.numVertsPerFace = cage.faceVertexCounts.cdata();
    desc.vertIndicesPerFace = cage.faceVertexIndices.cdata();
    desc.numCorners = int(cage.cornerIndices.size());
    desc.cornerVertexIndices = cage.cornerIndices.cdata();
    desc.cornerWeights = cage.cornerSharpnesses.cdata();
    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER);
    using Factory = Far::TopologyRefinerFactory<Far::TopologyDescriptor>;
    std::unique_ptr<Far::TopologyRefiner> refiner(
        Factory::Create(desc, Factory::Options(Sdc::SCHEME_CATMARK, options)));
    Check(bool(refiner), "support cage is valid Catmull-Clark topology");
    if (!refiner) return;
    refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(3));
    std::vector<SubdivisionPoint> previous(cage.points.size());
    for (size_t i = 0; i < previous.size(); ++i) previous[i].p = cage.points[i];
    std::vector<int> held(cage.cornerIndices.begin(), cage.cornerIndices.end());
    GfVec3f const corner = cage.points[size_t(bevelVertex)];
    int sideVertex = bevelVertex + 3;
    GfVec3f const chordMidpoint =
        (corner + cage.points[size_t(bevelVertex + 6)]) * 0.5f;
    bool pinned = true;
    for (int level = 1; level <= 3; ++level) {
        std::vector<SubdivisionPoint> next(size_t(refiner->GetLevel(level).GetNumVertices()));
        SubdivisionPoint *destination = next.data();
        Far::PrimvarRefiner(*refiner).Interpolate(level, previous.data(), destination);
        auto const &parent = refiner->GetLevel(level - 1);
        for (size_t i = 0; i < held.size(); ++i) {
            held[i] = parent.GetVertexChildVertex(held[i]);
            pinned &= (next[size_t(held[i])].p -
                cage.points[size_t(cage.cornerIndices[i])]).GetLength() < 1e-7f;
        }
        bevelVertex = parent.GetVertexChildVertex(bevelVertex);
        sideVertex = parent.GetVertexChildVertex(sideVertex);
        previous = std::move(next);
    }
    Check(pinned, "three actual subdivision levels preserve every root/tip boundary vertex");
    float const rounding = (previous[size_t(bevelVertex)].p - corner).GetLength();
    Check(rounding > 1e-5f && rounding < 0.03f,
          "longitudinal support edges allow a small rounded bevel without collapsing the profile");
    if (straightProfile)
        Check((previous[size_t(sideVertex)].p - chordMidpoint).GetLength() < 1e-6f,
              "actual Catmull-Clark subdivision keeps the broad straight side flat");
}

} // namespace

int
main(int argc, char **argv)
{
    {
        using namespace usdGenPomade;
        PomadeModel model;
        Check(model.GetDisplaySegments() == 8,
              "new models reveal the cubic profile by default");
        PomadeTubeShape shape;
        shape.rings = 4;
        Check(model.BuildTestTube(shape) && model.ScaleSectionRing(1, 0.3f) &&
                  model.ScaleSectionRing(3, 0.3f),
              "four-ring profile builds through model editing");
        PomadePublisher publisher;
        Check(publisher.Stage(model), "four-ring profile publishes");
        auto const &level = publisher.Current().levels.at(1);
        Check(level.points.size() == 25 * 8,
              "the sampled profile retains its authored CV indexing");
        auto const &cage = level.subdivision;
        Check(cage.points.size() == 27 * 48 &&
                  cage.faceVertexCounts.size() == 26 * 48 &&
                  cage.cornerIndices.size() == 96,
              "profile cage adds holding rails, four spans across each side and two end holding rows");
        Check((cage.points[1] - (level.points[0] * 0.95f +
                                level.points[1] * 0.05f)).GetLength() < 1e-7f &&
                  (cage.points[5] - (level.points[0] * 0.05f +
                                     level.points[1] * 0.95f)).GetLength() < 1e-7f,
              "support rails sit tightly on both sides of each corner");
        CheckSubdivision(cage, 13 * 48, 48);
    }
    {
        using namespace usdGenPomade;
        PomadeModel model;
        PomadeTubeShape shape;
        shape.rings = 4;
        shape.ringVerts = 4;
        bool built = model.BuildTestTube(shape);
        for (int ring = 0; ring < 4; ++ring)
            built &= model.TwistSectionRing(ring, 0.78539816339f);
        PomadePublisher publisher;
        Check(built && publisher.Stage(model), "an aligned square tube builds for side interpolation");
        auto const &level = publisher.Current().levels.at(1);
        float rawMaxX = -1e30f;
        for (auto const &p : level.points) rawMaxX = std::max(rawMaxX, p[0]);
        bool bounded = true;
        for (auto const &p : level.subdivision.points) {
            for (int axis = 0; axis < 3; ++axis)
                bounded &= p[axis] >= level.extentMin[axis] &&
                           p[axis] <= level.extentMax[axis] && std::isfinite(p[axis]);
        }
        Check(bounded && std::abs(level.extentMax[0] - rawMaxX) < 1e-7f,
              "added spans stay inside the authored square without an outward bulge");
        CheckSubdivision(level.subdivision, 13 * 24, 24, true);
    }
    // The default (and USDGENPOMADE_TEST_TUBE=0, what record_usd.ps1 sets):
    // the constructor publishes nothing — no tube under the root, GetPrim
    // empty. The ctest entry testUsdGenPomadeIndexNoTestTube drives this
    // branch. The ambient variable is cleared first so this pins the
    // default itself, not merely the explicit-off mode; the setting is
    // first read below, so nothing has cached it yet. The tool path still
    // works: create a model, activate it, and the level prims appear.
    if (argc > 1 && std::string(argv[1]) == "--no-test-tube") {
#ifdef _WIN32
        _putenv("USDGENPOMADE_TEST_TUBE=");
#else
        unsetenv("USDGENPOMADE_TEST_TUBE");
#endif
        HdRetainedSceneIndexRefPtr input = HdRetainedSceneIndex::New();
        HdSceneIndexBaseRefPtr pomade = UsdGenPomadeSceneIndex::New(input);
        SdfPath const root = UsdGenPomadeSceneIndex::RootPath();
        SdfPath const tube = UsdGenPomadeSceneIndex::TestTubePath();
        Check(!HasChild(*pomade, root, tube),
              "no-test-tube: testTube not announced under root");
        HdSceneIndexPrim tubePrim = pomade->GetPrim(tube);
        Check(tubePrim.primType.IsEmpty() && !tubePrim.dataSource,
              "no-test-tube: GetPrim(testTube) is empty");
        PomadeModelContext *ctx = nullptr;
        Check(Pomade_Create(&ctx) == POMADE_OK && ctx,
              "no-test-tube: model creates");
        Check(Pomade_BuildTestTube(ctx, 0, 0, 0.0f, 0.0f) == POMADE_OK,
              "no-test-tube: tube builds");
        Check(Pomade_Activate(ctx) == POMADE_OK, "no-test-tube: activate");
        Check(HasChild(*pomade, UsdGenPomadeSceneIndex::TubesScopePath(),
                       UsdGenPomadeSceneIndex::TubesPath(1)),
              "no-test-tube: tubes/L1 announced after activate");
        Check(!HasChild(*pomade, root, tube),
              "no-test-tube: the test tube never appears");
        Pomade_Destroy(ctx);
        std::printf("%d failure(s)\n", g_failures);
        return g_failures ? 1 : 0;
    }

    usdGenPomade::PomadeRegistry &registry = usdGenPomade::PomadeRegistry::Get();
    SdfPath const root = UsdGenPomadeSceneIndex::RootPath();
    SdfPath const testTube = UsdGenPomadeSceneIndex::TestTubePath();
    SdfPath const tubesL1 = UsdGenPomadeSceneIndex::TubesPath(1);
    SdfPath const tubesL2 = UsdGenPomadeSceneIndex::TubesPath(2);
    SdfPath const centersL1 = UsdGenPomadeSceneIndex::CentersPath(1);
    SdfPath const centerCVsL1 = UsdGenPomadeSceneIndex::CenterCVsPath(1);
    SdfPath const ringsL1 = UsdGenPomadeSceneIndex::RingsPath(1);
    SdfPath const ringCVsL1 = UsdGenPomadeSceneIndex::RingCVsPath(1);
    SdfPath const centersL2 = UsdGenPomadeSceneIndex::CentersPath(2);
    SdfPath const centerCVsL2 = UsdGenPomadeSceneIndex::CenterCVsPath(2);
    SdfPath const ringsL2 = UsdGenPomadeSceneIndex::RingsPath(2);
    SdfPath const ringCVsL2 = UsdGenPomadeSceneIndex::RingCVsPath(2);
    SdfPath const tubesL3 = UsdGenPomadeSceneIndex::TubesPath(3);
    SdfPath const centersL3 = UsdGenPomadeSceneIndex::CentersPath(3);
    SdfPath const ringCVsL3 = UsdGenPomadeSceneIndex::RingCVsPath(3);

    Check(registry.IndexCount() == 0, "registry starts with no index");

    HdRetainedSceneIndexRefPtr input = HdRetainedSceneIndex::New();
    HdSceneIndexBaseRefPtr pomade = UsdGenPomadeSceneIndex::New(input);
    UsdGenPomadeSceneIndex *pomadeRaw =
        dynamic_cast<UsdGenPomadeSceneIndex *>(pomade.operator->());
    Check(pomadeRaw != nullptr, "pomade index constructs over empty input");
    if (!pomadeRaw) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    Check(registry.IndexCount() == 1,
          "the index attaches to the registry on construction");

    // -- the test tube, published while no model is active -----------------
    Check(root == SdfPath("/__usdGenPomade"), "root path is /__usdGenPomade");
    Check(HasChild(*pomade, SdfPath::AbsoluteRootPath(), root),
          "root announced under /");
    Check(HasChild(*pomade, root, testTube), "testTube announced under root");
    Check(pomadeRaw->HasTestTube(), "the index reports the test tube");
    {
        HdSceneIndexPrim prim = pomade->GetPrim(testTube);
        Check(prim.primType == TfToken("mesh"), "test tube primType is mesh");
        VtVec3fArray points;
        Check(ArraySize(*pomade, testTube, "points", 40, &points),
              "test tube has 5 x 8 = 40 points");
        if (points.size() == 40) {
            // Ring 0, slot 0: angle 0 -> (0.5, 0, 0). Ring 4, slot 4
            // (index 4*8+4 = 36): angle pi -> (-0.5, 4, 0).
            Check(std::abs(points[0][0] - 0.5f) < 1e-6f &&
                      std::abs(points[0][1]) < 1e-6f &&
                      std::abs(points[0][2]) < 1e-6f,
                  "ring-0 slot-0 point is (0.5, 0, 0)");
            Check(std::abs(points[36][0] + 0.5f) < 1e-5f &&
                      std::abs(points[36][1] - 4.0f) < 1e-6f &&
                      std::abs(points[36][2]) < 1e-5f,
                  "ring-4 slot-4 point is (-0.5, 4, 0)");
        }
        VtVec3fArray normals;
        Check(ArraySize(*pomade, testTube, "normals", 128, &normals),
              "test tube has one normal per face corner");
        auto interpolation = SampledAt(prim.dataSource,
            HdDataSourceLocator(TfToken("primvars"), TfToken("normals"),
                                TfToken("interpolation")));
        Check(interpolation && interpolation->GetValue(0.0f) ==
                  VtValue(TfToken("faceVarying")),
              "tube normals are face-varying in Hydra");
        if (normals.size() == 128) {
            Check(GfDot(normals[1], normals[4]) < 0.9f &&
                      GfDot(normals[0], normals[29]) < 0.9f,
                  "adjacent CV rails are hard, including the wrap seam");
            Check(normals[3] == normals[32] && normals[2] == normals[33],
                  "normals are shared along the longitudinal strip");
        }
        // Uniform means per face: 32 entries, not the single-element
        // `constant` the P3 index published.
        Check(ArraySize<VtIntArray>(*pomade, testTube, "tubeId", 32),
              "test tube tubeId is uniform, one per face");
        Check(ArraySize<VtVec3fArray>(*pomade, testTube, "clumpColor", 32),
              "test tube clumpColor is uniform, one per face");
        Check(ArraySize<VtFloatArray>(*pomade, testTube, "selected", 32),
              "test tube selected is uniform, one per face");
        Check(ArraySize<VtIntArray>(*pomade, testTube, "hierarchyLevel", 1),
              "hierarchyLevel is constant");
        HdSampledDataSourceHandle counts =
            SampledAt(prim.dataSource,
                      HdDataSourceLocator(TfToken("mesh"),
                                          TfToken("topology"),
                                          TfToken("faceVertexCounts")));
        Check(counts && counts->GetValue(0.0f)
                            .UncheckedGet<VtIntArray>()
                            .size() == 32,
              "test tube has 32 quad faces");
    }
    Check(HasChild(*pomade, root, UsdGenPomadeSceneIndex::TubeMaterialPath()),
          "tube material announced");
    Check(HasChild(*pomade, root, UsdGenPomadeSceneIndex::OverlayMaterialPath()),
          "overlay material announced");
    Check(pomade->GetPrim(UsdGenPomadeSceneIndex::OverlayMaterialPath())
                  .primType == TfToken("material"),
          "overlay material primType is material");

    // -- activate a model: the test tube goes, the levels arrive -----------
    HdSceneIndexObserverPtr observer(new RecordingObserver());
    pomade->AddObserver(observer);
    RecordingObserver *rec =
        static_cast<RecordingObserver *>(observer.operator->());

    PomadeModelContext *ctx = nullptr;
    Check(Pomade_Create(&ctx) == POMADE_OK && ctx != nullptr, "model creates");
    if (!ctx) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    Check(Pomade_GetModelId(ctx) > 0, "the model registers and gets an id");
    Check(registry.ModelCount() == 1, "the registry holds one model");
    Check(Pomade_BuildTestTube(ctx, 0, 0, 0.0f, 0.0f) == POMADE_OK,
          "the model builds its tube");
    Check(Pomade_Activate(ctx) == POMADE_OK, "activate succeeds");
    Check(registry.GetActiveId() == Pomade_GetModelId(ctx),
          "the registry records the active model");
    Check(rec->WasRemoved(testTube),
          "activating removes the test tube (PrimsRemoved)");
    Check(!pomadeRaw->HasTestTube(), "the test tube is gone");
    Check(rec->WasAdded(tubesL1), "activating adds tubes/L1");
    Check(rec->WasAdded(centersL1) && rec->WasAdded(centerCVsL1) &&
              rec->WasAdded(ringsL1) && rec->WasAdded(ringCVsL1),
          "activating adds the L1 overlays");

    // -- the L1 payload ----------------------------------------------------
    Check(pomade->GetPrim(tubesL1).primType == TfToken("mesh"),
          "tubes/L1 primType is mesh");
    {
        auto mesh = pomade->GetPrim(tubesL1).dataSource;
        auto scheme = SampledAt(mesh, HdDataSourceLocator(
            TfToken("mesh"), TfToken("subdivisionScheme")));
        auto corners = SampledAt(mesh, HdDataSourceLocator(
            TfToken("mesh"), TfToken("subdivisionTags"), TfToken("cornerIndices")));
        Check(scheme && scheme->GetValue(0.0f) == VtValue(TfToken("catmullClark")) &&
                  corners && corners->GetValue(0.0f).Get<VtIntArray>().size() == 96 &&
                  !Primvar(*pomade, tubesL1, "normals"),
              "Hydra receives Catmull-Clark, pinned ends and smooth limit normals");
    }
    Check(ArraySize<VtVec3fArray>(*pomade, tubesL1, "points", 336),
          "tubes/L1 carries 336 support-cage points");
    Check(ArraySize<VtIntArray>(*pomade, tubesL1, "tubeId", 288),
          "tubes/L1 tubeId is uniform, one per face");
    Check(ArraySize<VtVec3fArray>(*pomade, tubesL1, "clumpColor", 288),
          "tubes/L1 clumpColor is uniform, one per face");
    Check(ArraySize<VtFloatArray>(*pomade, tubesL1, "selected", 288),
          "tubes/L1 selected is uniform, one per face");
    {
        VtIntArray level;
        Check(ArraySize(*pomade, tubesL1, "hierarchyLevel", 1, &level) &&
                  level[0] == 1,
              "tubes/L1 hierarchyLevel is a constant 1");
        VtFloatArray xray;
        Check(ArraySize(*pomade, tubesL1, "xray", 1, &xray) &&
                  xray[0] == 0.0f,
              "tubes/L1 xray is a constant 0");
    }
    Check(pomade->GetPrim(centersL1).primType == TfToken("basisCurves"),
          "centers/L1 primType is basisCurves");
    Check(pomade->GetPrim(centerCVsL1).primType == TfToken("points"),
          "centerCVs/L1 primType is points");
    Check(ArraySize<VtVec3fArray>(*pomade, centersL1, "points", 5),
          "centers/L1 has the 5 center CVs");
    Check(ArraySize<VtVec3fArray>(*pomade, centersL1, "displayColor", 1),
          "centers/L1 displayColor is uniform, one per curve");
    Check(ArraySize<VtFloatArray>(*pomade, centersL1, "widths", 1),
          "centers/L1 widths is uniform, one per curve");
    Check(ArraySize<VtVec3fArray>(*pomade, centerCVsL1, "displayColor", 5),
          "centerCVs/L1 displayColor is per vertex");
    // -- V6: the ring visibility rule (plan/18 §2.4a) ----------------------
    //
    // Rings and ring CV dots draw in Tube mode's Ring/Section sub-modes and
    // on selected tubes everywhere else, so the default (Rings_Selected,
    // nothing selected) publishes no ring geometry at all.
    Check(Pomade_GetRingDisplay(ctx) == 1,
          "rings default to 'selected tubes only'");
    Check(ArraySize<VtVec3fArray>(*pomade, ringsL1, "points", 0),
          "rings/L1 is empty while nothing is selected");
    Check(ArraySize<VtVec3fArray>(*pomade, ringCVsL1, "points", 0),
          "ringCVs/L1 is empty while nothing is selected");
    {
        int const tubeZero = 0;
        Check(Pomade_SelectSet(ctx, usdGenPomade::PomadePick_TubeVert,
                              &tubeZero, nullptr, nullptr, 1) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1,
              "selecting the tube publishes its rings");
        Check(ArraySize<VtVec3fArray>(*pomade, ringsL1, "points", 45),
              "rings/L1 has 5 closed rings of 9 CVs once selected");
        Check(Pomade_SelectClear(ctx, 0) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1 &&
                  ArraySize<VtVec3fArray>(*pomade, ringsL1, "points", 0),
              "deselecting takes them away again");
    }
    // Ring mode shows every tube's rings; the rest of this test reads them,
    // so it stays on from here.
    Check(Pomade_SetRingDisplay(ctx, 2) == POMADE_OK &&
              Pomade_Publish(ctx, 0) == 1,
          "Ring sub-mode turns every tube's rings on");
    // 5 rings of 8 verts: 5 closed curves of 9 CVs, 40 ring CV dots.
    Check(ArraySize<VtVec3fArray>(*pomade, ringsL1, "points", 45),
          "rings/L1 has 5 closed rings of 9 CVs");
    Check(ArraySize<VtVec3fArray>(*pomade, ringCVsL1, "points", 40),
          "ringCVs/L1 has the 40 ring CVs");
    {
        int faces = 0, points = 0, tubes = 0;
        Check(Pomade_GetPublishedLevelInfo(ctx, 1, &faces, &points, &tubes) ==
                      POMADE_OK &&
                  faces == 288 && points == 336 && tubes == 1,
              "Pomade_GetPublishedLevelInfo reports L1 as 288/336/1");
        Check(Pomade_GetPublishedLevelInfo(ctx, 7, &faces, &points, &tubes) ==
                  POMADE_ERROR,
              "Pomade_GetPublishedLevelInfo fails for an unpublished level");
    }

    // -- a single-tube move: leaf-exact, one level, one restage ------------
    rec->Clear();
    Check(Pomade_MoveCenterRing(ctx, 2, 1.0f, 0.0f) == POMADE_OK,
          "MoveCenterRing(2, +1 X) applies");
    Check(Pomade_Publish(ctx, 0) == 1, "publish reaches the one index");
    Check(rec->added.empty() && rec->removed.empty(),
          "a move sends no added/removed");
    {
        HdDataSourceLocatorSet const expected{
            Pv("points"), kExtentMin, kExtentMax,
        };
        Check(rec->DirtiedFor(tubesL1) == expected,
              "the move dirties tubes/L1 leaf-exactly: " +
                  ToString(rec->DirtiedFor(tubesL1)));
        HdDataSourceLocatorSet const curveExpected{
            Pv("points"), kExtentMin, kExtentMax,
        };
        Check(rec->DirtiedFor(centersL1) == curveExpected,
              "the move dirties centers/L1 leaf-exactly: " +
                  ToString(rec->DirtiedFor(centersL1)));
        Check(rec->DirtiedFor(ringCVsL1) == curveExpected,
              "the move dirties ringCVs/L1 leaf-exactly");
    }
    Check(pomadeRaw->LastRestagedTubeCount() == 1,
          "the move re-tessellated exactly one tube");
    {
        VtVec3fArray points;
        ArraySize(*pomade, tubesL1, "points", 336, &points);
        Check(points.size() == 336 &&
                  std::abs(points[144][0] - 1.5f) < 1e-5f &&
                  std::abs(points[144][1] - 2.0f) < 1e-6f,
              "ring-2 slot-0 point moved to (1.5, 2, 0)");
    }

    // -- subdivide: the whole L2 family arrives ----------------------------
    rec->Clear();
    int kids[4] = {0, 0, 0, 0};
    int kidCount = 0;
    Check(Pomade_SubdivideTube(ctx, 0, 4, "kmeans", 7, kids, 4, &kidCount) ==
                  POMADE_OK &&
              kidCount == 4,
          "Pomade_SubdivideTube splits tube 0 into four children");
    Check(Pomade_Publish(ctx, 0) == 1, "publish after subdivide");
    Check(rec->WasAdded(tubesL2), "subdivide adds tubes/L2");
    Check(rec->WasAdded(centersL2) && rec->WasAdded(centerCVsL2) &&
              rec->WasAdded(ringsL2) && rec->WasAdded(ringCVsL2),
          "subdivide adds the L2 overlays");
    {
        int faces = 0, points = 0, tubes = 0;
        Check(Pomade_GetPublishedLevelInfo(ctx, 2, &faces, &points, &tubes) ==
                      POMADE_OK &&
                  tubes == 4,
              "L2 publishes the four children in one mesh");
        // The staged grid is the model's own census: (sections - 1) *
        // displaySegments + 1 rows of ringVerts, per child.
        int expectFaces = 0;
        int expectPoints = 0;
        int const segments = Pomade_GetDisplaySegments(ctx);
        for (int i = 0; i < kidCount; ++i) {
            int const sections = Pomade_GetTubeSectionCount(ctx, kids[i]);
            float t = 0.0f, scale = 0.0f, twist = 0.0f;
            float uv[128] = {0.0f};
            int cvCount = 0;
            Pomade_GetTubeSection(ctx, kids[i], 0, &t, uv, 128, &cvCount,
                                 &scale, &twist);
            int const rings = (sections - 1) * segments + 1;
            expectFaces += (rings + 1) * cvCount * 6;
            expectPoints += (rings + 2) * cvCount * 6;
        }
        Check(expectFaces > 0 && faces == expectFaces &&
                  points == expectPoints,
              "L2 stages every child's grid (want " +
                  std::to_string(expectFaces) + "/" +
                  std::to_string(expectPoints) + ", got " +
                  std::to_string(faces) + "/" + std::to_string(points) + ")");
        VtIntArray tubeIds;
        Check(ArraySize(*pomade, tubesL2, "tubeId", size_t(faces), &tubeIds),
              "tubes/L2 tubeId is uniform, one per face");
        bool distinct = false;
        for (size_t i = 1; i < tubeIds.size(); ++i) {
            distinct = distinct || tubeIds[i] != tubeIds[0];
        }
        Check(distinct, "tubes/L2 carries more than one tube id");
        VtIntArray level;
        Check(ArraySize(*pomade, tubesL2, "hierarchyLevel", 1, &level) &&
                  level[0] == 2,
              "tubes/L2 hierarchyLevel is a constant 2");
    }
    Check(ArraySize<VtVec3fArray>(*pomade, centersL2, "displayColor", 4),
          "centers/L2 has one colour per child curve");
    Check(pomadeRaw->PublishedLevels() == std::vector<int>({1, 2}),
          "the index reports exactly the two published levels");

    // -- per-tube slice restage: moving tube 0 leaves L2 alone -------------
    rec->Clear();
    Check(Pomade_MoveCenterRing(ctx, 1, 0.25f, 0.0f) == POMADE_OK,
          "a second move applies");
    Check(Pomade_Publish(ctx, 0) == 1, "publish after the second move");
    Check(pomadeRaw->LastRestagedTubeCount() == 1,
          "with five tubes staged, one move re-tessellates one tube");
    Check(rec->WasDirtied(tubesL1), "the second move dirties L1");
    Check(!rec->WasDirtied(tubesL2), "the second move leaves L2 clean");

    // -- display: x-ray, visibility and focus are separate leaves ----------
    rec->Clear();
    Check(Pomade_SetLevelDisplay(ctx, 2, 1, 1) == POMADE_OK, "L2 x-ray on");
    Check(Pomade_Publish(ctx, 0) == 1, "publish the x-ray change");
    {
        // V9: turning x-ray on also swaps the level onto the translucent
        // twin of the tube material — that is what stops the ghosted tube
        // writing depth over the curves inside it — so the binding is a
        // second leaf, and the only second leaf.
        HdDataSourceLocatorSet const expected{
            Pv("xray"), HdDataSourceLocator(TfToken("materialBindings"))};
        Check(rec->DirtiedFor(tubesL2) == expected,
              "x-ray dirties exactly the xray primvar and the binding: " +
                  ToString(rec->DirtiedFor(tubesL2)));
        Check(!rec->WasDirtied(centersL2),
              "x-ray does not touch the overlays");
        Check(!rec->WasDirtied(tubesL1), "x-ray does not touch L1");
    }
    {
        VtFloatArray xray;
        Check(ArraySize(*pomade, tubesL2, "xray", 1, &xray) &&
                  std::fabs(xray[0] -
                            usdGenPomade::PomadeModel::kDefaultXrayOpacity) <
                      1e-6f,
              "tubes/L2 xray reads the level's alpha, not a 0/1 flag");
        Check(MaterialBindingOf(*pomade, tubesL2) ==
                  UsdGenPomadeSceneIndex::TubeXrayMaterialPath(),
              "an x-rayed level binds the translucent tube material");
        Check(MaterialBindingOf(*pomade, tubesL1) ==
                  UsdGenPomadeSceneIndex::TubeMaterialPath(),
              "an opaque level keeps the opaque tube material");
    }
    rec->Clear();
    Check(Pomade_SetLevelDisplay(ctx, 2, 0, 1) == POMADE_OK, "L2 hidden");
    Check(Pomade_Publish(ctx, 0) == 1, "publish the visibility change");
    {
        HdDataSourceLocatorSet const expected{kVisibility};
        Check(rec->DirtiedFor(tubesL2) == expected,
              "hiding a level dirties exactly visibility: " +
                  ToString(rec->DirtiedFor(tubesL2)));
        Check(rec->DirtiedFor(centersL2) == expected,
              "the level's curves dirty exactly visibility");
        Check(rec->DirtiedFor(ringCVsL2) == expected,
              "the level's CV dots dirty exactly visibility");
        Check(!rec->WasDirtied(tubesL1), "hiding L2 does not touch L1");
    }
    {
        int visible = 1, xray = 0;
        Check(Pomade_GetLevelDisplay(ctx, 2, &visible, &xray) == POMADE_OK &&
                  visible == 0 && xray == 1,
              "Pomade_GetLevelDisplay reads back hidden + x-ray");
        HdSampledDataSourceHandle vis =
            SampledAt(pomade->GetPrim(tubesL2).dataSource, kVisibility);
        Check(vis && !vis->GetValue(0.0f).UncheckedGet<bool>(),
              "tubes/L2 visibility is false");
    }
    rec->Clear();
    float unfocusedWidth = 0.0f;
    {
        VtFloatArray widths;
        ArraySize(*pomade, centersL1, "widths", 1, &widths);
        unfocusedWidth = widths.empty() ? 0.0f : widths[0];
    }
    Check(unfocusedWidth > 0.0f, "an unfocused center curve has a width");
    Check(Pomade_SetFocusLevel(ctx, 1) == POMADE_OK, "focus L1");
    Check(Pomade_GetFocusLevel(ctx) == 1, "focus level reads back");
    Check(Pomade_Publish(ctx, 0) == 1, "publish the focus change");
    {
        HdDataSourceLocatorSet const expected{Pv("widths")};
        Check(rec->DirtiedFor(centersL1) == expected,
              "focus dirties exactly the center widths: " +
                  ToString(rec->DirtiedFor(centersL1)));
        Check(rec->DirtiedFor(centerCVsL1) == expected,
              "focus dirties exactly the CV widths");
        Check(!rec->WasDirtied(tubesL1),
              "focus does not touch the tube mesh");
    }
    {
        VtFloatArray widths;
        ArraySize(*pomade, centersL1, "widths", 1, &widths);
        Check(widths.size() == 1 && widths[0] > unfocusedWidth,
              "the focused level's center curve is the thicker one");
    }
    Check(pomadeRaw->LastRestagedTubeCount() == 0,
          "no display change re-tessellates anything");

    // -- selection drives the `selected` primvars, leaf-exactly ------------
    //
    // A click must not restage a vertex and must not dirty anything but the
    // one leaf that carries it: `selected` on the mesh and the curves,
    // displayColor on the CV dots (a dot shows its state as its colour).
    {
        SdfPath const gizmo = UsdGenPomadeSceneIndex::GizmoPath();
        SdfPath const brush = UsdGenPomadeSceneIndex::BrushRingPath();
        // L2 was left hidden + x-ray by the display block above; put it
        // back so the selection asserts read a normal level.
        Check(Pomade_SetLevelDisplay(ctx, 2, 1, 0) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1,
              "selection: L2 visible again");
        VtVec3fArray l1PaletteBefore;
        Check(ArraySize(*pomade, tubesL1, "clumpColor", 288,
                        &l1PaletteBefore),
              "selection: capture the unselected L1 clump palette");
        rec->Clear();
        int const tube0 = 0;
        Check(Pomade_SelectSet(ctx, usdGenPomade::PomadePick_TubeVert, &tube0,
                              nullptr, nullptr, 1) == POMADE_OK,
              "selection: select tube 0");
        Check(Pomade_Publish(ctx, 0) == 1, "selection: publish the click");
        Check(pomadeRaw->LastRestagedTubeCount() == 0,
              "selection: a click re-tessellates nothing");
        {
            HdDataSourceLocatorSet const expected{Pv("selected")};
            Check(rec->DirtiedFor(tubesL1) == expected,
                  "selection dirties exactly the selected primvar: " +
                      ToString(rec->DirtiedFor(tubesL1)));
            Check(rec->DirtiedFor(centersL1) == expected,
                  "selection dirties exactly `selected` on the centers");
            Check(rec->DirtiedFor(ringsL1) == expected,
                  "selection dirties exactly `selected` on the rings");
        }
        {
            HdDataSourceLocatorSet const expected{Pv("displayColor")};
            Check(rec->DirtiedFor(centerCVsL1) == expected,
                  "selection dirties exactly the CV colours: " +
                      ToString(rec->DirtiedFor(centerCVsL1)));
            Check(rec->DirtiedFor(ringCVsL1) == expected,
                  "selection dirties exactly the ring CV colours");
        }
        Check(!rec->WasDirtied(tubesL2),
              "selecting an L1 tube leaves L2 alone");
        {
            VtFloatArray selected;
            Check(ArraySize(*pomade, tubesL1, "selected", 288, &selected) &&
                      selected[0] == 1.0f && selected[31] == 1.0f,
                  "tubes/L1 `selected` reads 1 on every face of the tube");
            // V9: float, like the mesh's. `selected` is one of the three
            // primvars pomadeTube.glslfx declares, so its published type
            // has to be the one the shader reads on every prim that
            // binds the material, curves included.
            VtFloatArray curveSelected;
            Check(ArraySize(*pomade, centersL1, "selected", 1,
                            &curveSelected) &&
                      curveSelected[0] == 1.0f,
                  "centers/L1 `selected` reads 1 as a float");
        }
        // Hover is state 2, and it wins over selected on the same item.
        rec->Clear();
        Check(Pomade_SetHover(ctx, usdGenPomade::PomadePick_TubeVert, 0, -1,
                             -1) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1,
              "selection: hover the same tube");
        {
            VtFloatArray selected;
            Check(ArraySize(*pomade, tubesL1, "selected", 288, &selected) &&
                      selected[0] == 2.0f,
                  "a hovered tube reads 2, not 1");
            HdDataSourceLocatorSet const expected{Pv("selected")};
            Check(rec->DirtiedFor(tubesL1) == expected,
                  "hover dirties exactly the selected primvar");
        }
        Check(Pomade_SetHover(ctx, 0, -1, -1, -1) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1,
              "selection: clear the hover");
        // A center CV turns white; its neighbours keep the clump colour.
        rec->Clear();
        {
            int const ids[1] = {0};
            int const subs[1] = {2};
            Check(Pomade_SelectSet(ctx, usdGenPomade::PomadePick_CenterCV, ids,
                                  subs, nullptr, 1) == POMADE_OK &&
                      Pomade_Publish(ctx, 0) == 1,
                  "selection: select one center CV");
            VtVec3fArray colors;
        Check(ArraySize(*pomade, centerCVsL1, "displayColor", 5,
                            &colors) &&
                      colors[2] == GfVec3f(1.0f, 1.0f, 1.0f) &&
                      colors[1] != GfVec3f(1.0f, 1.0f, 1.0f),
                  "the selected CV dot is white and its neighbour is not");
        VtFloatArray ownerSelected;
        VtVec3fArray ownerPalette;
        Check(ArraySize(*pomade, tubesL1, "selected", 288, &ownerSelected) &&
                      ownerSelected[0] == 1.0f &&
                      ownerSelected[31] == 1.0f &&
                      ArraySize(*pomade, tubesL1, "clumpColor", 288,
                                &ownerPalette) &&
                      ownerPalette == l1PaletteBefore,
                  "a selected center CV highlights its owner tube without "
                  "replacing the clump palette");
        Check(Pomade_SetHover(ctx, usdGenPomade::PomadePick_CenterCV, 0, 2,
                             -1) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1 &&
                  ArraySize(*pomade, tubesL1, "selected", 288,
                            &ownerSelected) &&
                  ownerSelected[0] == 2.0f && ownerSelected[31] == 2.0f,
              "hovering a center CV gives its owner tube the hover cue");
        Check(Pomade_SetHover(ctx, 0, -1, -1, -1) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1,
              "selection: clear the center-CV hover");
        }
        // A level selection is every tube at that level and nothing else.
        rec->Clear();
        {
            Check(Pomade_SelectClear(ctx, 0) == POMADE_OK,
                  "selection: start the level test from nothing selected");
            int const level2[1] = {2};
            Check(Pomade_SelectSet(ctx, usdGenPomade::PomadePick_Level, level2,
                                  nullptr, nullptr, 1) == POMADE_OK &&
                      Pomade_Publish(ctx, 0) == 1,
                  "selection: select hierarchy level 2");
            int faces = 0;
            Check(Pomade_GetPublishedLevelInfo(ctx, 2, &faces, nullptr,
                                              nullptr) == POMADE_OK &&
                      faces > 0,
                  "selection: L2 reports its face count");
            VtFloatArray l2;
            Check(ArraySize(*pomade, tubesL2, "selected", size_t(faces),
                            &l2) && l2[0] == 1.0f &&
                      l2[size_t(faces) - 1] == 1.0f,
                  "every face of every L2 tube reads selected");
            VtFloatArray l1;
            Check(ArraySize(*pomade, tubesL1, "selected", 288, &l1) &&
                      l1[0] == 0.0f,
                  "the L1 tube is not selected by a level-2 selection");
        }
        Check(Pomade_SelectClear(ctx, 0) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1,
              "selection: clear everything");
        {
            VtFloatArray selected;
            Check(ArraySize(*pomade, tubesL1, "selected", 288, &selected) &&
                      selected[0] == 0.0f,
                  "clearing the selection puts `selected` back to 0");
        }

        // -- the gizmo and the brush ring appear and disappear -------------
        rec->Clear();
        Check(!HasChild(*pomade, root, gizmo),
              "no gizmo prim while the model holds no gizmo record");
        float const origin[3] = {0.0f, 1.0f, 0.0f};
        float const frame[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f};
        Check(Pomade_SetGizmo(ctx, 1, origin, frame, 1.0f, 1) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1,
              "publish a translate gizmo");
        Check(rec->WasAdded(gizmo), "the gizmo prim arrives");
        Check(pomade->GetPrim(gizmo).primType == TfToken("basisCurves"),
              "the gizmo is basisCurves");
        Check(ArraySize<VtVec3fArray>(*pomade, gizmo, "points", 50),
              "a translate gizmo has Maya axes, arrows and move squares");
        {
            VtIntArray handles;
            VtIntArray active;
            bool activeMatches = false;
            if (ArraySize(*pomade, gizmo, "handleId", 19, &handles) &&
                ArraySize(*pomade, gizmo, "active", 19, &active)) {
                activeMatches = true;
                for (size_t i = 0; i < handles.size(); ++i) {
                    bool const want = handles[i] == 1;
                    activeMatches = activeMatches &&
                        (active[i] == (want ? 1 : 0));
                }
            }
            Check(activeMatches,
                  "every curve of the selected gizmo handle is flagged");
        }
        // Moving it dirties its own leaves and nothing else.
        rec->Clear();
        float const moved[3] = {0.5f, 1.0f, 0.0f};
        Check(Pomade_SetGizmo(ctx, 1, moved, frame, 1.0f, 1) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1,
              "move the gizmo");
        {
            HdDataSourceLocatorSet const expected{
                Pv("points"),  Pv("displayColor"), Pv("widths"),
                Pv("active"),  kExtentMin,         kExtentMax,
            };
            Check(rec->DirtiedFor(gizmo) == expected,
                  "a gizmo move dirties exactly its own leaves: " +
                      ToString(rec->DirtiedFor(gizmo)));
            Check(!rec->WasDirtied(tubesL1),
                  "a gizmo move does not touch the geometry");
        }
        Check(pomadeRaw->LastRestagedTubeCount() == 0,
              "a gizmo move re-tessellates nothing");
        // The brush ring is its own prim with its own life.
        rec->Clear();
        float const center[3] = {0.0f, 2.0f, 0.0f};
        float const normal[3] = {0.0f, 1.0f, 0.0f};
        Check(Pomade_SetBrushRing(ctx, center, normal, 0.4f) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1,
              "publish a brush ring");
        Check(rec->WasAdded(brush), "the brush ring prim arrives");
        Check(ArraySize<VtVec3fArray>(
                  *pomade, brush, "points",
                  size_t(usdGenPomade::PomadeGizmoCircleSegments()) + 1),
              "the brush ring is one closed circle");
        Check(!rec->WasDirtied(gizmo),
              "setting the brush does not dirty the gizmo");
        rec->Clear();
        Check(Pomade_SetBrushRing(ctx, center, normal, 0.0f) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1,
              "clear the brush ring");
        Check(rec->WasRemoved(brush), "the brush ring prim goes away");
        Check(!HasChild(*pomade, root, brush),
              "the brush ring is no longer announced");
        rec->Clear();
        Check(Pomade_SetGizmo(ctx, 0, origin, frame, 1.0f, -1) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1,
              "clear the gizmo");
        Check(rec->WasRemoved(gizmo), "the gizmo prim goes away");
        Check(!HasChild(*pomade, root, gizmo),
              "the gizmo is no longer announced");
    }

    // -- notice builders are pure functions --------------------------------
    {
        Check(UsdGenPomadeSceneIndex::NoticesFor(
                  usdGenPomade::PomadeDirty_Clean)
                  .IsEmpty(),
              "NoticesFor(clean) is empty");
        Check(LocatorSetSize(UsdGenPomadeSceneIndex::NoticesFor(
                  usdGenPomade::PomadeDirty_Points)) == 3,
              "NoticesFor(points) has 3 leaves (subdivision derives normals)");
        Check(LocatorSetSize(UsdGenPomadeSceneIndex::NoticesFor(
                  usdGenPomade::PomadeDirty_Topology)) == 8,
              "NoticesFor(topology) has 8 leaves");
        Check(LocatorSetSize(UsdGenPomadeSceneIndex::NoticesFor(
                  UsdGenPomadeSceneIndex::Dirty_Xray)) == 1,
              "NoticesFor(xray) has 1 leaf");
        Check(LocatorSetSize(UsdGenPomadeSceneIndex::CurveNoticesFor(
                  UsdGenPomadeSceneIndex::Dirty_Widths)) == 1,
              "CurveNoticesFor(widths) has 1 leaf");
        Check(LocatorSetSize(UsdGenPomadeSceneIndex::PointNoticesFor(
                  usdGenPomade::PomadeDirty_Points)) == 3,
              "PointNoticesFor(points) has 3 leaves");
        Check(UsdGenPomadeSceneIndex::GuideNoticesFor(
                  usdGenPomade::PomadeDirty_Points, false)
                  .IsEmpty(),
              "GuideNoticesFor(points) is empty (no guide co-dirty)");
        Check(LocatorSetSize(UsdGenPomadeSceneIndex::GuideNoticesFor(
                  usdGenPomade::PomadeDirty_Guides, false)) == 5,
              "GuideNoticesFor(guides) has 5 leaves");
        Check(LocatorSetSize(UsdGenPomadeSceneIndex::GuideNoticesFor(
                  usdGenPomade::PomadeDirty_Guides, true)) == 6,
              "GuideNoticesFor(guides, countChanged) has 6 leaves");
        Check(LocatorSetSize(UsdGenPomadeSceneIndex::NoticesFor(
                  usdGenPomade::PomadeDirty_Selection)) == 1,
              "NoticesFor(selection) has 1 leaf");
        Check(LocatorSetSize(UsdGenPomadeSceneIndex::CurveNoticesFor(
                  usdGenPomade::PomadeDirty_Selection)) == 1,
              "CurveNoticesFor(selection) has 1 leaf");
        Check(LocatorSetSize(UsdGenPomadeSceneIndex::PointNoticesFor(
                  usdGenPomade::PomadeDirty_Selection)) == 1,
              "PointNoticesFor(selection) has 1 leaf");
        Check(UsdGenPomadeSceneIndex::OverlayNoticesFor(
                  usdGenPomade::PomadeDirty_Points, false)
                  .IsEmpty(),
              "OverlayNoticesFor(points) is empty (no overlay co-dirty)");
        Check(LocatorSetSize(UsdGenPomadeSceneIndex::OverlayNoticesFor(
                  usdGenPomade::PomadeDirty_Gizmo, false)) == 6,
              "OverlayNoticesFor(gizmo) has 6 leaves");
        Check(LocatorSetSize(UsdGenPomadeSceneIndex::OverlayNoticesFor(
                  usdGenPomade::PomadeDirty_Gizmo, true)) == 8,
              "OverlayNoticesFor(gizmo, countChanged) has 8 leaves");
    }

    // -- the guide preview publishes per level -----------------------------
    {
        SdfPath const guidesL1 = UsdGenPomadeSceneIndex::GuidesPath(1);
        SdfPath const guidesL2 = UsdGenPomadeSceneIndex::GuidesPath(2);
        Check(!HasChild(*pomade, UsdGenPomadeSceneIndex::GuidesScopePath(),
                        guidesL1) &&
                  !HasChild(*pomade, UsdGenPomadeSceneIndex::GuidesScopePath(),
                            guidesL2),
              "no guides anywhere before the first refill");
        // Tube 0 is suspended (four children), so the global fill runs
        // nowhere: each leaf fills from its own params and the four sets
        // merge at the children's level.
        Check(Pomade_SetFillParams(ctx, 16.0f, 8, 7, 0.0f, nullptr, 0) ==
                  POMADE_OK,
              "fill params set");
        bool leavesSet = (kidCount == 4);
        for (int k = 0; leavesSet && k < kidCount; ++k) {
            leavesSet =
                Pomade_SetTubeFillParams(ctx, kids[k], 16.0f, 8, 7, 0.0f,
                                        nullptr, 0) == POMADE_OK;
        }
        Check(leavesSet, "each leaf takes its own fill params");
        Check(Pomade_RefillGuides(ctx, 1.0f) == POMADE_OK,
              "full-density refill runs");
        rec->Clear();
        Check(Pomade_Publish(ctx, 0) == 1, "publish the refill");
        Check(rec->WasAdded(guidesL2), "the refill adds guides/L2");
        Check(!HasChild(*pomade, UsdGenPomadeSceneIndex::GuidesScopePath(),
                        guidesL1),
              "and the suspended parent stages no guides/L1");
        Check(pomade->GetPrim(guidesL2).primType == TfToken("basisCurves"),
              "guides/L2 primType is basisCurves");
        Check(ArraySize<VtVec3fArray>(*pomade, guidesL2, "points", 512),
              "guides/L2 has 4 leaves x 16 guides x 8 CVs");
        Check(ArraySize<VtFloatArray>(*pomade, guidesL2, "widths", 512),
              "guides/L2 has 512 widths");
        VtFloatArray hairT;
        Check(ArraySize(*pomade, guidesL2, "hairT", 512, &hairT) &&
                  std::abs(hairT[0]) < 1e-6f &&
                  std::abs(hairT[7] - 1.0f) < 1e-6f,
              "guides/L2 hairT ramps 0 -> 1 per guide");
        Check(ArraySize<VtVec3fArray>(*pomade, guidesL2, "displayColor", 64),
              "guides/L2 displayColor is uniform, one per curve");
        rec->Clear();
        Check(Pomade_RefillGuides(ctx, 1.0f) == POMADE_OK,
              "same-density refill runs");
        Check(Pomade_Publish(ctx, 0) == 1, "publish the same-count refill");
        HdDataSourceLocatorSet const expected{
            Pv("points"), Pv("widths"), Pv("hairT"), kExtentMin, kExtentMax,
        };
        Check(rec->DirtiedFor(guidesL2) == expected,
              "a same-count refill dirties the guide leaves exactly: " +
                  ToString(rec->DirtiedFor(guidesL2)));
    }

    // -- tubes use the graph's region colour, in Hydra's linear space ------
    {
        Check(usdGenPomade::PomadeClumpPaletteSize() == 16,
              "the fallback clump palette holds 16 entries");
        bool matchesGraph = true;
        for (int region = 0; region < 32; ++region) {
            float srgb[3];
            usdGenPomade::PomadeRegionColor(region, srgb);
            usdGenPomade::PomadeRgb const color =
                usdGenPomade::PomadeClumpColor(region, 1, -1);
            float const linear[3] = {color.r, color.g, color.b};
            for (int channel = 0; channel < 3; ++channel) {
                // Round trip the published linear value to graph sRGB.
                float const v = linear[channel];
                float const encoded = v <= 0.0031308f ? v * 12.92f
                    : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
                matchesGraph &= std::fabs(encoded - srgb[channel]) < 1e-6f;
            }
        }
        Check(matchesGraph,
              "tube colours match graph region colours, including ids above 15");
        auto const fallback = usdGenPomade::PomadeClumpColor(-1, 1, -1);
        auto const slot15 = usdGenPomade::PomadeClumpPaletteEntry(15);
        Check(fallback.r == slot15.r && fallback.g == slot15.g &&
                  fallback.b == slot15.b,
              "unrooted tubes retain their fallback colour");
        usdGenPomade::PomadeRgb const child0 =
            usdGenPomade::PomadeClumpColor(3, 2, 0);
        usdGenPomade::PomadeRgb const child1 =
            usdGenPomade::PomadeClumpColor(3, 2, 1);
        usdGenPomade::PomadeRgb const parent =
            usdGenPomade::PomadeClumpColor(3, 1, -1);
        Check(child0.r > parent.r && child1.r < parent.r,
              "children step lighter and darker around the parent hue");
    }

    // -- V9: the clump HUE FAMILY (plan/18 section 2.4a) -------------------
    //
    // "Children of one L1 tube take the parent's hue with +/- lightness
    // steps per childIndex so a lock stays one hue family." Two claims,
    // both checkable: siblings must be visibly different, and they must be
    // the same hue. The step V8 shipped (0.12/(level-1)) satisfied only the
    // first, and only on paper: +/-6 % at L3 is below the spread the
    // shader's own key/fill rig puts across one curved tube.
    {
        // Hue as the standard max/min formula, in [0, 6).
        auto hue = [](usdGenPomade::PomadeRgb const &c) {
            float const mx = std::max(c.r, std::max(c.g, c.b));
            float const mn = std::min(c.r, std::min(c.g, c.b));
            float const d = mx - mn;
            if (d <= 0.0f) {
                return 0.0f;
            }
            float h = 0.0f;
            if (mx == c.r) {
                h = (c.g - c.b) / d;
            } else if (mx == c.g) {
                h = 2.0f + (c.b - c.r) / d;
            } else {
                h = 4.0f + (c.r - c.g) / d;
            }
            return h < 0.0f ? h + 6.0f : h;
        };
        auto luma = [](usdGenPomade::PomadeRgb const &c) {
            return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
        };
        bool sameHue = true;
        bool separated = true;
        bool withinBand = true;
        usdGenPomade::PomadeRgb const base =
            usdGenPomade::PomadeClumpColor(3, 1, -1);
        float const baseHue = hue(base);
        for (int level = 2; level <= 4; ++level) {
            for (int child = 0; child < 4; ++child) {
                usdGenPomade::PomadeRgb const c =
                    usdGenPomade::PomadeClumpColor(3, level, child);
                if (std::fabs(hue(c) - baseHue) > 1e-4f) {
                    sameHue = false;
                }
                // Every sibling pair has to separate; the neighbour one
                // index away is the hardest case.
                usdGenPomade::PomadeRgb const other =
                    usdGenPomade::PomadeClumpColor(3, level, child + 1);
                if (std::fabs(luma(c) - luma(other)) < 0.02f) {
                    separated = false;
                }
                if (std::fabs(luma(c) - luma(base)) > 0.45f) {
                    withinBand = false;
                }
            }
        }
        Check(sameHue, "siblings at every level keep their L1 ancestor's hue");
        Check(separated,
              "neighbouring siblings differ in luma by more than 2 %");
        Check(withinBand,
              "a sibling stays inside its ancestor's lightness band");
        Check(usdGenPomade::PomadeClumpColor(3, 2, -1).r == base.r,
              "a tube with no childIndex is the plain palette entry");
    }

    // -- V9: THE display policy (plan/18 section 2.4a) ---------------------
    //
    // The table itself, asserted row by row. It is a pure function, so this
    // is a T0 inside the T1 binary: no model, no index, no publish.
    {
        using usdGenPomade::PomadeModel;
        auto policy = [](char const *mode, char const *sub, int level,
                         int focus) {
            int const id = usdGenPomade::PomadeDisplayModeFromNames(mode, sub);
            return usdGenPomade::PomadePolicyLevelDisplay(id, level, focus);
        };
        auto rings = [](char const *mode, char const *sub) {
            return usdGenPomade::PomadePolicyRingDisplay(
                usdGenPomade::PomadeDisplayModeFromNames(mode, sub));
        };
        Check(usdGenPomade::PomadeDisplayModeFromNames("nonsense", "") < 0,
              "an unknown mode is rejected, not guessed");
        Check(usdGenPomade::PomadeDisplayModeFromNames("tube", "ring") ==
                      usdGenPomade::PomadeDisplayMode_TubeRing &&
                  usdGenPomade::PomadeDisplayModeFromNames("tube", "section") ==
                      usdGenPomade::PomadeDisplayMode_TubeRing &&
                  usdGenPomade::PomadeDisplayModeFromNames("tube", "center") ==
                      usdGenPomade::PomadeDisplayMode_TubeCenter,
              "Ring and Section share Tube's ring row, Center does not");
        Check(usdGenPomade::PomadeDisplayModeFromNames("tube", "tube") ==
                      usdGenPomade::PomadeDisplayMode_TubeObject &&
                  usdGenPomade::PomadeDisplayMode_TubeObject == 7,
              "Whole Tube has an appended display row without renumbering");

        // Graph hides authoring helpers behind the region surface.
        {
            char const *mode = "graph";
            PomadeModel::LevelDisplay const focused = policy(mode, "", 2, 2);
            PomadeModel::LevelDisplay const other = policy(mode, "", 1, 2);
            Check(!focused.xray && !other.xray,
                  std::string(mode) + " keeps every level opaque");
            Check(!focused.centers && !other.centers,
                  std::string(mode) + " draws no centers or CV dots");
            Check(rings(mode, "") == PomadeModel::Rings_Off,
                  std::string(mode) + " draws no rings");
        }

        // Output shows committed amplified tiles from the stage, not the
        // authoring tube/cage/preview geometry that would occlude them.
        // `visible` affects only Pomade's helper families; it does not hide
        // the external amplified tile prims.
        {
            char const *mode = "output";
            PomadeModel::LevelDisplay const focused = policy(mode, "", 2, 2);
            PomadeModel::LevelDisplay const other = policy(mode, "", 1, 2);
            Check(!focused.visible && !other.visible &&
                      !focused.xray && !other.xray,
                  "Output hides opaque authoring tube surfaces at every level");
            Check(!focused.centers && !other.centers &&
                      !focused.guides && !other.guides,
                  "Output hides the cage and interactive guide preview");
            Check(rings(mode, "") == PomadeModel::Rings_Off,
                  "Output draws no section rings");
        }

        {
            PomadeModel::LevelDisplay const focused =
                policy("tube", "tube", 2, 2);
            PomadeModel::LevelDisplay const other =
                policy("tube", "tube", 1, 2);
            Check(!focused.xray && other.xray &&
                      !focused.centers && !other.centers &&
                      !focused.guides && !other.guides &&
                      rings("tube", "tube") == PomadeModel::Rings_Off,
                  "Whole Tube is an opaque body-only selection display");
        }

        // Tube / Ring: the focused level stays opaque (its rings lie on the
        // surface), everything behind it ghosts.
        {
            PomadeModel::LevelDisplay const focused =
                policy("tube", "ring", 2, 2);
            PomadeModel::LevelDisplay const other =
                policy("tube", "ring", 1, 2);
            Check(!focused.xray && focused.centers,
                  "Tube/Ring keeps the focused level opaque with centers");
            Check(other.xray &&
                      other.xrayOpacity == PomadeModel::kDefaultXrayOpacity,
                  "Tube/Ring x-rays the other levels at 25 %");
            Check(rings("tube", "section") == PomadeModel::Rings_All,
                  "Tube/Section shows every ring");
        }

        // Tube / Center, Hierarchy, Sculpt: the ladder of two x-ray
        // strengths, centers everywhere.
        char const *ladder[3][2] = {
            {"tube", "center"}, {"hierarchy", ""}, {"sculpt", ""}};
        for (auto const &pair : ladder) {
            PomadeModel::LevelDisplay const focused =
                policy(pair[0], pair[1], 2, 2);
            PomadeModel::LevelDisplay const other =
                policy(pair[0], pair[1], 1, 2);
            std::string const what = std::string(pair[0]) + "/" + pair[1];
            Check(focused.xray &&
                      focused.xrayOpacity == PomadeModel::kDefaultXrayOpacity,
                  what + " x-rays the focused level at 25 %");
            Check(other.xray &&
                      other.xrayOpacity == PomadeModel::kFaintXrayOpacity,
                  what + " x-rays the levels behind it at 10 %");
            Check(focused.centers && other.centers,
                  what + " draws the center curves and CV dots");
            Check(rings(pair[0], pair[1]) == PomadeModel::Rings_Selected,
                  what + " rings follow the selection");
        }

        // Fill: every level at the same 25 %, so the guide preview reads
        // from root to tip across the whole groom.
        {
            PomadeModel::LevelDisplay const focused = policy("fill", "", 2, 2);
            PomadeModel::LevelDisplay const other = policy("fill", "", 1, 2);
            Check(focused.xray && other.xray &&
                      focused.xrayOpacity == other.xrayOpacity &&
                      other.xrayOpacity == PomadeModel::kDefaultXrayOpacity,
                  "Fill x-rays every level at the same 25 %");
        }

        // No focus is not "everything is a ghost".
        {
            PomadeModel::LevelDisplay const a = policy("sculpt", "", 1, 0);
            PomadeModel::LevelDisplay const b = policy("sculpt", "", 3, 0);
            Check(a.xrayOpacity == PomadeModel::kDefaultXrayOpacity &&
                      b.xrayOpacity == PomadeModel::kDefaultXrayOpacity,
                  "with no focused level every level reads as the focused one");
        }
        Check(!policy("graph", "", 1, 1).visible,
              "Graph hides tube levels behind the region surface");
    }

    // -- input passthrough --------------------------------------------------
    rec->Clear();
    {
        SdfPath const inputPath("/inputMesh");
        HdRetainedSceneIndex::AddedPrimEntries inputAdded = {
            {inputPath, TfToken("mesh"), HdRetainedContainerDataSource::New()},
        };
        input->AddPrims(inputAdded);
        Check(HasChild(*pomade, SdfPath::AbsoluteRootPath(), inputPath),
              "input prim passes through GetChildPrimPaths");
        Check(pomade->GetPrim(inputPath).primType == TfToken("mesh"),
              "input prim passes through GetPrim");
        Check(HasChild(*pomade, SdfPath::AbsoluteRootPath(), root),
              "pomade root still announced alongside input prims");
        Check(rec->added.size() == 1 && rec->added[0].primPath == inputPath,
              "input PrimsAdded forwards");
    }

    // -- V6 / G7: "show amplified hair" is a real switch -------------------
    //
    // plan/17 §3.2: while the cook's amplified tiles are up, the Pomade guide
    // preview steps aside; while a gesture is live the tiles are hidden
    // (they describe a groom older than the drag) and the preview comes
    // back. Before V6 the flag was written by the panel and read nowhere.
    rec->Clear();
    {
        // The leaves' guides stage at L2 (their own level); the suspended
        // parent stages none.
        SdfPath const guidesLive = UsdGenPomadeSceneIndex::GuidesPath(2);
        // A tile exactly where the groom scene index publishes them.
        SdfPath const tile("/Groom/Hair/__usdGenRender/tile0");
        input->AddPrims({{tile, TfToken("basisCurves"),
                          HdRetainedContainerDataSource::New()}});
        Check(pomadeRaw->AmplifiedTileCount() == 1,
              "G7: the index remembers the amplified tile");

        auto tileVisible = [&]() {
            HdSampledDataSourceHandle vis =
                SampledAt(pomade->GetPrim(tile).dataSource, kVisibility);
            // No opinion at all means the groom index's own visibility
            // stands, which is "visible".
            return !vis || vis->GetValue(0.0f).UncheckedGet<bool>();
        };
        auto guidesVisible = [&]() {
            HdSampledDataSourceHandle vis = SampledAt(
                pomade->GetPrim(guidesLive).dataSource, kVisibility);
            return !vis || vis->GetValue(0.0f).UncheckedGet<bool>();
        };

        Check(Pomade_GetAmplifiedHair(ctx) == 0,
              "G7: amplified hair is off by default");
        Check(!pomadeRaw->AmplifiedTilesVisible() && !tileVisible(),
              "G7: with it off the tiles are hidden");
        Check(guidesVisible(), "G7: and the guide preview draws");

        rec->Clear();
        Check(Pomade_SetAmplifiedHair(ctx, 1) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1,
              "G7: turning amplified hair on publishes");
        Check(pomadeRaw->AmplifiedTilesVisible() && tileVisible(),
              "G7: the tiles come up");
        Check(!guidesVisible(),
              "G7: and the guide preview steps aside for them");
        Check(rec->WasDirtied(tile),
              "G7: the flip dirties the tile, not the whole scene");

        rec->Clear();
        Check(Pomade_BeginGesture(ctx, "drag") == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1,
              "G7: a gesture starts");
        Check(!pomadeRaw->AmplifiedTilesVisible() && !tileVisible(),
              "G7: visibility=false goes over the tiles for the gesture");
        Check(guidesVisible(),
              "G7: and the guide preview is what the artist drags against");
        Check(Pomade_GetAmplifiedHair(ctx) == 1,
              "G7: the artist's own setting is untouched by the gesture");

        rec->Clear();
        Check(Pomade_EndGesture(ctx) == POMADE_OK && Pomade_Publish(ctx, 0) == 1,
              "G7: the gesture ends");
        Check(pomadeRaw->AmplifiedTilesVisible() && tileVisible(),
              "G7: release restores the tiles");
        Check(!guidesVisible(), "G7: and hides the preview again");

        Check(Pomade_SetAmplifiedHair(ctx, 0) == POMADE_OK &&
                  Pomade_Publish(ctx, 0) == 1 && !tileVisible() &&
                  guidesVisible(),
              "G7: turning it back off restores the preview");
        input->RemovePrims({{tile}});
        Check(pomadeRaw->AmplifiedTileCount() == 0,
              "G7: a removed tile is forgotten");
    }

    // -- V8: the scalp tint and the tube agree, and the overlays are
    // -- sized in pixels (plan/18 section 2.4a) ----------------------------
    //
    // A second model, because this one has a test tube and no scalp: the
    // tint only exists over a bound scalp with a rasterised region, and the
    // point of the check is that the region's patch on the head and the
    // tube rooted in that region come out of the palette at the same slot.
    {
        Check(Pomade_Deactivate(ctx) == POMADE_OK, "V8: park the first model");
        PomadeModelContext *sc = nullptr;
        Check(Pomade_Create(&sc) == POMADE_OK && sc != nullptr,
              "V8: the scalp model creates");
        // The 4x4 quad grid in XZ the T3 scripts use: face f = ix * 4 + iz
        // covers x in [ix, ix+1] and z in [iz, iz+1].
        std::vector<float> points;
        for (int ix = 0; ix < 5; ++ix) {
            for (int iz = 0; iz < 5; ++iz) {
                points.push_back(float(ix));
                points.push_back(0.0f);
                points.push_back(float(iz));
            }
        }
        std::vector<int> counts(16, 4);
        std::vector<int> indices;
        for (int ix = 0; ix < 4; ++ix) {
            for (int iz = 0; iz < 4; ++iz) {
                int const a = ix * 5 + iz;
                indices.push_back(a);
                indices.push_back(a + 5);
                indices.push_back(a + 6);
                indices.push_back(a + 1);
            }
        }
        Check(Pomade_BindScalp(sc, points.data(), int(points.size()),
                              counts.data(), int(counts.size()),
                              indices.data(), int(indices.size())) ==
                  POMADE_OK,
              "V8: the 4x4 scalp binds");
        Pomade_SetSnapRadius(sc, 0.1f);
        float const corners[4][2] = {
            {0.0f, 1.0f}, {2.0f, 1.0f}, {2.0f, 3.0f}, {0.0f, 3.0f}};
        std::vector<int> faces;
        std::vector<float> uvs;
        for (int c = 0; c < 4; ++c) {
            for (int k = 0; k < 5; ++k) {
                float const t = float(k) / 5.0f;
                float const x = corners[c][0] +
                                (corners[(c + 1) % 4][0] - corners[c][0]) * t;
                float const z = corners[c][1] +
                                (corners[(c + 1) % 4][1] - corners[c][1]) * t;
                int const ix = std::min(std::max(int(std::floor(x)), 0), 3);
                int const iz = std::min(std::max(int(std::floor(z)), 0), 3);
                faces.push_back(ix * 4 + iz);
                // Quad face coordinates follow the topology winding:
                // u advances from p0 to p1 (X here), v from p0 to p3 (Z).
                // Swapping them makes a multi-face rectangle self-cross at
                // every coarse-face seam, which is invisible to a per-face
                // map but invalid for an exact clipped display contour.
                uvs.push_back(x - float(ix));
                uvs.push_back(z - float(iz));
            }
        }
        faces.push_back(1);
        uvs.push_back(0.0f);
        uvs.push_back(0.0f);
        std::vector<int> chain(256);
        int chainCount = 0, closed = 0, weldStart = 0, weldEnd = 0;
        Check(Pomade_GraphStroke(sc, faces.data(), uvs.data(),
                                int(faces.size()), 0.1f, 0.05f, chain.data(),
                                int(chain.size()), &chainCount, &closed,
                                &weldStart, &weldEnd) == POMADE_OK &&
                  closed == 1,
              "V8: the stroke closes into a region");
        Check(Pomade_Rasterise(sc) == POMADE_OK, "V8: K3 rasterises it");
        Check(Pomade_BuildTubeFromRegion(sc, 0, 5, 8, 3.0f) == POMADE_OK,
              "V8: an L1 tube builds from region 0");
        Check(Pomade_Activate(sc) == POMADE_OK && Pomade_Publish(sc, ~0u) >= 1,
              "V8: the scalp model activates and publishes");

        SdfPath const regions = UsdGenPomadeSceneIndex::GraphRegionsPath();
        SdfPath const nodes = UsdGenPomadeSceneIndex::GraphNodesPath();
        VtVec3fArray faceColor;
        VtIntArray faceRegion;
        VtVec3fArray clump;
        {
            HdSampledDataSourceHandle colors =
                Primvar(*pomade, regions, "displayColor");
            HdSampledDataSourceHandle ids =
                Primvar(*pomade, regions, "usdGen:pomadeRegion");
            VtValue const cv = colors ? colors->GetValue(0.0f) : VtValue();
            VtValue const iv = ids ? ids->GetValue(0.0f) : VtValue();
            if (cv.IsHolding<VtVec3fArray>()) {
                faceColor = cv.UncheckedGet<VtVec3fArray>();
            }
            if (iv.IsHolding<VtIntArray>()) {
                faceRegion = iv.UncheckedGet<VtIntArray>();
            }
        }
        Check(faceColor.size() == faceRegion.size() && faceColor.size() > 16,
              "V8: graph/regions carries sparse clipped patch faces");
        {
            HdSampledDataSourceHandle ds = Primvar(*pomade, tubesL1,
                                                   "clumpColor");
            VtValue v = ds ? ds->GetValue(0.0f) : VtValue();
            Check(v.IsHolding<VtVec3fArray>(),
                  "V8: tubes/L1 carries a clumpColor");
            if (v.IsHolding<VtVec3fArray>()) {
                clump = v.UncheckedGet<VtVec3fArray>();
            }
        }
        // The one assertion this block exists for. Every face the region
        // claims must be painted the colour the tube rooted in that region
        // draws with: one palette, one index, no interp-id detour.
        int covered = 0;
        bool matched = !clump.empty();
        for (size_t f = 0; f < faceRegion.size() && f < faceColor.size();
             ++f) {
            if (faceRegion[f] < 0) {
                continue;
            }
            ++covered;
            if (!clump.empty() &&
                (faceColor[f] - clump[0]).GetLength() > 1e-6f) {
                matched = false;
            }
        }
        Check(covered > 0, "V8: the region claims scalp faces");
        Check(matched,
              "V8: every claimed face is painted the clumpColor of the tube "
              "rooted in its region");
        float graphRgb[3] = {};
        int colorCount = 0;
        bool graphMatched = Pomade_ReadRegionColors(sc, graphRgb, 1,
                                                    &colorCount) == POMADE_OK &&
                            colorCount == 1 && !clump.empty();
        for (int c = 0; graphMatched && c < 3; ++c) {
            float const v = clump[0][c];
            float const encoded = v <= 0.0031308f ? v * 12.92f
                : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
            graphMatched &= std::fabs(encoded - graphRgb[c]) < 1e-6f;
        }
        Check(graphMatched,
              "the published tube and patch use the persisted graph colour");
        {
            // Uncovered faces keep their own warning colour, so the match
            // above is a statement about the region and not about the mesh
            // having one colour.
            bool warned = false;
            for (size_t f = 0; f < faceRegion.size() && f < faceColor.size();
                 ++f) {
                if (faceRegion[f] < 0 && !clump.empty() &&
                    (faceColor[f] - clump[0]).GetLength() > 1e-3f) {
                    warned = true;
                }
            }
            Check(warned, "V8: uncovered faces are not painted the clump "
                          "colour");
        }
        {
            // The tint mesh is lifted off the scalp it copies. Coincident
            // opaque meshes resolve by depth-buffer luck, and the luck ran
            // against the tint: the artist saw an untinted head.
            VtVec3fArray tintPoints;
            HdSampledDataSourceHandle ds = Primvar(*pomade, regions, "points");
            VtValue const value = ds ? ds->GetValue(0.0f) : VtValue();
            if (value.IsHolding<VtVec3fArray>()) {
                tintPoints = value.UncheckedGet<VtVec3fArray>();
            }
            Check(tintPoints.size() > 25,
                  "V8: graph/regions retains the scalp and adds patch points");
            float lift = 0.0f;
            bool everyPointLifted = tintPoints.size() >= 25;
            for (size_t i = 0; i < 25 && i < tintPoints.size(); ++i) {
                GfVec3f const source(points[i * 3 + 0], points[i * 3 + 1],
                                     points[i * 3 + 2]);
                float const d = (tintPoints[i] - source).GetLength();
                lift = std::max(lift, d);
                if (!(d > 0.0f)) {
                    everyPointLifted = false;
                }
            }
            Check(everyPointLifted,
                  "V8: no tint vertex is coincident with the scalp");
            // Small enough to be invisible as parallax (the 4x4 scalp's
            // diagonal is about 5.7 units), large enough to beat depth
            // precision by orders of magnitude.
            Check(lift > 1e-4f && lift < 0.05f,
                  "V8: and the lift stays sub-pixel at any sane framing");
        }

        // -- the display scale sizes the overlays in pixels ---------------
        SdfPath const scCenters = UsdGenPomadeSceneIndex::CentersPath(1);
        SdfPath const scCenterCVs = UsdGenPomadeSceneIndex::CenterCVsPath(1);
        SdfPath const scRingCVs = UsdGenPomadeSceneIndex::RingCVsPath(1);
        auto firstWidth = [&](SdfPath const &path) -> float {
            HdSampledDataSourceHandle ds = Primvar(*pomade, path, "widths");
            if (!ds) {
                return -1.0f;
            }
            VtValue v = ds->GetValue(0.0f);
            if (!v.IsHolding<VtFloatArray>()) {
                return -1.0f;
            }
            VtFloatArray const &a = v.UncheckedGet<VtFloatArray>();
            return a.empty() ? -1.0f : a[0];
        };
        float const fallbackCV = firstWidth(scCenterCVs);
        Check(fallbackCV > 0.0f,
              "V8: with no camera the CV dots keep their radius-relative "
              "width");
        Check(Pomade_SetRingDisplay(sc, 2) == POMADE_OK &&
                  Pomade_Publish(sc, 0) >= 1,
              "V8: rings on, so the ring CV dots publish too");
        float const fallbackRingCV = firstWidth(scRingCVs);
        Check(fallbackRingCV > 0.0f, "V8: and so do their widths");

        float scale = 0.0f;
        Check(Pomade_GetDisplayScale(sc, &scale) == POMADE_OK && scale == 0.0f,
              "V8: a model with no camera reports display scale 0");
        rec->Clear();
        float const perPixel = 0.004f;
        Check(Pomade_SetDisplayScale(sc, perPixel) == POMADE_OK &&
                  Pomade_GetDisplayScale(sc, &scale) == POMADE_OK &&
                  scale == perPixel,
              "V8: Pomade_SetDisplayScale round trips");
        Check(Pomade_Publish(sc, 0) >= 1, "V8: the scale publishes");
        // The focused level is 0 here (nothing focused), so the "else"
        // column of the section 2.4a table is what these must equal.
        using Px = usdGenPomade::PomadeOverlayPixels;
        Check(std::fabs(firstWidth(scCenterCVs) -
                        Px::kCenterCVOther * perPixel) < 1e-6f,
              "V8: centerCVs are 5 px wide at the display scale");
        Check(std::fabs(firstWidth(scRingCVs) - Px::kRingCV * perPixel) <
                  1e-6f,
              "V8: ringCVs are 5 px wide at the display scale");
        Check(std::fabs(firstWidth(scCenters) -
                        Px::kCenterCurveOther * perPixel) < 1e-6f,
              "V8: an unfocused level's center curves are 1 px");
        Check(std::fabs(firstWidth(nodes) - Px::kGraphNode * perPixel) <
                  1e-6f,
              "V8: graph node dots are 6 px");
        Check(rec->WasDirtied(scCenterCVs) && rec->WasDirtied(scCenters),
              "V8: a camera move dirties the overlay widths");
        Check(!rec->WasDirtied(tubesL1) ||
                  rec->DirtiedFor(tubesL1).Intersects(
                      HdDataSourceLocatorSet{Pv("points")}) == false,
              "V8: and re-tessellates nothing");

        Check(Pomade_SetFocusLevel(sc, 1) == POMADE_OK &&
                  Pomade_Publish(sc, 0) >= 1,
              "V8: focusing L1 republishes");
        Check(std::fabs(firstWidth(scCenterCVs) -
                        Px::kCenterCVFocused * perPixel) < 1e-6f,
              "V8: the focused level's CV dots step up to 8 px");
        Check(std::fabs(firstWidth(scCenters) -
                        Px::kCenterCurveFocused * perPixel) < 1e-6f,
              "V8: and its control curves to 3 px");

        // Double the scale (the artist dollies out), halve nothing else.
        Check(Pomade_SetDisplayScale(sc, perPixel * 2.0f) == POMADE_OK &&
                  Pomade_Publish(sc, 0) >= 1,
              "V8: doubling the scale republishes");
        Check(std::fabs(firstWidth(scCenterCVs) -
                        Px::kCenterCVFocused * perPixel * 2.0f) < 1e-6f,
              "V8: the world width follows the scale linearly");

        // Back to the unfocused state the fallback widths were read in:
        // the radius-relative column has its own focused/unfocused split,
        // so comparing across a focus change would prove nothing.
        Check(Pomade_SetFocusLevel(sc, 0) == POMADE_OK &&
                  Pomade_SetDisplayScale(sc, 0.0f) == POMADE_OK &&
                  Pomade_Publish(sc, 0) >= 1 &&
                  std::fabs(firstWidth(scRingCVs) - fallbackRingCV) < 1e-6f &&
                  std::fabs(firstWidth(scCenterCVs) - fallbackCV) < 1e-6f,
              "V8: clearing the scale restores the radius-relative widths");
        Check(Pomade_SetDisplayScale(sc, -1.0f) == POMADE_ERROR,
              "V8: a negative scale is refused");

        Check(Pomade_Deactivate(sc) == POMADE_OK,
              "V8: the scalp model deactivates for the patch regression");
        // Two completed loops fit in one authored quad.  The K3 primvar is
        // necessarily one value for that quad; the display patch mesh must
        // nevertheless expose both region colours.  Each side has an
        // intermediate sample, which also guards ear clipping against the
        // collinear K2 samples a real graph contour carries.
        PomadeModelContext *patchCtx = nullptr;
        Check(Pomade_Create(&patchCtx) == POMADE_OK && patchCtx,
              "V8: the same-face patch model creates");
        if (patchCtx) {
            float const patchPoints[] = {
                0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
            int const patchCounts[] = {4};
            int const patchIndices[] = {0, 1, 2, 3};
            Check(Pomade_BindScalp(patchCtx, patchPoints, 12, patchCounts, 1,
                                  patchIndices, 4) == POMADE_OK &&
                      Pomade_SetSnapRadius(patchCtx, 0.02f) == POMADE_OK,
                  "V8: the one-quad patch scalp binds");
            auto drawPatchLoop = [&](float v0, float v1, bool concave) {
                // (u, v) follows the scalp face convention.  The first
                // contour has a real inward notch; both retain intermediate
                // edge samples, as K2 contours do in production.
                std::vector<float> uvs = concave
                    ? std::vector<float>{
                          0.10f, v0, 0.50f, v0, 0.90f, v0,
                          0.90f, v1, 0.60f, v1, 0.60f,
                          0.5f * (v0 + v1), 0.45f,
                          0.5f * (v0 + v1), 0.45f, v1,
                          0.10f, v1, 0.10f, v0}
                    : std::vector<float>{
                          0.10f, v0, 0.50f, v0, 0.90f, v0,
                          0.90f, 0.5f * (v0 + v1), 0.90f, v1,
                          0.50f, v1, 0.10f, v1,
                          0.10f, 0.5f * (v0 + v1), 0.10f, v0};
                std::vector<int> faces(uvs.size() / 2, 0);
                int chain[64] = {};
                int chainCount = 0, closed = 0, weldStart = 0, weldEnd = 0;
                return Pomade_GraphStroke(patchCtx, faces.data(), uvs.data(),
                                         int(faces.size()),
                                         0.01f, 0.001f, chain, 64,
                                         &chainCount, &closed, &weldStart,
                                         &weldEnd) == POMADE_OK && closed == 1;
            };
            Check(drawPatchLoop(0.10f, 0.42f, true) &&
                      drawPatchLoop(0.58f, 0.90f, false) &&
                      Pomade_Rasterise(patchCtx) == POMADE_OK &&
                      Pomade_Activate(patchCtx) == POMADE_OK &&
                      Pomade_Publish(patchCtx, ~0u) >= 1,
                  "V8: two same-face graph regions publish");
            VtVec3fArray patchColors;
            VtIntArray patchIds;
            {
                HdSampledDataSourceHandle colors =
                    Primvar(*pomade, regions, "displayColor");
                HdSampledDataSourceHandle ids =
                    Primvar(*pomade, regions, "usdGen:pomadeRegion");
                VtValue const cv =
                    colors ? colors->GetValue(0.0f) : VtValue();
                VtValue const iv = ids ? ids->GetValue(0.0f) : VtValue();
                if (cv.IsHolding<VtVec3fArray>()) {
                    patchColors = cv.UncheckedGet<VtVec3fArray>();
                }
                if (iv.IsHolding<VtIntArray>()) {
                    patchIds = iv.UncheckedGet<VtIntArray>();
                }
            }
            std::vector<GfVec3f> distinct;
            for (size_t i = 0; i < patchIds.size() && i < patchColors.size();
                 ++i) {
                if (patchIds[i] < 0) {
                    continue;
                }
                bool seen = false;
                for (GfVec3f const &colour : distinct) {
                    seen = seen ||
                        (colour - patchColors[i]).GetLength() < 1.0e-5f;
                }
                if (!seen) {
                    distinct.push_back(patchColors[i]);
                }
            }
            Check(distinct.size() >= 2,
                  "V8: one scalp quad exposes two distinct region colours");

            // A hierarchy cut is per branch, not a global numeric level.
            // Build two independent L1 roots, expose one root's L2 children,
            // then expand exactly one of those children. The resulting
            // frontier deliberately spans L1 (the unrelated root), L2 (the
            // unexpanded sibling), and L3 (the expanded child's children).
            int cutRootB = -1;
            int cutKids[2] = {-1, -1};
            int cutKidCount = 0;
            int cutGrandkids[2] = {-1, -1};
            int cutGrandkidCount = 0;
            Check(Pomade_BuildTubeFromRegion(patchCtx, 0, 5, 0, 2.0f) ==
                      POMADE_OK &&
                      Pomade_BuildTubeFromRegion(patchCtx, 1, 5, 0, 2.0f) ==
                      POMADE_OK,
                  "cut: two graph regions build independent L1 roots");
            int l1Roots[2] = {-1, -1};
            int l1RootCount = 0;
            bool const readRoots =
                Pomade_ReadL1TubeIds(patchCtx, l1Roots, 2, &l1RootCount) ==
                    POMADE_OK && l1RootCount == 2 && l1Roots[0] == 0;
            if (readRoots) {
                cutRootB = l1Roots[1];
            }
            Check(readRoots && cutRootB >= 0,
                  "cut: both region roots have stable L1 identities");
            Check(Pomade_SubdivideTube(patchCtx, 0, 2, "kmeans", 7, cutKids,
                                      2, &cutKidCount) == POMADE_OK &&
                      cutKidCount == 2 &&
                      Pomade_SubdivideTube(patchCtx, cutKids[0], 2, "kmeans",
                                          7, cutGrandkids, 2,
                                          &cutGrandkidCount) == POMADE_OK &&
                      cutGrandkidCount == 2,
                  "cut: one L1 root has nested L2 and L3 descendants");
            Check(Pomade_SetRingDisplay(patchCtx, 2) == POMADE_OK &&
                      Pomade_SetDisplayScale(patchCtx, 0.004f) == POMADE_OK &&
                      Pomade_SetActiveCutEnabled(patchCtx, 1) == POMADE_OK &&
                      Pomade_Publish(patchCtx, ~0u) >= 1,
                  "cut: enabling Levels publishes its collapsed L1 frontier");
            int faces = 0, points = 0, tubes = 0;
            Check(Pomade_GetPublishedLevelInfo(patchCtx, 1, &faces, &points,
                                              &tubes) == POMADE_OK &&
                      tubes == 2 &&
                      Pomade_GetPublishedLevelInfo(patchCtx, 2, nullptr,
                                                  nullptr, nullptr) ==
                          POMADE_ERROR &&
                      Pomade_IsTubeVisible(patchCtx, 0) == 1 &&
                      Pomade_IsTubeVisible(patchCtx, cutRootB) == 1 &&
                      Pomade_IsTubeVisible(patchCtx, cutKids[0]) == 0,
                  "cut: collapsed roots hide every descendant from staging");
            Check(Pomade_SetTubeExpanded(patchCtx, 0, 1) == POMADE_OK &&
                      Pomade_Publish(patchCtx, ~0u) >= 1,
                  "cut: expanding only root A republishes");
            Check(Pomade_GetPublishedLevelInfo(patchCtx, 1, &faces, &points,
                                              &tubes) == POMADE_OK &&
                      tubes == 1 &&
                      Pomade_GetPublishedLevelInfo(patchCtx, 2, &faces,
                                                  &points, &tubes) == POMADE_OK &&
                      tubes == 2 &&
                      Pomade_GetPublishedLevelInfo(patchCtx, 3, nullptr,
                                                  nullptr, nullptr) ==
                          POMADE_ERROR &&
                      Pomade_IsTubeVisible(patchCtx, 0) == 0 &&
                      Pomade_IsTubeVisible(patchCtx, cutRootB) == 1 &&
                      Pomade_IsTubeVisible(patchCtx, cutKids[0]) == 1,
                  "cut: root A's L2 siblings coexist with unrelated root B");
            Check(Pomade_SetTubeExpanded(patchCtx, cutKids[0], 1) == POMADE_OK &&
                      Pomade_Publish(patchCtx, ~0u) >= 1,
                  "cut: expanding one L2 child republishes mixed levels");
            Check(Pomade_GetPublishedLevelInfo(patchCtx, 1, &faces, &points,
                                              &tubes) == POMADE_OK &&
                      tubes == 1 &&
                      Pomade_GetPublishedLevelInfo(patchCtx, 2, &faces,
                                                  &points, &tubes) == POMADE_OK &&
                      tubes == 1 &&
                      Pomade_GetPublishedLevelInfo(patchCtx, 3, &faces,
                                                  &points, &tubes) == POMADE_OK &&
                      tubes == cutGrandkidCount &&
                      Pomade_IsTubeVisible(patchCtx, cutKids[0]) == 0 &&
                      Pomade_IsTubeVisible(patchCtx, cutKids[1]) == 1 &&
                      Pomade_IsTubeVisible(patchCtx, cutGrandkids[0]) == 1 &&
                      Pomade_IsTubeVisible(patchCtx, cutGrandkids[1]) == 1,
                  "cut: L1, L2 and L3 batches contain only their frontier members");
            auto tubeHasRegionShade = [&](int tubeId, int level,
                                          int regionId, int childIndex) {
                SdfPath const path = UsdGenPomadeSceneIndex::TubesPath(level);
                auto colors = Primvar(*pomade, path, "clumpColor");
                auto ids = Primvar(*pomade, path, "tubeId");
                if (!colors || !ids) {
                    return false;
                }
                VtValue const cv = colors->GetValue(0.0f);
                VtValue const iv = ids->GetValue(0.0f);
                if (!cv.IsHolding<VtVec3fArray>() || !iv.IsHolding<VtIntArray>()) {
                    return false;
                }
                auto const &rgb = cv.UncheckedGet<VtVec3fArray>();
                auto const &tubeIds = iv.UncheckedGet<VtIntArray>();
                if (rgb.size() != tubeIds.size()) {
                    return false;
                }
                auto const shade = usdGenPomade::PomadeClumpColor(
                    regionId, level, childIndex);
                bool found = false;
                for (size_t i = 0; i < tubeIds.size(); ++i) {
                    if (tubeIds[i] == tubeId) {
                        found = true;
                        if ((rgb[i] - GfVec3f(shade.r, shade.g, shade.b))
                                .GetLength() > 1e-6f) {
                            return false;
                        }
                    }
                }
                return found;
            };
            Check(tubeHasRegionShade(cutRootB, 1, 1, -1) &&
                      tubeHasRegionShade(cutKids[1], 2, 0, 1) &&
                      tubeHasRegionShade(cutGrandkids[0], 3, 0, 0) &&
                      tubeHasRegionShade(cutGrandkids[1], 3, 0, 1),
                  "mixed-depth tubes retain their underlying region's colour family");
            Check(ArraySize<VtVec3fArray>(*pomade, centersL1, "points", 5) &&
                      ArraySize<VtVec3fArray>(*pomade, centersL2, "points", 5) &&
                      ArraySize<VtVec3fArray>(*pomade, centersL3, "points", 10) &&
                      ArrayNonEmpty<VtVec3fArray>(*pomade, ringCVsL1, "points") &&
                      ArrayNonEmpty<VtVec3fArray>(*pomade, ringCVsL2, "points") &&
                      ArrayNonEmpty<VtVec3fArray>(*pomade, ringCVsL3, "points"),
                  "cut: center and ring controls match each visible branch batch");
            using CutPx = usdGenPomade::PomadeOverlayPixels;
            Check(std::fabs(firstWidth(centersL1) -
                            CutPx::kCenterCurveFocused * 0.004f) < 1e-6f &&
                      std::fabs(firstWidth(centersL2) -
                            CutPx::kCenterCurveFocused * 0.004f) < 1e-6f &&
                      std::fabs(firstWidth(centersL3) -
                            CutPx::kCenterCurveFocused * 0.004f) < 1e-6f,
                  "cut: every mixed-depth frontier batch has active edit cues");
            Check(Pomade_SetTubeExpanded(patchCtx, 0, 0) == POMADE_OK &&
                      Pomade_Publish(patchCtx, ~0u) >= 1,
                  "cut: collapsing root A republishes the two roots");
            Check(Pomade_GetPublishedLevelInfo(patchCtx, 1, &faces, &points,
                                              &tubes) == POMADE_OK &&
                      tubes == 2 &&
                      pomade->GetPrim(tubesL2).primType.IsEmpty() &&
                      pomade->GetPrim(centersL2).primType.IsEmpty() &&
                      pomade->GetPrim(ringCVsL2).primType.IsEmpty() &&
                      pomade->GetPrim(tubesL3).primType.IsEmpty() &&
                      pomade->GetPrim(centersL3).primType.IsEmpty() &&
                      pomade->GetPrim(ringCVsL3).primType.IsEmpty() &&
                      Pomade_IsTubeVisible(patchCtx, 0) == 1 &&
                      Pomade_IsTubeVisible(patchCtx, cutRootB) == 1 &&
                      Pomade_IsTubeVisible(patchCtx, cutKids[0]) == 0 &&
                      Pomade_IsTubeVisible(patchCtx, cutGrandkids[0]) == 0,
                  "cut: collapsing hides descendant meshes and component overlays");
            Check(Pomade_Deactivate(patchCtx) == POMADE_OK &&
                      Pomade_Destroy(patchCtx) == POMADE_OK,
                  "V8: the same-face patch model destroys");
        }
        Check(Pomade_Destroy(sc) == POMADE_OK, "V8: the scalp model destroys");

        // plan/02 §2.20: a face GeomSubset scalp tints its own faces only.
        // The faces a subset leaves out are not scalp, so the coarse base
        // must not paint them as uncovered (or at all).
        PomadeModelContext *subsetCtx = nullptr;
        Check(Pomade_Create(&subsetCtx) == POMADE_OK && subsetCtx != nullptr,
              "subset tint: the model creates");
        if (subsetCtx) {
            int const keep[] = {0, 1, 2, 3, 4, 6, 7, 8,
                                9, 10, 11, 12, 13, 14, 15};  // all but 5
            Check(Pomade_BindScalpSubset(subsetCtx, points.data(),
                                        int(points.size()), counts.data(),
                                        int(counts.size()), indices.data(),
                                        int(indices.size()), keep, 15) ==
                          POMADE_OK &&
                      Pomade_Rasterise(subsetCtx) == POMADE_OK &&
                      Pomade_Activate(subsetCtx) == POMADE_OK &&
                      Pomade_Publish(subsetCtx, ~0u) >= 1,
                  "subset tint: the subset scalp binds and publishes");
            VtIntArray tintCounts, tintIndices, tintRegions;
            HdContainerDataSourceHandle const tintPrim =
                pomade->GetPrim(regions).dataSource;
            if (HdSampledDataSourceHandle const ds = SampledAt(
                    tintPrim, HdDataSourceLocator(TfToken("mesh"),
                                                  TfToken("topology"),
                                                  TfToken("faceVertexCounts")))) {
                VtValue const v = ds->GetValue(0.0f);
                if (v.IsHolding<VtIntArray>()) {
                    tintCounts = v.UncheckedGet<VtIntArray>();
                }
            }
            if (HdSampledDataSourceHandle const ds = SampledAt(
                    tintPrim, HdDataSourceLocator(TfToken("mesh"),
                                                  TfToken("topology"),
                                                  TfToken("faceVertexIndices")))) {
                VtValue const v = ds->GetValue(0.0f);
                if (v.IsHolding<VtIntArray>()) {
                    tintIndices = v.UncheckedGet<VtIntArray>();
                }
            }
            ArraySize<VtIntArray>(*pomade, regions, "usdGen:pomadeRegion", 15,
                                  &tintRegions);
            bool uncovered = tintRegions.size() == 15;
            for (int id : tintRegions) {
                uncovered = uncovered && id == -1;
            }
            // Face 5 is the quad (6, 11, 12, 7) of the parent grid.
            bool holeDrawn = false;
            for (size_t f = 0; f < tintIndices.size() / 4; ++f) {
                holeDrawn = holeDrawn || (tintIndices[f * 4 + 0] == 6 &&
                                          tintIndices[f * 4 + 1] == 11 &&
                                          tintIndices[f * 4 + 2] == 12 &&
                                          tintIndices[f * 4 + 3] == 7);
            }
            Check(tintCounts.size() == 15 && tintIndices.size() == 60 &&
                      uncovered && !holeDrawn,
                  "subset tint: only the 15 subset faces are tinted uncovered "
                  "(got " + std::to_string(tintCounts.size()) + ")");
            Check(Pomade_Deactivate(subsetCtx) == POMADE_OK &&
                      Pomade_Destroy(subsetCtx) == POMADE_OK,
                  "subset tint: the model destroys");
        }
        Check(Pomade_Activate(ctx) == POMADE_OK && Pomade_Publish(ctx, ~0u) >= 1,
              "V8: the first model comes back for the teardown checks");
    }

    // -- V9 T1: the published prims follow the display policy -------------
    //
    // The table is proven pure above; this proves the wiring. One model,
    // two levels, one Pomade_SetDisplayPolicy per mode, and the assertion
    // is on what the INDEX published: the xray alpha, the material the
    // mesh binds, and whether the centers and CV dots are visible.
    {
        PomadeModelContext *dc = nullptr;
        Check(Pomade_Create(&dc) == POMADE_OK, "V9: a policy model creates");
        Check(Pomade_BuildTestTube(dc, 0, 0, 0.0f, 0.0f) == POMADE_OK,
              "V9: the policy model builds its root tube");
        int dKids[2] = {0, 0};
        int dKidCount = 0;
        Check(Pomade_SubdivideTube(dc, 0, 2, "kmeans", 7, dKids, 2,
                                  &dKidCount) == POMADE_OK &&
                  dKidCount == 2,
              "V9: it subdivides into two children");
        Check(Pomade_Activate(dc) == POMADE_OK && Pomade_Publish(dc, ~0u) >= 1,
              "V9: the policy model activates and publishes");

        SdfPath const dTubes1 = UsdGenPomadeSceneIndex::TubesPath(1);
        SdfPath const dTubes2 = UsdGenPomadeSceneIndex::TubesPath(2);
        SdfPath const dCenters2 = UsdGenPomadeSceneIndex::CentersPath(2);
        SdfPath const dCVs2 = UsdGenPomadeSceneIndex::CenterCVsPath(2);
        SdfPath const dRingCVs2 = UsdGenPomadeSceneIndex::RingCVsPath(2);
        SdfPath const dGuides2 = UsdGenPomadeSceneIndex::GuidesPath(2);
        auto xrayOf = [&](SdfPath const &path) -> float {
            VtFloatArray a;
            return ArraySize(*pomade, path, "xray", 1, &a) && !a.empty()
                       ? a[0]
                       : -1.0f;
        };
        auto visibleOf = [&](SdfPath const &path) -> bool {
            HdSampledDataSourceHandle ds = SampledAt(
                pomade->GetPrim(path).dataSource, kVisibility);
            if (!ds) {
                return false;
            }
            VtValue const v = ds->GetValue(0.0f);
            return v.IsHolding<bool>() && v.UncheckedGet<bool>();
        };

        // Graph: region patches and their white boundary/CV controls must be
        // readable without a tube wall in front of them.
        Check(Pomade_SetDisplayPolicy(dc, "graph", "", 0) == POMADE_OK &&
                  Pomade_Publish(dc, 0) >= 1,
              "V9: the Graph policy applies and publishes");
        Check(!visibleOf(dTubes1) && !visibleOf(dTubes2),
              "V9: Graph hides every tube level");
        Check(!visibleOf(dCenters2) && !visibleOf(dCVs2),
              "V9: Graph hides the center curves and CV dots");

        // Hierarchy at L2 focus: focused level 25 %, the level behind 10 %,
        // centers back on, and the ghosted levels on the OIT material so
        // they stop writing depth over what is inside them.
        Check(Pomade_SetDisplayPolicy(dc, "hierarchy", "", 2) == POMADE_OK &&
                  Pomade_Publish(dc, 0) >= 1,
              "V9: the Hierarchy policy applies at L2 focus");
        Check(std::fabs(xrayOf(dTubes2) -
                        usdGenPomade::PomadeModel::kDefaultXrayOpacity) < 1e-6f,
              "V9: the focused level publishes 25 % x-ray");
        Check(std::fabs(xrayOf(dTubes1) -
                        usdGenPomade::PomadeModel::kFaintXrayOpacity) < 1e-6f,
              "V9: the level behind it publishes 10 %");
        Check(MaterialBindingOf(*pomade, dTubes1) ==
                      UsdGenPomadeSceneIndex::TubeXrayMaterialPath() &&
                  MaterialBindingOf(*pomade, dTubes2) ==
                      UsdGenPomadeSceneIndex::TubeXrayMaterialPath(),
              "V9: both x-rayed levels bind the translucent tube material");
        Check(visibleOf(dCenters2) && visibleOf(dCVs2),
              "V9: and the center curves and CV dots come back");
        {
            int focus = 0;
            focus = Pomade_GetFocusLevel(dc);
            Check(focus == 2, "V9: the policy sets the focus level too");
            Check(Pomade_GetRingDisplay(dc) == 1,
                  "V9: Hierarchy leaves the rings on the selection");
        }
        {
            float opacity = -1.0f;
            int centers = -1;
            Check(Pomade_GetLevelDraw(dc, 1, &opacity, &centers) == POMADE_OK &&
                      std::fabs(opacity -
                                usdGenPomade::PomadeModel::kFaintXrayOpacity) <
                          1e-6f &&
                      centers == 1,
                  "V9: Pomade_GetLevelDraw reads the policy back");
        }

        // The Tube component rows show exactly the point controls their
        // selection domain accepts. Curves remain orientation guides; point
        // glyphs never advertise an inactive component kind.
        Check(Pomade_SetDisplayPolicy(dc, "tube", "tube", 2) == POMADE_OK &&
                  Pomade_Publish(dc, 0) >= 1 &&
                  !visibleOf(dCVs2) && !visibleOf(dRingCVs2),
              "V9: Tube/Object hides all component dots");
        Check(Pomade_SetDisplayPolicy(dc, "tube", "center", 2) == POMADE_OK &&
                  Pomade_Publish(dc, 0) >= 1 &&
                  visibleOf(dCVs2) && !visibleOf(dRingCVs2),
              "V9: Tube/Center shows only center CV dots");

        // Tube / Ring: focused opaque with every ring, others ghosted, and
        // only section vertices are displayed as point controls.
        Check(Pomade_SetDisplayPolicy(dc, "tube", "ring", 2) == POMADE_OK &&
                  Pomade_Publish(dc, 0) >= 1,
              "V9: the Tube/Ring policy applies");
        Check(xrayOf(dTubes2) == 0.0f,
              "V9: Tube/Ring keeps the focused level opaque");
        Check(std::fabs(xrayOf(dTubes1) -
                        usdGenPomade::PomadeModel::kDefaultXrayOpacity) < 1e-6f,
              "V9: and ghosts the level behind it at 25 %");
        Check(Pomade_GetRingDisplay(dc) == 2,
              "V9: Tube/Ring shows every ring");
        Check(!visibleOf(dCVs2) && visibleOf(dRingCVs2),
              "V9: Tube/Ring shows only section CV dots");
        Check(Pomade_SetDisplayPolicy(dc, "tube", "section", 2) == POMADE_OK &&
                  Pomade_Publish(dc, 0) >= 1 &&
                  !visibleOf(dCVs2) && visibleOf(dRingCVs2),
              "V9: Tube/Section keeps only section CV dots");
        Check(Pomade_SetDisplayPolicy(dc, "nonsense", "", 0) == POMADE_ERROR,
              "V9: an unknown mode is an error and changes nothing");
        Check(xrayOf(dTubes2) == 0.0f,
              "V9: the rejected call left the published state alone");

        // Guide preview visibility is policy-owned.  Its path and topology
        // stay stable while Fill switches to Tube/Center, so Hydra needs a
        // visibility dirty on the existing curves or it keeps drawing the
        // dense Fill preview over the Tube controls.
        bool dLeavesSet = (dKidCount == 2);
        for (int k = 0; dLeavesSet && k < dKidCount; ++k) {
            dLeavesSet =
                Pomade_SetTubeFillParams(dc, dKids[k], 8.0f, 6, 11, 0.0f,
                                        nullptr, 0) == POMADE_OK;
        }
        Check(dLeavesSet && Pomade_RefillGuides(dc, 1.0f) == POMADE_OK &&
                  Pomade_SetDisplayPolicy(dc, "fill", "", 2) == POMADE_OK &&
                  Pomade_Publish(dc, 0) >= 1,
              "V9: the Fill guide preview publishes");
        Check(visibleOf(dGuides2),
              "V9: Fill displays the live generated curves");
        rec->Clear();
        Check(Pomade_SetDisplayPolicy(dc, "tube", "center", 2) == POMADE_OK &&
                  Pomade_Publish(dc, 0) >= 1,
              "V9: Tube/Center publishes after Fill");
        Check(!visibleOf(dGuides2),
              "V9: Tube/Center hides the live generated curves");
        Check(rec->DirtiedFor(dGuides2) == HdDataSourceLocatorSet{kVisibility},
              "V9: changing only guide policy dirties guide visibility");
        Check(!rec->WasAdded(dGuides2) && !rec->WasRemoved(dGuides2),
              "V9: hiding generated curves does not change their topology");

        // -- the committed guides hide while the model is live -----------
        //
        // <groom>/Guides is last commit's curves, an ordinary BasisCurves
        // with no displayColor, so Storm draws it plain white over the
        // tubes. The index hides it in Hydra only.
        SdfPath const groom("/World/PomadeGroom");
        SdfPath const committed = groom.AppendChild(TfToken("Guides"));
        input->AddPrims({{committed, TfToken("basisCurves"),
                          HdRetainedContainerDataSource::New()}});
        Check(UsdGenPomadeSceneIndex::CommittedGuidesPath(
                  "/World/PomadeGroom") == committed,
              "V9: the hidden path is <groom>/Guides");
        Check(UsdGenPomadeSceneIndex::CommittedGuidesPath("").IsEmpty(),
              "V9: no groom path hides nothing");
        // The input publishes no visibility of its own, so "drawing" here
        // means "this index authors no opinion over it".
        auto hiddenOf = [&](SdfPath const &path) {
            HdSampledDataSourceHandle ds = SampledAt(
                pomade->GetPrim(path).dataSource, kVisibility);
            if (!ds) {
                return false;
            }
            VtValue const v = ds->GetValue(0.0f);
            return v.IsHolding<bool>() && !v.UncheckedGet<bool>();
        };
        Check(pomadeRaw->HiddenGuidesPath().IsEmpty(),
              "V9: with no groom path set, nothing is hidden");
        Check(!hiddenOf(committed),
              "V9: and the committed guides draw");
        rec->Clear();
        Check(Pomade_SetGroomPath(dc, "/World/PomadeGroom") == POMADE_OK &&
                  Pomade_Publish(dc, 0) >= 1,
              "V9: Pomade_SetGroomPath publishes");
        Check(pomadeRaw->HiddenGuidesPath() == committed,
              "V9: the index now hides that prim");
        Check(hiddenOf(committed),
              "V9: the committed guides are invisible while the model lives");
        Check(rec->WasDirtied(committed),
              "V9: and their visibility was dirtied, not resynced");
        {
            char buf[64] = {0};
            Check(Pomade_GetGroomPath(dc, buf, int(sizeof(buf))) == POMADE_OK &&
                      std::string(buf) == "/World/PomadeGroom",
                  "V9: Pomade_GetGroomPath round trips");
        }
        // Every other input prim is untouched by the override.
        SdfPath const bystander("/World/Scalp");
        input->AddPrims({{bystander, TfToken("mesh"),
                          HdRetainedContainerDataSource::New()}});
        Check(!hiddenOf(bystander) &&
                  pomade->GetPrim(bystander).primType == TfToken("mesh"),
              "V9: an unrelated input prim passes through unchanged");
        rec->Clear();
        Check(Pomade_Deactivate(dc) == POMADE_OK && Pomade_Publish(dc, 0) >= 0,
              "V9: the model deactivates");
        Check(pomadeRaw->HiddenGuidesPath().IsEmpty(),
              "V9: deactivating stops hiding the committed guides");
        Check(!hiddenOf(committed),
              "V9: and they draw again");
        Check(Pomade_Destroy(dc) == POMADE_OK, "V9: the policy model destroys");
    }
    Check(Pomade_Activate(ctx) == POMADE_OK && Pomade_Publish(ctx, ~0u) >= 1,
          "V9: the first model comes back for the teardown checks");

    // -- deactivate: the levels go, the test tube does not return ----------
    rec->Clear();
    Check(Pomade_Deactivate(ctx) == POMADE_OK, "deactivate succeeds");
    Check(rec->WasRemoved(tubesL1) && rec->WasRemoved(tubesL2),
          "deactivating removes the level prims");
    Check(!rec->WasAdded(testTube) && !pomadeRaw->HasTestTube(),
          "the test tube stays retired after deactivate");
    Check(registry.GetActiveId() == 0, "no model is active");

    pomade->RemoveObserver(observer);
    Check(Pomade_Destroy(ctx) == POMADE_OK, "model destroys");
    Check(registry.ModelCount() == 0, "destroying unregisters the model");

    // -- the index detaches on destruction ---------------------------------
    pomade = nullptr;
    Check(registry.IndexCount() == 0,
          "the index detaches from the registry on destruction");

    // -- V6 / G6: what the un-aliased VtArray fill actually costs ----------
    //
    // plan/17 §4.3 phase A promised VtArrays that ALIAS the pinned block, so
    // a publish would make no second copy. VtArray has no external-buffer
    // form (that was a Hydra 1.x HdVtBufferSource spelling), so a publish
    // memcpys the pinned block into the array Hydra retains. This measures
    // the copy that promise was about, at the plan/17 §7 reference tube
    // vertex count, so the decision to keep it is a number and not a guess.
    {
        size_t const verts = 770000;
        usdGenPomade::PomadePinnedStaging positions, normals;
        float *pinnedP = positions.Ensure(verts * 3);
        float *pinnedN = normals.Ensure(verts * 3);
        Check(pinnedP != nullptr && pinnedN != nullptr,
              "G6: the reference-scale pinned block allocates");
        if (pinnedP && pinnedN) {
            for (size_t i = 0; i < verts * 3; ++i) {
                pinnedP[i] = float(i) * 1e-5f;
                pinnedN[i] = 0.0f;
            }
            VtVec3fArray pts, nrm;
            auto const t0 = std::chrono::steady_clock::now();
            pts.resize(verts);
            nrm.resize(verts);
            std::memcpy(pts.data(), pinnedP, verts * 3 * sizeof(float));
            std::memcpy(nrm.data(), pinnedN, verts * 3 * sizeof(float));
            auto const t1 = std::chrono::steady_clock::now();
            double const ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::printf("info: G6 pinned -> VtArray at %zu verts "
                        "(pos + normals, %.1f MB): %.3f ms, pinned=%d\n",
                        verts, double(verts * 6 * sizeof(float)) / 1048576.0,
                        ms, int(positions.IsPinned()));
            Check(pts.size() == verts && nrm.size() == verts,
                  "G6: the staged arrays are the reference size");
            // The §7 per-move budget is 8 ms. The copy is charged against it
            // on every publish, so it has to be a small fraction of one.
            Check(ms < 8.0,
                  "G6: the retained-array copy fits inside one move budget");
        }
    }

    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
