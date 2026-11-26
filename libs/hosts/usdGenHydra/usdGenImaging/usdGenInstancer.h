// usdGen imaging — UsdGenInstance instancer (02-schema.md §2.11, 06 §4.3;
// v1, M6).
//
// The ONLY operator that emits no curves: evaluated curves + Instance params
// become one Hydra `instancer` prim per UsdGenInstance,
// <description>/__usdGenRender/inst_<opName> (02 §3.2), whose prototypes are
// the description's Prototypes re-rooted as namespace children of the
// instancer (S33). This unit is self-contained and engine-free: it bakes
// per-instance transforms/primvars from a UsdGenCurveBuffer and assembles
// the Hydra data sources; the scene-index plugin owns session membership,
// prototype mirroring and notice delivery (see the integration note in
// BuildInstancerDataSource).
//
// One instance per evaluated curve. Per-instance hashed choices
// (prototype, scale, twist) key on the stable curveId (R12) and are drawn
// once here, never per frame (02 §6: weights/scaleRandom/twistRandom are
// capture class).
//
// Fail-closed inputs (Bake returns false + error, result untouched):
//   * primitive "archives" (subtree re-root mirroring is plugin work, not
//     fitted here) or any other token than cards|spheres;
//   * orient "camera" (needs live camera state, unavailable at publish) or
//     any other token than surfaceFrame|curveTangent|world;
//   * empty prototype list, empty/mismatched curve buffer, missing curveId,
//     missing root frames for a frame-relative orient, bad weights
//     (size mismatch, negative, all zero), negative width/length.
// An unknown width:interpolation token falls back to linear, mirroring
// usdGenMath/ramp.cpp (NOT fail-closed). Variation names with no baked
// source publish nothing and are reported in unresolvedPrimvars (02 §4:
// the plugin raises one TF_WARN per name).
//
// Self-containment: usdGenMath is PRIVATE to usdGen/usdGenSchema (N-7) and
// is NOT linked into usdGenImaging, and no installed header may include
// usdGenMath/... — so the pinned SplitMix64 draws and the R11 ramp
// evaluation are implemented in the .cpp, citing their canonical sources.
#ifndef USDGEN_IMAGING_INSTANCER_H
#define USDGEN_IMAGING_INSTANCER_H

#include "usdGen/curveBuffer.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/quatf.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/dataSourceLocator.h"
#include "pxr/usd/sdf/path.h"

#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

// Evaluated-curve input bundle for UsdGenInstancer::Bake.
struct UsdGenInstanceCurves
{
    // Terminal input of the UsdGenInstance node: roots (first CV of each
    // span), tangents (root segment), rootT/rootN/rootB frames and curveId.
    // Must outlive the Bake call; never retained.
    usdGen::UsdGenCurveBuffer const *curves = nullptr;
    // Per-curve look bake feeding "displayColor" (02 §2.11 example). Empty =
    // absent. A per-CV array (size == totalCvs) is sampled at each curve's
    // root CV, mirroring the tile displayColor uniform-or-vertex duality
    // (06 §4.1); any other non-empty size fails Bake closed.
    VtVec3fArray displayColor;
};

// Authored UsdGenInstance parameters (schema.usda UsdGenInstance block;
// 02 §2.11). Defaults match the schema; `seed` is the operator prim's
// usdGen:seed (the per-instance draws key on (seed, curveId)).
struct UsdGenInstanceParams
{
    TfToken primitive = TfToken("cards");   // cards | spheres (archives fail closed)
    std::vector<SdfPath> prototypes;        // authored usdGen:prototypes targets
    VtFloatArray weights;                   // empty = uniform choice
    TfToken orient = TfToken("surfaceFrame");// surfaceFrame | curveTangent | world
    float scale = 1.0f;
    GfVec2f scaleRandom = GfVec2f(1.0f, 1.0f);
    float twist = 0.0f;                     // degrees
    float twistRandom = 0.0f;               // degrees, centered ± draw
    float normalOffset = 0.0f;              // stage units along rootN
    float width = 0.02f;                    // cards: card width; spheres: diameter
    float length = 0.09f;                   // cards only; ignored for spheres
    VtVec2fArray widthKnots;                // (position, value); empty = flat 1.0
    TfToken widthInterpolation = TfToken("catmullRom");
    VtArray<TfToken> variationPrimvars;     // default {"displayColor"} when empty
    int seed = 0;
};

// Baked per-instance output: Hydra instancer topology + instance-rate data.
struct UsdGenInstanceResult
{
    SdfPath instancerPath;                  // <description>/__usdGenRender/inst_<op>
    std::vector<SdfPath> prototypePaths;    // re-rooted instancer children
    std::vector<VtIntArray> instanceIndices;// one VtIntArray per prototype
    VtBoolArray mask;                       // empty == all true (never written here)
    // Instance-rate arrays, one entry per curve, ascending curve order:
    VtVec3fArray translations;              // primvars/hydra:instanceTranslations
    VtQuatfArray rotations;                 // baked at full float precision;
                                            // the data source publishes
                                            // VtQuathArray (USD's own
                                            // orientations encoding, required
                                            // by hdMoonray)
    VtVec3fArray scales;                    // primvars/hydra:instanceScales
    VtIntArray prototypeIndex;              // owning prototype per instance
    // Instance-interpolated varyings (displayColor + extraCurve planes) as
    // flat float/int planes. Float arity 2/3 pack to VtVec2f/3fArray at
    // assembly (hdMoonray ignores elementSize); displayColor publishes
    // with the color role.
    std::vector<usdGen::UsdGenPlane> varyings;
    // Requested variation names with no baked source: publish nothing.
    std::vector<TfToken> unresolvedPrimvars;
    // Hand-authored by the plugin before BuildInstancerDataSource, exactly
    // as tiles inherit them from the description (06 §4.1): empty purpose
    // omits the container; visibility "invisible" writes false.
    TfToken purpose;
    TfToken visibility;
};

class UsdGenInstancer
{
public:
    // -- vocabulary ------------------------------------------------------
    static TfToken const &PrimitiveCards();
    static TfToken const &PrimitiveSpheres();
    static TfToken const &PrimitiveArchives();
    static TfToken const &OrientSurfaceFrame();
    static TfToken const &OrientCurveTangent();
    static TfToken const &OrientWorld();

    // -- paths (02 §3.2; 06 §4.3) ------------------------------------------
    /// <description>/__usdGenRender/inst_<opName>. `opName` is the Instance
    /// prim's name (TEXT, never a path): SurvivorsOfDescription-style
    /// sanitising is the plugin's job; an empty name fails closed here.
    static SdfPath InstancerPath(SdfPath const &descriptionPath,
                                 std::string const &opName);
    /// <instancer>/Prototypes/<protoName>: the re-rooted namespace child.
    static SdfPath PrototypePath(SdfPath const &instancerPath,
                                 std::string const &protoName);

    // -- bake ---------------------------------------------------------------
    /// Validate params alone (tokens, weights shape, width/length signs).
    /// `prototypeCount` is prototypes.size(); the full Bake re-checks.
    static bool Validate(UsdGenInstanceParams const &params,
                         size_t prototypeCount,
                         std::string *error = nullptr);

    /// Bake evaluated curves + params into per-instance topology,
    /// transforms and primvars. Pure compute: no Hydra, no scene index,
    /// deterministic in (params, curves, seed). False + `error`, with
    /// `result` untouched, on any fail-closed input (see header doc).
    ///
    /// Conventions (all asserted by testUsdGenInstance):
    /// - surfaceFrame: card axes (x,y,z) = (rootT, rootB, rootN),
    ///   re-orthogonalized; twist rotates about local Y (the length axis).
    /// - curveTangent: y = root-segment tangent (rootB fallback for
    ///   single/degenerate CV spans), z = rootN with the tangent projected
    ///   out (rootT fallback when parallel), x = y cross z; twist as above.
    /// - world: identity rotation. spheres: identity rotation always
    ///   (twist is a documented no-op); uniform scale = width * s.
    /// - cards assume a unit 1x1 prototype in XY: scale = (width * ramp *
    ///   s, length * s, s) with ramp = width knots at mid-strand u = 0.5.
    /// - scale s = scale * lerp(scaleRandom) drawn per instance; twist is
    ///   degrees twist + twistRandom * (draw * 2 - 1).
    static bool Bake(UsdGenInstanceParams const &params,
                     UsdGenInstanceCurves const &input,
                     SdfPath const &instancerPath,
                     UsdGenInstanceResult *result,
                     std::string *error = nullptr);

    // -- Hydra assembly (06 §4.3 contract table) ------------------------------
    /// The `instancer` prim data source: instancerTopology/{prototypes,
    /// instanceIndices[, mask]} (HdIntArrayVectorSchema — a vector of
    /// VtIntArray, one per prototype), primvars/{hydra:instanceTranslations,
    /// hydra:instanceRotations, hydra:instanceScales} + varyings at
    /// `instance` interpolation, identity xform + resetXformStack (instance
    /// translations are final, as tile points are), purpose/visibility and
    /// primOrigin/scenePath. Absolute primOrigin outside prototypes (02 §3.2).
    ///
    /// Wire types follow USD's own PointInstancer encoding so every Hydra
    /// consumer (Storm and hdMoonray alike) reads the prim (docs/moonray-
    /// fur.md): instanceRotations publishes VtQuathArray, float varyings of
    /// arity 3/2 publish VtVec3f/2fArray with no elementSize, and
    /// displayColor carries the color role. Scalar int varyings ride
    /// through as VtIntArray; only Storm honors elementSize on multi-arity
    /// int planes.
    ///
    /// INTEGRATION: UsdGenGroomSceneIndex::GetPrim answers this alongside
    /// the tile map (groomSceneIndexPlugin.cpp, next to the
    /// `g.tiles->find(path)` branch), and GetChildPrimPaths appends the
    /// instancer + re-rooted Prototypes/<n> children under the render scope.
    /// Re-rooted prototypes mirror their upstream prim overlaid with
    /// BuildInstancedByDataSource(); value edits dirty Translations, never
    /// the topology (see NoticesFor).
    static HdContainerDataSourceHandle BuildInstancerDataSource(
        UsdGenInstanceResult const &result,
        SdfPath const &primOrigin);

    /// The `instancedBy` overlay for a re-rooted prototype: paths holds
    /// EXACTLY ONE path (the instancer) and prototypeRoots its own
    /// re-rooted path, per 06 §4.3 (more than one path raises
    /// TF_CODING_ERROR downstream).
    static HdContainerDataSourceHandle BuildInstancedByDataSource(
        SdfPath const &instancerPath,
        SdfPath const &prototypeRoot);

    // -- notices (06 §5.1 instancer rows) --------------------------------------
    struct InstanceNotices
    {
        // Any count/binding change -> instancerTopology (index rebuild).
        bool topologyDirty = false;
        // Any card/hair transform edit at constant count ->
        // primvars/hydra:instanceTranslations (one BAR re-upload), NEVER
        // instancerTopology (06 §4.3).
        bool transformsDirty = false;
        // Varying-only edits -> primvars/<name>/primvarValue each.
        std::vector<TfToken> dirtyVaryings;
        std::vector<HdDataSourceLocator> all() const;
    };

    static InstanceNotices NoticesFor(bool topologyChanged,
                                      bool transformsChanged,
                                      std::vector<TfToken> const &varyingNames =
                                          std::vector<TfToken>());
};

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_INSTANCER_H
