// usdGen engine — UsdGenGraphDesc, the engine's pure-value input (ADR §4.2.3,
// 03-execution-engine.md §2.2). Built by usdGenImaging from Hydra data sources;
// the engine never sees a UsdStage, a scene index or an SdfLayer (S8).
#ifndef USDGEN_GRAPH_DESC_H
#define USDGEN_GRAPH_DESC_H

#include "usdGen/types.h"
#include "usdGen/expressions/context.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/value.h"
#include "pxr/usd/sdf/path.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenImagePayload;

/// One authored usdGen:* property, resolved. `name` is the property name with
/// the `usdGen:` prefix stripped (C1 registry, docs/freezes/C1.md).
struct UsdGenParamValue
{
    TfToken name;                // "clump:size", "width:knots", "surface"
    VtValue value;               // scalar, token, array, or resolved asset path
    bool    animated = false;    // authored with a .spline / time samples
};

// Invalid is deliberately distinct from CpuReference: a present but
// malformed/unknown backend request must fail closed instead of silently
// selecting the legacy CPU implementation.  Keep the existing values of the
// two established backends stable for descriptor compatibility.
enum class UsdGenExecutionBackend : uint8_t {
    // Keep the original descriptor values stable. These values are carried
    // by some host-side graph snapshots, so adding portable backend names
    // must not renumber either established backend or Invalid.
    CpuReference = 0,
    Cuda = 1,
    Invalid = 2,
    Metal = 3,
    Vulkan = 4
};

struct UsdGenExpressionOutputDesc { TfToken name{"result"}; TfToken nativeType{"float"}; expr::ValueShape shape; };

/// One `input:<name>` relationship of a UsdGenExpression prim: the external
/// data geoSampler("<name>", ...) and ptex("<name>") read. `targets` is the
/// authored (forwarded) target list; the builder resolves it into gprims in
/// UsdGenGraphDesc::geometries (a non-gprim target contributes its descendant
/// gprims, in namespace order) and map prims in UsdGenGraphDesc::maps.
struct UsdGenExpressionInputDesc
{
    TfToken       name;         // "guideCurves" for input:guideCurves
    SdfPathVector targets;
    SdfPathVector geometries;   // resolved gprim paths, in order
    SdfPathVector maps;         // resolved UsdGenMap prim paths, in order
};

struct UsdGenExpressionDesc
{
    SdfPath path;
    std::string source;
    std::vector<UsdGenExpressionOutputDesc> outputs;
    std::vector<UsdGenExpressionInputDesc> inputs;
};

/// A gprim an expression samples (geoSampler), by value. Points are in the
/// prim's object space; worldMatrix places them.
enum class UsdGenGeometryKind : uint8_t { Mesh, Curves, Points };

struct UsdGenGeometryDesc
{
    SdfPath            path;
    UsdGenGeometryKind kind = UsdGenGeometryKind::Points;
    VtIntArray         counts;       // faceVertexCounts | curveVertexCounts | empty
    VtIntArray         indices;      // faceVertexIndices (mesh)
    VtVec3fArray       points;       // at UsdGenGraphDesc::time
    VtVec3fArray       rest;         // Default-time rest when the prim has one, else empty
    VtVec3fArray       normals;      // points: optional per-point normals
    VtArray<uint64_t>  ids;          // curves: usdGen:curveId; points: ids (optional)
    // A GeomSubset input (R15): `path` is the subset, the arrays above its
    // parent Mesh's, and only subsetFaces (sorted parent face ids) and their
    // points are sampled. isSubset keeps an empty subset empty.
    VtIntArray         subsetFaces;
    bool               isSubset = false;
    GfMatrix4d         worldMatrix{1.0};
    /// Content identity of everything above, computed by the builder. Any
    /// edit of the prim's data changes it, so a consumer's capture identity
    /// can fold it instead of rehashing arrays.
    uint64_t           generation = 0;
};
struct UsdGenExpressionBinding {
    // `output` empty means the connection named the expression PRIM rather
    // than one of its outputs; UsdGenFindExpressionOutput resolves it.
    SdfPath expression; TfToken output{"result"}; TfToken nativeType{"float"}; TfToken destination;
    expr::ValueShape destinationShape; expr::Domain domain = expr::Domain::Groom;
    VtValue literal;
};

/// The output a binding names on `expression`.  A named output must exist.
/// An EMPTY name is a prim-path connection: it resolves to "result" when the
/// expression declares it, otherwise to its single declared output.  Returns
/// nullptr when nothing matches, or when a prim-path connection is ambiguous
/// because the expression declares several non-`result` outputs.
inline UsdGenExpressionOutputDesc const *
UsdGenFindExpressionOutput(UsdGenExpressionDesc const &expression,
                           TfToken const &output)
{
    if (!output.IsEmpty()) {
        for (UsdGenExpressionOutputDesc const &o : expression.outputs)
            if (o.name == output) return &o;
        return nullptr;
    }
    for (UsdGenExpressionOutputDesc const &o : expression.outputs)
        if (o.name == TfToken("result")) return &o;
    return expression.outputs.size() == 1 ? &expression.outputs.front() : nullptr;
}

/// S11 ramp encodings, already resolved by the adapter (02-schema.md §2.17).
struct UsdGenRampDesc
{
    VtVec2fArray knots;          // scalar ramp: (position, value), sorted by x
    VtFloatArray positions;     // colour ramp
    VtVec3fArray colors;        // colour ramp
    TfToken      interpolation; // linear | catmullRom | bspline | constant (default catmullRom, R11)
};

/// The authored relationship slot which consumes a map.  usdGen:map is the
/// only map relationship; the binding retains its exact authored name so a
/// diagnostic can quote it.  Empty is valid for direct descriptor clients
/// which only populate UsdGenNodeDesc::maps.
struct UsdGenMapBindingDesc
{
    SdfPath  map;
    TfToken  relationship;
};

/// A named plane authored on a C3 curve set. Descriptor-only transport keeps
/// the source path backend-neutral: Point becomes vertex, Primitive uniform,
/// and Groom one constant value in the CPU CurveSource buffer.
enum class UsdGenAuthoredPlaneType : uint8_t { Float32, Int32 };
enum class UsdGenAuthoredPlaneDomain : uint8_t { Point, Primitive, Groom };

struct UsdGenAuthoredPlaneDesc
{
    TfToken name;
    UsdGenAuthoredPlaneType type = UsdGenAuthoredPlaneType::Float32;
    UsdGenAuthoredPlaneDomain domain = UsdGenAuthoredPlaneDomain::Point;
    uint8_t arity = 1;                 // [1, 16]
    VtFloatArray floatValues;          // populated only for Float32
    VtIntArray intValues;              // populated only for Int32
};

/// Immutable surface-cage controls captured alongside a CurveSource's sparse
/// C3 rails.  The arrays are deliberately stage-free and preserve the
/// authored owner/triangle order used by the runtime interpolator.  Chart
/// positions are captured independently of the surface map so a density or
/// profile edit changes the capture identity without re-reading USD.
struct UsdGenSurfaceCagePayload
{
    VtIntArray   ownerIds;
    VtFloatArray ownerDensities;
    VtIntArray   ownerSeeds;
    VtIntArray   ownerCvCounts;
    VtFloatArray ownerEdgeBias;
    VtIntArray   ownerLengthProfileOffsets;
    VtVec2fArray ownerLengthProfile;
    VtFloatArray normalizedT;           // flattened per-point rail parameter
    VtVec3iArray triangles;              // guide-curve indices
    VtIntArray   triangleOwnerIndices;
    VtVec2fArray triangleRootCharts;     // 3 entries per triangle
    VtVec2fArray ownerChartCentroids;
    VtFloatArray ownerChartMeanRadii;
};

/// The source curves' own `primvars:displayColor`, forwarded as an authored
/// plane under this reserved name. It rides the ordinary named-plane machinery
/// so resampling and compaction carry it correctly, but the session cooker
/// consumes it into the tile's displayColor instead of publishing it, so it
/// never appears as a stray primvar. `displayColor` itself cannot be used: the
/// compiler reserves that name for authored planes.
inline TfToken const &UsdGenSourceColorPlane()
{
    static TfToken const name("usdGenSourceColor");
    return name;
}

struct UsdGenNodeDesc
{
    SdfPath                      path;         // the operator prim's scene path (identity + Kahn tie-break)
    TfToken                      type;         // "UsdGenClump", "UsdGenScatter", ...
    TfToken                      mode;         // usdGen:mode, when the type has one
    bool                         enabled = true;
    int                          seed = 0;
    SdfPathVector                inputs;       // compiler-owned hierarchy dependency edges
    SdfPathVector                references;   // guide sets, clump centres, card roots
    SdfPathVector                curves;       // usdGen:guides / usdGen:curves / usdGen:frozen:curves
                                               //   -> indices into UsdGenGraphDesc::curveSets
    SdfPathVector                surfaces;     // the Description's usdGen:surface targets (ADR R15)
    // Legacy untyped map paths.  Keep this populated by builders and accept
    // it from old direct clients; mapBindings is the canonical typed form.
    SdfPathVector                maps;
    std::vector<UsdGenMapBindingDesc> mapBindings;
    std::vector<UsdGenParamValue> params;      // EVERY mapped locator of this prim (S14 pull-all)
    std::vector<UsdGenExpressionBinding> expressionBindings;
    std::vector<UsdGenRampDesc>   ramps;
};

/// One authored C3 `BasisCurves` input: a UsdGenGuideSet child, a freeze, an
/// import, a sim cache. The only way curve data reaches the engine (R23).
struct UsdGenCurveSetDesc
{
    SdfPath         path;
    UsdGenRole      role;              // Reference (guides, clump centres) | Curves (a CurveSource)
    TfToken         curveRole;         // primvars:usdGen:role: hair | guide (C3 marker)
    VtIntArray      curveVertexCounts;
    VtVec3fArray    points, rest;      // rest == primvars:rest; may share points' buffer
    // Compatibility builders may lack a Default-time C3 snapshot. CUDA
    // admission must not mistake a current-frame fallback for bound rest.
    bool            restFromCurrentPoints = false;
    GfMatrix4d      worldMatrix{1.0}; // source object space must be explicit for deformation
    VtFloatArray    widths;
    TfToken         type{"cubic"};
    TfToken         basis{"bspline"};
    TfToken         wrap{"pinned"};
    TfToken         widthsInterpolation{"vertex"};
    VtIntArray      skinPrim;          // primvars:skinprim (uniform int)
    VtArray<uint64_t> curveId;         // primvars:usdGen:curveId (uniform uint64[], R12)
    VtVec2fArray    skinPrimUv;        // primvars:skinprimuv (uniform texCoord2f, not "st")
    VtMatrix4dArray rootFrame;         // primvars:usdGen:rootFrame; may be empty
    std::string     frozenEpoch;       // constant string primvar, "usdgen1:sha1:..." (S42)
    std::vector<UsdGenAuthoredPlaneDesc> authoredPlanes;
    std::shared_ptr<const UsdGenSurfaceCagePayload> surfaceCage;
    uint64_t        curveGeneration = 0;   // bumped by any points/topology/id change on the prim
};

/// A surface geometry sample at an absolute time (R23).
struct UsdGenSurfaceSample { double time; VtVec3fArray points; };

// The Default-time mesh-normal interpolation transported by UsdGenRestAPI.
// Invalid is deliberately distinct from None: a nonempty malformed snapshot
// must not be silently converted into geometric-normal fallback by CUDA.
enum class UsdGenSurfaceNormalDomain : uint8_t {
    None,
    Constant,
    Uniform,
    Vertex,
    FaceVarying,
    Invalid
};

struct UsdGenSurfaceDesc
{
    SdfPath        path;
    UsdGenSurfaceId id = 0;
    VtIntArray     faceVertexCounts, faceVertexIndices;
    // USD subdivision opinions, captured identically by stage/Hydra builders.
    TfToken subdivisionScheme{"none"}, orientation{"rightHanded"};
    TfToken interpolateBoundary{"edgeAndCorner"}, faceVaryingLinearInterpolation{"cornersPlus1"};
    TfToken triangleSubdivisionRule{"catmullClark"}, creaseMethod{"uniform"};
    VtIntArray holeIndices, creaseIndices, creaseLengths, cornerIndices;
    VtFloatArray creaseSharpnesses, cornerSharpnesses;
    VtVec3fArray   restPoints;        // usdGen/rest/points (S12), UsdTimeCode::Default()
    VtVec3fArray   restNormals;       // usdGen/rest/normals, Default-time Mesh normals
    UsdGenSurfaceNormalDomain restNormalDomain = UsdGenSurfaceNormalDomain::None;
    bool          restFromCurrentPoints = false; // compatibility fallback, never a valid RBF binding
    VtVec3fArray   points;            // deformed, at UsdGenGraphDesc::time
    /// Sorted by time; samples[0].time == UsdGenGraphDesc::time (R23).
    std::vector<UsdGenSurfaceSample> samples;
    VtVec3fArray   velocities;        // motion profile P1 only; empty otherwise
    VtVec2fArray   uv;                // the surface's primary uv set
    // A GeomSubset target (R15) carries its parent Mesh's full geometry
    // above, so face ids stay parent-mesh ids, plus its sorted, unique faces
    // here. isSubset makes subsetFaces authoritative even when empty (an
    // empty subset selects no face); otherwise empty == whole mesh.
    VtIntArray     subsetFaces;
    bool           isSubset = false;
    VtFloatArray   densityMultiplier; // per-face scatter density scale from
                                     // usdGen:paint:density (face means,
                                     // clamped >= 0, non-finite -> 0);
                                     // empty == all 1.0
    GfMatrix4d     worldMatrix{1.0};  // post-flattening, resetXformStack (S4)
    uint64_t       surfaceGeneration = 0; // bumped by any points/topology change
    // Sub-hash of the slowly-varying inputs (rest points/normals, topology):
    // the Hydra builder memoizes it across cooks while its shares hold, so a
    // deform re-hashes only the moving points. Set by Finalize alongside the
    // generation; in-memory equality-only like the generation itself.
    uint64_t       surfaceInvariantGeneration = 0;
};

/// Whether consumers restrict a surface to subsetFaces: a GeomSubset target
/// (even an empty one) or a hand-built desc that names faces.
inline bool
UsdGenSurfaceRestricted(UsdGenSurfaceDesc const &surface)
{
    return surface.isSubset || !surface.subsetFaces.empty();
}

/// The Mesh prim a surface's geometry and primvars live on: the target
/// itself, or a GeomSubset's parent (a subset is always a Mesh's child).
inline SdfPath
UsdGenSurfaceMeshPath(UsdGenSurfaceDesc const &surface)
{
    return surface.isSubset ? surface.path.GetParentPath() : surface.path;
}

struct UsdGenMapDesc
{
    SdfPath      path;
    TfToken      type;                // UsdGenImageMap | UsdGenPtexMap | UsdGenExprMap | ...
    std::string  resolvedAssetPath;   // stage-free (S13)
    uint64_t     textureGeneration = 0;  // bumped by ReloadMaps()
    // UsdGenPaintMap payload (empty unless type == UsdGenPaintMap): the
    // surface primvar snapshotted at capture, scalar-folded through
    // usdGen:map:channel. v1 carries faceVarying float data only; any other
    // interpolation, a missing primvar, or a size mismatch leaves
    // paintValues empty and the sampler fails closed.
    SdfPath      paintSurface;
    TfToken      paintPrimvar;
    TfToken      paintInterpolation;
    VtFloatArray paintValues;
    std::vector<UsdGenParamValue> params;
    // Optional stage-free decoded pixels for a typed ImageMap consumer. The
    // immutable payload is backend-neutral; callers replacing it must also
    // advance textureGeneration, which is the deterministic cache identity.
    std::shared_ptr<const UsdGenImagePayload> imagePayload;
};

/// Compiler-resolved map value.  It contains no stage or backend handle: the
/// identity is a deterministic digest of the fully resolved descriptor value
/// and can therefore be retained safely by a CPU capture (or rejected by a
/// backend which has no transport for it).
struct UsdGenResolvedMapValue
{
    SdfPath     path;
    TfToken     type;
    std::string resolvedAssetPath;
    uint64_t    textureGeneration = 0;
    uint64_t    identity = 0;
};

/// The description's look block (02-schema.md §2.12 defaults; 07 §1.2 bake
/// order). M1 bakes the CPU half of the displayColor contract: base colour
/// (white, no maps until M4), the root->tip ramp and the per-curve jitter.
struct UsdGenLookDesc
{
    GfVec3f  rootColor{0.035f, 0.018f, 0.008f};
    GfVec3f  tipColor{0.210f, 0.115f, 0.045f};
    VtVec3fArray rampColors;          // usdGen:look:colorRamp:colors (empty == two-colour mix)
    VtFloatArray rampPositions;       // usdGen:look:colorRamp:positions
    TfToken  rampInterpolation{"catmullRom"};
    float    rampExponent = 1.0f;     // usdGen:look:rampExponent
    TfToken  bakeMode{"perCurve"};    // perCurve | perCV
    TfToken  bakeTarget{"displayColor"};  // displayColor | primvar | none
    TfToken  bakePrimvar{"displayColor"};
    float    hueJitter = 0.0f;
    float    valueJitter = 0.0f;
    int      jitterSeed = 0;
};

/// usdGen:preview:* on the description: a viewport-only colour override that
/// shows a value on the strands (07 §7.3 "value preview"). `source` is the
/// first authored target, unresolved: a UsdGenExpression prim, a
/// UsdGenPtexMap prim (the builders add it to `maps`) or an operator
/// attribute path. The session cooker resolves it against the compiled graph.
struct UsdGenPreviewDesc
{
    SdfPath  source;                    // empty == no preview
    TfToken  colorMap{"heat"};          // heat | viridis | gray | ids | rgb
    GfVec2f  range{0.0f, 1.0f};
    TfToken  evaluation{"primitive"};   // for an expression prim source
    TfToken  shading{"lit"};            // lit | flat

    bool Active() const { return !source.IsEmpty(); }
};

struct UsdGenGraphDesc
{
    SdfPath                        description;   // the UsdGenDescription prim
    SdfPath                        terminal;      // hierarchy-derived final operator
    std::vector<UsdGenNodeDesc>    nodes;         // composed execution order from the builder
    std::vector<UsdGenCurveSetDesc> curveSets;    // every C3 BasisCurves the graph names (R23)
    std::vector<UsdGenSurfaceDesc> surfaces;
    std::vector<UsdGenMapDesc>     maps;
    std::vector<UsdGenExpressionDesc> expressions;
    std::vector<UsdGenGeometryDesc> geometries;   // every gprim an expression input names
    UsdGenLookDesc                 look;
    UsdGenPreviewDesc              preview;
    GfMatrix4d                     xformMatrix{1.0}; // description world matrix (post-flattening, S4)
    TfToken                        purpose;        // inherited by hand to every tile (C2)
    TfToken                        visibility;     // inherited by hand to every tile (C2)
    SdfPath                        materialPath;   // the description's bound Material (C2)
    float    defaultWidth = 0.01f;                 // usdGen:width:default
    int      tileTarget = 64;                      // uniform int usdGen:tileTarget
    TfToken  curveBasis{"bspline"};                // usdGen:curve:basis (C2)
    double   time = 0.0;
    double   timeCodesPerSecond = 24.0; // USD default; expressions expose seconds
    UsdGenExecutionBackend executionBackend = UsdGenExecutionBackend::CpuReference;
    std::vector<std::string> validationErrors;
};

}  // namespace usdGen

#endif  // USDGEN_GRAPH_DESC_H
