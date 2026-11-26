// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGen engine — UsdGenGraphSpec, the engine's pure-value input (step 7.2).
// Every identity is a host-assigned UsdGenNodeKey; the host (usdGenUsd)
// lowers its path-based UsdGenGraphDesc into this. Path text survives only as
// labels, for diagnostics.
#ifndef USDGEN_API_GRAPH_SPEC_H
#define USDGEN_API_GRAPH_SPEC_H

#include "usdGen/api/nodeKey.h"
#include "usdGen/compile/types.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/handles.h"
#include "usdGen/registries/typeRegistry.h"
#include "usdGenExpr/expressions/context.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/value.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenImagePayload;

/// One authored usdGen:* property, resolved. 'name' is the property name with
/// the 'usdGen:' prefix stripped (C1 registry, docs/freezes/C1.md).
struct UsdGenParamValue
{
    TfToken name;                // "clump:size", "width:knots", "surface"
    VtValue value;               // scalar, token, array, or resolved asset path
    bool    animated = false;    // authored with a .spline / time samples
};

// Invalid is deliberately distinct from CpuReference: a present but
// malformed/unknown backend request must fail closed instead of silently
// selecting the legacy CPU implementation. Values are stable: host-side graph
// snapshots carry them.
enum class UsdGenExecutionBackend : uint8_t {
    CpuReference = 0,
    Cuda = 1,
    Invalid = 2,
    Metal = 3,
    Vulkan = 4
};

struct UsdGenExpressionOutputDesc { TfToken name = UsdGenTokens().result; TfToken nativeType = UsdGenTokens().float_; expr::ValueShape shape; };

/// One input:<name> relationship of an expression: the external data
/// geoSampler("<name>", ...) and ptex("<name>") read. 'targets' is the
/// authored (forwarded) target list; 'geometries' the gprims it resolves to,
/// in namespace order; 'maps' the map prims, in order.
struct UsdGenExpressionSlot
{
    TfToken name;                            // "guideCurves" for input:guideCurves
    std::vector<UsdGenNodeKey> targets;
    std::vector<UsdGenNodeKey> geometries;
    std::vector<UsdGenNodeKey> maps;
};

struct UsdGenExpressionInput
{
    UsdGenNodeKey key;
    std::string source;
    std::vector<UsdGenExpressionOutputDesc> outputs;
    std::vector<UsdGenExpressionSlot> inputs;
};

/// A gprim an expression samples (geoSampler), by value. Points are in the
/// prim's object space; worldMatrix places them.
enum class UsdGenGeometryKind : uint8_t { Mesh, Curves, Points };

/// The payload of a sampled gprim, shared by the host desc and the engine input.
struct UsdGenGeometryData
{
    UsdGenGeometryKind kind = UsdGenGeometryKind::Points;
    VtIntArray         counts;       // faceVertexCounts | curveVertexCounts | empty
    VtIntArray         indices;      // faceVertexIndices (mesh)
    VtVec3fArray       points;       // at the graph time
    VtVec3fArray       rest;         // Default-time rest when the prim has one, else empty
    VtVec3fArray       normals;      // points: optional per-point normals
    VtArray<uint64_t>  ids;          // curves: usdGen:curveId; points: ids (optional)
    // A GeomSubset input (R15): the arrays above are its parent Mesh's, and
    // only subsetFaces (sorted parent face ids) and their points are sampled.
    // isSubset keeps an empty subset empty.
    VtIntArray         subsetFaces;
    bool               isSubset = false;
    GfMatrix4d         worldMatrix{1.0};
    /// Content identity of everything above, computed by the builder. Any
    /// edit of the prim's data changes it, so a consumer's capture identity
    /// can fold it instead of rehashing arrays.
    uint64_t           generation = 0;
};

struct UsdGenGeometryInput : UsdGenGeometryData
{
    UsdGenNodeKey key;
};

/// One connected parameter (was UsdGenExpressionBinding). 'output' empty means
/// the connection named the expression PRIM rather than one of its outputs;
/// UsdGenFindExpressionOutput resolves it.
struct UsdGenParamBinding {
    UsdGenNodeKey expression; TfToken output = UsdGenTokens().result; TfToken nativeType = UsdGenTokens().float_; TfToken destination;
    expr::ValueShape destinationShape; expr::Domain domain = expr::Domain::Groom;
    VtValue literal;
};

/// The output a binding names on 'expression' (a host desc or an engine
/// input). A named output must exist. An EMPTY name is a prim connection: it
/// resolves to "result" when the expression declares it, otherwise to its
/// single declared output. Returns nullptr when nothing matches, or when a
/// prim connection is ambiguous.
template <class Expression>
inline UsdGenExpressionOutputDesc const *
UsdGenFindExpressionOutput(Expression const &expression, TfToken const &output)
{
    if (!output.IsEmpty()) {
        for (UsdGenExpressionOutputDesc const &o : expression.outputs)
            if (o.name == output) return &o;
        return nullptr;
    }
    for (UsdGenExpressionOutputDesc const &o : expression.outputs)
        if (o.name == UsdGenTokens().result) return &o;
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

/// The authored relationship slot which consumes a map. usdGen:map is the
/// only map relationship; the binding keeps its exact authored name so a
/// diagnostic can quote it. Empty is valid for clients which only populate
/// UsdGenOpNode::maps.
struct UsdGenMapBinding
{
    UsdGenNodeKey map;
    TfToken       relationship;
};

/// A named plane authored on a C3 curve set: Point becomes vertex, Primitive
/// uniform, and Groom one constant value in the CPU CurveSource buffer.
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
/// C3 rails, in the authored owner/triangle order the runtime interpolator
/// uses. Chart positions are captured independently of the surface map so a
/// density or profile edit changes the capture identity without re-reading.
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

/// The source curves' own primvars:displayColor, forwarded as an authored
/// plane under this reserved name. It rides the named-plane machinery so
/// resampling and compaction carry it, but the session cooker consumes it
/// into the tile's displayColor instead of publishing it. 'displayColor'
/// itself is reserved by the compiler for authored planes.
inline TfToken const &UsdGenSourceColorPlane()
{
    static TfToken const name("usdGenSourceColor");
    return name;
}

/// One operator (was UsdGenNodeDesc). Builder order is the Kahn tie-break.
struct UsdGenOpNode
{
    UsdGenNodeKey                key;          // the operator prim (identity)
    TfToken                      type;         // the operator's schema type name
    TfToken                      mode;         // usdGen:mode, when the type has one
    bool                         enabled = true;
    int                          seed = 0;
    std::vector<UsdGenNodeKey>   inputs;       // compiler-owned hierarchy dependency edges
    std::vector<UsdGenNodeKey>   references;   // guide sets, clump centres, card roots
    std::vector<UsdGenNodeKey>   curves;       // usdGen:guides / usdGen:curves / usdGen:frozen:curves
                                               //   -> UsdGenGraphSpec::curveSets
    std::vector<UsdGenNodeKey>   surfaces;     // the Description's usdGen:surface targets (ADR R15)
    // Untyped map keys. Builders populate both; mapBindings is the canonical form.
    std::vector<UsdGenNodeKey>   maps;
    std::vector<UsdGenMapBinding> mapBindings;
    std::vector<UsdGenParamValue> params;      // EVERY mapped locator of this prim (S14 pull-all)
    std::vector<UsdGenParamBinding> expressionBindings;
    std::vector<UsdGenRampDesc>   ramps;
    UsdGenOpId                   opId = UsdGenOpId::Invalid;   // engine-filled at intake (UsdGenResolveDescIds)
};

/// One authored C3 BasisCurves input: a guide set, a freeze, an import, a sim
/// cache. The only way curve data reaches the engine (R23).
struct UsdGenCurveSetData
{
    UsdGenLane      role;              // Reference (guides, clump centres) | Chunked (a CurveSource)
    TfToken         curveRole;         // primvars:usdGen:role: hair | guide (C3 marker)
    VtIntArray      curveVertexCounts;
    VtVec3fArray    points, rest;      // rest == primvars:rest; may share points' buffer
    // Compatibility builders may lack a Default-time C3 snapshot. CUDA
    // admission must not mistake a current-frame fallback for bound rest.
    bool            restFromCurrentPoints = false;
    GfMatrix4d      worldMatrix{1.0}; // source object space must be explicit for deformation
    VtFloatArray    widths;
    TfToken         type = UsdGenTokens().cubic;
    TfToken         basis = UsdGenTokens().bspline;
    TfToken         wrap = UsdGenTokens().pinned;
    TfToken         widthsInterpolation = UsdGenTokens().vertexWidths;
    VtIntArray      skinPrim;          // primvars:skinprim (uniform int)
    VtArray<uint64_t> curveId;         // primvars:usdGen:curveId (uniform uint64[], R12)
    VtVec2fArray    skinPrimUv;        // primvars:skinprimuv (uniform texCoord2f, not "st")
    VtMatrix4dArray rootFrame;         // primvars:usdGen:rootFrame; may be empty
    std::string     frozenEpoch;       // constant string primvar, "usdgen1:sha1:..." (S42)
    std::vector<UsdGenAuthoredPlaneDesc> authoredPlanes;
    std::shared_ptr<const UsdGenSurfaceCagePayload> surfaceCage;
    uint64_t        curveGeneration = 0;   // bumped by any points/topology/id change on the prim
};

struct UsdGenCurveSetInput : UsdGenCurveSetData
{
    UsdGenNodeKey key;
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

struct UsdGenSurfaceData
{
    UsdGenSurfaceId id = 0;
    VtIntArray     faceVertexCounts, faceVertexIndices;
    // USD subdivision opinions, captured identically by stage/Hydra builders.
    TfToken subdivisionScheme = UsdGenTokens().none, orientation = UsdGenTokens().rightHanded;
    TfToken interpolateBoundary = UsdGenTokens().edgeAndCorner,
        faceVaryingLinearInterpolation = UsdGenTokens().cornersPlus1;
    TfToken triangleSubdivisionRule = UsdGenTokens().catmullClark,
        creaseMethod = UsdGenTokens().uniform;
    VtIntArray holeIndices, creaseIndices, creaseLengths, cornerIndices;
    VtFloatArray creaseSharpnesses, cornerSharpnesses;
    VtVec3fArray   restPoints;        // usdGen/rest/points (S12), UsdTimeCode::Default()
    VtVec3fArray   restNormals;       // usdGen/rest/normals, Default-time Mesh normals
    UsdGenSurfaceNormalDomain restNormalDomain = UsdGenSurfaceNormalDomain::None;
    bool          restFromCurrentPoints = false; // compatibility fallback, never a valid RBF binding
    VtVec3fArray   points;            // deformed, at the graph time
    /// Sorted by time; samples[0].time == the graph time (R23).
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
};

struct UsdGenSurfaceInput : UsdGenSurfaceData
{
    UsdGenNodeKey key;
    UsdGenNodeKey owner;   // the Mesh a GeomSubset belongs to (host-filled); else invalid
};

/// Whether consumers restrict a surface to subsetFaces: a GeomSubset target
/// (even an empty one) or a hand-built input that names faces.
inline bool
UsdGenSurfaceRestricted(UsdGenSurfaceData const &surface)
{
    return surface.isSubset || !surface.subsetFaces.empty();
}

/// The Mesh a surface's geometry and primvars live on: the target itself, or
/// a GeomSubset's parent.
inline UsdGenNodeKey
UsdGenSurfaceMeshKey(UsdGenSurfaceInput const &surface)
{
    return surface.isSubset ? surface.owner : surface.key;
}

struct UsdGenMapData
{
    TfToken      type;                // UsdGenImageMap | UsdGenPtexMap | UsdGenExprMap | ...
    std::string  resolvedAssetPath;   // stage-free (S13)
    uint64_t     textureGeneration = 0;  // bumped by ReloadMaps()
    // UsdGenPaintMap payload (empty unless type == UsdGenPaintMap): the
    // surface primvar snapshotted at capture, scalar-folded through
    // usdGen:map:channel. v1 carries faceVarying float data only; any other
    // interpolation, a missing primvar, or a size mismatch leaves
    // paintValues empty and the sampler fails closed.
    TfToken      paintPrimvar;
    TfToken      paintInterpolation;
    VtFloatArray paintValues;
    std::vector<UsdGenParamValue> params;
    // Optional stage-free decoded pixels for a typed ImageMap consumer. The
    // immutable payload is backend-neutral; callers replacing it must also
    // advance textureGeneration, which is the deterministic cache identity.
    std::shared_ptr<const UsdGenImagePayload> imagePayload;
    UsdGenTypeId typeId = UsdGenTypeId::Invalid;   // engine-filled at intake (UsdGenResolveDescIds)
};

struct UsdGenMapInput : UsdGenMapData
{
    UsdGenNodeKey key;
    UsdGenNodeKey paintSurface;       // UsdGenPaintMap: the surface carrying the primvar
};

/// Compiler-resolved map value. It contains no stage or backend handle: the
/// identity is a deterministic digest of the fully resolved input and can be
/// retained safely by a CPU capture (or rejected by a backend which has no
/// transport for it).
struct UsdGenResolvedMapValue
{
    UsdGenNodeKey key;
    TfToken     type;
    std::string resolvedAssetPath;
    uint64_t    textureGeneration = 0;
    uint64_t    identity = 0;
    UsdGenTypeId typeId = UsdGenTypeId::Invalid;
    UsdGenMapOptions options;            // the map type's parseOptions, run once at compile
};

/// The map's type id: the intake-resolved one, else a registry probe (inputs
/// built outside the compiler). Never interns.
inline UsdGenTypeId UsdGenMapTypeId(UsdGenMapData const &map) noexcept
{
    return map.typeId != UsdGenTypeId::Invalid ? map.typeId
                                               : UsdGenTypeRegistry::Get().Find(map.type);
}

/// True when the map's registered type carries \p trait (kUsdGenTypeMap*).
inline bool UsdGenMapHas(UsdGenMapData const &map, uint32_t trait) noexcept
{
    return UsdGenTypeRegistry::Get().Has(UsdGenMapTypeId(map), trait);
}

/// The description's look block (02-schema.md §2.12 defaults; 07 §1.2 bake
/// order): base colour, the root->tip ramp and the per-curve jitter.
struct UsdGenLookDesc
{
    GfVec3f  rootColor{0.035f, 0.018f, 0.008f};
    GfVec3f  tipColor{0.210f, 0.115f, 0.045f};
    VtVec3fArray rampColors;          // usdGen:look:colorRamp:colors (empty == two-colour mix)
    VtFloatArray rampPositions;       // usdGen:look:colorRamp:positions
    TfToken  rampInterpolation = UsdGenTokens().catmullRom;
    float    rampExponent = 1.0f;     // usdGen:look:rampExponent
    TfToken  bakeMode = UsdGenTokens().perCurve;           // perCurve | perCV
    TfToken  bakeTarget = UsdGenTokens().displayColor;     // displayColor | primvar | none
    TfToken  bakePrimvar = UsdGenTokens().displayColor;
    float    hueJitter = 0.0f;
    float    valueJitter = 0.0f;
    int      jitterSeed = 0;
};

/// usdGen:preview:* on the description: a viewport-only colour override that
/// shows a value on the strands (07 §7.3). 'source' is the first authored
/// target, resolved by the host to a prim and an attribute: a UsdGenExpression
/// prim, a UsdGenPtexMap prim (the builders add it to 'maps') or an operator
/// attribute ('attribute' non-empty). The session cooker resolves it against
/// the compiled graph.
struct UsdGenPreviewSpec
{
    UsdGenNodeKey source;                         // invalid == no preview
    TfToken  attribute;                           // the operator attribute; empty for a prim
    TfToken  colorMap = UsdGenTokens().heat;          // heat | viridis | gray | ids | rgb
    GfVec2f  range{0.0f, 1.0f};
    TfToken  evaluation = UsdGenTokens().primitive;   // for an expression prim source
    TfToken  shading = UsdGenTokens().lit;            // lit | flat

    bool Active() const { return source.IsValid(); }
};

/// The text a key stands for (the host's prim path), for diagnostics only.
/// A derived key's text is its parent's text, '/', then 'text'.
struct UsdGenNodeLabel
{
    UsdGenNodeKey key;
    UsdGenNodeKey parent;     // valid for a derived key
    TfToken       text;
};

/// FNV-1a64 of a path text: the key a host gives a path that collides with
/// nothing (usdGenUsd NodeKeyMap); 1 for a zero hash.
UsdGenNodeKey UsdGenNodeKeyOfText(std::string const &text) noexcept;

struct UsdGenGraphSpec
{
    UsdGenNodeKey                  description;   // the UsdGenDescription prim
    UsdGenNodeKey                  terminal;      // hierarchy-derived final operator
    std::vector<UsdGenOpNode>      nodes;         // composed execution order from the builder
    std::vector<UsdGenCurveSetInput> curveSets;   // every C3 BasisCurves the graph names (R23)
    std::vector<UsdGenSurfaceInput> surfaces;
    std::vector<UsdGenMapInput>    maps;
    std::vector<UsdGenExpressionInput> expressions;
    std::vector<UsdGenGeometryInput> geometries;  // every gprim an expression input names
    UsdGenLookDesc                 look;
    UsdGenPreviewSpec              preview;
    GfMatrix4d                     xformMatrix{1.0}; // description world matrix (post-flattening, S4)
    TfToken                        purpose;        // inherited by hand to every tile (C2)
    TfToken                        visibility;     // inherited by hand to every tile (C2)
    UsdGenNodeKey                  material;       // the description's bound Material (C2)
    float    defaultWidth = 0.01f;                 // usdGen:width:default
    int      tileTarget = 64;                      // uniform int usdGen:tileTarget
    TfToken  curveBasis = UsdGenTokens().bspline;      // usdGen:curve:basis (C2)
    double   time = 0.0;
    double   timeCodesPerSecond = 24.0; // USD default; expressions expose seconds
    UsdGenExecutionBackend executionBackend = UsdGenExecutionBackend::CpuReference;
    std::vector<std::string> validationErrors;
    /// Sorted by key: the text of every key above.
    std::vector<UsdGenNodeLabel> labels;
    /// Opaque host data (usdGenUsd: this spec's key -> path table). The engine
    /// copies it into the generations and routing snapshots it makes from this
    /// spec and never reads it.
    std::shared_ptr<const void> host;

    /// The text of 'key' (its prim path), else empty. Allocates: diagnostics only.
    std::string Label(UsdGenNodeKey key) const;
    /// The host label token of 'key' without allocating; empty for a
    /// derived or unknown key.
    TfToken const &LabelToken(UsdGenNodeKey key) const noexcept;
    /// The child name of a derived key (DeriveKey); empty for a host key.
    TfToken const &DerivedName(UsdGenNodeKey key) const noexcept;
    /// The description's path text (expr::DescriptionId hashes it).
    char const *DescriptionText() const noexcept { return LabelToken(description).GetText(); }
    /// The key of a synthetic child of 'parent' (an engine-made node or
    /// input), keyed like the host keys the path <parent>/<child>, and its
    /// label. Compile phase: allocates.
    UsdGenNodeKey DeriveKey(UsdGenNodeKey parent, TfToken const &child);
    /// Label(a) < Label(b): the order of the host's paths (prim paths order as
    /// their text).
    bool LabelLess(UsdGenNodeKey a, UsdGenNodeKey b) const;
};

/// spec->Label(key), or empty without a spec.
inline std::string UsdGenLabel(UsdGenGraphSpec const *spec, UsdGenNodeKey key)
{
    return spec ? spec->Label(key) : std::string();
}

}  // namespace usdGen

#endif  // USDGEN_API_GRAPH_SPEC_H
