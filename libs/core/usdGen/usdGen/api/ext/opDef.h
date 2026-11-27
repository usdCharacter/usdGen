// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// What an op library tells the engine about one op: plain data plus hooks.
#pragma once
#include "usdGen/api/export.h"
#include "usdGen/api/span.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/valueShape.h"        // expr::Domain, expr::ScalarType
#include "usdGen/api/types.h"                 // UsdGenCountsAtLevel, UsdGenLane

#include "usdGen/api/span.h"
#include <cstdint>
#include <memory>
#include <string>

namespace usdGen {

class UsdGenOp;
class UsdGenOpRecord;
struct UsdGenParamView;
struct UsdGenGraphSpec;
struct UsdGenOpNode;
struct UsdGenDeformBinding;
struct UsdGenInputRecord;

/// Dirty class of one parameter (02 §6). Structural == Merkle digest term.
enum class UsdGenParamClass : uint8_t { Structural, Topology, Capture, Value };

/// What an expression may drive. domains == None: not connectable.
struct UsdGenExprAdmission {
    expr::ScalarType scalar = expr::ScalarType::Invalid;
    uint8_t components = 1;
    char const *nativeType = nullptr;    // "bool", "int", "float", "float2", "vector3f"
    expr::Domain domains = expr::Domain::None;
};

struct UsdGenParamDecl {
    char const *name;                    // C1 name without "usdGen:", e.g. "length:mode"
    UsdGenParamClass cls;
    bool routed = true;                  // appears in the node's dirty-routing table
    UsdGenExprAdmission expr{};
    UsdGenSpan<char const *const> enumerants{};   // token params read with GetEnum
};

enum class UsdGenPlaneRate : uint8_t { Vertex, Uniform, Constant };
enum class UsdGenPlaneScalar : uint8_t { Float, Int };

struct UsdGenPlaneDecl {
    char const *name;
    UsdGenPlaneRate rate;
    UsdGenPlaneScalar scalar;
    uint8_t arity = 1;
};

struct UsdGenMapSlotDecl {
    char const *relationship;            // "usdGen:map", "usdGen:lengthMap", ...
};

/// Receives emitted planes during compile; interns names (compile scope).
class USDGEN_API UsdGenPlaneSink {
public:
    virtual void Add(std::string const &name, UsdGenPlaneRate, UsdGenPlaneScalar,
                     uint8_t arity) = 0;
protected:
    ~UsdGenPlaneSink() = default;
};

/// A run of indexed plane names (clumpId_<base + i>) a node needs.
struct UsdGenIndexedPlaneRequest {
    int count = 0;                       // 0: none; Clump 1..4
    int pinnedBase = -1;                 // >= 0: authored usdGen:clump:level
};

/// Facts the engine computes for the animation-binding hook (4a).
struct UsdGenAnimationBindingQuery {
    UsdGenOpNode const *terminalDesc = nullptr;
    UsdGenParamView const *terminalParams = nullptr;
    UsdGenInputRecord const *surface = nullptr;    // null: terminal binds no surface
    uint32_t terminalInputCount = 0;
    bool terminalHasOutputPlanes = false;
    bool graphStatic = false;            // no ReadsTime op, no expression binding, no animated param
    uint32_t nodesWithTerminalOp = 0;    // nodes with the terminal's OpId, terminal included
};

struct UsdGenOpHooks {
    /// Compile-time node validation beyond arity (WidthBlend, compiler.cpp:1294-1319).
    bool (*validateNode)(UsdGenGraphSpec const &, UsdGenOpNode const &,
                         UsdGenParamView const &, std::string *error) = nullptr;
    /// 4a. Indexed plane run this node needs.
    UsdGenIndexedPlaneRequest (*indexedPlanes)(UsdGenParamView const &) = nullptr;
    /// 4a. Output planes for this node; indexedBase is -1 without a request.
    void (*emittedPlanes)(UsdGenParamView const &, int indexedBase, UsdGenPlaneSink &) = nullptr;
    /// Roots carry limit-surface UVs (Scatter with subdivisionLevel > 0).
    bool (*emitsLimitSurfaceUv)(UsdGenParamView const &) = nullptr;
    /// Evaluate remaps every inherited vertex plane for this node's params
    /// (Collide in cutThenCollide); ORed with kUsdGenOpRemapsVertexPlanes.
    bool (*remapsVertexPlanes)(UsdGenParamView const &) = nullptr;
    /// 4a. Fill sampleIndices and field; false: no binding.
    bool (*makeAnimationBinding)(UsdGenAnimationBindingQuery const &, UsdGenOp const &,
                                 UsdGenDeformBinding *out) = nullptr;
    /// The diagnostic for a nonzero UsdGenEvalContext::failureCode; null:
    /// the generic "operator evaluation failed".
    char const *(*failureMessage)(uint32_t code) = nullptr;
};

enum UsdGenOpTrait : uint32_t {
    kUsdGenOpGenerator                = 1u << 0,   // op.h:395
    kUsdGenOpReadsTime                = 1u << 1,   // op.h:332
    kUsdGenOpSamplesFrameInputs       = 1u << 2,   // op.h:335
    kUsdGenOpGeometryInputsOrdered    = 1u << 3,   // op.h:362
    kUsdGenOpCaptureDigestComplete    = 1u << 4,   // op.h:372
    kUsdGenOpRemapsVertexPlanes       = 1u << 5,   // op.h:352
    kUsdGenOpCaptureOwnsRestAndPlanes = 1u << 6,   // scheduler.cpp:386-392, 440-446
    kUsdGenOpRecordsChunkExtents      = 1u << 7,   // scheduler.cpp:1031
    kUsdGenOpPosesSpace               = 1u << 8,   // scheduler.cpp:1076
    kUsdGenOpFollowsInputUntilOwning  = 1u << 9,   // graph.cpp:438
    kUsdGenOpFixedOutputPlaneLayout   = 1u << 10,  // scheduler.cpp:228-236
};

/// UsdGenSpan has no C-array constructor: `d.params = UsdGenSpan(kParams);`
/// (api/span.h) converts through UsdGenSpan's container constructor.

using UsdGenOpCreateFn = std::unique_ptr<UsdGenOp> (*)(UsdGenOpRecord const &, void *context);

/// Filled by assignment inside one register function and copied by
/// RegisterOp. Every char const* is a literal or lives as long as the process.
struct UsdGenOpDef {
    char const *name = nullptr;          // C1 schema type name
    char const *displayName = nullptr;   // diagnostics, e.g. "Length"
    UsdGenOpCreateFn create = nullptr;
    void *createContext = nullptr;
    uint8_t countChanges = 0;                // bit L: element counts at level L may change
    UsdGenLane lane = UsdGenLane::Chunked;
    uint8_t geometryInputArity = 1;
    /// Typed geometry ports by registered type name; null is the default
    /// primitive. Every geometry input accepts inputTypes (empty: the default
    /// only; a null entry adds the default). The compiler rejects other edges.
    char const *outputType = nullptr;
    UsdGenSpan<char const *const> inputTypes;
    uint32_t traits = 0;
    UsdGenSpan<const UsdGenParamDecl> params;
    UsdGenSpan<char const *const> referenceInputs;      // was ReferenceInputs(), op.h:344
    UsdGenSpan<const UsdGenPlaneDecl> inputPlanes;      // was InputPrimvars(), op.h:348
    UsdGenSpan<const UsdGenPlaneDecl> outputPlanes;     // static outputs; unused with hooks.emittedPlanes
    char const *indexedPlaneFamily = nullptr;       // "clumpId_": authored-collision prefix
    UsdGenSpan<const UsdGenMapSlotDecl> mapSlots;
    UsdGenSpan<char const *const> tokens;               // other names the op compares; Record().Token(i)
    UsdGenOpHooks hooks;
};

}  // namespace usdGen
