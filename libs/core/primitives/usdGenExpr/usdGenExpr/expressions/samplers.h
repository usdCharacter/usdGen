// usdGen — the host-side data an expression may sample: geoSampler() over
// external geometry, and per-strand map values (ptex()).
//
// An expression names external data only through the `input:<name>`
// relationships of its UsdGenExpression prim. The frontend lowers each such
// call to IROp::Sample with a slot index; the CPU parameter evaluator resolves
// every slot once per compile (cpuParameters.cpp) and hands the table to
// irExec.h's interpreter through CpuExpressionInputs::samplers.
//
// geoSampler(input, expression [, iterate [, reduce [, query]]])
//   iterate  "point" | "prim" | "geometry"   (default "prim")
//   reduce   "nearest" | "nearest2" | "min" | "max" | "sum" | "mean"
//            (default "nearest")
//   query    a position; defaults to the calling element's $P
// The element expression is one SeExpr expression evaluated per element of
// the input. It reads the element's own variables ($P, $Pref, $N, $index,
// $count, $id, $primIndex, $t, $rootP, ...) plus the query position $Q and
// the element's distance to it, $Qdist. `nearest`/`nearest2` evaluate it at
// the nearest/second-nearest element only; the other reductions fold it over
// every element. An element expression that reads neither $Q nor $Qdist is
// evaluated once per build and then only looked up.
//
// Deliberately free of pxr types, like cpuEvaluator.h.
#ifndef USDGEN_EXPRESSIONS_SAMPLERS_H
#define USDGEN_EXPRESSIONS_SAMPLERS_H

#include "usdGen/expressions/cpuEvaluator.h"
#include "usdGen/expressions/ir.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace usdGen::expr {

/// One gprim a geoSampler() iterates, already in the sampling space (the space
/// the groom is generated in).
struct SamplerGeometrySource {
    enum class Kind : uint8_t { Mesh, Curves, Points };
    Kind kind = Kind::Points;
    std::vector<float> points;      // 3 * pointCount, at the cook time
    std::vector<float> rest;        // empty, or 3 * pointCount at Default time
    std::vector<float> normals;     // empty, or 3 * pointCount (points only)
    std::vector<int> counts;        // mesh: faceVertexCounts; curves: curveVertexCounts
    std::vector<int> indices;       // mesh: faceVertexIndices
    std::vector<uint64_t> ids;      // optional stable id per prim (curve / point)
    // Mesh only: a GeomSubset input (R15) visits just `faces` (sorted parent
    // face ids) and the points they use; ids and prim indices stay the
    // parent mesh's, so widening the subset renumbers nothing.
    bool subset = false;
    std::vector<int> faces;
};

class GeometrySampler {
public:
    GeometrySampler();
    ~GeometrySampler();
    GeometrySampler(GeometrySampler const &) = delete;
    GeometrySampler &operator=(GeometrySampler const &) = delete;

    /// Builds the element table for `spec.iterate` over `sources` (their order
    /// is the element order) and the proximity index. `controls` supplies
    /// $frame/$time/$seed/$descId. Fails with a diagnostic for malformed
    /// geometry, or for an element expression that reads a variable this
    /// geometry cannot supply.
    bool Build(std::vector<SamplerGeometrySource> const &sources, IRSampler const &spec,
               Context const &controls, std::string *error);

    /// The sampled value at query position `q`. NaN when the input has no
    /// elements. Thread-safe.
    double Sample(double const q[3], unsigned component) const;

    size_t ElementCount() const noexcept;
    unsigned Components() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/// The table IROp::Sample dispatches through: one slot per
/// IRProgram::samplers entry, in the same order.
class CpuExpressionSamplers {
public:
    struct Slot {
        SamplerKind kind = SamplerKind::Geometry;
        GeometrySampler const *geometry = nullptr;   // Geometry
        // Ptex (and any other per-strand value): `count` primitives times
        // `components` values, indexed by the owning strand.
        double const *values = nullptr;
        size_t count = 0;
        unsigned components = 1;
    };
    std::vector<Slot> slots;
};

} // namespace usdGen::expr
#endif
