#include "usdGen/expressions/samplers.h"

#include "usdGen/expressions/irExec.h"

#include <nanoflann.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <set>

namespace usdGen::expr {
namespace {

std::atomic<uint64_t> s_buildGeneration{0};

struct ElementCloud {
    std::vector<double> const *points = nullptr;
    size_t kdtree_get_point_count() const { return points->size() / 3; }
    double kdtree_get_pt(size_t index, size_t dimension) const
    {
        return (*points)[index * 3 + dimension];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX &) const { return false; }
};
using ElementIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, ElementCloud>, ElementCloud, 3, uint32_t>;

bool Fail(std::string *error, std::string const &message)
{
    if (error) *error = message;
    return false;
}

bool Finite(std::vector<float> const &values)
{
    for (float v : values)
        if (!std::isfinite(v)) return false;
    return true;
}

void Push3(std::vector<double> &out, float const *p)
{
    out.push_back(p[0]); out.push_back(p[1]); out.push_back(p[2]);
}
void Push3(std::vector<double> &out, double x, double y, double z)
{
    out.push_back(x); out.push_back(y); out.push_back(z);
}

/// Area-weighted vertex normals and Newell face normals of a polygon mesh.
void MeshNormals(SamplerGeometrySource const &mesh, std::vector<double> *vertex,
                 std::vector<double> *face)
{
    size_t const pointCount = mesh.points.size() / 3;
    vertex->assign(pointCount * 3, 0.0);
    face->clear();
    face->reserve(mesh.counts.size() * 3);
    size_t corner = 0;
    for (int count : mesh.counts) {
        double n[3] = {0, 0, 0};
        for (int k = 0; k < count; ++k) {
            float const *a = &mesh.points[size_t(mesh.indices[corner + k]) * 3];
            float const *b = &mesh.points[size_t(mesh.indices[corner + (k + 1) % count]) * 3];
            n[0] += (double(a[1]) - b[1]) * (double(a[2]) + b[2]);
            n[1] += (double(a[2]) - b[2]) * (double(a[0]) + b[0]);
            n[2] += (double(a[0]) - b[0]) * (double(a[1]) + b[1]);
        }
        // The un-normalised Newell vector is twice the area vector, which is
        // exactly the area weight a vertex normal wants.
        for (int k = 0; k < count; ++k) {
            size_t const v = size_t(mesh.indices[corner + k]) * 3;
            (*vertex)[v] += n[0]; (*vertex)[v + 1] += n[1]; (*vertex)[v + 2] += n[2];
        }
        double const length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (length > 0.0) Push3(*face, n[0] / length, n[1] / length, n[2] / length);
        else Push3(*face, 0.0, 1.0, 0.0);
        corner += size_t(count);
    }
    for (size_t v = 0; v < vertex->size(); v += 3) {
        double *n = vertex->data() + v;
        double const length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (length > 0.0) { n[0] /= length; n[1] /= length; n[2] /= length; }
        else { n[0] = 0.0; n[1] = 1.0; n[2] = 0.0; }
    }
}

bool ValidateSource(SamplerGeometrySource const &s, size_t index, std::string *error)
{
    std::string const where = "geoSampler input geometry " + std::to_string(index);
    if (s.points.size() % 3 != 0) return Fail(error, where + " has a malformed point array");
    size_t const pointCount = s.points.size() / 3;
    if (!Finite(s.points)) return Fail(error, where + " has non-finite points");
    if (!s.rest.empty() && (s.rest.size() != s.points.size() || !Finite(s.rest)))
        return Fail(error, where + " has a malformed rest point array");
    if (!s.normals.empty() && (s.normals.size() != s.points.size() || !Finite(s.normals)))
        return Fail(error, where + " has a malformed normal array");
    if ((s.subset || !s.faces.empty()) && s.kind != SamplerGeometrySource::Kind::Mesh)
        return Fail(error, where + " restricts faces but is not a mesh");
    for (int face : s.faces)
        if (face < 0 || size_t(face) >= s.counts.size())
            return Fail(error, where + " has a subset face out of range");
    switch (s.kind) {
    case SamplerGeometrySource::Kind::Points:
        if (!s.ids.empty() && s.ids.size() != pointCount)
            return Fail(error, where + " has an id per point mismatch");
        return true;
    case SamplerGeometrySource::Kind::Curves: {
        size_t total = 0;
        for (int count : s.counts) {
            if (count < 1) return Fail(error, where + " has an empty curve");
            total += size_t(count);
        }
        if (total != pointCount) return Fail(error, where + " has inconsistent curve vertex counts");
        if (!s.ids.empty() && s.ids.size() != s.counts.size())
            return Fail(error, where + " has an id per curve mismatch");
        return true;
    }
    case SamplerGeometrySource::Kind::Mesh: {
        size_t total = 0;
        for (int count : s.counts) {
            if (count < 1) return Fail(error, where + " has an empty face");
            total += size_t(count);
        }
        if (total != s.indices.size()) return Fail(error, where + " has inconsistent face vertex counts");
        for (int v : s.indices)
            if (v < 0 || size_t(v) >= pointCount)
                return Fail(error, where + " has a face vertex index out of range");
        return true;
    }
    }
    return Fail(error, where + " has an unknown geometry kind");
}

} // namespace

struct GeometrySampler::Impl {
    IRSampler spec;
    Context controls;
    size_t count = 0;
    uint64_t generation = 0;
    // Point-rate element fields over the element table.
    std::vector<double> P, PRef, rootP, rootPRef, N, id, idLo, idHi, primIndex, primCount,
        pointIndex, pointCount, t, cLength;
    CpuExpressionInputs inputs;
    ElementCloud cloud;
    std::unique_ptr<ElementIndex> index;
    bool queryDependent = false;
    std::vector<double> constant;   // count * components, when !queryDependent
    double folded[4]{};             // the query-independent reduction

    uint16_t Output(unsigned c) const
    {
        IRProgram const &program = *spec.element;
        return program.outputCount ? program.output[c] : program.result;
    }

    /// Evaluates the element expression at element `e`; `in` must already
    /// carry $Q and the storage $Qdist reads through `qdist`.
    void Evaluate(CpuExpressionInputs const &in, size_t e, double *qdist, double distance,
                  double *out) const
    {
        *qdist = distance;
        double registers[kExprRegisters];
        IRProgram const &program = *spec.element;
        ExecuteElement(program.instructions.data(), program.instructions.size(), registers,
                       in, e);
        for (unsigned c = 0; c < spec.components; ++c) out[c] = registers[Output(c)];
    }

    double Distance(double const q[3], size_t e) const
    {
        double const dx = P[e * 3] - q[0], dy = P[e * 3 + 1] - q[1], dz = P[e * 3 + 2] - q[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    void Fold(double const *value, bool first, double *accumulator) const
    {
        for (unsigned c = 0; c < spec.components; ++c) {
            double &a = accumulator[c];
            switch (spec.reduce) {
            case SampleReduce::Min: a = first ? value[c] : std::min(a, value[c]); break;
            case SampleReduce::Max: a = first ? value[c] : std::max(a, value[c]); break;
            case SampleReduce::Sum:
            case SampleReduce::Mean: a = first ? value[c] : a + value[c]; break;
            default: break;
            }
            // min/max would quietly drop a NaN; the language poisons instead.
            if (!std::isfinite(value[c])) a = NAN;
        }
    }

    void Compute(double const q[3], double *out) const
    {
        for (unsigned c = 0; c < 4; ++c) out[c] = NAN;
        if (count == 0 || !std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(q[2]))
            return;
        CpuExpressionInputs in = inputs;
        in.fields[unsigned(Variable::Q)] = {q, 1, Domain::Groom, 3};
        double qdist = 0.0;
        in.fields[unsigned(Variable::QDist)] = {&qdist, 1, Domain::Groom, 1};

        if (spec.reduce == SampleReduce::Nearest || spec.reduce == SampleReduce::Nearest2) {
            size_t const want = spec.reduce == SampleReduce::Nearest ? 1 : std::min<size_t>(2, count);
            uint32_t indices[2]{};
            double distances[2]{};
            size_t const found = index->knnSearch(q, want, indices, distances);
            if (found == 0) return;
            // With a single element "second nearest" degrades to the nearest,
            // so f2 - f1 reads 0 instead of poisoning a one-guide input.
            size_t const e = indices[found - 1];
            if (!queryDependent) {
                for (unsigned c = 0; c < spec.components; ++c)
                    out[c] = constant[e * spec.components + c];
                return;
            }
            Evaluate(in, e, &qdist, std::sqrt(distances[found - 1]), out);
            return;
        }
        if (!queryDependent) {
            for (unsigned c = 0; c < spec.components; ++c) out[c] = folded[c];
            return;
        }
        double value[4]{};
        double accumulator[4]{};
        for (size_t e = 0; e < count; ++e) {
            Evaluate(in, e, &qdist, Distance(q, e), value);
            Fold(value, e == 0, accumulator);
        }
        for (unsigned c = 0; c < spec.components; ++c)
            out[c] = spec.reduce == SampleReduce::Mean ? accumulator[c] / double(count)
                                                       : accumulator[c];
    }
};

GeometrySampler::GeometrySampler() : _impl(new Impl) {}
GeometrySampler::~GeometrySampler() = default;

size_t GeometrySampler::ElementCount() const noexcept { return _impl->count; }
unsigned GeometrySampler::Components() const noexcept { return _impl->spec.components; }

bool GeometrySampler::Build(std::vector<SamplerGeometrySource> const &sources,
                            IRSampler const &spec, Context const &controls,
                            std::string *error)
{
    auto fresh = std::make_unique<Impl>();
    Impl &impl = *fresh;
    impl.spec = spec;
    impl.controls = controls;
    impl.generation = ++s_buildGeneration;
    if (spec.kind != SamplerKind::Geometry || !spec.element ||
        (spec.components != 1 && spec.components != 3))
        return Fail(error, "geoSampler has no valid element expression");
    if (!ValidProgram(*spec.element) || !spec.element->samplers.empty())
        return Fail(error, "geoSampler element expression is not a valid program");
    for (size_t s = 0; s < sources.size(); ++s)
        if (!ValidateSource(sources[s], s, error)) return false;

    bool hasN = true, curvesOnly = !sources.empty();
    size_t primTotal = 0;
    for (auto const &source : sources) {
        size_t const prims = source.kind == SamplerGeometrySource::Kind::Points
            ? source.points.size() / 3 : source.counts.size();
        primTotal += prims;
        if (source.kind != SamplerGeometrySource::Kind::Curves) curvesOnly = false;
        switch (spec.iterate) {
        case SampleIterate::Point:
        case SampleIterate::Prim:
            if (source.kind == SamplerGeometrySource::Kind::Curves ||
                (source.kind == SamplerGeometrySource::Kind::Points && source.normals.empty()))
                hasN = false;
            break;
        case SampleIterate::Geometry:
            hasN = false;
            break;
        }
    }
    if (spec.iterate == SampleIterate::Geometry) {
        curvesOnly = false;
        primTotal = sources.size();
    }
    if (sources.empty()) hasN = false;

    // --- the element table -------------------------------------------------
    auto element = [&](float const *p, float const *rest, float const *root,
                       float const *rootRest, double n[3], uint64_t stableId, size_t prim,
                       size_t pointInPrim, size_t pointsInPrim, double param, double length) {
        Push3(impl.P, p);
        Push3(impl.PRef, rest ? rest : p);
        Push3(impl.rootP, root);
        Push3(impl.rootPRef, rootRest ? rootRest : root);
        if (hasN) Push3(impl.N, n ? n[0] : 0.0, n ? n[1] : 1.0, n ? n[2] : 0.0);
        impl.id.push_back(double(stableId));
        impl.idLo.push_back(double(uint32_t(stableId)));
        impl.idHi.push_back(double(uint32_t(stableId >> 32)));
        impl.primIndex.push_back(double(prim));
        impl.primCount.push_back(double(primTotal));
        impl.pointIndex.push_back(double(pointInPrim));
        impl.pointCount.push_back(double(pointsInPrim));
        impl.t.push_back(param);
        if (curvesOnly) impl.cLength.push_back(length);
    };

    size_t primBase = 0;
    for (size_t s = 0; s < sources.size(); ++s) {
        SamplerGeometrySource const &src = sources[s];
        float const *pts = src.points.data();
        float const *rest = src.rest.empty() ? nullptr : src.rest.data();
        size_t const pointCount = src.points.size() / 3;
        auto restAt = [&](size_t i) { return rest ? rest + i * 3 : nullptr; };

        // A mesh subset visits only its faces' corners (offsets[f] ..
        // offsets[f + 1]) and the points they use.
        std::vector<size_t> offsets;
        std::vector<char> faceUsed, pointUsed;
        if (src.kind == SamplerGeometrySource::Kind::Mesh) {
            offsets.assign(src.counts.size() + 1, 0);
            for (size_t f = 0; f < src.counts.size(); ++f)
                offsets[f + 1] = offsets[f] + size_t(src.counts[f]);
            if (src.subset) {
                faceUsed.assign(src.counts.size(), 0);
                pointUsed.assign(pointCount, 0);
                for (int f : src.faces) {
                    faceUsed[size_t(f)] = 1;
                    for (size_t c = offsets[size_t(f)]; c < offsets[size_t(f) + 1]; ++c)
                        pointUsed[size_t(src.indices[c])] = 1;
                }
            }
        }
        auto visitsPoint = [&](size_t i) { return pointUsed.empty() || pointUsed[i]; };
        auto visitsFace = [&](size_t f) { return faceUsed.empty() || faceUsed[f]; };

        if (spec.iterate == SampleIterate::Geometry) {
            double centroid[3]{}, restCentroid[3]{};
            size_t visited = 0;
            for (size_t i = 0; i < pointCount; ++i) {
                if (!visitsPoint(i)) continue;
                ++visited;
                for (int d = 0; d < 3; ++d) {
                    centroid[d] += pts[i * 3 + d];
                    restCentroid[d] += (rest ? rest : pts)[i * 3 + d];
                }
            }
            if (src.subset && !visited) continue;  // an empty subset is no element
            float c[3]{}, rc[3]{};
            for (int d = 0; d < 3; ++d) {
                c[d] = visited ? float(centroid[d] / double(visited)) : 0.0f;
                rc[d] = visited ? float(restCentroid[d] / double(visited)) : 0.0f;
            }
            element(c, rc, c, rc, nullptr, uint64_t(s), s, 0, visited, 0.0, 0.0);
            continue;
        }

        switch (src.kind) {
        case SamplerGeometrySource::Kind::Points:
            for (size_t i = 0; i < pointCount; ++i) {
                double n[3]{};
                if (!src.normals.empty())
                    for (int d = 0; d < 3; ++d) n[d] = src.normals[i * 3 + d];
                uint64_t const stable = src.ids.empty() ? uint64_t(primBase + i) : src.ids[i];
                element(pts + i * 3, restAt(i), pts + i * 3, restAt(i),
                        src.normals.empty() ? nullptr : n, stable, primBase + i, 0, 1, 0.0, 0.0);
            }
            primBase += pointCount;
            break;
        case SamplerGeometrySource::Kind::Curves: {
            size_t first = 0;
            for (size_t c = 0; c < src.counts.size(); ++c) {
                size_t const n = size_t(src.counts[c]);
                double length = 0.0;
                for (size_t i = 1; i < n; ++i) {
                    float const *a = pts + (first + i - 1) * 3, *b = pts + (first + i) * 3;
                    double const dx = double(b[0]) - a[0], dy = double(b[1]) - a[1],
                                 dz = double(b[2]) - a[2];
                    length += std::sqrt(dx * dx + dy * dy + dz * dz);
                }
                uint64_t const stable = src.ids.empty() ? uint64_t(primBase + c) : src.ids[c];
                size_t const visit = spec.iterate == SampleIterate::Point ? n : 1;
                for (size_t i = 0; i < visit; ++i) {
                    double const param = n > 1 ? double(i) / double(n - 1) : 0.0;
                    element(pts + (first + i) * 3, restAt(first + i), pts + first * 3,
                            restAt(first), nullptr, stable, primBase + c, i, n, param, length);
                }
                first += n;
            }
            primBase += src.counts.size();
            break;
        }
        case SamplerGeometrySource::Kind::Mesh: {
            std::vector<double> vertexNormals, faceNormals;
            MeshNormals(src, &vertexNormals, &faceNormals);
            if (spec.iterate == SampleIterate::Point) {
                std::vector<size_t> owner(pointCount, primBase);
                std::vector<char> seen(pointCount, 0);
                for (size_t f = 0; f < src.counts.size(); ++f) {
                    if (!visitsFace(f)) continue;
                    for (size_t c = offsets[f]; c < offsets[f + 1]; ++c) {
                        size_t const v = size_t(src.indices[c]);
                        if (!seen[v]) { seen[v] = 1; owner[v] = primBase + f; }
                    }
                }
                for (size_t i = 0; i < pointCount; ++i)
                    if (visitsPoint(i))
                        element(pts + i * 3, restAt(i), pts + i * 3, restAt(i),
                                vertexNormals.data() + i * 3, uint64_t(i), owner[i], 0, 1,
                                0.0, 0.0);
            } else {
                for (size_t f = 0; f < src.counts.size(); ++f) {
                    if (!visitsFace(f)) continue;
                    size_t const corner = offsets[f];
                    int const n = src.counts[f];
                    double centroid[3]{}, restCentroid[3]{};
                    for (int k = 0; k < n; ++k) {
                        size_t const v = size_t(src.indices[corner + size_t(k)]);
                        for (int d = 0; d < 3; ++d) {
                            centroid[d] += pts[v * 3 + d];
                            restCentroid[d] += (rest ? rest : pts)[v * 3 + d];
                        }
                    }
                    float c[3]{}, rc[3]{};
                    for (int d = 0; d < 3; ++d) {
                        c[d] = float(centroid[d] / double(n));
                        rc[d] = float(restCentroid[d] / double(n));
                    }
                    element(c, rc, c, rc, faceNormals.data() + f * 3, uint64_t(f),
                            primBase + f, 0, size_t(n), 0.0, 0.0);
                }
            }
            primBase += src.counts.size();
            break;
        }
        }
    }
    impl.count = impl.P.size() / 3;
    if (impl.count > size_t(std::numeric_limits<uint32_t>::max()))
        return Fail(error, "geoSampler input has too many elements");

    // --- what the element expression may read ------------------------------
    std::set<Variable> available{
        Variable::Frame, Variable::Time, Variable::Index, Variable::Count, Variable::Seed,
        Variable::DescId, Variable::P, Variable::PRef, Variable::RootP, Variable::RootPRef,
        Variable::Id, Variable::IdLo, Variable::IdHi, Variable::PrimIndex,
        Variable::PrimCount, Variable::PointIndex, Variable::PointCount, Variable::T,
        Variable::Q, Variable::QDist};
    if (hasN) { available.insert(Variable::N); available.insert(Variable::NRef); }
    if (curvesOnly) available.insert(Variable::CLength);
    for (IRInstruction const &op : spec.element->instructions) {
        if (op.op != IROp::LoadVariable) continue;
        if (!available.count(op.variable)) {
            VariableInfo const *info = Registry::Get().Find(op.variable);
            return Fail(error, std::string("geoSampler(\"") + spec.input + "\", ...): variable " +
                (info ? info->name : "?") + " is not available when iterating " +
                SampleIterateName(spec.iterate) + " elements of this geometry");
        }
        if (op.variable == Variable::Q || op.variable == Variable::QDist)
            impl.queryDependent = true;
    }

    // --- the evaluation inputs ---------------------------------------------
    impl.inputs = CpuExpressionInputs{};
    impl.inputs.context = controls;
    impl.inputs.context.domain = Domain::Point;
    impl.inputs.context.count = static_cast<uint32_t>(impl.count);
    impl.inputs.count = impl.count;
    impl.inputs.primitiveCount = impl.count;
    auto bind = [&](Variable v, std::vector<double> const &data, unsigned components) {
        if (data.empty()) return;
        impl.inputs.fields[unsigned(v)] = {data.data(), impl.count, Domain::Point, components};
    };
    bind(Variable::P, impl.P, 3);
    bind(Variable::PRef, impl.PRef, 3);
    bind(Variable::RootP, impl.rootP, 3);
    bind(Variable::RootPRef, impl.rootPRef, 3);
    bind(Variable::N, impl.N, 3);
    bind(Variable::NRef, impl.N, 3);
    bind(Variable::Id, impl.id, 1);
    bind(Variable::IdLo, impl.idLo, 1);
    bind(Variable::IdHi, impl.idHi, 1);
    bind(Variable::PrimIndex, impl.primIndex, 1);
    bind(Variable::PrimCount, impl.primCount, 1);
    bind(Variable::PointIndex, impl.pointIndex, 1);
    bind(Variable::PointCount, impl.pointCount, 1);
    bind(Variable::T, impl.t, 1);
    bind(Variable::CLength, impl.cLength, 1);

    // --- the query-independent values ---------------------------------------
    if (!impl.queryDependent && impl.count) {
        double const q[3] = {0.0, 0.0, 0.0};
        CpuExpressionInputs in = impl.inputs;
        in.fields[unsigned(Variable::Q)] = {q, 1, Domain::Groom, 3};
        double qdist = 0.0;
        in.fields[unsigned(Variable::QDist)] = {&qdist, 1, Domain::Groom, 1};
        impl.constant.resize(impl.count * spec.components);
        for (size_t e = 0; e < impl.count; ++e)
            impl.Evaluate(in, e, &qdist, 0.0, impl.constant.data() + e * spec.components);
        bool const folds = spec.reduce != SampleReduce::Nearest &&
                           spec.reduce != SampleReduce::Nearest2;
        if (folds) {
            for (size_t e = 0; e < impl.count; ++e)
                impl.Fold(impl.constant.data() + e * spec.components, e == 0, impl.folded);
            if (spec.reduce == SampleReduce::Mean)
                for (unsigned c = 0; c < spec.components; ++c)
                    impl.folded[c] /= double(impl.count);
        }
    }

    // --- the proximity index -------------------------------------------------
    impl.cloud.points = &impl.P;
    if (impl.count) {
        impl.index = std::make_unique<ElementIndex>(
            3, impl.cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    }
    _impl = std::move(fresh);
    // The element fields point into this Impl's own vectors, which moved with
    // the unique_ptr, so the addresses stay valid.
    return true;
}

double GeometrySampler::Sample(double const q[3], unsigned component) const
{
    Impl const &impl = *_impl;
    if (component >= impl.spec.components) return NAN;
    // One call per component of the same query is the common pattern (the IR
    // reads a vector sample component by component), so keep the last result.
    thread_local struct {
        uint64_t generation = 0;
        double q[3]{};
        double value[4]{};
    } last;
    if (last.generation == impl.generation && last.q[0] == q[0] && last.q[1] == q[1] &&
        last.q[2] == q[2])
        return last.value[component];
    impl.Compute(q, last.value);
    last.generation = impl.generation;
    last.q[0] = q[0]; last.q[1] = q[1]; last.q[2] = q[2];
    return last.value[component];
}

double SampleSlot(CpuExpressionSamplers const &samplers, CpuExpressionInputs const &inputs,
                  unsigned slot, double const *args, unsigned argCount, size_t index,
                  unsigned component)
{
    if (slot >= samplers.slots.size()) return NAN;
    CpuExpressionSamplers::Slot const &s = samplers.slots[slot];
    if (component >= s.components) return NAN;
    switch (s.kind) {
    case SamplerKind::Geometry: {
        if (!s.geometry || argCount != 3) return NAN;
        double const q[3] = {args[0], args[1], args[2]};
        return s.geometry->Sample(q, component);
    }
    case SamplerKind::Ptex: {
        size_t strand = index;
        if (inputs.context.domain == Domain::Point) {
            if (!inputs.pointToPrimitive.data || index >= inputs.pointToPrimitive.size)
                return NAN;
            strand = inputs.pointToPrimitive.data[index];
        } else if (inputs.context.domain != Domain::Primitive) {
            return NAN;
        }
        if (!s.values || strand >= s.count) return NAN;
        return s.values[strand * s.components + component];
    }
    }
    return NAN;
}

} // namespace usdGen::expr
