// usdGen engine — session implementation (03 §5.4/§5.6/§6.1/§9.1).
//
// Private serial work-owner implementation. The command owner supplies a
// copied request and alone decides whether this candidate may publish.
#include "usdGen/sessionCooker.h"
#include "usdGen/limitSurface.h"
#include "usdGen/debugCodes.h"
#include "usdGen/digest.h"
#include "usdGen/furOcclusion.h"
#include "usdGen/cudaExecution.h"
#include "usdGen/digest.h"
#include "usdGen/executionBackend.h"
#include "usdGen/executionTaskGraph.h"

#include "usdGenMath/usdGenMath/hash.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/envSetting.h"
#include "pxr/base/trace/trace.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>

#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"

PXR_NAMESPACE_USING_DIRECTIVE

// Hair shadowing the scalp is on by default and costs one extra sweep of the
// density volume plus a small translucent mesh. Set USDGEN_SCALP_SHADOW=0 to
// publish no cap at all: nothing else in the groom changes.
TF_DEFINE_ENV_SETTING(USDGEN_SCALP_SHADOW, true,
                      "Publish the usdGen scalp-shadow cap (hair shadowing the "
                      "groom's emitting surface).");

namespace usdGen {

namespace {

// USDGEN_COMMIT: where one cook's time went, printed as a single line.
class _CommitPhases
{
public:
    _CommitPhases()
        : _enabled(TfDebug::IsEnabled(USDGEN_COMMIT)),
          _start(std::chrono::steady_clock::now()), _mark(_start) {}

    void Phase(char const *name, size_t rebuilt = SIZE_MAX, size_t total = 0)
    {
        if (!_enabled) return;
        auto const now = std::chrono::steady_clock::now();
        double const ms = std::chrono::duration<double, std::milli>(now - _mark).count();
        char buffer[160];
        if (rebuilt == SIZE_MAX)
            std::snprintf(buffer, sizeof(buffer), "%s%s %.2f",
                          _text.empty() ? "" : ", ", name, ms);
        else
            std::snprintf(buffer, sizeof(buffer), "%s%s %.2f (%zu/%zu rebuilt)",
                          _text.empty() ? "" : ", ", name, ms, rebuilt, total);
        _text += buffer;
        _mark = now;
    }

    void Print(double frame, std::string const &compile, char const *outcome,
               SdfPath const &description) const
    {
        if (!_enabled) return;
        double const total = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - _start).count();
        TfDebug::Helper().Msg("usdGen commit    %s frame %g: %s; %s; total %.2f ms [ms: %s]\n",
                              description.GetText(), frame, outcome,
                              compile.empty() ? "no compile" : compile.c_str(), total,
                              _text.c_str());
    }

private:
    bool _enabled;
    std::chrono::steady_clock::time_point _start, _mark;
    std::string _text;
};

// Device revisions are allocated in the immutable plan's topological operator
// order. Derive terminal topology from its actual value producer, never from
// a returned generation or a source-relative arithmetic guess.
bool _TerminalTopologyRevision(UsdGenExecutionPlanMetadata const& metadata,
    uint64_t source, uint64_t final, std::vector<uint64_t> const& intermediates,
    uint64_t* result)
{
    auto const* terminal = metadata.FindTask(metadata.TerminalTask());
    if (!result || !terminal || terminal->kind != UsdGenExecutionTaskKind::Publication) return false;
    uint32_t producer = UINT32_MAX;
    bool found = false;
    for (auto const& use : terminal->resources) {
        if (use.resource != UsdGenExecutionDataKind::CurveTopology) continue;
        if (found || use.access != UsdGenExecutionResourceAccess::Read ||
            use.inputValue >= metadata.Values().size()) return false;
        auto const& value = metadata.Values()[use.inputValue];
        if (value.resource != UsdGenExecutionDataKind::CurveTopology ||
            value.producerTask != use.producerTask) return false;
        found = true; producer = value.producerTask;
    }
    auto const* task = found ? metadata.FindTask(producer) : nullptr;
    if (!task) return false;
    if (task->kind == UsdGenExecutionTaskKind::Source) { *result = source; return true; }
    if (task->kind != UsdGenExecutionTaskKind::Operator) return false;
    size_t ordinal = 0;
    for (auto const& candidate : metadata.Tasks()) {
        if (candidate.kind != UsdGenExecutionTaskKind::Operator) continue;
        if (candidate.id == producer) {
            if (ordinal > intermediates.size()) return false;
            *result = ordinal == intermediates.size() ? final : intermediates[ordinal];
            return true;
        }
        ++ordinal;
    }
    return false;
}

// Cook-digest feeds. Keys are internal equality-only cache identity (the
// digest.h contract names cook digests explicitly), so the bulk feeds use
// the shared 4-lane word-wise mixer instead of byte-at-a-time FNV-1a:
// same coverage, same lanes, ~8x fewer multiplies with 4-way chain
// overlap. Key values change; groom outputs do not.
void _CacheMix(uint64_t *hash, uint64_t value)
{
    UsdGenDigestMixWord(*hash, value);
}

void _CacheMixText(uint64_t *hash, std::string const &text)
{
    _CacheMix(hash, static_cast<uint64_t>(text.size()));
    if (!text.empty())
        *hash = UsdGenDigestBytes(text.data(), text.size(), *hash);
}

// Deferred array term: the serial plan-digest walk records every hashed
// array (lane, bytes) instead of hashing bytes inline, so the ~40MB bench
// surface digests in parallel below. Sizes still mix serially in walk
// order; term digests join their lane in recorded order afterwards, so the
// key stays deterministic, input-sensitive and order-sensitive (digest.h
// contract: values are equality-only internal keys, never persisted).
struct _CacheArrayTerm
{
    uint64_t *lane;
    void const *data;
    size_t bytes;
};

struct _CacheArrayTerms
{
    std::vector<_CacheArrayTerm> terms;
    size_t totalBytes = 0;
};

template<class Array>
void _CacheMixArray(uint64_t *hash, Array const &array,
                    _CacheArrayTerms *terms)
{
    _CacheMix(hash, static_cast<uint64_t>(array.size()));
    if (!array.empty() && terms) {
        size_t const bytes = array.size() * sizeof(array[0]);
        terms->terms.push_back(_CacheArrayTerm{hash, array.cdata(), bytes});
        terms->totalBytes += bytes;
    }
}

// Two-arg serial fallback for call sites that hash small arrays inline
// (generation-gated paths); the three-arg form above defers large arrays
// into terms for the parallel digest.
template<class Array>
void _CacheMixArray(uint64_t *hash, Array const &array)
{
    _CacheMix(hash, static_cast<uint64_t>(array.size()));
    if (!array.empty())
        *hash = UsdGenDigestBytes(array.cdata(),
                                  array.size() * sizeof(array[0]), *hash);
}

struct _CacheDigestJob
{
    void const *data;
    size_t bytes;
    uint64_t seed;
    uint64_t *out;
};

// 4MB segments: the bench key spreads its 4/16/12/8MB arrays over 10 jobs.
size_t constexpr _CacheDigestSegmentBytes = size_t(4) << 20;

void _CacheDigestTerms(_CacheArrayTerms const &terms)
{
    if (terms.terms.empty())
        return;
    // Flatten terms into segment jobs (segmentation is a deterministic
    // function of each term's bytes); the ordered folds below restore term
    // order, and per-segment seeds domain-separate segment positions.
    std::vector<_CacheDigestJob> jobs;
    size_t jobCount = 0;
    for (_CacheArrayTerm const &term : terms.terms)
        jobCount += (term.bytes + _CacheDigestSegmentBytes - 1) /
            _CacheDigestSegmentBytes;
    jobs.reserve(jobCount);
    std::vector<uint64_t> digests(jobCount);
    size_t job = 0;
    for (_CacheArrayTerm const &term : terms.terms) {
        unsigned char const *base =
            static_cast<unsigned char const *>(term.data);
        size_t remaining = term.bytes, seg = 0;
        while (remaining != 0) {
            size_t const len =
                std::min(remaining, _CacheDigestSegmentBytes);
            jobs.push_back(_CacheDigestJob{
                base, len,
                UsdGenDigestOffset ^ (seg * 0x9E3779B97F4A7C15ULL),
                &digests[job]});
            base += len;
            remaining -= len;
            ++seg;
            ++job;
        }
    }
    // Threaded over jobs for big keys (plain TBB like tile publication:
    // Cook owns no scheduler context here); chunks cap at 8. Small keys
    // keep the serial driver, computing the identical function.
    int const keyWorkers = tbb::this_task_arena::max_concurrency();
    size_t const keyChunks =
        (keyWorkers > 1 && terms.totalBytes > 1048576)
            ? std::min({size_t(keyWorkers), size_t(8), jobs.size()})
            : 1;
    if (keyChunks > 1) {
        size_t const n = jobs.size();
        _CacheDigestJob const *jobData = jobs.data();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, keyChunks),
            [&](tbb::blocked_range<size_t> const &range) {
                for (size_t c = range.begin(); c != range.end(); ++c) {
                    size_t const i0 = (c * n) / keyChunks;
                    size_t const i1 = ((c + 1) * n) / keyChunks;
                    for (size_t i = i0; i < i1; ++i)
                        *jobData[i].out = UsdGenDigestBytes(
                            jobData[i].data, jobData[i].bytes,
                            jobData[i].seed);
                }
            });
    } else {
        for (_CacheDigestJob const &j : jobs)
            *j.out = UsdGenDigestBytes(j.data, j.bytes, j.seed);
    }
    // Ordered fold: each term's segment digests chain, then terms join
    // their lane in recorded walk order.
    size_t k = 0;
    for (_CacheArrayTerm const &term : terms.terms) {
        uint64_t acc = UsdGenDigestOffset;
        size_t remaining = term.bytes;
        while (remaining != 0) {
            size_t const len =
                std::min(remaining, _CacheDigestSegmentBytes);
            UsdGenDigestMixWord(acc, digests[k++]);
            remaining -= len;
        }
        _CacheMix(term.lane, acc);
    }
}

void _CacheMixVec3(uint64_t *hash, GfVec3f const &value)
{
    for (int i = 0; i != 3; ++i) {
        uint32_t bits = 0;
        float component = value[i];
        std::memcpy(&bits, &component, sizeof(bits));
        _CacheMix(hash, bits);
    }
}

void _CacheMixMatrix(uint64_t *hash, GfMatrix4d const &value)
{
    for (int row = 0; row != 4; ++row) for (int col = 0; col != 4; ++col) {
        uint64_t bits = 0;
        double component = value[row][col];
        std::memcpy(&bits, &component, sizeof(bits));
        _CacheMix(hash, bits);
    }
}

void _CacheMixShape(uint64_t *hash, expr::ValueShape const &shape)
{
    _CacheMix(hash, static_cast<uint64_t>(shape.scalar));
    _CacheMix(hash, shape.elementCount);
    _CacheMix(hash, shape.components);
    _CacheMix(hash, shape.rows);
    _CacheMix(hash, shape.columns);
    _CacheMix(hash, static_cast<uint64_t>(shape.isArray));
}

void _CacheMixPaths(uint64_t *hash, std::vector<SdfPath> const &paths)
{
    _CacheMix(hash, static_cast<uint64_t>(paths.size()));
    for (SdfPath const &path : paths) _CacheMixText(hash, path.GetString());
}

UsdGenEpoch _ExecutionPlanDigest(UsdGenGraph const &graph,
                                UsdGenGraphDesc const &desc)
{
    uint64_t h0 = 1469598103934665603ULL;
    uint64_t h1 = 1099511628211ULL;
    _CacheArrayTerms terms;
    _CacheMix(&h0, static_cast<uint64_t>(graph.NodeCount()));
    _CacheMix(&h1, static_cast<uint64_t>(graph.TerminalNodeId()));
    for (UsdGenNodeId id = 0; id != static_cast<UsdGenNodeId>(graph.NodeCount()); ++id) {
        UsdGenCompiledNode const &node = graph.Node(id);
        _CacheMixText(&h0, node.type.GetString());
        _CacheMixText(&h1, node.desc ? node.desc->path.GetString() : std::string());
        _CacheMix(&h1, node.structuralDigest[0]);
        _CacheMix(&h0, node.structuralDigest[1]);
        // structuralDigest intentionally excludes value/capture-class
        // parameters for incremental recompilation. Cache identity cannot:
        // include the compiled value digest and all resolved descriptor terms
        // so a terminal value edit is a miss even when its plan is reusable.
        _CacheMix(&h1, node.paramValueDigest);
        if (node.desc) {
            UsdGenNodeDesc const &nd = *node.desc;
            _CacheMix(&h0, static_cast<uint64_t>(nd.enabled));
            _CacheMix(&h1, static_cast<uint64_t>(nd.seed));
            _CacheMixText(&h1, nd.mode.GetString());
            _CacheMixPaths(&h1, nd.inputs);
            _CacheMixPaths(&h0, nd.references);
            _CacheMixPaths(&h1, nd.curves);
            _CacheMixPaths(&h0, nd.surfaces);
            _CacheMixPaths(&h1, nd.maps);
            _CacheMix(&h0, static_cast<uint64_t>(nd.mapBindings.size()));
            for (UsdGenMapBindingDesc const &binding : nd.mapBindings) {
                _CacheMixText(&h0, binding.map.GetString());
                _CacheMixText(&h0, binding.relationship.GetString());
            }
            _CacheMix(&h0, static_cast<uint64_t>(nd.params.size()));
            for (UsdGenParamValue const &param : nd.params) {
                _CacheMixText(&h0, param.name.GetString());
                _CacheMix(&h1, param.value.GetHash());
                _CacheMix(&h0, static_cast<uint64_t>(param.animated));
            }
            _CacheMix(&h1, static_cast<uint64_t>(nd.ramps.size()));
            for (UsdGenRampDesc const &ramp : nd.ramps) {
                _CacheMixArray(&h0, ramp.knots, &terms);
                _CacheMixArray(&h1, ramp.positions, &terms);
                _CacheMixArray(&h0, ramp.colors, &terms);
                _CacheMixText(&h1, ramp.interpolation.GetString());
            }
        }
        for (UsdGenNodeId input : node.inputs) _CacheMix(&h1, input);
    }
    // Descriptor-wide terms can affect evaluation or presentation while a
    // compiled node remains incrementally reusable.
    _CacheMixText(&h0, desc.curveBasis.GetString());
    uint32_t widthBits = 0;
    std::memcpy(&widthBits, &desc.defaultWidth, sizeof(widthBits));
    _CacheMix(&h1, widthBits);
    _CacheMixText(&h0, desc.look.bakeMode.GetString());
    _CacheMixText(&h1, desc.look.bakeTarget.GetString());
    _CacheMixText(&h0, desc.look.bakePrimvar.GetString());
    _CacheMixArray(&h1, desc.look.rampColors, &terms);
    _CacheMixArray(&h0, desc.look.rampPositions, &terms);
    _CacheMixVec3(&h1, desc.look.rootColor);
    _CacheMixVec3(&h0, desc.look.tipColor);
    _CacheMixText(&h1, desc.look.rampInterpolation.GetString());
    uint32_t rampExponentBits = 0, hueJitterBits = 0, valueJitterBits = 0;
    std::memcpy(&rampExponentBits, &desc.look.rampExponent, sizeof(rampExponentBits));
    std::memcpy(&hueJitterBits, &desc.look.hueJitter, sizeof(hueJitterBits));
    std::memcpy(&valueJitterBits, &desc.look.valueJitter, sizeof(valueJitterBits));
    _CacheMix(&h1, rampExponentBits);
    _CacheMix(&h0, hueJitterBits);
    _CacheMix(&h1, valueJitterBits);
    _CacheMix(&h0, static_cast<uint64_t>(desc.look.jitterSeed));
    _CacheMixText(&h1, desc.preview.source.GetString());
    _CacheMixText(&h0, desc.preview.colorMap.GetString());
    _CacheMixText(&h1, desc.preview.evaluation.GetString());
    _CacheMixText(&h0, desc.preview.shading.GetString());
    uint32_t previewLoBits = 0, previewHiBits = 0;
    std::memcpy(&previewLoBits, &desc.preview.range[0], sizeof(previewLoBits));
    std::memcpy(&previewHiBits, &desc.preview.range[1], sizeof(previewHiBits));
    _CacheMix(&h1, previewLoBits);
    _CacheMix(&h0, previewHiBits);
    _CacheMixMatrix(&h1, desc.xformMatrix);
    _CacheMixText(&h0, desc.purpose.GetString());
    _CacheMixText(&h1, desc.visibility.GetString());
    _CacheMixText(&h0, desc.materialPath.GetString());
    uint64_t timeBits = 0, cpsBits = 0;
    std::memcpy(&timeBits, &desc.time, sizeof(timeBits));
    std::memcpy(&cpsBits, &desc.timeCodesPerSecond, sizeof(cpsBits));
    _CacheMix(&h0, timeBits);
    _CacheMix(&h1, cpsBits);
    _CacheMix(&h0, static_cast<uint64_t>(desc.expressions.size()));
    for (UsdGenExpressionDesc const &expression : desc.expressions) {
        _CacheMixText(&h0, expression.path.GetString());
        _CacheMixText(&h1, expression.source);
        _CacheMix(&h0, static_cast<uint64_t>(expression.outputs.size()));
        for (UsdGenExpressionOutputDesc const &output : expression.outputs) {
            _CacheMixText(&h1, output.name.GetString());
            _CacheMixText(&h0, output.nativeType.GetString());
            _CacheMixShape(&h1, output.shape);
        }
        _CacheMix(&h0, static_cast<uint64_t>(expression.inputs.size()));
        for (UsdGenExpressionInputDesc const &input : expression.inputs) {
            _CacheMixText(&h1, input.name.GetString());
            for (SdfPath const &path : input.targets) _CacheMixText(&h0, path.GetString());
            for (SdfPath const &path : input.geometries) _CacheMixText(&h1, path.GetString());
            for (SdfPath const &path : input.maps) _CacheMixText(&h0, path.GetString());
        }
    }
    // Geometry an expression samples is an execution input exactly like a
    // curve set: its content moves the key.
    _CacheMix(&h1, static_cast<uint64_t>(desc.geometries.size()));
    for (UsdGenGeometryDesc const &geometry : desc.geometries) {
        _CacheMixText(&h0, geometry.path.GetString());
        _CacheMix(&h1, static_cast<uint64_t>(geometry.kind));
        _CacheMixArray(&h0, geometry.counts, &terms);
        _CacheMixArray(&h1, geometry.indices, &terms);
        _CacheMixArray(&h0, geometry.points, &terms);
        _CacheMixArray(&h1, geometry.rest, &terms);
        _CacheMixArray(&h0, geometry.normals, &terms);
        _CacheMixArray(&h1, geometry.ids, &terms);
        _CacheMixArray(&h0, geometry.subsetFaces, &terms);
        _CacheMix(&h1, geometry.isSubset ? 1u : 0u);
        _CacheMixMatrix(&h0, geometry.worldMatrix);
        _CacheMix(&h1, geometry.generation);
    }
    for (UsdGenNodeDesc const &node : desc.nodes) {
        _CacheMix(&h0, static_cast<uint64_t>(node.expressionBindings.size()));
        for (UsdGenExpressionBinding const &binding : node.expressionBindings) {
            _CacheMixText(&h1, binding.expression.GetString());
            _CacheMixText(&h0, binding.output.GetString());
            _CacheMixText(&h1, binding.nativeType.GetString());
            _CacheMixText(&h0, binding.destination.GetString());
            _CacheMixShape(&h1, binding.destinationShape);
            _CacheMix(&h0, static_cast<uint64_t>(binding.domain));
            _CacheMix(&h1, binding.literal.GetHash());
        }
    }
    for (UsdGenMapDesc const &map : desc.maps) {
        _CacheMixText(&h0, map.path.GetString());
        _CacheMixText(&h1, map.type.GetString());
        _CacheMixText(&h0, map.resolvedAssetPath);
        _CacheMix(&h1, map.textureGeneration);
        // A paint map's snapshot is capture content, not identity: the
        // path, type and generation never move under a repaint, so the
        // values themselves join the key or a length stroke would hit the
        // pre-stroke cache entry and the groom would freeze (the density
        // multiplier's row below is the same contract).
        if (map.type == TfToken("UsdGenPaintMap"))
            _CacheMixArray(&h0, map.paintValues, &terms);
        _CacheMix(&h0, static_cast<uint64_t>(map.params.size()));
        for (UsdGenParamValue const &param : map.params) {
            _CacheMixText(&h1, param.name.GetString());
            _CacheMix(&h0, param.value.GetHash());
            _CacheMix(&h1, static_cast<uint64_t>(param.animated));
        }
    }
    // A finalized input carries its content hash as its generation
    // (UsdGenFinalizeInputGenerations, never zero), so the key reads the
    // bulk arrays through it instead of re-hashing tens of megabytes the
    // capture just hashed: same coverage (every content byte feeds the
    // generation), one word per input. The key's input-versions tuple
    // already trusts generations for equality, so this adds no new
    // trust; key values change, groom outputs do not. A zero generation
    // means a hand-built, unfinalized desc, which keeps raw feeds below
    // (same coverage; feed order is the new function's own).
    for (UsdGenCurveSetDesc const &curve : desc.curveSets) {
        _CacheMixText(&h0, curve.path.GetString());
        _CacheMix(&h1, static_cast<uint64_t>(curve.role));
        _CacheMixText(&h0, curve.curveRole.GetString());
        _CacheMixText(&h1, curve.type.GetString());
        _CacheMixText(&h0, curve.basis.GetString());
        _CacheMixText(&h1, curve.wrap.GetString());
        _CacheMixText(&h0, curve.widthsInterpolation.GetString());
        _CacheMix(&h1, static_cast<uint64_t>(curve.restFromCurrentPoints));
        _CacheMixMatrix(&h1, curve.worldMatrix);
        _CacheMixText(&h0, curve.frozenEpoch);
        _CacheMix(&h1, static_cast<uint64_t>(curve.authoredPlanes.size()));
        for (UsdGenAuthoredPlaneDesc const& plane : curve.authoredPlanes) {
            _CacheMixText(&h0, plane.name.GetString());
            _CacheMix(&h1, static_cast<uint64_t>(plane.type));
            _CacheMix(&h0, static_cast<uint64_t>(plane.domain));
            _CacheMix(&h1, plane.arity);
            if (curve.curveGeneration) continue;
            _CacheMixArray(&h0, plane.floatValues);
            _CacheMixArray(&h1, plane.intValues);
        }
        if (curve.curveGeneration) {
            _CacheMix(&h0, curve.curveGeneration);
            continue;
        }
        _CacheMixArray(&h0, curve.curveVertexCounts);
        _CacheMixArray(&h1, curve.points);
        _CacheMixArray(&h0, curve.rest);
        _CacheMixArray(&h0, curve.widths);
        _CacheMixArray(&h1, curve.skinPrim);
        _CacheMixArray(&h0, curve.curveId);
        _CacheMixArray(&h1, curve.skinPrimUv);
        _CacheMixArray(&h0, curve.rootFrame);
        _CacheMix(&h1, curve.curveGeneration);
    }
    for (UsdGenSurfaceDesc const &surface : desc.surfaces) {
        // Tags-only subdivision cover: faceVertexCounts/faceVertexIndices
        // reach this key through the generation (raw below when
        // unfinalized). Coverage is unchanged.
        _CacheMix(&h0, UsdGenSubdivisionTagsDigest(surface));
        _CacheMixText(&h0, surface.path.GetString());
        _CacheMix(&h1, surface.id);
        _CacheMix(&h0, static_cast<uint64_t>(surface.restNormalDomain));
        _CacheMix(&h1, static_cast<uint64_t>(surface.restFromCurrentPoints));
        _CacheMix(&h1, static_cast<uint64_t>(surface.samples.size()));
        for (UsdGenSurfaceSample const& sample : surface.samples) {
            uint64_t sampleTimeBits = 0;
            std::memcpy(&sampleTimeBits, &sample.time, sizeof(sampleTimeBits));
            _CacheMix(&h0, sampleTimeBits);
            if (surface.surfaceGeneration) continue;
            _CacheMixArray(&h1, sample.points);
        }
        _CacheMix(&h1, surface.isSubset ? 1u : 0u);
        // Paint density the brush bakes: a repaint must miss the cache,
        // or the stale pre-stroke roots publish and the groom freezes.
        // The generation does not cover it, so it still mixes raw.
        _CacheMixArray(&h1, surface.densityMultiplier);
        _CacheMixMatrix(&h0, surface.worldMatrix);
        if (surface.surfaceGeneration) {
            _CacheMix(&h0, surface.surfaceGeneration);
            continue;
        }
        _CacheMixArray(&h0, surface.faceVertexCounts);
        _CacheMixArray(&h1, surface.faceVertexIndices);
        _CacheMixArray(&h0, surface.restPoints);
        _CacheMixArray(&h1, surface.restNormals);
        _CacheMixArray(&h0, surface.points);
        _CacheMixArray(&h0, surface.velocities);
        _CacheMixArray(&h1, surface.uv);
        _CacheMixArray(&h0, surface.subsetFaces);
        _CacheMix(&h1, surface.surfaceGeneration);
    }
    _CacheDigestTerms(terms);
    return {h0, h1};
}

uint64_t _ExecutionLayoutDigest(UsdGenGraph const &graph,
                                UsdGenGraphDesc const &desc)
{
    uint64_t hash = 1469598103934665603ULL;
    _CacheMix(&hash, static_cast<uint64_t>(graph.ChunkSize()));
    _CacheMix(&hash, static_cast<uint64_t>(graph.TileTarget()));
    _CacheMix(&hash, static_cast<uint64_t>(graph.ChunksPerTile()));
    _CacheMix(&hash, static_cast<uint64_t>(desc.tileTarget));
    for (auto const &surface : desc.surfaces) {
        _CacheMixText(&hash, surface.path.GetString());
        _CacheMix(&hash, surface.faceVertexCounts.size());
        _CacheMix(&hash, surface.restPoints.size());
    }
    for (auto const &curve : desc.curveSets) {
        _CacheMixText(&hash, curve.path.GetString());
        _CacheMix(&hash, curve.curveVertexCounts.size());
        _CacheMix(&hash, curve.points.size());
    }
    return hash;
}

UsdGenExecutionContext _ExecutionCacheContext(
    UsdGenGraphDesc const &desc, UsdGenContext evaluationContext,
    int deviceIndex = -1, uint64_t deviceGeneration = 0)
{
    uint32_t const capabilityVersion =
        desc.executionBackend == UsdGenExecutionBackend::Cuda
            ? GetCudaExecutionCapabilityMatrix().Version() :
              GetUsdGenExecutionBackendContract(desc.executionBackend).capabilityVersion;
    return MakeUsdGenExecutionContext(desc.executionBackend, capabilityVersion,
                                      deviceIndex, deviceGeneration,
                                      evaluationContext);
}

UsdGenExecutionCacheKey _MakeExecutionCacheKey(
    UsdGenGraph const &graph, UsdGenGraphDesc const &desc,
    UsdGenExecutionContext context, double frame)
{
    return UsdGenExecutionCacheKey::Make(
        desc.description, _ExecutionPlanDigest(graph, desc), graph.InputVersions(),
        context, _ExecutionLayoutDigest(graph, desc), frame);
}

bool _AddBytes(size_t *total, size_t count, size_t element)
{
    if (count != 0 && element > std::numeric_limits<size_t>::max() / count)
        return false;
    size_t const bytes = count * element;
    if (bytes > std::numeric_limits<size_t>::max() - *total) return false;
    *total += bytes;
    return true;
}

bool _GenerationBytes(UsdGenGeneration const &generation, size_t *result)
{
    if (!result) return false;
    size_t total = sizeof(UsdGenGeneration);
    auto addTile = [&total](UsdGenTilePublication const &tile) {
        auto addPlane = [&total](UsdGenPlane const &plane) {
            return _AddBytes(&total, 1, sizeof(UsdGenPlane)) &&
                _AddBytes(&total, plane.f.size(), sizeof(float)) &&
                _AddBytes(&total, plane.i.size(), sizeof(int));
        };
        if (!_AddBytes(&total, 1, sizeof(UsdGenTilePublication)) ||
            !_AddBytes(&total, tile.primPath.GetString().size(), sizeof(char)) ||
            !_AddBytes(&total, tile.basis.size(), sizeof(char)) ||
            !_AddBytes(&total, tile.curveVertexCounts.size(), sizeof(int)) ||
            !_AddBytes(&total, tile.points.size(), sizeof(GfVec3f)) ||
            !_AddBytes(&total, tile.widths.size(), sizeof(float)) ||
            !_AddBytes(&total, tile.hairT.size(), sizeof(float)) ||
            !_AddBytes(&total, tile.hairId.size(), sizeof(float)) ||
            !_AddBytes(&total, tile.st.size(), sizeof(GfVec2f)) ||
            !_AddBytes(&total, tile.displayColor.size(), sizeof(GfVec3f)) ||
            !_AddBytes(&total, tile.bakeColor.size(), sizeof(GfVec3f)) ||
            !_AddBytes(&total, tile.velocities.size(), sizeof(GfVec3f)))
            return false;
        for (UsdGenPlane const &plane : tile.extraUniform)
            if (!addPlane(plane)) return false;
        return true;
    };
    for (auto const &tile : generation.tiles)
        if (!addTile(tile)) return false;
    for (auto const &tile : generation.guides)
        if (!addTile(tile)) return false;
    for (UsdGenInstancerPublication const &instancer : generation.instancers) {
        if (!_AddBytes(&total, 1, sizeof(UsdGenInstancerPublication)) ||
            !_AddBytes(&total, instancer.primPath.GetString().size(), sizeof(char)) ||
            !_AddBytes(&total, instancer.prototypes.size(), sizeof(SdfPath)) ||
            !_AddBytes(&total, instancer.instanceIndices.size(), sizeof(VtIntArray)) ||
            !_AddBytes(&total, instancer.mask.size(), sizeof(bool)))
            return false;
        for (SdfPath const &prototype : instancer.prototypes)
            if (!_AddBytes(&total, prototype.GetString().size(), sizeof(char)))
                return false;
        for (VtIntArray const &indices : instancer.instanceIndices)
            if (!_AddBytes(&total, indices.size(), sizeof(int))) return false;
    }
    if (!_AddBytes(&total, 1, sizeof(UsdGenPrimSetSignature)) ||
        !_AddBytes(&total, generation.signature.primPaths.size(), sizeof(std::string)) ||
        !_AddBytes(&total, generation.signature.primTypes.size(), sizeof(std::string)) ||
        !_AddBytes(&total, generation.signature.primvarNames.size(),
                   sizeof(std::vector<std::string>))) return false;
    for (std::string const &path : generation.signature.primPaths)
        if (!_AddBytes(&total, path.size(), sizeof(char))) return false;
    for (std::string const &type : generation.signature.primTypes)
        if (!_AddBytes(&total, type.size(), sizeof(char))) return false;
    for (auto const &names : generation.signature.primvarNames) {
        if (!_AddBytes(&total, names.size(), sizeof(std::string))) return false;
        for (std::string const &name : names)
            if (!_AddBytes(&total, name.size(), sizeof(char))) return false;
    }
    if (generation.device) {
        // Cache residency must account for every immutable COW predecessor
        // reachable through the owner. The exclusive value is correct for
        // device-pool permits, but would undercharge a derived cache entry
        // after its base cache record is evicted.
        size_t const deviceBytes = generation.device->InclusiveRetainedBytes();
        if (deviceBytes == 0 || deviceBytes == std::numeric_limits<size_t>::max() ||
            deviceBytes > std::numeric_limits<size_t>::max() - total)
            return false;
        total += deviceBytes;
    }
    *result = total;
    return true;
}

SdfPath _RenderNamespace(SdfPath const &description)
{
    return description.AppendChild(TfToken("__usdGenRender"));
}

// <description>/__usdGenRender/tile_NNNN — must stay byte-identical to the
// imaging-side UsdGenTilePublisher::TilePath formula (06 §4.2).
SdfPath _TilePath(SdfPath const &description, UsdGenTileId tile)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "tile_%04d", static_cast<int>(tile));
    return _RenderNamespace(description).AppendChild(TfToken(buf));
}

struct _PresentationScalars
{
    GfMatrix4d xformMatrix{1.0};
    TfToken purpose;
    TfToken visibility;
    SdfPath materialPath;
    TfToken materialPurpose{TfToken("allPurpose")};
    int refineLevel = 2;
    SdfPath primOrigin;
    SdfPath dependencySurface;
};

_PresentationScalars _BuildPresentationScalars(
    UsdGenGraphDesc const &desc, SdfPath const &dependencySurface)
{
    _PresentationScalars result;
    result.xformMatrix = desc.xformMatrix;
    result.purpose = desc.purpose;
    result.visibility = desc.visibility;
    result.materialPath = desc.materialPath;
    result.primOrigin = desc.description;
    result.dependencySurface = dependencySurface;
    return result;
}

// CUDA admission permits at most one CurveSource root-binding surface and
// ExecuteCudaGraph rejects a named binding that is absent from desc.surfaces.
// Return no dependency for the valid empty/no-binding case; never invent one.
SdfPath _CudaSourceDependencySurface(UsdGenGraphDesc const &desc)
{
    if (desc.nodes.empty() || desc.nodes.front().surfaces.size() != 1)
        return SdfPath();
    SdfPath const &candidate = desc.nodes.front().surfaces.front();
    auto found = std::find_if(desc.surfaces.begin(), desc.surfaces.end(),
        [&](UsdGenSurfaceDesc const &surface) { return surface.path == candidate; });
    return found == desc.surfaces.end() ? SdfPath() : found->path;
}

UsdGenDevicePresentationMetadata _BuildDevicePresentation(
    UsdGenGraphDesc const &desc)
{
    _PresentationScalars const scalars = _BuildPresentationScalars(
        desc, _CudaSourceDependencySurface(desc));
    UsdGenDevicePresentationMetadata result;
    result.description = desc.description;
    result.renderNamespace = _RenderNamespace(desc.description);
    result.xformMatrix = scalars.xformMatrix;
    result.purpose = scalars.purpose;
    result.visibility = scalars.visibility;
    result.materialPath = scalars.materialPath;
    result.materialPurpose = scalars.materialPurpose;
    result.refineLevel = scalars.refineLevel;
    result.primOrigin = scalars.primOrigin;
    result.dependencySurface = scalars.dependencySurface;
    return result;
}

// Sample a named curve buffer plane into the publication's uniform array.
// "uniform" planes index by curve, "vertex" planes by CV.
void _GatherPlane(const UsdGenPlane &plane, uint32_t curveIdx, uint32_t cvIdx,
                  bool wantUniform, VtFloatArray *outF)
{
    // Every TfToken(const char*) construction takes the registry lock; these
    // spellings are built once for the process, not once per gathered CV.
    static TfToken const intType("int");
    static TfToken const vertexInterp("vertex");
    if (plane.type == intType || plane.arity == 0) return;
    const size_t idx =
        (wantUniform && plane.interpolation != vertexInterp)
            ? curveIdx : cvIdx;
    if ((idx + 1) * plane.arity > plane.f.size()) return;
    for (uint8_t k = 0; k < plane.arity; ++k)
        outF->push_back(plane.f[idx * plane.arity + k]);
}

bool _GatherPlaneElement(UsdGenPlane const& plane, size_t index,
                         UsdGenPlane* out)
{
    static TfToken const intType("int");
    static TfToken const floatType("float");
    if (!out || plane.arity == 0) return false;
    size_t const begin = index * plane.arity;
    if (plane.type == intType) {
        if (begin + plane.arity > plane.i.size()) return false;
        for (uint8_t component = 0; component != plane.arity; ++component)
            out->i.push_back(plane.i[begin + component]);
        return true;
    }
    if (plane.type != floatType || begin + plane.arity > plane.f.size())
        return false;
    for (uint8_t component = 0; component != plane.arity; ++component)
        out->f.push_back(plane.f[begin + component]);
    return true;
}

// Color planes may be float3 (arity 3) or float (arity 1, grey).
void _GatherColor(const UsdGenPlane &plane, uint32_t curveIdx, uint32_t cvIdx,
                  VtVec3fArray *out)
{
    VtFloatArray tmp;
    _GatherPlane(plane, curveIdx, cvIdx, /*wantUniform=*/true, &tmp);
    if (plane.arity >= 3) {
        for (size_t q = 0; q + 3 <= tmp.size(); q += 3)
            out->push_back(GfVec3f(tmp[q], tmp[q + 1], tmp[q + 2]));
    } else if (plane.arity == 1) {
        for (float v : tmp) out->push_back(GfVec3f(v, v, v));
    }
}

const UsdGenPlane *_FindPlane(const std::vector<UsdGenPlane> &planes,
                              TfToken name)
{
    for (const UsdGenPlane &p : planes)
        if (p.name == name) return &p;
    return nullptr;
}

// Bulk tile path (bit-identical): when the tile's chunks are all uniform,
// the terminal planes cover every emitted element, no extra plane needs a
// per-element gather, and the color mode is a per-tile constant (Root) or
// empty (NoColor), the serial push_back loop becomes per-chunk bulk copies
// True when no node that ran this cook could have written a non-point
// plane: every node-stats entry names a points-only UsdGenDeform (whose
// capture and sweep write only point planes and scratch). Nodes absent
// from the stats neither captured nor swept -- fully skipped, or disabled
// pass-through aliases over unchanged upstream -- so with an unmoved
// partition every tile gather input except the point planes is
// byte-identical to the previous cook. Any other op, a null op, or an
// unknown node id fails closed into a full re-gather.
bool _OnlyPointsDeformsRan(UsdGenGraph const& graph, UsdGenRunResult const& result)
{
    static TfToken const deformType("UsdGenDeform");
    for (UsdGenNodeRunStats const& st : result.nodeStats) {
        if (st.id >= UsdGenNodeId(graph.NodeCount())) return false;
        UsdGenOp const* op = graph.Node(st.id).op.get();
        if (!op || op->Type() != deformType) return false;
        if (op->PlanesTouched() != UsdGenOp::kPlanePoints) return false;
    }
    return true;
}
// (points interleave, widths/hairT/st memcpy, counts fill, displayColor
// assign) plus a branch-free per-curve hairId loop, with the extent
// reduction fused into the interleave. Coverage is verified up front (any
// gap returns false with `pub` untouched and the serial loop runs); sizes
// are proven against the tile view, so every no-init element is
// overwritten and no garbage is observed. wantColor selects Root (NoColor
// passes false); every other color mode stays on the serial loop.
bool _BuildTilePublicationBulk(
    UsdGenTileView const &tv, UsdGenCurveBuffer const &term,
    UsdGenCompiledNode const &tn, bool wantColor, GfVec3f const &rootColor,
    UsdGenTilePublication *pub, uint32_t *depSurface, bool *hasDep)
{
    // Pre-scan: uniform chunks, plane coverage, view-size agreement. The
    // serial loop breaks the CV loop on px overrun and falls back per
    // element for short width/hairT/curveId/rootUV; any of that needs the
    // serial loop, so coverage must be total here.
    uint64_t scanCurves = 0, scanCvs = 0;
    bool const needWidth = !term.width.empty();
    bool const needHairT = !term.hairT.empty();
    bool const needSt = !term.rootUV.empty();
    for (uint32_t i = 0; i < tv.chunkCount; ++i) {
        UsdGenChunkDesc const &cd = tn.chunks[tv.firstChunk + i];
        if (cd.cvCount == 0)
            return false;
        uint64_t const n = uint64_t(cd.liveCount) * cd.cvCount;
        uint64_t const cvEnd = uint64_t(cd.firstCv) + n;
        if (cvEnd > term.px.size() || cvEnd > term.py.size() ||
            cvEnd > term.pz.size())
            return false;
        if (needWidth && cvEnd > term.width.size())
            return false;
        if (needHairT && cvEnd > term.hairT.size())
            return false;
        uint64_t const gEnd = uint64_t(cd.firstCurve) + cd.liveCount;
        if (gEnd > term.curveId.size())
            return false;
        if (needSt && gEnd > term.rootUV.size())
            return false;
        scanCurves += cd.liveCount;
        scanCvs += n;
    }
    if (scanCurves != tv.totalLiveCurves || scanCvs != tv.totalLiveCvs)
        return false;
    size_t const nCurves = tv.totalLiveCurves, nCvs = tv.totalLiveCvs;
    auto noInit = [](auto *b, auto *e) {
        (void)b;
        (void)e;
    };
    pub->curveVertexCounts.resize(nCurves, noInit);
    pub->points.resize(nCvs, noInit);
    pub->widths.resize(nCvs, noInit);
    pub->hairT.resize(nCvs, noInit);
    pub->hairId.resize(nCurves, noInit);
    if (needSt)
        pub->st.resize(nCurves, noInit);
    if (wantColor)
        pub->displayColor.assign(nCurves, rootColor);
    int *counts = pub->curveVertexCounts.data();
    GfVec3f *points = nCvs ? pub->points.data() : nullptr;
    float *widths = nCvs ? pub->widths.data() : nullptr;
    float *hairT = nCvs ? pub->hairT.data() : nullptr;
    float *hairIds = nCurves ? pub->hairId.data() : nullptr;
    GfVec2f *st = (needSt && nCurves) ? pub->st.data() : nullptr;
    float const *px = term.px.cdata(), *py = term.py.cdata(),
                *pz = term.pz.cdata();
    // Fused extent: same init (SetEmpty) and per-point compares as the
    // serial GfRange3f pass over the identical emission order, including
    // the empty-range tail below (NaN points stay ignored, as there).
    float const inf = std::numeric_limits<float>::max();
    float minX = inf, minY = inf, minZ = inf;
    float maxX = -inf, maxY = -inf, maxZ = -inf;
    size_t curOff = 0, cvOff = 0;
    for (uint32_t i = 0; i < tv.chunkCount; ++i) {
        UsdGenChunkDesc const &cd = tn.chunks[tv.firstChunk + i];
        uint32_t const L = cd.liveCount;
        if (L == 0)
            continue;
        size_t const n = size_t(L) * cd.cvCount;
        uint32_t const G = cd.firstCurve, P = cd.firstCv;
        std::fill(counts + curOff, counts + curOff + L, int(cd.cvCount));
        for (uint32_t c = 0; c < L; ++c)
            hairIds[curOff + c] = UsdGenHairId(term.curveId[G + c]);
        if (needSt)
            std::memcpy(st + curOff, term.rootUV.cdata() + G,
                        L * sizeof(GfVec2f));
        if (needWidth)
            std::memcpy(widths + cvOff, term.width.cdata() + P,
                        n * sizeof(float));
        else
            std::fill(widths + cvOff, widths + cvOff + n, 0.01f);
        if (needHairT)
            std::memcpy(hairT + cvOff, term.hairT.cdata() + P,
                        n * sizeof(float));
        else
            std::fill(hairT + cvOff, hairT + cvOff + n, 0.0f);
        for (size_t k = 0; k < n; ++k) {
            float const x = px[P + k], y = py[P + k], z = pz[P + k];
            points[cvOff + k] = GfVec3f(x, y, z);
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            if (z < minZ) minZ = z;
            if (z > maxZ) maxZ = z;
        }
        curOff += L;
        cvOff += n;
    }
    GfRange3f e(GfVec3f(minX, minY, minZ), GfVec3f(maxX, maxY, maxZ));
    if (e.IsEmpty()) e = GfRange3f(GfVec3f(0.f), GfVec3f(0.f));
    pub->extentMin = GfVec3d(e.GetMin()[0], e.GetMin()[1], e.GetMin()[2]);
    pub->extentMax = GfVec3d(e.GetMax()[0], e.GetMax()[1], e.GetMax()[2]);
    if (tv.chunkCount != 0) {
        *depSurface = tn.chunks[tv.firstChunk].surface;
        *hasDep = true;
    }
    return true;
}

}  // namespace

UsdGenSessionCooker::UsdGenSessionCooker(
    int threadLimit, size_t executionCacheBytes)
    : UsdGenSessionCooker(threadLimit, executionCacheBytes, {}) {}

UsdGenSessionCooker::UsdGenSessionCooker(int threadLimit,
                                         size_t executionCacheBytes,
    std::shared_ptr<UsdGenExecutionCacheDomain> executionCacheDomain)
    : _scheduler(threadLimit), _executionCacheBytes(executionCacheBytes),
      _cacheDomainExplicit(bool(executionCacheDomain)),
      _executionCacheDomain(std::move(executionCacheDomain))
{
    if (!_executionCacheDomain)
        _executionCacheDomain = std::make_shared<UsdGenExecutionCacheDomain>(
            UsdGenExecutionCacheDomainKey{
                UsdGenDeviceBackend::CpuReference, -1, 0},
            executionCacheBytes);
}

UsdGenSessionCooker::~UsdGenSessionCooker() = default;

bool UsdGenSessionCooker::_SelectExecutionCacheDomain(
    UsdGenDeviceBackend backend, int32_t deviceIndex, uint64_t contextIdentity)
{
    UsdGenExecutionCacheDomainKey key{backend, deviceIndex, contextIdentity};
    auto selectedDomain = std::atomic_load(&_executionCacheDomain);
    if (selectedDomain) {
        auto const& selected = selectedDomain->Key();
        // contextIdentity is the adapter-owned stable identity of this
        // domain. The cooker selects only backend/device and must preserve a
        // nonzero identity supplied by an explicit Metal/Vulkan/CUDA owner.
        if (selected.backend == backend && selected.deviceIndex == deviceIndex &&
            (!contextIdentity || selected.contextIdentity == contextIdentity))
            return true;
    }
    if (_cacheDomainExplicit) {
        _lastDiagnostics.Error("execution cache domain identity does not match backend/device");
        return false;
    }
    auto replacement = std::make_shared<UsdGenExecutionCacheDomain>(
        key, _executionCacheBytes);
    _observedCacheDomainEpoch = replacement->Epoch();
    std::atomic_store(&_executionCacheDomain, std::move(replacement));
    return true;
}

bool UsdGenSessionCooker::_ObserveExecutionCacheDomainEpoch()
{
    if (!_executionCacheDomain) return false;
    uint64_t const epoch = _executionCacheDomain->Epoch();
    if (epoch == _observedCacheDomainEpoch) return true;
    // Another Session sharing this domain invalidated the backend context.
    // The serialized cooker lane is the only owner of this workspace, so it
    // can quarantine it before the next job borrows any native allocation.
    _cacheCandidate.reset();
    if (_cudaWorkspace) {
        _cudaWorkspace->MarkContextLost();
        _cudaWorkspace.reset();
        _cudaWorkspaceDescription = SdfPath();
    }
    _observedCacheDomainEpoch = epoch;
    return true;
}

void UsdGenSessionCooker::_BeginCoalesced(
    std::shared_ptr<UsdGenExecutionCacheDomain> const& domain,
    UsdGenExecutionCacheKey const& key, CoalescedHooks const& hooks)
{
    if (!domain || !hooks.callback) return;
    auto fence = domain->CapturePublicationFence(key);
    if (!fence) return;
    _coalescedDomain = domain;
    _coalescedKey = key;
    _hasCoalescedKey = true;
    auto resident = std::make_shared<std::atomic<bool>>(false);
    auto registration = domain->BeginCoalesced(key,
        [callback=hooks.callback, domain, fence, resident](
            UsdGenExecutionCacheDomain::CoalescedStatus status,
            std::optional<UsdGenExecutionCacheDomain::Lease> lease) mutable {
            if (status == UsdGenExecutionCacheDomain::CoalescedStatus::Resident)
                resident->store(true, std::memory_order_release);
            try {
                callback(status, std::move(lease),
                    ExecutionPublicationFence{domain, *fence});
            } catch (...) {}
        }, false);
    if (!registration) {
        if (resident->load(std::memory_order_acquire))
            _coalescedRole = CoalescedRole::Resident;
        return;
    }
    _coalescedRegistration = *registration;
    _coalescedRole = registration->leader
        ? CoalescedRole::Leader : CoalescedRole::Follower;
    if (hooks.registered) hooks.registered(domain, *registration);
}

bool UsdGenSessionCooker::_Prepare(std::shared_ptr<const UsdGenGraphDesc> desc,
    UsdGenContext context, bool devicePublicationEnabled,
    UsdGenPendingDirty& pending, UsdGenGenerationConstPtr previous,
    UsdGenStats publishedStats, bool invalidateValues,
    uint64_t previousPublishedWorkerEpoch,
    UsdGenCompiler::DevicePlanCompiler const& deviceCompiler,
    std::shared_ptr<const UsdGenExecutionPlanHandle>* devicePlan)
{
    _coalescedRole = CoalescedRole::None;
    _coalescedDomain.reset();
    _hasCoalescedKey = false;
    _coalescedRegistration.reset();
    _cacheCandidate.reset();
    _publicationFence.reset();
    _acceptedDeviceIdentity.reset();
    _activeDeviceIdentity.reset();
    _stats = std::move(publishedStats);
    _lastReport = UsdGenDirtyReport{};
    const bool newDesc = desc != _descIdentity;
    const bool staleBaseline = previousPublishedWorkerEpoch != _lastCookedEpoch;
    _descIdentity = std::move(desc);
    _desc = *_descIdentity;
    _context = context;
    _devicePublicationEnabled = devicePublicationEnabled;
    _store = UsdGenGenerationStore(std::move(previous));
    if (newDesc || staleBaseline || _graphBaselineDetached)
        pending.structural = true;
    _lastDiagnostics = UsdGenDiagnostics{};
    _traceCompile.clear();
    // The published tiles may not come from this graph's last run: a cache
    // hit published another result, or another worker published. The graph
    // itself stays valid (every capture is re-validated against its inputs),
    // but no tile may be carried over from that baseline.
    _rebuildAllTiles = staleBaseline || _graphBaselineDetached;
    if (pending.structural || _graph.NodeCount() == 0 || deviceCompiler) {
        // Device plans capture the exact immutable request. Keep all common
        // reset, baseline and dirty handling above; only compilation is
        // supplied by the explicitly injected provider.
        TRACE_SCOPE("usdGen compile");
        auto const compileStart = std::chrono::steady_clock::now();
        bool const firstCompile = _graph.NodeCount() == 0;
        // A new descriptor recompiles incrementally (gate E-6): an operator
        // whose structural digest is unchanged keeps its op, capture and
        // buffer, and re-captures only when its capture digest or its input
        // moved. A graph a failed cook left behind is not trusted and is
        // rebuilt from scratch.
        bool const incremental = !deviceCompiler && !firstCompile && !_graphUntrusted;
        UsdGenCompileResult cr = deviceCompiler
            ? _compiler.CompileInjectedDevice(_desc, &_graph, deviceCompiler, devicePlan)
            : incremental ? _compiler.Recompile(_desc, &_graph)
                          : _compiler.Compile(_desc, &_graph);
        ++_stats.recompiles;
        if (TfDebug::IsEnabled(USDGEN_COMMIT)) {
            char buffer[256];
            std::snprintf(buffer, sizeof(buffer),
                "%s compile (%s) %.2f ms, %zu of %d nodes rebuilt%s",
                incremental ? "incremental" : "full",
                firstCompile ? "first" :
                _graphUntrusted ? "after a failed cook" :
                newDesc ? "new description" :
                staleBaseline ? "stale baseline" :
                _graphBaselineDetached ? "after a cache hit" : "structural dirt",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - compileStart).count(),
                cr.rebuilt.size(), _graph.NodeCount(),
                _rebuildAllTiles ? ", every tile rebuilt" : "");
            _traceCompile = buffer;
        }
        if (!cr.ok) {
            // A CPU recompile edits the graph in place and can fail halfway;
            // never reuse what it left. (A CUDA recompile is transactional
            // and keeps the last usable plan.)
            if (incremental && _desc.executionBackend != UsdGenExecutionBackend::Cuda)
                _graph = UsdGenGraph();
            _lastDiagnostics.errors = std::move(cr.errors);
            return false;
        }
        _graphUntrusted = false;
        _stats.cookedNodes = static_cast<uint64_t>(_graph.NodeCount());
    } else {
        for (auto const& kv : pending.nodeBits)
            if (kv.first < static_cast<UsdGenNodeId>(_graph.NodeCount())) _graph.MarkNode(kv.first, kv.second);
        for (auto const& kv : pending.surfaceBits) _graph.DirtySurface(kv.first, kv.second);
        if (pending.surfaceTopology)
            for (auto const& s : _desc.surfaces) _graph.DirtySurface(s.id, UsdGenDirtySurfaceTopo);
    }
    if (invalidateValues) for (UsdGenNodeId id=0; id<static_cast<UsdGenNodeId>(_graph.NodeCount()); ++id) {
        auto& node=_graph.Node(id); node.paramValueDigest=~uint64_t(0); node.lastParamDigest=0;
    }
    return true;
}

bool UsdGenSessionCooker::_FinalizeDevicePublication(
    std::shared_ptr<const UsdGenDeviceGeneration> device,
    std::shared_ptr<UsdGenExecutionCacheDomain> const& domain,
    UsdGenExecutionCacheKey const& key, double frame)
{
    if (!device || !domain) return false;
    UsdGenGeneration generation;
    generation.frame = frame;
    generation.device = std::move(device);
    generation.devicePresentation = _BuildDevicePresentation(_desc);
    auto prepared = _store.PreparePublication(std::move(generation));
    auto fence = domain->CapturePublicationFence(key);
    if (!fence || !domain->PublishIfCurrent(*fence, [this, &prepared] {
            return _store.PublishPrepared(std::move(prepared));
        })) return false;
    _publicationFence = ExecutionPublicationFence{domain, *fence};
    _graphBaselineDetached = false;
    _lastReport = UsdGenDirtyReport{};
    ++_stats.commits;
    _stats.publishedTiles = 0;
    _lastNodeStats.clear();
    size_t bytes = 0;
    auto const accepted = _store.Get();
    if (_GenerationBytes(*accepted, &bytes))
        _cacheCandidate = ExecutionCacheCandidate{key, accepted, bytes};
    return true;
}

struct UsdGenSessionCooker::CudaStages {
    std::shared_ptr<UsdGenCudaExecutionJob> job;
    UsdGenGenerationConstPtr previous;
    // Retain the exact shared domain selected when this asynchronous job was
    // created.  A later cooker turn cannot substitute a different domain for
    // its terminal publication.
    std::shared_ptr<UsdGenExecutionCacheDomain> cacheDomain;
    UsdGenExecutionCacheKey cacheKey;
    double frame = 0.0;
};

bool UsdGenSessionCooker::_InvalidatePoisonedCudaWorkspace()
{
    if (!_cudaWorkspace || !_cudaWorkspace->IsPoisoned()) return true;
    // The poisoned destructor deliberately quarantines its native state. Drop
    // cooker reachability only after all jobs borrowing the workspace have
    // released it; callers enforce that ordering. Cached/public generations
    // retain their independent immutable owners and retirement tickets.
    bool const cleared = _executionCacheDomain && _executionCacheDomain->Invalidate();
    _cacheCandidate.reset();
    if (!cleared) {
        // Keep the poisoned identity installed so every retry returns here;
        // replacing it while old entries remain resident would hide a failed
        // invalidation behind a new epoch instead of actually removing it.
        _lastDiagnostics.Error("CUDA execution cache poison invalidation failed");
        return false;
    }
    _cudaWorkspace.reset();
    _cudaWorkspaceDescription = SdfPath();
    return true;
}

bool UsdGenSessionCooker::ApplyDeviceContextLoss(
    UsdGenDeviceContextLoss const& loss)
{
    if (!loss.IsValid()) return loss.serial == 0;
    // State retains an ordered prefix until the batch's publication is
    // accepted. A superseded batch or a commit can therefore replay an older
    // member after a newer batch already applied the whole prefix. Serial
    // assignment and sorted delivery guarantee that member was already seen
    // by this serialized cooker, so the replay is idempotent.
    if (loss.serial <= _lastDeviceContextLossSerial) return true;

    if (loss.backend == UsdGenDeviceBackend::Cuda && _executionCacheDomain &&
        _executionCacheDomain->Key().backend == UsdGenDeviceBackend::Cuda &&
        (loss.deviceIndex < 0 ||
         loss.deviceIndex == _executionCacheDomain->Key().deviceIndex)) {
        // This method runs only on the serialized cooker work lane. Any older
        // async job has reached terminal completion before this turn begins,
        // so marking and retiring the borrowed workspace cannot race it.
        if (_cudaWorkspace && (loss.deviceIndex < 0 ||
                               loss.deviceIndex == _cudaWorkspace->DeviceIndex())) {
            _cudaWorkspace->MarkContextLost();
            if (!_InvalidatePoisonedCudaWorkspace()) return false;
        } else if (!_executionCacheDomain->Invalidate()) {
            _lastDiagnostics.Error("CUDA execution cache context invalidation failed");
            return false;
        }
    }
    if (loss.backend == UsdGenDeviceBackend::Vulkan && _deviceProvider) {
        auto const identity = _deviceProvider->Identity();
        if (loss.deviceIndex < 0 || loss.deviceIndex == identity.deviceIndex) {
            // Another Session may already have latched the shared provider.
            // Cache invalidation and serial acknowledgement still belong to
            // this cooker, irrespective of NotifyContextLost's return value.
            (void)_deviceProvider->NotifyContextLost(identity);
            if (_executionCacheDomain &&
                _executionCacheDomain->Key().backend == UsdGenDeviceBackend::Vulkan &&
                (loss.deviceIndex < 0 ||
                 loss.deviceIndex == _executionCacheDomain->Key().deviceIndex) &&
                !_executionCacheDomain->Invalidate()) {
                _lastDiagnostics.Error("Vulkan execution cache context invalidation failed");
                return false;
            }
        }
    }
    _lastDeviceContextLossSerial = loss.serial;
    return true;
}

void UsdGenSessionCooker::SetDeviceProvider(
    std::shared_ptr<UsdGenSessionDeviceProvider> provider)
{
    _deviceProvider = std::move(provider);
}

void UsdGenSessionCooker::CookCudaAsync(UsdGenExecutionRuntime& runtime,
    UsdGenExecutionPipeline::Cancellation const& cancellation,
    std::shared_ptr<const UsdGenGraphDesc> desc, UsdGenContext context,
    bool devicePublicationEnabled, UsdGenPendingDirty pending, double frame,
    UsdGenCommitReason reason, UsdGenGenerationConstPtr previous,
    UsdGenStats publishedStats, bool invalidateValues,
    uint64_t previousPublishedWorkerEpoch, int callerDevice, CudaCompletion completion,
    CoalescedHooks hooks)
{
    // Preparation owns the serial compiler/workspace state. It deliberately
    // ends before the ready-task job is submitted; source/operators/final can
    // then interleave with other sessions on the shared device dispatcher.
    // Preparation mutates private compiler/store state before subsequent
    // allocations or admission can throw. Record this epoch on every exit so
    // a later cook against an unchanged public baseline rebuilds that state.
    // This function returns before the pipeline can start this session's next
    // work item (the async completion/run-return gate retains the work lane),
    // so this is the sole writer: task-graph completion must not race it.
    struct LastCooked {
        uint64_t& out;
        uint64_t epoch;
        ~LastCooked() { out = epoch; }
    } lastCooked{_lastCookedEpoch, cancellation.epoch};
    TF_UNUSED(reason);
    if (!_Prepare(std::move(desc), context, devicePublicationEnabled, pending,
                  std::move(previous), std::move(publishedStats), invalidateValues,
                  previousPublishedWorkerEpoch)) {
        completion(_store.Get(), {});
        return;
    }
    if (callerDevice==-2) _lastDiagnostics.Error("CUDA caller device capture failed");
    else if (!_devicePublicationEnabled) _lastDiagnostics.Error("CUDA publication requires a device-aware consumer; renderer graphics interop is unavailable");
    else if (!_graph.CudaPlan()) _lastDiagnostics.Error("CUDA graph has no compiled execution plan");
    if (_lastDiagnostics.HasErrors()) {
        completion(_store.Get(), {});
        return;
    }
    if (!_InvalidatePoisonedCudaWorkspace()) {
        completion(_store.Get(), {});
        return;
    }
    if (_cudaWorkspace && _executionCacheDomain &&
        _executionCacheDomain->Key().backend == UsdGenDeviceBackend::Cuda &&
        _executionCacheDomain->Key().deviceIndex == _cudaWorkspace->DeviceIndex() &&
        !_ObserveExecutionCacheDomainEpoch()) {
        completion(_store.Get(), {});
        return;
    }
    const bool newWorkspace=!_cudaWorkspace || _cudaWorkspaceDescription!=_desc.description || (callerDevice>=0 && _cudaWorkspace->DeviceIndex()!=callerDevice);
    if (newWorkspace) {
        auto replacement = CreateCudaExecutionWorkspace(callerDevice, &_lastDiagnostics);
        if (!replacement) {
            completion(_store.Get(), {});
            return;
        }
        _cudaWorkspace = std::move(replacement);
        _cudaWorkspaceDescription = _desc.description;
    }
    if (!_SelectExecutionCacheDomain(UsdGenDeviceBackend::Cuda,
                                     _cudaWorkspace->DeviceIndex())) {
        completion(_store.Get(), {});
        return;
    }
    _observedCacheDomainEpoch = _executionCacheDomain->Epoch();
    auto cacheDomain = _executionCacheDomain;
    UsdGenExecutionCacheKey const cacheKey = _MakeExecutionCacheKey(
        _graph, _desc, _ExecutionCacheContext(_desc, _context,
            _cudaWorkspace->DeviceIndex(), cacheDomain->Epoch()), frame);
    ++_stats.executionCacheMisses;
    if (auto lease = cacheDomain->Lookup(cacheKey)) {
        UsdGenGeneration const& cached = **lease;
        auto const& device = cached.device;
        if (device && device->Owner() && device->Owner()->ProducerReady() &&
            device->Identity().backend == UsdGenDeviceBackend::Cuda &&
            device->Identity().deviceIndex == _cudaWorkspace->DeviceIndex()) {
            std::string republishReason;
            uint64_t const publicationGeneration =
                cacheDomain->AllocatePublicationGeneration(
                    static_cast<uint64_t>(_store.NextId()));
            auto republished = publicationGeneration
                ? device->Republish(publicationGeneration, &republishReason)
                : std::shared_ptr<const UsdGenDeviceGeneration>{};
            if (republished) {
                UsdGenGeneration hit = cached;
                hit.id = -1;
                hit.frame = frame;
                hit.device = std::move(republished);
                auto prepared = _store.PreparePublication(std::move(hit));
                auto fence = cacheDomain->CapturePublicationFence(cacheKey);
                if (fence && cacheDomain->PublishIfCurrent(*fence, [this, &prepared] {
                        return _store.PublishPrepared(std::move(prepared));
                    })) {
                    _publicationFence = ExecutionPublicationFence{
                        cacheDomain, *fence};
                    --_stats.executionCacheMisses;
                    ++_stats.executionCacheHits;
                    _graphBaselineDetached = true;
                    _lastReport = UsdGenDirtyReport{};
                    ++_stats.commits;
                    _stats.publishedTiles = 0;
                    _lastNodeStats.clear();
                    completion(_store.Get(), {});
                    return;
                }
            }
        }
    }
    _BeginCoalesced(cacheDomain, cacheKey, hooks);
    if (_coalescedRole == CoalescedRole::Follower ||
        _coalescedRole == CoalescedRole::Resident) {
        return;
    }
    auto stages = std::make_shared<CudaStages>();
    stages->previous = _store.Get();
    stages->cacheDomain = std::move(cacheDomain);
    stages->cacheKey = cacheKey;
    stages->frame = frame;
    stages->job = CreateCudaExecutionJob(_graph.CudaPlan(), *_cudaWorkspace, frame,
        static_cast<uint64_t>(_store.NextId()), &_lastDiagnostics,
        stages->previous && !newWorkspace ? stages->previous->device : nullptr);
    if (!stages->job) {
        _lastDiagnostics.Error("CUDA execution job creation failed");
        completion(_store.Get(), {});
        return;
    }
    auto metadata = GetCudaExecutionPlanMetadata(*_graph.CudaPlan());
    if (!metadata ||
        (metadata->Shape() != UsdGenExecutionPlanShape::LinearAuthoredChain &&
         metadata->Shape() != UsdGenExecutionPlanShape::SourceRootedUnaryDag &&
         metadata->Shape() != UsdGenExecutionPlanShape::SourceRootedValueDag) ||
        metadata->Operators().empty() ||
        metadata->Tasks().size() != CudaExecutionJobOperatorCount(*stages->job) + 2 ||
        !metadata->FindTask(metadata->TerminalTask())) {
        completion(_store.Get(), std::make_exception_ptr(
            std::runtime_error("CUDA execution plan metadata is missing or incompatible")));
        return;
    }
    for (uint32_t i = 0; i != metadata->Tasks().size(); ++i) {
        if (!metadata->FindTask(i)) {
            completion(_store.Get(), std::make_exception_ptr(
                std::runtime_error("CUDA execution plan task ids are not dense")));
            return;
        }
    }
    std::string dependencyReason;
    if (!UsdGenExecutionDependencyCompiler::Validate(
            metadata->Tasks(), metadata->Values(), &dependencyReason)) {
        completion(_store.Get(), std::make_exception_ptr(std::runtime_error(
            "CUDA execution plan dependencies are invalid: " + dependencyReason)));
        return;
    }
    auto dispatcher = UsdGenExecutionTaskGraph::GetOrCreate(
        runtime, "cuda", _cudaWorkspace->DeviceIndex());
    if (!dispatcher) {
        completion(_store.Get(), std::make_exception_ptr(
            std::runtime_error("CUDA dispatcher unavailable")));
        return;
    }
    UsdGenExecutionTaskGraph::Job graph;
    graph.cancellation = cancellation;
    auto const* sourceMetadata = metadata->FindTask(0);
    if (!sourceMetadata || sourceMetadata->kind != UsdGenExecutionTaskKind::Source ||
        sourceMetadata->id != graph.tasks.size()) {
        completion(_store.Get(), std::make_exception_ptr(
            std::runtime_error("CUDA source task metadata is missing")));
        return;
    }
    graph.tasks.push_back({sourceMetadata->dependencies, [stages](auto const& taskCancel, auto done) {
        auto completion = std::make_shared<UsdGenExecutionTaskGraph::TaskCompletion>(
            std::move(done));
        try {
            if (taskCancel.Superseded()) {
                (*completion)({}, std::make_exception_ptr(
                    std::runtime_error("CUDA source stage failed")));
                return;
            }
            if (!ExecuteCudaJobSourceAsync(stages->job,
                    [completion](bool ok) {
                if (ok) (*completion)({}, {});
                else (*completion)({}, std::make_exception_ptr(
                    std::runtime_error("CUDA source stage failed")));
            })) {
                (*completion)({}, std::make_exception_ptr(
                    std::runtime_error("CUDA source stage failed")));
            }
        } catch (...) { (*completion)({}, std::current_exception()); }
    }});
    const size_t operators=CudaExecutionJobOperatorCount(*stages->job);
    for (size_t i = 0; i < operators; ++i) {
        auto const* operatorMetadata = metadata->FindTask(static_cast<uint32_t>(i + 1));
        if (!operatorMetadata || operatorMetadata->kind != UsdGenExecutionTaskKind::Operator ||
            operatorMetadata->id != graph.tasks.size()) {
            completion(_store.Get(), std::make_exception_ptr(
                std::runtime_error("CUDA operator task metadata is missing")));
            return;
        }
        graph.tasks.push_back({operatorMetadata->dependencies,
                               [stages, i](auto const& taskCancel, auto done) {
            auto completion = std::make_shared<UsdGenExecutionTaskGraph::TaskCompletion>(
                std::move(done));
            try {
                if (taskCancel.Superseded()) {
                    (*completion)({}, std::make_exception_ptr(
                        std::runtime_error("CUDA operator stage " + std::to_string(i) + " superseded")));
                    return;
                }
                if (!ExecuteCudaJobOperatorAsync(stages->job, i,
                        [completion, i](bool ok) {
                    if (ok) (*completion)({}, {});
                    else (*completion)({}, std::make_exception_ptr(
                        std::runtime_error("CUDA operator stage " + std::to_string(i) + " completion failed")));
                })) {
                    (*completion)({}, std::make_exception_ptr(
                        std::runtime_error("CUDA operator stage " + std::to_string(i) + " submission failed")));
                }
            } catch (...) { (*completion)({}, std::current_exception()); }
        }});
    }
    auto const* terminalMetadata = metadata->FindTask(metadata->TerminalTask());
    if (!terminalMetadata || terminalMetadata->kind != UsdGenExecutionTaskKind::Publication ||
        terminalMetadata->id != graph.tasks.size()) {
        completion(_store.Get(), std::make_exception_ptr(
            std::runtime_error("CUDA publication task metadata is missing")));
        return;
    }
    graph.tasks.push_back({terminalMetadata->dependencies,
                           [this, stages](auto const&, auto done) {
        auto completion = std::make_shared<UsdGenExecutionTaskGraph::TaskCompletion>(
            std::move(done));
        try {
            if (!FinalizeCudaExecutionJobAsync(stages->job,
                    [this, stages, completion](std::shared_ptr<const UsdGenDeviceGeneration> device) {
                try {
                    if (!device) {
                        (*completion)({}, std::make_exception_ptr(
                            std::runtime_error("CUDA final stage failed")));
                        return;
                    }
                    // Allocate native diagnostic snapshots before advancing
                    // the shared private publication state.
                    auto bindingStats = std::make_shared<const std::vector<UsdGenCudaBindingStats>>(
                        GetCudaBindingStats(*_cudaWorkspace));
                    if (!_FinalizeDevicePublication(std::move(device),
                            stages->cacheDomain, stages->cacheKey, stages->frame)) {
                        (*completion)({}, std::make_exception_ptr(std::runtime_error(
                            "CUDA execution domain was invalidated before publication")));
                        return;
                    }
                    // This is private cooker state for the sole in-flight
                    // Session work lane.  Public snapshot/pending acceptance
                    // remains the pipeline publication produced by session.cpp.
                    std::atomic_store(&_cudaBindingStats, std::move(bindingStats));
                    (*completion)([] {}, {});
                } catch (...) { (*completion)({}, std::current_exception()); }
            })) {
                (*completion)({}, std::make_exception_ptr(
                    std::runtime_error("CUDA final stage failed")));
            }
        } catch (...) { (*completion)({}, std::current_exception()); }
    }});
    graph.finalTask = terminalMetadata->id;
    auto terminal=std::make_shared<CudaCompletion>(std::move(completion));
    graph.completion = [this, stages, terminal](
        auto, std::exception_ptr error) mutable {
        // Task completion proves no dispatcher closure can still borrow the
        // job workspace. Destroy the job first, because its destructor can be
        // the operation that marks an unprovable transaction poisoned.
        stages->job.reset();
        if (!_InvalidatePoisonedCudaWorkspace() && !error)
            error = std::make_exception_ptr(std::runtime_error(
                "CUDA execution cache poison invalidation failed"));
        if (*terminal) (*terminal)(_store.Get(), error);
    };
    if (!dispatcher->Submit(std::move(graph))) {
        if (*terminal) {
            (*terminal)(_store.Get(), std::make_exception_ptr(
                std::runtime_error("CUDA dispatcher rejected job")));
        }
    }
}

void UsdGenSessionCooker::CookDeviceAsync(UsdGenExecutionRuntime& runtime,
    UsdGenExecutionPipeline::Cancellation const& cancellation,
    std::shared_ptr<const UsdGenGraphDesc> desc, UsdGenContext context,
    bool devicePublicationEnabled, UsdGenPendingDirty pending, double frame,
    UsdGenCommitReason reason, UsdGenGenerationConstPtr previous,
    UsdGenStats publishedStats, bool invalidateValues,
    uint64_t previousPublishedWorkerEpoch, int callerDevice,
    std::shared_ptr<UsdGenSessionDeviceProvider> provider,
    DeviceReturnBinder bindReturn, CudaCompletion completion, CoalescedHooks hooks)
{
    TF_UNUSED(runtime); TF_UNUSED(reason); TF_UNUSED(callerDevice);
    struct LastCooked { uint64_t& out; uint64_t epoch; ~LastCooked() { out = epoch; } }
        lastCooked{_lastCookedEpoch, cancellation.epoch};
    if (!provider || !bindReturn) { _lastDiagnostics.Error("Vulkan Session provider is not injected"); completion(_store.Get(), {}); return; }
    _deviceProvider = provider;
    std::shared_ptr<const UsdGenExecutionPlanHandle> plan;
    if (!_Prepare(std::move(desc), context, devicePublicationEnabled, pending,
                  std::move(previous), std::move(publishedStats), invalidateValues,
                  previousPublishedWorkerEpoch,
                  [provider](UsdGenGraphDesc const& d, UsdGenDiagnostics* diag) {
                      return provider->Compile(d, diag);
                  }, &plan)) { completion(_store.Get(), {}); return; }
    auto const identity = provider->Identity();
    if (!_devicePublicationEnabled || !plan ||
        identity.backend != UsdGenDeviceBackend::Vulkan ||
        !identity.capabilityVersion || cancellation.Superseded()) {
        _lastDiagnostics.Error("Vulkan device publication is unavailable for this request");
        completion(_store.Get(), {}); return;
    }
    _activeDeviceIdentity = identity;
    if (!_SelectExecutionCacheDomain(identity.backend, identity.deviceIndex,
                                    identity.logicalContextIdentity) ||
        !_ObserveExecutionCacheDomainEpoch()) { completion(_store.Get(), {}); return; }
    auto domain = _executionCacheDomain;
    if (!domain || domain->Key().contextIdentity != identity.logicalContextIdentity) {
        _lastDiagnostics.Error("Vulkan execution cache domain logical context identity mismatch");
        completion(_store.Get(), {}); return;
    }
    _observedCacheDomainEpoch = domain->Epoch();
    auto cacheContext = MakeUsdGenExecutionContext(UsdGenExecutionBackend::Vulkan,
        identity.capabilityVersion, identity.deviceIndex, domain->Epoch(), _context);
    auto key = _MakeExecutionCacheKey(_graph, _desc, cacheContext, frame);
    ++_stats.executionCacheMisses;
    if (auto lease = domain->Lookup(key)) {
        UsdGenGeneration const& cached = **lease;
        auto const& cachedDevice = cached.device;
        if (provider->Identity() == identity && cachedDevice && cachedDevice->Owner() && cachedDevice->Owner()->ProducerReady() &&
            cachedDevice->Identity().backend == UsdGenDeviceBackend::Vulkan &&
            cachedDevice->Identity().deviceIndex == identity.deviceIndex) {
            std::string republishReason;
            uint64_t const publicationGeneration = domain->AllocatePublicationGeneration(
                static_cast<uint64_t>(_store.NextId()));
            auto republished = publicationGeneration
                ? cachedDevice->RepublishPreservingRevisions(publicationGeneration, &republishReason)
                : std::shared_ptr<const UsdGenDeviceGeneration>{};
            if (republished) {
                // Republish changes only the consumer/publication identity;
                // cached topology/value revisions remain immutable.
                UsdGenGeneration hit = cached;
                hit.id = -1; hit.frame = frame; hit.device = std::move(republished);
                auto prepared = _store.PreparePublication(std::move(hit));
                auto fence = domain->CapturePublicationFence(key);
                if (fence && domain->PublishIfCurrent(*fence, [this, &prepared] {
                        return _store.PublishPrepared(std::move(prepared));
                    })) {
                    _publicationFence = ExecutionPublicationFence{domain, *fence};
                    _acceptedDeviceIdentity = identity;
                    --_stats.executionCacheMisses; ++_stats.executionCacheHits;
                    _graphBaselineDetached = true; _lastReport = UsdGenDirtyReport{};
                    ++_stats.commits; _stats.publishedTiles = 0; _lastNodeStats.clear();
                    completion(_store.Get(), {}); return;
                }
            }
        }
    }
    _BeginCoalesced(domain, key, hooks);
    if (_coalescedRole == CoalescedRole::Follower || _coalescedRole == CoalescedRole::Resident) return;
    auto metadata = plan->Metadata();
    size_t modifiers = 0;
    if (metadata) for (auto const& task : metadata->Tasks())
        if (task.kind == UsdGenExecutionTaskKind::Operator) ++modifiers;
    if (!metadata || !modifiers ||
        (metadata->Shape() != UsdGenExecutionPlanShape::LinearAuthoredChain &&
         metadata->Shape() != UsdGenExecutionPlanShape::SourceRootedUnaryDag &&
         metadata->Shape() != UsdGenExecutionPlanShape::SourceRootedValueDag)) {
        _lastDiagnostics.Error("Vulkan publication revision contract is missing");
        completion(_store.Get(), {}); return;
    }
    std::vector<uint64_t> intermediates(modifiers - 1);
    uint64_t const source = domain->AllocatePublicationGeneration(
        static_cast<uint64_t>(_store.NextId()));
    uint64_t previousRevision = source;
    for (auto& value : intermediates) {
        if (!previousRevision) break;
        value = domain->AllocatePublicationGeneration(previousRevision);
        if (value <= previousRevision) { previousRevision = 0; break; }
        previousRevision = value;
    }
    uint64_t const final = previousRevision ? domain->AllocatePublicationGeneration(previousRevision) : 0;
    if (!source || !final || final <= previousRevision) {
        _lastDiagnostics.Error("Vulkan publication revision allocation failed"); completion(_store.Get(), {}); return;
    }
    uint64_t expectedTopology = 0;
    if (!_TerminalTopologyRevision(*metadata, source, final, intermediates, &expectedTopology)) {
        _lastDiagnostics.Error("Vulkan terminal topology revision contract is missing");
        completion(_store.Get(), {}); return;
    }
    auto handler = [this, provider, domain, key, frame, identity, source, final, intermediates, expectedTopology,
                    completion](
        std::shared_ptr<const UsdGenSessionDeviceResult> result) mutable {
        if (result && result->error) {
            try { std::rethrow_exception(result->error); }
            catch (std::exception const& error) {
                _lastDiagnostics.Error(std::string("Vulkan device execution failed: ") + error.what());
            } catch (...) {
                _lastDiagnostics.Error("Vulkan device execution failed with an unknown error");
            }
            completion(_store.Get(), result->error);
            return;
        }
        if (!result || result->error || !result->generation || result->identity != identity ||
            provider->Identity() != identity ||
            result->revisions.topologyVersion != source ||
            result->revisions.sourceValueVersion != source ||
            result->revisions.finalValueVersion != final ||
            result->revisions.intermediateValueVersions != intermediates ||
            !result->generation->Owner() || !result->generation->Owner()->ProducerReady() ||
            result->generation->Identity().backend != UsdGenDeviceBackend::Vulkan ||
            result->generation->Identity().deviceIndex != identity.deviceIndex ||
            result->generation->Identity().generation != final ||
            result->generation->Geometry().topologyVersion != expectedTopology ||
            result->generation->Geometry().valueVersion != final) {
            _lastDiagnostics.Error("Vulkan device result failed immutable identity or readiness validation");
            completion(_store.Get(), result && result->error ? result->error :
                std::make_exception_ptr(std::runtime_error("Vulkan device result validation failed"))); return;
        }
        if (!_FinalizeDevicePublication(result->generation, domain, key, frame)) {
            completion(_store.Get(), std::make_exception_ptr(std::runtime_error(
                "Vulkan execution domain was invalidated before publication"))); return;
        }
        _acceptedDeviceIdentity = identity;
        completion(_store.Get(), {});
    };
    UsdGenSessionDeviceRequest request;
    request.plan = std::move(plan); request.identity = identity;
    request.publicationGeneration = final;
    request.evaluationFrame = frame;
    request.authoritativeRevisions = UsdGenSessionDeviceRevisions{source, source, final, std::move(intermediates)};
    request.cancellation = cancellation;
    request.tool.snapshotVersion = final;
    // Binder state can retain the Session ingress relay. Drop it before
    // provider admission so handler/result/provider cannot close a cycle back
    // through the binder after the one return route has been materialized.
    auto binder = std::move(bindReturn);
    try { request.returnTransport = binder(std::move(handler)); }
    catch (...) {
        _lastDiagnostics.Error("Vulkan Session return transport binding failed");
        completion(_store.Get(), std::current_exception()); return;
    }
    binder = {};
    if (!request.returnTransport || !provider->Submit(std::move(request))) {
        _lastDiagnostics.Error("Vulkan device provider rejected owner admission");
        completion(_store.Get(), {});
    }
}

std::vector<UsdGenCudaBindingStats> UsdGenSessionCooker::CudaBindingStats() const {
    return *std::atomic_load(&_cudaBindingStats);
}

std::optional<UsdGenSessionCooker::ExecutionCacheCandidate>
UsdGenSessionCooker::TakeCacheCandidate()
{
    std::optional<ExecutionCacheCandidate> result = std::move(_cacheCandidate);
    _cacheCandidate.reset();
    return result;
}

std::optional<UsdGenSessionCooker::ExecutionPublicationFence>
UsdGenSessionCooker::TakePublicationFence()
{
    std::optional<ExecutionPublicationFence> result = std::move(_publicationFence);
    _publicationFence.reset();
    return result;
}

bool UsdGenSessionCooker::AdoptCoalesced(
    ExecutionPublicationFence const& publication,
    CoalescedLease const& lease, double frame,
    UsdGenGenerationConstPtr* generation)
{
    if (!generation || !publication || !lease ||
        !_coalescedDomain || publication.domain != _coalescedDomain ||
        !_hasCoalescedKey || lease.Key() != _coalescedKey) return false;
    auto payload = lease.SharedPayload();
    if (!payload) return false;
    auto previous = _store.Get();
    UsdGenGeneration hit = *payload;
    hit.id = -1;
    hit.frame = frame;
    if (_desc.executionBackend == UsdGenExecutionBackend::Cuda) {
        if (!_cudaWorkspace || _cudaWorkspace->IsPoisoned() || !hit.device ||
            !hit.device->Owner() || !hit.device->Owner()->ProducerReady() ||
            hit.device->Identity().deviceIndex != _cudaWorkspace->DeviceIndex())
            return false;
        std::string reason;
        uint64_t const id = _coalescedDomain->AllocatePublicationGeneration(
            static_cast<uint64_t>(_store.NextId()));
        auto republished = id ? hit.device->Republish(id, &reason)
                              : std::shared_ptr<const UsdGenDeviceGeneration>{};
        if (!republished) {
            _lastDiagnostics.Error(reason.empty()
                ? "coalesced CUDA cache result could not be republished" : reason);
            return false;
        }
        hit.device = std::move(republished);
    } else if (_desc.executionBackend == UsdGenExecutionBackend::Vulkan) {
        if (!_deviceProvider || !hit.device || !hit.device->Owner() ||
            !hit.device->Owner()->ProducerReady() ||
            _coalescedDomain->Key().backend != UsdGenDeviceBackend::Vulkan) return false;
        auto const identity = _deviceProvider->Identity();
        if (identity.backend != UsdGenDeviceBackend::Vulkan ||
            hit.device->Identity().backend != UsdGenDeviceBackend::Vulkan ||
            hit.device->Identity().deviceIndex != identity.deviceIndex ||
            _coalescedDomain->Key().deviceIndex != identity.deviceIndex ||
            _coalescedDomain->Key().contextIdentity != identity.logicalContextIdentity ||
            _coalescedKey.context.deviceGeneration != _coalescedDomain->Epoch()) return false;
        // The native generation's topology/value revisions are immutable
        // cache payload. Allocate only a fresh consumer/publication identity.
        std::string reason;
        uint64_t const id = _coalescedDomain->AllocatePublicationGeneration(
            static_cast<uint64_t>(_store.NextId()));
        auto republished = id ? hit.device->RepublishPreservingRevisions(id, &reason)
                              : std::shared_ptr<const UsdGenDeviceGeneration>{};
        if (!republished) {
            _lastDiagnostics.Error(reason.empty()
                ? "coalesced Vulkan cache result could not be republished" : reason);
            return false;
        }
        hit.device = std::move(republished);
    }
    if (_desc.executionBackend == UsdGenExecutionBackend::Vulkan &&
        (!_deviceProvider || !_activeDeviceIdentity ||
         _deviceProvider->Identity() != *_activeDeviceIdentity)) return false;
    auto prepared = _store.PreparePublication(std::move(hit));
    if (!_coalescedDomain->PublishIfCurrent(publication.fence, [this, &prepared] {
            return _store.PublishPrepared(std::move(prepared));
        })) return false;
    _publicationFence = publication;
    if (_desc.executionBackend == UsdGenExecutionBackend::Vulkan)
        _acceptedDeviceIdentity = _activeDeviceIdentity;
    --_stats.executionCacheMisses;
    ++_stats.executionCacheHits;
    if (_coalescedRole == CoalescedRole::Follower)
        ++_stats.executionCacheCoalesced;
    _graphBaselineDetached = true;
    _lastReport = (_desc.executionBackend == UsdGenExecutionBackend::Cuda ||
                   _desc.executionBackend == UsdGenExecutionBackend::Vulkan)
        ? UsdGenDirtyReport{}
        : _store.Diff(previous ? *previous : UsdGenGeneration{}, *_store.Get());
    ++_stats.commits;
    _stats.publishedTiles = static_cast<uint64_t>(_store.Get()->tiles.size());
    _lastNodeStats.clear();
    *generation = _store.Get();
    _coalescedRole = CoalescedRole::None;
    _coalescedDomain.reset();
    _hasCoalescedKey = false;
    _coalescedRegistration.reset();
    return true;
}

void UsdGenSessionCooker::AbandonCoalesced(bool notify)
{
    auto domain = std::move(_coalescedDomain);
    auto registration = std::move(_coalescedRegistration);
    _coalescedRole = CoalescedRole::None;
    _hasCoalescedKey = false;
    if (!domain || !registration) return;
    if (notify) domain->CancelCoalescedAndNotify(*registration);
    else domain->CancelCoalesced(*registration);
}

bool UsdGenSessionCooker::PublishIfCurrent(
    ExecutionPublicationFence const& publication,
    std::function<std::shared_ptr<const void>()> publish)
{
    return publication && publish && publication.domain->PublishIfCurrent(
        publication.fence, std::move(publish));
}

bool UsdGenSessionCooker::CommitCacheCandidate(
    ExecutionCacheCandidate const &candidate)
{
    // The candidate is tied to the exact immutable generation accepted by the
    // cooker. The command owner calls this only from the current publication
    // action, after publishing its snapshot; a superseded/cancelled action
    // never reaches here.
    auto rejectBecause = [this, &candidate](char const *why) {
        ++_stats.executionCacheAdmissionFailures;
        TF_DEBUG(USDGEN_COMMIT).Msg("usdGen cache     key %016llx not admitted: %s\n",
                                    (unsigned long long)candidate.key.Hash(), why);
        return false;
    };
    auto reject = [&rejectBecause] { return rejectBecause("backend identity"); };
    if (!candidate.generation || candidate.generation != _store.Get() ||
        !candidate.key.IsValid()) return rejectBecause("stale or invalid candidate");
    double frame = 0.0;
    std::memcpy(&frame, &candidate.key.frameBits, sizeof(frame));
    UsdGenExecutionContext context = _ExecutionCacheContext(
        _desc, _context, -1,
        _executionCacheDomain ? _executionCacheDomain->Epoch() : 0);
    if (_desc.executionBackend == UsdGenExecutionBackend::Cuda) {
        if (!_cudaWorkspace || _cudaWorkspace->IsPoisoned() ||
            !candidate.generation->device ||
            !candidate.generation->device->Owner() ||
            !candidate.generation->device->Owner()->ProducerReady()) return reject();
        context = _ExecutionCacheContext(_desc, _context,
            _cudaWorkspace->DeviceIndex(), _executionCacheDomain->Epoch());
        auto const& identity = candidate.generation->device->Identity();
        if (identity.backend != UsdGenDeviceBackend::Cuda ||
            identity.deviceIndex != context.deviceIndex) return reject();
    } else if (_desc.executionBackend == UsdGenExecutionBackend::Vulkan) {
        if (!_deviceProvider || !candidate.generation->device ||
            !candidate.generation->device->Owner() ||
            !candidate.generation->device->Owner()->ProducerReady() ||
            !_executionCacheDomain) return reject();
        auto const identity = _deviceProvider->Identity();
        if (!_acceptedDeviceIdentity || *_acceptedDeviceIdentity != identity) return reject();
        if (_executionCacheDomain->Key().contextIdentity != identity.logicalContextIdentity ||
            candidate.key.context.backend != UsdGenDeviceBackend::Vulkan ||
            candidate.key.context.capabilityVersion != identity.capabilityVersion ||
            candidate.key.context.deviceIndex != identity.deviceIndex ||
            candidate.key.context.deviceGeneration != _executionCacheDomain->Epoch()) return reject();
        context = MakeUsdGenExecutionContext(UsdGenExecutionBackend::Vulkan,
            identity.capabilityVersion, identity.deviceIndex,
            _executionCacheDomain->Epoch(), _context);
        auto const& deviceIdentity = candidate.generation->device->Identity();
        if (identity.backend != UsdGenDeviceBackend::Vulkan ||
            deviceIdentity.backend != UsdGenDeviceBackend::Vulkan ||
            deviceIdentity.deviceIndex != identity.deviceIndex) return reject();
    }
    size_t exactBytes = 0;
    if (!_GenerationBytes(*candidate.generation, &exactBytes) ||
        exactBytes != candidate.bytes) return rejectBecause("size changed");
    // An entry larger than the whole domain budget can never be admitted:
    // the store refuses without eviction. Prove that first, before the full
    // key re-hash below re-streams every input array (~50 ms at 1M).
    bool const settleCoalesced = _coalescedRegistration &&
        _coalescedRegistration->leader &&
        _coalescedDomain == _executionCacheDomain &&
        _coalescedRegistration->key == candidate.key;
    auto const &targetDomain = settleCoalesced
        ? _coalescedDomain : _executionCacheDomain;
    if (!targetDomain) return rejectBecause("store refused");
    if (targetDomain->MaxBytes() == 0 ||
        candidate.bytes > targetDomain->MaxBytes()) {
        // Same terminal state a refused ResolveCoalesced would deliver:
        // the flight settles Failed, waiters get nullopt, the debug reason
        // matches the store's own budget message (stale epochs still report
        // stale, exactly as the resolve path would).
        std::string budgetReason =
            "execution cache entry exceeds its byte budget";
        if (settleCoalesced) {
            _coalescedDomain->RejectCoalesced(*_coalescedRegistration,
                UsdGenExecutionCacheDomain::CoalescedStatus::Failed,
                &budgetReason);
            _coalescedRegistration.reset();
            _coalescedDomain.reset();
            _coalescedRole = CoalescedRole::None;
        }
        return rejectBecause(budgetReason.c_str());
    }
    UsdGenExecutionCacheKey current = _MakeExecutionCacheKey(
        _graph, _desc, context, frame);
    if (current != candidate.key)
        return rejectBecause("the key moved between cook and publication");
    std::string reason;
    bool admitted = false;
    if (settleCoalesced) {
        admitted = _coalescedDomain->ResolveCoalesced(
            *_coalescedRegistration, candidate.generation, candidate.bytes,
            &reason);
        _coalescedRegistration.reset();
        _coalescedDomain.reset();
        _coalescedRole = CoalescedRole::None;
    } else {
        admitted = _executionCacheDomain && _executionCacheDomain->Insert(
            candidate.key, candidate.generation, candidate.bytes, &reason);
    }
    if (!admitted) return rejectBecause(reason.empty() ? "store refused" : reason.c_str());
    ++_stats.executionCacheAdmissions;
    TF_DEBUG(USDGEN_COMMIT).Msg("usdGen cache     key %016llx admitted, %.1f MB (domain %.1f of %.1f MB, %zu entries)\n",
        (unsigned long long)candidate.key.Hash(), candidate.bytes / 1048576.0,
        _executionCacheDomain ? _executionCacheDomain->Bytes() / 1048576.0 : 0.0,
        _executionCacheDomain ? _executionCacheDomain->MaxBytes() / 1048576.0 : 0.0,
        _executionCacheDomain ? _executionCacheDomain->Size() : size_t(0));
    return true;
}

UsdGenGenerationConstPtr UsdGenSessionCooker::Cook(
    std::shared_ptr<const UsdGenGraphDesc> desc, UsdGenContext context,
    bool devicePublicationEnabled, UsdGenPendingDirty pending, double frame,
    UsdGenCommitReason reason, UsdGenGenerationConstPtr previous,
UsdGenStats publishedStats, bool invalidateValues,
    uint64_t previousPublishedWorkerEpoch, uint64_t workEpoch, int callerDevice,
    CoalescedHooks hooks, UsdGenSession::TileProgressCallback progress,
    std::function<bool()> superseded)
{
    // _Prepare resets private accounting from the last published baseline.
    struct LastCooked { uint64_t& out; uint64_t epoch; ~LastCooked() { out = epoch; } }
        lastCooked{_lastCookedEpoch, workEpoch};
    const auto t0 = std::chrono::steady_clock::now();
    TRACE_FUNCTION();
    _CommitPhases phases;
    TF_UNUSED(reason);
    if (!_Prepare(std::move(desc), context, devicePublicationEnabled, pending,
                  std::move(previous), std::move(publishedStats), invalidateValues,
                  previousPublishedWorkerEpoch)) return _store.Get();
    phases.Phase("prepare");

    const uint64_t myReq = workEpoch;
    if (_desc.executionBackend != UsdGenExecutionBackend::Cuda &&
        !_SelectExecutionCacheDomain(UsdGenDeviceBackend::CpuReference, -1))
        return _store.Get();
    if (_desc.executionBackend != UsdGenExecutionBackend::Cuda &&
        !_ObserveExecutionCacheDomainEpoch())
        return _store.Get();
    // Keep the exact domain alive for every publication in this cook.  CUDA
    // may select its device-specific domain below; CPU keeps this one.
    auto cacheDomain = _executionCacheDomain;
    UsdGenExecutionCacheKey cacheKey = [&] {
        TRACE_SCOPE("usdGen execution cache key");
        return _MakeExecutionCacheKey(_graph, _desc,
            _ExecutionCacheContext(_desc, _context, -1,
                cacheDomain ? cacheDomain->Epoch() : 0), frame);
    }();
    phases.Phase("cache key");
    TF_DEBUG(USDGEN_COMMIT).Msg("usdGen cache     frame %g looks up key %016llx\n",
                                frame, (unsigned long long)cacheKey.Hash());

    // Cache lookup occurs after preparation so the key contains the compiled
    // graph's exact input tuple and current layout contract. A hit is copied
    // into a new generation publication; the cache's immutable payload is
    // never handed to the mutable generation store by alias.
    if (_desc.executionBackend != UsdGenExecutionBackend::Cuda) {
        ++_stats.executionCacheMisses;
        auto lease = cacheDomain->Lookup(cacheKey);
        if (lease) {
            UsdGenGeneration hit = **lease;
            hit.id = -1;
            hit.frame = frame;
            UsdGenGenerationConstPtr prev = _store.Get();
            auto prepared = _store.PreparePublication(std::move(hit));
            auto fence = cacheDomain->CapturePublicationFence(cacheKey);
            if (fence && cacheDomain->PublishIfCurrent(*fence, [this, &prepared] {
                    return _store.PublishPrepared(std::move(prepared));
                })) {
                _publicationFence = ExecutionPublicationFence{
                    cacheDomain, *fence};
                --_stats.executionCacheMisses;
                ++_stats.executionCacheHits;
                UsdGenGenerationConstPtr next = _store.Get();
                _lastReport = _store.Diff(prev ? *prev : UsdGenGeneration{}, *next);
                ++_stats.commits;
                _stats.publishedTiles = static_cast<uint64_t>(next->tiles.size());
                _lastNodeStats.clear();
                _graphBaselineDetached = true;
                phases.Phase("cache hit publish");
                phases.Print(frame, _traceCompile,
                             "cache hit (operators not run; next cook recompiles)",
                             _desc.description);
                return next;
            }
        }
        phases.Phase("cache lookup");
        _BeginCoalesced(cacheDomain, cacheKey, hooks);
        if (_coalescedRole == CoalescedRole::Follower ||
            _coalescedRole == CoalescedRole::Resident)
            return _store.Get();
    }

    if (_desc.executionBackend == UsdGenExecutionBackend::Cuda) {
        auto reject = [&]() {
            return _store.Get();
        };
        // -2 is the command boundary's explicit cudaGetDevice failure
        // sentinel.  Reusing a former workspace in that case would silently
        // execute on an unrelated device selected by a worker thread.
        if (callerDevice == -2) {
            _lastDiagnostics.Error("CUDA caller device capture failed");
            return reject();
        }
        if (!_devicePublicationEnabled) {
            _lastDiagnostics.Error("CUDA publication requires a device-aware consumer; renderer graphics interop is unavailable");
            return reject();
        }
        if (!_graph.CudaPlan()) {
            _lastDiagnostics.Error("CUDA graph has no compiled execution plan");
            return reject();
        }
        if (!_InvalidatePoisonedCudaWorkspace()) return reject();
        if (_cudaWorkspace && _executionCacheDomain &&
            _executionCacheDomain->Key().backend == UsdGenDeviceBackend::Cuda &&
            _executionCacheDomain->Key().deviceIndex == _cudaWorkspace->DeviceIndex() &&
            !_ObserveExecutionCacheDomainEpoch()) return reject();
        const bool newWorkspace = !_cudaWorkspace ||
            _cudaWorkspaceDescription != _desc.description ||
            (callerDevice >= 0 && _cudaWorkspace->DeviceIndex() != callerDevice);
        if (newWorkspace) {
            auto replacement = CreateCudaExecutionWorkspace(
                callerDevice, &_lastDiagnostics);
            if (!replacement) return reject();
            _cudaWorkspace = std::move(replacement);
            _cudaWorkspaceDescription = _desc.description;
        }
        if (!_cudaWorkspace) return reject();
        if (!_SelectExecutionCacheDomain(UsdGenDeviceBackend::Cuda,
                                         _cudaWorkspace->DeviceIndex()))
            return reject();
        _observedCacheDomainEpoch = _executionCacheDomain->Epoch();
        cacheDomain = _executionCacheDomain;
        cacheKey = _MakeExecutionCacheKey(_graph, _desc,
            _ExecutionCacheContext(_desc, _context,
                _cudaWorkspace->DeviceIndex(), cacheDomain->Epoch()), frame);
        ++_stats.executionCacheMisses;
        if (auto lease = cacheDomain->Lookup(cacheKey)) {
            UsdGenGeneration const& cached = **lease;
            auto const& cachedDevice = cached.device;
            if (cachedDevice && cachedDevice->Owner() &&
                cachedDevice->Owner()->ProducerReady() &&
                cachedDevice->Identity().backend == UsdGenDeviceBackend::Cuda &&
                cachedDevice->Identity().deviceIndex == _cudaWorkspace->DeviceIndex()) {
                std::string republishReason;
                uint64_t const publicationGeneration =
                    cacheDomain->AllocatePublicationGeneration(
                        static_cast<uint64_t>(_store.NextId()));
                auto republished = publicationGeneration
                    ? cachedDevice->Republish(publicationGeneration, &republishReason)
                    : std::shared_ptr<const UsdGenDeviceGeneration>{};
                if (republished) {
                    UsdGenGeneration hit = cached;
                    hit.id = -1;
                    hit.frame = frame;
                    hit.device = std::move(republished);
                    auto prepared = _store.PreparePublication(std::move(hit));
                    auto fence = cacheDomain->CapturePublicationFence(cacheKey);
                    if (fence && cacheDomain->PublishIfCurrent(*fence, [this, &prepared] {
                            return _store.PublishPrepared(std::move(prepared));
                        })) {
                        _publicationFence = ExecutionPublicationFence{
                            cacheDomain, *fence};
                        --_stats.executionCacheMisses;
                        ++_stats.executionCacheHits;
                        _lastReport = UsdGenDirtyReport{};
                        ++_stats.commits;
                        _stats.publishedTiles = 0;
                        _lastNodeStats.clear();
                        _graphBaselineDetached = true;
                        return _store.Get();
                    }
                }
            }
        }
        if (!_desc.preview.Active())
            _cudaPreviewWarned = false;
        else if (!std::exchange(_cudaPreviewWarned, true))
            _lastDiagnostics.Warn("usdGen:preview: " + _desc.description.GetString() +
                                  " runs on the CUDA lane, which cannot preview values; "
                                  "the look is shown");
        auto priorGeneration = _store.Get();
        auto device = ExecuteCudaGraph(*_graph.CudaPlan(), *_cudaWorkspace, frame,
            static_cast<uint64_t>(_store.NextId()), &_lastDiagnostics,
            priorGeneration && !newWorkspace ? priorGeneration->device : nullptr);
        if (!device || _lastDiagnostics.HasErrors()) {
            _InvalidatePoisonedCudaWorkspace();
            return reject();
        }
        std::atomic_store(&_cudaBindingStats,
            std::make_shared<const std::vector<UsdGenCudaBindingStats>>(
                GetCudaBindingStats(*_cudaWorkspace)));
        UsdGenGeneration gen;
        gen.frame = frame;
        gen.device = std::move(device);
        // `_desc` is the copied command descriptor that compiled the accepted
        // CUDA plan above.  Capture its presentation once with the device
        // payload; publishing must not consult a later graph descriptor.
        gen.devicePresentation = _BuildDevicePresentation(_desc);
        // Do not call the host scheduler or _BuildTilePublication for a
        // device generation. Tools retain this snapshot directly.
        auto prepared = _store.PreparePublication(std::move(gen));
        auto fence = cacheDomain->CapturePublicationFence(cacheKey);
        if (!fence || !cacheDomain->PublishIfCurrent(*fence, [this, &prepared] {
                return _store.PublishPrepared(std::move(prepared));
            })) return reject();
        _publicationFence = ExecutionPublicationFence{cacheDomain, *fence};
        _lastReport = UsdGenDirtyReport{};
        ++_stats.commits;
        _stats.publishedTiles = 0;
        _lastNodeStats.clear();
        _graphBaselineDetached = false;
        UsdGenGenerationConstPtr const next = _store.Get();
        size_t cacheBytes = 0;
        if (_GenerationBytes(*next, &cacheBytes))
            _cacheCandidate = ExecutionCacheCandidate{
                std::move(cacheKey), next, cacheBytes};
        return next;
    }

    // -- steps 2-5: reference lane, capture, evaluate, interleave (03 §5.4)
    UsdGenEvalContext evalCtx;
    evalCtx.time = frame;
    evalCtx.desc = &_desc;
    // Tile copies may start on terminal sweep workers. A previous cook's
    // preview colors must never be visible there; active preview is deferred
    // until its whole-output color pass below has completed.
    _previewColors = UsdGenPreviewColors{};
    std::mutex progressMutex;
    uint64_t progressSequence = 0;
    std::map<UsdGenTileId, UsdGenTilePublication> streamedTiles;
    auto emitProgress = [&](UsdGenTilePublication tile) {
        if (!progress || (superseded && superseded())) return;
        auto update = std::make_shared<UsdGenSession::TileProgress>();
        update->epoch = workEpoch;
        update->frame = frame;
        update->tile = std::move(tile);
        std::lock_guard<std::mutex> lock(progressMutex);
        if (superseded && superseded()) return;
        update->sequence = ++progressSequence;
        try { progress(std::move(update)); } catch (...) {
            // A provisional render consumer cannot invalidate the cook.
        }
    };
    auto tileCompleted = progress && !_desc.preview.Active()
        ? UsdGenScheduler::TileCompleted([&](UsdGenTileView const& tile,
                                               UsdGenCurveBuffer const& output) {
            if (superseded && superseded()) return;
            UsdGenRunResult partial;
            partial.terminalOutput = &output;
            // Streamed mid-sweep, before the run's writer set is known: never reuse.
            UsdGenTilePublication publication =
                _BuildTilePublication(tile, partial, {}, TileReuse{});
            {
                std::lock_guard<std::mutex> lock(progressMutex);
                // VtArray copies share their immutable storage. Reuse this
                // exact geometry in the final generation instead of gathering
                // the tile's points a second time.
                streamedTiles.emplace(tile.tile, publication);
            }
            emitProgress(std::move(publication));
        }) : UsdGenScheduler::TileCompleted{};
    UsdGenRunResult result = _scheduler.Run(_graph, evalCtx, myReq,
                                             std::move(tileCompleted));
    _lastDiagnostics = result.diagnostics;
    phases.Phase("run operators");

    if (result.diagnostics.HasErrors()) {
        // A partially executed graph is never a publishable generation.
        // Rebuild capture/evaluation state on retry; retain the last completed
        // generation for readers, tools and renderers.
        _graphUntrusted = true;
        return _store.Get();
    }

    if (result.superseded) {
        // Newer request: publish nothing, return the PREVIOUS generation,
        // The command owner retains its pending inputs for the retry.
        ++_stats.supersessions;
        _graphUntrusted = true;
        return _store.Get();
    }

    // -- step 6: build the immutable generation (03 §6.1) ------------------
    UsdGenGenerationConstPtr prev = _store.Get();
    UsdGenGeneration gen;
    gen.frame = frame;
    gen.tiles.reserve(result.tiles.size());

    // usdGen:preview recolours every tile, including those whose geometry
    // this cook did not touch.
    {
        std::vector<std::string> previewWarnings;
        _previewColors = _preview.Build(_desc, _graph, *result.terminalOutput, frame,
                                        &previewWarnings);
        for (std::string const &warning : previewWarnings) _lastDiagnostics.Warn(warning);
    }
    gen.colorDigest = _previewColors.digest;
    // ...and so does usdGen:look, which is baked into every tile's
    // displayColor and (since the look travels as constant primvars too)
    // into its extra planes. A look-only edit touches no point and no
    // topology, so without this it would reach the synthetic material and
    // nothing else -- the strands would keep the previous root colour.
    {
        auto mix = [](uint64_t *h, uint64_t v) {
            *h ^= v + 0x9e3779b97f4a7c15ull + (*h << 6) + (*h >> 2);
            *h ^= *h >> 30; *h *= 0xbf58476d1ce4e5b9ull;
            *h ^= *h >> 27; *h *= 0x94d049bb133111ebull;
            *h ^= *h >> 31;
        };
        auto mixF = [&](uint64_t *h, float v) {
            uint32_t bits = 0; std::memcpy(&bits, &v, sizeof(bits));
            mix(h, bits);
        };
        UsdGenLookDesc const &lk = _desc.look;
        uint64_t h = gen.colorDigest;
        for (int i = 0; i < 3; ++i) { mixF(&h, lk.rootColor[i]); mixF(&h, lk.tipColor[i]); }
        for (GfVec3f const &c : lk.rampColors)
            for (int i = 0; i < 3; ++i) mixF(&h, c[i]);
        for (float p : lk.rampPositions) mixF(&h, p);
        mixF(&h, lk.rampExponent);
        mixF(&h, lk.hueJitter);
        mixF(&h, lk.valueJitter);
        mix(&h, static_cast<uint64_t>(static_cast<uint32_t>(lk.jitterSeed)));
        mix(&h, lk.rampInterpolation.Hash());
        mix(&h, lk.bakeMode.Hash());
        mix(&h, lk.bakeTarget.Hash());
        mix(&h, lk.bakePrimvar.Hash());
        gen.colorDigest = h;
    }
    const bool recolour = prev && prev->colorDigest != gen.colorDigest;
    phases.Phase("preview");

    { TRACE_SCOPE("usdGen build tile publications");
    // Tile builds are independent: each gathers its own publication from the
    // immutable terminal output (the progress path already builds them on
    // sweep workers), so fan them out over the scheduler pool — idle since
    // Run returned — and collect in tile order. The generation, the progress
    // sequence, and the rebuilt count match the serial loop exactly; only
    // best-effort progress emission under a mid-build supersession flip can
    // differ (fewer provisional updates, same final generation).
    size_t rebuiltTiles = 0;
    size_t const nTiles = result.tiles.size();
    // Per-array reuse: a rebuilt tile still re-gathers its points, but every
    // other array is carried from the previous tile when the run wrote only
    // point planes (deforms alone) and the partition, baseline, and colors
    // are unchanged. The carried VtArrays share buffers, so step 7 and the
    // drain diff see IsIdentical == true instead of memcmp'ing them.
    bool const arraysStable = prev && !result.topologyChanged &&
        !_rebuildAllTiles && _OnlyPointsDeformsRan(_graph, result);
    bool const colorsStable = arraysStable && !recolour;
    std::vector<UsdGenTilePublication const*> reuse(nTiles, nullptr);
    std::vector<char> carried(nTiles, 0);
    std::vector<char> emit(nTiles, 0);
    std::vector<TileReuse> tileReuse(nTiles);
    std::vector<size_t> buildIndex;
    buildIndex.reserve(nTiles);
    for (size_t i = 0; i < nTiles; ++i) {
        UsdGenTileView const &tv = result.tiles[i];
        // E-4: untouched tiles carry over their publication wholesale (the
        // VtArray copies share buffers, so step 7 sees IsIdentical == true).
        const bool rebuild =
            result.topologyChanged || tv.pointsDirty || tv.widthsDirty || recolour ||
            _rebuildAllTiles;
        const UsdGenTilePublication *carry = nullptr;
        const UsdGenTilePublication *prevTile = nullptr;
        if (prev) {
            auto it = std::lower_bound(
                prev->tiles.begin(), prev->tiles.end(), tv.tile,
                [](UsdGenTilePublication const &p, UsdGenTileId t) {
                    return p.tile < t;
                });
            if (it != prev->tiles.end() && it->tile == tv.tile) prevTile = &*it;
        }
        if (!rebuild) carry = prevTile;
        if (carry) {
            reuse[i] = carry;
            carried[i] = 1;
            continue;
        }
        if (arraysStable) {
            tileReuse[i].prevTile = prevTile;
            tileReuse[i].arraysStable = true;
            tileReuse[i].colorsStable = colorsStable;
        }
        auto streamed = streamedTiles.find(tv.tile);
        if (streamed != streamedTiles.end()) {
            reuse[i] = &streamed->second;
            continue;
        }
        emit[i] = progress ? 1 : 0;
        buildIndex.push_back(i);
    }
    std::vector<UsdGenTilePublication> built(buildIndex.size());
    if (buildIndex.size() > 1) {
        TileBuildWork work;
        work.cooker = this;
        work.result = &result;
        work.prev = &prev;
        work.tileIndex = buildIndex.data();
        work.built = built.data();
        work.reuse = tileReuse.data();
        _scheduler.MakeWorkDispatcher().ParallelFor(
            buildIndex.size(), &_BuildTileWork, &work);
    } else if (buildIndex.size() == 1) {
        built[0] = _BuildTilePublication(
            result.tiles[buildIndex[0]], result, prev,
            tileReuse[buildIndex[0]]);
    }
    for (size_t i = 0, b = 0; i < nTiles; ++i) {
        if (reuse[i]) {
            // A streamed tile counts as rebuilt, as in the serial loop; a
            // wholesale carry does not. Neither re-emits progress (streamed
            // tiles were already emitted by the sweep).
            if (!carried[i]) ++rebuiltTiles;
            gen.tiles.push_back(*reuse[i]);
            continue;
        }
        ++rebuiltTiles;
        gen.tiles.push_back(std::move(built[b++]));
        if (emit[i])
            emitProgress(gen.tiles.back());
    }
    std::sort(gen.tiles.begin(), gen.tiles.end(),
              [](UsdGenTilePublication const &a, UsdGenTilePublication const &b) {
                  return a.tile < b.tile;
              });
    phases.Phase("tiles", rebuiltTiles, gen.tiles.size());
    }

    // Whole-groom optical depth includes neighboring tiles. Unchanged
    // geometry reuses the immutable planes, including on look-only edits.
    {
        UsdGenWorkDispatcher dispatcher = _scheduler.MakeWorkDispatcher();
        UsdGenFurOcclusionParams occlusion;
        occlusion.dispatcher = &dispatcher;
        // The emitting surfaces are opaque: without them light reaches hair
        // through the scalp (a host renderer injects the opaque depth for the same
        // reason). Deformed points at this frame, in the same world space the
        // tiles were transformed into.
        // A GeomSubset surface carries its whole parent mesh, which occludes
        // whole and once, however many subsets of it the groom binds.
        occlusion.occluders.reserve(_desc.surfaces.size());
        std::set<SdfPath> occludingMeshes;
        for (UsdGenSurfaceDesc const &surface : _desc.surfaces) {
            if (surface.points.empty() || surface.faceVertexIndices.empty())
                continue;
            if (!occludingMeshes.insert(UsdGenSurfaceMeshPath(surface)).second)
                continue;
            UsdGenFurOccluder occluder;
            occluder.points = surface.points;
            occluder.faceVertexCounts = surface.faceVertexCounts;
            occluder.faceVertexIndices = surface.faceVertexIndices;
            occluder.worldMatrix = surface.worldMatrix;
            occlusion.occluders.push_back(std::move(occluder));
        }
        // Hair shadowing the scalp. Stock Storm casts no shadow from any
        // UsdLux light, so a synthetic cap over the emitting surface is the
        // only route; it is built from the same volume, at no extra sweep
        // beyond the hair-only one it needs.
        occlusion.scalpShadow = !occlusion.occluders.empty() &&
                                TfGetEnvSetting(USDGEN_SCALP_SHADOW);
        // The reuse path leaves the cap untouched, so seed it: unchanged
        // geometry shares the previous immutable arrays.
        if (prev) gen.scalpShadow = prev->scalpShadow;
        occlusion.occluderCache = &_furOccluderBuild;
        occlusion.scalpScratch = &_furScalpScratch;
        occlusion.bakeScratch = &_furBakeScratch;
        UsdGenBuildFurOcclusion(&gen.tiles, prev ? &prev->tiles : nullptr,
                                occlusion, &_furVolumeKey, &gen.scalpShadow);
        if (!gen.scalpShadow.IsEmpty()) {
            _PresentationScalars const scalars = _BuildPresentationScalars(
                _desc, SdfPath());
            gen.scalpShadow.primPath =
                _RenderNamespace(_desc.description).AppendChild(
                    TfToken("scalpShadow"));
            gen.scalpShadow.purpose = scalars.purpose;
            gen.scalpShadow.visibility = scalars.visibility;
            gen.scalpShadow.materialPath =
                _RenderNamespace(_desc.description).AppendChild(
                    TfToken("material_scalpShadow"));
        }
    }
    phases.Phase("occlusion");

    // Signature: the prim-set identity step 7 diffs structurally (03 §6.1).
    gen.signature.tileCount = static_cast<uint32_t>(gen.tiles.size());
    gen.signature.instancerCount = 0;
    // The end of the look's path into Storm: the colour a tile actually
    // publishes. It is baked from _desc.look, so this and the INGRESS capture
    // line together say whether an authored look survived the whole trip --
    // the failure mode is silent, because a defaulted look is still a
    // plausible dark brown.
    if (TfDebug::IsEnabled(USDGEN_INGRESS) && !gen.tiles.empty() &&
        !gen.tiles.front().displayColor.empty()) {
        GfVec3f const &c = gen.tiles.front().displayColor.front();
        TF_DEBUG(USDGEN_INGRESS).Msg(
            "usdGen tile displayColor[0] = %.4f %.4f %.4f (look rootColor "
            "%.4f %.4f %.4f)\n",
            double(c[0]), double(c[1]), double(c[2]),
            double(_desc.look.rootColor[0]), double(_desc.look.rootColor[1]),
            double(_desc.look.rootColor[2]));
    }
    gen.signature.primPaths.reserve(gen.tiles.size());
    gen.signature.primTypes.assign(gen.tiles.size(), "basisCurves");
    gen.signature.primvarNames.reserve(gen.tiles.size());
    for (UsdGenTilePublication const &t : gen.tiles) {
        gen.signature.primPaths.push_back(t.primPath.GetString());
        std::vector<std::string> names{"points", "widths", "hairT", "hairId"};
        if (!t.st.empty())            names.emplace_back("st");
        if (!t.displayColor.empty())  names.emplace_back("displayColor");
        if (!t.bakeColor.empty())     names.emplace_back("bakeColor");
        if (!t.velocities.empty())    names.emplace_back("velocities");
        for (UsdGenPlane const &p : t.extraUniform)
            names.push_back(p.name.GetString());
        std::sort(names.begin(), names.end());
        names.erase(std::unique(names.begin(), names.end()), names.end());
        gen.signature.primvarNames.push_back(std::move(names));
    }

    auto prepared = _store.PreparePublication(std::move(gen));
    auto fence = cacheDomain->CapturePublicationFence(cacheKey);
    if (!fence || !cacheDomain->PublishIfCurrent(*fence, [this, &prepared] {
            return _store.PublishPrepared(std::move(prepared));
        })) return _store.Get();
    _publicationFence = ExecutionPublicationFence{cacheDomain, *fence};
    _graphBaselineDetached = false;
    phases.Phase("publish");

    // -- step 7: diff vs the previous generation (06 §5.1) -----------------
    UsdGenGenerationConstPtr next = _store.Get();
    _lastReport = _store.Diff(prev ? *prev : UsdGenGeneration{}, *next);
    phases.Phase("diff");

    // -- stats (03 §9.2) ----------------------------------------------------
    ++_stats.commits;
    _stats.publishedTiles = static_cast<uint64_t>(next->tiles.size());
    _lastNodeStats.clear();
    double captureMs = 0.0, evalMs = 0.0;
    for (UsdGenNodeRunStats const &rs : result.nodeStats) {
        _lastNodeStats[rs.id] = rs;
        captureMs += rs.captureMs;
        evalMs += rs.evalMs;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> dt = t1 - t0;
    UsdGenCommitTiming &slot = _stats.ring[_stats.ringHead];
    slot = UsdGenCommitTiming{};
    slot.generation = static_cast<uint64_t>(next->id);
    slot.captureMs = captureMs;
    slot.evaluateMs = evalMs;
    slot.totalMs = dt.count();
    _stats.ringHead = (_stats.ringHead + 1) % _stats.ring.size();

    size_t cacheBytes = 0;
    if (_GenerationBytes(*next, &cacheBytes)) {
        _cacheCandidate = ExecutionCacheCandidate{
            std::move(cacheKey), next, cacheBytes};
    }

    phases.Phase("cache bytes");
    phases.Print(frame, _traceCompile, "cache miss, cooked", _desc.description);

    // Remaining chunk dirt (skipped no-op nodes are clean; anything the
    // scheduler could not run stays dirty; 03 §5.4).
    return next;
}

/*static*/
void UsdGenSessionCooker::_BuildTileWork(size_t slot, void *payload)
{
    TileBuildWork *w = static_cast<TileBuildWork *>(payload);
    w->built[slot] = w->cooker->_BuildTilePublication(
        w->result->tiles[w->tileIndex[slot]], *w->result, *w->prev,
        w->reuse[w->tileIndex[slot]]);
}

UsdGenTilePublication UsdGenSessionCooker::_BuildTilePublication(
    UsdGenTileView const &tv, UsdGenRunResult const &result,
    UsdGenGenerationConstPtr const &prev, TileReuse reuse)
{
    TF_UNUSED(prev);
    UsdGenTilePublication pub;
    pub.tile = tv.tile;
    pub.primPath = _TilePath(_desc.description, tv.tile);
    if (!_desc.curveBasis.IsEmpty())
        pub.basis = _desc.curveBasis.GetString();
    pub.refineLevel = 2;

    UsdGenCurveBuffer const &term = *result.terminalOutput;
    UsdGenCompiledNode const &tn = _graph.Node(_graph.TerminalNodeId());

    // Per-array reuse resolves here; every flag below fails closed into a
    // fresh gather. The previous arrays are gathered from byte-identical
    // inputs (stable planes, stable mapping), so carrying their storage is
    // bit-identical; the size checks additionally refuse any tile whose
    // previous shape does not match this tile's mapping exactly.
    UsdGenTilePublication const *rp =
        (reuse.arraysStable && reuse.prevTile) ? reuse.prevTile : nullptr;
    size_t expectCurves = 0;
    if (rp) {
        if (size_t(tv.firstChunk) + size_t(tv.chunkCount) <= tn.chunks.size()) {
            for (uint32_t i = 0; i < tv.chunkCount; ++i)
                expectCurves += tn.chunks[tv.firstChunk + i].liveCount;
        } else {
            rp = nullptr;
        }
    }
    bool const reuseCounts = rp &&
        rp->curveVertexCounts.size() == expectCurves;
    bool const reuseWidths = rp &&
        rp->widths.size() == tv.totalLiveCvs;
    bool const reuseHairT = rp &&
        rp->hairT.size() == tv.totalLiveCvs;
    bool const reuseHairId = rp &&
        rp->hairId.size() == expectCurves;
    bool const reuseSt = rp &&
        rp->st.size() == expectCurves;

    // Exact emission bound, walked locally: the tile view's counts cannot
    // size the raw stores below. Interleave fills graph tile views in
    // place AFTER the sweep streams them, so a streamed tile view carries
    // zero counts (reserve(0) leaves data() null and the first store
    // segfaults), and an untouched tile keeps stale counts from an older
    // cook. The walk mirrors the emission loop's len rule exactly (same
    // branches, minus the px-size break which only emits fewer), so the
    // emitted CVs never exceed emitCap; uniform chunks cost O(1) each.
    size_t emitCap = 0;
    for (uint32_t i = 0; i < tv.chunkCount; ++i) {
        UsdGenChunkDesc const& cd = tn.chunks[tv.firstChunk + i];
        if (cd.cvCount != 0) {
            emitCap += uint64_t(cd.liveCount) * cd.cvCount;
        } else {
            for (uint32_t c = 0; c < cd.liveCount; ++c) {
                size_t const g = size_t(cd.firstCurve) + c;
                if (g + 1 >= term.cvOffsets.size()) break;
                emitCap += size_t(term.cvOffsets[g + 1]) - size_t(term.cvOffsets[g]);
            }
        }
    }
    pub.points.reserve(emitCap);
    if (!reuseCounts) pub.curveVertexCounts.reserve(tv.totalLiveCurves);
    if (!reuseWidths) pub.widths.reserve(emitCap);
    if (!reuseHairT) pub.hairT.reserve(emitCap);
    if (!reuseHairId) pub.hairId.reserve(tv.totalLiveCurves);
    // Raw-pointer emission (cf. the k==1 cap path): the per-CV push_backs'
    // capacity checks and size stores cost more than the ~7 stores they
    // frame. reserve() owns the capacity; data() detaches when shared, so
    // the pointers are uniquely owned; nothing between here and the
    // closing resizes can reallocate. Same values in the same order:
    // bit-identical.
    GfVec3f* emitPts = pub.points.data();
    float* emitW = !reuseWidths ? pub.widths.data() : nullptr;
    float* emitH = !reuseHairT ? pub.hairT.data() : nullptr;
    size_t nEmit = 0;
    // The published extent reduces during emission (same ExtendBy calls in
    // the same order as the retired scan over pub.points), so no second
    // pass over the points is needed below.
    GfRange3f extent;

    // Loop-invariant token spellings: every TfToken(const char*)
    // construction takes the registry lock, so the per-curve/per-CV loop
    // below must never build one (1M curves x lock = tens of ms).
    static TfToken const displayColorName("displayColor");
    static TfToken const constantInterp("constant");
    static TfToken const uniformInterp("uniform");
    static TfToken const vertexInterp("vertex");
    static TfToken const intType("int");
    static TfToken const floatType("float");
    static TfToken const noBake("none");
    UsdGenPlane const *displayColor = _FindPlane(term.extraCurve, displayColorName);
    static TfToken const furTauP("furTauP");
    static TfToken const furTauN("furTauN");
    struct ExtraPlanePublication {
        UsdGenPlane const* source = nullptr;
        UsdGenPlane output;
        // Resolved once at setup: the gather loops below test these, never
        // the spelling, so a many-extras tile pays no token compare per CV.
        bool isVertex = false;
        bool isUniform = false;
        // Previous tile's matching plane, when its payload is carried
        // instead of re-gathered (same layout, full-size payload).
        UsdGenPlane const* reuseFrom = nullptr;
    };
    std::vector<ExtraPlanePublication> extraPlanes;
    // The forwarded source colour is consumed into the tile's displayColor
    // below, never published: it is an input to the colour decision, not a
    // primvar of its own.
    UsdGenPlane const *sourceColor = _FindPlane(term.extraCurve,
                                                UsdGenSourceColorPlane());
    if (!sourceColor) sourceColor = _FindPlane(term.extraCv,
                                               UsdGenSourceColorPlane());
    auto addExtraPlane = [&](UsdGenPlane const& plane, TfToken expectedInterpolation) {
        if (plane.name == displayColorName ||
            plane.name == UsdGenSourceColorPlane()) return;
        size_t const elements = expectedInterpolation == constantInterp ? 1 :
            expectedInterpolation == vertexInterp ? term.totalCvs : term.totalCurves;
        size_t const values = elements * plane.arity;
        bool const validPayload = plane.type == intType
            ? plane.i.size() == values && plane.f.empty()
            : plane.type == floatType && plane.f.size() == values && plane.i.empty();
        if (plane.name.IsEmpty() || plane.arity == 0 ||
            plane.arity > kUsdGenMaxExtraPlaneSlots ||
            plane.interpolation != expectedInterpolation || !validPayload)
            throw std::runtime_error("invalid extra-plane publication layout for '" +
                                     plane.name.GetString() + "'");
        if (std::any_of(extraPlanes.begin(), extraPlanes.end(), [&](auto const& entry) {
                return entry.output.name == plane.name;
            }))
            throw std::runtime_error("duplicate extra-plane publication name '" +
                                     plane.name.GetString() + "'");
        ExtraPlanePublication entry;
        entry.source = &plane;
        entry.output.name = plane.name;
        entry.output.interpolation = plane.interpolation;
        entry.output.type = plane.type;
        entry.output.arity = plane.arity;
        entry.isVertex = plane.interpolation == vertexInterp;
        entry.isUniform = plane.interpolation == uniformInterp;
        if (plane.interpolation == constantInterp)
            _GatherPlaneElement(plane, 0, &entry.output);
        // The occlusion planes are appended fresh after the build, never
        // carried; constants re-synthesize above for a few floats. Anything
        // else with a same-layout, full-size previous payload is carried.
        if (rp && entry.source && plane.name != furTauP && plane.name != furTauN &&
            (entry.isVertex || entry.isUniform)) {
            size_t const expectElements = entry.isVertex
                ? size_t(tv.totalLiveCvs) : expectCurves;
            for (UsdGenPlane const &candidate : rp->extraUniform) {
                if (candidate.name != plane.name ||
                    candidate.interpolation != plane.interpolation ||
                    candidate.type != plane.type ||
                    candidate.arity != plane.arity)
                    continue;
                size_t const payload = plane.type == intType
                    ? candidate.i.size() : candidate.f.size();
                bool const otherEmpty = plane.type == intType
                    ? candidate.f.empty() : candidate.i.empty();
                if (otherEmpty && payload == expectElements * plane.arity)
                    entry.reuseFrom = &candidate;
                break;
            }
        }
        extraPlanes.push_back(std::move(entry));
    };
    for (UsdGenPlane const &p : term.extraCurve) {
        if (p.interpolation == constantInterp)
            addExtraPlane(p, constantInterp);
        else if (p.interpolation == uniformInterp)
            addExtraPlane(p, uniformInterp);
        else if (p.name != displayColorName)
            throw std::runtime_error("invalid per-curve extra-plane interpolation for '" +
                                     p.name.GetString() + "'");
    }
    for (UsdGenPlane const& p : term.extraCv)
        addExtraPlane(p, vertexInterp);

    // Precedence (02 §2.14): an AUTHORED look on the description wins;
    // otherwise the source curves' own displayColor, which is what a
    // character asset already carries; otherwise the look's schema
    // defaults. "Authored" has to mean an opinion that differs from the
    // fallback -- a description that merely applies UsdGenLookAPI has a full
    // set of schema-default look values, and treating those as authored would
    // silently discard every asset's own colour.
    static UsdGenLookDesc const lookFallback;
    bool const authoredLook =
        _desc.look.rootColor != lookFallback.rootColor ||
        _desc.look.tipColor != lookFallback.tipColor ||
        !_desc.look.rampColors.empty();

    // The rest of the look as CONSTANT primvars, so the shading of a groom
    // does not depend on the tile keeping the synthetic material: a user who
    // binds their own UsdGenHairStrands material gets the Sdr defaults for
    // everything the tile does not carry, which used to mean blond tips on a
    // black-haired groom. rootColor already travels as the baked per-curve
    // displayColor, so it is deliberately NOT repeated here -- a second
    // channel for it would only be a way for the two to disagree.
    //
    // Each one is published only where the look RESOLVES it to something
    // other than its schema fallback. An unauthored look publishes none of
    // them, the shader's `#ifdef HD_HAS_*` is false, and the material input
    // applies exactly as before: the default-material path is unchanged by
    // construction, and a forwarded asset colour (one flat colour per curve)
    // never acquires an invented tip.
    auto addLookConstant = [&](char const* name, std::initializer_list<float> v) {
        // An authored plane of the same name wins: the user asked for it by
        // hand, and two planes with one name is not a publishable tile.
        if (std::any_of(extraPlanes.begin(), extraPlanes.end(),
                        [&](auto const& e) { return e.output.name == name; }))
            return;
        UsdGenPlane plane;
        plane.name = TfToken(name);
        plane.interpolation = constantInterp;
        plane.type = floatType;
        plane.arity = static_cast<decltype(plane.arity)>(v.size());
        plane.f.assign(v.begin(), v.end());
        ExtraPlanePublication entry;
        entry.source = nullptr;   // synthesized; the uniform gather never runs
        entry.output = std::move(plane);
        // Constant interpolation: both gathers stay off (bools default false).
        extraPlanes.push_back(std::move(entry));
    };
    if (authoredLook) {
        // Ramp-aware, matching UsdGenTilePublisher::_LookColors: the shader
        // has one lerp, so a multi-stop ramp presents as its last stop.
        GfVec3f const tip = _desc.look.rampColors.size() >= 2
            ? _desc.look.rampColors[_desc.look.rampColors.size() - 1]
            : _desc.look.tipColor;
        addLookConstant("hairTipColor", {tip[0], tip[1], tip[2]});
    }
    if (_desc.look.rampExponent != lookFallback.rampExponent)
        addLookConstant("hairColorRamp",
                        {std::max(_desc.look.rampExponent, 0.001f)});
    if (_desc.look.hueJitter != lookFallback.hueJitter)
        addLookConstant("hairRandomHue", {std::max(_desc.look.hueJitter, 0.0f)});
    if (_desc.look.valueJitter != lookFallback.valueJitter)
        addLookConstant("hairRandomValue",
                        {std::max(_desc.look.valueJitter, 0.0f)});

    GfRange3f bounds;
    uint32_t depSurface = 0;
    bool hasDep = false;
    bool firstChunk = true;

    // DisplayColor pushes exactly one value per gathered curve on the
    // preview-uniform and constant-look paths (mirroring the loop's branch
    // conditions below); the plane and preview-per-CV paths re-gather.
    bool const previewUniformColors =
        _previewColors.active && !_previewColors.perCv;
    bool const constantLookColors = !_previewColors.active && !displayColor &&
        _desc.look.bakeTarget != noBake && (authoredLook || !sourceColor);
    bool const reuseDisplayColor = rp && reuse.colorsStable &&
        (previewUniformColors || constantLookColors) &&
        rp->displayColor.size() == expectCurves;

    for (uint32_t i = 0; i < tv.chunkCount; ++i) {
        UsdGenChunkDesc const &cd = tn.chunks[tv.firstChunk + i];
        // id 0 is a legitimate surface (compiler.cpp assigns dense indices
        // from 0): the first chunk of the tile always establishes the
        // dependency, never a `!= 0` sentinel.
        if (firstChunk) { depSurface = cd.surface; hasDep = true; firstChunk = false; }
        for (uint32_t c = 0; c < cd.liveCount; ++c) {
            const uint32_t g = cd.firstCurve + c;
            uint32_t p0 = 0, len = 0;
            if (cd.cvCount != 0) {
                len = cd.cvCount;
                p0 = cd.firstCv + c * cd.cvCount;
            } else if (g + 1 < term.cvOffsets.size()) {   // ragged chunk
                p0 = static_cast<uint32_t>(term.cvOffsets[g]);
                len = static_cast<uint32_t>(term.cvOffsets[g + 1]) - p0;
            }
            if (!reuseCounts)
                pub.curveVertexCounts.push_back(static_cast<int>(len));
            for (uint32_t v = 0; v < len; ++v) {
                const uint32_t p = p0 + v;
                if (p >= term.px.size()) break;
                emitPts[nEmit] = GfVec3f(term.px[p], term.py[p], term.pz[p]);
                extent.ExtendBy(emitPts[nEmit]);
                // C2 (06 §4.1): widths/hairT are vertex channels on every
                // tile. A chain whose terminal never wrote them (grow-only:
                // kPlanePoints|kPlaneHairT) still publishes FULL-SIZE planes
                // from desc defaults (02 §2.6/§2.14: width 0.01, look bake),
                // never a wrong-size array — SI-1 sizes every non-empty
                // vertex plane against points.
                if (!reuseWidths) {
                    float w = 0.01f;
                    if (p < term.width.size()) w = term.width[p];
                    else if (!term.width.empty()) w = term.width.back();
                    emitW[nEmit] = w;
                }
                if (!reuseHairT)
                    emitH[nEmit] = p < term.hairT.size() ? term.hairT[p] : 0.0f;
                ++nEmit;
                for (auto& extra : extraPlanes)
                    if (extra.isVertex && !extra.reuseFrom)
                        _GatherPlaneElement(*extra.source, p, &extra.output);
            }
            // hairId: UsdGenHash32(curveId, 0) / 2^32 in [0,1) (06, S29).
            if (!reuseHairId)
                pub.hairId.push_back(g < term.curveId.size()
                    ? UsdGenHairId(term.curveId[g]) : 0.0f);
            if (!reuseSt && !term.rootUV.empty() && g < term.rootUV.size())
                pub.st.push_back(term.rootUV[g]);
            if (!reuseDisplayColor) {
            if (_previewColors.active) {
                std::vector<GfVec3f> const &colors = _previewColors.colors;
                if (_previewColors.perCv) {
                    for (uint32_t v = 0; v < len; ++v)
                        pub.displayColor.push_back(size_t(p0) + v < colors.size()
                            ? colors[size_t(p0) + v] : UsdGenPreviewMissingColor());
                } else {
                    pub.displayColor.push_back(g < colors.size()
                        ? colors[g] : UsdGenPreviewMissingColor());
                }
            } else if (displayColor)
                _GatherColor(*displayColor, g, p0, &pub.displayColor);
            else if (_desc.look.bakeTarget != noBake) {
                // `authoredLook` above carries the 02 §2.14 precedence rule.
                if (!authoredLook && sourceColor)
                    _GatherColor(*sourceColor, g, p0, &pub.displayColor);
                else
                    pub.displayColor.push_back(_desc.look.rootColor);
            }
            }
            for (auto& extra : extraPlanes)
                if (extra.isUniform && !extra.reuseFrom)
                    _GatherPlaneElement(*extra.source, g, &extra.output);
        }
    }
    // Publish the raw-pointer emission above: emitCap was walked from
    // the same len rule the loop uses (minus the px-size break, which
    // only emits fewer), so nEmit <= emitCap always; the resizes only set
    // sizes over the already-written elements.
    {
        auto noInit = [](auto* b, auto* e) { (void)b; (void)e; };
        pub.points.resize(nEmit, noInit);
        if (!reuseWidths) pub.widths.resize(nEmit, noInit);
        if (!reuseHairT) pub.hairT.resize(nEmit, noInit);
    }
    // Carried arrays share the previous tile's immutable storage (VtArray
    // copy-on-write), so downstream identity checks see them as unchanged.
    if (reuseCounts) pub.curveVertexCounts = rp->curveVertexCounts;
    if (reuseWidths) pub.widths = rp->widths;
    if (reuseHairT) pub.hairT = rp->hairT;
    if (reuseHairId) pub.hairId = rp->hairId;
    if (reuseSt) pub.st = rp->st;
    if (reuseDisplayColor) pub.displayColor = rp->displayColor;
    for (auto& extra : extraPlanes) {
        if (!extra.reuseFrom) continue;
        if (extra.reuseFrom->type == intType)
            extra.output.i = extra.reuseFrom->i;
        else
            extra.output.f = extra.reuseFrom->f;
    }
    for (auto& extra : extraPlanes)
        pub.extraUniform.push_back(std::move(extra.output));
    std::sort(pub.extraUniform.begin(), pub.extraUniform.end(),
              [](UsdGenPlane const& left, UsdGenPlane const& right) {
                  return left.name < right.name;
              });

    // Extent is a pure function of the published points (03 §6.3: min/max
    // fused into the interleave loop — but InterleaveTile skips untouched
    // tiles, so tv.extent is empty on any tile this commit did not evaluate
    // while its points are real). It reduced during emission above (the
    // same ExtendBy calls in the same order as a scan over pub.points),
    // so only the empty-tiles fallback remains here.
    if (extent.IsEmpty()) extent = GfRange3f(GfVec3f(0.f), GfVec3f(0.f));
    pub.extentMin = GfVec3d(extent.GetMin()[0], extent.GetMin()[1], extent.GetMin()[2]);
    pub.extentMax = GfVec3d(extent.GetMax()[0], extent.GetMax()[1], extent.GetMax()[2]);

    SdfPath dependencySurface;
    if (hasDep) {
        for (UsdGenSurfaceDesc const &s : _desc.surfaces)
            if (s.id == depSurface) { dependencySurface = s.path; break; }
    }
    _PresentationScalars const scalars = _BuildPresentationScalars(
        _desc, dependencySurface);
    pub.xformMatrix = scalars.xformMatrix;
    pub.purpose = scalars.purpose;
    pub.visibility = scalars.visibility;
    pub.materialPath = scalars.materialPath;
    pub.materialPurpose = scalars.materialPurpose;
    if (_previewColors.active) {
        // The look's shader blends toward the tip colour; the preview shows
        // exactly the published displayColor, over any authored material.
        static TfToken const allPurpose("allPurpose");
        pub.materialPath = UsdGenPreviewMaterialPath(_desc.description, _desc.preview.shading);
        pub.materialPurpose = allPurpose;
    }
    pub.refineLevel = scalars.refineLevel;
    pub.primOrigin = scalars.primOrigin;
    pub.dependencySurface = scalars.dependencySurface;
    return pub;
}

std::unordered_map<UsdGenNodeId, UsdGenNodeStats> UsdGenSessionCooker::NodeStats() const
{
    std::unordered_map<UsdGenNodeId, UsdGenNodeStats> out;
    for (auto const& entry : _lastNodeStats) {
        if (entry.first >= static_cast<UsdGenNodeId>(_graph.NodeCount())) continue;
        UsdGenCompiledNode const& n = _graph.Node(entry.first);
        UsdGenNodeStats& s = out[entry.first];
        s.type = n.type; s.path = n.desc ? n.desc->path : SdfPath();
        s.captureMs = entry.second.captureMs; s.lastEvalMs = entry.second.evalMs;
        s.meanEvalMs = entry.second.evalMs; s.curvesOut = n.buffer.totalCurves;
        for (uint8_t bit : n.chunkDirty) if (bit) ++s.chunksDirty;
        s.chunksTotal = n.chunks.size();
    }
    return out;
}

std::shared_ptr<const UsdGenGraphRoutingSnapshot> UsdGenSessionCooker::Routing() const
{ return _graph.RoutingSnapshot(); }
UsdGenGraphDesc UsdGenSessionCooker::GraphDesc() const
{ return _graph.NodeCount() ? _graph.Desc() : _desc; }
std::shared_ptr<const UsdGenCudaExecutionPlan> UsdGenSessionCooker::CudaPlan() const
{ return _graph.CudaPlan(); }

}  // namespace usdGen
