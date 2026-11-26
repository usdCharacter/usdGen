// usdGen engine — session implementation (03 §5.4/§5.6/§6.1/§9.1).
//
// Private serial work-owner implementation. The command owner supplies a
// copied request and alone decides whether this candidate may publish.
#include "usdGen/sessionCooker.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/token.h"
#include "usdGen/debugCodes.h"
#include "usdGen/cudaExecution.h"
#include "usdGen/digest.h"
#include "usdGen/executionBackend.h"
#include "usdGen/executionTaskGraph.h"
#include "usdGen/taskScheduler.h"


#include "pxr/pxr.h"
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
#include <optional>
#include <string>
#include <string_view>
#include <utility>

PXR_NAMESPACE_USING_DIRECTIVE
namespace usdGen {

namespace {

// A streamed tile past the cook's slot count.
struct _StreamedTile
{
    _StreamedTile *next = nullptr;
    UsdGenTileId tile{};
    UsdGenRef<const UsdGenPublication> publication;
};

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
               UsdGenGraphSpec const &spec) const
    {
        if (!_enabled) return;
        double const total = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - _start).count();
        TfDebug::Helper().Msg("usdGen commit    %s frame %g: %s; %s; total %.2f ms [ms: %s]\n",
                              spec.DescriptionText(), frame, outcome,
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
        if (use.resource != UsdGenExecutionDataKind::ElementTopology) continue;
        if (found || use.access != UsdGenExecutionResourceAccess::Read ||
            use.inputValue >= metadata.Values().size()) return false;
        auto const& value = metadata.Values()[use.inputValue];
        if (value.resource != UsdGenExecutionDataKind::ElementTopology ||
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

void _CacheMix(uint64_t *hash, uint64_t value)
{
    for (unsigned shift = 0; shift != 64; shift += 8) {
        *hash ^= (value >> shift) & 0xffu;
        *hash *= 0x100000001b3ULL;
    }
}

void _CacheMixText(uint64_t *hash, std::string_view text)
{
    _CacheMix(hash, static_cast<uint64_t>(text.size()));
    for (unsigned char byte : text) {
        *hash ^= byte;
        *hash *= 0x100000001b3ULL;
    }
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
    // Threaded over jobs for big keys (on the process scheduler: the key
    // owns no scheduler context here); chunks cap at 8. Small keys keep the
    // serial driver, computing the identical function.
    int const keyWorkers = UsdGenParallelForConcurrency();
    size_t const keyChunks =
        (keyWorkers > 1 && terms.totalBytes > 1048576)
            ? std::min({size_t(keyWorkers), size_t(8), jobs.size()})
            : 1;
    if (keyChunks > 1) {
        size_t const n = jobs.size();
        _CacheDigestJob const *jobData = jobs.data();
        struct Payload { _CacheDigestJob const *jobs; size_t n, chunks; }
            payload{jobData, n, keyChunks};
        UsdGenParallelFor(keyChunks, [](size_t c, void *p) {
            auto const &pl = *static_cast<Payload const *>(p);
            size_t const i0 = (c * pl.n) / pl.chunks;
            size_t const i1 = ((c + 1) * pl.n) / pl.chunks;
            for (size_t i = i0; i < i1; ++i)
                *pl.jobs[i].out = UsdGenDigestBytes(
                    pl.jobs[i].data, pl.jobs[i].bytes, pl.jobs[i].seed);
        }, &payload);
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

void _CacheMixKeys(uint64_t *hash, std::vector<UsdGenNodeKey> const &keys)
{
    _CacheMix(hash, static_cast<uint64_t>(keys.size()));
    for (UsdGenNodeKey const key : keys) _CacheMix(hash, key.value);
}

// An input kind's digest hook writes into the plan digest's lanes.
class _CacheInputDigest final : public UsdGenInputDigest
{
public:
    _CacheInputDigest(uint64_t *h0, uint64_t *h1, _CacheArrayTerms *terms)
        : _lanes{h0, h1}, _terms(terms) {}
    void Mix(int lane, uint64_t value) override { _CacheMix(_Lane(lane), value); }
    void MixText(int lane, std::string_view text) override { _CacheMixText(_Lane(lane), text); }
    void MixArray(int lane, void const *data, size_t count, size_t elementBytes) override
    {
        uint64_t *const hash = _Lane(lane);
        _CacheMix(hash, static_cast<uint64_t>(count));
        if (count != 0 && _terms) {
            size_t const bytes = count * elementBytes;
            _terms->terms.push_back(_CacheArrayTerm{hash, data, bytes});
            _terms->totalBytes += bytes;
        }
    }

private:
    uint64_t *_Lane(int lane) const { return _lanes[lane & 1]; }
    uint64_t *_lanes[2];
    _CacheArrayTerms *_terms;
};

/// Every input record through its kind's digest hook, record order.
void _CacheMixInputs(UsdGenGraphSpec const &desc, uint64_t *h0, uint64_t *h1,
                     _CacheArrayTerms *terms, bool layout)
{
    _CacheInputDigest sink(h0, h1, terms);
    for (UsdGenInputRecord const &record : desc.inputs) {
        UsdGenInputDesc const *input = UsdGenFindInputDesc(record.kind);
        if (input && input->digest) input->digest(record, sink, layout);
    }
}

UsdGenEpoch _ExecutionPlanDigest(UsdGenGraph const &graph,
                                UsdGenGraphSpec const &desc)
{
    uint64_t h0 = 1469598103934665603ULL;
    uint64_t h1 = 1099511628211ULL;
    _CacheArrayTerms terms;
    _CacheMix(&h0, static_cast<uint64_t>(graph.NodeCount()));
    _CacheMix(&h1, static_cast<uint64_t>(graph.TerminalNodeId()));
    for (UsdGenNodeId id = 0; id != static_cast<UsdGenNodeId>(graph.NodeCount()); ++id) {
        UsdGenCompiledNode const &node = graph.Node(id);
        _CacheMixText(&h0, node.record ? node.record->Name().GetString() : std::string());
        _CacheMix(&h1, node.desc ? node.desc->key.value : 0);
        _CacheMix(&h1, node.structuralDigest[0]);
        _CacheMix(&h0, node.structuralDigest[1]);
        // structuralDigest intentionally excludes value/capture-class
        // parameters for incremental recompilation. Cache identity cannot:
        // include the compiled value digest and all resolved descriptor terms
        // so a terminal value edit is a miss even when its plan is reusable.
        _CacheMix(&h1, node.paramValueDigest);
        if (node.desc) {
            UsdGenOpNode const &nd = *node.desc;
            _CacheMix(&h0, static_cast<uint64_t>(nd.enabled));
            _CacheMix(&h1, static_cast<uint64_t>(nd.seed));
            _CacheMixText(&h1, nd.mode.GetString());
            _CacheMixKeys(&h1, nd.inputs);
            _CacheMixKeys(&h0, nd.references);
            _CacheMixKeys(&h1, nd.inputSets);
            _CacheMixKeys(&h0, nd.surfaces);
            _CacheMixKeys(&h1, nd.maps);
            _CacheMix(&h0, static_cast<uint64_t>(nd.mapBindings.size()));
            for (UsdGenMapBinding const &binding : nd.mapBindings) {
                _CacheMix(&h0, binding.map.value);
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
    _CacheMixText(&h0, desc.basis.GetString());
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
    _CacheMix(&h1, desc.preview.source.value);
    _CacheMixText(&h1, desc.preview.attribute.GetString());
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
    _CacheMix(&h0, desc.material.value);
    uint64_t timeBits = 0, cpsBits = 0;
    std::memcpy(&timeBits, &desc.time, sizeof(timeBits));
    std::memcpy(&cpsBits, &desc.timeCodesPerSecond, sizeof(cpsBits));
    _CacheMix(&h0, timeBits);
    _CacheMix(&h1, cpsBits);
    for (UsdGenOpNode const &node : desc.nodes) {
        _CacheMix(&h0, static_cast<uint64_t>(node.expressionBindings.size()));
        for (UsdGenParamBinding const &binding : node.expressionBindings) {
            _CacheMix(&h1, binding.expression.value);
            _CacheMixText(&h0, binding.output.GetString());
            _CacheMixText(&h1, binding.nativeType.GetString());
            _CacheMixText(&h0, binding.destination.GetString());
            _CacheMixShape(&h1, binding.destinationShape);
            _CacheMix(&h0, static_cast<uint64_t>(binding.domain));
            _CacheMix(&h1, binding.literal.GetHash());
        }
    }
    _CacheMixInputs(desc, &h0, &h1, &terms, false);
    _CacheDigestTerms(terms);
    return {h0, h1};
}

uint64_t _ExecutionLayoutDigest(UsdGenGraph const &graph,
                                UsdGenGraphSpec const &desc)
{
    uint64_t hash = 1469598103934665603ULL;
    _CacheMix(&hash, static_cast<uint64_t>(graph.ChunkSize()));
    _CacheMix(&hash, static_cast<uint64_t>(graph.TileTarget()));
    _CacheMix(&hash, static_cast<uint64_t>(graph.ChunksPerTile()));
    _CacheMix(&hash, static_cast<uint64_t>(desc.tileTarget));
    _CacheMixInputs(desc, &hash, &hash, nullptr, true);
    return hash;
}

UsdGenExecutionContext _ExecutionCacheContext(
    UsdGenGraphSpec const &desc, UsdGenContext evaluationContext,
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
    UsdGenGraph const &graph, UsdGenGraphSpec const &desc,
    UsdGenExecutionContext context, double frame)
{
    return UsdGenExecutionCacheKey::Make(
        desc.description, _ExecutionPlanDigest(graph, desc), graph.InputVersions(),
        context, _ExecutionLayoutDigest(graph, desc), frame);
}


bool _GenerationBytes(UsdGenGeneration const &generation, size_t *result)
{
    if (!result) return false;
    size_t total = sizeof(UsdGenGeneration);
    if (generation.publication) {
        UsdGenPublisherDef const *publisher = UsdGenFindPublisher(generation.publication->type);
        size_t const bytes = publisher && publisher->bytes ? publisher->bytes(*generation.publication) : 0;
        if (bytes == 0 || bytes > std::numeric_limits<size_t>::max() - total) return false;
        total += bytes;
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

// The publisher's diff of two generations; null when `next` has no
// publication (device lane) or its type has no diff.
UsdGenRef<const UsdGenPublication> _DiffGenerations(UsdGenGeneration const *prev,
                                                    UsdGenGeneration const &next)
{
    if (!next.publication) return {};
    UsdGenPublisherDef const *publisher = UsdGenFindPublisher(next.publication->type);
    if (!publisher || !publisher->diff) return {};
    UsdGenPublication const *const before =
        prev && prev->publication && prev->publication->type == next.publication->type
            ? prev->publication.Get() : nullptr;
    return publisher->diff(before, *next.publication);
}

// Groups the generation publishes (0 on the device lane).
uint64_t _PublishedGroups(UsdGenGeneration const *generation)
{
    return generation && generation->publication ? generation->publication->groups : 0;
}

struct _PresentationScalars
{
    GfMatrix4d xformMatrix{1.0};
    TfToken purpose;
    TfToken visibility;
    UsdGenPrimRef material;
    TfToken materialPurpose{UsdGenTokens().allPurpose};
    int refineLevel = 2;
    UsdGenPrimRef origin;
    UsdGenPrimRef dependency;
};

_PresentationScalars _BuildPresentationScalars(
    UsdGenGraphSpec const &desc, UsdGenNodeKey dependencySurface)
{
    _PresentationScalars result;
    result.xformMatrix = desc.xformMatrix;
    result.purpose = desc.purpose;
    result.visibility = desc.visibility;
    result.material = UsdGenPrimRef{desc.material};
    result.origin = UsdGenPrimRef{desc.description};
    result.dependency = UsdGenPrimRef{dependencySurface};
    return result;
}

// CUDA admission permits at most one source root-binding surface and
// ExecuteCudaGraph rejects a named binding that is absent from desc.surfaces.
// Return no dependency for the valid empty/no-binding case; never invent one.
UsdGenNodeKey _CudaSourceDependencySurface(UsdGenGraphSpec const &desc)
{
    if (desc.nodes.empty() || desc.nodes.front().surfaces.size() != 1)
        return UsdGenNodeKey();
    UsdGenNodeKey const candidate = desc.nodes.front().surfaces.front();
    UsdGenInputRecord const *found =
        desc.FindInput(UsdGenInputKindWithRole(UsdGenInputRole::Surface), candidate);
    return found ? found->key : UsdGenNodeKey();
}

UsdGenDevicePresentationMetadata _BuildDevicePresentation(
    UsdGenGraphSpec const &desc)
{
    _PresentationScalars const scalars = _BuildPresentationScalars(
        desc, _CudaSourceDependencySurface(desc));
    UsdGenDevicePresentationMetadata result;
    result.description = desc.description;
    result.xformMatrix = scalars.xformMatrix;
    result.purpose = scalars.purpose;
    result.visibility = scalars.visibility;
    result.material = scalars.material;
    result.materialPurpose = scalars.materialPurpose;
    result.refineLevel = scalars.refineLevel;
    result.origin = scalars.origin;
    result.dependency = scalars.dependency;
    return result;
}

}  // namespace

struct UsdGenSessionCooker::_ProgressSlot
{
    UsdGenTileId tile{};
    UsdGenRef<const UsdGenPublication> publication;
    std::atomic<bool> ready{false};
};

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

void UsdGenSessionCooker::_PublishCudaBindingStats(
    std::vector<UsdGenCudaBindingStats> stats)
{
    auto ref = UsdGenMakeRef<_BindingStats>();
    ref->stats = std::move(stats);
    _cudaBindingStats.Store(std::move(ref));
}

UsdGenSessionCooker::~UsdGenSessionCooker() = default;

bool UsdGenSessionCooker::_SelectExecutionCacheDomain(
    UsdGenDeviceBackend backend, int32_t deviceIndex, uint64_t contextIdentity)
{
    UsdGenExecutionCacheDomainKey key{backend, deviceIndex, contextIdentity};
    auto selectedDomain = _executionCacheDomain;
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
    _executionCacheDomain = std::move(replacement);
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
        _cudaWorkspaceDescription = UsdGenNodeKey();
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

bool UsdGenSessionCooker::_Prepare(std::shared_ptr<const UsdGenGraphSpec> desc,
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
    _lastReport.Reset();
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
            // Recompile is transactional: a rejected edit keeps the last
            // usable graph. Discard it only if the compiler reports that a
            // failure got past its preflight.
            if (!cr.graphIntact)
                _graph = UsdGenGraph();
            // The kept graph still describes the previous descriptor, so the
            // next cook must compile again rather than run it as current.
            _descIdentity.reset();
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
            if (UsdGenInputDesc const* surfaceInput =
                    UsdGenFindInputDesc(UsdGenInputKindWithRole(UsdGenInputRole::Surface)))
                for (UsdGenInputRecord const& s : _desc.Inputs(surfaceInput->kind))
                    _graph.DirtySurface(surfaceInput->surfaceId
                                            ? UsdGenSurfaceId(surfaceInput->surfaceId(s)) : UsdGenSurfaceId(0),
                                        UsdGenDirtySurfaceTopo);
    }
    if (invalidateValues) for (UsdGenNodeId id=0; id<static_cast<UsdGenNodeId>(_graph.NodeCount()); ++id) {
        auto& node=_graph.Node(id); node.paramValueDigest=~uint64_t(0); node.lastParamDigest=0;
    }
    // The publisher's state is made here, before the cook's exec phase.
    if (_graph.NodeCount() != 0)
        _SelectPublisher(_graph.NodeOutput(_graph.TerminalNodeId()));
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
    generation.host = _desc.host;
    auto prepared = _store.PreparePublication(std::move(generation));
    auto fence = domain->CapturePublicationFence(key);
    if (!fence || !domain->PublishIfCurrent(*fence, [this, &prepared] {
            return _store.PublishPrepared(std::move(prepared));
        })) return false;
    _publicationFence = ExecutionPublicationFence{domain, *fence};
    _graphBaselineDetached = false;
    _lastReport.Reset();
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
    _cudaWorkspaceDescription = UsdGenNodeKey();
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
    std::shared_ptr<const UsdGenGraphSpec> desc, UsdGenContext context,
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
    // Exec phase from here to the job's completion (registries/token.h).
    UsdGenCookTokenCheck const tokenCheck;
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
                    _lastReport.Reset();
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
                           [this, stages, tokenCheck](auto const&, auto done) {
        auto completion = std::make_shared<UsdGenExecutionTaskGraph::TaskCompletion>(
            std::move(done));
        try {
            if (!FinalizeCudaExecutionJobAsync(stages->job,
                    [this, stages, completion, tokenCheck](std::shared_ptr<const UsdGenDeviceGeneration> device) {
                try {
                    if (!device) {
                        (*completion)({}, std::make_exception_ptr(
                            std::runtime_error("CUDA final stage failed")));
                        return;
                    }
                    if (UsdGenCookInternFails(tokenCheck, nullptr)) {
                        (*completion)({}, std::make_exception_ptr(std::runtime_error(
                            "usdGen: a token was interned during the cook")));
                        return;
                    }
                    // Allocate native diagnostic snapshots before advancing
                    // the shared private publication state.
                    auto bindingStats = UsdGenMakeRef<_BindingStats>();
                    bindingStats->stats = GetCudaBindingStats(*_cudaWorkspace);
                    if (!_FinalizeDevicePublication(std::move(device),
                            stages->cacheDomain, stages->cacheKey, stages->frame)) {
                        (*completion)({}, std::make_exception_ptr(std::runtime_error(
                            "CUDA execution domain was invalidated before publication")));
                        return;
                    }
                    // This is private cooker state for the sole in-flight
                    // Session work lane.  Public snapshot/pending acceptance
                    // remains the pipeline publication produced by session.cpp.
                    _cudaBindingStats.Store(std::move(bindingStats));
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
    std::shared_ptr<const UsdGenGraphSpec> desc, UsdGenContext context,
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
                  [provider](UsdGenGraphSpec const& d, UsdGenDiagnostics* diag) {
                      return provider->Compile(d, diag);
                  }, &plan)) { completion(_store.Get(), {}); return; }
    // Exec phase from here to the device result (registries/token.h).
    UsdGenCookTokenCheck const tokenCheck;
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
                    _graphBaselineDetached = true; _lastReport.Reset();
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
                    completion, tokenCheck](
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
        if (UsdGenCookInternFails(tokenCheck, &_lastDiagnostics)) {
            completion(_store.Get(), std::make_exception_ptr(std::runtime_error(
                "usdGen: a token was interned during the cook"))); return;
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
    return _cudaBindingStats.Load()->stats;
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
        ? UsdGenRef<const UsdGenPublication>()
        : _DiffGenerations(previous.get(), *_store.Get());
    ++_stats.commits;
    _stats.publishedTiles = _PublishedGroups(_store.Get().get());
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
    UsdGenExecutionCacheKey current = _MakeExecutionCacheKey(
        _graph, _desc, context, frame);
    if (current != candidate.key)
        return rejectBecause("the key moved between cook and publication");
    size_t exactBytes = 0;
    if (!_GenerationBytes(*candidate.generation, &exactBytes) ||
        exactBytes != candidate.bytes) return rejectBecause("size changed");
    std::string reason;
    bool admitted = false;
    if (_coalescedRegistration && _coalescedRegistration->leader &&
        _coalescedDomain == _executionCacheDomain &&
        _coalescedRegistration->key == candidate.key) {
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
    std::shared_ptr<const UsdGenGraphSpec> desc, UsdGenContext context,
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
    // Once populated, replace the complete pose together. Streaming updates
    // would display mixed poses and disable capture-direct point updates.
    // Geometry notices can precede the host's current-frame notice, so the
    // frame number alone cannot distinguish playback from an authoring edit.
    if (_PublishedGroups(previous.get()) != 0) progress = {};
    if (!_Prepare(std::move(desc), context, devicePublicationEnabled, pending,
                  std::move(previous), std::move(publishedStats), invalidateValues,
                  previousPublishedWorkerEpoch)) return _store.Get();
    phases.Phase("prepare");
    // Exec phase from here: an intern fails the cook (registries/token.h).
    UsdGenCookTokenCheck const tokenCheck;

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
                _lastReport = _DiffGenerations(prev.get(), *next);
                ++_stats.commits;
                _stats.publishedTiles = _PublishedGroups(next.get());
                _lastNodeStats.clear();
                _graphBaselineDetached = true;
                phases.Phase("cache hit publish");
                phases.Print(frame, _traceCompile,
                             "cache hit (operators not run; next cook recompiles)",
                             _desc);
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
                        _lastReport.Reset();
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
            _lastDiagnostics.Warn("usdGen:preview: " + _desc.Label(_desc.description) +
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
        _PublishCudaBindingStats(GetCudaBindingStats(*_cudaWorkspace));
        UsdGenGeneration gen;
        gen.frame = frame;
        gen.device = std::move(device);
        // `_desc` is the copied command descriptor that compiled the accepted
        // CUDA plan above.  Capture its presentation once with the device
        // payload; publishing must not consult a later graph descriptor.
        gen.devicePresentation = _BuildDevicePresentation(_desc);
        gen.host = _desc.host;
        // Do not call the host scheduler or the publisher for a
        // device generation. Tools retain this snapshot directly.
        auto prepared = _store.PreparePublication(std::move(gen));
        auto fence = cacheDomain->CapturePublicationFence(cacheKey);
        if (!fence || !cacheDomain->PublishIfCurrent(*fence, [this, &prepared] {
                return _store.PublishPrepared(std::move(prepared));
            })) return reject();
        _publicationFence = ExecutionPublicationFence{cacheDomain, *fence};
        _lastReport.Reset();
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
    // Streamed tiles, without a lock: each completed tile claims a slot by
    // fetch_add and publishes it with a release store (tiles past the slot
    // count go on an overflow stack). Whoever holds `delivering` emits the
    // ready slot prefix and the overflow in order, so progress stays serial.
    size_t const slotCount = progress && !_desc.preview.Active()
        ? size_t(std::max(0, _graph.NumGroups())) : 0;
    if (slotCount > _progressSlotCapacity) {
        _progressSlots.reset(new _ProgressSlot[slotCount]);
        _progressSlotCapacity = slotCount;
    }
    _ProgressSlot *const progressSlots = _progressSlots.get();
    std::atomic<size_t> slotsClaimed{0};
    std::atomic<size_t> slotsDelivered{0};
    std::atomic<bool> delivering{false};
    UsdGenTreiberStack<_StreamedTile, &_StreamedTile::next> overflowReady;
    _StreamedTile *overflowDelivered = nullptr;  // guarded by `delivering`
    uint64_t progressSequence = 0;               // guarded by `delivering`
    struct StreamReset {
        _ProgressSlot *slots;
        size_t count;
        std::atomic<size_t> const &claimed;
        _StreamedTile *&overflow;
        ~StreamReset() {
            size_t const used = std::min(count, claimed.load(std::memory_order_acquire));
            for (size_t i = 0; i < used; ++i) {
                slots[i].publication.Reset();
                slots[i].ready.store(false, std::memory_order_relaxed);
            }
            while (_StreamedTile *n = overflow) {
                overflow = n->next;
                delete n;
            }
        }
    } streamReset{progressSlots, slotCount, slotsClaimed, overflowDelivered};
    auto emitProgress = [&](UsdGenRef<const UsdGenPublication> tile) {
        if (!progress || (superseded && superseded())) return;
        auto update = std::make_shared<UsdGenSession::TileProgress>();
        update->epoch = workEpoch;
        update->frame = frame;
        update->tile = std::move(tile);
        update->host = _desc.host;
        if (superseded && superseded()) return;
        update->sequence = ++progressSequence;
        try { progress(std::move(update)); } catch (...) {
            // A provisional render consumer cannot invalidate the cook.
        }
    };
    auto deliverStreamed = [&] {
        for (;;) {
            if (delivering.exchange(true, std::memory_order_acquire)) return;
            size_t d = slotsDelivered.load(std::memory_order_relaxed);
            while (d < slotCount && progressSlots[d].ready.load(std::memory_order_acquire)) {
                try { emitProgress(progressSlots[d].publication); } catch (...) {}
                slotsDelivered.store(++d, std::memory_order_relaxed);
            }
            for (_StreamedTile *n = overflowReady.PopAll(); n;) {
                _StreamedTile *next = n->next;
                try { emitProgress(n->publication); } catch (...) {}
                n->next = overflowDelivered;
                overflowDelivered = n;
                n = next;
            }
            delivering.store(false, std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (!(d < slotCount && progressSlots[d].ready.load(std::memory_order_acquire)) &&
                overflowReady.Empty())
                return;
        }
    };
    // Streams only without a usdGen:preview: a group built here never carries
    // preview colours (the whole-output colour pass runs after the run).
    auto tileCompleted = progress && !_desc.preview.Active()
        ? UsdGenScheduler::TileCompleted([&](UsdGenChunkGroup const& tile,
                                               UsdGenElementSet const& output) {
            if (superseded && superseded()) return;
            UsdGenPublisherDef const *const streamPublisher =
                UsdGenFindPublisher(output.Channels().Layout().primitive);
            if (!streamPublisher || !streamPublisher->buildGroup) return;
            UsdGenCompiledNode const &terminal = _graph.Node(_graph.TerminalNodeId());
            UsdGenPublishContext group;
            group.spec = &_desc;
            group.graph = &_graph;
            group.terminal = &output;
            group.chunks = {terminal.chunks.data(), terminal.chunks.size()};
            group.groups = {&tile, 1};
            group.frame = frame;
            UsdGenRef<const UsdGenPublication> publication =
                streamPublisher->buildGroup(group, 0);
            // The final publication reuses this group instead of building it
            // a second time.
            size_t const i = slotsClaimed.fetch_add(1, std::memory_order_relaxed);
            if (i < slotCount) {
                progressSlots[i].tile = tile.id;
                progressSlots[i].publication = std::move(publication);
                progressSlots[i].ready.store(true, std::memory_order_release);
            } else {
                auto *n = new _StreamedTile;
                n->tile = tile.id;
                n->publication = std::move(publication);
                overflowReady.Push(n);
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
            deliverStreamed();
        }) : UsdGenScheduler::TileCompleted{};
    UsdGenRunResult result = _scheduler.Run(_graph, evalCtx, myReq,
                                             std::move(tileCompleted), superseded);
    _lastDiagnostics = result.diagnostics;
    phases.Phase("run operators");

    // A points-only run: every node that did work writes nothing but points
    // (no generator, no output planes, no remapped vertex planes), so the
    // terminal's other channels are the ones the published tiles were built
    // from -- if they were (_tileChannelsSettled). Cleared here whatever
    // happens next, so a failed or superseded run never leaves it set.
    bool pointsOnlyRun = !result.topologyChanged;
    for (UsdGenNodeRunStats const &rs : result.nodeStats) {
        if (!pointsOnlyRun) break;
        if (rs.id >= static_cast<UsdGenNodeId>(_graph.NodeCount())) { pointsOnlyRun = false; break; }
        UsdGenCompiledNode const &n = _graph.Node(rs.id);
        pointsOnlyRun = n.op && !n.record->Has(kUsdGenOpGenerator) && !n.remapsVertexPlanes &&
            n.op->PlanesTouched() == UsdGenOp::kPlanePoints && n.outputPrimvars.empty();
    }
    bool const channelsSettled = _tileChannelsSettled && pointsOnlyRun;
    _tileChannelsSettled = false;

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
    gen.host = _desc.host;
    if (UsdGenPublisherDef const *publisher = _SelectPublisher(*result.terminalOutput)) {
        // Run has joined every group callback: the streamed set is final.
        std::vector<UsdGenPublishedGroup> streamed;
        if (size_t const claimed = slotsClaimed.load(std::memory_order_acquire)) {
            streamed.reserve(claimed);
            for (size_t i = 0, n = std::min(claimed, slotCount); i < n; ++i)
                streamed.push_back({progressSlots[i].tile, progressSlots[i].publication.Get()});
            for (_StreamedTile const *n = overflowDelivered; n; n = n->next)
                streamed.push_back({n->tile, n->publication.Get()});
            // First completion wins, as with map::emplace.
            std::stable_sort(streamed.begin(), streamed.end(),
                [](UsdGenPublishedGroup const &a, UsdGenPublishedGroup const &b) {
                    return a.group < b.group;
                });
        }
        UsdGenCompiledNode const &terminal = _graph.Node(_graph.TerminalNodeId());
        UsdGenWorkDispatcher dispatcher = _scheduler.MakeWorkDispatcher();
        UsdGenPublishContext ctx;
        ctx.spec = &_desc;
        ctx.graph = &_graph;
        ctx.terminal = result.terminalOutput;
        ctx.chunks = {terminal.chunks.data(), terminal.chunks.size()};
        ctx.groups = result.groups;
        ctx.streamed = {streamed.data(), streamed.size()};
        ctx.previous = prev ? prev->publication.Get() : nullptr;
        ctx.state = _publisherState.get();
        ctx.frame = frame;
        ctx.topologyChanged = result.topologyChanged;
        ctx.rebuildAll = _rebuildAllTiles;
        ctx.pointsOnly = channelsSettled;
        ctx.dispatcher = &dispatcher;
        ctx.diag = &_lastDiagnostics;
        if (progress) {
            ctx.emit = [](void *user, UsdGenRef<const UsdGenPublication> group) {
                (*static_cast<decltype(emitProgress) *>(user))(std::move(group));
            };
            ctx.emitUser = &emitProgress;
        }
        ctx.phase = [](void *user, char const *name, size_t rebuilt, size_t total) {
            static_cast<_CommitPhases *>(user)->Phase(name, rebuilt, total);
        };
        ctx.phaseUser = &phases;
        gen.publication = publisher->build(ctx);
    }

    if (UsdGenCookInternFails(tokenCheck, &_lastDiagnostics)) {
        _graphUntrusted = true;
        return _store.Get();
    }
    auto prepared = _store.PreparePublication(std::move(gen));
    auto fence = cacheDomain->CapturePublicationFence(cacheKey);
    if (!fence || !cacheDomain->PublishIfCurrent(*fence, [this, &prepared] {
            return _store.PublishPrepared(std::move(prepared));
        })) return _store.Get();
    _publicationFence = ExecutionPublicationFence{cacheDomain, *fence};
    _graphBaselineDetached = false;
    _tileChannelsSettled = true;   // published from this graph's current buffers
    phases.Phase("publish");

    // -- step 7: diff vs the previous generation (06 §5.1) -----------------
    UsdGenGenerationConstPtr next = _store.Get();
    _lastReport = _DiffGenerations(prev.get(), *next);
    phases.Phase("diff");

    // -- stats (03 §9.2) ----------------------------------------------------
    ++_stats.commits;
    _stats.publishedTiles = _PublishedGroups(next.get());
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
    phases.Print(frame, _traceCompile, "cache miss, cooked", _desc);

    // Remaining chunk dirt (skipped no-op nodes are clean; anything the
    // scheduler could not run stays dirty; 03 §5.4).
    return next;
}

UsdGenPublisherDef const *UsdGenSessionCooker::_SelectPublisher(
    UsdGenElementSet const &terminal)
{
    UsdGenPublisherDef const *const publisher =
        UsdGenFindPublisher(terminal.Channels().Layout().primitive);
    if (!publisher || !publisher->build) return nullptr;
    if (publisher != _publisher) {
        _publisherState = publisher->createState ? publisher->createState() : nullptr;
        _publisher = publisher;
    }
    return publisher;
}

std::unordered_map<UsdGenNodeId, UsdGenNodeStats> UsdGenSessionCooker::NodeStats() const
{
    std::unordered_map<UsdGenNodeId, UsdGenNodeStats> out;
    for (auto const& entry : _lastNodeStats) {
        if (entry.first >= static_cast<UsdGenNodeId>(_graph.NodeCount())) continue;
        UsdGenCompiledNode const& n = _graph.Node(entry.first);
        UsdGenNodeStats& s = out[entry.first];
        s.type = n.record ? n.record->Name() : TfToken(); s.key = n.desc ? n.desc->key : UsdGenNodeKey();
        s.captureMs = entry.second.captureMs; s.lastEvalMs = entry.second.evalMs;
        s.meanEvalMs = entry.second.evalMs; s.elementsOut = _graph.NodeOutput(n.id).Count(1);
        for (UsdGenChunkDirtyBits bit : n.chunkDirty) if (bit) ++s.chunksDirty;
        s.chunksTotal = n.chunks.size();
    }
    return out;
}

std::shared_ptr<const UsdGenGraphRoutingSnapshot> UsdGenSessionCooker::Routing() const
{ return _graph.RoutingSnapshot(); }
UsdGenGraphSpec UsdGenSessionCooker::GraphSpec() const
{ return _graph.NodeCount() ? _graph.Spec() : _desc; }
std::shared_ptr<const UsdGenCudaExecutionPlan> UsdGenSessionCooker::CudaPlan() const
{ return _graph.CudaPlan(); }

}  // namespace usdGen
