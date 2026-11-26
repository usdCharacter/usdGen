#include "usdGen/executionResources.h"
#include "usdGenCuda/backendId.h"
#include "usdGen/executionRetirement.h"
#include "usdGen/gpu/cudaRetirement.h"
#include "usdGenCurves/cuda/curveCompaction.h"
#include "usdGenCurves/cuda/curveResample.h"
#include "usdGenCurves/cuda/curveSource.h"
#include "usdGenCurves/cuda/generation.h"
#include "usdGenCurves/cuda/namedChannelTopology.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; \
} } while (false)

namespace {

std::unique_ptr<CudaCurveSource> Source()
{
    static int32_t const counts[]{2, 3};
    static uint64_t const ids[]{101, 202};
    static float3 const points[]{
        make_float3(0, 0, 0), make_float3(10, 0, 0),
        make_float3(20, 0, 0), make_float3(30, 0, 0),
        make_float3(40, 0, 0)};
    auto source = std::make_unique<CudaCurveSource>();
    CurveSourceInput input;
    input.curveVertexCounts = {counts, 2};
    input.points = {points, 5};
    input.restPoints = {points, 5};
    input.stableIds = {ids, 2};
    input.fallbackWidth = 1;
    if (source->Set(input, nullptr) != CurveSourceStatus::Ok ||
        source->Finish(nullptr) != CurveSourceStatus::Ok)
        return {};
    return source;
}

std::shared_ptr<const UsdGenDeviceGeneration> Base(uint64_t generation)
{
    return MakeSourceGeneration(Source(), generation);
}

template<class T, size_t N>
CudaNamedChannelPlane Plane(char const* name, UsdGenDeviceValueType type,
                            UsdGenDeviceDomain domain, uint64_t elements,
                            uint32_t arity, std::array<T, N> const& values)
{
    CudaNamedChannelPlane result;
    result.metadata = {name, type, domain, elements, arity,
        uint32_t(sizeof(T) * arity), true,
        UsdGenDeviceChannelSemantic::Generic};
    result.bytes = std::make_unique<DeviceBuffer<unsigned char>>();
    if (result.bytes->reset(sizeof(values)) != cudaSuccess ||
        cudaMemcpy(result.bytes->data(), values.data(), sizeof(values),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        result.bytes.reset();
    return result;
}

template<class T, size_t N>
bool Read(std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
          char const* name, std::array<T, N>* values)
{
    auto lease = AcquireNamedChannel(generation, name, nullptr);
    return lease && lease.Bytes().size == sizeof(*values) &&
        cudaMemcpy(values->data(), lease.Bytes().data, sizeof(*values),
                   cudaMemcpyDeviceToHost) == cudaSuccess;
}

std::shared_ptr<const UsdGenDeviceGeneration> Named()
{
    auto base = Base(1);
    if (!base) return {};
    std::vector<CudaNamedChannelPlane> planes;
    planes.push_back(Plane("f", UsdGenDeviceValueType::Float32,
        UsdGenDeviceDomain::Point, 5, 1,
        std::array<float, 5>{{0, 10, 20, 30, 40}}));
    planes.push_back(Plane("i", UsdGenDeviceValueType::Int32,
        UsdGenDeviceDomain::Point, 5, 1,
        std::array<int, 5>{{0, 1, 2, 3, 4}}));
    planes.push_back(Plane("u", UsdGenDeviceValueType::Int32,
        UsdGenDeviceDomain::Primitive, 2, 2,
        std::array<int, 4>{{7, 8, 9, 10}}));
    planes.push_back(Plane("c", UsdGenDeviceValueType::Float32,
        UsdGenDeviceDomain::Groom, 1, 2,
        std::array<float, 2>{{3, 5}}));
    for (auto const& plane : planes)
        if (!plane.bytes) return {};
    return MakeNamedChannelRevisionGeneration(base, 2, std::move(planes));
}

std::shared_ptr<const UsdGenDeviceGeneration> Resampled()
{
    auto source = Source();
    auto resample = std::make_unique<CudaCurveResample>();
    if (!source || resample->Apply(source->view(), source->hairT(),
            source->rootPrim(), source->rootUV(), 4, nullptr) !=
                CurveResampleStatus::Ok ||
        resample->Finish(nullptr) != CurveResampleStatus::Ok)
        return {};
    return MakeResampledGeneration(std::move(source), std::move(resample), 3);
}

std::shared_ptr<const UsdGenDeviceGeneration> Compacted()
{
    auto source = Source();
    DeviceBuffer<unsigned char> keep;
    std::array<unsigned char, 2> const values{{0, 1}};
    if (!source || keep.reset(values.size()) != cudaSuccess ||
        cudaMemcpy(keep.data(), values.data(), values.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return {};
    auto compact = std::make_unique<CudaCurveCompaction>();
    if (compact->Apply(source->view(), source->hairT(), source->rootPrim(),
            source->rootUV(), {keep.data(), keep.size()}, nullptr) !=
                CurveCompactionStatus::Ok ||
        compact->Finish(nullptr) != CurveCompactionStatus::Ok)
        return {};
    return MakeCompactedGeneration(std::move(compact), 3);
}

bool Near(float left, float right)
{
    return std::fabs(left - right) < 1e-5f;
}

struct AsyncCompletion {
    std::atomic<int> status{std::numeric_limits<int>::min()};
    std::atomic<unsigned> calls{0};
};

void AsyncCallback(cudaStream_t, cudaError_t status, void* userdata) noexcept
{
    auto* completion = static_cast<AsyncCompletion*>(userdata);
    completion->status.store(static_cast<int>(status), std::memory_order_release);
    completion->calls.fetch_add(1, std::memory_order_relaxed);
}

struct RawInputLifetime {
    CudaGeometryLease source;
    CudaGeometryLease target;
    CudaNamedChannelLease channel;
    std::unique_ptr<DeviceBuffer<unsigned char>> readyPlane;
    std::unique_ptr<DeviceBuffer<uint32_t>> offsets;
    std::unique_ptr<DeviceBuffer<uint64_t>> ids;
};

} // namespace

int main()
{
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 77;
    CHECK(ConfigureCudaExecutionResources(device, {size_t{16} << 20, 0}));
    UsdGenExecutionResourceDevice const resourceKey{
        usdGen::UsdGenCudaBackendId(), device};
    auto resources = FindUsdGenExecutionResourcePool(resourceKey);
    auto retirement = GetOrCreateUsdGenExecutionRetirementService(
        resourceKey, {64});
    CHECK(resources && retirement);

    auto source = Named();
    auto resampled = Resampled();
    auto compacted = Compacted();
    CHECK(source && resampled && compacted);

    // Generation ordering is part of immutable COW identity. A failed
    // transform leaves all existing generations and accounting unchanged.
    auto const beforeOrderFailure = resources->Snapshot().usedBytes;
    std::string reason;
    CHECK(!TransformCudaNamedChannelsForResample(
        source, resampled, 3, nullptr, &reason));
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == beforeOrderFailure &&
          reason.find("ordered local CUDA generations") != std::string::npos);

    // A deliberately undersized aggregate reservation fails before launch
    // and returns its Pending credit without partially publishing an overlay.
    auto tinyReservation = resources->TryReserveMemory(1);
    CHECK(tinyReservation);
    reason.clear();
    CHECK(!TransformCudaNamedChannelsForResample(source, resampled, 4,
        nullptr, &reason, &*tinyReservation));
    tinyReservation->Release();
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == beforeOrderFailure &&
          reason.find("validation status") != std::string::npos);

    auto resampleResult = TransformCudaNamedChannelsForResample(
        source, resampled, 4, nullptr, &reason);
    CHECK(resampleResult &&
          resampleResult->Owner() != source->Owner() &&
          resampleResult->Owner() != resampled->Owner());
    std::array<float, 8> floatValues{};
    std::array<int, 8> intValues{};
    std::array<int, 4> uniformValues{};
    std::array<float, 2> constantValues{};
    CHECK(Read(resampleResult, "f", &floatValues) &&
          Read(resampleResult, "i", &intValues) &&
          Read(resampleResult, "u", &uniformValues) &&
          Read(resampleResult, "c", &constantValues));
    std::array<float, 8> const expectedFloat{{
        0, 10.f / 3, 20.f / 3, 10,
        20, 20.f + 20.f / 3, 20.f + 40.f / 3, 40}};
    for (size_t i = 0; i != floatValues.size(); ++i)
        CHECK(Near(floatValues[i], expectedFloat[i]));
    CHECK((intValues == std::array<int, 8>{{0, 0, 1, 1, 2, 3, 3, 4}} &&
          uniformValues == std::array<int, 4>{{7, 8, 9, 10}} &&
          constantValues == std::array<float, 2>{{3, 5}}));

    // The async candidate holds source/topology leases and private outputs
    // until its one native callback is proved by the host relay.  No output
    // generation exists before Commit, and attempting a second callback arm
    // is rejected rather than publishing twice.
    cudaStream_t asyncStream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&asyncStream, cudaStreamNonBlocking) == cudaSuccess);
    {
        CudaNamedChannelTopologyCandidate asyncCandidate;
        reason.clear();
        CHECK(asyncCandidate.Begin(source, resampled, 5, asyncStream,
            CudaNamedChannelTopologyMode::Resample, &reason) == cudaSuccess);
        CHECK(asyncCandidate.HasUnprovenWork());
        AsyncCompletion asyncCompletion;
        CHECK(asyncCandidate.FinishAsync(AsyncCallback, &asyncCompletion) == cudaSuccess);
        CHECK(asyncCandidate.FinishAsync(AsyncCallback, &asyncCompletion) ==
              cudaErrorInvalidResourceHandle);
        reason.clear();
        CHECK(!asyncCandidate.Commit(cudaSuccess, &reason));
        CHECK(reason.find("callback has not proved completion") != std::string::npos);
        CHECK(cudaStreamSynchronize(asyncStream) == cudaSuccess);
        CHECK(asyncCompletion.calls.load(std::memory_order_acquire) == 1 &&
              asyncCompletion.status.load(std::memory_order_acquire) == cudaSuccess);
        auto asyncResult = asyncCandidate.Commit(
            static_cast<cudaError_t>(asyncCompletion.status.load(std::memory_order_acquire)),
            &reason);
        CHECK(asyncResult && !asyncCandidate.HasUnprovenWork());
        std::array<float, 8> asyncFloat{};
        CHECK(Read(asyncResult, "f", &asyncFloat));
        for (size_t i = 0; i != asyncFloat.size(); ++i)
            CHECK(Near(asyncFloat[i], expectedFloat[i]));
        asyncResult.reset();
    }
    // A topology-producing execution job has raw candidate geometry rather
    // than a public target generation.  BeginRaw only borrows those views;
    // the job-owned lifetime keeps all input leases valid through the terminal
    // callback, and CommitPlanes transfers only private output buffers.
    {
        auto lifetime = std::make_shared<RawInputLifetime>();
        lifetime->source = AcquireGeometry(source, asyncStream);
        lifetime->target = AcquireGeometry(resampled, asyncStream);
        lifetime->channel = AcquireNamedChannel(source, "f", asyncStream);
        CHECK(lifetime->source && lifetime->target && lifetime->channel);
        CudaNamedChannelTopologyCandidate rawCandidate;
        std::vector<CudaNamedChannelTopologyCandidate::RawInputPlane> rawInputs{{
            *lifetime->channel.Metadata(), lifetime->channel.Bytes()}};
        reason.clear();
        CHECK(rawCandidate.BeginRaw(lifetime->source.Geometry(), lifetime->target.Geometry(),
            std::move(rawInputs), asyncStream, CudaNamedChannelTopologyMode::Resample,
            lifetime, &reason) == cudaSuccess);
        AsyncCompletion rawCompletion;
        CHECK(rawCandidate.FinishAsync(AsyncCallback, &rawCompletion) == cudaSuccess);
        CHECK(cudaStreamSynchronize(asyncStream) == cudaSuccess);
        std::vector<CudaNamedChannelPlane> rawOutputs;
        CHECK(rawCandidate.CommitPlanes(
            static_cast<cudaError_t>(rawCompletion.status.load(std::memory_order_acquire)),
            &rawOutputs, &reason));
        CHECK(rawOutputs.size() == 1 && rawOutputs.front().metadata.name == "f" &&
              rawOutputs.front().bytes->waitOn(nullptr) == cudaSuccess);
        std::array<float, 8> rawFloat{};
        CHECK(cudaMemcpy(rawFloat.data(), rawOutputs.front().bytes->data(), sizeof(rawFloat),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
        for (size_t i = 0; i != rawFloat.size(); ++i)
            CHECK(Near(rawFloat[i], expectedFloat[i]));
    }
    // Raw input readiness is an explicit event edge, not merely a lifetime
    // promise. A producer on another stream can therefore hand its immutable
    // buffer to the transform stream without a host synchronization.
    {
        cudaStream_t producer = nullptr;
        CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking) == cudaSuccess);
        auto lifetime = std::make_shared<RawInputLifetime>();
        lifetime->source = AcquireGeometry(source, asyncStream);
        lifetime->target = AcquireGeometry(resampled, asyncStream);
        lifetime->readyPlane = std::make_unique<DeviceBuffer<unsigned char>>();
        std::array<float, 5> values{{0, 10, 20, 30, 40}};
        CHECK(lifetime->source && lifetime->target &&
              lifetime->readyPlane->reset(sizeof(values)) == cudaSuccess &&
              cudaMemcpyAsync(lifetime->readyPlane->data(), values.data(), sizeof(values),
                              cudaMemcpyHostToDevice, producer) == cudaSuccess &&
              lifetime->readyPlane->recordUse(producer) == cudaSuccess);
        CudaNamedChannelTopologyCandidate candidate;
        CudaNamedChannelTopologyCandidate::RawInputPlane input{
            {"ready", UsdGenDeviceValueType::Float32, UsdGenDeviceDomain::Point,
             5, 1, sizeof(float), true, UsdGenDeviceChannelSemantic::Generic},
            {lifetime->readyPlane->data(), lifetime->readyPlane->size()},
            lifetime->readyPlane.get()};
        CHECK(candidate.BeginRaw(lifetime->source.Geometry(), lifetime->target.Geometry(),
            {input}, asyncStream, CudaNamedChannelTopologyMode::Resample,
            lifetime, &reason) == cudaSuccess);
        AsyncCompletion completion;
        CHECK(candidate.FinishAsync(AsyncCallback, &completion) == cudaSuccess &&
              cudaStreamSynchronize(asyncStream) == cudaSuccess);
        std::vector<CudaNamedChannelPlane> outputs;
        CHECK(candidate.CommitPlanes(static_cast<cudaError_t>(
            completion.status.load(std::memory_order_acquire)), &outputs, &reason));
        CHECK(outputs.size() == 1);
        CHECK(cudaStreamDestroy(producer) == cudaSuccess);
    }
    // Compaction cannot duplicate a survivor stable id. The raw target below
    // is otherwise shape-valid and used to slip through by copying curve 101
    // twice.
    {
        auto lifetime = std::make_shared<RawInputLifetime>();
        lifetime->source = AcquireGeometry(source, asyncStream);
        lifetime->channel = AcquireNamedChannel(source, "f", asyncStream);
        lifetime->offsets = std::make_unique<DeviceBuffer<uint32_t>>();
        lifetime->ids = std::make_unique<DeviceBuffer<uint64_t>>();
        std::array<uint32_t, 3> offsets{{0, 2, 4}};
        std::array<uint64_t, 2> ids{{101, 101}};
        CHECK(lifetime->source && lifetime->channel &&
              lifetime->offsets->reset(offsets.size()) == cudaSuccess &&
              lifetime->ids->reset(ids.size()) == cudaSuccess &&
              cudaMemcpyAsync(lifetime->offsets->data(), offsets.data(), sizeof(offsets),
                              cudaMemcpyHostToDevice, asyncStream) == cudaSuccess &&
              cudaMemcpyAsync(lifetime->ids->data(), ids.data(), sizeof(ids),
                              cudaMemcpyHostToDevice, asyncStream) == cudaSuccess);
        DeviceCurveGeometryView duplicateTarget{};
        duplicateTarget.curveCount = 2;
        duplicateTarget.pointCount = 4;
        duplicateTarget.curveOffsets =
            {lifetime->offsets->data(), lifetime->offsets->size()};
        duplicateTarget.stableIds =
            {lifetime->ids->data(), lifetime->ids->size()};
        CudaNamedChannelTopologyCandidate candidate;
        std::vector<CudaNamedChannelTopologyCandidate::RawInputPlane> inputs{{
            *lifetime->channel.Metadata(), lifetime->channel.Bytes()}};
        CHECK(candidate.BeginRaw(lifetime->source.Geometry(), duplicateTarget,
            std::move(inputs), asyncStream, CudaNamedChannelTopologyMode::Compaction,
            lifetime, &reason) == cudaSuccess);
        AsyncCompletion completion;
        CHECK(candidate.FinishAsync(AsyncCallback, &completion) == cudaSuccess &&
              cudaStreamSynchronize(asyncStream) == cudaSuccess);
        std::vector<CudaNamedChannelPlane> outputs;
        reason.clear();
        CHECK(!candidate.CommitPlanes(static_cast<cudaError_t>(
            completion.status.load(std::memory_order_acquire)), &outputs, &reason));
        CHECK(outputs.empty() && reason.find("duplicate survivor stable ids") != std::string::npos);
    }
    // Empty Point planes are legitimate immutable channels on empty geometry.
    {
        auto lifetime = std::make_shared<RawInputLifetime>();
        lifetime->offsets = std::make_unique<DeviceBuffer<uint32_t>>();
        uint32_t zero = 0;
        CHECK(lifetime->offsets->reset(1) == cudaSuccess &&
              cudaMemcpyAsync(lifetime->offsets->data(), &zero, sizeof(zero),
                              cudaMemcpyHostToDevice, asyncStream) == cudaSuccess);
        DeviceCurveGeometryView empty{};
        empty.curveOffsets =
            {lifetime->offsets->data(), lifetime->offsets->size()};
        CudaNamedChannelTopologyCandidate candidate;
        CudaNamedChannelTopologyCandidate::RawInputPlane input{
            {"empty", UsdGenDeviceValueType::Float32, UsdGenDeviceDomain::Point,
             0, 1, sizeof(float), true, UsdGenDeviceChannelSemantic::Generic},
            {nullptr, 0}};
        CHECK(candidate.BeginRaw(empty, empty, {input}, asyncStream,
            CudaNamedChannelTopologyMode::Resample, lifetime, &reason) == cudaSuccess);
        AsyncCompletion completion;
        CHECK(candidate.FinishAsync(AsyncCallback, &completion) == cudaSuccess &&
              cudaStreamSynchronize(asyncStream) == cudaSuccess);
        std::vector<CudaNamedChannelPlane> outputs;
        CHECK(candidate.CommitPlanes(static_cast<cudaError_t>(
            completion.status.load(std::memory_order_acquire)), &outputs, &reason));
        CHECK(outputs.size() == 1 && outputs.front().metadata.elementCount == 0 &&
              outputs.front().bytes && outputs.front().bytes->size() == 0);
    }
    CHECK(cudaStreamDestroy(asyncStream) == cudaSuccess);
    // Candidate-owned raw outputs/leases have retired before the following
    // accounting-invariance assertion takes its baseline.
    retirement->Drain();

    auto compactResult = TransformCudaNamedChannelsForCompaction(
        source, compacted, 4, nullptr, &reason);
    CHECK(compactResult && compactResult->Owner() != source->Owner());
    std::array<float, 3> compactFloat{};
    std::array<int, 3> compactInt{};
    std::array<int, 2> compactUniform{};
    CHECK(Read(compactResult, "f", &compactFloat) &&
          Read(compactResult, "i", &compactInt) &&
          Read(compactResult, "u", &compactUniform));
    CHECK((compactFloat == std::array<float, 3>{{20, 30, 40}} &&
          compactInt == std::array<int, 3>{{2, 3, 4}} &&
          compactUniform == std::array<int, 2>{{9, 10}}));

    // Revision factories apply the same strict metadata/cardinality contract
    // as first-generation named planes.
    std::vector<CudaNamedChannelPlane> malformedPlanes;
    malformedPlanes.push_back(Plane("badDomain", UsdGenDeviceValueType::Float32,
        UsdGenDeviceDomain::Topology, 5, 1,
        std::array<float, 5>{{0, 1, 2, 3, 4}}));
    reason.clear();
    CHECK(!MakeNamedChannelRevisionGeneration(
        source, 6, std::move(malformedPlanes), {}, &reason));
    CHECK(reason.find("incompatible generic plane") != std::string::npos);

    // Reading the accepted results must not mutate or replace their source
    // COW payload.
    std::array<float, 5> originalFloat{};
    CHECK((Read(source, "f", &originalFloat) &&
          originalFloat == std::array<float, 5>{{0, 10, 20, 30, 40}}));

    // A target-only plane would otherwise remain visible through topology's
    // base owner with stale cardinality. Reject that entire transaction.
    std::vector<CudaNamedChannelPlane> stalePlanes;
    stalePlanes.push_back(Plane("stale", UsdGenDeviceValueType::Int32,
        UsdGenDeviceDomain::Primitive, 2, 1,
        std::array<int, 2>{{1, 2}}));
    auto staleTarget = MakeNamedChannelRevisionGeneration(
        resampled, 4, std::move(stalePlanes));
    CHECK(staleTarget);
    auto const beforeStaleFailure = resources->Snapshot().usedBytes;
    reason.clear();
    CHECK(!TransformCudaNamedChannelsForResample(
        source, staleTarget, 5, nullptr, &reason));
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == beforeStaleFailure &&
          reason.find("unrelated target overlay") != std::string::npos);

    source.reset();
    resampled.reset();
    compacted.reset();
    resampleResult.reset();
    compactResult.reset();
    staleTarget.reset();
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == 0);
    std::puts("testUsdGenCudaNamedChannelTopology: PASS");
    return 0;
}
