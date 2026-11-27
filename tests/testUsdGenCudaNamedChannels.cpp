#include "usdGen/executionResources.h"
#include "testBackend.h"
#include "usdGen/executionRetirement.h"
#include "usdGen/gpu/cudaRetirement.h"
#include "usdGenCurves/cuda/generation.h"
#include "usdGen/gpu/deviceResources.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;
using namespace usdGen::gpu;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {

size_t KindBytes(UsdGenExecutionResourceSnapshot const& snapshot,
                 UsdGenExecutionResourceKind kind) {
    return snapshot.byKind[static_cast<size_t>(kind)];
}

std::unique_ptr<CudaCurveSource> Source() {
    auto source = std::make_unique<CudaCurveSource>();
    int32_t const counts[] = {2};
    float3 const points[] = {
        make_float3(0, 0, 0), make_float3(1, 0, 0)};
    CurveSourceInput input;
    input.curveVertexCounts = {counts, 1};
    input.points = {points, 2};
    input.restPoints = {points, 2};
    input.fallbackWidth = .1f;
    if (source->Set(input, nullptr) != CurveSourceStatus::Ok ||
        source->Finish(nullptr) != CurveSourceStatus::Ok)
        return {};
    return source;
}

std::shared_ptr<const UsdGenDeviceGeneration> BaseGeneration(uint64_t id) {
    return MakeSourceGeneration(Source(), id);
}

template <class T, size_t N>
CudaNamedChannelPlane Plane(char const* name, UsdGenDeviceValueType type,
                            UsdGenDeviceDomain domain, uint64_t elements,
                            uint32_t arity, std::array<T, N> const& values) {
    CudaNamedChannelPlane result;
    result.metadata = {TfToken(name), type, domain, elements, arity,
                       static_cast<uint32_t>(sizeof(T) * arity), true,
                       UsdGenDeviceChannelSemantic::Generic};
    result.bytes = std::make_unique<DeviceBuffer<unsigned char>>();
    if (result.bytes->reset(sizeof(values)) != cudaSuccess ||
        cudaMemcpy(result.bytes->data(), values.data(), sizeof(values),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        result.bytes->recordUse(nullptr) != cudaSuccess)
        result.bytes.reset();
    return result;
}

bool HasChannel(std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
                char const* name, uint32_t arity) {
    auto found = std::find_if(generation->Channels().begin(),
        generation->Channels().end(), [&](auto const& channel) {
            return channel.name == name;
        });
    return found != generation->Channels().end() &&
        found->semantic == UsdGenDeviceChannelSemantic::Generic &&
        found->arity == arity;
}

std::vector<CudaNamedChannelPlane> One(CudaNamedChannelPlane plane) {
    std::vector<CudaNamedChannelPlane> result;
    result.push_back(std::move(plane));
    return result;
}

template <class T, size_t N>
bool Read(CudaNamedChannelLease const& lease, cudaStream_t stream,
          std::array<T, N>* values) {
    return lease && lease.Bytes().size == sizeof(*values) &&
        cudaMemcpyAsync(values->data(), lease.Bytes().data, sizeof(*values),
                        cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess;
}

} // namespace

int main() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 77;
    CHECK(ConfigureCudaExecutionResources(device, {size_t{8} << 20, 0}));
    UsdGenExecutionResourceDevice const key{
        usdGenTestUtils::Backend("cuda"), device};
    auto resources = FindUsdGenExecutionResourcePool(key);
    CHECK(resources);
    auto retirement = GetOrCreateUsdGenExecutionRetirementService(key, {32});
    CHECK(retirement);

    // Source-owned named data is part of the same immutable publication, not
    // a synthetic later generation. This is the final ownership form used by
    // execution jobs after their private uploads/transforms prove complete.
    std::array<float, 2> const sourceValues{{2.f, 6.f}};
    auto sourcePlane = Plane("sourceValue", UsdGenDeviceValueType::Float32,
        UsdGenDeviceDomain::Point, 2, 1, sourceValues);
    std::string localCreateReason;
    auto local = MakeSourceGeneration(
        Source(), 1, &localCreateReason, false, {}, {}, UINT64_MAX,
        One(std::move(sourcePlane)));
    if (!local) std::fprintf(stderr, "local generation: %s\n", localCreateReason.c_str());
    CHECK(local && local->Identity().generation == 1 &&
          HasChannel(local, "sourceValue", 1));
    {
        auto lease = AcquireNamedChannel(local, TfToken("sourceValue"), nullptr);
        std::array<float, 2> read{};
        CHECK(Read(lease, nullptr, &read) && read == sourceValues);
    }
    auto const localBytes = resources->Snapshot().usedBytes;
    auto badLocalPlane = Plane("badLocal", UsdGenDeviceValueType::Int32,
        UsdGenDeviceDomain::Primitive, 2, 1, std::array<int, 2>{{1, 2}});
    std::string localReason;
    CHECK(!MakeSourceGeneration(
        Source(), 2, &localReason, false, {}, {}, UINT64_MAX,
        One(std::move(badLocalPlane))));
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == localBytes &&
          localReason.find("incompatible local named channel") != std::string::npos);
    local.reset();
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == 0);

    auto base = BaseGeneration(1);
    CHECK(base);
    auto const baseSnapshot = resources->Snapshot();
    CHECK(KindBytes(baseSnapshot, UsdGenExecutionResourceKind::Pinned) != 0);

    // Validation happens transactionally. Duplicate names, wrong domain
    // cardinality and attempts to shadow a built-in geometry channel all
    // leave the published ledger unchanged after ordinary retirement drains.
    {
        auto a = Plane("duplicate", UsdGenDeviceValueType::Int32,
            UsdGenDeviceDomain::Primitive, 1, 1, std::array<int, 1>{1});
        auto b = Plane("duplicate", UsdGenDeviceValueType::Int32,
            UsdGenDeviceDomain::Primitive, 1, 1, std::array<int, 1>{2});
        std::vector<CudaNamedChannelPlane> duplicates;
        duplicates.push_back(std::move(a));
        duplicates.push_back(std::move(b));
        std::string reason;
        CHECK(!MakeNamedChannelRevisionGeneration(
            base, 2, std::move(duplicates), {}, &reason));
        CHECK(reason.find("incompatible generic plane") != std::string::npos);
    }
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == baseSnapshot.usedBytes);
    {
        auto badCount = Plane("badCount", UsdGenDeviceValueType::Float32,
            UsdGenDeviceDomain::Point, 1, 1, std::array<float, 1>{1.f});
        std::string reason;
        CHECK(!MakeNamedChannelRevisionGeneration(
            base, 2, One(std::move(badCount)), {}, &reason));
    }
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == baseSnapshot.usedBytes);
    {
        auto points = Plane("points", UsdGenDeviceValueType::Float32,
            UsdGenDeviceDomain::Primitive, 1, 1, std::array<float, 1>{1.f});
        std::string reason;
        CHECK(!MakeNamedChannelRevisionGeneration(
            base, 2, One(std::move(points)), {}, &reason));
        CHECK(reason.find("cannot replace geometry channel") != std::string::npos);
    }
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == baseSnapshot.usedBytes &&
          KindBytes(resources->Snapshot(), UsdGenExecutionResourceKind::Pinned) ==
              KindBytes(baseSnapshot, UsdGenExecutionResourceKind::Pinned));

    // R24 scalar-array channels use one element per curve and arity three.
    // Publication transfers only the new byte plane from Active to Pinned.
    std::array<int, 3> const guideIndices{4, 7, 9};
    auto guideIndex = Plane("guideIndex", UsdGenDeviceValueType::Int32,
        UsdGenDeviceDomain::Primitive, 1, 3, guideIndices);
    CHECK(guideIndex.bytes);
    auto const beforeIndex = resources->Snapshot();
    size_t const indexBytes = sizeof(guideIndices);
    CHECK(beforeIndex.usedBytes == baseSnapshot.usedBytes + indexBytes);
    auto withIndex = MakeNamedChannelRevisionGeneration(
        base, 3, One(std::move(guideIndex)));
    CHECK(withIndex && withIndex->Owner() != base->Owner() &&
          HasChannel(withIndex, "guideIndex", 3));
    auto const afterIndex = resources->Snapshot();
    CHECK(afterIndex.usedBytes == beforeIndex.usedBytes &&
          KindBytes(afterIndex, UsdGenExecutionResourceKind::Pinned) ==
              KindBytes(baseSnapshot, UsdGenExecutionResourceKind::Pinned) +
                  indexBytes &&
          KindBytes(afterIndex, UsdGenExecutionResourceKind::Active) +
                  indexBytes ==
              KindBytes(beforeIndex, UsdGenExecutionResourceKind::Active));

    // A second overlay shares the first plane and privately owns only its new
    // plane. Named lookup through the newest generation sees both values.
    std::array<float, 3> const guideWeights{.2f, .3f, .5f};
    auto guideWeight = Plane("guideWeight", UsdGenDeviceValueType::Float32,
        UsdGenDeviceDomain::Primitive, 1, 3, guideWeights);
    CHECK(guideWeight.bytes);
    auto withBoth = MakeNamedChannelRevisionGeneration(
        withIndex, 4, One(std::move(guideWeight)));
    CHECK(withBoth && HasChannel(withBoth, "guideIndex", 3) &&
          HasChannel(withBoth, "guideWeight", 3));
    auto const afterBoth = resources->Snapshot();
    CHECK(KindBytes(afterBoth, UsdGenExecutionResourceKind::Pinned) ==
          KindBytes(baseSnapshot, UsdGenExecutionResourceKind::Pinned) +
              indexBytes + sizeof(guideWeights));

    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    {
        auto indexLease = AcquireNamedChannel(withBoth, TfToken("guideIndex"), stream);
        auto weightLease = AcquireNamedChannel(withBoth, TfToken("guideWeight"), stream);
        CHECK(indexLease && indexLease.Metadata() &&
              indexLease.Metadata()->type == UsdGenDeviceValueType::Int32 &&
              weightLease && weightLease.Metadata() &&
              weightLease.Metadata()->type == UsdGenDeviceValueType::Float32 &&
              !AcquireNamedChannel(withBoth, TfToken("missing"), stream));
        std::array<int, 3> readIndices{};
        std::array<float, 3> readWeights{};
        CHECK(Read(indexLease, stream, &readIndices));
        CHECK(Read(weightLease, stream, &readWeights));
        CHECK(readIndices == guideIndices);
        CHECK(readWeights == guideWeights);
    }
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);

    base.reset();
    withIndex.reset();
    withBoth.reset();
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == 0 &&
          KindBytes(resources->Snapshot(), UsdGenExecutionResourceKind::Pinned) == 0);
    std::puts("testUsdGenCudaNamedChannels: PASS");
    return 0;
}
