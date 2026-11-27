#include "usdGenCurves/cuda/topology.h"
#include "usdGenCuda/backendId.h"
#include "gpu/deviceBuffer.h"
#include "usdGen/api/ext/device/executionResources.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <optional>

using namespace usdGen;
using namespace usdGen::gpu;

namespace {

struct NativeProof {
    std::atomic<int> calls{0};
    std::atomic<cudaError_t> status{cudaErrorUnknown};

    void Reset() {
        calls.store(0, std::memory_order_relaxed);
        status.store(cudaErrorUnknown, std::memory_order_relaxed);
    }
};

void CUDART_CB RecordNativeProof(cudaStream_t, cudaError_t status,
                                 void* userData) {
    auto* const proof = static_cast<NativeProof*>(userData);
    proof->status.store(status, std::memory_order_relaxed);
    proof->calls.fetch_add(1, std::memory_order_relaxed);
}

bool IsProven(NativeProof const& proof) {
    return proof.calls.load(std::memory_order_relaxed) == 1 &&
           proof.status.load(std::memory_order_relaxed) == cudaSuccess;
}

} // namespace

int main() {
    int failures = 0;
    auto check = [&](bool condition, const char* message) {
        if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
    };
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream) == cudaSuccess, "create stream");
    if (failures) return failures;

    DeviceBuffer<uint32_t> offsetsA, offsetsB;
    DeviceBuffer<uint64_t> idsA, idsB;
    check(offsetsA.reset(3) == cudaSuccess && offsetsB.reset(3) == cudaSuccess &&
          idsA.reset(2) == cudaSuccess && idsB.reset(2) == cudaSuccess, "allocate topology");
    uint32_t offsets[]{0, 2, 5};
    uint64_t ids[]{17, 42};
    check(cudaMemcpy(offsetsA.data(), offsets, sizeof(offsets), cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(offsetsB.data(), offsets, sizeof(offsets), cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(idsA.data(), ids, sizeof(ids), cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(idsB.data(), ids, sizeof(ids), cudaMemcpyHostToDevice) == cudaSuccess, "upload topology");
    DeviceCurveGeometryView a;
    a.curveCount = 2; a.pointCount = 5;
    a.curveOffsets = {offsetsA.data(), offsetsA.size()};
    a.stableIds = {idsA.data(), idsA.size()};
    DeviceCurveGeometryView b = a;
    b.curveOffsets = {offsetsB.data(), offsetsB.size()};
    b.stableIds = {idsB.data(), idsB.size()};

    // The synchronous comparator consumes only its scalar result from the
    // caller's precharged balance.  Use the pool baseline (which includes all
    // topology buffers already allocated above) rather than assuming an empty
    // ledger.  The comparator's local result recycles its credit before it
    // returns; a held manual child then makes the Pending->Scratch transfer
    // and recycle observable without depending on an in-call synchronization.
    int resourceDevice = -1;
    check(cudaGetDevice(&resourceDevice) == cudaSuccess, "get resource device");
    auto resourcePool = FindUsdGenExecutionResourcePool(
        {usdGen::UsdGenCudaBackendId(), resourceDevice});
    check(bool(resourcePool), "find CUDA resource pool");
    auto const baseline = resourcePool ? resourcePool->Snapshot() :
        UsdGenExecutionResourceSnapshot{};
    constexpr size_t resultBytes = sizeof(CurveTopologyCompareResult);
    auto reservation = resourcePool ? resourcePool->TryReserveMemory(resultBytes) :
        std::optional<UsdGenExecutionMemoryReservation>{};
    check(bool(reservation), "reserve comparator result memory");
    if (reservation) {
        auto const charged = resourcePool->Snapshot();
        check(charged.usedBytes == baseline.usedBytes + resultBytes &&
              charged.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)] ==
                  baseline.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)] + resultBytes,
              "reservation charges Pending without changing baseline categories");
        bool reservedEqual = false;
        check(CompareCurveTopology(a, b, stream, &reservedEqual, &*reservation) == cudaSuccess &&
              reservedEqual,
              "reservation-backed topology comparison");
        auto const afterCompare = resourcePool->Snapshot();
        check(afterCompare.usedBytes == charged.usedBytes &&
              reservation->RemainingBytes() == resultBytes &&
              afterCompare.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)] ==
                  charged.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)],
              "comparator consume preserves total and recycles Pending credit");

        auto child = reservation->Consume(resultBytes,
                                          UsdGenExecutionResourceKind::Scratch);
        check(bool(child), "consume comparator reservation child");
        if (child) {
            auto const transferred = resourcePool->Snapshot();
            check(transferred.usedBytes == charged.usedBytes &&
                  transferred.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)] ==
                      charged.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)] - resultBytes &&
                  transferred.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Scratch)] ==
                      baseline.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Scratch)] + resultBytes,
                  "reservation child transfers category without total charge");
            child->Release();
            auto const recycled = resourcePool->Snapshot();
            check(recycled.usedBytes == charged.usedBytes &&
                  recycled.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)] ==
                      charged.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)] &&
                  recycled.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Scratch)] ==
                      baseline.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Scratch)],
                  "reservation child release recycles Pending credit");
        }
        reservation->Release();
        auto const restored = resourcePool->Snapshot();
        check(restored.usedBytes == baseline.usedBytes &&
              restored.byKind == baseline.byKind,
              "closing comparator reservation restores baseline");
    }
    bool equal = false;
    check(CompareCurveTopology(a, b, stream, &equal) == cudaSuccess && equal, "equal topology");

    DeviceBuffer<float3> pointsA, pointsB;
    check(pointsA.reset(5) == cudaSuccess && pointsB.reset(5) == cudaSuccess,
          "allocate ignored point channels");
    float3 positionsA[]{make_float3(0.f, 0.f, 0.f), make_float3(1.f, 0.f, 0.f),
                        make_float3(2.f, 0.f, 0.f), make_float3(3.f, 0.f, 0.f),
                        make_float3(4.f, 0.f, 0.f)};
    float3 positionsB[]{make_float3(0.f, 0.f, 0.f), make_float3(1.f, 0.f, 0.f),
                        make_float3(2.f, 0.f, 0.f), make_float3(3.f, 0.f, 0.f),
                        make_float3(400.f, 0.f, 0.f)};
    check(cudaMemcpy(pointsA.data(), positionsA, sizeof(positionsA), cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(pointsB.data(), positionsB, sizeof(positionsB), cudaMemcpyHostToDevice) == cudaSuccess,
          "upload different ignored point values");
    a.points = {pointsA.data(), pointsA.size()};
    b.points = {pointsB.data(), pointsB.size()};
    // Layout identity intentionally ignores point positions.
    equal = false;
    check(CompareCurveTopology(a, b, stream, &equal) == cudaSuccess && equal,
          "point values do not affect topology");

    ids[1] = 99;
    check(cudaMemcpy(idsB.data() + 1, ids + 1, sizeof(uint64_t), cudaMemcpyHostToDevice) == cudaSuccess,
          "change id");
    equal = true;
    check(CompareCurveTopology(a, b, stream, &equal) == cudaSuccess && !equal, "different id");

    uint64_t permuted[]{42,17};
    check(cudaMemcpy(idsB.data(), permuted, sizeof(permuted), cudaMemcpyHostToDevice) == cudaSuccess,
          "permute same stable-id set");
    equal = true;
    check(CompareCurveTopology(a, b, stream, &equal) == cudaSuccess && !equal,
          "stable-id order is part of topology");

    ids[1] = 42;
    check(cudaMemcpy(idsB.data(), ids, sizeof(ids), cudaMemcpyHostToDevice) == cudaSuccess,
          "restore id");
    uint32_t redistributed[]{0, 1, 5};
    check(cudaMemcpy(offsetsB.data(), redistributed, sizeof(redistributed), cudaMemcpyHostToDevice) == cudaSuccess,
          "change offset layout");
    equal = true;
    check(CompareCurveTopology(a, b, stream, &equal) == cudaSuccess && !equal,
          "different offset layout");
    check(cudaMemcpy(offsetsB.data(), offsets, sizeof(offsets), cudaMemcpyHostToDevice) == cudaSuccess,
          "restore offset layout");

    b.curveCount = 1;
    equal = false;
    check(CompareCurveTopology(a, b, stream, &equal) == cudaErrorInvalidValue && !equal,
          "unequal malformed shape is validated");
    b.curveCount = 2;

    DeviceBuffer<uint32_t> bad;
    check(bad.reset(3) == cudaSuccess, "allocate bad offsets");
    uint32_t malformed[]{0, 3, 2};
    check(cudaMemcpy(bad.data(), malformed, sizeof(malformed), cudaMemcpyHostToDevice) == cudaSuccess,
          "upload malformed offsets");
    b.curveOffsets = {bad.data(), bad.size()};
    equal = true;
    check(CompareCurveTopology(a, b, stream, &equal) == cudaErrorInvalidValue && equal,
          "malformed topology preserves result");

    b = a;
    b.curveOffsets = {offsetsB.data(), 2};
    equal = false;
    check(CompareCurveTopology(a, b, stream, &equal) == cudaErrorInvalidValue && !equal,
          "offset shape validation preserves result");
    b = a;
    b.stableIds = {nullptr, 2};
    equal = true;
    check(CompareCurveTopology(a, b, stream, &equal) == cudaErrorInvalidValue && equal,
          "null topology pointer validation preserves result");

    DeviceBuffer<uint32_t> emptyOffsetsA, emptyOffsetsB;
    DeviceBuffer<uint64_t> emptyIdsA, emptyIdsB;
    check(emptyOffsetsA.reset(1) == cudaSuccess && emptyOffsetsB.reset(1) == cudaSuccess &&
          emptyIdsA.reset(0) == cudaSuccess && emptyIdsB.reset(0) == cudaSuccess, "allocate empty topology");
    uint32_t zero = 0;
    check(cudaMemcpy(emptyOffsetsA.data(), &zero, sizeof(zero), cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(emptyOffsetsB.data(), &zero, sizeof(zero), cudaMemcpyHostToDevice) == cudaSuccess,
          "upload empty offsets");
    DeviceCurveGeometryView emptyA, emptyB;
    emptyA.curveOffsets = {emptyOffsetsA.data(), emptyOffsetsA.size()};
    emptyA.stableIds = {emptyIdsA.data(), emptyIdsA.size()};
    emptyB.curveOffsets = {emptyOffsetsB.data(), emptyOffsetsB.size()};
    emptyB.stableIds = {emptyIdsB.data(), emptyIdsB.size()};
    equal = false;
    check(CompareCurveTopology(emptyA, emptyB, stream, &equal) == cudaSuccess && equal,
          "empty topology");
    equal = true;
    check(CompareCurveTopology(a, emptyB, stream, &equal) == cudaSuccess && !equal,
          "valid different curve counts");

    // Fresh comparison owns only its scalar device/pinned result.  The parent
    // supplies the terminal-proof boundary before host-only commit; positions
    // remain intentionally irrelevant to topology identity.
    // Several legacy malformed-view checks above assign b=a.  Rebuild the
    // independent peer views before the fresh checks so their mutations
    // target idsB/offsetsB rather than the authoritative a buffers.
    b = a;
    b.points = {pointsB.data(), pointsB.size()};
    b.curveOffsets = {offsetsB.data(), offsetsB.size()};
    b.stableIds = {idsB.data(), idsB.size()};
    CudaCurveTopologyCompare fresh;
    bool freshEqual = false;
    check(fresh.BeginFresh(a, a, stream) == cudaSuccess && fresh.HasUnprovenWork(),
          "fresh begin equal");
    check(fresh.CommitFreshFinish(&freshEqual) == cudaErrorInvalidValue && !freshEqual &&
          fresh.BeginFresh(a, a, stream) == cudaErrorInvalidValue,
          "fresh early commit and phase guards");
    NativeProof proof;
    check(fresh.EnqueueFreshStatus(stream) == cudaSuccess &&
          cudaStreamAddCallback(stream, RecordNativeProof, &proof, 0) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess && IsProven(proof) &&
          fresh.CommitFreshFinish(&freshEqual) == cudaSuccess && freshEqual && !fresh.HasUnprovenWork(),
          "fresh pinned result becomes visible only after terminal proof");

    // Point channels do not participate in layout identity in either API.
    proof.Reset();
    check(fresh.BeginFresh(a, b, stream) == cudaSuccess &&
          fresh.EnqueueFreshStatus(stream) == cudaSuccess &&
          cudaStreamAddCallback(stream, RecordNativeProof, &proof, 0) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess && IsProven(proof) &&
          fresh.CommitFreshFinish(&freshEqual) == cudaSuccess && freshEqual,
          "fresh ignores different point values");

    ids[1] = 99;
    check(cudaMemcpy(idsB.data() + 1, ids + 1, sizeof(uint64_t), cudaMemcpyHostToDevice) == cudaSuccess,
          "fresh change id");
    freshEqual = true;
    proof.Reset();
    check(fresh.BeginFresh(a, b, stream) == cudaSuccess &&
          fresh.EnqueueFreshStatus(stream) == cudaSuccess &&
          cudaStreamAddCallback(stream, RecordNativeProof, &proof, 0) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess && IsProven(proof) &&
          fresh.CommitFreshFinish(&freshEqual) == cudaSuccess && !freshEqual,
          "fresh unequal IDs are valid false");
    ids[1] = 42;
    check(cudaMemcpy(idsB.data() + 1, ids + 1, sizeof(uint64_t), cudaMemcpyHostToDevice) == cudaSuccess,
          "fresh restore id");
    check(cudaMemcpy(offsetsB.data(), redistributed, sizeof(redistributed), cudaMemcpyHostToDevice) == cudaSuccess,
          "fresh change offsets");
    freshEqual = true;
    proof.Reset();
    check(fresh.BeginFresh(a, b, stream) == cudaSuccess &&
          fresh.EnqueueFreshStatus(stream) == cudaSuccess &&
          cudaStreamAddCallback(stream, RecordNativeProof, &proof, 0) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess && IsProven(proof) &&
          fresh.CommitFreshFinish(&freshEqual) == cudaSuccess && !freshEqual,
          "fresh unequal offsets are valid false");
    check(cudaMemcpy(offsetsB.data(), offsets, sizeof(offsets), cudaMemcpyHostToDevice) == cudaSuccess,
          "fresh restore offsets");

    // A malformed device layout is a proved semantic rejection: it does not
    // mutate the caller result and leaves this comparator reusable.
    b.curveOffsets = {bad.data(), bad.size()};
    freshEqual = true;
    proof.Reset();
    check(fresh.BeginFresh(a, b, stream) == cudaSuccess &&
          fresh.EnqueueFreshStatus(stream) == cudaSuccess &&
          cudaStreamAddCallback(stream, RecordNativeProof, &proof, 0) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess && IsProven(proof) &&
          fresh.CommitFreshFinish(&freshEqual) == cudaErrorInvalidValue && freshEqual &&
          !fresh.HasUnprovenWork(), "fresh malformed offsets preserve result");
    b = a;
    proof.Reset();
    check(fresh.BeginFresh(emptyA, emptyB, stream) == cudaSuccess &&
          fresh.EnqueueFreshStatus(stream) == cudaSuccess &&
          cudaStreamAddCallback(stream, RecordNativeProof, &proof, 0) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess && IsProven(proof) &&
          fresh.CommitFreshFinish(&freshEqual) == cudaSuccess && freshEqual,
          "fresh canonical empty topology and reuse");
    freshEqual = true;
    proof.Reset();
    check(fresh.BeginFresh(a, emptyB, stream) == cudaSuccess &&
          fresh.EnqueueFreshStatus(stream) == cudaSuccess &&
          cudaStreamAddCallback(stream, RecordNativeProof, &proof, 0) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess && IsProven(proof) &&
          fresh.CommitFreshFinish(&freshEqual) == cudaSuccess && !freshEqual,
          "fresh valid different counts");

    // Capture is rejected before fresh device/stream validation or allocation.
    CudaCurveTopologyCompare capture;
    cudaGraph_t graph = nullptr;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess &&
          capture.BeginFresh(a, b, stream) == cudaErrorInvalidValue &&
          cudaStreamEndCapture(stream, &graph) == cudaSuccess, "fresh capture rejection");
    if (graph) check(cudaGraphDestroy(graph) == cudaSuccess, "destroy capture graph");
    cudaStreamDestroy(stream);
    return failures;
}
