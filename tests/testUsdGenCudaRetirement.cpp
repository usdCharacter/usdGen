#include "usdGen/api/ext/device/executionResources.h"
#include "testBackend.h"
#include "usdGen/api/ext/device/executionRetirement.h"
#include "usdGenCuda/gpu/cudaRetirement.h"
#include "usdGenCurves/cuda/curveCompaction.h"
#include "usdGenCuda/gpu/deviceBuffer.h"
#include "usdGenCuda/gpu/deviceResources.h"
#include "usdGenCurves/cuda/generation.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (false)

static std::unique_ptr<CudaCurveSource> CompletedSource()
{
    auto source = std::make_unique<CudaCurveSource>();
    int32_t const counts[] = {2};
    float3 const points[] = {make_float3(0, 0, 0), make_float3(1, 0, 0)};
    CurveSourceInput input;
    input.curveVertexCounts = {counts, 1};
    input.points = {points, 2};
    input.restPoints = {points, 2};
    input.fallbackWidth = .1f;
    if (source->Set(input, nullptr) != CurveSourceStatus::Ok ||
        source->Finish(nullptr) != CurveSourceStatus::Ok) return {};
    return source;
}

static std::shared_ptr<const UsdGenDeviceGeneration> MakeGeneration(uint64_t id)
{
    auto source = CompletedSource();
    if (!source) return {};
    std::string reason;
    return MakeSourceGeneration(std::move(source), id, &reason);
}

struct HostGate {
    std::atomic<bool> entered{false};
    std::atomic<bool> open{false};
    std::atomic<bool> timedOut{false};
};

static void CUDART_CB HoldStream(void* data)
{
    auto& gate = *static_cast<HostGate*>(data);
    gate.entered.store(true, std::memory_order_release);
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!gate.open.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            gate.timedOut.store(true, std::memory_order_release);
            return;
        }
        std::this_thread::yield();
    }
}

static bool WaitForGate(HostGate const& gate)
{
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!gate.entered.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

static size_t KindBytes(UsdGenExecutionResourceSnapshot const& snapshot,
                        UsdGenExecutionResourceKind kind)
{
    return snapshot.byKind[static_cast<size_t>(kind)];
}

struct GateCleanup {
    HostGate& gate;
    cudaStream_t stream;
    ~GateCleanup() {
        gate.open.store(true, std::memory_order_release);
        if (stream) cudaStreamSynchronize(stream);
    }
    void Disarm() noexcept { stream = nullptr; }
};

int main(int argc, char** argv)
{
    // Reclassification is a ledger transfer, not a second charge.  Keep this
    // backend-neutral proof local so it remains deterministic even when CUDA
    // allocation sizes are rounded by the driver.
    {
        UsdGenExecutionResourcePool ledger(4096);
        auto permit = ledger.TryReserve(257, UsdGenExecutionResourceKind::Active);
        CHECK(permit);
        auto before = ledger.Snapshot();
        permit->Reclassify(UsdGenExecutionResourceKind::Pinned);
        auto after = ledger.Snapshot();
        CHECK(after.usedBytes == before.usedBytes);
        CHECK(KindBytes(after, UsdGenExecutionResourceKind::Active) + 257 ==
              KindBytes(before, UsdGenExecutionResourceKind::Active));
        CHECK(KindBytes(after, UsdGenExecutionResourceKind::Pinned) ==
              KindBytes(before, UsdGenExecutionResourceKind::Pinned) + 257);
        permit->Release();
        CHECK(ledger.Snapshot().usedBytes == 0);
    }

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 77;
    CHECK(ConfigureCudaExecutionResources(device, {size_t{8} << 20, 0}));
    UsdGenExecutionResourceDevice const key{usdGenTestUtils::Backend("cuda"), device};
    auto resources = FindUsdGenExecutionResourcePool(key);
    CHECK(resources);
    // One published owner plus one consumer is enough to force a factory
    // admission rejection while the consumer completion is intentionally held.
    auto retirement = GetOrCreateUsdGenExecutionRetirementService(key, {2});
    CHECK(retirement);

    // This mode deliberately leaves the normal owner-retirement work queued
    // at process return.  It covers the automatic atexit backend close with
    // a real CUDA generation, without an explicit Drain or test-only sync
    // that could hide shutdown ordering.
    if (argc == 2 && std::string(argv[1]) == "--exit-retire") {
        auto exitGeneration = MakeGeneration(100);
        CHECK(exitGeneration);
        exitGeneration.reset();
        return 0;
    }
    if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--exit-retire]\n", argv[0]);
        return 2;
    }

    auto generation = MakeGeneration(1);
    CHECK(generation);
    std::weak_ptr<const UsdGenDeviceOwner> owner = generation->Owner();
    auto publishedSnapshot = resources->Snapshot();
    size_t const publishedBytes = publishedSnapshot.usedBytes;
    CHECK(publishedBytes != 0);
    // A published source owns its allocation as Pinned.  Its publication
    // must not leave a new Active charge behind.
    CHECK(KindBytes(publishedSnapshot, UsdGenExecutionResourceKind::Pinned) != 0);

    cudaStream_t consumer = nullptr;
    CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking) == cudaSuccess);
    HostGate gate;
    auto lease = AcquireGeometry(generation, consumer);
    CHECK(lease && lease.Geometry().pointCount == 2);

    // The completed factory input owns charged DeviceBuffers.  With both
    // retirement slots occupied it is rejected, and ordinary factory cleanup
    // must return its charge rather than permanently quarantining it.
    DeviceBuffer<int> delayedWork;
    CHECK(delayedWork.reset(1) == cudaSuccess);
    size_t const delayedWorkBytes = resources->Snapshot().usedBytes;
    CHECK(delayedWorkBytes > publishedBytes);
    size_t const delayedOnlyBytes = delayedWorkBytes - publishedBytes;
    auto rejectedSource = CompletedSource();
    CHECK(rejectedSource);
    size_t const withRejectedInput = resources->Snapshot().usedBytes;
    CHECK(withRejectedInput > delayedWorkBytes);
    std::string reason;
    CHECK(!MakeSourceGeneration(std::move(rejectedSource), 2, &reason));
    CHECK(reason.find("retirement admission") != std::string::npos);
    auto afterRejected = resources->Snapshot();
    CHECK(afterRejected.usedBytes == delayedWorkBytes);
    CHECK(KindBytes(afterRejected, UsdGenExecutionResourceKind::Pinned) ==
          KindBytes(publishedSnapshot, UsdGenExecutionResourceKind::Pinned));

    // CUDA treats a wait on an unrecorded event as a no-op.  This bounded
    // host callback is actual queued CUDA stream work; the following device
    // memset and terminal event/callback cannot complete until it opens.
    // Keep this guard after all CUDA-owning locals: on a failing assertion it
    // opens/synchronizes the held stream before their destructors can free.
    GateCleanup gateCleanup{gate, consumer};
    CHECK(cudaLaunchHostFunc(consumer, HoldStream, &gate) == cudaSuccess);
    CHECK(cudaMemsetAsync(delayedWork.data(), 0, sizeof(int), consumer) == cudaSuccess);
    generation.reset();
    lease = {};
    if (!WaitForGate(gate)) {
        std::fprintf(stderr, "FAIL: CUDA retirement gate did not begin within deadline\n");
        return 1;
    }
    CHECK(!gate.timedOut.load(std::memory_order_acquire));
    CHECK(!owner.expired());
    auto held = resources->Snapshot();
    CHECK(held.usedBytes == delayedWorkBytes);
    CHECK(KindBytes(held, UsdGenExecutionResourceKind::Pinned) ==
          KindBytes(publishedSnapshot, UsdGenExecutionResourceKind::Pinned));

    gate.open.store(true, std::memory_order_release);
    CHECK(cudaStreamSynchronize(consumer) == cudaSuccess);
    // The callback may already have signalled the service.  Its payload does
    // not retain a native stream handle, so destruction before Drain is safe.
    CHECK(cudaStreamDestroy(consumer) == cudaSuccess);
    consumer = nullptr;
    gateCleanup.Disarm();
    retirement->Drain();
    CHECK(owner.expired());
    auto afterRetirement = resources->Snapshot();
    CHECK(afterRetirement.usedBytes == delayedOnlyBytes);
    CHECK(KindBytes(afterRetirement, UsdGenExecutionResourceKind::Pinned) == 0);
    delayedWork = {};
    CHECK(resources->Snapshot().usedBytes == 0);

    // A point revision adds one private published owner.  The shared source
    // remains charged exactly once, while the new point storage transfers
    // from Active to Pinned without changing the total.
    auto base = MakeGeneration(5);
    CHECK(base);
    auto baseSnapshot = resources->Snapshot();
    auto baseOwner = base->Owner();
    auto revisedPoints = std::make_unique<DeviceBuffer<float3>>();
    CHECK(revisedPoints->reset(base->Geometry().pointCount) == cudaSuccess);
    auto beforeRevision = resources->Snapshot();
    size_t const pointBytes = base->Geometry().pointCount * sizeof(float3);
    CHECK(beforeRevision.usedBytes == baseSnapshot.usedBytes + pointBytes);
    std::string revisionReason;
    auto revision = MakePointRevisionGeneration(base, 6, std::move(revisedPoints), {},
                                                &revisionReason);
    CHECK(revision);
    CHECK(revision->Owner() != baseOwner);
    auto afterRevision = resources->Snapshot();
    CHECK(afterRevision.usedBytes == beforeRevision.usedBytes);
    CHECK(KindBytes(afterRevision, UsdGenExecutionResourceKind::Pinned) ==
          KindBytes(baseSnapshot, UsdGenExecutionResourceKind::Pinned) + pointBytes);

    // Metadata is an immutable wrapper around the same owner and therefore
    // has no resource-category or lifetime effect.  Invalid metadata must
    // likewise fail before changing the ledger.
    std::string metadataReason;
    std::vector<UsdGenDeviceTileMetadata> tile{{0, 0, 1, 0, 2}};
    auto withMetadata = WithTileMetadata(revision, tile, &metadataReason);
    CHECK(withMetadata && withMetadata->Owner() == revision->Owner());
    auto afterMetadata = resources->Snapshot();
    CHECK(afterMetadata.usedBytes == afterRevision.usedBytes);
    CHECK(KindBytes(afterMetadata, UsdGenExecutionResourceKind::Pinned) ==
          KindBytes(afterRevision, UsdGenExecutionResourceKind::Pinned));
    std::vector<UsdGenDeviceTileMetadata> invalidTile{{0, 0, 1, 1, 2}};
    CHECK(!WithTileMetadata(revision, invalidTile, &metadataReason));
    auto afterInvalidMetadata = resources->Snapshot();
    CHECK(afterInvalidMetadata.usedBytes == afterMetadata.usedBytes);
    CHECK(KindBytes(afterInvalidMetadata, UsdGenExecutionResourceKind::Pinned) ==
          KindBytes(afterMetadata, UsdGenExecutionResourceKind::Pinned));

    // Keep the base alive while the private revision is retired, then prove
    // the shared source charge is released only after the final owner drain.
    withMetadata.reset();
    revision.reset();
    base.reset();
    baseOwner.reset();
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == 0);

    // Width revisions are private published storage too.  The complete
    // source+width candidate is charged before publication, then its entire
    // candidate charge is reclassified without changing the total.
    auto widthSource = CompletedSource();
    CHECK(widthSource);
    DeviceBuffer<float> widthRevision;
    CHECK(widthRevision.reset(2) == cudaSuccess);
    auto beforeWidthPublish = resources->Snapshot();
    size_t const widthCandidateBytes = beforeWidthPublish.usedBytes;
    size_t const widthCandidateActive =
        KindBytes(beforeWidthPublish, UsdGenExecutionResourceKind::Active);
    std::string widthReason;
    auto widthGeneration = MakeSourceGeneration(std::move(widthSource), 7,
                                                &widthReason, false,
                                                std::make_unique<DeviceBuffer<float>>(
                                                    std::move(widthRevision)));
    CHECK(widthGeneration);
    auto afterWidthPublish = resources->Snapshot();
    CHECK(afterWidthPublish.usedBytes == widthCandidateBytes);
    CHECK(KindBytes(afterWidthPublish, UsdGenExecutionResourceKind::Pinned) ==
          KindBytes(beforeWidthPublish, UsdGenExecutionResourceKind::Pinned) +
              widthCandidateBytes);
    CHECK(KindBytes(afterWidthPublish, UsdGenExecutionResourceKind::Active) ==
          widthCandidateActive - widthCandidateBytes);
    widthGeneration.reset();
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == 0);

    // A resampled publication retains both the authored source and the
    // resampler's completed output.  Its synchronous status buffer is also
    // retained by the owner, so the pre-publication Active/Scratch ledger is
    // the complete candidate charge to transfer.
    cudaStream_t transformStream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&transformStream, cudaStreamNonBlocking) == cudaSuccess);
    auto resampleSource = CompletedSource();
    CHECK(resampleSource);
    auto resampled = std::make_unique<CudaCurveResample>();
    CHECK(resampled->Apply(resampleSource->view(), resampleSource->hairT(),
                           resampleSource->rootPrim(), resampleSource->rootUV(),
                           2, transformStream) == CurveResampleStatus::Ok);
    CHECK(resampled->Finish(transformStream) == CurveResampleStatus::Ok);
    auto beforeResampledPublish = resources->Snapshot();
    size_t const resampledActive =
        KindBytes(beforeResampledPublish, UsdGenExecutionResourceKind::Active);
    size_t const resampledScratch =
        KindBytes(beforeResampledPublish, UsdGenExecutionResourceKind::Scratch);
    CHECK(resampledScratch != 0);
    std::string resampledReason;
    auto resampledGeneration = MakeResampledGeneration(
        std::move(resampleSource), std::move(resampled), 8, &resampledReason);
    CHECK(resampledGeneration);
    auto afterResampledPublish = resources->Snapshot();
    CHECK(afterResampledPublish.usedBytes == beforeResampledPublish.usedBytes);
    CHECK(afterResampledPublish.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)] ==
          beforeResampledPublish.byKind[static_cast<size_t>(UsdGenExecutionResourceKind::Pending)]);
    CHECK(KindBytes(afterResampledPublish, UsdGenExecutionResourceKind::Pinned) ==
          KindBytes(beforeResampledPublish, UsdGenExecutionResourceKind::Pinned) +
              resampledActive + resampledScratch);
    CHECK(KindBytes(afterResampledPublish, UsdGenExecutionResourceKind::Active) ==
          KindBytes(beforeResampledPublish, UsdGenExecutionResourceKind::Active) -
              resampledActive);
    CHECK(KindBytes(afterResampledPublish, UsdGenExecutionResourceKind::Scratch) ==
          KindBytes(beforeResampledPublish, UsdGenExecutionResourceKind::Scratch) -
              resampledScratch);
    resampledGeneration.reset();
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == 0);

    // Compaction retains its completed output, status/count readback, and
    // CUB scan workspace.  The keep mask is an external input and remains
    // Active; only the compactor's pre-factory Active/Scratch delta moves to
    // Pinned, proving the scan/status workspace is neither leaked nor
    // double-charged.
    auto compactSource = CompletedSource();
    CHECK(compactSource);
    auto compactKeep = std::make_unique<DeviceBuffer<unsigned char>>();
    CHECK(compactKeep->reset(1) == cudaSuccess);
    unsigned char const keep = 1;
    CHECK(cudaMemcpy(compactKeep->data(), &keep, sizeof(keep), cudaMemcpyHostToDevice) ==
          cudaSuccess);
    auto beforeCompactor = resources->Snapshot();
    auto compacted = std::make_unique<CudaCurveCompaction>();
    CHECK(compacted->Apply(compactSource->view(), compactSource->hairT(),
                           compactSource->rootPrim(), compactSource->rootUV(),
                           {compactKeep->data(), 1}, transformStream) ==
          CurveCompactionStatus::Ok);
    CHECK(compacted->Finish(transformStream) == CurveCompactionStatus::Ok);
    auto beforeCompactedSourceRelease = resources->Snapshot();
    compactSource.reset();
    auto beforeCompactedPublish = resources->Snapshot();
    size_t const compactedActive =
        KindBytes(beforeCompactedSourceRelease, UsdGenExecutionResourceKind::Active) -
        KindBytes(beforeCompactor, UsdGenExecutionResourceKind::Active);
    size_t const compactedScratch =
        KindBytes(beforeCompactedSourceRelease, UsdGenExecutionResourceKind::Scratch) -
        KindBytes(beforeCompactor, UsdGenExecutionResourceKind::Scratch);
    CHECK(compactedScratch != 0);
    std::string compactedReason;
    auto compactedGeneration = MakeCompactedGeneration(
        std::move(compacted), 9, &compactedReason);
    CHECK(compactedGeneration);
    auto afterCompactedPublish = resources->Snapshot();
    CHECK(afterCompactedPublish.usedBytes == beforeCompactedPublish.usedBytes);
    CHECK(KindBytes(afterCompactedPublish, UsdGenExecutionResourceKind::Pinned) ==
          KindBytes(beforeCompactedPublish, UsdGenExecutionResourceKind::Pinned) +
              compactedActive + compactedScratch);
    CHECK(KindBytes(afterCompactedPublish, UsdGenExecutionResourceKind::Active) ==
          KindBytes(beforeCompactedPublish, UsdGenExecutionResourceKind::Active) -
              compactedActive);
    CHECK(KindBytes(afterCompactedPublish, UsdGenExecutionResourceKind::Scratch) ==
          KindBytes(beforeCompactedPublish, UsdGenExecutionResourceKind::Scratch) -
              compactedScratch);
    compactedGeneration.reset();
    retirement->Drain();
    CHECK(KindBytes(resources->Snapshot(), UsdGenExecutionResourceKind::Pinned) == 0);
    compactKeep.reset();
    CHECK(resources->Snapshot().usedBytes == 0);
    CHECK(cudaStreamDestroy(transformStream) == cudaSuccess);

    // cudaStreamAddCallback is capture-incompatible.  Acquire rejects before
    // it records an event or changes the capture, leaving the caller's graph
    // usable for ordinary captured work.
    auto capturedGeneration = MakeGeneration(3);
    CHECK(capturedGeneration);
    DeviceBuffer<int> capturedWork;
    CHECK(capturedWork.reset(1) == cudaSuccess);
    // Control: this runtime/device accepts the same ordinary captured work
    // when no generation acquisition query is made.
    cudaStream_t control = nullptr;
    cudaGraph_t controlGraph = nullptr;
    CHECK(cudaStreamCreateWithFlags(&control, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(cudaStreamBeginCapture(control, cudaStreamCaptureModeGlobal) == cudaSuccess);
    CHECK(cudaMemsetAsync(capturedWork.data(), 0, sizeof(int), control) == cudaSuccess);
    CHECK(cudaStreamEndCapture(control, &controlGraph) == cudaSuccess && controlGraph);
    CHECK(cudaGraphDestroy(controlGraph) == cudaSuccess);
    CHECK(cudaStreamDestroy(control) == cudaSuccess);
    cudaStream_t capture = nullptr;
    cudaGraph_t graph = nullptr;
    CHECK(cudaStreamCreateWithFlags(&capture, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(cudaStreamBeginCapture(capture, cudaStreamCaptureModeGlobal) == cudaSuccess);
    CHECK(cudaMemsetAsync(capturedWork.data(), 0, sizeof(int), capture) == cudaSuccess);
    cudaStreamCaptureStatus before = cudaStreamCaptureStatusNone;
    cudaError_t const beforeStatus = cudaStreamIsCapturing(capture, &before);
    CHECK(!AcquireGeometry(capturedGeneration, capture));
    cudaStreamCaptureStatus after = cudaStreamCaptureStatusNone;
    cudaError_t const afterStatus = cudaStreamIsCapturing(capture, &after);
    cudaError_t const endStatus = cudaStreamEndCapture(capture, &graph);
    if (beforeStatus != cudaSuccess || before != cudaStreamCaptureStatusActive ||
        afterStatus != cudaSuccess || after != cudaStreamCaptureStatusActive ||
        endStatus != cudaSuccess || !graph) {
        std::fprintf(stderr,
            "capture diagnostics: before=%s/%d after=%s/%d end=%s\n",
            cudaGetErrorString(beforeStatus), int(before),
            cudaGetErrorString(afterStatus), int(after), cudaGetErrorString(endStatus));
    }
    CHECK(beforeStatus == cudaSuccess && before == cudaStreamCaptureStatusActive);
    CHECK(afterStatus == cudaSuccess && after == cudaStreamCaptureStatusActive);
    CHECK(endStatus == cudaSuccess && graph);
    CHECK(cudaGraphDestroy(graph) == cudaSuccess);
    CHECK(cudaStreamDestroy(capture) == cudaSuccess);
    capturedWork = {};
    capturedGeneration.reset();
    retirement->Drain();
    CHECK(resources->Snapshot().usedBytes == 0);

    // Explicit backend close is an irreversible embedding-runtime boundary,
    // not a device reset.  First admit both the owner and its consumer, then
    // close while CUDA itself is still valid.  The late drops below must not
    // touch CUDA: they transfer both charged payloads into neutral retention.
    auto closingGeneration = MakeGeneration(4);
    CHECK(closingGeneration);
    std::weak_ptr<const UsdGenDeviceOwner> closingOwner = closingGeneration->Owner();
    cudaStream_t closingConsumer = nullptr;
    CHECK(cudaStreamCreateWithFlags(&closingConsumer, cudaStreamNonBlocking) == cudaSuccess);
    auto closingLease = AcquireGeometry(closingGeneration, closingConsumer);
    CHECK(closingLease && closingLease.Geometry().pointCount == 2);
    size_t const closingBytes = resources->Snapshot().usedBytes;
    CHECK(closingBytes != 0);

    CloseCudaRetirementBackend();
    CHECK(!IsCudaRetirementBackendOpen());
    CHECK(!TryReserveCudaRetirement(device));

    // Complete observes the closed adapter before event/callback submission;
    // the consumer and the still-referenced owner remain quarantined.  Drain
    // is nevertheless finite because quarantine releases its wait credit.
    closingLease = {};
    closingGeneration.reset();
    retirement->Drain();
    CHECK(!closingOwner.expired());
    CHECK(resources->Snapshot().usedBytes == closingBytes);
    CHECK(cudaStreamDestroy(closingConsumer) == cudaSuccess);

    std::puts("testUsdGenCudaRetirement: PASS");
    return 0;
}
