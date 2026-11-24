#ifdef USDGEN_ENABLE_CUDA
#include "usdGen/gpu/generation.h"
#include "usdGen/gpu/cudaCompat.h"
#include "usdGen/gpu/curveCompaction.h"
#include "usdGen/gpu/curveGrow.h"
#include "usdGen/gpu/scatterGrow.h"
#include "usdGen/gpu/cudaRetirement.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <utility>

namespace usdGen::gpu {
namespace {
using Status = UsdGenDeviceStatus;

class CudaOwnerRetirementPayload final : public UsdGenExecutionRetirementPayload {
public:
    std::shared_ptr<const UsdGenDeviceGeneration> base;
    std::unique_ptr<CudaCurveSource> source;
    std::unique_ptr<CudaCurveResample> resampled;
    std::unique_ptr<CudaCurveCompaction> compacted;
    std::unique_ptr<CudaScatterGrow> scatterGrow;
    std::unique_ptr<CudaCurveGrow> curveGrow;
    std::unique_ptr<DeviceBuffer<float>> widths;
    std::unique_ptr<DeviceBuffer<float3>> points;
    std::vector<CudaNamedChannelPlane> namedChannels;
    int device = -1;

    bool IsComplete() noexcept override { return true; }
    bool Destroy() noexcept override {
        if (!IsCudaRetirementBackendOpen()) return false;
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess || device < 0 ||
            cudaSetDevice(device) != cudaSuccess) return false;
        // DeviceBuffer and the curve owners carry their byte permits.  Their
        // destructors run only here, on the retirement worker, never on a
        // publication/refcount-drop thread.
        source.reset(); resampled.reset(); compacted.reset(); scatterGrow.reset(); curveGrow.reset();
        widths.reset(); points.reset(); namedChannels.clear(); base.reset();
        if (previous != device) cudaSetDevice(previous);
        return true;
    }
    void Quarantine() noexcept override {
        // The service retains this complete payload permanently.  In
        // particular do not destroy a DeviceBuffer here: its destructor may
        // issue cudaFree in an unproven/lost context.
    }
};

class SourceOwner;

class CudaConsumerRetirementPayload final : public UsdGenExecutionRetirementPayload {
public:
    struct CallbackState {
        UsdGenExecutionRetirementSignal signal;
        std::atomic<bool>* quarantined = nullptr;
    };

    CudaConsumerRetirementPayload(std::shared_ptr<const SourceOwner> owner_,
                                  cudaEvent_t event_, int device_,
                                  UsdGenExecutionRetirementSignal signal_,
                                  std::atomic<bool>* quarantined_)
        : owner(std::move(owner_)), event(event_), device(device_),
          callback{std::move(signal_), quarantined_} {}

    bool IsComplete() noexcept override {
        if (!IsCudaRetirementBackendOpen()) return false;
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess || device < 0 ||
            cudaSetDevice(device) != cudaSuccess) return false;
        cudaError_t const status = cudaEventQuery(event);
        if (previous != device) cudaSetDevice(previous);
        // The callback is ordered after event recording.  A single query is
        // a proof check, never a retry/poll loop.
        return status == cudaSuccess;
    }
    bool Destroy() noexcept override {
        if (!IsCudaRetirementBackendOpen()) return false;
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess || device < 0 ||
            cudaSetDevice(device) != cudaSuccess) return false;
        if (cudaEventDestroy(event) != cudaSuccess) {
            if (previous != device) cudaSetDevice(previous);
            return false;
        }
        event = nullptr;
        owner.reset();
        if (previous != device) cudaSetDevice(previous);
        return true;
    }
    void Quarantine() noexcept override {
        // Retention by the service preserves the callback userdata, event and
        // strong owner reference without any CUDA call on an error path.
    }

    std::shared_ptr<const SourceOwner> owner;
    cudaEvent_t event = nullptr;
    int device = -1;
    CallbackState callback;
};

class SourceOwner final : public UsdGenDeviceOwner,
                          public std::enable_shared_from_this<SourceOwner> {
public:
    SourceOwner(std::unique_ptr<CudaOwnerRetirementPayload> payload,
                UsdGenExecutionRetirementTicket ticket)
        : payload_(std::move(payload)), ticket_(std::move(ticket)),
          signal_(ticket_.MakeSignal()) {}
    ~SourceOwner() override;
    bool ProducerReady() const noexcept override;
    size_t ExclusiveRetainedBytes() const noexcept override;
    size_t InclusiveRetainedBytes() const noexcept override;
    std::unique_ptr<UsdGenDeviceConsumer> AcquireConsumer(
        UsdGenDeviceStream stream) const noexcept override;
    DeviceView<const float> HairT() const noexcept {
        return payload_->base ? BaseOwner()->HairT() : payload_->curveGrow ? payload_->curveGrow->hairT() : payload_->scatterGrow ? payload_->scatterGrow->hairT() :
            payload_->compacted ? payload_->compacted->hairT() :
            payload_->resampled ? payload_->resampled->hairT() : payload_->source->hairT();
    }
    DeviceView<const int32_t> RootPrim() const noexcept {
        return payload_->base ? BaseOwner()->RootPrim() : payload_->curveGrow ? payload_->curveGrow->rootPrim() : payload_->scatterGrow ? payload_->scatterGrow->rootPrim() :
            payload_->compacted ? payload_->compacted->rootPrim() :
            payload_->resampled ? payload_->resampled->rootPrim() : payload_->source->rootPrim();
    }
    DeviceView<const float2> RootUV() const noexcept {
        return payload_->base ? BaseOwner()->RootUV() : payload_->curveGrow ? payload_->curveGrow->rootUV() : payload_->scatterGrow ? payload_->scatterGrow->rootUV() :
            payload_->compacted ? payload_->compacted->rootUV() :
            payload_->resampled ? payload_->resampled->rootUV() : payload_->source->rootUV();
    }
    DeviceCurveGeometryView Geometry() const noexcept {
        auto view = payload_->base ? BaseOwner()->Geometry() : payload_->curveGrow ? payload_->curveGrow->view() : payload_->scatterGrow ? payload_->scatterGrow->view() :
            payload_->compacted ? payload_->compacted->view() :
            payload_->resampled ? payload_->resampled->view() : payload_->source->view();
        if (payload_->widths) view.widths = {payload_->widths->data(), payload_->widths->size()};
        if (payload_->points) view.points = {payload_->points->data(), payload_->points->size()};
        return view;
    }
    DeviceView<const float3> RootT() const noexcept {
        return payload_->base ? BaseOwner()->RootT() : payload_->curveGrow
            ? payload_->curveGrow->rootT() : payload_->scatterGrow
            ? payload_->scatterGrow->rootT() : payload_->compacted
            ? payload_->compacted->frames().tangent : DeviceView<const float3>{};
    }
    DeviceView<const float3> RootB() const noexcept {
        return payload_->base ? BaseOwner()->RootB() : payload_->curveGrow
            ? payload_->curveGrow->rootB() : payload_->scatterGrow
            ? payload_->scatterGrow->rootB() : payload_->compacted
            ? payload_->compacted->frames().binormal : DeviceView<const float3>{};
    }
    DeviceView<const float3> RootN() const noexcept {
        return payload_->base ? BaseOwner()->RootN() : payload_->curveGrow
            ? payload_->curveGrow->rootN() : payload_->scatterGrow
            ? payload_->scatterGrow->rootN() : payload_->compacted
            ? payload_->compacted->frames().normal : DeviceView<const float3>{};
    }
    CudaNamedChannelPlane const* NamedChannel(std::string const& name) const noexcept {
        for (auto const& plane : payload_->namedChannels)
            if (plane.metadata.name == name) return &plane;
        return payload_->base ? BaseOwner()->NamedChannel(name) : nullptr;
    }
    // Called only after UsdGenDeviceGeneration::Create has accepted this
    // immutable owner.  This changes accounting categories only: base is an
    // older shared generation, retained owner-local status/workspace becomes
    // Pinned, and unrelated evaluator/RBF cache owners remain Cache.
    void ReclassifyPublishedResources() noexcept {
        if (payload_->source) payload_->source->ReclassifyPublishedGeneration();
        if (payload_->resampled) payload_->resampled->ReclassifyPublishedGeneration();
        if (payload_->compacted) payload_->compacted->ReclassifyPublishedGeneration();
        if (payload_->scatterGrow) payload_->scatterGrow->ReclassifyPublishedGeneration();
        if (payload_->curveGrow) payload_->curveGrow->ReclassifyPublishedGeneration();
        if (payload_->widths) payload_->widths->Reclassify(UsdGenExecutionResourceKind::Pinned);
        if (payload_->points) payload_->points->Reclassify(UsdGenExecutionResourceKind::Pinned);
        for (auto const& plane : payload_->namedChannels)
            if (plane.bytes) plane.bytes->Reclassify(UsdGenExecutionResourceKind::Pinned);
    }
private:
    class Consumer;
    SourceOwner const* BaseOwner() const noexcept {
        return static_cast<SourceOwner const*>(payload_->base->Owner().get());
    }
    Status WaitOn(cudaStream_t) const noexcept;
    std::unique_ptr<CudaOwnerRetirementPayload> payload_;
    UsdGenExecutionRetirementTicket ticket_;
    UsdGenExecutionRetirementSignal signal_;
    mutable std::atomic<bool> quarantined_{false};
};

class SourceOwner::Consumer final : public UsdGenDeviceConsumer {
public:
    Consumer(std::unique_ptr<CudaConsumerRetirementPayload> payload,
             UsdGenExecutionRetirementTicket ticket,
             UsdGenExecutionRetirementSignal signal, UsdGenDeviceStream stream)
        : payload_(std::move(payload)), ticket_(std::move(ticket)),
          signal_(std::move(signal)), stream_(stream) {}

    ~Consumer() override { Complete(); }

    UsdGenDeviceStream Stream() const noexcept override { return stream_; }

    Status WaitUntilReady() const noexcept override {
        if (!payload_ || !payload_->owner || completed_.load()) return Status::InvalidLease;
        if (!IsCudaRetirementBackendOpen() || payload_->owner->quarantined_.load()) return Status::ConsumerRejected;

        int current = -1;
        if (cudaGetDevice(&current) != cudaSuccess || current != payload_->device)
            return Status::ConsumerRejected;
        if (stream_ != 0) {
            int streamDevice = -1;
            if (cudaStreamGetDevice(reinterpret_cast<cudaStream_t>(stream_),
                                    &streamDevice) != cudaSuccess ||
                streamDevice != payload_->device)
                return Status::ConsumerRejected;
        }

        return payload_->owner->WaitOn(reinterpret_cast<cudaStream_t>(stream_));
    }

    void Complete() noexcept override {
        bool expected = false;
        if (!completed_.compare_exchange_strong(expected, true)) return;

        if (!payload_) return;
        if (!IsCudaRetirementBackendOpen()) {
            payload_->owner->quarantined_.store(true);
            ticket_.Retire(std::move(payload_));
            signal_.SignalFailure();
            return;
        }
        int previous = -1;
        bool const selected = cudaGetDevice(&previous) == cudaSuccess &&
            cudaSetDevice(payload_->device) == cudaSuccess;
        cudaStream_t const stream = reinterpret_cast<cudaStream_t>(stream_);
        bool const recorded = selected && cudaEventRecord(payload_->event, stream) == cudaSuccess;
        bool const callback = recorded &&
            cudaStreamAddCallback(stream, [](cudaStream_t, cudaError_t status, void *data) {
                // Copy first: the worker may release the payload as Signal
                // dispatches it, so never touch callback userdata afterward.
                auto const* callback = static_cast<CudaConsumerRetirementPayload::CallbackState*>(data);
                auto signal = callback->signal;
                auto* quarantined = callback->quarantined;
                if (status == cudaSuccess) signal.SignalSuccess();
                else {
                    if (quarantined) quarantined->store(true);
                    signal.SignalFailure();
                }
            }, &payload_->callback, 0) == cudaSuccess;
        if (selected && previous != payload_->device) cudaSetDevice(previous);
        if (!recorded || !callback) {
            // The event/callback sequence cannot prove this stream terminal.
            // Poison future acquires before handing the payload to retention.
            payload_->owner->quarantined_.store(true);
        }
        ticket_.Retire(std::move(payload_));
        if (!recorded || !callback) signal_.SignalFailure();
    }

private:
    std::unique_ptr<CudaConsumerRetirementPayload> payload_;
    UsdGenExecutionRetirementTicket ticket_;
    UsdGenExecutionRetirementSignal signal_;
    UsdGenDeviceStream stream_ = 0;
    std::atomic<bool> completed_{false};
};

SourceOwner::~SourceOwner()
{
    // Payload storage was allocated and ticket admission succeeded with this
    // owner.  Final shared-owner release therefore only transfers it.
    ticket_.Retire(std::move(payload_));
    if (!quarantined_.load()) signal_.SignalSuccess();
    else signal_.SignalFailure();
}

bool SourceOwner::ProducerReady() const noexcept
{
    if (!IsCudaRetirementBackendOpen() || quarantined_.load() || !payload_)
        return false;
    // Every owner-local producer must be terminally proved, even when (as
    // with source + resample) only one supplies the exposed geometry.
    if ((payload_->source && (payload_->source->pending() ||
                              payload_->source->HasUnprovenUpload())) ||
        (payload_->resampled && (payload_->resampled->pending() ||
                                 payload_->resampled->HasUnprovenUpload())) ||
        (payload_->compacted && (payload_->compacted->pending() ||
                                 payload_->compacted->HasUnprovenWork())) ||
        (payload_->scatterGrow && (payload_->scatterGrow->pending() ||
                                   payload_->scatterGrow->HasUnprovenWork())) ||
        (payload_->curveGrow && (payload_->curveGrow->pending() ||
                                 payload_->curveGrow->HasUnprovenWork())))
        return false;
    if (payload_->base)
        return payload_->base->Owner() && payload_->base->Owner()->ProducerReady();
    if (payload_->compacted) return payload_->compacted->generation() != 0;
    if (payload_->curveGrow) return payload_->curveGrow->generation() != 0;
    if (payload_->scatterGrow) return payload_->scatterGrow->generation() != 0;
    return payload_->source && payload_->source->generation() != 0;
}

size_t SourceOwner::ExclusiveRetainedBytes() const noexcept
{
    if (!payload_) return 0;
    size_t total = 0;
    auto add = [&total](size_t bytes) {
        if (bytes > std::numeric_limits<size_t>::max() - total)
            total = std::numeric_limits<size_t>::max();
        else total += bytes;
    };
    // `base` is another immutable COW owner and is intentionally excluded.
    // Its physical allocation is charged by its own cache entry/owner.
    if (payload_->source) add(payload_->source->ExclusiveRetainedBytes());
    if (payload_->resampled) add(payload_->resampled->ExclusiveRetainedBytes());
    if (payload_->compacted) add(payload_->compacted->ExclusiveRetainedBytes());
    if (payload_->scatterGrow) add(payload_->scatterGrow->ExclusiveRetainedBytes());
    if (payload_->curveGrow) add(payload_->curveGrow->ExclusiveRetainedBytes());
    if (payload_->widths) add(payload_->widths->bytes());
    if (payload_->points) add(payload_->points->bytes());
    for (auto const& plane : payload_->namedChannels)
        if (plane.bytes) add(plane.bytes->bytes());
    return total;
}

size_t SourceOwner::InclusiveRetainedBytes() const noexcept
{
    size_t total = ExclusiveRetainedBytes();
    auto add = [&total](size_t bytes) {
        if (bytes > std::numeric_limits<size_t>::max() - total)
            total = std::numeric_limits<size_t>::max();
        else total += bytes;
    };
    // A revision owner keeps its immutable predecessor alive even after the
    // predecessor's cache record is evicted. Include that reachable COW
    // footprint in this entry's conservative charge so the cache cannot
    // under-account physical allocations. The exclusive API intentionally
    // remains delta-only for resource-pool accounting.
    if (payload_ && payload_->base && payload_->base->Owner())
        add(payload_->base->Owner()->InclusiveRetainedBytes());
    return total;
}

std::unique_ptr<UsdGenDeviceConsumer>
SourceOwner::AcquireConsumer(UsdGenDeviceStream stream) const noexcept
{
    int device = -1;
    if (!ProducerReady() || cudaGetDevice(&device) != cudaSuccess || device != payload_->device)
        return {};
    // Capture detection must precede cudaStreamGetDevice: the latter is not a
    // capture-safe query on all supported CUDA runtimes and can invalidate an
    // otherwise usable capture.  Current-device provenance was checked above.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(stream), &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) return {};
    // Zero denotes the default stream and is associated with the current
    // device. cudaStreamGetDevice(nullptr) is not portable, so only query
    // explicitly supplied streams after capture has been ruled out.
    if (stream != 0) {
        int streamDevice = -1;
        if (cudaStreamGetDevice(reinterpret_cast<cudaStream_t>(stream),
                                &streamDevice) != cudaSuccess ||
            streamDevice != payload_->device)
            return {};
    }
    try {
        // Admission happens before an event/callback is created or a lease is
        // returned.  This keeps a full retirement queue a clean rejection.
        auto ticket = TryReserveCudaRetirement(payload_->device);
        if (!ticket) return {};
        auto signal = ticket->MakeSignal();
        struct UnsubmittedEvent {
            cudaEvent_t value = nullptr;
            ~UnsubmittedEvent() { if (value) cudaEventDestroy(value); }
        } event;
        if (cudaEventCreateWithFlags(&event.value, cudaEventDisableTiming) != cudaSuccess) return {};
        auto owner = shared_from_this();
        auto payload = std::make_unique<CudaConsumerRetirementPayload>(
            std::move(owner), event.value, device, signal, &quarantined_);
        auto consumer = std::make_unique<Consumer>(std::move(payload), std::move(*ticket),
                                                   std::move(signal), stream);
        event.value = nullptr; // ownership is now retained by the payload.
        return consumer;
    } catch (...) {
        return {};
    }
}

Status SourceOwner::WaitOn(cudaStream_t stream) const noexcept
{
    if (!IsCudaRetirementBackendOpen()) return Status::ConsumerRejected;
    if (payload_->base) {
        auto const* base = BaseOwner();
        if (!base || base->WaitOn(stream) != Status::Ok)
            return Status::SynchronizationFailed;
    }
    if ((payload_->source && payload_->source->waitOn(stream) != CurveSourceStatus::Ok) ||
        (payload_->resampled && payload_->resampled->waitOn(stream) != cudaSuccess) ||
        (payload_->compacted && payload_->compacted->waitOn(stream) != cudaSuccess) ||
        (payload_->scatterGrow && payload_->scatterGrow->waitOn(stream) !=
                                      ScatterGrowStatus::Ok) ||
        (payload_->curveGrow && payload_->curveGrow->waitOn(stream) !=
                                    CurveGrowStatus::Ok) ||
        (payload_->widths && payload_->widths->waitOn(stream) != cudaSuccess) ||
        (payload_->points && payload_->points->waitOn(stream) != cudaSuccess))
        return Status::SynchronizationFailed;
    for (auto const& plane : payload_->namedChannels)
        if (plane.bytes && plane.bytes->waitOn(stream) != cudaSuccess)
            return Status::SynchronizationFailed;
    return Status::Ok;
}

} // namespace

static bool AppendLocalNamedChannels(
    std::vector<CudaNamedChannelPlane> const& planes,
    DeviceCurveGeometryView const& geometry, int device,
    std::vector<UsdGenDeviceChannelMetadata>* channels, std::string* reason,
    bool replaceGeneric = false)
{
    if (!channels) return false;
    std::vector<std::string> names;
    names.reserve(planes.size());
    for (CudaNamedChannelPlane const& plane : planes) {
        auto const& metadata = plane.metadata;
        uint64_t expectedElements = 0;
        if (metadata.domain == UsdGenDeviceDomain::Point)
            expectedElements = geometry.pointCount;
        else if (metadata.domain == UsdGenDeviceDomain::Primitive)
            expectedElements = geometry.curveCount;
        else if (metadata.domain == UsdGenDeviceDomain::Groom)
            expectedElements = 1;
        else {
            if (reason) *reason = "CUDA named channel has unsupported domain";
            return false;
        }
        if (metadata.name.empty() ||
            metadata.semantic != UsdGenDeviceChannelSemantic::Generic ||
            !metadata.readOnly || metadata.arity == 0 || metadata.arity > 16 ||
            (metadata.type != UsdGenDeviceValueType::Float32 &&
             metadata.type != UsdGenDeviceValueType::Int32) ||
            metadata.strideBytes != metadata.arity * sizeof(uint32_t) ||
            metadata.elementCount != expectedElements ||
            metadata.elementCount >
                std::numeric_limits<size_t>::max() / metadata.strideBytes ||
            !plane.bytes || plane.bytes->size() !=
                size_t(metadata.elementCount) * metadata.strideBytes ||
            std::find(names.begin(), names.end(), metadata.name) != names.end()) {
            if (reason) *reason = "CUDA generation has incompatible local named channel";
            return false;
        }
        auto existing = std::find_if(channels->begin(), channels->end(),
            [&](auto const& channel) { return channel.name == metadata.name; });
        if (existing != channels->end() &&
            (!replaceGeneric ||
             existing->semantic != UsdGenDeviceChannelSemantic::Generic)) {
            if (reason) *reason = "CUDA generation has incompatible local named channel";
            return false;
        }
        if (plane.bytes->size()) {
            cudaPointerAttributes attributes{};
            if (cudaPointerGetAttributes(&attributes, plane.bytes->data()) != cudaSuccess ||
                attributes.type != cudaMemoryTypeDevice || attributes.device != device) {
                if (reason) *reason = "CUDA generation named channel has incompatible device";
                return false;
            }
        }
        names.push_back(metadata.name);
        if (existing != channels->end()) *existing = metadata;
        else channels->push_back(metadata);
    }
    return true;
}

static std::shared_ptr<const UsdGenDeviceGeneration> MakeGeneration(
    std::unique_ptr<CudaCurveSource> source, uint64_t generation, std::string* reason,
    bool alreadyDeformed, std::unique_ptr<DeviceBuffer<float>> widths,
    std::unique_ptr<DeviceBuffer<float3>> points, std::unique_ptr<CudaCurveCompaction> compacted,
    uint64_t topologyVersion, std::unique_ptr<CudaCurveResample> resampled,
    std::vector<CudaNamedChannelPlane> namedChannels,
    std::unique_ptr<CudaScatterGrow> scatterGrow = {},
    std::unique_ptr<CudaCurveGrow> curveGrow = {}) {
    if (!IsCudaRetirementBackendOpen()) {
        // A late static/embedded caller cannot safely run CUDA-owning
        // destructors after CloseCudaRetirementBackend. Preserve them for
        // process/context teardown instead of issuing a backend free.
        source.release(); resampled.release(); compacted.release();
        scatterGrow.release(); curveGrow.release();
        widths.release(); points.release();
        for (auto& plane : namedChannels)
            if (plane.bytes) plane.bytes->quarantine();
        if (reason) *reason = "CUDA retirement backend is closed";
        return {};
    }
    int device = -1;
    unsigned const primaryOwners = unsigned(bool(source)) + unsigned(bool(compacted)) +
        unsigned(bool(scatterGrow)) + unsigned(bool(curveGrow));
    if (primaryOwners != 1 || (compacted && (source || resampled || scatterGrow || curveGrow)) ||
        (scatterGrow && (source || resampled || compacted || curveGrow)) ||
        (curveGrow && (source || resampled || compacted || scatterGrow)) ||
        (resampled && !source) || cudaGetDevice(&device) != cudaSuccess ||
        (source && (source->pending() || source->HasUnprovenUpload() ||
                    !source->generation() || source->deviceIndex() != device)) ||
        (resampled && (resampled->pending() || resampled->HasUnprovenUpload() ||
                       resampled->deviceIndex() != device)) ||
        (compacted && (compacted->pending() || compacted->HasUnprovenWork() ||
                       !compacted->generation() || compacted->deviceIndex() != device)) ||
        (scatterGrow && (scatterGrow->pending() || scatterGrow->HasUnprovenWork() ||
                         !scatterGrow->generation() ||
                         scatterGrow->deviceIndex() != device)) ||
        (curveGrow && (curveGrow->pending() || curveGrow->HasUnprovenWork() ||
                       !curveGrow->generation() || curveGrow->deviceIndex() != device))) {
        if (reason) *reason = "CUDA source is not a completed generation";
        return {};
    }
    auto view = curveGrow ? curveGrow->view() : scatterGrow ? scatterGrow->view() : compacted ? compacted->view() :
        resampled ? resampled->view() : source->view();
    auto hairT = curveGrow ? curveGrow->hairT() : scatterGrow ? scatterGrow->hairT() : compacted ? compacted->hairT() :
        resampled ? resampled->hairT() : source->hairT();
    auto rootPrim = curveGrow ? curveGrow->rootPrim() : scatterGrow ? scatterGrow->rootPrim() :
        compacted ? compacted->rootPrim() : resampled ? resampled->rootPrim() : source->rootPrim();
    auto rootUV = curveGrow ? curveGrow->rootUV() : scatterGrow ? scatterGrow->rootUV() :
        compacted ? compacted->rootUV() : resampled ? resampled->rootUV() : source->rootUV();
    auto compactedFrames = compacted ? compacted->frames() : RestRootFrames{};
    auto rootT = curveGrow ? curveGrow->rootT() : compacted ? compactedFrames.tangent : DeviceView<const float3>{};
    auto rootB = curveGrow ? curveGrow->rootB() : compacted ? compactedFrames.binormal : DeviceView<const float3>{};
    auto rootN = curveGrow ? curveGrow->rootN() : compacted ? compactedFrames.normal : DeviceView<const float3>{};
    bool const compactedHasFrames = compacted && (rootT.size || rootB.size || rootN.size ||
                                                   compactedFrames.stableIds.size);
    if (view.points.size != view.pointCount || view.restPoints.size != view.pointCount ||
        view.widths.size != view.pointCount || view.stableIds.size != view.curveCount ||
        view.curveOffsets.size != view.curveCount + 1 || hairT.size != view.pointCount ||
        !view.points || !view.restPoints || !view.widths || !view.stableIds || !view.curveOffsets || !hairT ||
        (curveGrow && (rootPrim.size != view.curveCount || rootUV.size != view.curveCount ||
                       rootT.size != view.curveCount || rootB.size != view.curveCount ||
                       rootN.size != view.curveCount || !rootPrim || !rootUV || !rootT || !rootB || !rootN)) ||
        (compactedHasFrames && (rootT.size != view.curveCount || rootB.size != view.curveCount ||
                       rootN.size != view.curveCount || !rootT || !rootB || !rootN))) {
        if (reason) *reason = "CUDA geometry revision is missing required C3 channels";
        return {};
    }
    if (widths) {
        cudaPointerAttributes attributes{};
        if (widths->size() != view.pointCount ||
            (widths->size() && (cudaPointerGetAttributes(&attributes, widths->data()) != cudaSuccess ||
              attributes.type != cudaMemoryTypeDevice || attributes.device != device))) {
            if (reason) *reason = "CUDA width revision has incompatible shape or device";
            return {};
        }
    }
    if (points) {
        cudaPointerAttributes attributes{};
        if (points->size() != view.pointCount ||
            (points->size() && (cudaPointerGetAttributes(&attributes, points->data()) != cudaSuccess ||
              attributes.type != cudaMemoryTypeDevice || attributes.device != device))) {
            if (reason) *reason = "CUDA point revision has incompatible shape or device";
            return {};
        }
    }
    UsdGenDeviceGeneration::CreateInfo info;
    info.identity = {UsdGenDeviceBackend::Cuda, device, generation};
    info.geometry.topologyVersion = generation;
    info.geometry.valueVersion = generation;
    info.geometry.curveCount = view.curveCount;
    info.geometry.pointCount = view.pointCount;
    if (topologyVersion != UINT64_MAX) info.geometry.topologyVersion = topologyVersion;
    info.geometry.alreadyDeformed = alreadyDeformed;
    using Type = UsdGenDeviceValueType;
    using Domain = UsdGenDeviceDomain;
    using Semantic = UsdGenDeviceChannelSemantic;
    auto channel = [&](char const* name, Type type, Domain domain,
                       size_t count, unsigned arity, unsigned stride, Semantic semantic) {
        info.channels.push_back({name, type, domain, count, arity, stride, true, semantic});
    };
    channel("points", Type::Float32x3, Domain::Point, view.pointCount, 3, sizeof(float3), Semantic::Points);
    channel("rest", Type::Float32x3, Domain::Point, view.pointCount, 3, sizeof(float3), Semantic::RestPoints);
    channel("widths", Type::Float32, Domain::Point, view.pointCount, 1, sizeof(float), Semantic::Widths);
    channel("hairT", Type::Float32, Domain::Point, view.pointCount, 1, sizeof(float), Semantic::HairT);
    // Offsets are topology storage (C+1), not a per-primitive value (C).
    channel("curveOffsets", Type::UInt32, Domain::Topology, view.curveOffsets.size, 1, sizeof(uint32_t), Semantic::CurveOffsets);
    channel("curveId", Type::UInt64, Domain::Primitive, view.curveCount, 1, sizeof(uint64_t), Semantic::StableIds);
    if (rootPrim.size)
        channel("skinprim", Type::Int32, Domain::Primitive, view.curveCount, 1, sizeof(int32_t), Semantic::RootPrim);
    if (rootUV.size)
        channel("skinprimuv", Type::Float32x2, Domain::Primitive, view.curveCount, 2, sizeof(float2), Semantic::RootUV);
    if (!AppendLocalNamedChannels(namedChannels, view, device, &info.channels, reason))
        return {};
    auto ticket = TryReserveCudaRetirement(device);
    if (!ticket) {
        if (reason) *reason = "CUDA generation retirement admission is exhausted";
        return {};
    }
    std::unique_ptr<CudaOwnerRetirementPayload> payload;
    try {
        payload = std::make_unique<CudaOwnerRetirementPayload>();
        payload->source = std::move(source);
        payload->resampled = std::move(resampled);
        payload->compacted = std::move(compacted);
        payload->scatterGrow = std::move(scatterGrow);
        payload->curveGrow = std::move(curveGrow);
        payload->widths = std::move(widths);
        payload->points = std::move(points);
        payload->namedChannels = std::move(namedChannels);
        payload->device = device;
        auto owner = std::make_shared<SourceOwner>(std::move(payload), std::move(*ticket));
        info.owner = owner;
        auto generationResult = UsdGenDeviceGeneration::Create(std::move(info), reason);
        if (generationResult) owner->ReclassifyPublishedResources();
        return generationResult;
    } catch (...) {
        // This is still factory-side, before a published owner exists.  The
        // inputs were validated completed above, so ordinary destruction is
        // safe and avoids converting recoverable admission pressure into a
        // permanent charged leak.
        if (reason) *reason = "CUDA generation retirement payload allocation failed";
        return {};
    }
}

std::shared_ptr<const UsdGenDeviceGeneration> MakeResampledGeneration(
    std::unique_ptr<CudaCurveSource> source, std::unique_ptr<CudaCurveResample> resampled,
    uint64_t generation, std::string* reason, bool alreadyDeformed,
    std::unique_ptr<DeviceBuffer<float>> widths, std::unique_ptr<DeviceBuffer<float3>> points,
    uint64_t topologyVersion,
    std::vector<CudaNamedChannelPlane> namedChannels) {
    return MakeGeneration(std::move(source), generation, reason, alreadyDeformed,
                          std::move(widths), std::move(points), {},
                          topologyVersion, std::move(resampled),
                          std::move(namedChannels));
}

std::shared_ptr<const UsdGenDeviceGeneration> MakeSourceGeneration(
    std::unique_ptr<CudaCurveSource> source, uint64_t generation, std::string* reason,
    bool alreadyDeformed, std::unique_ptr<DeviceBuffer<float>> widths,
    std::unique_ptr<DeviceBuffer<float3>> points, uint64_t topologyVersion,
    std::vector<CudaNamedChannelPlane> namedChannels) {
    return MakeGeneration(std::move(source), generation, reason, alreadyDeformed,
        std::move(widths), std::move(points), {}, topologyVersion, {},
        std::move(namedChannels));
}

std::shared_ptr<const UsdGenDeviceGeneration> MakeCompactedGeneration(
    std::unique_ptr<CudaCurveCompaction> compacted, uint64_t generation, std::string* reason,
    bool alreadyDeformed, std::unique_ptr<DeviceBuffer<float>> widths,
    std::unique_ptr<DeviceBuffer<float3>> points, uint64_t topologyVersion,
    std::vector<CudaNamedChannelPlane> namedChannels) {
    return MakeGeneration({}, generation, reason, alreadyDeformed,
        std::move(widths), std::move(points), std::move(compacted), topologyVersion,
        {}, std::move(namedChannels));
}

std::shared_ptr<const UsdGenDeviceGeneration> MakeScatterGrowGeneration(
    std::unique_ptr<CudaScatterGrow> geometry, uint64_t generation,
    std::string* reason, bool alreadyDeformed, uint64_t topologyVersion,
    std::vector<CudaNamedChannelPlane> namedChannels,
    std::unique_ptr<DeviceBuffer<float>> widths,
    std::unique_ptr<DeviceBuffer<float3>> points) {
    return MakeGeneration({}, generation, reason, alreadyDeformed,
        std::move(widths), std::move(points), {},
        topologyVersion, {}, std::move(namedChannels), std::move(geometry));
}

std::shared_ptr<const UsdGenDeviceGeneration> MakeCurveGrowGeneration(
    std::unique_ptr<CudaCurveGrow> geometry, uint64_t generation,
    std::string* reason, bool alreadyDeformed, uint64_t topologyVersion,
    std::vector<CudaNamedChannelPlane> namedChannels,
    std::unique_ptr<DeviceBuffer<float>> widths,
    std::unique_ptr<DeviceBuffer<float3>> points) {
    return MakeGeneration({}, generation, reason, alreadyDeformed,
        std::move(widths), std::move(points), {}, topologyVersion, {},
        std::move(namedChannels), {}, std::move(geometry));
}

std::shared_ptr<const UsdGenDeviceGeneration> MakePointRevisionGeneration(
    std::shared_ptr<const UsdGenDeviceGeneration> base, uint64_t generation,
    std::unique_ptr<DeviceBuffer<float3>> points, UsdGenDeviceToolMetadata tool,
    std::string* reason) {
    if (!IsCudaRetirementBackendOpen()) {
        points.release();
        if (reason) *reason = "CUDA retirement backend is closed";
        return {};
    }
    int device = -1;
    if (!base || base->Identity().backend != UsdGenDeviceBackend::Cuda ||
        !std::dynamic_pointer_cast<SourceOwner const>(base->Owner()) ||
        !base->Owner()->ProducerReady() || cudaGetDevice(&device) != cudaSuccess ||
        device != base->Identity().deviceIndex || generation <= base->Identity().generation) {
        if (reason) *reason = "CUDA point edit requires a completed local base and newer generation";
        return {};
    }
    cudaPointerAttributes attributes{};
    if (points && (points->size() != base->Geometry().pointCount ||
        (points->size() && (cudaPointerGetAttributes(&attributes, points->data()) != cudaSuccess ||
         attributes.type != cudaMemoryTypeDevice || attributes.device != device)))) {
        if (reason) *reason = "CUDA point edit has incompatible shape or device";
        return {};
    }
    UsdGenDeviceGeneration::CreateInfo info;
    info.identity = {UsdGenDeviceBackend::Cuda, device, generation};
    info.geometry = base->Geometry();
    info.geometry.valueVersion = generation;
    if (points) {
        for (UsdGenDeviceTileMetadata &tile : info.geometry.tiles) {
            tile.boundsValid = false;
        }
    }
    info.channels = base->Channels();
    info.tool = std::move(tool);
    auto ticket = TryReserveCudaRetirement(device);
    if (!ticket) {
        if (reason) *reason = "CUDA point edit retirement admission is exhausted";
        return {};
    }
    std::unique_ptr<CudaOwnerRetirementPayload> payload;
    try {
        payload = std::make_unique<CudaOwnerRetirementPayload>();
        payload->base = std::move(base);
        payload->points = std::move(points);
        payload->device = device;
        auto owner = std::make_shared<SourceOwner>(std::move(payload), std::move(*ticket));
        info.owner = owner;
        auto generationResult = UsdGenDeviceGeneration::Create(std::move(info), reason);
        if (generationResult) owner->ReclassifyPublishedResources();
        return generationResult;
    } catch (...) {
        if (reason) *reason = "CUDA point edit retirement payload allocation failed";
        return {};
    }
}

std::shared_ptr<const UsdGenDeviceGeneration> MakeNamedChannelRevisionGeneration(
    std::shared_ptr<const UsdGenDeviceGeneration> base, uint64_t generation,
    std::vector<CudaNamedChannelPlane> planes, UsdGenDeviceToolMetadata tool,
    std::string* reason) {
    if (!IsCudaRetirementBackendOpen()) {
        if (reason) *reason = "CUDA retirement backend is closed";
        return {};
    }
    int device = -1;
    auto baseOwner = base ? std::dynamic_pointer_cast<SourceOwner const>(base->Owner()) : nullptr;
    if (!base || !baseOwner || base->Identity().backend != UsdGenDeviceBackend::Cuda ||
        !base->Owner()->ProducerReady() || cudaGetDevice(&device) != cudaSuccess ||
        device != base->Identity().deviceIndex || generation <= base->Identity().generation ||
        planes.empty()) {
        if (reason) *reason = "CUDA channel revision requires a completed local base and planes";
        return {};
    }
    auto channels = base->Channels();
    for (auto const& plane : planes) {
        auto const existing = std::find_if(channels.begin(), channels.end(),
            [&](auto const& channel) { return channel.name == plane.metadata.name; });
        if (existing != channels.end() &&
            existing->semantic != UsdGenDeviceChannelSemantic::Generic) {
            if (reason) *reason = "CUDA channel revision cannot replace geometry channel";
            return {};
        }
    }
    if (!AppendLocalNamedChannels(planes, baseOwner->Geometry(), device,
                                  &channels, reason, true)) {
        if (reason) *reason = "CUDA channel revision has incompatible generic plane";
        return {};
    }
    UsdGenDeviceGeneration::CreateInfo info;
    info.identity = {UsdGenDeviceBackend::Cuda, device, generation};
    info.geometry = base->Geometry();
    info.geometry.valueVersion = generation;
    info.channels = std::move(channels);
    info.tool = std::move(tool);
    auto ticket = TryReserveCudaRetirement(device);
    if (!ticket) {
        if (reason) *reason = "CUDA channel revision retirement admission is exhausted";
        return {};
    }
    try {
        auto payload = std::make_unique<CudaOwnerRetirementPayload>();
        payload->base = std::move(base);
        payload->namedChannels = std::move(planes);
        payload->device = device;
        auto owner = std::make_shared<SourceOwner>(std::move(payload), std::move(*ticket));
        info.owner = owner;
        auto result = UsdGenDeviceGeneration::Create(std::move(info), reason);
        if (result) owner->ReclassifyPublishedResources();
        return result;
    } catch (...) {
        if (reason) *reason = "CUDA channel revision retirement payload allocation failed";
        return {};
    }
}

CudaNamedChannelLease AcquireNamedChannel(
    std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
    std::string const& name, cudaStream_t stream) {
    CudaNamedChannelLease result;
    if (!generation || name.empty() || generation->Identity().backend != UsdGenDeviceBackend::Cuda)
        return result;
    auto owner = std::dynamic_pointer_cast<SourceOwner const>(generation->Owner());
    auto const* plane = owner ? owner->NamedChannel(name) : nullptr;
    if (!plane || !plane->bytes) return result;
    auto lease = generation->AcquireLease(reinterpret_cast<UsdGenDeviceStream>(stream));
    if (!lease || lease.WaitUntilReady() != Status::Ok) return result;
    result.metadata_ = plane->metadata;
    result.bytes_ = static_cast<DeviceBuffer<unsigned char> const&>(*plane->bytes).view();
    result.lease_ = std::move(lease);
    return result;
}

CudaGeometryLease AcquireGeometry(
    std::shared_ptr<const UsdGenDeviceGeneration> const& generation, cudaStream_t stream) {
    CudaGeometryLease result;
    if (!generation || generation->Identity().backend != UsdGenDeviceBackend::Cuda) return result;
    auto owner = std::dynamic_pointer_cast<SourceOwner const>(generation->Owner());
    if (!owner) return result;
    auto lease = generation->AcquireLease(reinterpret_cast<UsdGenDeviceStream>(stream));
    if (!lease || lease.WaitUntilReady() != Status::Ok) return result;
    result.geometry_ = owner->Geometry();
    result.hairT_ = owner->HairT();
    result.rootPrim_ = owner->RootPrim();
    result.rootUV_ = owner->RootUV();
    result.rootT_ = owner->RootT();
    result.rootB_ = owner->RootB();
    result.rootN_ = owner->RootN();
    result.lease_ = std::move(lease);
    return result;
}

namespace {
template <class T>
bool _Slice(DeviceView<const T> source, uint64_t begin, uint64_t count,
            DeviceView<const T>* result)
{
    if (begin > source.size || count > source.size - begin ||
        begin > std::numeric_limits<size_t>::max() ||
        count > std::numeric_limits<size_t>::max()) return false;
    size_t const b = static_cast<size_t>(begin);
    size_t const n = static_cast<size_t>(count);
    if (n && !source.data) return false;
    result->data = n ? source.data + b : nullptr;
    result->size = n;
    return true;
}
}

CudaGeometryTileLease AcquireGeometryTile(
    std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
    uint32_t tileId, uint64_t expectedGeneration, cudaStream_t stream)
{
    CudaGeometryTileLease result;
    if (!generation || generation->Identity().backend != UsdGenDeviceBackend::Cuda ||
        generation->Identity().generation != expectedGeneration)
        return result;

    UsdGenDeviceTileMetadata const* tile = nullptr;
    for (UsdGenDeviceTileMetadata const& candidate : generation->Geometry().tiles) {
        if (candidate.tile == tileId) { tile = &candidate; break; }
    }
    if (!tile) return result;

    CudaGeometryLease parent = AcquireGeometry(generation, stream);
    if (!parent) return result;
    DeviceCurveGeometryView const geometry = parent.Geometry();
    DeviceView<const uint32_t> offsets{geometry.curveOffsets.data,
                                       geometry.curveOffsets.size};
    CudaGeometryTileView view;
    view.range = *tile;
    view.generation = generation->Identity().generation;
    view.topologyVersion = generation->Geometry().topologyVersion;
    view.valueVersion = generation->Geometry().valueVersion;

    // The extra offset sentinel is part of every non-empty and empty tile.
    if (tile->curveCount > tile->pointCount ||
        (tile->curveCount == 0 && tile->pointCount != 0) ||
        tile->firstCurve > geometry.curveCount ||
        tile->curveCount > geometry.curveCount - tile->firstCurve ||
        tile->curveCount == std::numeric_limits<uint64_t>::max() ||
        tile->firstPoint > std::numeric_limits<uint32_t>::max() ||
        tile->pointCount > std::numeric_limits<uint32_t>::max() - tile->firstPoint)
        return result;
    uint64_t const offsetCount = tile->curveCount + 1;
    if (!_Slice(offsets, tile->firstCurve, offsetCount, &view.curveOffsets) ||
        !_Slice(parent.HairT(), tile->firstPoint, tile->pointCount, &view.hairT) ||
        !_Slice(geometry.points, tile->firstPoint, tile->pointCount, &view.points) ||
        !_Slice(geometry.restPoints, tile->firstPoint, tile->pointCount, &view.restPoints) ||
        !_Slice(geometry.widths, tile->firstPoint, tile->pointCount, &view.widths) ||
        !_Slice(geometry.stableIds, tile->firstCurve, tile->curveCount, &view.stableIds))
        return result;

    DeviceView<const int32_t> const roots = parent.RootPrim();
    DeviceView<const float2> const uvs = parent.RootUV();
    if ((roots.size == 0) != (uvs.size == 0)) return result;
    if (roots.size == 0) view.rootPrim = {};
    else if (!_Slice(roots, tile->firstCurve, tile->curveCount, &view.rootPrim)) return result;
    if (uvs.size == 0) view.rootUV = {};
    else if (!_Slice(uvs, tile->firstCurve, tile->curveCount, &view.rootUV)) return result;
    result.view_ = view;
    result.parent_ = std::move(parent);
    return result;
}

std::shared_ptr<const UsdGenDeviceGeneration> WithTileMetadata(
    std::shared_ptr<const UsdGenDeviceGeneration> const& candidate,
    std::vector<UsdGenDeviceTileMetadata> tiles, std::string* reason,
    UsdGenDeviceCurveTopologyMetadata curveTopology)
{
    if (!candidate || !candidate->Owner()) {
        if (reason) *reason = "tile metadata requires an owned generation";
        return {};
    }
    auto const& geometry = candidate->Geometry();
    uint64_t curveEnd = 0, pointEnd = 0, previousId = 0;
    bool first = true;
    for (UsdGenDeviceTileMetadata const& tile : tiles) {
        if (!first && tile.tile <= previousId) {
            if (reason) *reason = "tile IDs must be strictly increasing";
            return {};
        }
        first = false;
        previousId = tile.tile;
        if (tile.curveCount > tile.pointCount ||
            (tile.curveCount == 0 && tile.pointCount != 0) ||
            tile.firstPoint > std::numeric_limits<uint32_t>::max() ||
            tile.pointCount > std::numeric_limits<uint32_t>::max() - tile.firstPoint ||
            tile.firstCurve != curveEnd || tile.firstPoint != pointEnd ||
            tile.curveCount > geometry.curveCount - curveEnd ||
            tile.pointCount > geometry.pointCount - pointEnd ||
            tile.curveCount > std::numeric_limits<uint64_t>::max() - curveEnd ||
            tile.pointCount > std::numeric_limits<uint64_t>::max() - pointEnd) {
            if (reason) *reason = "tile ranges must be contiguous and in bounds";
            return {};
        }
        curveEnd += tile.curveCount;
        pointEnd += tile.pointCount;
    }
    if (tiles.empty()) {
        if (geometry.curveCount != 0 || geometry.pointCount != 0) {
            if (reason) *reason = "non-empty geometry requires tile metadata";
            return {};
        }
    } else if (curveEnd != geometry.curveCount || pointEnd != geometry.pointCount) {
        if (reason) *reason = "tile metadata does not cover geometry";
        return {};
    }
    UsdGenDeviceGeneration::CreateInfo info;
    info.identity = candidate->Identity();
    info.geometry = geometry;
    info.geometry.tiles = std::move(tiles);
    if (curveTopology.type != UsdGenDeviceCurveType::Unknown ||
        curveTopology.basis != UsdGenDeviceCurveBasis::Unknown ||
        curveTopology.wrap != UsdGenDeviceCurveWrap::Unknown) {
        info.geometry.curveTopology = curveTopology;
    }
    info.tool = candidate->Tool();
    info.channels = candidate->Channels();
    info.owner = candidate->Owner();
    return UsdGenDeviceGeneration::Create(std::move(info), reason);
}
} // namespace usdGen::gpu
#endif
