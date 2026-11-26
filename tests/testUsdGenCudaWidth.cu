#include "usdGenCurves/cuda/width.h"
#include "usdGenCurves/cuda/widthBlend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <atomic>
#include <utility>

using namespace usdGen;
using namespace usdGen::gpu;

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", \
                                      #condition, __FILE__, __LINE__); return 1; } \
} while (false)

static bool Near(float a, float b) { return std::fabs(a - b) < 2.0e-5f; }
static std::atomic<int> freshCallbacks{0};
static std::atomic<int> freshStatus{int(cudaErrorUnknown)};
static void FreshSignal(cudaStream_t, cudaError_t status, void*) noexcept {
    freshStatus.store(int(status), std::memory_order_release);
    freshCallbacks.fetch_add(1, std::memory_order_release);
}

template <class T>
static bool Upload(DeviceBuffer<T> &device, std::vector<T> const &host,
                   cudaStream_t stream) {
    return cudaMemcpyAsync(device.data(), host.data(),
                           host.size() * sizeof(T), cudaMemcpyHostToDevice,
                           stream) == cudaSuccess;
}

template <class T>
static bool Download(DeviceBuffer<T> const &device, std::vector<T> &host,
                     cudaStream_t stream) {
    return cudaMemcpyAsync(host.data(), device.data(),
                           host.size() * sizeof(T), cudaMemcpyDeviceToHost,
                           stream) == cudaSuccess &&
           cudaStreamSynchronize(stream) == cudaSuccess;
}

static DeviceCurveGeometryView Geometry(DeviceBuffer<float> const &widths,
                                        DeviceBuffer<uint32_t> const &offsets) {
    return {{}, {}, {widths.data(), widths.size()},
            {offsets.data(), offsets.size()}, {}, 2, 5};
}

static WidthParameters Defaults(DeviceView<const float> profile) {
    WidthParameters p;
    p.widthProfile = profile;
    return p;
}

// Test-local ownership for a single, already-instantiated CUDA graph.  This
// deliberately owns no DeviceBuffer: the caller keeps every COW plane alive,
// records its completion event after each launch, and synchronizes before this
// object destroys the native graph handles.
class WidthBlendGraphReplay {
public:
    WidthBlendGraphReplay() = default;
    ~WidthBlendGraphReplay() { Reset(); }
    WidthBlendGraphReplay(WidthBlendGraphReplay const&) = delete;
    WidthBlendGraphReplay& operator=(WidthBlendGraphReplay const&) = delete;

    bool Capture(DeviceView<const float> left, DeviceView<const float> right,
                 float blend, DeviceView<float> output, cudaStream_t stream) {
        // A zero-width blend is a successful production no-op, but does not
        // contain a launch to capture. Reject it before beginning capture so
        // the stream remains available for the next legal capture.
        if (!stream || !left.size || left.size != right.size ||
            left.size != output.size || !left.data || !right.data || !output.data)
            return false;
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &status) != cudaSuccess ||
            status != cudaStreamCaptureStatusNone)
            return false;
        cudaGraph_t graph = nullptr;
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) != cudaSuccess)
            return false;
        bool const launched = LaunchWidthBlend(left, right, blend, output, stream);
        cudaError_t const ended = cudaStreamEndCapture(stream, &graph);
        if (!launched || ended != cudaSuccess || !graph) {
            if (graph) cudaGraphDestroy(graph);
            return false;
        }
        cudaGraphExec_t executable = nullptr;
        if (cudaGraphInstantiate(&executable, graph, 0) != cudaSuccess) {
            cudaGraphDestroy(graph);
            return false;
        }
        Reset();
        graph_ = graph;
        executable_ = executable;
        return true;
    }

    bool Launch(cudaStream_t stream) const {
        return executable_ && stream && cudaGraphLaunch(executable_, stream) == cudaSuccess;
    }

    void Reset() noexcept {
        if (executable_) cudaGraphExecDestroy(executable_);
        if (graph_) cudaGraphDestroy(graph_);
        executable_ = nullptr;
        graph_ = nullptr;
    }

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
};

static DeviceView<const float> Read(DeviceBuffer<float> const& buffer) {
    return buffer.view();
}

static bool SameWidthBlend(std::vector<float> const& left,
                           std::vector<float> const& right, float blend,
                           std::vector<float> const& actual) {
    if (left.size() != right.size() || actual.size() != left.size()) return false;
    for (size_t i = 0; i != actual.size(); ++i) {
        float const expected = blend == 0.0f ? left[i] :
            blend == 1.0f ? right[i] : left[i] + (right[i] - left[i]) * blend;
        if (blend == 0.0f || blend == 1.0f) {
            if (std::memcmp(&expected, &actual[i], sizeof(float)) != 0) return false;
        } else if (!Near(expected, actual[i])) return false;
    }
    return true;
}

static bool RunWidthBlendGraphCase(size_t count, float blend, bool swapped,
                                   bool expectGraph) {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
        return false;
    DeviceBuffer<float> left, right, uncaptured, captured;
    WidthBlendGraphReplay graph;
    bool ok = left.reset(count) == cudaSuccess && right.reset(count) == cudaSuccess &&
        uncaptured.reset(count) == cudaSuccess && captured.reset(count) == cudaSuccess;
    std::vector<float> hostLeft(count), hostRight(count), actual(count);
    for (size_t i = 0; i != count; ++i) {
        hostLeft[i] = i == 0 ? -0.0f : float(i) * 0.125f + 0.25f;
        hostRight[i] = i == 0 ? 0.0f : 7.0f - float(i) * 0.0625f;
    }
    if (swapped) std::swap(hostLeft, hostRight);
    if (ok && count) {
        ok = Upload(left, hostLeft, stream) && Upload(right, hostRight, stream) &&
            cudaStreamSynchronize(stream) == cudaSuccess;
    }
    if (ok) {
        ok = LaunchWidthBlend(Read(left), Read(right), blend, uncaptured.view(), stream) &&
            uncaptured.recordUse(stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess;
    }
    if (ok && count) {
        ok = Download(uncaptured, actual, stream) &&
            SameWidthBlend(hostLeft, hostRight, blend, actual);
    }
    bool const capturedGraph = ok && graph.Capture(Read(left), Read(right), blend,
                                                    captured.view(), stream);
    if (capturedGraph != expectGraph) ok = false;
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusInvalidated;
    if (ok && cudaStreamIsCapturing(stream, &captureStatus) != cudaSuccess)
        ok = false;
    if (ok && captureStatus != cudaStreamCaptureStatusNone) ok = false;
    if (ok && expectGraph) {
        // Replay the same captured single-kernel graph twice.  Device buffers
        // remain caller-owned and their use event is recorded after each launch.
        ok = graph.Launch(stream) && captured.recordUse(stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess && Download(captured, actual, stream) &&
            SameWidthBlend(hostLeft, hostRight, blend, actual) &&
            graph.Launch(stream) && captured.recordUse(stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess && Download(captured, actual, stream) &&
            SameWidthBlend(hostLeft, hostRight, blend, actual);
    }
    if (ok && !expectGraph) {
        // The rejected empty entry must leave this very stream capable of a
        // later legal capture; these are fresh preallocated COW test planes.
        DeviceBuffer<float> laterLeft, laterRight, laterOutput;
        WidthBlendGraphReplay later;
        std::vector<float> const oneLeft{-0.0f}, oneRight{3.5f};
        std::vector<float> oneActual(1);
        bool const laterOk = laterLeft.reset(1) == cudaSuccess && laterRight.reset(1) == cudaSuccess &&
            laterOutput.reset(1) == cudaSuccess && Upload(laterLeft, oneLeft, stream) &&
            Upload(laterRight, oneRight, stream) && cudaStreamSynchronize(stream) == cudaSuccess;
        bool const capturedLater = laterOk && later.Capture(Read(laterLeft), Read(laterRight), .25f,
                                                             laterOutput.view(), stream);
        bool const launchedLater = capturedLater && later.Launch(stream) &&
            laterOutput.recordUse(stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess;
        bool const readLater = launchedLater && Download(laterOutput, oneActual, stream) &&
            SameWidthBlend(oneLeft, oneRight, .25f, oneActual);
        ok = readLater;
        later.Reset();
    }
    graph.Reset();
    if (cudaStreamDestroy(stream) != cudaSuccess) ok = false;
    return ok;
}

int main() {
    cudaStream_t producer = nullptr, consumer = nullptr;
    CHECK(cudaStreamCreate(&producer) == cudaSuccess);
    CHECK(cudaStreamCreate(&consumer) == cudaSuccess);

    DeviceBuffer<float> widths, output, profile, maskField, pointField;
    DeviceBuffer<uint32_t> offsets;
    CHECK(widths.reset(5) == cudaSuccess && output.reset(5) == cudaSuccess &&
          offsets.reset(3) == cudaSuccess && profile.reset(257) == cudaSuccess &&
          maskField.reset(5) == cudaSuccess && pointField.reset(5) == cudaSuccess);
    CHECK(Upload(widths, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, producer));
    CHECK(Upload(offsets, {0u, 2u, 5u}, producer));
    CHECK(Upload(profile, std::vector<float>(257, 1.0f), producer));
    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
    DeviceCurveGeometryView geometry = Geometry(widths, offsets);

    // Variable 2/3-CV curves, supplied hairT, target profile arithmetic, and
    // cross-stream Finish. The profile is flat here, so every CV takes the
    // authored base width.
    std::vector<float> hairTHost{0.0f, 1.0f, 0.0f, .5f, 1.0f};
    DeviceBuffer<float> hairT;
    CHECK(hairT.reset(5) == cudaSuccess && Upload(hairT, hairTHost, producer));
    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
    WidthParameters p = Defaults({profile.data(), profile.size()});
    p.width = ScalarField::Literal(2.0f);
    CHECK(CudaWidth{}.pending() == false);
    CudaWidth op;
    CHECK(op.Apply(geometry, {hairT.data(), hairT.size()}, p,
                   {output.data(), output.size()}, producer) == StyleStatus::Ok);
    CHECK(op.Finish(consumer) == StyleStatus::Ok);
    std::vector<float> got(5);
    CHECK(Download(output, got, consumer));
    CHECK(Near(got[0], 2.0f) && Near(got[1], 2.0f) &&
          Near(got[2], 2.0f) && Near(got[3], 2.0f) && Near(got[4], 2.0f));

    // Primitive-domain broadcasting and replace=false multiplication.
    DeviceBuffer<float> primitive;
    CHECK(primitive.reset(2) == cudaSuccess &&
          Upload(primitive, {2.0f, 4.0f}, producer));
    p = Defaults({profile.data(), profile.size()});
    p.width = ScalarField::Device({primitive.data(), 2}, expr::Domain::Primitive);
    p.replace = BoolField::Literal(false);
    CHECK(op.Apply(geometry, {}, p, {output.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer));
    CHECK(Near(got[0], 2.0f) && Near(got[1], 4.0f) &&
          Near(got[2], 12.0f) && Near(got[3], 16.0f) && Near(got[4], 20.0f));

    // Point-domain controls, a non-flat width profile, and a POINT-domain
    // usdGen:mask envelope (the former mask ramp is now just an expression
    // evaluated per CV).  The canonical t per point is {0, 1, 0, .5, 1}.
    std::vector<float> profileHost(257);
    for (size_t i = 0; i < 257; ++i) profileHost[i] = 1.0f + float(i) / 256.0f;
    CHECK(Upload(profile, profileHost, producer) &&
          Upload(maskField, {.5f, 1.0f, .5f, .75f, 1.0f}, producer) &&
          Upload(pointField, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, producer));
    p = Defaults({profile.data(), profile.size()});
    p.width = ScalarField::Device({pointField.data(), 5}, expr::Domain::Point);
    p.mask = ScalarField::Device({maskField.data(), 5}, expr::Domain::Point);
    CHECK(op.Apply(geometry, {}, p, {output.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer));
    // At each canonical t, target = pointWidth * widthProfile[t], then the
    // per-CV mask is the envelope. The first and last values are exact.
    CHECK(Near(got[0], 1.0f) && Near(got[1], 4.0f) &&
          Near(got[2], 3.0f) && Near(got[3], 5.5f) &&
          Near(got[4], 10.0f));

    // Exact-zero enabled/envelope paths copy the upstream width bitwise.
    std::vector<float> sentinel{1.0f, -0.0f, 3.0f, 4.0f, 5.0f};
    CHECK(Upload(widths, sentinel, producer));
    p = Defaults({profile.data(), profile.size()});
    p.mask = ScalarField::Literal(0.0f);
    CHECK(op.Apply(geometry, {}, p, {output.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer));
    CHECK(std::memcmp(got.data(), sentinel.data(), got.size() * sizeof(float)) == 0);
    p.mask = ScalarField::Literal(1.0f);
    p.enabled = BoolField::Literal(false);
    CHECK(op.Apply(geometry, {}, p, {output.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer));
    CHECK(std::memcmp(got.data(), sentinel.data(), got.size() * sizeof(float)) == 0);

    // Device diagnostics reject malformed offsets, nonfinite/negative source
    // widths and invalid profile values without publishing staging output.
    CHECK(Upload(widths, std::vector<float>(5, 91.0f), producer) &&
          Upload(output, std::vector<float>(5, 91.0f), producer) &&
          Upload(offsets, {0u, 2u, 9u}, producer));
    CHECK(op.Apply(geometry, {}, Defaults({profile.data(), profile.size()}),
                   {output.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(op.Finish(consumer) == StyleStatus::InvalidArgument);
    CHECK(Download(output, got, consumer));
    for (float value : got) CHECK(Near(value, 91.0f));
    CHECK(Upload(offsets, {0u, 2u, 5u}, producer));
    std::vector<float> badWidths{1.0f, NAN, 3.0f, 4.0f, 5.0f};
    CHECK(Upload(widths, badWidths, producer));
    p = Defaults({profile.data(), profile.size()});
    CHECK(op.Apply(geometry, {}, p, {output.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(op.Finish(consumer) == StyleStatus::NonFiniteInput);
    CHECK(Upload(widths, std::vector<float>(5, 91.0f), producer));
    p.width = ScalarField::Literal(-1.0f);
    CHECK(op.Apply(geometry, {}, p, {output.data(), 5}, producer) ==
          StyleStatus::InvalidValue);
    std::vector<float> badProfile(257, 1.0f);
    badProfile[12] = -1.0f;
    CHECK(Upload(profile, badProfile, producer));
    p = Defaults({profile.data(), profile.size()});
    CHECK(op.Apply(geometry, {}, p, {output.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(op.Finish(consumer) == StyleStatus::InvalidValue);
    CHECK(Upload(profile, std::vector<float>(257, 1.0f), producer));

    // Host-invalid field counts queue no work; a valid operation can follow
    // immediately on another stream. Device bool values are byte-validated.
    p = Defaults({profile.data(), profile.size()});
    p.width = ScalarField::Device({primitive.data(), 1}, expr::Domain::Primitive);
    CHECK(op.Apply(geometry, {}, p, {output.data(), 5}, producer) ==
          StyleStatus::InvalidArgument);
    p.width = ScalarField::Literal(2.0f);
    std::vector<unsigned char> badBool{2u, 1u};
    DeviceBuffer<unsigned char> boolBytes;
    CHECK(boolBytes.reset(2) == cudaSuccess && Upload(boolBytes, badBool, producer));
    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
    p.replace = BoolField::Device(
        {reinterpret_cast<const uint8_t *>(boolBytes.data()), 2},
        expr::Domain::Primitive);
    CHECK(op.Apply(geometry, {}, p, {output.data(), 5}, consumer) == StyleStatus::Ok);
    CHECK(op.Finish(producer) == StyleStatus::InvalidValue);
    p.replace = BoolField::Literal(true);
    CHECK(op.Apply(geometry, {}, p, {output.data(), 5}, consumer) == StyleStatus::Ok);
    CHECK(op.Finish(producer) == StyleStatus::Ok);

    // Fresh Width publishes only after its stream terminal callback.  Commit
    // before that proof and a duplicate terminal installation are rejected.
    CudaWidth fresh;
    CHECK(Upload(widths, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, producer));
    p = Defaults({profile.data(), profile.size()}); p.width = ScalarField::Literal(2.0f);
    freshCallbacks.store(0); freshStatus.store(int(cudaErrorUnknown));
    CHECK(fresh.ApplyFresh(geometry, {}, p, {output.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(fresh.Apply(geometry, {}, p, {output.data(), 5}, producer) == StyleStatus::InvalidArgument);
    CHECK(fresh.Finish(producer) == StyleStatus::InvalidArgument);
    CHECK(fresh.CommitFreshFinish() == StyleStatus::InvalidArgument);
    CHECK(fresh.FinishFreshAsync(producer, FreshSignal, nullptr) == StyleStatus::Ok);
    CHECK(fresh.FinishFreshAsync(producer, FreshSignal, nullptr) == StyleStatus::InvalidArgument);
    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
    CHECK(freshCallbacks.load(std::memory_order_acquire) == 1 &&
          cudaError_t(freshStatus.load(std::memory_order_acquire)) == cudaSuccess);
    CHECK(fresh.CommitFreshFinish() == StyleStatus::Ok);
    CHECK(Download(output, got, consumer));
    for (float value : got) CHECK(Near(value, 2.0f));

    // A device-semantic failure reaches the callback but conditional
    // publication leaves output unchanged.
    CHECK(Upload(output, std::vector<float>(5, 73.0f), producer));
    std::vector<float> freshBadProfile(257, 1.0f); freshBadProfile[7] = NAN;
    CHECK(Upload(profile, freshBadProfile, producer));
    freshCallbacks.store(0); freshStatus.store(int(cudaErrorUnknown));
    CHECK(fresh.ApplyFresh(geometry, {}, Defaults({profile.data(), profile.size()}),
                           {output.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(fresh.FinishFreshAsync(producer, FreshSignal, nullptr) == StyleStatus::Ok);
    CHECK(cudaStreamSynchronize(producer) == cudaSuccess && freshCallbacks == 1);
    CHECK(fresh.CommitFreshFinish() == StyleStatus::NonFiniteInput);
    CHECK(Download(output, got, consumer));
    for (float value : got) CHECK(Near(value, 73.0f));
    CHECK(Upload(profile, std::vector<float>(257, 1.0f), producer));

    // Offset validation runs before WidthKernel dereferences a span; fresh
    // conditional publication preserves the caller's output on that error.
    CHECK(Upload(output, std::vector<float>(5, 61.0f), producer) &&
          Upload(offsets, {0u, UINT32_MAX, 5u}, producer));
    freshCallbacks.store(0); freshStatus.store(int(cudaErrorUnknown));
    CHECK(fresh.ApplyFresh(geometry, {}, Defaults({profile.data(), profile.size()}),
                           {output.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(fresh.FinishFreshAsync(producer, FreshSignal, nullptr) == StyleStatus::Ok);
    CHECK(cudaStreamSynchronize(producer) == cudaSuccess && freshCallbacks == 1);
    CHECK(fresh.CommitFreshFinish() == StyleStatus::InvalidArgument);
    CHECK(Download(output, got, consumer));
    for (float value : got) CHECK(Near(value, 61.0f));
    CHECK(Upload(offsets, {0u, 2u, 5u}, producer));

    // Capture is rejected before fresh allocation/submission.
    cudaGraph_t graph = nullptr;
    CHECK(cudaStreamBeginCapture(producer, cudaStreamCaptureModeGlobal) == cudaSuccess);
    CHECK(fresh.ApplyFresh(geometry, {}, Defaults({profile.data(), profile.size()}),
                           {output.data(), 5}, producer) == StyleStatus::InvalidArgument);
    CHECK(cudaStreamEndCapture(producer, &graph) == cudaSuccess);
    if (graph) CHECK(cudaGraphDestroy(graph) == cudaSuccess);
    CHECK(fresh.Apply(geometry, {}, Defaults({profile.data(), profile.size()}),
                      {output.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(fresh.Finish(consumer) == StyleStatus::Ok);

    // Empty geometry is a successful no-op; profile shape remains required.
    DeviceCurveGeometryView empty{{}, {}, {}, {}, {}, 0, 0};
    CudaWidth emptyOp;
    CHECK(emptyOp.Apply(empty, {}, Defaults({profile.data(), profile.size()}),
                        {}, producer) == StyleStatus::Ok);
    CHECK(emptyOp.Finish(consumer) == StyleStatus::Ok);

    // CUDA Graph scope is deliberately one preallocated, allocation-free
    // WidthBlend launch. This test does not make it an execution-plan cache.
    // Zero remains the leaf API's valid no-op; the helper rejects an empty
    // capture before BeginCapture and proves the same stream can then capture
    // a legal nonempty blend.
    CHECK(RunWidthBlendGraphCase(0, .25f, false, false));
    CHECK(RunWidthBlendGraphCase(1, .25f, false, true));
    CHECK(RunWidthBlendGraphCase(255, .25f, false, true));
    CHECK(RunWidthBlendGraphCase(256, .25f, false, true));
    CHECK(RunWidthBlendGraphCase(257, .25f, false, true));
    CHECK(RunWidthBlendGraphCase(257, 0.0f, false, true));
    CHECK(RunWidthBlendGraphCase(257, 1.0f, false, true));
    CHECK(RunWidthBlendGraphCase(257, .25f, true, true));

    // A terminal stream-synchronization failure cannot prove the producer
    // event or diagnostic buffer is complete.  Capture makes the failure
    // deterministic without passing a stale native stream handle to CUDA.
    {
        CudaWidth unproven;
        CHECK(unproven.Apply(geometry, {}, Defaults({profile.data(), profile.size()}),
                             {output.data(), 5}, producer) == StyleStatus::Ok);
        cudaGraph_t graph = nullptr;
        CHECK(cudaStreamBeginCapture(consumer, cudaStreamCaptureModeGlobal) == cudaSuccess);
        CHECK(unproven.Finish(consumer) != StyleStatus::Ok &&
              unproven.HasUnprovenWork());
        CHECK(cudaStreamEndCapture(consumer, &graph) != cudaSuccess);
        if (graph) CHECK(cudaGraphDestroy(graph) == cudaSuccess);
    }

    cudaStreamDestroy(producer);
    cudaStreamDestroy(consumer);
    return 0;
}
