#include "usdGenCurves/cuda/curveSource.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace usdGen::gpu;

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", \
                                      #condition, __FILE__, __LINE__); return 1; } \
} while (false)

static float3 V(float x, float y, float z) { return make_float3(x, y, z); }
static bool Near(float a, float b) { return std::fabs(a - b) < 2.0e-5f; }
static bool Near(float3 a, float3 b) {
    return Near(a.x, b.x) && Near(a.y, b.y) && Near(a.z, b.z);
}

static void FreshCallback(cudaStream_t, cudaError_t, void*) noexcept {}

template <class T>
static bool Download(DeviceView<const T> view, std::vector<T> &host,
                     cudaStream_t stream) {
    return cudaMemcpyAsync(host.data(), view.data, host.size() * sizeof(T),
                           cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
           cudaStreamSynchronize(stream) == cudaSuccess;
}

int main() {
    int originalDevice = -1, deviceCount = 0;
    CHECK(cudaGetDevice(&originalDevice) == cudaSuccess);
    CHECK(cudaGetDeviceCount(&deviceCount) == cudaSuccess);
    cudaStream_t upload = nullptr, consumer = nullptr;
    CHECK(cudaStreamCreate(&upload) == cudaSuccess);
    CHECK(cudaStreamCreate(&consumer) == cudaSuccess);

    std::vector<int32_t> counts{2, 3};
    std::vector<float3> points{V(0, 0, 0), V(0, 0, 1), V(1, 0, 0),
                               V(1, 0, 1), V(1, 0, 2)};
    std::vector<float3> rest{V(0, 0, 0), V(0, 0, .9f), V(1, 0, 0),
                             V(1, 0, .8f), V(1, 0, 1.7f)};
    std::vector<float> widths{.2f, .1f, .4f, .3f, .2f};
    std::vector<float> authoredT{0, 1, 0, .25f, 1};
    std::vector<uint64_t> ids{41, 99};
    std::vector<int32_t> rootPrim{4, 8};
    std::vector<float2> rootUV{make_float2(.1f, .2f), make_float2(.3f, .4f)};
    CurveSourceInput input{{counts.data(), counts.size()},
                           {points.data(), points.size()},
                           {rest.data(), rest.size()},
                           {widths.data(), widths.size()},
                           {authoredT.data(), authoredT.size()},
                           {ids.data(), ids.size()},
                           {rootPrim.data(), rootPrim.size()},
                           {rootUV.data(), rootUV.size()}};

    CudaCurveSource source;
    CHECK(source.Finish(consumer) == CurveSourceStatus::NoPendingUpdate);
    CHECK(source.Set(input, upload) == CurveSourceStatus::Ok);
    CHECK(source.generation() == 0 && source.view().curveCount == 0);
    CHECK(source.Finish(consumer) == CurveSourceStatus::Ok);
    CHECK(source.generation() == 1 && source.curveCount() == 2 &&
          source.pointCount() == 5 &&
          source.warningFlags() == CurveSourceWarningNone);
    CHECK(source.deviceIndex() == originalDevice);
    DeviceCurveGeometryView first = source.view();
    CHECK(first.points.size == 5 && first.restPoints.size == 5 &&
          first.widths.size == 5 && first.curveOffsets.size == 3 &&
          first.stableIds.size == 2);
    std::vector<float> gotT(5), gotWidths(5);
    std::vector<float3> gotPoints(5), gotRest(5);
    std::vector<uint32_t> gotOffsets(3);
    std::vector<uint64_t> gotIds(2);
    std::vector<int32_t> gotRootPrim(2);
    std::vector<float2> gotRootUV(2);
    CHECK(Download(source.hairT(), gotT, consumer) &&
          Download(first.points, gotPoints, consumer) &&
          Download(first.widths, gotWidths, consumer) &&
          Download(first.restPoints, gotRest, consumer) &&
          Download(first.curveOffsets, gotOffsets, consumer) &&
          Download(first.stableIds, gotIds, consumer) &&
          Download(source.rootPrim(), gotRootPrim, consumer) &&
          Download(source.rootUV(), gotRootUV, consumer));
    for (size_t i = 0; i < 5; ++i)
        CHECK(Near(gotT[i], authoredT[i]) && Near(gotPoints[i], points[i]) &&
              Near(gotWidths[i], widths[i]) &&
              Near(gotRest[i], rest[i]));
    CHECK(gotOffsets[0] == 0 && gotOffsets[1] == 2 && gotOffsets[2] == 5 &&
          gotIds[0] == 41 && gotIds[1] == 99);
    CHECK(gotRootPrim[0] == 4 && gotRootPrim[1] == 8 &&
          Near(gotRootUV[0].x, .1f) && Near(gotRootUV[0].y, .2f) &&
          Near(gotRootUV[1].x, .3f) && Near(gotRootUV[1].y, .4f));

    // A source is bound to the device on its first upload.  Both a stream
    // created on another device and a changed current device are rejected
    // before any host-to-device work is queued.
    if (deviceCount > 1) {
        const int otherDevice = originalDevice == 0 ? 1 : 0;
        cudaStream_t foreign = nullptr;
        CHECK(cudaSetDevice(otherDevice) == cudaSuccess);
        CHECK(cudaStreamCreate(&foreign) == cudaSuccess);
        CHECK(cudaSetDevice(originalDevice) == cudaSuccess);
        CHECK(source.Set(input, foreign) == CurveSourceStatus::InvalidArgument);
        CHECK(source.deviceIndex() == originalDevice && !source.pending());
        CHECK(cudaSetDevice(otherDevice) == cudaSuccess);
        CHECK(source.Set(input, upload) == CurveSourceStatus::InvalidArgument);
        CHECK(cudaSetDevice(originalDevice) == cudaSuccess);
        CHECK(source.Set(input, upload) == CurveSourceStatus::Ok);
        CHECK(source.Finish(foreign) == CurveSourceStatus::InvalidArgument);
        CHECK(source.pending());
        CHECK(cudaSetDevice(otherDevice) == cudaSuccess);
        CHECK(cudaStreamDestroy(foreign) == cudaSuccess);
        CHECK(cudaSetDevice(originalDevice) == cudaSuccess);
        CHECK(source.Finish(consumer) == CurveSourceStatus::Ok);
        CHECK(source.generation() == 2);
    }
    const uint64_t initialGeneration = deviceCount > 1 ? 2 : 1;

    // Every host-invalid update leaves the last valid device generation intact.
    std::vector<int32_t> badCounts{2, 0};
    CurveSourceInput badTopology = input;
    badTopology.curveVertexCounts = {badCounts.data(), badCounts.size()};
    CHECK(source.Set(badTopology, upload) == CurveSourceStatus::InvalidTopology);
    CHECK(source.generation() == initialGeneration && source.view().curveCount == 2);
    std::vector<float3> badPoints = points;
    badPoints[3].x = NAN;
    CurveSourceInput badFinite = input;
    badFinite.points = {badPoints.data(), badPoints.size()};
    CHECK(source.Set(badFinite, upload) == CurveSourceStatus::NonFiniteInput);
    CHECK(source.generation() == initialGeneration);
    std::vector<uint64_t> duplicateIds{41, 41};
    CurveSourceInput badIds = input;
    badIds.stableIds = {duplicateIds.data(), duplicateIds.size()};
    CHECK(source.Set(badIds, upload) == CurveSourceStatus::DuplicateStableId);
    CHECK(source.generation() == initialGeneration);
    CurveSourceInput badOptional = input;
    badOptional.rootUV = {nullptr, 1};
    CHECK(source.Set(badOptional, upload) == CurveSourceStatus::InvalidArgument);
    CHECK(source.generation() == initialGeneration);

    // Optional channels use canonical fallback data, while the ragged counts
    // and authored points remain unchanged.
    CurveSourceInput fallback{input.curveVertexCounts, input.points, {}, {}, {}, {}};
    CHECK(source.Set(fallback, upload) == CurveSourceStatus::Ok);
    CHECK(source.Finish(consumer) == CurveSourceStatus::Ok);
    CHECK(source.warningFlags() ==
          (CurveSourceWarningSynthesizedRest | CurveSourceWarningSynthesizedWidths |
           CurveSourceWarningSynthesizedHairT | CurveSourceWarningSynthesizedStableIds));
    DeviceCurveGeometryView second = source.view();
    CHECK(second.curveCount == 2 && second.pointCount == 5);
    CHECK(Download(source.hairT(), gotT, consumer) &&
          Download(second.widths, gotWidths, consumer) &&
          Download(second.restPoints, gotRest, consumer) &&
          Download(second.stableIds, gotIds, consumer));
    CHECK(Near(gotT[0], 0) && Near(gotT[1], 1) && Near(gotT[2], 0) &&
          Near(gotT[3], .5f) && Near(gotT[4], 1));
    for (float width : gotWidths) CHECK(Near(width, .01f));
    for (size_t i = 0; i < 5; ++i) CHECK(Near(gotRest[i], points[i]));
    CHECK(gotIds[0] == 0 && gotIds[1] == 1);

    // Fresh async commit is never allowed to replace an active generation;
    // legacy Finish remains the general replacement path and can commit it.
    const uint64_t activeBeforeFreshReject = source.generation();
    CHECK(source.Set(input, upload) == CurveSourceStatus::Ok);
    CHECK(source.CommitFreshFinish() == CurveSourceStatus::InvalidArgument &&
          source.generation() == activeBeforeFreshReject && source.pending());
    CHECK(source.Finish(consumer) == CurveSourceStatus::Ok &&
          source.generation() == activeBeforeFreshReject + 1);

    // Capture rejects the fresh callback path after Set has queued its
    // upload, marking the source unsafe until legacy Finish proves terminal
    // completion and commits the pending generation.
    CudaCurveSource captureRecovery;
    CHECK(captureRecovery.Set(input, upload) == CurveSourceStatus::Ok);
    CHECK(cudaStreamBeginCapture(upload, cudaStreamCaptureModeGlobal) == cudaSuccess);
    CHECK(captureRecovery.FinishFreshAsync(upload, FreshCallback, nullptr) ==
          CurveSourceStatus::InvalidArgument && captureRecovery.HasUnprovenUpload());
    cudaGraph_t capturedGraph = nullptr;
    CHECK(cudaStreamEndCapture(upload, &capturedGraph) == cudaSuccess);
    if (capturedGraph) CHECK(cudaGraphDestroy(capturedGraph) == cudaSuccess);
    CHECK(captureRecovery.Finish(upload) == CurveSourceStatus::Ok &&
          !captureRecovery.HasUnprovenUpload() && captureRecovery.generation() == 1);

    // A known consumer use is ordered before replacing the active storage.
    const uint64_t generationBeforeLegacyReplacement = source.generation();
    CHECK(source.recordUse(consumer) == CurveSourceStatus::Ok);
    CHECK(source.Set(input, upload) == CurveSourceStatus::Ok);
    CHECK(source.Finish(upload) == CurveSourceStatus::Ok);
    CHECK(source.generation() == generationBeforeLegacyReplacement + 1);

    // Empty source is a valid owned generation; non-empty point payload without
    // curves is not a valid topology.
    CurveSourceInput empty{};
    CHECK(source.Set(empty, upload) == CurveSourceStatus::Ok);
    CHECK(source.Finish(consumer) == CurveSourceStatus::Ok);
    CHECK(source.view().curveCount == 0 && source.view().pointCount == 0 &&
          source.view().curveOffsets.size == 1);
    CurveSourceInput impossible{};
    impossible.points = {points.data(), 1};
    CHECK(source.Set(impossible, upload) == CurveSourceStatus::InvalidArgument);

    cudaStreamDestroy(upload);
    cudaStreamDestroy(consumer);
    return 0;
}
