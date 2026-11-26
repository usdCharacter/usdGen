// Test-only readback oracle for device-only conservative per-tile bounds.
#include "usdGenCurves/cuda/curveTileBounds.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace usdGen::gpu;

namespace {

int failures = 0;

void Check(bool value, char const* label) {
    if (!value) { ++failures; std::fprintf(stderr, "FAIL: %s\n", label); }
}

void Cuda(cudaError_t value, char const* label) {
    if (value != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(value));
        std::exit(1);
    }
}

template <class T> DeviceView<const T> Read(DeviceBuffer<T> const& buffer) {
    return buffer.view();
}

template <class T> std::vector<T> Download(DeviceBuffer<T> const& buffer) {
    std::vector<T> result(buffer.size());
    if (!result.empty()) Cuda(cudaMemcpy(result.data(), buffer.data(),
        result.size() * sizeof(T), cudaMemcpyDeviceToHost), "bounds oracle readback");
    return result;
}

bool Near(float actual, float expected) {
    return actual <= expected && actual >= expected - 1.0e-4f;
}

bool LowerNear(float3 actual, float3 expected) {
    return Near(actual.x, expected.x) && Near(actual.y, expected.y) &&
        Near(actual.z, expected.z);
}

bool UpperNear(float3 actual, float3 expected) {
    return actual.x >= expected.x && actual.x <= expected.x + 1.0e-4f &&
        actual.y >= expected.y && actual.y <= expected.y + 1.0e-4f &&
        actual.z >= expected.z && actual.z <= expected.z + 1.0e-4f;
}

constexpr float3 kSentinel = {1234.0f, -5678.0f, 91011.0f};

struct Fixture {
    cudaStream_t stream = nullptr;
    DeviceBuffer<float3> points;
    DeviceBuffer<float> widths;
    DeviceBuffer<CurveTileSpan> spans;
    DeviceBuffer<CurveTileBoundsScratch> scratch;
    DeviceBuffer<float3> minimums;
    DeviceBuffer<float3> maximums;
    DeviceBuffer<uint32_t> status;

    Fixture() {
        Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
             "bounds stream");
        Cuda(points.reset(5), "bounds points allocation");
        Cuda(widths.reset(5), "bounds widths allocation");
        Cuda(spans.reset(3), "bounds span allocation");
        Cuda(scratch.reset(3), "bounds scratch allocation");
        Cuda(minimums.reset(3), "bounds minimum allocation");
        Cuda(maximums.reset(3), "bounds maximum allocation");
        Cuda(status.reset(1), "bounds status allocation");
    }
    ~Fixture() { if (stream) cudaStreamDestroy(stream); }

    CurveTileBoundsInput Input() const {
        return {Read(points), Read(widths), Read(spans)};
    }
    CurveTileBoundsWorkspace Workspace() { return {scratch.view()}; }
    CurveTileBoundsOutput Output() {
        return {minimums.view(), maximums.view(), status.view()};
    }
};

std::array<float3, 5> const kPoints{{
    {1.0f, 2.0f, 3.0f}, {3.0f, 4.0f, 5.0f}, {0.0f, 0.0f, 0.0f},
    {8.0f, 4.0f, 2.0f}, {3.0f, 1.0f, 1.0f}}};
std::array<float, 5> const kWidths{{2.0f, 4.0f, 1.0f, 4.0f, 2.0f}};
std::array<CurveTileSpan, 3> const kSpans{{
    {7, 0, 1, 0, 2}, {42, 1, 1, 2, 3}, {99, 2, 0, 5, 0}}};

void UploadFixture(Fixture* fixture) {
    Cuda(cudaMemcpyAsync(fixture->points.data(), kPoints.data(),
        sizeof(kPoints), cudaMemcpyHostToDevice, fixture->stream), "bounds point upload");
    Cuda(cudaMemcpyAsync(fixture->widths.data(), kWidths.data(),
        sizeof(kWidths), cudaMemcpyHostToDevice, fixture->stream), "bounds width upload");
    Cuda(cudaMemcpyAsync(fixture->spans.data(), kSpans.data(),
        sizeof(kSpans), cudaMemcpyHostToDevice, fixture->stream), "bounds span upload");
}

void SetSentinel(Fixture* fixture) {
    std::array<float3, 3> values{{kSentinel, kSentinel, kSentinel}};
    Cuda(cudaMemcpyAsync(fixture->minimums.data(), values.data(), sizeof(values),
        cudaMemcpyHostToDevice, fixture->stream), "bounds minimum sentinel");
    Cuda(cudaMemcpyAsync(fixture->maximums.data(), values.data(), sizeof(values),
        cudaMemcpyHostToDevice, fixture->stream), "bounds maximum sentinel");
    // `values` is stack storage. Retire its asynchronous transfers before
    // returning, rather than relying on pageable-copy implementation details.
    Cuda(cudaStreamSynchronize(fixture->stream), "bounds sentinel upload completion");
}

cudaError_t Enqueue(Fixture* fixture, CurveTileBoundsBasis basis) {
    return BuildCurveTileBounds({basis}, fixture->Input(), fixture->Workspace(),
                                fixture->Output(), fixture->stream);
}

bool SentinelOutputs(Fixture const& fixture) {
    auto const minimums = Download(fixture.minimums);
    auto const maximums = Download(fixture.maximums);
    for (size_t i = 0; i != minimums.size(); ++i) {
        if (minimums[i].x != kSentinel.x || minimums[i].y != kSentinel.y ||
            minimums[i].z != kSentinel.z || maximums[i].x != kSentinel.x ||
            maximums[i].y != kSentinel.y || maximums[i].z != kSentinel.z)
            return false;
    }
    return true;
}

void CheckBounds(Fixture const& fixture, size_t tile, float3 minimum,
                 float3 maximum, char const* label) {
    auto const actualMinimums = Download(fixture.minimums);
    auto const actualMaximums = Download(fixture.maximums);
    Check(tile < actualMinimums.size() && LowerNear(actualMinimums[tile], minimum) &&
          UpperNear(actualMaximums[tile], maximum), label);
}

uint32_t Status(Fixture const& fixture) { return Download(fixture.status)[0]; }

void ValidBounds() {
    Fixture fixture;
    UploadFixture(&fixture);
    SetSentinel(&fixture);
    Cuda(Enqueue(&fixture, CurveTileBoundsBasis::Linear), "linear bounds enqueue");
    Cuda(cudaStreamSynchronize(fixture.stream), "linear bounds completion");
    Check(Status(fixture) == 0, "linear bounds status is valid");
    // Tile 1 deliberately has a nonzero global point base. Outputs are by
    // span-array position, not the stable tile IDs 7/42/99.
    CheckBounds(fixture, 0, {-1, 0, 1}, {5, 6, 7},
                "linear whole-span max-width pad");
    CheckBounds(fixture, 1, {-2, -2, -2}, {10, 6, 4},
                "linear nonzero-base tile is addressed by span position");
    CheckBounds(fixture, 2, {0, 0, 0}, {0, 0, 0},
                "empty retained tile has deterministic zero bounds");

    SetSentinel(&fixture);
    Cuda(Enqueue(&fixture, CurveTileBoundsBasis::BSpline), "bspline bounds enqueue");
    Cuda(cudaStreamSynchronize(fixture.stream), "bspline bounds completion");
    Check(Status(fixture) == 0, "bspline bounds status is valid");
    CheckBounds(fixture, 1, {-2, -2, -2}, {10, 6, 4},
                "convex B-spline hull needs only width radius");

    SetSentinel(&fixture);
    Cuda(Enqueue(&fixture, CurveTileBoundsBasis::CatmullRom), "catmull bounds enqueue");
    Cuda(cudaStreamSynchronize(fixture.stream), "catmull bounds completion");
    Check(Status(fixture) == 0, "catmull bounds status is valid");
    CheckBounds(fixture, 1, {-3.25f, -2.75f, -2.5f},
                {11.25f, 6.75f, 4.5f},
                "Catmull-Rom covers signed basis overshoot and widened width");
}

void InvalidDeviceInput() {
    Fixture fixture;
    UploadFixture(&fixture);
    auto invalid = [&](char const* label) {
        SetSentinel(&fixture);
        Cuda(Enqueue(&fixture, CurveTileBoundsBasis::BSpline), "invalid bounds enqueue");
        Cuda(cudaStreamSynchronize(fixture.stream), "invalid bounds completion");
        Check(Status(fixture) == 1 && SentinelOutputs(fixture), label);
    };

    auto spans = kSpans;
    spans[1].firstPoint = UINT32_MAX;
    spans[1].pointCount = 1;
    Cuda(cudaMemcpyAsync(fixture.spans.data(), spans.data(), sizeof(spans),
        cudaMemcpyHostToDevice, fixture.stream), "invalid span upload");
    invalid("out-of-range span rejects without publishing any tile output");

    UploadFixture(&fixture);
    auto points = kPoints;
    points[0].x = std::numeric_limits<float>::quiet_NaN();
    Cuda(cudaMemcpyAsync(fixture.points.data(), points.data(), sizeof(points),
        cudaMemcpyHostToDevice, fixture.stream), "nonfinite point upload");
    invalid("nonfinite point rejects without publishing any tile output");

    UploadFixture(&fixture);
    auto widths = kWidths;
    widths[0] = -1.0f;
    Cuda(cudaMemcpyAsync(fixture.widths.data(), widths.data(), sizeof(widths),
        cudaMemcpyHostToDevice, fixture.stream), "negative width upload");
    invalid("negative width rejects without publishing any tile output");

    UploadFixture(&fixture);
    points = kPoints;
    points[0].x = std::numeric_limits<float>::max();
    Cuda(cudaMemcpyAsync(fixture.points.data(), points.data(), sizeof(points),
        cudaMemcpyHostToDevice, fixture.stream), "overflow point upload");
    invalid("unrepresentable final float bound rejects without publishing output");
}

void ShapeAndEmptyInput() {
    Fixture fixture;
    UploadFixture(&fixture);
    SetSentinel(&fixture);
    uint32_t sentinel = 0x5a5a5a5aU;
    Cuda(cudaMemcpyAsync(fixture.status.data(), &sentinel, sizeof(sentinel),
        cudaMemcpyHostToDevice, fixture.stream), "shape status sentinel");
    CurveTileBoundsInput badWidth{Read(fixture.points),
        {fixture.widths.data(), fixture.widths.size() - 1}, Read(fixture.spans)};
    Check(BuildCurveTileBounds({}, badWidth, fixture.Workspace(), fixture.Output(),
                               fixture.stream) == cudaErrorInvalidValue,
          "mismatched point and width shapes reject on the host");
    Cuda(cudaStreamSynchronize(fixture.stream), "shape rejection completion");
    Check(Status(fixture) == sentinel && SentinelOutputs(fixture),
          "host shape rejection enqueues no status or output write");
    Check(BuildCurveTileBounds({static_cast<CurveTileBoundsBasis>(99)}, fixture.Input(),
                               fixture.Workspace(), fixture.Output(), fixture.stream) ==
              cudaErrorInvalidValue,
          "unsupported basis rejects on the host");

    DeviceBuffer<uint32_t> status;
    Cuda(status.reset(1), "empty status allocation");
    uint32_t nonzero = 99;
    Cuda(cudaMemcpyAsync(status.data(), &nonzero, sizeof(nonzero), cudaMemcpyHostToDevice,
        fixture.stream), "empty status sentinel");
    CurveTileBoundsInput emptyInput{};
    CurveTileBoundsWorkspace emptyWorkspace{};
    CurveTileBoundsOutput emptyOutput{{}, {}, status.view()};
    Cuda(BuildCurveTileBounds({}, emptyInput, emptyWorkspace, emptyOutput,
                              fixture.stream), "empty bounds enqueue");
    Cuda(cudaStreamSynchronize(fixture.stream), "empty bounds completion");
    Check(Download(status)[0] == 0, "zero-tile capture is valid and clears status");
}

void GraphCaptureReplay() {
    Fixture fixture;
    UploadFixture(&fixture);
    SetSentinel(&fixture);
    Cuda(cudaStreamBeginCapture(fixture.stream, cudaStreamCaptureModeGlobal),
         "bounds graph capture begin");
    Cuda(Enqueue(&fixture, CurveTileBoundsBasis::BSpline), "bounds graph capture enqueue");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    Cuda(cudaStreamEndCapture(fixture.stream, &graph), "bounds graph capture end");
    Cuda(cudaGraphInstantiate(&executable, graph, 0), "bounds graph instantiate");
    Cuda(cudaGraphLaunch(executable, fixture.stream), "bounds graph launch");
    Cuda(cudaStreamSynchronize(fixture.stream), "bounds graph completion");
    Check(Status(fixture) == 0, "captured bounds graph is valid");
    CheckBounds(fixture, 1, {-2, -2, -2}, {10, 6, 4},
                "captured graph initial nonzero-base bounds");

    auto points = kPoints;
    points[3].x = 10.0f;
    Cuda(cudaMemcpyAsync(fixture.points.data(), points.data(), sizeof(points),
        cudaMemcpyHostToDevice, fixture.stream), "bounds graph replay point upload");
    Cuda(cudaGraphLaunch(executable, fixture.stream), "bounds graph replay");
    Cuda(cudaStreamSynchronize(fixture.stream), "bounds graph replay completion");
    Check(Status(fixture) == 0, "replayed bounds graph is valid");
    CheckBounds(fixture, 1, {-2, -2, -2}, {12, 6, 4},
                "replayed graph reduces changed device points");
    Cuda(cudaGraphExecDestroy(executable), "bounds graph executable destroy");
    Cuda(cudaGraphDestroy(graph), "bounds graph destroy");
}

} // namespace

int main() {
    ValidBounds();
    InvalidDeviceInput();
    ShapeAndEmptyInput();
    GraphCaptureReplay();
    std::printf("GPU curve tile bounds: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
