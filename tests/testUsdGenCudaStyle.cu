#include "usdGenCurves/cuda/styleOps.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace usdGen::gpu;
using namespace usdGen;

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", \
                                      #condition, __FILE__, __LINE__); return 1; } \
} while (false)

static float3 V(float x, float y, float z) { return make_float3(x, y, z); }
static bool Near(float a, float b) { return std::fabs(a - b) < 2.0e-5f; }
static bool Near(float3 a, float3 b) {
    return Near(a.x, b.x) && Near(a.y, b.y) && Near(a.z, b.z);
}

template <class T>
static bool Upload(DeviceBuffer<T> &device, std::vector<T> const &host,
                   cudaStream_t stream) {
    return cudaMemcpyAsync(device.data(), host.data(), host.size() * sizeof(T),
                           cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

template <class T>
static bool Download(std::vector<T> &host, DeviceBuffer<T> const &device,
                     cudaStream_t stream) {
    return cudaMemcpyAsync(host.data(), device.data(), host.size() * sizeof(T),
                           cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
           cudaStreamSynchronize(stream) == cudaSuccess;
}

int main() {
    cudaStream_t producer = nullptr, consumer = nullptr;
    CHECK(cudaStreamCreate(&producer) == cudaSuccess);
    CHECK(cudaStreamCreate(&consumer) == cudaSuccess);

    DeviceBuffer<float3> points, roots, normals, vectorOutput;
    DeviceBuffer<uint32_t> offsets;
    DeviceBuffer<float> primitive, point, widths;
    CHECK(points.reset(5) == cudaSuccess && roots.reset(2) == cudaSuccess &&
          normals.reset(2) == cudaSuccess && vectorOutput.reset(5) == cudaSuccess &&
          offsets.reset(3) == cudaSuccess && primitive.reset(2) == cudaSuccess &&
          point.reset(5) == cudaSuccess && widths.reset(5) == cudaSuccess);
    CHECK(Upload(points, {V(0, 0, 0), V(0, 0, 1), V(1, 0, 0),
                          V(1, 0, 1), V(1, 0, 2)}, producer));
    CHECK(Upload(offsets, {0u, 2u, 5u}, producer));
    CHECK(Upload(roots, {V(0, 0, 0), V(1, 0, 0)}, producer));
    CHECK(Upload(normals, {V(0, 2, 0), V(0, 0, 3)}, producer));
    CHECK(Upload(primitive, {2.0f, 4.0f}, producer));
    CHECK(Upload(point, {0.0f, .25f, 0.0f, .5f, 1.0f}, producer));
    CHECK(Upload(widths, {-7.0f, -7.0f, -7.0f, -7.0f, -7.0f}, producer));
    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);

    DeviceCurveGeometryView geometry{
        {points.data(), 5}, {}, {}, {offsets.data(), 3}, {}, 2, 5};
    CudaStyleOps style;

    // Primitive fields broadcast across every CV of their owning curve.
    CHECK(style.WidthRamp(geometry, ScalarField::Literal(1.0f),
                          ScalarField::Device({primitive.data(), 2}, expr::Domain::Primitive),
                          {widths.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
    std::vector<float> beforePublish(5, -7.0f);
    CHECK(Download(beforePublish, widths, producer));
    for (float value : beforePublish) CHECK(Near(value, -7.0f));
    CHECK(style.Finish(consumer) == StyleStatus::Ok);
    std::vector<float> widthResult(5);
    CHECK(Download(widthResult, widths, consumer));
    CHECK(Near(widthResult[0], 1.0f) && Near(widthResult[1], 2.0f) &&
          Near(widthResult[2], 1.0f) && Near(widthResult[3], 2.5f) &&
          Near(widthResult[4], 4.0f));

    // Point-domain fields are indexed per CV (rather than broadcast by curve).
    CHECK(style.WidthRamp(geometry,
                          ScalarField::Device({point.data(), 5}, expr::Domain::Point),
                          ScalarField::Literal(10.0f), {widths.data(), 5}, producer) ==
          StyleStatus::Ok);
    CHECK(style.Finish(consumer) == StyleStatus::Ok);
    CHECK(Download(widthResult, widths, consumer));
    CHECK(Near(widthResult[0], 0.0f) && Near(widthResult[1], 10.0f) &&
          Near(widthResult[2], 0.0f) && Near(widthResult[3], 5.25f) &&
          Near(widthResult[4], 10.0f));

    // Length remains root anchored while its scale is broadcast by curve.
    CHECK(style.Length(geometry,
                       ScalarField::Device({primitive.data(), 2}, expr::Domain::Primitive),
                       {vectorOutput.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(style.Finish(consumer) == StyleStatus::Ok);
    std::vector<float3> vectorResult(5);
    CHECK(Download(vectorResult, vectorOutput, consumer));
    CHECK(Near(vectorResult[0], V(0, 0, 0)) && Near(vectorResult[1], V(0, 0, 2)) &&
          Near(vectorResult[2], V(1, 0, 0)) && Near(vectorResult[3], V(1, 0, 4)) &&
          Near(vectorResult[4], V(1, 0, 8)));

    // Supplied hairT is used instead of index interpolation by Grow.
    CHECK(style.Grow(geometry, {roots.data(), 2}, {normals.data(), 2},
                     ScalarField::Device({primitive.data(), 2}, expr::Domain::Primitive),
                     {point.data(), 5}, {vectorOutput.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(style.Finish(consumer) == StyleStatus::Ok);
    CHECK(Download(vectorResult, vectorOutput, consumer));
    CHECK(Near(vectorResult[0], V(0, 0, 0)) && Near(vectorResult[1], V(0, .5f, 0)) &&
          Near(vectorResult[2], V(1, 0, 0)) && Near(vectorResult[3], V(1, 0, 2)) &&
          Near(vectorResult[4], V(1, 0, 4)));

    // A malformed offset is detected before any offset-derived point read,
    // and the staged output is not published on failure.
    std::vector<float> sentinel(5, 91.0f);
    CHECK(Upload(widths, sentinel, producer));
    CHECK(Upload(offsets, {0u, 2u, 9u}, producer));
    CHECK(style.WidthRamp(geometry, 1.0f, 2.0f, {widths.data(), 5}, producer) ==
          StyleStatus::Ok);
    CHECK(style.Finish(consumer) == StyleStatus::InvalidArgument);
    CHECK(Download(sentinel, widths, consumer));
    for (float value : sentinel) CHECK(Near(value, 91.0f));
    CHECK(Upload(offsets, {1u, 2u, 5u}, producer));
    CHECK(style.WidthRamp(geometry, 1.0f, 2.0f, {widths.data(), 5}, producer) ==
          StyleStatus::Ok);
    CHECK(style.Finish(consumer) == StyleStatus::InvalidArgument);
    CHECK(Download(sentinel, widths, consumer));
    for (float value : sentinel) CHECK(Near(value, 91.0f));

    // Device non-finite values and negative widths are terminal diagnostics.
    CHECK(Upload(offsets, {0u, 2u, 5u}, producer));
    CHECK(Upload(primitive, {NAN, 4.0f}, producer));
    CHECK(style.WidthRamp(geometry, ScalarField::Literal(1.0f),
                          ScalarField::Device({primitive.data(), 2}, expr::Domain::Primitive),
                          {widths.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(style.Finish(consumer) == StyleStatus::NonFiniteInput);
    CHECK(style.WidthRamp(geometry, -1.0f, 2.0f, {widths.data(), 5}, producer) ==
          StyleStatus::Ok);
    CHECK(style.Finish(consumer) == StyleStatus::InvalidValue);
    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);

    // Host-invalid metadata must enqueue no memset or event. A valid call on
    // another stream can therefore follow immediately without clobbering its
    // diagnostic state.
    CHECK(style.WidthRamp(geometry,
                          ScalarField::Device({primitive.data(), 1}, expr::Domain::Primitive),
                          ScalarField::Literal(2.0f), {widths.data(), 5}, producer) ==
          StyleStatus::InvalidArgument);
    CHECK(style.WidthRamp(geometry, 1.0f, 2.0f, {widths.data(), 5}, consumer) ==
          StyleStatus::Ok);
    CHECK(style.Finish(producer) == StyleStatus::Ok);

    DeviceCurveGeometryView mismatchedCounts = geometry;
    mismatchedCounts.curveCount = 0;
    CHECK(style.WidthRamp(mismatchedCounts, 1.0f, 2.0f, {widths.data(), 5}, producer) ==
          StyleStatus::InvalidArgument);

    CHECK(Upload(points, {V(0, 0, 0), V(0, 0, 1), V(1, 0, 0),
                          V(NAN, 0, 1), V(1, 0, 2)}, producer));
    CHECK(style.Length(geometry, 1.0f, {vectorOutput.data(), 5}, producer) ==
          StyleStatus::Ok);
    CHECK(style.Finish(consumer) == StyleStatus::NonFiniteInput);
    CHECK(Upload(points, {V(0, 0, 0), V(0, 0, 1), V(1, 0, 0),
                          V(1, 0, 1), V(1, 0, 2)}, producer));

    // A zero normal is rejected without changing the caller's output.
    CHECK(Upload(primitive, {2.0f, 4.0f}, producer));
    CHECK(Upload(normals, {V(0, 0, 0), V(0, 0, 3)}, producer));
    CHECK(Upload(vectorOutput, {V(77, 77, 77), V(77, 77, 77), V(77, 77, 77),
                                V(77, 77, 77), V(77, 77, 77)}, producer));
    CHECK(style.Grow(geometry, {roots.data(), 2}, {normals.data(), 2}, 2.0f,
                     {vectorOutput.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(style.Finish(consumer) == StyleStatus::InvalidValue);
    CHECK(Download(vectorResult, vectorOutput, consumer));
    for (float3 value : vectorResult) CHECK(Near(value, V(77, 77, 77)));
    CHECK(Upload(normals, {V(0, 2, 0), V(NAN, 0, 3)}, producer));
    CHECK(style.Grow(geometry, {roots.data(), 2}, {normals.data(), 2}, 2.0f,
                     {vectorOutput.data(), 5}, producer) == StyleStatus::Ok);
    CHECK(style.Finish(consumer) == StyleStatus::NonFiniteInput);

    // Empty geometry is a valid no-op, while Noise is explicitly unfinished.
    DeviceCurveGeometryView empty{{}, {}, {}, {}, {}, 0, 0};
    CudaStyleOps emptyStyle;
    CHECK(emptyStyle.WidthRamp(empty, 1.0f, 2.0f, {}, producer) == StyleStatus::Ok);
    CHECK(emptyStyle.Finish(consumer) == StyleStatus::Ok);
    CHECK(emptyStyle.Noise(empty, 1.0f, 7, {}, producer) == StyleStatus::NotSupported);

    cudaStreamDestroy(producer);
    cudaStreamDestroy(consumer);
    return 0;
}
