#include "usdGenCurves/cuda/pointOverride.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace usdGen::gpu;

#define CHECK(X) do { if (!(X)) { \
    std::fprintf(stderr, "check failed at %d: %s\n", __LINE__, #X); return 1; \
} } while (0)

static float3 P(float x, float y, float z) { return make_float3(x, y, z); }
static bool Same(float3 a, float3 b) {
    return std::fabs(a.x-b.x) < 1.e-5f && std::fabs(a.y-b.y) < 1.e-5f &&
           std::fabs(a.z-b.z) < 1.e-5f;
}

template<class T>
static bool Upload(DeviceBuffer<T>& buffer, std::vector<T> const& values) {
    return buffer.reset(values.size()) == cudaSuccess &&
        (values.empty() || cudaMemcpy(buffer.data(), values.data(),
                                      values.size() * sizeof(T),
                                      cudaMemcpyHostToDevice) == cudaSuccess);
}

template<class T>
static std::vector<T> Read(DeviceView<const T> view, cudaStream_t stream) {
    std::vector<T> result(view.size);
    if (view.size && (cudaMemcpyAsync(result.data(), view.data,
                                      view.size * sizeof(T),
                                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                      cudaStreamSynchronize(stream) != cudaSuccess))
        result.clear();
    return result;
}

template<class T>
static DeviceView<const T> Const(DeviceView<T> view) {
    return {view.data, view.size};
}

int main() {
    cudaStream_t producer = nullptr, consumer = nullptr;
    CHECK(cudaStreamCreate(&producer) == cudaSuccess);
    CHECK(cudaStreamCreate(&consumer) == cudaSuccess);
    {
        const std::vector<float3> base = {
            P(0,0,0), P(1,0,0), P(2,0,0), P(3,0,0), P(4,0,0)};
        DeviceBuffer<float3> dBase, dPositions;
        DeviceBuffer<int32_t> dIndices;
        CHECK(Upload(dBase, base));
        CudaPointOverride override;

        CHECK(Upload(dIndices, std::vector<int32_t>{3,1}));
        CHECK(Upload(dPositions, std::vector<float3>{P(30,31,32), P(10,11,12)}));
        DeviceView<const float3> baseView = Const(dBase.view());
        CHECK(override.Apply(baseView, Const(dIndices.view()), Const(dPositions.view()), producer) ==
              PointOverrideStatus::Ok);
        CHECK(override.pending() && override.view().size == 0);
        CHECK(override.releaseCompleted() == nullptr);
        CHECK(override.Finish(consumer) == PointOverrideStatus::Ok);
        CHECK(override.generation() == 1 && override.view().size == base.size());
        const auto expected = std::vector<float3>{
            P(0,0,0), P(10,11,12), P(2,0,0), P(30,31,32), P(4,0,0)};
        auto result = Read(override.view(), consumer);
        CHECK(result.size() == expected.size());
        for (size_t i = 0; i < result.size(); ++i) CHECK(Same(result[i], expected[i]));
        auto untouched = Read(Const(dBase.view()), consumer);
        CHECK(untouched.size() == base.size());
        for (size_t i = 0; i < base.size(); ++i) CHECK(Same(untouched[i], base[i]));

        // The published generation remains unchanged after rejected edits.
        CHECK(Upload(dIndices, std::vector<int32_t>{1,1}));
        CHECK(override.Apply(baseView, Const(dIndices.view()), Const(dPositions.view()), producer) ==
              PointOverrideStatus::DuplicateIndex);
        CHECK(override.view().size == base.size());
        CHECK(Upload(dIndices, std::vector<int32_t>{5,}));
        CHECK(override.Apply(baseView, Const(dIndices.view()),
                             {dPositions.data(), 1}, producer) ==
              PointOverrideStatus::InvalidArgument);
        auto nanPosition = std::vector<float3>{P(NAN, 0, 0)};
        CHECK(Upload(dIndices, std::vector<int32_t>{0}) && Upload(dPositions, nanPosition));
        CHECK(override.Apply(baseView, Const(dIndices.view()), Const(dPositions.view()), producer) ==
              PointOverrideStatus::NonFiniteInput);
        CHECK(override.view().size == base.size());
        auto retained = override.releaseCompleted();
        CHECK(retained && retained->size() == base.size());
        CHECK(override.releaseCompleted() == nullptr);

        // Sparse input is allowed to be empty and is an exact device copy.
        DeviceView<const int32_t> noIndices{};
        DeviceView<const float3> noPositions{};
        CHECK(override.Apply(baseView, noIndices, noPositions, producer) ==
              PointOverrideStatus::Ok);
        CHECK(override.Finish(consumer) == PointOverrideStatus::Ok);
        result = Read(override.view(), consumer);
        CHECK(result.size() == base.size());
        for (size_t i = 0; i < result.size(); ++i) CHECK(Same(result[i], base[i]));

        // Use/release fences stay with the completed allocation.
        CHECK(override.recordUse(producer) == cudaSuccess);
        CHECK(override.waitOn(consumer) == cudaSuccess);
        auto owned = override.releaseCompleted();
        CHECK(owned && override.view().size == 0);
        result = Read(Const(owned->view()), consumer);
        CHECK(result.size() == base.size());
        for (size_t i = 0; i < result.size(); ++i) CHECK(Same(result[i], base[i]));

        // A host pointer is not silently accepted as a CUDA input.
        CHECK(override.Apply({base.data(), base.size()}, noIndices, noPositions, producer) ==
              PointOverrideStatus::InvalidArgument);

        CudaPointOverride emptyOverride;
        DeviceView<const float3> emptyBase{};
        CHECK(emptyOverride.Apply(emptyBase, noIndices, noPositions, producer) ==
              PointOverrideStatus::Ok);
        CHECK(emptyOverride.Finish(consumer) == PointOverrideStatus::Ok);
        CHECK(emptyOverride.view().size == 0 && emptyOverride.generation() == 1);
        auto emptyOwned = emptyOverride.releaseCompleted();
        CHECK(emptyOwned && emptyOwned->size() == 0);
        CHECK(emptyOverride.releaseCompleted() == nullptr);
    }
    CHECK(cudaStreamDestroy(consumer) == cudaSuccess);
    CHECK(cudaStreamDestroy(producer) == cudaSuccess);
    std::puts("testUsdGenCudaPointOverride: PASS");
    return 0;
}
