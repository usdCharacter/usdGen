// Small CUDA-only fixture shared by graphics interop tests.  The returned
// generation owns every device allocation; callers own the consumer stream.
#ifndef USDGEN_TESTS_CUDA_GL_FIXTURE_H
#define USDGEN_TESTS_CUDA_GL_FIXTURE_H

#include "usdGenCurves/cuda/generation.h"
#include "usdGenCurves/cuda/curveSource.h"
#include "usdGenCurves/cuda/width.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace usdGenTest {

inline std::shared_ptr<const usdGen::UsdGenDeviceGeneration>
MakeCudaGlFixture(float width, uint64_t generation = 1, float xOffset = 0.0f)
{
    using namespace usdGen::gpu;
    constexpr size_t pointCount = 5;
    constexpr size_t curveCount = 2;
    cudaStream_t stream = nullptr; // default stream; caller owns GL stream/context

    auto source = std::make_unique<CudaCurveSource>();
    const int32_t counts[] = {2, 3};
    const float3 points[] = {
        make_float3(-.5f + xOffset, -.6f, 0), make_float3(-.5f + xOffset, .6f, 0),
        make_float3(.5f + xOffset, -.6f, 0), make_float3(.5f + xOffset, 0, 0),
        make_float3(.5f + xOffset, .6f, 0)};
    const int32_t rootPrim[] = {4, 8};
    const float2 rootUV[] = {make_float2(.1f, .2f), make_float2(.3f, .4f)};
    const uint64_t stableIds[] = {0x123456780000005bULL,
                                  0xabcdef0100000025ULL};
    CurveSourceInput input;
    input.curveVertexCounts = {counts, curveCount};
    input.points = {points, pointCount};
    input.restPoints = {points, pointCount};
    input.rootPrim = {rootPrim, curveCount};
    input.rootUV = {rootUV, curveCount};
    input.stableIds = {stableIds, curveCount};
    input.fallbackWidth = 1.0f;
    auto status = source->Set(input, stream);
    if (status == CurveSourceStatus::Ok) status = source->Finish(stream);
    if (status != CurveSourceStatus::Ok) {
        std::fprintf(stderr, "MakeCudaGlFixture: source upload failed status=%d CUDA=%s\n",
                     static_cast<int>(status), cudaGetErrorString(cudaGetLastError()));
        return {};
    }

    DeviceBuffer<float> lut;
    if (lut.reset(257) != cudaSuccess) return {};
    std::vector<float> flat(257, 1.0f);
    if (cudaMemcpy(lut.data(), flat.data(), flat.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) return {};

    auto output = std::make_unique<DeviceBuffer<float>>();
    if (output->reset(pointCount) != cudaSuccess) return {};
    CudaWidth op;
    WidthParameters parameters;
    parameters.width = ScalarField::Literal(width);
    parameters.widthProfile = {lut.data(), lut.size()};
    auto widthStatus = op.Apply(source->view(), source->hairT(), parameters, output->view(), stream);
    if (widthStatus == StyleStatus::Ok) widthStatus = op.Finish(stream);
    if (widthStatus != StyleStatus::Ok ||
        output->recordUse(stream) != cudaSuccess) {
        std::fprintf(stderr, "MakeCudaGlFixture: width evaluation failed status=%d CUDA=%s\n",
                     static_cast<int>(widthStatus), cudaGetErrorString(cudaGetLastError()));
        return {};
    }

    std::string reason;
    auto generationResult = MakeSourceGeneration(std::move(source), generation, &reason,
                                           false, std::move(output));
    if (!generationResult) {
        std::fprintf(stderr, "MakeCudaGlFixture: generation failed: %s\n",
                     reason.c_str());
    }
    return generationResult;
}

} // namespace usdGenTest

#endif
