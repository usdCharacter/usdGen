#include "usdGenSurface/cuda/rootFrames.h"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec3d.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace usdGen::gpu;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (false)

template <class T>
bool Upload(DeviceBuffer<T>& device, std::vector<T> const& host, cudaStream_t stream) {
    return device.reset(host.size()) == cudaSuccess &&
        (host.empty() || cudaMemcpyAsync(device.data(), host.data(), host.size()*sizeof(T),
                                         cudaMemcpyHostToDevice, stream) == cudaSuccess);
}
template <class T>
DeviceView<const T> View(DeviceBuffer<T> const& device) { return device.view(); }
bool UploadUv(DeviceBuffer<unsigned char>& device, std::vector<float2> const& host,
              cudaStream_t stream) {
    return device.reset(host.size()*sizeof(float2)) == cudaSuccess &&
        (host.empty() || cudaMemcpyAsync(device.data(), host.data(), host.size()*sizeof(float2),
                                         cudaMemcpyHostToDevice, stream) == cudaSuccess);
}
DeviceView<const float2> UvView(DeviceBuffer<unsigned char> const& device) {
    return {reinterpret_cast<float2 const*>(device.data()), device.size()/sizeof(float2)};
}
template <class T>
bool Download(DeviceView<const T> device, std::vector<T>* host, cudaStream_t stream) {
    host->resize(device.size);
    return (host->empty() || cudaMemcpyAsync(host->data(), device.data, host->size()*sizeof(T),
                                              cudaMemcpyDeviceToHost, stream) == cudaSuccess) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
}
bool Near(float a, float b) { return std::fabs(a-b) < 1.0e-5f; }
bool Near(float3 a, float3 b) { return Near(a.x,b.x) && Near(a.y,b.y) && Near(a.z,b.z); }

int main() {
    cudaStream_t stream = nullptr, consumer = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking) == cudaSuccess);
    {
    DeviceBuffer<float3> vertices, normals;
    DeviceBuffer<uint32_t> offsets, indices;
    DeviceBuffer<int32_t> roots;
    DeviceBuffer<unsigned char> uv;
    DeviceBuffer<double> matrices;

    // A triangle in z=0 and a bilinear quad in z=1. The second root proves
    // quads accept independent u/v rather than triangle barycentric u+v.
    std::vector<float3> const hostVertices{
        make_float3(0,0,0), make_float3(2,0,0), make_float3(0,2,0),
        make_float3(0,0,1), make_float3(2,0,1), make_float3(2,2,1), make_float3(0,2,1)};
    CHECK(Upload(vertices, hostVertices, stream) &&
          Upload(offsets, std::vector<uint32_t>{0,3,7}, stream) &&
          Upload(indices, std::vector<uint32_t>{0,1,2,3,4,5,6}, stream) &&
          Upload(roots, std::vector<int32_t>{0,1}, stream) &&
          UploadUv(uv, std::vector<float2>{make_float2(.25f,.5f), make_float2(.8f,.7f)}, stream));

    CudaRestRootFrames frames;
    CHECK(frames.Apply(View(vertices), View(offsets), 2, View(indices), View(roots), UvView(uv),
                       {}, RootFrameNormalDomain::None, {}, stream) == RootFrameStatus::Ok);
    CHECK(frames.Finish(stream) == RootFrameStatus::Ok && frames.badRootCount() == 0);
    std::vector<float3> origin, tangent, binormal, normal;
    std::vector<uint8_t> valid, drop;
    CHECK(Download(frames.origins(), &origin, stream) && Download(frames.frames().tangent, &tangent, stream) &&
          Download(frames.frames().binormal, &binormal, stream) && Download(frames.frames().normal, &normal, stream) &&
          Download(frames.valid(), &valid, stream) && Download(frames.dropMask(), &drop, stream));
    CHECK(Near(origin[0], make_float3(.5f,1.0f,0)) && Near(origin[1], make_float3(1.6f,1.4f,1)));
    CHECK(Near(tangent[0], make_float3(1,0,0)) && Near(binormal[0], make_float3(0,1,0)) &&
          Near(normal[0], make_float3(0,0,1)) && valid == std::vector<uint8_t>({1,1}) &&
          drop == std::vector<uint8_t>({0,0}));
    uint64_t const generation = frames.generation();

    // A consumer fence is retained by the active owner and Finish waits before
    // swapping storage for a newer generation.
    CHECK(frames.recordUse(consumer) == cudaSuccess && frames.waitOn(consumer) == cudaSuccess);
    DeviceBuffer<uint32_t> malformedOffsets;
    CHECK(Upload(malformedOffsets, std::vector<uint32_t>{0,2,7}, stream));
    CHECK(frames.Apply(View(vertices), View(malformedOffsets), 2, View(indices), View(roots), UvView(uv),
                       {}, RootFrameNormalDomain::None, {}, stream) == RootFrameStatus::InvalidTopology &&
          frames.generation() == generation);
    CHECK(Upload(roots, std::vector<int32_t>{2,1}, stream));
    CHECK(frames.Apply(View(vertices), View(offsets), 2, View(indices), View(roots), UvView(uv),
                       {}, RootFrameNormalDomain::None, {}, stream) == RootFrameStatus::InvalidRootBinding &&
          frames.generation() == generation);
    CHECK(Upload(roots, std::vector<int32_t>{0,1}, stream));
    CHECK(frames.Apply(View(vertices), View(offsets), 2, View(indices), View(roots), UvView(uv),
                       {}, RootFrameNormalDomain::Vertex, {}, stream) == RootFrameStatus::InvalidArgument &&
          frames.generation() == generation);
    DeviceBuffer<float3> nonFiniteVertices;
    auto badVertices = hostVertices;
    badVertices[0].x = NAN;
    CHECK(Upload(nonFiniteVertices, badVertices, stream));
    CHECK(frames.Apply(View(nonFiniteVertices), View(offsets), 2, View(indices), View(roots), UvView(uv),
                       {}, RootFrameNormalDomain::None, {}, stream) == RootFrameStatus::NonFiniteInput &&
          frames.generation() == generation);

    // Vertex normals interpolate without silently repairing artist orientation.
    CHECK(Upload(normals, std::vector<float3>(hostVertices.size(), make_float3(0,0,-1)), stream));
    CHECK(frames.Apply(View(vertices), View(offsets), 2, View(indices), View(roots), UvView(uv),
                       View(normals), RootFrameNormalDomain::Vertex, {}, stream) == RootFrameStatus::Ok &&
          frames.Finish(stream) == RootFrameStatus::Ok);
    CHECK(Download(frames.frames().normal, &normal, stream) && Near(normal[0], make_float3(0,0,-1)) &&
          Download(frames.frames().binormal, &binormal, stream) && Near(binormal[0], make_float3(0,-1,0)));
    CHECK(Upload(normals, std::vector<float3>{make_float3(0,0,-1)}, stream));
    CHECK(frames.Apply(View(vertices), View(offsets), 2, View(indices), View(roots), UvView(uv),
                       View(normals), RootFrameNormalDomain::Constant, {}, stream) == RootFrameStatus::Ok &&
          frames.Finish(stream) == RootFrameStatus::Ok);
    CHECK(Upload(normals, std::vector<float3>{make_float3(0,0,-1), make_float3(0,0,-1)}, stream));
    CHECK(frames.Apply(View(vertices), View(offsets), 2, View(indices), View(roots), UvView(uv),
                       View(normals), RootFrameNormalDomain::Uniform, {}, stream) == RootFrameStatus::Ok &&
          frames.Finish(stream) == RootFrameStatus::Ok);
    CHECK(Upload(normals, std::vector<float3>(7, make_float3(0,0,-1)), stream));
    CHECK(frames.Apply(View(vertices), View(offsets), 2, View(indices), View(roots), UvView(uv),
                       View(normals), RootFrameNormalDomain::FaceVarying, {}, stream) == RootFrameStatus::Ok &&
          frames.Finish(stream) == RootFrameStatus::Ok);

    // A Gf row-vector matrix deliberately has a nonzero translation. Rows
    // map TransformDir(ex/ey/ez), and row 3 maps Transform(0).
    pxr::GfMatrix4d matrix(1.0);
    matrix[0][0] = 0; matrix[0][1] = 1; matrix[1][0] = -1; matrix[1][1] = 0;
    matrix[3][0] = 7; matrix[3][1] = 8; matrix[3][2] = 9;
    std::vector<double> hostMatrices(32, 0.0);
    for (int r = 0; r != 4; ++r) for (int c = 0; c != 4; ++c) hostMatrices[16+r*4+c] = matrix[r][c];
    // The first identity frame makes matrix data per-curve, not a broadcast.
    hostMatrices[0] = hostMatrices[5] = hostMatrices[10] = hostMatrices[15] = 1.0;
    CHECK(Upload(matrices, hostMatrices, stream));
    CHECK(frames.Apply(View(vertices), View(offsets), 2, View(indices), View(roots), UvView(uv),
                       {}, RootFrameNormalDomain::None, View(matrices), stream) == RootFrameStatus::Ok &&
          frames.Finish(stream) == RootFrameStatus::Ok);
    CHECK(Download(frames.origins(), &origin, stream) && Download(frames.frames().tangent, &tangent, stream) &&
          Download(frames.frames().binormal, &binormal, stream) && Download(frames.frames().normal, &normal, stream));
    pxr::GfVec3d const x = matrix.TransformDir(pxr::GfVec3d(1,0,0));
    pxr::GfVec3d const y = matrix.TransformDir(pxr::GfVec3d(0,1,0));
    pxr::GfVec3d const z = matrix.TransformDir(pxr::GfVec3d(0,0,1));
    pxr::GfVec3d const p = matrix.Transform(pxr::GfVec3d(0));
    CHECK(Near(tangent[1], make_float3(float(x[0]),float(x[1]),float(x[2]))) &&
          Near(binormal[1], make_float3(float(y[0]),float(y[1]),float(y[2]))) &&
          Near(normal[1], make_float3(float(z[0]),float(z[1]),float(z[2]))) &&
          Near(origin[1], make_float3(float(p[0]),float(p[1]),float(p[2]))));

    // Degenerate geometry is per-curve data: it publishes a drop rather than
    // inventing an axis. Bad topology/frame input rejects the whole candidate
    // and leaves the prior published generation intact.
    DeviceBuffer<float3> collapsed;
    CHECK(Upload(collapsed, std::vector<float3>{make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)}, stream) &&
          Upload(offsets, std::vector<uint32_t>{0,3}, stream) && Upload(indices, std::vector<uint32_t>{0,1,2}, stream) &&
          Upload(roots, std::vector<int32_t>{0}, stream) && UploadUv(uv, std::vector<float2>{make_float2(0,0)}, stream));
    CHECK(frames.Apply(View(collapsed), View(offsets), 1, View(indices), View(roots), UvView(uv), {},
                       RootFrameNormalDomain::None, {}, stream) == RootFrameStatus::Ok &&
          frames.Finish(stream) == RootFrameStatus::Ok && frames.badRootCount() == 1);
    CHECK(Download(frames.valid(), &valid, stream) && Download(frames.dropMask(), &drop, stream) &&
          valid == std::vector<uint8_t>({0}) && drop == std::vector<uint8_t>({1}));
    CHECK(Download(frames.origins(), &origin, stream) && Download(frames.frames().tangent, &tangent, stream) &&
          Download(frames.frames().binormal, &binormal, stream) && Download(frames.frames().normal, &normal, stream) &&
          Near(origin[0], make_float3(0,0,0)) && Near(tangent[0], make_float3(0,0,0)) &&
          Near(binormal[0], make_float3(0,0,0)) && Near(normal[0], make_float3(0,0,0)));
    uint64_t const degenerateGeneration = frames.generation();
    CHECK(Upload(matrices, std::vector<double>(16, 0.0), stream));
    CHECK(frames.Apply(View(collapsed), View(offsets), 1, View(indices), View(roots), UvView(uv), {},
                       RootFrameNormalDomain::None, View(matrices), stream) == RootFrameStatus::Ok &&
          frames.Finish(stream) == RootFrameStatus::InvalidAuthoredFrame &&
          frames.generation() == degenerateGeneration);
    std::vector<double> overflowMatrix(16, 0.0);
    overflowMatrix[0] = overflowMatrix[5] = overflowMatrix[10] = overflowMatrix[15] = 1.0;
    overflowMatrix[12] = 1.0e300;
    CHECK(Upload(matrices, overflowMatrix, stream));
    CHECK(frames.Apply(View(collapsed), View(offsets), 1, View(indices), View(roots), UvView(uv), {},
                       RootFrameNormalDomain::None, View(matrices), stream) == RootFrameStatus::Ok &&
          frames.Finish(stream) == RootFrameStatus::InvalidAuthoredFrame &&
          frames.generation() == degenerateGeneration);
    CHECK(generation < frames.generation());

    // Empty root sets have no per-root fields or outputs.
    CHECK(frames.Apply(View(vertices), View(offsets), 1, View(indices), {}, {}, {},
                       RootFrameNormalDomain::None, {}, stream) == RootFrameStatus::Ok &&
          frames.Finish(stream) == RootFrameStatus::Ok && frames.origins().size == 0);
    }
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    CHECK(cudaStreamDestroy(consumer) == cudaSuccess);
    std::puts("testUsdGenCudaRootFrames: PASS");
    return 0;
}
