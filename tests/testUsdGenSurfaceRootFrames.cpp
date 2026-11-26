#include "usdGen/surfaceRootFrames.h"
#include "usdGenUsd/lower.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#ifdef USDGEN_TEST_CUDA_ROOT_FRAME_PARITY
#include "gpu/rootFrames.h"
#include <cuda_runtime.h>
#endif

using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

static bool Near(double a, double b) { return std::abs(a-b) < 1.e-6; }
static UsdGenSurfaceDesc Surface() {
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/Scalp");
    s.restPoints = {{0,0,0}, {2,0,0}, {0,2,0}, {2,2,0}, {-1,1,0}};
    s.points = s.restPoints;
    s.faceVertexCounts = {3,4,5};
    s.faceVertexIndices = {0,1,2, 0,1,3,2, 0,1,3,2,4};
    return s;
}

#ifdef USDGEN_TEST_CUDA_ROOT_FRAME_PARITY
namespace {

template <class T>
bool Upload(gpu::DeviceBuffer<T> *device, std::vector<T> const &host,
            cudaStream_t stream)
{
    return device->reset(host.size()) == cudaSuccess &&
        (host.empty() || cudaMemcpyAsync(device->data(), host.data(),
            host.size() * sizeof(T), cudaMemcpyHostToDevice, stream) == cudaSuccess);
}
template <class T>
bool Download(gpu::DeviceView<const T> device, std::vector<T> *host,
              cudaStream_t stream)
{
    host->resize(device.size);
    return (host->empty() || cudaMemcpyAsync(host->data(), device.data,
                host->size() * sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
}
template <class T>
gpu::DeviceView<const T> View(gpu::DeviceBuffer<T> const &device) { return device.view(); }

gpu::RootFrameNormalDomain GpuDomain(UsdGenSurfaceNormalDomain domain)
{
    switch (domain) {
    case UsdGenSurfaceNormalDomain::None: return gpu::RootFrameNormalDomain::None;
    case UsdGenSurfaceNormalDomain::Constant: return gpu::RootFrameNormalDomain::Constant;
    case UsdGenSurfaceNormalDomain::Uniform: return gpu::RootFrameNormalDomain::Uniform;
    case UsdGenSurfaceNormalDomain::Vertex: return gpu::RootFrameNormalDomain::Vertex;
    case UsdGenSurfaceNormalDomain::FaceVarying: return gpu::RootFrameNormalDomain::FaceVarying;
    default: return gpu::RootFrameNormalDomain::None;
    }
}

bool NearGpu(float value, double expected) { return std::fabs(value - float(expected)) < 1.e-5f; }

bool CompareNativeCuda(UsdGenSurfaceDesc const &surface, VtIntArray const &roots,
                       VtVec2fArray const &uv)
{
    UsdGenSurfaceRootFrameResult host;
    std::string error;
    if (!UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(surface), roots, uv, &host, &error)) {
        std::fprintf(stderr, "host root-frame construction failed: %s\n", error.c_str());
        return false;
    }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return false;
    bool ok = false;
    {
        std::vector<float3> vertices;
        vertices.reserve(surface.restPoints.size());
        for (GfVec3f const &p : surface.restPoints) vertices.push_back(make_float3(p[0], p[1], p[2]));
        std::vector<uint32_t> offsets(1, 0), indices;
        for (int count : surface.faceVertexCounts) offsets.push_back(offsets.back() + uint32_t(count));
        indices.reserve(surface.faceVertexIndices.size());
        for (int index : surface.faceVertexIndices) indices.push_back(uint32_t(index));
        std::vector<int32_t> rootValues(roots.begin(), roots.end());
        std::vector<float2> uvValues;
        uvValues.reserve(uv.size());
        for (GfVec2f const &value : uv) uvValues.push_back(make_float2(value[0], value[1]));
        std::vector<float3> normals;
        normals.reserve(surface.restNormals.size());
        for (GfVec3f const &n : surface.restNormals) normals.push_back(make_float3(n[0], n[1], n[2]));

        gpu::DeviceBuffer<float3> dVertices, dNormals;
        gpu::DeviceBuffer<uint32_t> dOffsets, dIndices;
        gpu::DeviceBuffer<int32_t> dRoots;
        gpu::DeviceBuffer<float2> dUv;
        if (!Upload(&dVertices, vertices, stream) || !Upload(&dOffsets, offsets, stream) ||
            !Upload(&dIndices, indices, stream) || !Upload(&dRoots, rootValues, stream) ||
            !Upload(&dUv, uvValues, stream) || !Upload(&dNormals, normals, stream)) {
            cudaStreamSynchronize(stream);
        } else {
            gpu::CudaRestRootFrames frames;
            if (frames.Apply(View(dVertices), View(dOffsets), surface.faceVertexCounts.size(),
                    View(dIndices), View(dRoots), View(dUv), View(dNormals),
                    GpuDomain(surface.restNormalDomain), {}, stream) == gpu::RootFrameStatus::Ok &&
                frames.Finish(stream) == gpu::RootFrameStatus::Ok) {
                std::vector<float3> origin, tangent, binormal, normal;
                std::vector<uint8_t> valid, drop;
                if (Download(frames.origins(), &origin, stream) &&
                    Download(frames.frames().tangent, &tangent, stream) &&
                    Download(frames.frames().binormal, &binormal, stream) &&
                    Download(frames.frames().normal, &normal, stream) &&
                    Download(frames.valid(), &valid, stream) && Download(frames.dropMask(), &drop, stream) &&
                    valid.size() == host.valid.size() && drop.size() == host.dropMask.size() &&
                    origin.size() == host.frames.size()) {
                    ok = true;
                    for (size_t i = 0; i != host.frames.size(); ++i) {
                        if (valid[i] != host.valid[i] || drop[i] != host.dropMask[i]) { ok = false; break; }
                        if (!host.valid[i]) continue;
                        GfMatrix4d const &f = host.frames[i];
                        ok = NearGpu(origin[i].x, f[3][0]) && NearGpu(origin[i].y, f[3][1]) && NearGpu(origin[i].z, f[3][2]) &&
                             NearGpu(tangent[i].x, f[0][0]) && NearGpu(tangent[i].y, f[0][1]) && NearGpu(tangent[i].z, f[0][2]) &&
                             NearGpu(binormal[i].x, f[1][0]) && NearGpu(binormal[i].y, f[1][1]) && NearGpu(binormal[i].z, f[1][2]) &&
                             NearGpu(normal[i].x, f[2][0]) && NearGpu(normal[i].y, f[2][1]) && NearGpu(normal[i].z, f[2][2]);
                        if (!ok) break;
                    }
                }
            }
        }
    } // Device/frame owners retire before their stream.
    cudaError_t const destroyed = cudaStreamDestroy(stream);
    return ok && destroyed == cudaSuccess;
}

UsdGenSurfaceDesc CudaSurface()
{
    auto surface = Surface();
    surface.faceVertexCounts = {3,4};
    surface.faceVertexIndices = {0,1,2, 0,1,3,2};
    return surface;
}

} // namespace
#endif

int main() {
#ifdef USDGEN_TEST_CUDA_ROOT_FRAME_PARITY
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
#endif
    auto s = Surface();
    UsdGenSurfaceRootFrameResult frames;
    std::string error;
    CHECK(UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(s), {0,1,2},
        {{.25f,.5f}, {.25f,.5f}, {1.25f,.5f}}, &frames, &error));
    CHECK(frames.dropped == 0 && frames.valid.size() == 3);
    for (auto const& frame : frames.frames)
        CHECK(UsdGenValidateAuthoredRootFrame(frame));
    CHECK(Near(frames.frames[0][3][0], .5) && Near(frames.frames[0][3][1], 1));
    CHECK(Near(frames.frames[1][3][0], .5) && Near(frames.frames[1][3][1], 1));
    CHECK(Near(frames.frames[2][3][0], .5) && Near(frames.frames[2][3][1], 1.5));
    CHECK(Near(frames.frames[0][0][0], 1) && Near(frames.frames[0][2][2], 1));
    CHECK(UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(s), {0}, {{.6f,.4f}}, &frames, &error));
    CHECK(frames.valid[0]);
#ifdef USDGEN_TEST_CUDA_ROOT_FRAME_PARITY
    CHECK(CompareNativeCuda(CudaSurface(), {0}, {{.6f,.4f}}));
#endif

    // Parent face indices are independent of Scatter subsets and transforms
    // are transport metadata, not baked into surface-local frames.
    s.subsetFaces = {2};
    s.worldMatrix.SetTranslate(GfVec3d(100,200,300));
    for (auto& p : s.points) p[2] = 50;
    CHECK(UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(s), {0}, {{.25f,.5f}}, &frames, &error));
    CHECK(Near(frames.frames[0][3][0], .5) && Near(frames.frames[0][3][2], 0));
    for (auto domain : {UsdGenSurfaceNormalDomain::Constant,
                        UsdGenSurfaceNormalDomain::Uniform,
                        UsdGenSurfaceNormalDomain::Vertex,
                        UsdGenSurfaceNormalDomain::FaceVarying}) {
        s.restNormalDomain = domain;
        size_t const size = domain == UsdGenSurfaceNormalDomain::Constant ? 1 :
            domain == UsdGenSurfaceNormalDomain::Uniform ? 3 :
            domain == UsdGenSurfaceNormalDomain::Vertex ? 5 : 12;
        s.restNormals.assign(size, GfVec3f(0,1,1));
        CHECK(UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(s), {0,1,2},
            {{.25f,.5f},{.25f,.5f},{1.25f,.5f}}, &frames, &error));
        CHECK(frames.dropped == 0);
        for (auto const& f : frames.frames) {
            CHECK(UsdGenValidateAuthoredRootFrame(f));
            CHECK(Near(f[2][1], std::sqrt(.5)) && Near(f[2][2], std::sqrt(.5)));
        }
    }
    auto const oldFrames = frames.frames;
    {
        // Nonuniform values catch wrong interpolation indices/weights; equal
        // normals alone would not distinguish the supported normal domains.
        auto varying = Surface();
        varying.restNormalDomain = UsdGenSurfaceNormalDomain::Vertex;
        varying.restNormals = {{0,0,1},{0,1,1},{0,2,1},{0,3,1},{0,4,1}};
        UsdGenSurfaceRootFrameResult interpolated;
        CHECK(UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(varying),{0},{{.25f,.5f}},&interpolated,&error));
        CHECK(Near(interpolated.frames[0][2][1],1.25/std::sqrt(2.5625)) &&
              Near(interpolated.frames[0][2][2],1/std::sqrt(2.5625)));
#ifdef USDGEN_TEST_CUDA_ROOT_FRAME_PARITY
        varying.faceVertexCounts = {3,4};
        varying.faceVertexIndices = {0,1,2,0,1,3,2};
        CHECK(CompareNativeCuda(varying,{0,1},{{.25f,.5f},{.4f,.6f}}));
        varying.restNormalDomain = UsdGenSurfaceNormalDomain::FaceVarying;
        varying.restNormals = {{0,0,1},{0,1,1},{0,2,1},
                               {0,4,1},{0,3,1},{0,2,1},{0,1,1}};
        CHECK(CompareNativeCuda(varying,{1,0},{{.4f,.6f},{.25f,.5f}}));
        varying.restNormalDomain = UsdGenSurfaceNormalDomain::Uniform;
        varying.restNormals = {{0,0,1},{0,1,1}};
        CHECK(CompareNativeCuda(varying,{1,0},{{.4f,.6f},{.25f,.5f}}));
#endif
    }
    auto reject = [&](UsdGenSurfaceDesc const& invalid, VtIntArray prim,
                      VtVec2fArray uv) {
        return !UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(invalid), prim, uv, &frames, &error) &&
            !error.empty() && frames.frames == oldFrames;
    };
    CHECK(reject(s, {0}, {{.8f,.8f}}));
    CHECK(reject(s, {2}, {{std::numeric_limits<float>::max(),0}}));
    CHECK(reject(s, {3}, {{0,0}}));
    CHECK(reject(s, {0}, {}));
    auto invalid = s;
    invalid.restFromCurrentPoints = true;
    CHECK(reject(invalid, {0}, {{0,0}}));
    invalid = s; invalid.restNormals.pop_back();
    CHECK(reject(invalid, {0}, {{0,0}}));
    invalid = s; invalid.faceVertexIndices[0] = 100;
    CHECK(reject(invalid, {0}, {{0,0}}));
    invalid = s; invalid.restPoints[0][0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(reject(invalid, {0}, {{0,0}}));
    invalid = s; invalid.worldMatrix[0][0] = std::numeric_limits<double>::infinity();
    CHECK(reject(invalid, {0}, {{0,0}}));

    // Degenerate geometry produces explicit drop bits, never identity frames.
    s = Surface(); s.restPoints[1] = s.restPoints[0];
    CHECK(UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(s), {0,1}, {{0,0},{.5f,.5f}}, &frames, &error));
    CHECK(frames.dropped == 1 && frames.dropMask[0] == 1 && frames.valid[0] == 0 &&
          frames.frames[0] == GfMatrix4d(0.0) && frames.valid[1] == 1);
    // A collapsed quad derivative falls back to the first projected edge.
    s = Surface(); s.restPoints[3] = s.restPoints[2];
    CHECK(UsdGenBuildRestSurfaceRootFrames(usdGenUsd::Lower(s), {1}, {{0,1}}, &frames, &error));
    CHECK(frames.valid[0] && UsdGenValidateAuthoredRootFrame(frames.frames[0]));

    GfMatrix4d authored(1);
    CHECK(UsdGenValidateAuthoredRootFrame(authored));
    authored[0][0] = 2;
    CHECK(!UsdGenValidateAuthoredRootFrame(authored));
    authored = GfMatrix4d(1); authored[2][2] = -1;
    CHECK(!UsdGenValidateAuthoredRootFrame(authored));
    authored = GfMatrix4d(1); authored[0][3] = 1;
    CHECK(!UsdGenValidateAuthoredRootFrame(authored));
    authored = GfMatrix4d(1); authored[3][0] = std::numeric_limits<double>::max();
    CHECK(!UsdGenValidateAuthoredRootFrame(authored));
#ifdef USDGEN_TEST_CUDA_ROOT_FRAME_PARITY
    // The shared host implementation must agree with the native device
    // helper for its supported tri/quad domain.  The 5-gon above deliberately
    // remains host-only: gpu/rootFrames.cu accepts only tri/quad input.
    auto cudaSurface = CudaSurface();
    CHECK(CompareNativeCuda(cudaSurface, {0,1}, {{.25f,.5f},{.25f,.5f}}));
    for (auto domain : {UsdGenSurfaceNormalDomain::Constant,
                        UsdGenSurfaceNormalDomain::Uniform,
                        UsdGenSurfaceNormalDomain::Vertex,
                        UsdGenSurfaceNormalDomain::FaceVarying}) {
        cudaSurface.restNormalDomain = domain;
        size_t const count = domain == UsdGenSurfaceNormalDomain::Constant ? 1 :
            domain == UsdGenSurfaceNormalDomain::Uniform ? 2 :
            domain == UsdGenSurfaceNormalDomain::Vertex ? cudaSurface.restPoints.size() :
            cudaSurface.faceVertexIndices.size();
        cudaSurface.restNormals.assign(count, GfVec3f(0,1,1));
        CHECK(CompareNativeCuda(cudaSurface, {0,1}, {{.25f,.5f},{.25f,.5f}}));
    }
    cudaSurface = CudaSurface();
    cudaSurface.restPoints[1] = cudaSurface.restPoints[0];
    CHECK(CompareNativeCuda(cudaSurface, {0}, {{0,0}}));
    cudaSurface = CudaSurface();
    cudaSurface.restPoints[3] = cudaSurface.restPoints[2];
    CHECK(CompareNativeCuda(cudaSurface, {1}, {{0,1}}));
#endif
    std::puts("surface rest root frame tests passed");
    return 0;
}
