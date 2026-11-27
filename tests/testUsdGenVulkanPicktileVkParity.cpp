// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenVulkanPicktileVkParity — CUDA-equivalence proof for the Vulkan
// picking + tile spans/bounds/index port (vulkan/picktileVk.h vs
// gpu/picking.cu, gpu/curveTiles.cu, gpu/curveTileBounds.cu,
// gpu/curveIndices.cu). Every section runs the same fixture on both backends
// and compares within a tolerance that cites the CUDA test's own contract:
//
//   * pick indices/stable ids: exact; distancePx within 1e-6 (the CUDA
//     picking test's own fabs bound);
//   * footprint indices/count/status: exact (CUDA footprint order is
//     ascending by construction);
//   * tile spans/status: exact (CUDA golden ranges are exact);
//   * tile bounds: exact expected, accepted within the CUDA bounds test's own
//     directional 1e-4 Near/UpperNear windows;
//   * index streams/primitive params/record counts/draw counts/status: exact
//     (CUDA goldens are exact).
//
// Host-only sections (requirements parity, admission, leases, metadata) run
// without a device. Device sections need one NVIDIA Vulkan device; without
// one the test exits 77. vkCreateDevice failures retry (environmental tenant
// load): 5 attempts with a 60s wait between them.
#include "usdGenVulkan/vulkan/deviceContext.h"
#include "testBackend.h"
#include "usdGenVulkan/vulkan/chargedBuffer.h"
#include "usdGenCurves/vulkan/picktileVk.h"

#include "usdGenCurves/cuda/picking.h"
#include "usdGenCurves/cuda/curveTiles.h"
#include "usdGenCurves/cuda/curveTileBounds.h"
#include "usdGenCurves/cuda/curveIndices.h"
#include "usdGenCurves/cuda/generation.h"

#include <cuda_runtime.h>

#include "vulkanNativeFixture.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;
using namespace usdGen::gpu;

namespace {

int g_fail = 0;

void Check(bool ok, char const* label) {
    if (ok) {
        std::printf("PASS: %s\n", label);
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", label);
    }
}

#define CHECK(x) Check((x), #x)

// Ordinal/layout correspondence between the Vulkan port and the CUDA twins.
void StaticCorrespondence() {
    CHECK(int(CurveTileBoundsBasis::Linear) == int(PicktileVkTileBoundsBasis::Linear));
    CHECK(int(CurveTileBoundsBasis::BSpline) == int(PicktileVkTileBoundsBasis::BSpline));
    CHECK(int(CurveTileBoundsBasis::CatmullRom) == int(PicktileVkTileBoundsBasis::CatmullRom));
    CHECK(int(CurveTileOrder::SortedSurvivorSubset) ==
        int(PicktileVkTileOrder::SortedSurvivorSubset));
    CHECK(int(CurveTileOrder::IdentityCaptureOrder) ==
        int(PicktileVkTileOrder::IdentityCaptureOrder));
    CHECK(int(CurveTileOrder::CaptureOrderSurvivorSubset) ==
        int(PicktileVkTileOrder::CaptureOrderSurvivorSubset));
    CHECK(int(CurveIndexBasis::Linear) == int(PicktileVkIndexBasis::Linear));
    CHECK(int(CurveIndexBasis::Bezier) == int(PicktileVkIndexBasis::Bezier));
    CHECK(int(CurveIndexBasis::BSpline) == int(PicktileVkIndexBasis::BSpline));
    CHECK(int(CurveIndexBasis::CatmullRom) == int(PicktileVkIndexBasis::CatmullRom));
    CHECK(int(CurveIndexBasis::CentripetalCatmullRom) ==
        int(PicktileVkIndexBasis::CentripetalCatmullRom));
    CHECK(int(CurveIndexWrap::Nonperiodic) == int(PicktileVkIndexWrap::Nonperiodic));
    CHECK(int(CurveIndexWrap::Periodic) == int(PicktileVkIndexWrap::Periodic));
    CHECK(int(CurveIndexWrap::Pinned) == int(PicktileVkIndexWrap::Pinned));
    CHECK(int(CurveIndexWrap::Segmented) == int(PicktileVkIndexWrap::Segmented));
    CHECK(int(CurveIndexMode::Curves) == int(PicktileVkIndexMode::Curves));
    CHECK(int(CurveIndexMode::Hull) == int(PicktileVkIndexMode::Hull));
    CHECK(int(CurveIndexMode::Points) == int(PicktileVkIndexMode::Points));
    CHECK(sizeof(PicktileVkTileSpan) == sizeof(CurveTileSpan));
    CHECK(offsetof(PicktileVkTileSpan, tile) == offsetof(CurveTileSpan, tile));
    CHECK(offsetof(PicktileVkTileSpan, firstCurve) == offsetof(CurveTileSpan, firstCurve));
    CHECK(offsetof(PicktileVkTileSpan, curveCount) == offsetof(CurveTileSpan, curveCount));
    CHECK(offsetof(PicktileVkTileSpan, firstPoint) == offsetof(CurveTileSpan, firstPoint));
    CHECK(offsetof(PicktileVkTileSpan, pointCount) == offsetof(CurveTileSpan, pointCount));
    CHECK(int(PickingStatus::Ok) == int(PicktileVkPickPipeline::Semantic::Ok));
    CHECK(int(PickingStatus::InvalidArgument) ==
        int(PicktileVkPickPipeline::Semantic::InvalidArgument));
    CHECK(int(PickingStatus::NonFiniteInput) ==
        int(PicktileVkPickPipeline::Semantic::NonFiniteInput));
}

bool CudaTilesReq(CurveTileOptions options, size_t capture, size_t survivors, size_t points,
    CurveTileRequirements* out) {
    return GetCurveTileRequirements(options, capture, survivors, points, out, nullptr) ==
        cudaSuccess;
}

// Host-only requirements parity: identical scalar cores.
void RequirementsParity() {
    struct Case {
        uint32_t chunk, target;
        size_t capture, survivors, points;
    };
    Case const cases[] = {
        {128, 64, 300, 6, 27},
        {0, 0, 100000, 0, 0},
        {512, 32, 100000, 0, 0},
        {1, 999, 300, 0, 0},
        {0, 0, 300, 0, 0},
        {128, 64, 33 * 128, 33 * 128, 33 * 128},
        {128, 32, 65 * 128, 65 * 128, 65 * 128},
        {1024, 256, 7, 7, 14},
    };
    for (auto const& c : cases) {
        CurveTileRequirements cuda{};
        PicktileVkTileRequirements vk{};
        bool const okCuda = CudaTilesReq({c.chunk, c.target}, c.capture, c.survivors, c.points,
            &cuda);
        bool const okVk = GetPicktileVkTileRequirements(c.chunk, c.target, c.capture,
            c.survivors, c.points, &vk);
        char label[160];
        std::snprintf(label, sizeof(label),
            "tile requirements parity (%u/%u #%zu/%zu/%zu)", c.chunk, c.target, c.capture,
            c.survivors, c.points);
        Check(okCuda && okVk && cuda.chunkCount == vk.chunkCount &&
                cuda.tileCount == vk.tileCount && cuda.chunksPerTile == vk.chunksPerTile &&
                cuda.curvesPerTile == vk.curvesPerTile && cuda.chunkSize == vk.chunkSize &&
                cuda.tileTarget == vk.tileTarget,
            label);
    }
    struct Bad {
        size_t capture, survivors, points;
    };
    Bad const bad[] = {
        {size_t(UINT32_MAX) + 1, 0, 0},
        {3, 4, 4},
        {3, 0, 1},
        {3, 3, 2},
    };
    for (auto const& c : bad) {
        CurveTileRequirements cuda{};
        PicktileVkTileRequirements vk{};
        char label[160];
        std::snprintf(label, sizeof(label), "tile requirements reject #%zu/%zu/%zu", c.capture,
            c.survivors, c.points);
        Check(!CudaTilesReq({}, c.capture, c.survivors, c.points, &cuda) &&
                !GetPicktileVkTileRequirements(0, 0, c.capture, c.survivors, c.points, &vk),
            label);
    }
    // 100k golden partition from the CUDA tile test.
    {
        PicktileVkTileRequirements vk{};
        CHECK(GetPicktileVkTileRequirements(0, 0, 100000, 0, 0, &vk) &&
            vk.chunkCount == 196 && vk.chunksPerTile == 4 && vk.tileCount == 49);
    }
    // Index requirements sweep over the full basis/wrap/mode taxonomy.
    for (int b = 0; b <= 4; ++b)
        for (int w = 0; w <= 3; ++w)
            for (int m = 0; m <= 2; ++m) {
                CurveIndexOptions cudaOptions{CurveIndexBasis(b), CurveIndexWrap(w),
                    CurveIndexMode(m)};
                CurveIndexRequirements cuda{};
                PicktileVkIndexRequirements vk{};
                bool const okCuda = GetCurveIndexRequirements(cudaOptions, 3, 9, &cuda,
                                        nullptr) == cudaSuccess;
                bool const okVk = GetPicktileVkIndexRequirements(PicktileVkIndexBasis(b),
                    PicktileVkIndexWrap(w), PicktileVkIndexMode(m), 3, 9, &vk);
                char label[160];
                std::snprintf(label, sizeof(label), "index requirements parity b%d/w%d/m%d", b,
                    w, m);
                bool same = okCuda == okVk;
                if (same && okCuda)
                    same = cuda.maxRecords == vk.maxRecords &&
                        cuda.indexArity == vk.indexArity;
                Check(same, label);
            }
    {
        PicktileVkIndexRequirements vk{};
        CHECK(GetPicktileVkIndexRequirements(PicktileVkIndexBasis::BSpline,
                PicktileVkIndexWrap::Pinned, PicktileVkIndexMode::Curves, 3, 9, &vk) &&
            vk.maxRecords == 18 && vk.indexArity == 4);
        CHECK(!GetPicktileVkIndexRequirements(PicktileVkIndexBasis::Bezier,
            PicktileVkIndexWrap::Segmented, PicktileVkIndexMode::Curves, 3, 9, &vk));
        CHECK(!GetPicktileVkIndexRequirements(PicktileVkIndexBasis::Linear,
            PicktileVkIndexWrap::Nonperiodic, PicktileVkIndexMode::Curves, 0, 1, &vk));
    }
}

std::vector<uint32_t> LoadSpv(char const* path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    if (bytes.empty() || bytes.size() % 4 != 0) return {};
    std::vector<uint32_t> code(bytes.size() / 4);
    std::memcpy(code.data(), bytes.data(), bytes.size());
    return code;
}

std::shared_ptr<NativeOwner> CreateNativeRetry(bool* unavailable) {
    *unavailable = false;
    for (int attempt = 0; attempt < 5; ++attempt) {
        bool local = false;
        auto native = CreateTimelineNative(&local);
        if (native) return native;
        if (local) {
            *unavailable = true;
            return {};
        }
        // Environmental (tenant load can fail vkCreateDevice with -3): wait
        // 60s and retry; never edit code to chase it.
        std::printf("picktileVk: device creation attempt %d failed; waiting 60s\n",
            attempt + 1);
        std::this_thread::sleep_for(std::chrono::seconds(60));
    }
    return {};
}

bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) ==
        VK_SUCCESS;
}

std::shared_ptr<const ChargedBuffer> Upload(std::shared_ptr<DeviceContext> const& context,
    void const* data, VkDeviceSize bytes) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes ? bytes : 4;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto buffer = ChargedBuffer::Create(context, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!buffer) return {};
    if (bytes) {
        void* mapped = nullptr;
        if (vkMapMemory(context->device(), buffer->memory(), 0, bytes, 0, &mapped) !=
            VK_SUCCESS)
            return {};
        std::memcpy(mapped, data, bytes);
        vkUnmapMemory(context->device(), buffer->memory());
    }
    return buffer;
}

template <class T>
bool ReadBuffer(std::shared_ptr<DeviceContext> const& context,
    std::shared_ptr<const ChargedBuffer> const& buffer, std::vector<T>* out) {
    if (!buffer || !out) return false;
    size_t const count = size_t(buffer->sizeBytes() / sizeof(T));
    out->resize(count);
    if (count == 0) return true;
    void* mapped = nullptr;
    if (vkMapMemory(context->device(), buffer->memory(), 0, count * sizeof(T), 0, &mapped) !=
        VK_SUCCESS)
        return false;
    std::memcpy(out->data(), mapped, count * sizeof(T));
    vkUnmapMemory(context->device(), buffer->memory());
    return true;
}

PickQuery IdentityQuery(float x, float y, float radius) {
    PickQuery query{};
    query.viewProj[0] = query.viewProj[5] = query.viewProj[10] = query.viewProj[15] = 1.0f;
    query.width = query.height = 100;
    query.x = x;
    query.y = y;
    query.radiusPx = radius;
    return query;
}

PicktileVkQuery ToVk(PickQuery const& query) {
    PicktileVkQuery out{};
    std::memcpy(out.viewProj, query.viewProj, sizeof(out.viewProj));
    out.width = uint32_t(query.width);
    out.height = uint32_t(query.height);
    out.x = query.x;
    out.y = query.y;
    out.radiusPx = query.radiusPx;
    return out;
}

template <class T>
DeviceView<const T> Read(DeviceBuffer<T> const& buffer) {
    return buffer.view();
}

template <class T>
bool CudaUpload(DeviceBuffer<T>& device, std::vector<T> const& host) {
    if (device.reset(host.size()) != cudaSuccess) return false;
    if (host.empty()) return true;
    return cudaMemcpy(device.data(), host.data(), host.size() * sizeof(T),
               cudaMemcpyHostToDevice) == cudaSuccess;
}

bool SamePick(PickResult const& cuda, PicktileVkResult const& vk) {
    // Exact-equality first: a miss on both sides reports +inf, and
    // fabs(inf - inf) is NaN, which never satisfies the tolerance leg.
    return cuda.hit == vk.hit && cuda.curve == vk.curve && cuda.cv == vk.cv &&
        cuda.flatIndex == vk.flatIndex && cuda.stableId == vk.stableId &&
        (cuda.distancePx == vk.distancePx ||
         std::fabs(cuda.distancePx - vk.distancePx) <= 1e-6f);
}

} // namespace

// Minimal neutral owner with Vulkan identity for the fail-closed proofs.
namespace {
class VulkanIdentityOwner final : public UsdGenDeviceOwner {
public:
    bool ProducerReady() const noexcept override { return true; }
    std::unique_ptr<UsdGenDeviceConsumer> AcquireConsumer(
        UsdGenDeviceStream) const noexcept override {
        return {};
    }
};
} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
            "usage: %s pick.spv footprint.spv tiles.spv bounds.spv indices.spv\n", argv[0]);
        return 1;
    }

    StaticCorrespondence();

    // CUDA twin health preflight: a cold CUDA context can fail its first
    // allocations spuriously under tenant load (later calls in the same
    // process succeed), so warm up with a malloc roundtrip before trusting
    // any twin comparison. A pass afterwards is meaningful; mid-run flakes
    // fail loud.
    {
        bool healthy = false;
        for (int attempt = 0; attempt < 12 && !healthy; ++attempt) {
            (void)cudaGetLastError(); // clear any sticky cold-start error
            int count = 0;
            void* probe = nullptr;
            char host[64] = {7};
            char back[64] = {};
            healthy = cudaGetDeviceCount(&count) == cudaSuccess && count > 0 &&
                cudaMalloc(&probe, 4096) == cudaSuccess &&
                cudaMemcpy(probe, host, sizeof(host), cudaMemcpyHostToDevice) ==
                    cudaSuccess &&
                cudaMemcpy(back, probe, sizeof(back), cudaMemcpyDeviceToHost) ==
                    cudaSuccess &&
                back[0] == 7;
            if (probe) cudaFree(probe);
            if (!healthy) {
                std::printf("picktileVk: CUDA preflight attempt %d failed; waiting 5s\n",
                    attempt + 1);
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        }
        if (!healthy) {
            std::fprintf(stderr, "picktileVk: CUDA twin retry-pending\n");
            return 1;
        }
    }

    RequirementsParity();

    // ---- Host-only admission/metadata/lease proofs (no device needed). ----
    {
        UsdGenDeviceGeometryMetadata geometry{};
        geometry.elementCount = 2;
        geometry.pointCount = 5;
        std::vector<UsdGenDeviceChannelMetadata> channels{
            {TfToken("points"), UsdGenDeviceValueType::Float32x3, UsdGenDeviceDomain::Point, 5, 3, 12,
                true, UsdGenDeviceChannelSemantic::Points},
            {TfToken("curveOffsets"), UsdGenDeviceValueType::UInt32, UsdGenDeviceDomain::Topology, 3,
                1, 4, true, UsdGenDeviceChannelSemantic::ElementOffsets},
            {TfToken("stableIds"), UsdGenDeviceValueType::UInt64, UsdGenDeviceDomain::Primitive, 2,
                1, 8, true, UsdGenDeviceChannelSemantic::StableIds},
        };
        CHECK(ValidatePicktileVkToolChannels(geometry, channels));
        CHECK(!ValidatePicktileVkToolChannels(geometry, {}));
        auto bad = channels;
        bad[1].elementCount = 2;
        CHECK(!ValidatePicktileVkToolChannels(geometry, bad));
        bad = channels;
        bad[2].elementCount = 9;
        CHECK(!ValidatePicktileVkToolChannels(geometry, bad));

        PicktileVkTileSpan spans[2]{{0, 0, 1, 0, 2}, {1, 1, 1, 2, 3}};
        float minimums[6]{-1, 0, 1, -2, -2, -2};
        float maximums[6]{5, 6, 7, 10, 6, 4};
        std::vector<UsdGenDeviceTileMetadata> tiles;
        CHECK(PicktileVkPublishTileMetadata(spans, minimums, maximums, 2, &tiles) &&
            tiles.size() == 2 && tiles[0].tile == 0 && tiles[1].firstElement == 1 &&
            tiles[1].pointCount == 3 && tiles[0].boundsValid &&
            tiles[1].extentMax[0] == 10.0f);
        float flipped[6]{6, 7, 8, 11, 7, 5};
        CHECK(!PicktileVkPublishTileMetadata(spans, flipped, maximums, 2, &tiles));
        float nan[6]{-1, 0, 1, std::numeric_limits<float>::quiet_NaN(), -2, -2};
        CHECK(!PicktileVkPublishTileMetadata(spans, nan, maximums, 2, &tiles));
        CHECK(!PicktileVkPublishTileMetadata(nullptr, minimums, maximums, 2, &tiles));

        auto owner = std::make_shared<VulkanIdentityOwner>();
        UsdGenDeviceGeneration::CreateInfo info;
        info.identity.backend = usdGenTestUtils::Backend("vulkan");
        info.identity.deviceIndex = 0;
        info.identity.generation = 7;
        info.geometry.elementCount = 2;
        info.geometry.pointCount = 5;
        info.owner = owner;
        std::string reason;
        auto generation = UsdGenDeviceGeneration::Create(info, &reason);
        CHECK(bool(generation));
        if (generation) {
            // CUDA leases keep failing closed for Vulkan state (GapRejections
            // invariant, preserved by this port).
            CHECK(!AcquireGeometry(generation, nullptr));
            CHECK(!AcquireGeometryTile(generation, 0, 7, nullptr));
            // The Vulkan leases fail closed for a foreign (non-adapter) owner.
            CHECK(!AcquirePicktileVkGeometry(generation, 0));
            CHECK(!AcquirePicktileVkTile(generation, 0, 7, 0));
            std::vector<UsdGenDeviceTileMetadata> attach{
                {0, 0, 1, 0, 2, {-1, 0, 1}, {5, 6, 7}, true},
                {1, 1, 1, 2, 3, {-2, -2, -2}, {10, 6, 4}, true},
            };
            auto wrapped = PicktileVkWithTileMetadata(generation, attach, &reason);
            CHECK(bool(wrapped) && wrapped->Geometry().tiles.size() == 2 &&
                wrapped->Identity().generation == 7 &&
                wrapped->Geometry().tiles[1].boundsValid);
            std::vector<UsdGenDeviceTileMetadata> gap{{0, 0, 1, 1, 2}};
            CHECK(!PicktileVkWithTileMetadata(generation, gap, &reason));
            // Generation-mismatch and unknown-tile fail closed.
            CHECK(!AcquirePicktileVkTile(wrapped, 0, 8, 0));
            CHECK(!AcquirePicktileVkTile(wrapped, 99, 7, 0));
        }
    }

    // ---- Device sections. ----
    auto pickCode = LoadSpv(argv[1]);
    auto footprintCode = LoadSpv(argv[2]);
    auto tilesCode = LoadSpv(argv[3]);
    auto boundsCode = LoadSpv(argv[4]);
    auto indicesCode = LoadSpv(argv[5]);
    CHECK(!pickCode.empty());
    CHECK(!footprintCode.empty());
    CHECK(!tilesCode.empty());
    CHECK(!boundsCode.empty());
    CHECK(!indicesCode.empty());
    if (g_fail) return 1;

    bool unavailable = false;
    auto native = CreateNativeRetry(&unavailable);
    if (unavailable) return 77;
    if (!native) {
        std::fprintf(stderr, "picktileVk: NVIDIA device retry-pending\n");
        return 1;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(native->physical, &props);
    if (props.limits.maxPerStageDescriptorStorageBuffers < 8 ||
        props.limits.maxPushConstantsSize < 48 ||
        props.limits.maxComputeWorkGroupInvocations < 256 ||
        props.limits.maxComputeWorkGroupSize[0] < 256)
        return 77;
    DeviceContext::CreateInfo ci;
    ci.instance = native->instance;
    ci.physicalDevice = native->physical;
    ci.device = native->device;
    ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 8011;
    ci.nativeLifetime = native;
    ci.timelineSemaphoreEnabled = true;
    ci.resources = {size_t{64} << 20, 0};
    auto context = DeviceContext::Create(ci);
    CHECK(bool(context));
    if (!context) return 1;
    VkResult status = VK_SUCCESS;
    auto pickPipe = PicktileVkPickPipeline::Create(context, pickCode, &status);
    CHECK(pickPipe && status == VK_SUCCESS);
    auto footprintPipe = PicktileVkFootprintPipeline::Create(context, footprintCode, &status);
    CHECK(footprintPipe && status == VK_SUCCESS);
    auto tilesPipe = PicktileVkTilesPipeline::Create(context, tilesCode, &status);
    CHECK(tilesPipe && status == VK_SUCCESS);
    auto boundsPipe = PicktileVkBoundsPipeline::Create(context, boundsCode, &status);
    CHECK(boundsPipe && status == VK_SUCCESS);
    auto indicesPipe = PicktileVkIndicesPipeline::Create(context, indicesCode, &status);
    CHECK(indicesPipe && status == VK_SUCCESS);
    if (g_fail) return 1;

    auto provePollPick = [&](std::unique_ptr<PicktileVkPickPipeline::Candidate>& candidate,
                             PicktileVkPickPipeline::Semantic* semantic) {
        CHECK(Prove(native));
        VkResult proved = candidate->Poll(semantic);
        for (int spin = 0; spin < 100 && proved == VK_NOT_READY; ++spin) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            proved = candidate->Poll(semantic);
        }
        CHECK(proved == VK_SUCCESS);
        return proved == VK_SUCCESS;
    };
    auto provePollFootprint =
        [&](std::unique_ptr<PicktileVkFootprintPipeline::Candidate>& candidate,
            PicktileVkFootprintPipeline::Semantic* semantic) {
            CHECK(Prove(native));
            VkResult proved = candidate->Poll(semantic);
            for (int spin = 0; spin < 100 && proved == VK_NOT_READY; ++spin) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                proved = candidate->Poll(semantic);
            }
            CHECK(proved == VK_SUCCESS);
            return proved == VK_SUCCESS;
        };
    auto provePollTiles = [&](std::unique_ptr<PicktileVkTilesPipeline::Candidate>& candidate,
                              uint32_t* deviceStatus) {
        CHECK(Prove(native));
        VkResult proved = candidate->Poll(deviceStatus);
        for (int spin = 0; spin < 100 && proved == VK_NOT_READY; ++spin) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            proved = candidate->Poll(deviceStatus);
        }
        CHECK(proved == VK_SUCCESS);
        return proved == VK_SUCCESS;
    };
    auto provePollBounds = [&](std::unique_ptr<PicktileVkBoundsPipeline::Candidate>& candidate,
                               uint32_t* deviceStatus) {
        CHECK(Prove(native));
        VkResult proved = candidate->Poll(deviceStatus);
        for (int spin = 0; spin < 100 && proved == VK_NOT_READY; ++spin) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            proved = candidate->Poll(deviceStatus);
        }
        CHECK(proved == VK_SUCCESS);
        return proved == VK_SUCCESS;
    };
    auto provePollIndices =
        [&](std::unique_ptr<PicktileVkIndicesPipeline::Candidate>& candidate,
            uint32_t* deviceStatus) {
            CHECK(Prove(native));
            VkResult proved = candidate->Poll(deviceStatus);
            for (int spin = 0; spin < 100 && proved == VK_NOT_READY; ++spin) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                proved = candidate->Poll(deviceStatus);
            }
            CHECK(proved == VK_SUCCESS);
            return proved == VK_SUCCESS;
        };

    // ============================ PICK ============================
    {
        std::vector<float> points{0, 0, 0, 0.2f, 0, 0, -0.5f, 0, 0, 0.5f, 0, 0, 0, 0, 0};
        std::vector<uint32_t> offsets{0, 2, 5};
        std::vector<uint64_t> ids{100, 200};
        auto vkPoints = Upload(context, points.data(), points.size() * 4);
        auto vkOffsets = Upload(context, offsets.data(), offsets.size() * 4);
        auto vkIds = Upload(context, ids.data(), ids.size() * 8);
        CHECK(vkPoints && vkOffsets && vkIds);

        DeviceBuffer<::float3> cuPoints;
        DeviceBuffer<uint32_t> cuOffsets;
        DeviceBuffer<uint64_t> cuIds;
        std::vector<::float3> cuSource{{0, 0, 0}, {0.2f, 0, 0}, {-0.5f, 0, 0}, {0.5f, 0, 0},
            {0, 0, 0}};
        CHECK(CudaUpload(cuPoints, cuSource) && CudaUpload(cuOffsets, offsets) &&
            CudaUpload(cuIds, ids));
        DeviceCurveGeometryView geometry{{cuPoints.data(), 5}, {}, {},
            {cuOffsets.data(), 3}, {cuIds.data(), 2}, 2, 5};
        CudaPicking picker;

        auto runPick = [&](PickQuery const& query, char const* label) {
            PickingStatus cudaStatus = picker.ApplyPick(geometry, query, nullptr);
            PickResult cudaResult{};
            if (cudaStatus == PickingStatus::Ok) {
                cudaStatus = picker.Finish(nullptr);
                cudaResult = picker.result();
            }
            auto candidate = pickPipe->Begin(vkPoints, vkOffsets, vkIds, 2, 5, ToVk(query),
                &status);
            bool vkBegan = bool(candidate);
            PicktileVkPickPipeline::Semantic semantic;
            bool vkOk = false;
            PicktileVkResult vkResult{};
            if (vkBegan && provePollPick(candidate, &semantic)) {
                vkOk = semantic == PicktileVkPickPipeline::Semantic::Ok;
                vkResult = candidate->result();
            }
            char text[192];
            std::snprintf(text, sizeof(text), "pick parity %s (cuda=%d vk=%d)", label,
                int(cudaStatus), vkBegan ? int(semantic) : -1);
            bool same = (cudaStatus == PickingStatus::Ok) == (vkBegan && vkOk) &&
                (cudaStatus != PickingStatus::Ok || SamePick(cudaResult, vkResult));
            if (!same)
                std::printf("pick mismatch %s: cuda{hit=%d curve=%u cv=%u flat=%u id=%llu d=%.6g} "
                    "vk{hit=%d curve=%u cv=%u flat=%u id=%llu d=%.6g}\n", label,
                    int(cudaResult.hit), cudaResult.curve, cudaResult.cv,
                    cudaResult.flatIndex, (unsigned long long)cudaResult.stableId,
                    double(cudaResult.distancePx), int(vkResult.hit), vkResult.curve,
                    vkResult.cv, vkResult.flatIndex,
                    (unsigned long long)vkResult.stableId, double(vkResult.distancePx));
            Check(same, text);
        };
        runPick(IdentityQuery(50, 50, 1), "identity tie -> flat 0");
        PickQuery translated = IdentityQuery(60, 50, 1);
        translated.viewProj[12] = 0.2f;
        runPick(translated, "row-vector translation");
        runPick(IdentityQuery(75, 50, 1), "clean miss");

        // Perspective-style varying w, no stable ids (index fallback).
        {
            std::vector<float> ppoints{0, 0, 0, 0.5f, 0, 2};
            std::vector<uint32_t> poffsets{0, 2};
            auto vkP = Upload(context, ppoints.data(), ppoints.size() * 4);
            auto vkO = Upload(context, poffsets.data(), poffsets.size() * 4);
            DeviceBuffer<::float3> cuP;
            DeviceBuffer<uint32_t> cuO;
            CHECK(CudaUpload(cuP, std::vector<::float3>{{0, 0, 0}, {0.5f, 0, 2}}) &&
                CudaUpload(cuO, poffsets));
            DeviceCurveGeometryView perspective{{cuP.data(), 2}, {}, {},
                {cuO.data(), 2}, {}, 1, 2};
            PickQuery query = IdentityQuery(50, 50, 1);
            query.viewProj[11] = 0.5f;
            CHECK(picker.ApplyPick(perspective, query, nullptr) == PickingStatus::Ok &&
                picker.Finish(nullptr) == PickingStatus::Ok);
            PickResult cudaResult = picker.result();
            auto candidate = pickPipe->Begin(vkP, vkO, nullptr, 1, 2, ToVk(query), &status);
            CHECK(bool(candidate));
            if (!candidate) return 1;
            PicktileVkPickPipeline::Semantic semantic;
            CHECK(provePollPick(candidate, &semantic) &&
                semantic == PicktileVkPickPipeline::Semantic::Ok &&
                SamePick(cudaResult, candidate->result()) && cudaResult.hit &&
                cudaResult.stableId == 0);
        }

        // Footprint parity: exact ascending indices.
        auto runFootprint = [&](PickQuery const& query, std::vector<int32_t> const& want,
                                char const* label) {
            CHECK(picker.ApplyFootprint(geometry, query, nullptr) == PickingStatus::Ok &&
                picker.Finish(nullptr) == PickingStatus::Ok);
            std::vector<int32_t> cudaIndices(picker.footprintCount());
            if (!cudaIndices.empty())
                CHECK(cudaMemcpy(cudaIndices.data(), picker.footprint().data,
                          cudaIndices.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
            auto candidate = footprintPipe->Begin(vkPoints, vkOffsets, 2, 5, ToVk(query),
                &status);
            CHECK(bool(candidate));
            if (!candidate) return;
            PicktileVkFootprintPipeline::Semantic semantic;
            bool ok = provePollFootprint(candidate, &semantic) &&
                semantic == PicktileVkFootprintPipeline::Semantic::Ok;
            std::vector<int32_t> vkIndices;
            ok = ok && ReadBuffer(context, candidate->footprint(), &vkIndices);
            vkIndices.resize(candidate->footprintCount());
            char text[160];
            std::snprintf(text, sizeof(text), "footprint parity %s", label);
            Check(ok && cudaIndices == want && vkIndices == want, text);
        };
        runFootprint(IdentityQuery(50, 50, 11), {0, 1, 4}, "three-hit ascending");

        // 1025-point all-hit footprint (multi-stride device scan parity).
        {
            std::vector<float> many(1025 * 3, 0.0f);
            std::vector<uint32_t> manyOffsets{0, 1025};
            auto vkMany = Upload(context, many.data(), many.size() * 4);
            auto vkManyO = Upload(context, manyOffsets.data(), manyOffsets.size() * 4);
            DeviceBuffer<::float3> cuMany;
            DeviceBuffer<uint32_t> cuManyO;
            CHECK(CudaUpload(cuMany, std::vector<::float3>(1025, {0, 0, 0})) &&
                CudaUpload(cuManyO, manyOffsets));
            DeviceCurveGeometryView manyView{{cuMany.data(), 1025}, {}, {},
                {cuManyO.data(), 2}, {}, 1, 1025};
            CHECK(picker.ApplyFootprint(manyView, IdentityQuery(50, 50, 1), nullptr) ==
                    PickingStatus::Ok &&
                picker.Finish(nullptr) == PickingStatus::Ok &&
                picker.footprintCount() == 1025);
            std::vector<int32_t> cudaIndices(1025);
            CHECK(cudaMemcpy(cudaIndices.data(), picker.footprint().data, 1025 * 4,
                      cudaMemcpyDeviceToHost) == cudaSuccess);
            auto candidate = footprintPipe->Begin(vkMany, vkManyO, 1, 1025,
                ToVk(IdentityQuery(50, 50, 1)), &status);
            CHECK(bool(candidate));
            if (!candidate) return 1;
            PicktileVkFootprintPipeline::Semantic semantic;
            CHECK(provePollFootprint(candidate, &semantic) &&
                semantic == PicktileVkFootprintPipeline::Semantic::Ok &&
                candidate->footprintCount() == 1025);
            std::vector<int32_t> vkIndices;
            CHECK(ReadBuffer(context, candidate->footprint(), &vkIndices));
            bool ascending = vkIndices.size() == 1025;
            for (size_t i = 0; ascending && i < 1025; ++i)
                ascending = vkIndices[i] == int32_t(i) && cudaIndices[i] == int32_t(i);
            CHECK(ascending);
        }

        // Behind-camera miss.
        {
            PickQuery clipped = IdentityQuery(50, 50, 100);
            clipped.viewProj[15] = -1.0f;
            runPick(clipped, "behind-camera miss");
        }
        // Non-finite matrix: host rejects on both (CUDA: no pending op).
        {
            PickQuery bad = IdentityQuery(0, 0, 1);
            bad.viewProj[0] = std::numeric_limits<float>::quiet_NaN();
            CHECK(picker.ApplyPick(geometry, bad, nullptr) == PickingStatus::NonFiniteInput &&
                !picker.pending());
            auto candidate =
                pickPipe->Begin(vkPoints, vkOffsets, vkIds, 2, 5, ToVk(bad), &status);
            CHECK(!candidate);
        }
        // Malformed device offsets diagnose on device on both.
        {
            std::vector<uint32_t> badOffsets{0, 2, 99};
            auto vkBad = Upload(context, badOffsets.data(), badOffsets.size() * 4);
            CHECK(CudaUpload(cuOffsets, badOffsets));
            CHECK(picker.ApplyFootprint(geometry, IdentityQuery(50, 50, 11), nullptr) ==
                    PickingStatus::Ok &&
                picker.Finish(nullptr) == PickingStatus::InvalidArgument);
            auto candidate = footprintPipe->Begin(vkPoints, vkBad, 2, 5,
                ToVk(IdentityQuery(50, 50, 11)), &status);
            CHECK(bool(candidate));
            if (!candidate) return 1;
            PicktileVkFootprintPipeline::Semantic semantic;
            CHECK(provePollFootprint(candidate, &semantic) &&
                semantic == PicktileVkFootprintPipeline::Semantic::InvalidArgument &&
                !candidate->succeeded() && !candidate->footprint());
            CHECK(CudaUpload(cuOffsets, offsets));
        }
        // Non-finite point diagnoses on device on both.
        {
            std::vector<float> badPoints = points;
            badPoints[3] = std::numeric_limits<float>::quiet_NaN();
            auto vkBad = Upload(context, badPoints.data(), badPoints.size() * 4);
            auto badSource = cuSource;
            badSource[1].x = std::numeric_limits<float>::quiet_NaN();
            CHECK(CudaUpload(cuPoints, badSource));
            CHECK(picker.ApplyPick(geometry, IdentityQuery(50, 50, 1), nullptr) ==
                    PickingStatus::Ok &&
                picker.Finish(nullptr) == PickingStatus::NonFiniteInput);
            auto candidate = pickPipe->Begin(vkBad, vkOffsets, vkIds, 2, 5,
                ToVk(IdentityQuery(50, 50, 1)), &status);
            CHECK(bool(candidate));
            if (!candidate) return 1;
            PicktileVkPickPipeline::Semantic semantic;
            CHECK(provePollPick(candidate, &semantic) &&
                semantic == PicktileVkPickPipeline::Semantic::NonFiniteInput &&
                !candidate->succeeded());
            CHECK(CudaUpload(cuPoints, cuSource));
        }
        // Empty geometry is valid on both.
        {
            std::vector<uint32_t> emptyOffsets{0};
            auto vkEmpty = Upload(context, emptyOffsets.data(), emptyOffsets.size() * 4);
            auto vkNoPoints = Upload(context, nullptr, 0);
            DeviceBuffer<uint32_t> cuEmpty;
            CHECK(CudaUpload(cuEmpty, emptyOffsets));
            DeviceCurveGeometryView empty{{}, {}, {}, {cuEmpty.data(), 1}, {}, 0, 0};
            CudaPicking emptyPicker;
            CHECK(emptyPicker.ApplyPick(empty, IdentityQuery(1, 1, 2), nullptr) ==
                    PickingStatus::Ok &&
                emptyPicker.Finish(nullptr) == PickingStatus::Ok &&
                !emptyPicker.result().hit);
            auto candidate = pickPipe->Begin(vkNoPoints, vkEmpty, nullptr, 0, 0,
                ToVk(IdentityQuery(1, 1, 2)), &status);
            CHECK(bool(candidate));
            if (!candidate) return 1;
            PicktileVkPickPipeline::Semantic semantic;
            CHECK(provePollPick(candidate, &semantic) &&
                semantic == PicktileVkPickPipeline::Semantic::Ok &&
                !candidate->result().hit);
            CHECK(emptyPicker.ApplyFootprint(empty, IdentityQuery(1, 1, 2), nullptr) ==
                    PickingStatus::Ok &&
                emptyPicker.Finish(nullptr) == PickingStatus::Ok &&
                emptyPicker.footprintCount() == 0);
            auto foot = footprintPipe->Begin(vkNoPoints, vkEmpty, 0, 0,
                ToVk(IdentityQuery(1, 1, 2)), &status);
            CHECK(bool(foot));
            if (!foot) return 1;
            PicktileVkFootprintPipeline::Semantic fsemantic;
            CHECK(provePollFootprint(foot, &fsemantic) &&
                fsemantic == PicktileVkFootprintPipeline::Semantic::Ok &&
                foot->footprintCount() == 0);
        }
    }

    // ============================ TILES ============================
    {
        auto runTiles = [&](CurveTileOptions options, std::vector<uint64_t> const& capture,
                            std::vector<uint64_t> const& survivors,
                            std::vector<uint32_t> const& offsets, size_t pointCount,
                            CurveTileOrder order, char const* label,
                            std::vector<CurveTileSpan> const* want) {
            // CUDA twin.
            CurveTileRequirements cudaReq{};
            bool const cudaReqOk =
                CudaTilesReq(options, capture.size(), survivors.size(), pointCount, &cudaReq);
            cudaError_t cudaEnqueue = cudaErrorInvalidValue;
            uint32_t cudaStatus = UINT32_MAX;
            std::vector<CurveTileSpan> cudaSpans;
            if (cudaReqOk) {
                DeviceBuffer<uint64_t> cuCap, cuSurv, cuSorted;
                DeviceBuffer<uint32_t> cuOff, cuStatus, cuOrd;
                DeviceBuffer<CurveTileSpan> cuSpans;
                std::vector<uint64_t> lookupIds;
                std::vector<uint32_t> lookupOrdinals;
                if (order == CurveTileOrder::CaptureOrderSurvivorSubset) {
                    std::vector<std::pair<uint64_t, uint32_t>> pairs;
                    for (size_t i = 0; i < capture.size(); ++i)
                        pairs.emplace_back(capture[i], uint32_t(i));
                    std::sort(pairs.begin(), pairs.end());
                    for (auto const& entry : pairs) {
                        lookupIds.push_back(entry.first);
                        lookupOrdinals.push_back(entry.second);
                    }
                }
                bool up = CudaUpload(cuCap, capture) && CudaUpload(cuSurv, survivors) &&
                    CudaUpload(cuOff, offsets) && CudaUpload(cuSorted, lookupIds) &&
                    CudaUpload(cuOrd, lookupOrdinals) && cuStatus.reset(1) == cudaSuccess &&
                    cuSpans.reset(cudaReq.tileCount) == cudaSuccess;
                if (up) {
                    CurveTileInput input{Read(cuCap), Read(cuSurv), Read(cuOff),
                        capture.size(), survivors.size(), pointCount, order};
                    input.sortedCaptureStableIds = Read(cuSorted);
                    input.sortedCaptureOrdinals = Read(cuOrd);
                    CurveTileOutput output{cuSpans.view(), cuStatus.view()};
                    cudaEnqueue = BuildCurveTiles(input, cudaReq, output, nullptr);
                    if (cudaEnqueue == cudaSuccess &&
                        cudaDeviceSynchronize() == cudaSuccess) {
                        cudaSpans.resize(cuSpans.size());
                        if (cudaMemcpy(&cudaStatus, cuStatus.data(), 4,
                                cudaMemcpyDeviceToHost) == cudaSuccess &&
                            (cudaSpans.empty() ||
                                cudaMemcpy(cudaSpans.data(), cuSpans.data(),
                                    cudaSpans.size() * sizeof(CurveTileSpan),
                                    cudaMemcpyDeviceToHost) == cudaSuccess)) {
                        } else {
                            cudaStatus = UINT32_MAX;
                        }
                    }
                }
            }
            // Vulkan port.
            PicktileVkTileRequirements vkReq{};
            bool const vkReqOk = GetPicktileVkTileRequirements(options.chunkSize,
                options.tileTarget, capture.size(), survivors.size(), pointCount, &vkReq);
            bool vkBegan = false;
            uint32_t vkStatus = UINT32_MAX;
            std::vector<PicktileVkTileSpan> vkSpans;
            if (vkReqOk) {
                std::vector<uint64_t> lookupIds;
                std::vector<uint32_t> lookupOrdinals;
                if (order == CurveTileOrder::CaptureOrderSurvivorSubset) {
                    std::vector<std::pair<uint64_t, uint32_t>> pairs;
                    for (size_t i = 0; i < capture.size(); ++i)
                        pairs.emplace_back(capture[i], uint32_t(i));
                    std::sort(pairs.begin(), pairs.end());
                    for (auto const& entry : pairs) {
                        lookupIds.push_back(entry.first);
                        lookupOrdinals.push_back(entry.second);
                    }
                }
                auto vkCap = Upload(context, capture.data(), capture.size() * 8);
                auto vkSurv = Upload(context, survivors.data(), survivors.size() * 8);
                auto vkOff = Upload(context, offsets.data(), offsets.size() * 4);
                std::shared_ptr<const ChargedBuffer> vkSorted, vkOrd;
                if (order == CurveTileOrder::CaptureOrderSurvivorSubset) {
                    vkSorted = Upload(context, lookupIds.data(), lookupIds.size() * 8);
                    vkOrd = Upload(context, lookupOrdinals.data(), lookupOrdinals.size() * 4);
                }
                auto candidate = tilesPipe->Begin(vkCap, vkSurv, vkOff, vkSorted, vkOrd,
                    uint32_t(capture.size()), uint32_t(survivors.size()),
                    uint32_t(pointCount), PicktileVkTileOrder(order), vkReq, &status);
                vkBegan = bool(candidate);
                if (vkBegan && provePollTiles(candidate, &vkStatus) && vkStatus == 0)
                    CHECK(ReadBuffer(context, candidate->spans(), &vkSpans));
            }
            char text[220];
            std::snprintf(text, sizeof(text), "tiles parity %s (cudaReq=%d cudaQ=%d/%u vk=%d/%u)",
                label, int(cudaReqOk), int(cudaEnqueue), cudaStatus, int(vkBegan), vkStatus);
            bool ok = cudaReqOk == vkReqOk &&
                (cudaEnqueue == cudaSuccess) == vkBegan && cudaStatus == vkStatus;
            if (ok && cudaEnqueue == cudaSuccess && cudaStatus == 0) {
                ok = cudaSpans.size() == vkSpans.size() &&
                    std::memcmp(cudaSpans.data(), vkSpans.data(),
                        cudaSpans.size() * sizeof(CurveTileSpan)) == 0;
                if (ok && want)
                    ok = want->size() == cudaSpans.size() &&
                        std::memcmp(want->data(), cudaSpans.data(),
                            want->size() * sizeof(CurveTileSpan)) == 0;
            }
            Check(ok, text);
        };
        std::vector<uint64_t> capture300(300);
        for (uint64_t i = 0; i < 300; ++i) capture300[i] = i;
        std::vector<CurveTileSpan> const golden{{0, 0, 2, 0, 5}, {1, 2, 3, 5, 15},
            {2, 5, 1, 20, 7}};
        runTiles({128, 64}, capture300, {0, 127, 128, 129, 255, 299}, {0, 2, 5, 9, 14, 20, 27},
            27, CurveTileOrder::SortedSurvivorSubset, "ragged golden", &golden);
        std::vector<CurveTileSpan> const sparse{{0, 0, 1, 0, 2}, {1, 1, 0, 2, 0},
            {2, 1, 1, 2, 7}};
        runTiles({128, 64}, capture300, {0, 299}, {0, 2, 9}, 9,
            CurveTileOrder::SortedSurvivorSubset, "sparse retained tile", &sparse);
        runTiles({128, 64}, capture300, {}, {0}, 0, CurveTileOrder::SortedSurvivorSubset,
            "all-culled", nullptr);
        runTiles({128, 64}, {}, {}, {0}, 0, CurveTileOrder::SortedSurvivorSubset,
            "empty capture", nullptr);
        std::vector<uint64_t> reversed(300);
        std::vector<uint32_t> reversedOffsets(301);
        for (size_t i = 0; i < 300; ++i) {
            reversed[i] = 300 - i;
            reversedOffsets[i] = uint32_t(i * 2);
        }
        reversedOffsets.back() = 600;
        std::vector<CurveTileSpan> const identity{{0, 0, 128, 0, 256}, {1, 128, 128, 256, 256},
            {2, 256, 44, 512, 88}};
        runTiles({128, 64}, reversed, reversed, reversedOffsets, 600,
            CurveTileOrder::IdentityCaptureOrder, "identity order", &identity);
        auto mismatch = reversed;
        mismatch[17] += 1;
        runTiles({128, 64}, reversed, mismatch, reversedOffsets, 600,
            CurveTileOrder::IdentityCaptureOrder, "identity mismatch -> status 1", nullptr);
        std::vector<uint64_t> shuffled(300);
        for (size_t i = 0; i < 300; ++i) shuffled[i] = 1000 + (i * 137) % 300;
        std::vector<uint64_t> coSurv{shuffled[0], shuffled[127], shuffled[128], shuffled[255],
            shuffled[299]};
        std::vector<CurveTileSpan> const coWant{
            {0, 0, 2, 0, 5}, {1, 2, 2, 5, 9}, {2, 4, 1, 14, 6}};
        runTiles({128, 64}, shuffled, coSurv, {0, 2, 5, 9, 14, 20}, 20,
            CurveTileOrder::CaptureOrderSurvivorSubset, "capture-order survivors", &coWant);
        runTiles({128, 64}, shuffled, {shuffled[0], 99999}, {0, 2, 5}, 5,
            CurveTileOrder::CaptureOrderSurvivorSubset, "unknown survivor -> status 1",
            nullptr);
        runTiles({128, 64}, {10, 20, 30}, {10, 25}, {0, 1, 2}, 2,
            CurveTileOrder::SortedSurvivorSubset, "unknown id -> status 1", nullptr);
        runTiles({128, 64}, {10, 20, 30}, {10, 10}, {0, 1, 2}, 2,
            CurveTileOrder::SortedSurvivorSubset, "duplicate ids -> status 1", nullptr);
        runTiles({128, 64}, {10, 20, 30}, {20, 10}, {0, 1, 2}, 2,
            CurveTileOrder::SortedSurvivorSubset, "reordered ids -> status 1", nullptr);
        runTiles({128, 64}, {10, 20, 30}, {10, 20}, {1, 2, 3}, 3,
            CurveTileOrder::SortedSurvivorSubset, "nonzero first offset -> status 1", nullptr);
        runTiles({128, 64}, {10, 20, 30}, {10, 20}, {0, 2, 2}, 3,
            CurveTileOrder::SortedSurvivorSubset, "bad terminal offset -> status 1", nullptr);
        runTiles({128, 64}, {10, 10, 30}, {10}, {0, 1}, 1,
            CurveTileOrder::SortedSurvivorSubset, "duplicate capture -> status 1", nullptr);
        // Host rejections agree.
        {
            PicktileVkTileRequirements vkReq{};
            CHECK(GetPicktileVkTileRequirements(128, 64, 3, 2, 2, &vkReq));
            auto vkCap = Upload(context, std::vector<uint64_t>{10, 20, 30}.data(), 24);
            auto vkSurv = Upload(context, std::vector<uint64_t>{10, 20}.data(), 16);
            auto vkOff = Upload(context, std::vector<uint32_t>{0, 1, 2}.data(), 12);
            auto badOrder = tilesPipe->Begin(vkCap, vkSurv, vkOff, nullptr, nullptr, 3, 2, 2,
                PicktileVkTileOrder(99), vkReq, &status);
            CHECK(!badOrder);
            auto tampered = vkReq;
            tampered.tileCount += 1;
            auto mismatchReq = tilesPipe->Begin(vkCap, vkSurv, vkOff, nullptr, nullptr, 3, 2,
                2, PicktileVkTileOrder::SortedSurvivorSubset, tampered, &status);
            CHECK(!mismatchReq);
        }
    }

    // ============================ BOUNDS ============================
    {
        std::vector<float> points{1, 2, 3, 3, 4, 5, 0, 0, 0, 8, 4, 2, 3, 1, 1};
        std::vector<float> widths{2, 4, 1, 4, 2};
        std::vector<PicktileVkTileSpan> spans{{7, 0, 1, 0, 2}, {42, 1, 1, 2, 3},
            {99, 2, 0, 5, 0}};
        auto runBounds = [&](CurveTileBoundsBasis basis, char const* label) {
            DeviceBuffer<::float3> cuP;
            DeviceBuffer<float> cuW;
            DeviceBuffer<CurveTileSpan> cuS;
            DeviceBuffer<CurveTileBoundsScratch> cuScratch;
            DeviceBuffer<::float3> cuMin, cuMax;
            DeviceBuffer<uint32_t> cuStatus;
            std::vector<::float3> cuPoints{{1, 2, 3}, {3, 4, 5}, {0, 0, 0}, {8, 4, 2},
                {3, 1, 1}};
            std::vector<CurveTileSpan> cuSpans{{7, 0, 1, 0, 2}, {42, 1, 1, 2, 3},
                {99, 2, 0, 5, 0}};
            cudaError_t cudaEnqueue = cudaErrorInvalidValue;
            uint32_t cudaStatus = UINT32_MAX;
            std::vector<::float3> cudaMin(3), cudaMax(3);
            if (CudaUpload(cuP, cuPoints) && CudaUpload(cuW, widths) &&
                CudaUpload(cuS, cuSpans) && cuScratch.reset(3) == cudaSuccess &&
                cuMin.reset(3) == cudaSuccess && cuMax.reset(3) == cudaSuccess &&
                cuStatus.reset(1) == cudaSuccess) {
                CurveTileBoundsInput input{Read(cuP), Read(cuW), Read(cuS)};
                CurveTileBoundsWorkspace workspace{cuScratch.view()};
                CurveTileBoundsOutput output{cuMin.view(), cuMax.view(), cuStatus.view()};
                cudaEnqueue = BuildCurveTileBounds({basis}, input, workspace, output, nullptr);
                if (cudaEnqueue == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
                    cudaMemcpy(&cudaStatus, cuStatus.data(), 4, cudaMemcpyDeviceToHost) ==
                        cudaSuccess &&
                    cudaMemcpy(cudaMin.data(), cuMin.data(), sizeof(::float3) * 3,
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
                    cudaMemcpy(cudaMax.data(), cuMax.data(), sizeof(::float3) * 3,
                        cudaMemcpyDeviceToHost) == cudaSuccess) {
                } else {
                    cudaStatus = UINT32_MAX;
                }
            }
            auto vkP = Upload(context, points.data(), points.size() * 4);
            auto vkW = Upload(context, widths.data(), widths.size() * 4);
            auto vkS = Upload(context, spans.data(), spans.size() * 20);
            auto candidate = boundsPipe->Begin(vkP, vkW, vkS, 5, 3,
                PicktileVkTileBoundsBasis(basis), &status);
            bool vkBegan = bool(candidate);
            uint32_t vkStatus = UINT32_MAX;
            std::vector<float> vkMin, vkMax;
            if (vkBegan && provePollBounds(candidate, &vkStatus) && vkStatus == 0) {
                CHECK(ReadBuffer(context, candidate->minimums(), &vkMin));
                CHECK(ReadBuffer(context, candidate->maximums(), &vkMax));
            }
            char text[220];
            std::snprintf(text, sizeof(text), "bounds parity %s (cudaQ=%d/%u vk=%d/%u)",
                label, int(cudaEnqueue), cudaStatus, int(vkBegan), vkStatus);
            bool ok = (cudaEnqueue == cudaSuccess) == vkBegan && cudaStatus == vkStatus;
            bool exact = ok;
            if (ok && cudaEnqueue == cudaSuccess && cudaStatus == 0) {
                // Accepted within the CUDA bounds test's own directional 1e-4
                // Near/UpperNear windows; the port targets exact equality.
                ok = vkMin.size() == 9 && vkMax.size() == 9;
                for (size_t i = 0; ok && i < 9; ++i) {
                    float const* cMin = reinterpret_cast<float const*>(cudaMin.data());
                    float const* cMax = reinterpret_cast<float const*>(cudaMax.data());
                    float lo = vkMin[i], clo = cMin[i], hi = vkMax[i], chi = cMax[i];
                    if (!(lo <= clo && lo >= clo - 1e-4f && hi >= chi && hi <= chi + 1e-4f))
                        ok = false;
                    if (lo != clo || hi != chi) exact = false;
                }
            }
            Check(ok, text);
            char exactText[160];
            std::snprintf(exactText, sizeof(exactText), "bounds exact %s", label);
            Check(exact, exactText);
        };
        runBounds(CurveTileBoundsBasis::Linear, "linear");
        runBounds(CurveTileBoundsBasis::BSpline, "bspline");
        runBounds(CurveTileBoundsBasis::CatmullRom, "catmull-rom");
        // The CUDA side reproduces its own test's goldens (proves the twin ran).
        {
            DeviceBuffer<::float3> cuP;
            DeviceBuffer<float> cuW;
            DeviceBuffer<CurveTileSpan> cuS;
            DeviceBuffer<CurveTileBoundsScratch> cuScratch;
            DeviceBuffer<::float3> cuMin, cuMax;
            DeviceBuffer<uint32_t> cuStatus;
            std::vector<::float3> cuPoints{{1, 2, 3}, {3, 4, 5}, {0, 0, 0}, {8, 4, 2},
                {3, 1, 1}};
            std::vector<CurveTileSpan> cuSpans{{7, 0, 1, 0, 2}, {42, 1, 1, 2, 3},
                {99, 2, 0, 5, 0}};
            CHECK(CudaUpload(cuP, cuPoints) && CudaUpload(cuW, widths) &&
                CudaUpload(cuS, cuSpans) && cuScratch.reset(3) == cudaSuccess &&
                cuMin.reset(3) == cudaSuccess && cuMax.reset(3) == cudaSuccess &&
                cuStatus.reset(1) == cudaSuccess);
            CurveTileBoundsInput input{Read(cuP), Read(cuW), Read(cuS)};
            CurveTileBoundsWorkspace workspace{cuScratch.view()};
            CurveTileBoundsOutput output{cuMin.view(), cuMax.view(), cuStatus.view()};
            CHECK(BuildCurveTileBounds({CurveTileBoundsBasis::Linear}, input, workspace,
                      output, nullptr) == cudaSuccess &&
                cudaDeviceSynchronize() == cudaSuccess);
            std::vector<::float3> lo(3), hi(3);
            uint32_t st = 99;
            CHECK(cudaMemcpy(&st, cuStatus.data(), 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(lo.data(), cuMin.data(), sizeof(::float3) * 3,
                    cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(hi.data(), cuMax.data(), sizeof(::float3) * 3,
                    cudaMemcpyDeviceToHost) == cudaSuccess);
            auto near = [](float actual, float expected) {
                return actual <= expected && actual >= expected - 1e-4f;
            };
            auto upper = [](float actual, float expected) {
                return actual >= expected && actual <= expected + 1e-4f;
            };
            CHECK(st == 0 && near(lo[1].x, -2) && near(lo[1].y, -2) && near(lo[1].z, -2) &&
                upper(hi[1].x, 10) && upper(hi[1].y, 6) && upper(hi[1].z, 4) && lo[2].x == 0 &&
                hi[2].x == 0);
        }
        // Invalid device inputs fail closed on both, publishing nothing.
        auto runBoundsInvalid = [&](std::vector<float> badPoints, std::vector<float> badWidths,
                                    std::vector<PicktileVkTileSpan> badSpans,
                                    char const* label) {
            DeviceBuffer<::float3> cuP;
            DeviceBuffer<float> cuW;
            DeviceBuffer<CurveTileSpan> cuS;
            DeviceBuffer<CurveTileBoundsScratch> cuScratch;
            DeviceBuffer<::float3> cuMin, cuMax;
            DeviceBuffer<uint32_t> cuStatus;
            std::vector<::float3> cuPoints(badPoints.size() / 3);
            std::memcpy(cuPoints.data(), badPoints.data(), badPoints.size() * 4);
            std::vector<CurveTileSpan> cuSpans(badSpans.size());
            for (size_t i = 0; i < badSpans.size(); ++i)
                cuSpans[i] = {badSpans[i].tile, badSpans[i].firstCurve,
                    badSpans[i].curveCount, badSpans[i].firstPoint,
                    badSpans[i].pointCount};
            uint32_t cudaStatus = UINT32_MAX;
            if (CudaUpload(cuP, cuPoints) && CudaUpload(cuW, badWidths) &&
                CudaUpload(cuS, cuSpans) && cuScratch.reset(3) == cudaSuccess &&
                cuMin.reset(3) == cudaSuccess && cuMax.reset(3) == cudaSuccess &&
                cuStatus.reset(1) == cudaSuccess) {
                CurveTileBoundsInput input{Read(cuP), Read(cuW), Read(cuS)};
                CurveTileBoundsWorkspace workspace{cuScratch.view()};
                CurveTileBoundsOutput output{cuMin.view(), cuMax.view(), cuStatus.view()};
                if (BuildCurveTileBounds({}, input, workspace, output, nullptr) ==
                        cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess)
                    CHECK(cudaMemcpy(&cudaStatus, cuStatus.data(), 4,
                              cudaMemcpyDeviceToHost) == cudaSuccess);
            }
            auto vkP = Upload(context, badPoints.data(), badPoints.size() * 4);
            auto vkW = Upload(context, badWidths.data(), badWidths.size() * 4);
            auto vkS = Upload(context, badSpans.data(), badSpans.size() * 20);
            auto candidate = boundsPipe->Begin(vkP, vkW, vkS, 5, 3,
                PicktileVkTileBoundsBasis::BSpline, &status);
            uint32_t vkStatus = UINT32_MAX;
            bool vkBegan = bool(candidate);
            if (vkBegan) CHECK(provePollBounds(candidate, &vkStatus));
            char text[200];
            std::snprintf(text, sizeof(text), "bounds invalid parity %s (%u/%u)", label,
                cudaStatus, vkStatus);
            Check(vkBegan && cudaStatus == 1 && vkStatus == 1 && !candidate->succeeded() &&
                    !candidate->minimums() && !candidate->maximums(),
                text);
        };
        auto oob = spans;
        oob[1].firstPoint = UINT32_MAX;
        oob[1].pointCount = 1;
        runBoundsInvalid(points, widths, oob, "out-of-range span");
        auto nanPoints = points;
        nanPoints[0] = std::numeric_limits<float>::quiet_NaN();
        runBoundsInvalid(nanPoints, widths, spans, "nonfinite point");
        auto negWidths = widths;
        negWidths[0] = -1.0f;
        runBoundsInvalid(points, negWidths, spans, "negative width");
        auto hugePoints = points;
        hugePoints[0] = std::numeric_limits<float>::max();
        runBoundsInvalid(hugePoints, widths, spans, "unrepresentable bound");
        // Host rejections agree; empty capture is valid on both.
        {
            auto vkP = Upload(context, points.data(), points.size() * 4);
            auto vkW = Upload(context, widths.data(), (widths.size() - 1) * 4);
            auto vkS = Upload(context, spans.data(), spans.size() * 20);
            CHECK(!boundsPipe->Begin(vkP, vkW, vkS, 5, 3,
                PicktileVkTileBoundsBasis::BSpline, &status));
            auto vkWFull = Upload(context, widths.data(), widths.size() * 4);
            CHECK(!boundsPipe->Begin(vkP, vkWFull, vkS, 5, 3,
                PicktileVkTileBoundsBasis(99), &status));
            auto empty = boundsPipe->Begin(Upload(context, nullptr, 0),
                Upload(context, nullptr, 0), Upload(context, nullptr, 0), 0, 0,
                PicktileVkTileBoundsBasis::BSpline, &status);
            CHECK(bool(empty));
            uint32_t vkStatus = 99;
            CHECK(empty->Poll(&vkStatus) == VK_SUCCESS && vkStatus == 0);
        }
    }

    // ============================ INDICES ============================
    {
        auto runCounts = [&](CurveIndexOptions options, std::vector<uint32_t> counts,
                             char const* label, std::vector<int32_t> const* wantIndices,
                             std::vector<int32_t> const* wantPrims) {
            std::vector<uint32_t> offsets{0};
            for (auto c : counts) offsets.push_back(offsets.back() + c);
            size_t const pointCount = offsets.back();
            // CUDA twin.
            CurveIndexRequirements cudaReq{};
            bool const cudaReqOk = GetCurveIndexRequirements(options, counts.size(),
                                         pointCount, &cudaReq, nullptr) == cudaSuccess;
            uint32_t cudaStatus = UINT32_MAX;
            uint64_t cudaTotal = UINT64_MAX;
            std::vector<int32_t> cudaIndices, cudaPrims;
            if (cudaReqOk) {
                DeviceBuffer<uint32_t> cuIn, cuStatus;
                DeviceBuffer<uint64_t> cuCounts, cuPrefix, cuTotal;
                DeviceBuffer<unsigned char> cuScratch;
                DeviceBuffer<int32_t> cuIndices, cuPrims;
                if (CudaUpload(cuIn, offsets) && cuStatus.reset(1) == cudaSuccess &&
                    cuTotal.reset(1) == cudaSuccess &&
                    cuCounts.reset(offsets.size()) == cudaSuccess &&
                    cuPrefix.reset(offsets.size()) == cudaSuccess &&
                    cuScratch.reset(cudaReq.scanBytes) == cudaSuccess &&
                    cuIndices.reset(cudaReq.maxRecords * cudaReq.indexArity) == cudaSuccess &&
                    cuPrims.reset(cudaReq.maxRecords) == cudaSuccess) {
                    CurveIndexWorkspace workspace{cuCounts.view(), cuPrefix.view(),
                        cuScratch.view()};
                    CurveIndexOutput output{{cuIndices.data(),
                                                cudaReq.maxRecords * cudaReq.indexArity},
                        {cuPrims.data(), cudaReq.maxRecords}, cuTotal.view(),
                        cuStatus.view()};
                    if (BuildCurveIndices(options, counts.size(), pointCount, Read(cuIn),
                            cudaReq, workspace, output, nullptr) == cudaSuccess &&
                        cudaDeviceSynchronize() == cudaSuccess &&
                        cudaMemcpy(&cudaStatus, cuStatus.data(), 4,
                            cudaMemcpyDeviceToHost) == cudaSuccess &&
                        cudaMemcpy(&cudaTotal, cuTotal.data(), 8,
                            cudaMemcpyDeviceToHost) == cudaSuccess) {
                        cudaIndices.resize(cuIndices.size());
                        cudaPrims.resize(cuPrims.size());
                        if (cudaMemcpy(cudaIndices.data(), cuIndices.data(),
                                cudaIndices.size() * 4, cudaMemcpyDeviceToHost) !=
                                cudaSuccess ||
                            cudaMemcpy(cudaPrims.data(), cuPrims.data(),
                                cudaPrims.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
                            cudaStatus = UINT32_MAX;
                        }
                    }
                }
            }
            // Vulkan port.
            PicktileVkIndexRequirements vkReq{};
            bool const vkReqOk = GetPicktileVkIndexRequirements(
                PicktileVkIndexBasis(options.basis), PicktileVkIndexWrap(options.wrap),
                PicktileVkIndexMode(options.mode), counts.size(), pointCount, &vkReq);
            bool vkBegan = false;
            uint32_t vkStatus = UINT32_MAX;
            uint64_t vkTotal = UINT64_MAX;
            uint32_t vkDraw = UINT32_MAX;
            std::vector<int32_t> vkIndices, vkPrims;
            if (vkReqOk) {
                auto vkOff = Upload(context, offsets.data(), offsets.size() * 4);
                auto candidate = indicesPipe->Begin(vkOff, uint32_t(counts.size()),
                    uint32_t(pointCount), 0, PicktileVkIndexBasis(options.basis),
                    PicktileVkIndexWrap(options.wrap), PicktileVkIndexMode(options.mode),
                    vkReq, &status);
                vkBegan = bool(candidate);
                if (vkBegan && provePollIndices(candidate, &vkStatus)) {
                    vkTotal = candidate->recordCount();
                    vkDraw = candidate->drawCount();
                    if (vkStatus == 0) {
                        CHECK(ReadBuffer(context, candidate->indices(), &vkIndices));
                        CHECK(ReadBuffer(context, candidate->primitiveParam(), &vkPrims));
                    }
                }
            }
            // CUDA draw-count twin for the same total/status.
            uint32_t cudaDraw = UINT32_MAX;
            if (cudaReqOk && cudaStatus != UINT32_MAX) {
                DeviceBuffer<uint64_t> cuRec;
                DeviceBuffer<uint32_t> cuSt, cuOut;
                if (CudaUpload(cuRec, std::vector<uint64_t>{cudaTotal}) &&
                    CudaUpload(cuSt, std::vector<uint32_t>{cudaStatus}) &&
                    cuOut.reset(1) == cudaSuccess &&
                    PackCurveDrawCount(Read(cuRec), Read(cuSt), cudaReq.indexArity,
                        cudaReq.maxRecords, cuOut.view(), nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess)
                    CHECK(cudaMemcpy(&cudaDraw, cuOut.data(), 4, cudaMemcpyDeviceToHost) ==
                        cudaSuccess);
            }
            char text[230];
            std::snprintf(text, sizeof(text),
                "indices parity %s (req=%d/%d status=%u/%u total=%llu/%llu draw=%u/%u)",
                label, int(cudaReqOk), int(vkReqOk), cudaStatus, vkStatus,
                (unsigned long long)cudaTotal, (unsigned long long)vkTotal, cudaDraw,
                vkDraw);
            bool ok = cudaReqOk == vkReqOk;
            if (ok && cudaReqOk) {
                ok = vkBegan && cudaStatus == vkStatus && cudaTotal == vkTotal &&
                    cudaDraw == vkDraw;
                if (ok && cudaStatus == 0) {
                    size_t const records = size_t(cudaTotal);
                    size_t const wantIndexWords =
                        size_t(cudaReq.maxRecords) * cudaReq.indexArity;
                    size_t const wantPrimWords = size_t(cudaReq.maxRecords);
                    // Zero-record buffers carry the documented 4-byte
                    // allocation floor (MakeHostBuffer: Vulkan cannot
                    // materialize zero-size buffers), so an empty groom
                    // reads back 1 word; parity is over logical content.
                    size_t const floorWords = size_t(4 / sizeof(int32_t));
                    ok = (vkIndices.size() == wantIndexWords ||
                          (wantIndexWords == 0 && vkIndices.size() == floorWords)) &&
                        (vkPrims.size() == wantPrimWords ||
                         (wantPrimWords == 0 && vkPrims.size() == floorWords));
                    for (size_t i = 0; ok && i < records * cudaReq.indexArity; ++i)
                        ok = vkIndices[i] == cudaIndices[i];
                    for (size_t i = 0; ok && i < records; ++i)
                        ok = vkPrims[i] == cudaPrims[i];
                    if (ok && wantIndices)
                        ok = wantIndices->size() == records * cudaReq.indexArity &&
                            std::equal(wantIndices->begin(), wantIndices->end(),
                                cudaIndices.begin()) &&
                            std::equal(
                                wantIndices->begin(), wantIndices->end(), vkIndices.begin());
                    if (ok && wantPrims)
                        ok = wantPrims->size() == records &&
                            std::equal(wantPrims->begin(), wantPrims->end(),
                                cudaPrims.begin()) &&
                            std::equal(
                                wantPrims->begin(), wantPrims->end(), vkPrims.begin());
                }
            }
            Check(ok, text);
        };
        CurveIndexOptions const cubic{CurveIndexBasis::BSpline, CurveIndexWrap::Pinned,
            CurveIndexMode::Curves};
        std::vector<int32_t> const cubicWant{
            0, 0, 0, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2};
        std::vector<int32_t> const cubicPrims{0, 0, 0, 0, 0};
        runCounts(cubic, {3}, "bspline pinned cubic", &cubicWant, &cubicPrims);
        CurveIndexOptions const linear{CurveIndexBasis::Linear, CurveIndexWrap::Nonperiodic,
            CurveIndexMode::Curves};
        std::vector<int32_t> const linearWant{0, 1, 2, 3, 3, 4};
        std::vector<int32_t> const linearPrims{0, 1, 1};
        runCounts(linear, {2, 3}, "linear nonperiodic", &linearWant, &linearPrims);
        CurveIndexOptions const points{CurveIndexBasis::Linear, CurveIndexWrap::Nonperiodic,
            CurveIndexMode::Points};
        std::vector<int32_t> const pointsWant{0, 1, 2, 3, 4};
        std::vector<int32_t> const pointsPrims{0, 0, 1, 1, 1};
        runCounts(points, {2, 3}, "points mode", &pointsWant, &pointsPrims);
        CurveIndexOptions const segmented{CurveIndexBasis::Linear, CurveIndexWrap::Segmented,
            CurveIndexMode::Curves};
        std::vector<int32_t> const segWant{0, 1, 2, 3};
        std::vector<int32_t> const segPrims{0, 0};
        runCounts(segmented, {4}, "segmented linear", &segWant, &segPrims);
        runCounts(segmented, {3}, "segmented odd -> status 2", nullptr, nullptr);
        runCounts(linear, {}, "empty groom", nullptr, nullptr);
        CurveIndexOptions const hull{CurveIndexBasis::CatmullRom, CurveIndexWrap::Periodic,
            CurveIndexMode::Hull};
        runCounts(hull, {5}, "catmull periodic hull", nullptr, nullptr);
        CurveIndexOptions const bezier{CurveIndexBasis::Bezier, CurveIndexWrap::Nonperiodic,
            CurveIndexMode::Curves};
        runCounts(bezier, {7}, "bezier nonperiodic", nullptr, nullptr);
        // Corrupt terminal offset fails closed on both.
        {
            std::vector<uint32_t> bad{0, 2, 6};
            DeviceBuffer<uint32_t> cuIn, cuStatus;
            DeviceBuffer<uint64_t> cuCounts, cuPrefix, cuTotal;
            DeviceBuffer<unsigned char> cuScratch;
            DeviceBuffer<int32_t> cuIndices, cuPrims;
            CurveIndexRequirements cudaReq{};
            CHECK(GetCurveIndexRequirements(linear, 2, 5, &cudaReq, nullptr) == cudaSuccess);
            CHECK(CudaUpload(cuIn, bad) && cuStatus.reset(1) == cudaSuccess &&
                cuTotal.reset(1) == cudaSuccess &&
                cuCounts.reset(3) == cudaSuccess && cuPrefix.reset(3) == cudaSuccess &&
                cuScratch.reset(cudaReq.scanBytes) == cudaSuccess &&
                cuIndices.reset(cudaReq.maxRecords * cudaReq.indexArity) == cudaSuccess &&
                cuPrims.reset(cudaReq.maxRecords) == cudaSuccess);
            CurveIndexWorkspace workspace{cuCounts.view(), cuPrefix.view(), cuScratch.view()};
            CurveIndexOutput output{{cuIndices.data(), cudaReq.maxRecords * cudaReq.indexArity},
                {cuPrims.data(), cudaReq.maxRecords}, cuTotal.view(), cuStatus.view()};
            CHECK(BuildCurveIndices(linear, 2, 5, Read(cuIn), cudaReq, workspace, output,
                      nullptr) == cudaSuccess &&
                cudaDeviceSynchronize() == cudaSuccess);
            uint32_t cudaStatus = 0;
            uint64_t cudaTotal = 99;
            CHECK(cudaMemcpy(&cudaStatus, cuStatus.data(), 4, cudaMemcpyDeviceToHost) ==
                    cudaSuccess &&
                cudaMemcpy(&cudaTotal, cuTotal.data(), 8, cudaMemcpyDeviceToHost) ==
                    cudaSuccess);
            PicktileVkIndexRequirements vkReq{};
            CHECK(GetPicktileVkIndexRequirements(PicktileVkIndexBasis::Linear,
                PicktileVkIndexWrap::Nonperiodic, PicktileVkIndexMode::Curves, 2, 5, &vkReq));
            auto vkOff = Upload(context, bad.data(), bad.size() * 4);
            auto candidate = indicesPipe->Begin(vkOff, 2, 5, 0, PicktileVkIndexBasis::Linear,
                PicktileVkIndexWrap::Nonperiodic, PicktileVkIndexMode::Curves, vkReq, &status);
            CHECK(bool(candidate));
            if (!candidate) return 1;
            uint32_t vkStatus = 99;
            CHECK(provePollIndices(candidate, &vkStatus));
            CHECK(cudaStatus == 1 && cudaTotal == 0 && vkStatus == 1 &&
                candidate->recordCount() == 0 && candidate->drawCount() == 0 &&
                !candidate->succeeded());
        }
        // Tile-local span with a nonzero global point base.
        {
            std::vector<uint32_t> global{5, 9, 14};
            DeviceBuffer<uint32_t> cuIn, cuStatus;
            DeviceBuffer<uint64_t> cuCounts, cuPrefix, cuTotal;
            DeviceBuffer<unsigned char> cuScratch;
            DeviceBuffer<int32_t> cuIndices, cuPrims;
            CurveIndexRequirements cudaReq{};
            CHECK(GetCurveIndexRequirements(linear, 2, 9, &cudaReq, nullptr) == cudaSuccess);
            CHECK(CudaUpload(cuIn, global) && cuStatus.reset(1) == cudaSuccess &&
                cuTotal.reset(1) == cudaSuccess && cuCounts.reset(3) == cudaSuccess &&
                cuPrefix.reset(3) == cudaSuccess &&
                cuScratch.reset(cudaReq.scanBytes) == cudaSuccess &&
                cuIndices.reset(cudaReq.maxRecords * cudaReq.indexArity) == cudaSuccess &&
                cuPrims.reset(cudaReq.maxRecords) == cudaSuccess);
            CurveIndexWorkspace workspace{cuCounts.view(), cuPrefix.view(), cuScratch.view()};
            CurveIndexOutput output{{cuIndices.data(), cudaReq.maxRecords * cudaReq.indexArity},
                {cuPrims.data(), cudaReq.maxRecords}, cuTotal.view(), cuStatus.view()};
            CHECK(BuildCurveIndices(linear, CurveIndexSpan{Read(cuIn), 2, 9, 5}, cudaReq,
                      workspace, output, nullptr) == cudaSuccess &&
                cudaDeviceSynchronize() == cudaSuccess);
            uint32_t cudaStatus = 99;
            uint64_t cudaTotal = 99;
            std::vector<int32_t> cudaIndices(cudaReq.maxRecords * cudaReq.indexArity),
                cudaPrims(cudaReq.maxRecords);
            CHECK(cudaMemcpy(&cudaStatus, cuStatus.data(), 4, cudaMemcpyDeviceToHost) ==
                    cudaSuccess &&
                cudaMemcpy(&cudaTotal, cuTotal.data(), 8, cudaMemcpyDeviceToHost) ==
                    cudaSuccess &&
                cudaMemcpy(cudaIndices.data(), cuIndices.data(), cudaIndices.size() * 4,
                    cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(cudaPrims.data(), cuPrims.data(), cudaPrims.size() * 4,
                    cudaMemcpyDeviceToHost) == cudaSuccess);
            PicktileVkIndexRequirements vkReq{};
            CHECK(GetPicktileVkIndexRequirements(PicktileVkIndexBasis::Linear,
                PicktileVkIndexWrap::Nonperiodic, PicktileVkIndexMode::Curves, 2, 9, &vkReq));
            auto vkOff = Upload(context, global.data(), global.size() * 4);
            auto candidate = indicesPipe->Begin(vkOff, 2, 9, 5, PicktileVkIndexBasis::Linear,
                PicktileVkIndexWrap::Nonperiodic, PicktileVkIndexMode::Curves, vkReq, &status);
            CHECK(bool(candidate));
            if (!candidate) return 1;
            uint32_t vkStatus = 99;
            CHECK(provePollIndices(candidate, &vkStatus) && vkStatus == 0);
            std::vector<int32_t> vkIndices, vkPrims;
            CHECK(ReadBuffer(context, candidate->indices(), &vkIndices));
            CHECK(ReadBuffer(context, candidate->primitiveParam(), &vkPrims));
            std::vector<int32_t> const want{0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8};
            std::vector<int32_t> const wantPrims{0, 0, 0, 1, 1, 1, 1};
            CHECK(cudaStatus == 0 && cudaTotal == 7 && candidate->recordCount() == 7 &&
                std::equal(want.begin(), want.end(), cudaIndices.begin()) &&
                std::equal(want.begin(), want.end(), vkIndices.begin()) &&
                std::equal(wantPrims.begin(), wantPrims.end(), cudaPrims.begin()) &&
                std::equal(wantPrims.begin(), wantPrims.end(), vkPrims.begin()));
        }
        // Span overflow fails closed on device on both.
        {
            std::vector<uint32_t> offsets{0, 2, 5};
            DeviceBuffer<uint32_t> cuIn, cuStatus;
            DeviceBuffer<uint64_t> cuCounts, cuPrefix, cuTotal;
            DeviceBuffer<unsigned char> cuScratch;
            DeviceBuffer<int32_t> cuIndices, cuPrims;
            CurveIndexRequirements cudaReq{};
            CHECK(GetCurveIndexRequirements(linear, 2, 5, &cudaReq, nullptr) == cudaSuccess);
            CHECK(CudaUpload(cuIn, offsets) && cuStatus.reset(1) == cudaSuccess &&
                cuTotal.reset(1) == cudaSuccess && cuCounts.reset(3) == cudaSuccess &&
                cuPrefix.reset(3) == cudaSuccess &&
                cuScratch.reset(cudaReq.scanBytes) == cudaSuccess &&
                cuIndices.reset(cudaReq.maxRecords * cudaReq.indexArity) == cudaSuccess &&
                cuPrims.reset(cudaReq.maxRecords) == cudaSuccess);
            CurveIndexWorkspace workspace{cuCounts.view(), cuPrefix.view(), cuScratch.view()};
            CurveIndexOutput output{{cuIndices.data(), cudaReq.maxRecords * cudaReq.indexArity},
                {cuPrims.data(), cudaReq.maxRecords}, cuTotal.view(), cuStatus.view()};
            CHECK(BuildCurveIndices(linear, CurveIndexSpan{Read(cuIn), 2, 5, UINT32_MAX},
                      cudaReq, workspace, output, nullptr) == cudaSuccess &&
                cudaDeviceSynchronize() == cudaSuccess);
            uint32_t cudaStatus = 0;
            uint64_t cudaTotal = 99;
            CHECK(cudaMemcpy(&cudaStatus, cuStatus.data(), 4, cudaMemcpyDeviceToHost) ==
                    cudaSuccess &&
                cudaMemcpy(&cudaTotal, cuTotal.data(), 8, cudaMemcpyDeviceToHost) ==
                    cudaSuccess);
            PicktileVkIndexRequirements vkReq{};
            CHECK(GetPicktileVkIndexRequirements(PicktileVkIndexBasis::Linear,
                PicktileVkIndexWrap::Nonperiodic, PicktileVkIndexMode::Curves, 2, 5, &vkReq));
            auto vkOff = Upload(context, offsets.data(), offsets.size() * 4);
            auto candidate = indicesPipe->Begin(vkOff, 2, 5, UINT32_MAX,
                PicktileVkIndexBasis::Linear, PicktileVkIndexWrap::Nonperiodic,
                PicktileVkIndexMode::Curves, vkReq, &status);
            CHECK(bool(candidate));
            if (!candidate) return 1;
            uint32_t vkStatus = 99;
            CHECK(provePollIndices(candidate, &vkStatus));
            CHECK(cudaStatus == 1 && cudaTotal == 0 && vkStatus == 1 &&
                candidate->recordCount() == 0);
        }
        // Host rejections agree.
        {
            PicktileVkIndexRequirements vkReq{};
            CHECK(!GetPicktileVkIndexRequirements(PicktileVkIndexBasis::Bezier,
                PicktileVkIndexWrap::Segmented, PicktileVkIndexMode::Curves, 2, 5, &vkReq));
            CHECK(GetPicktileVkIndexRequirements(PicktileVkIndexBasis::Linear,
                PicktileVkIndexWrap::Nonperiodic, PicktileVkIndexMode::Curves, 2, 5, &vkReq));
            auto vkOff = Upload(context, std::vector<uint32_t>{0, 2, 5}.data(), 12);
            CHECK(!indicesPipe->Begin(vkOff, 2, 5, 0, PicktileVkIndexBasis::Bezier,
                PicktileVkIndexWrap::Segmented, PicktileVkIndexMode::Curves, vkReq, &status));
            auto tampered = vkReq;
            tampered.maxRecords += 1;
            CHECK(!indicesPipe->Begin(vkOff, 2, 5, 0, PicktileVkIndexBasis::Linear,
                PicktileVkIndexWrap::Nonperiodic, PicktileVkIndexMode::Curves, tampered,
                &status));
        }
    }

    // ============================ TOOL SESSION ============================
    {
        std::string reason;
        auto session = PicktileVkToolSession::Create(context, pickCode, footprintCode, &reason);
        CHECK(bool(session));
        if (session) {
            std::vector<float> points{0, 0, 0, 0.2f, 0, 0, -0.5f, 0, 0, 0.5f, 0, 0, 0, 0, 0};
            std::vector<uint32_t> offsets{0, 2, 5};
            std::vector<uint64_t> ids{100, 200};
            PicktileVkChargedGeometry geometry{Upload(context, points.data(), 60),
                Upload(context, offsets.data(), 12), Upload(context, ids.data(), 16), 2, 5};
            PicktileVkResult result{};
            CHECK(session->Pick(geometry, ToVk(IdentityQuery(50, 50, 1)), &result, &reason) &&
                result.hit && result.flatIndex == 0 && result.curve == 0 && result.cv == 0 &&
                result.stableId == 100 && std::fabs(result.distancePx) < 1e-6f);
            PickQuery bad = IdentityQuery(0, 0, 1);
            bad.viewProj[0] = std::numeric_limits<float>::quiet_NaN();
            CHECK(!session->Pick(geometry, ToVk(bad), &result, &reason));
            // The last successful result survives the failed op (CudaPicking parity).
            CHECK(session->result().hit && session->result().flatIndex == 0);
            CHECK(session->Footprint(geometry, ToVk(IdentityQuery(50, 50, 11)), &reason) &&
                session->footprintCount() == 3);
            std::vector<int32_t> indices;
            CHECK(ReadBuffer(context, session->footprint(), &indices));
            indices.resize(session->footprintCount());
            CHECK(indices == std::vector<int32_t>({0, 1, 4}));
            PickQuery badRadius = IdentityQuery(50, 50, -1);
            CHECK(!session->Footprint(geometry, ToVk(badRadius), &reason));
            CHECK(session->footprintCount() == 3 && bool(session->footprint()));
        }
    }

    std::printf("picktileVk parity: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
