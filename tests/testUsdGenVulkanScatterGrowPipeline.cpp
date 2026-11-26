// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/scatterGrowPipeline.h"
#include "vulkanNativeFixture.h"
#include "floatUlpFixture.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan ScatterGrow pipeline check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

// Test-only queue proof. Production Candidate::Poll never waits or reads back
// geometry. This fence is submitted after the candidate on its confined queue.
static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}

template <typename T>
static std::shared_ptr<ChargedBuffer> Upload(std::shared_ptr<DeviceContext> const& context,
    std::shared_ptr<NativeOwner> const& native, std::vector<T> const& values) {
    // Vulkan forbids zero-size buffers; clamp to a minimum bindable size so
    // empty planes (curveCount==0 paths) still bind. The pipeline only ever
    // reads elements up to its documented count, so padding is invisible.
    VkDeviceSize bytes = values.empty() ? 4 : static_cast<VkDeviceSize>(values.size() * sizeof(T));
    VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto buffer = ChargedBuffer::Create(context, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Active);
    if (!buffer) return buffer;
    void* mapped = nullptr;
    if (vkMapMemory(native->device, buffer->memory(), 0, bytes, 0, &mapped) != VK_SUCCESS)
        return std::shared_ptr<ChargedBuffer>{};
    if (!values.empty()) std::memcpy(mapped, values.data(), values.size() * sizeof(T));
    vkUnmapMemory(native->device, buffer->memory());
    return buffer;
}

template <typename T>
static bool Download(std::shared_ptr<NativeOwner> const& native,
    std::shared_ptr<const ChargedBuffer> const& source, std::vector<T>* values) {
    VkDeviceSize bytes = values->size() * sizeof(T);
    VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto staging = ChargedBuffer::Create(source->context(), bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!staging) return false;
    VkCommandBufferAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = native->commands; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(native->device, &ai, &command) != VK_SUCCESS) return false;
    VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) {
        vkFreeCommandBuffers(native->device, native->commands, 1, &command);
        return false;
    }
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(command, source->buffer(), staging->buffer(), 1, &region);
    VkMemoryBarrier barrier{}; barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(command) != VK_SUCCESS ||
        vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(native->device, native->commands, 1, &command);
        return false;
    }
    struct Keep { std::shared_ptr<NativeOwner> native; std::shared_ptr<const ChargedBuffer> source; };
    auto keep = std::make_shared<Keep>(Keep{native, source});
    if (staging->MarkSubmitted(native->fence, keep) != VK_SUCCESS) {
        vkFreeCommandBuffers(native->device, native->commands, 1, &command);
        return false;
    }
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    if (vkQueueSubmit(native->queue, 1, &submit, native->fence) != VK_SUCCESS ||
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) != VK_SUCCESS ||
        staging->PollComplete() != VK_SUCCESS) {
        vkFreeCommandBuffers(native->device, native->commands, 1, &command);
        return false;
    }
    void* mapped = nullptr;
    if (vkMapMemory(native->device, staging->memory(), 0, bytes, 0, &mapped) != VK_SUCCESS) {
        vkFreeCommandBuffers(native->device, native->commands, 1, &command);
        return false;
    }
    std::memcpy(values->data(), mapped, bytes); vkUnmapMemory(native->device, staging->memory());
    vkFreeCommandBuffers(native->device, native->commands, 1, &command);
    return true;
}

// Deterministic roots with a varied basis (rootT/rootB/rootN) per root.
struct Roots {
    uint32_t curves;
    std::vector<float> positions;
    std::vector<uint64_t> stableIds;
    std::vector<int32_t> rootPrim;
    std::vector<float> rootUV;
    std::vector<float> rootT, rootB, rootN;
    Roots(uint32_t c) : curves(c) {
        stableIds.resize(c);
        rootPrim.resize(c);
        rootUV.resize(2 * c);
        rootT.resize(3 * c); rootB.resize(3 * c); rootN.resize(3 * c);
        for (uint32_t i = 0; i < c; ++i) {
            positions.push_back(float(i) * 10.0f + 1.0f);
            positions.push_back(float(i) * 2.0f + 0.5f);
            positions.push_back(0.25f + float(i));
            stableIds[i] = uint64_t(i + 1000);
            rootPrim[i] = int32_t(i * 3 + 1);
            float u = float(i) / float(c ? c : 1), v = 0.25f;
            rootUV[2 * i] = u; rootUV[2 * i + 1] = v;
            // Distinct unit frames per root.
            rootT[3 * i] = 1.0f; rootT[3 * i + 1] = 0.0f; rootT[3 * i + 2] = 0.0f;
            rootB[3 * i] = 0.0f; rootB[3 * i + 1] = 0.0f; rootB[3 * i + 2] = 1.0f;
            rootN[3 * i] = 0.0f; rootN[3 * i + 1] = 1.0f; rootN[3 * i + 2] = 0.0f;
        }
    }
};

int main(int argc, char** argv) {
    bool quarantine = argc == 3 && std::string(argv[2]) == "--quarantine";
    CHECK(argc == 2 || quarantine);
    std::ifstream shader(argv[1], std::ios::binary);
    CHECK(shader);
    std::vector<char> bytes((std::istreambuf_iterator<char>(shader)), {});
    CHECK(!bytes.empty() && bytes.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> code(bytes.size() / sizeof(uint32_t));
    std::memcpy(code.data(), bytes.data(), bytes.size());
    bool unavailable = false;
    auto native = CreateNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    {   // scatter grow kernel needs 20 storage-buffer bindings; skip weak devices (77).
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(native->physical, &props);
        if (props.limits.maxPerStageDescriptorStorageBuffers < 20 ||
            props.limits.maxDescriptorSetStorageBuffers < 20) return 77;
    }
    std::weak_ptr<NativeOwner> weakNative = native;
    DeviceContext::CreateInfo info;
    info.instance = native->instance; info.physicalDevice = native->physical;
    info.device = native->device; info.computeQueue = native->queue;
    info.computeQueueFamily = native->family; info.physicalIndex = native->physicalIndex;
    info.resourceDeviceId = 7004; info.nativeLifetime = native;
    info.resources = {size_t{16} << 20, 0};
    auto context = DeviceContext::Create(info);
    CHECK(context); info.nativeLifetime.reset();
    auto pool = context->resources(); auto baseline = pool->Snapshot();
    VkResult status = VK_SUCCESS;
    CHECK(!ScatterGrowPipeline::Create(context, {0x07230203u}, &status));
    CHECK(status == VK_ERROR_INITIALIZATION_FAILED);
    auto pipeline = ScatterGrowPipeline::Create(context, code, &status);
    CHECK(pipeline && status == VK_SUCCESS);

    // ------------------------------------------------------------------ (a)
    uint32_t const curves = 4, cv = 8;
    Roots roots(curves);
    ScatterGrowPipeline::Controls controls;
    controls.cvCount = cv; controls.seed = 7; controls.length = 1.5;
    controls.randomLo = 0.5; controls.randomHi = 2.0;  // per-root random targets
    controls.lift = 0.0f; controls.fallbackWidth = 0.02f;
    controls.direction = ScatterGrowPipeline::Direction::RootNormal;
    auto targets = ScatterGrowPipeline::BuildTargets(roots.stableIds, controls);
    CHECK(targets.size() == curves);
    // BuildTargets must be deterministic and per-root distinct.
    CHECK(targets[0] != targets[1]);
    {
        auto before = pool->Snapshot();
        auto in = ScatterGrowPipeline::Input{};
        in.positions = Upload(context, native, roots.positions);
        in.stableIds = Upload(context, native, roots.stableIds);
        in.rootPrim = Upload(context, native, roots.rootPrim);
        in.rootUV = Upload(context, native, roots.rootUV);
        in.rootT = Upload(context, native, roots.rootT);
        in.rootB = Upload(context, native, roots.rootB);
        in.rootN = Upload(context, native, roots.rootN);
        in.targets = Upload(context, native, targets);
        CHECK(in.positions && in.stableIds && in.rootPrim && in.rootUV &&
              in.rootT && in.rootB && in.rootN && in.targets);
        auto cand = pipeline->Begin(native, in, curves, controls, &status);
        CHECK(cand && status == VK_SUCCESS && !cand->output() &&
              cand->curveCount() == curves && cand->pointCount() == curves * cv);
        CHECK(Prove(native));
        ScatterGrowPipeline::Candidate::Semantic semantic;
        CHECK(cand->Poll(&semantic) == VK_SUCCESS && semantic == ScatterGrowPipeline::Candidate::Semantic::Ok);

        // Oracle: exact same host inputs.
        ScatterGrowPipeline::Outputs oracle;
        bool ok = ScatterGrowPipeline::BuildCpu(roots.positions, roots.stableIds,
            roots.rootPrim, roots.rootUV, roots.rootT, roots.rootB, roots.rootN,
            targets, controls, &oracle);
        CHECK(ok);

        uint32_t outPts = curves * cv;
        std::vector<float> got; std::vector<uint32_t> gotOff; std::vector<uint64_t> gotId;
        std::vector<int32_t> gotPrim; std::vector<float> gotUv, gotT, gotB, gotN;
        got.resize(outPts * 3); CHECK(Download(native, cand->output(), &got));
        // Float planes compare within 4 ULPs: the GLSL kernel fuses mul+add
        // where the host oracle BuildCpu does not.
        CHECK(FloatBytesWithinUlps(got.data(), oracle.points.data(),
                                   got.size() * sizeof(float), 4u));
        got.resize(outPts * 3); CHECK(Download(native, cand->rest(), &got));
        CHECK(FloatBytesWithinUlps(got.data(), oracle.rest.data(),
                                   got.size() * sizeof(float), 4u));
        got.resize(outPts); CHECK(Download(native, cand->widths(), &got));
        CHECK(FloatBytesWithinUlps(got.data(), oracle.widths.data(),
                                   got.size() * sizeof(float), 4u));
        got.resize(outPts); CHECK(Download(native, cand->hairT(), &got));
        CHECK(std::memcmp(got.data(), oracle.hairT.data(), got.size() * sizeof(float)) == 0);
        gotOff.resize(curves + 1); CHECK(Download(native, cand->offsets(), &gotOff));
        CHECK(gotOff == oracle.offsets);
        gotId.resize(curves); CHECK(Download(native, cand->ids(), &gotId));
        CHECK(gotId == oracle.ids);
        gotPrim.resize(curves); CHECK(Download(native, cand->rootPrim(), &gotPrim));
        CHECK(gotPrim == oracle.rootPrim);
        gotUv.resize(curves * 2); CHECK(Download(native, cand->rootUV(), &gotUv));
        CHECK(FloatBytesWithinUlps(gotUv.data(), oracle.rootUV.data(),
                                   gotUv.size() * sizeof(float), 4u));
        gotT.resize(curves * 3); CHECK(Download(native, cand->rootT(), &gotT));
        CHECK(FloatBytesWithinUlps(gotT.data(), oracle.rootT.data(),
                                   gotT.size() * sizeof(float), 4u));
        gotB.resize(curves * 3); CHECK(Download(native, cand->rootB(), &gotB));
        CHECK(FloatBytesWithinUlps(gotB.data(), oracle.rootB.data(),
                                   gotB.size() * sizeof(float), 4u));
        gotN.resize(curves * 3); CHECK(Download(native, cand->rootN(), &gotN));
        CHECK(FloatBytesWithinUlps(gotN.data(), oracle.rootN.data(),
                                   gotN.size() * sizeof(float), 4u));
        cand.reset();
        // Structural hairT must match CUDA's rounded quotient exactly for
        // every admitted output count, even when geometry allows a few ULPs.
        // The one pipeline and uploaded roots are reused across all counts.
        // Computing the small integer ratio in double before its binary32
        // conversion supplies an oracle independent of the shader algorithm.
        auto const inputCharge = pool->Snapshot();
        for (uint32_t outputCvs = 2; outputCvs <= 64; ++outputCvs) {
            auto sweep = controls; sweep.cvCount = outputCvs;
            auto candidate = pipeline->Begin(native, in, curves, sweep, &status);
            CHECK(candidate && status == VK_SUCCESS && Prove(native));
            ScatterGrowPipeline::Candidate::Semantic sweepSemantic;
            CHECK(candidate->Poll(&sweepSemantic) == VK_SUCCESS &&
                sweepSemantic == ScatterGrowPipeline::Candidate::Semantic::Ok);
            CHECK(candidate->pointCount() == curves * outputCvs);
            std::vector<uint32_t> hairTBits(curves * outputCvs);
            CHECK(Download(native, candidate->hairT(), &hairTBits));
            for (uint32_t i = 0; i < outputCvs; ++i) {
                float const expected = float(double(i) / double(outputCvs - 1));
                uint32_t expectedBits;
                std::memcpy(&expectedBits, &expected, sizeof(expectedBits));
                for (uint32_t curve = 0; curve < curves; ++curve) {
                    uint32_t const actualBits = hairTBits[curve * outputCvs + i];
                    if (actualBits != expectedBits)
                        std::fprintf(stderr, "ScatterGrow hairT cvCount=%u curve=%u cv=%u: 0x%08x != 0x%08x\n",
                            outputCvs, curve, i, actualBits, expectedBits);
                    CHECK(actualBits == expectedBits);
                }
            }
            candidate.reset();
            CHECK(pool->Snapshot().usedBytes == inputCharge.usedBytes &&
                pool->Snapshot().byKind == inputCharge.byKind);
        }
        in = ScatterGrowPipeline::Input{};
        CHECK(pool->Snapshot().usedBytes == before.usedBytes && pool->Snapshot().byKind == before.byKind);
    }

    // ------------------------------------------------------------- (b)
    // RootTangent direction: per-root direction must differ from RootNormal.
    {
        auto inTangent = controls; inTangent.direction = ScatterGrowPipeline::Direction::RootTangent;
        auto targetsT = ScatterGrowPipeline::BuildTargets(roots.stableIds, inTangent);
        auto inNorm = controls; // RootNormal

        auto inTan = ScatterGrowPipeline::Input{};
        inTan.positions = Upload(context, native, roots.positions);
        inTan.stableIds = Upload(context, native, roots.stableIds);
        inTan.rootPrim = Upload(context, native, roots.rootPrim);
        inTan.rootUV = Upload(context, native, roots.rootUV);
        inTan.rootT = Upload(context, native, roots.rootT);
        inTan.rootB = Upload(context, native, roots.rootB);
        inTan.rootN = Upload(context, native, roots.rootN);
        inTan.targets = Upload(context, native, targetsT);
        auto candTan = pipeline->Begin(native, inTan, curves, inTangent, &status);
        CHECK(candTan && status == VK_SUCCESS && Prove(native));
        ScatterGrowPipeline::Candidate::Semantic semTan;
        CHECK(candTan->Poll(&semTan) == VK_SUCCESS && semTan == ScatterGrowPipeline::Candidate::Semantic::Ok);

        auto inNormal = ScatterGrowPipeline::Input{};
        inNormal.positions = Upload(context, native, roots.positions);
        inNormal.stableIds = Upload(context, native, roots.stableIds);
        inNormal.rootPrim = Upload(context, native, roots.rootPrim);
        inNormal.rootUV = Upload(context, native, roots.rootUV);
        inNormal.rootT = Upload(context, native, roots.rootT);
        inNormal.rootB = Upload(context, native, roots.rootB);
        inNormal.rootN = Upload(context, native, roots.rootN);
        inNormal.targets = Upload(context, native, targetsT);
        auto candNorm = pipeline->Begin(native, inNormal, curves, inNorm, &status);
        CHECK(candNorm && status == VK_SUCCESS && Prove(native));
        ScatterGrowPipeline::Candidate::Semantic semNorm;
        CHECK(candNorm->Poll(&semNorm) == VK_SUCCESS && semNorm == ScatterGrowPipeline::Candidate::Semantic::Ok);

        std::vector<float> tanPts(curves * cv * 3), normPts(curves * cv * 3);
        CHECK(Download(native, candTan->output(), &tanPts) &&
              Download(native, candNorm->output(), &normPts));
        // rootT=(1,0,0), rootN=(0,1,0): tangential grow differs from normal grow.
        float maxDelta = 0.0f;
        for (size_t i = 0; i < tanPts.size(); ++i)
            maxDelta = std::max(maxDelta, std::fabs(tanPts[i] - normPts[i]));
        CHECK(maxDelta > 0.001f);
        // And the tangential output must match its own CPU oracle.
        ScatterGrowPipeline::Outputs oracleT;
        CHECK(ScatterGrowPipeline::BuildCpu(roots.positions, roots.stableIds,
            roots.rootPrim, roots.rootUV, roots.rootT, roots.rootB, roots.rootN,
            targetsT, inTangent, &oracleT));
        CHECK(FloatBytesWithinUlps(tanPts.data(), oracleT.points.data(),
                                   tanPts.size() * sizeof(float), 4u));
        candTan.reset(); candNorm.reset();
        inTan = ScatterGrowPipeline::Input{}; inNormal = ScatterGrowPipeline::Input{};
    }

    // ------------------------------------------------------------- (c)
    // Non-finite root position -> device NonFinite status.
    {
        auto badRoots = roots;
        badRoots.positions[0] = std::numeric_limits<float>::quiet_NaN();
        auto in = ScatterGrowPipeline::Input{};
        in.positions = Upload(context, native, badRoots.positions);
        in.stableIds = Upload(context, native, badRoots.stableIds);
        in.rootPrim = Upload(context, native, badRoots.rootPrim);
        in.rootUV = Upload(context, native, badRoots.rootUV);
        in.rootT = Upload(context, native, badRoots.rootT);
        in.rootB = Upload(context, native, badRoots.rootB);
        in.rootN = Upload(context, native, badRoots.rootN);
        in.targets = Upload(context, native, targets);
        auto cand = pipeline->Begin(native, in, curves, controls, &status);
        CHECK(cand && status == VK_SUCCESS && Prove(native));
        ScatterGrowPipeline::Candidate::Semantic semantic;
        CHECK(cand->Poll(&semantic) == VK_SUCCESS &&
              semantic == ScatterGrowPipeline::Candidate::Semantic::NonFinite);
        CHECK(!cand->output());
        cand.reset(); in = ScatterGrowPipeline::Input{};
    }

    // ------------------------------------------------------------- (d)
    // Host control rejections: bad cvCount / negative length / bad lift.
    {
        auto in = ScatterGrowPipeline::Input{};
        in.positions = Upload(context, native, roots.positions);
        in.stableIds = Upload(context, native, roots.stableIds);
        in.rootPrim = Upload(context, native, roots.rootPrim);
        in.rootUV = Upload(context, native, roots.rootUV);
        in.rootT = Upload(context, native, roots.rootT);
        in.rootB = Upload(context, native, roots.rootB);
        in.rootN = Upload(context, native, roots.rootN);
        in.targets = Upload(context, native, targets);

        auto badCv = controls; badCv.cvCount = 1;
        auto r1 = pipeline->Begin(native, in, curves, badCv, &status);
        CHECK(!r1 && status == VK_ERROR_INITIALIZATION_FAILED);

        auto negLen = controls; negLen.length = -1.0;
        auto r2 = pipeline->Begin(native, in, curves, negLen, &status);
        CHECK(!r2 && status == VK_ERROR_INITIALIZATION_FAILED);

        auto badLift = controls; badLift.lift = 120.0f;
        auto r3 = pipeline->Begin(native, in, curves, badLift, &status);
        CHECK(!r3 && status == VK_ERROR_INITIALIZATION_FAILED);

        auto badAz = controls; badAz.azimuth = 361.0f;
        auto r4 = pipeline->Begin(native, in, curves, badAz, &status);
        CHECK(!r4 && status == VK_ERROR_INITIALIZATION_FAILED);

        auto badAzNeg = controls; badAzNeg.azimuth = -360.5f;
        auto r5 = pipeline->Begin(native, in, curves, badAzNeg, &status);
        CHECK(!r5 && status == VK_ERROR_INITIALIZATION_FAILED);

        auto badAzRnd = controls; badAzRnd.azimuthRandom = 1.5f;
        auto r6 = pipeline->Begin(native, in, curves, badAzRnd, &status);
        CHECK(!r6 && status == VK_ERROR_INITIALIZATION_FAILED);

        auto nanAz = controls; nanAz.azimuth = std::numeric_limits<float>::quiet_NaN();
        auto r7 = pipeline->Begin(native, in, curves, nanAz, &status);
        CHECK(!r7 && status == VK_ERROR_INITIALIZATION_FAILED);
        in = ScatterGrowPipeline::Input{};
    }

    // ------------------------------------------------------------- (e)
    // Span size: a targets buffer smaller than curveCount floats must reject
    // on the host before any allocation.
    {
        auto before = pool->Snapshot();
        auto shortTargets = targets; shortTargets.pop_back();  // curves-1 entries
        auto in = ScatterGrowPipeline::Input{};
        in.positions = Upload(context, native, roots.positions);
        in.stableIds = Upload(context, native, roots.stableIds);
        in.rootPrim = Upload(context, native, roots.rootPrim);
        in.rootUV = Upload(context, native, roots.rootUV);
        in.rootT = Upload(context, native, roots.rootT);
        in.rootB = Upload(context, native, roots.rootB);
        in.rootN = Upload(context, native, roots.rootN);
        in.targets = Upload(context, native, shortTargets);
        auto spanBroken = pipeline->Begin(native, in, curves, controls, &status);
        CHECK(!spanBroken && status == VK_ERROR_INITIALIZATION_FAILED);
        in = ScatterGrowPipeline::Input{};
        CHECK(pool->Snapshot().usedBytes == before.usedBytes && pool->Snapshot().byKind == before.byKind);
    }

    // ------------------------------------------------------------- (f)
    // Azimuth: literal rotation plus the per-root random draw must match the
    // CPU oracle, and must move RootTangent output (rotation about root N).
    {
        auto az = controls; az.direction = ScatterGrowPipeline::Direction::RootTangent;
        az.lift = 30.0f; az.azimuth = 45.0f; az.azimuthRandom = 1.0f;
        auto azTargets = ScatterGrowPipeline::BuildTargets(roots.stableIds, az);
        auto in = ScatterGrowPipeline::Input{};
        in.positions = Upload(context, native, roots.positions);
        in.stableIds = Upload(context, native, roots.stableIds);
        in.rootPrim = Upload(context, native, roots.rootPrim);
        in.rootUV = Upload(context, native, roots.rootUV);
        in.rootT = Upload(context, native, roots.rootT);
        in.rootB = Upload(context, native, roots.rootB);
        in.rootN = Upload(context, native, roots.rootN);
        in.targets = Upload(context, native, azTargets);
        CHECK(in.positions && in.stableIds && in.rootPrim && in.rootUV &&
              in.rootT && in.rootB && in.rootN && in.targets);
        auto cand = pipeline->Begin(native, in, curves, az, &status);
        CHECK(cand && status == VK_SUCCESS && Prove(native));
        ScatterGrowPipeline::Candidate::Semantic semAz;
        CHECK(cand->Poll(&semAz) == VK_SUCCESS && semAz == ScatterGrowPipeline::Candidate::Semantic::Ok);
        ScatterGrowPipeline::Outputs oracleAz;
        CHECK(ScatterGrowPipeline::BuildCpu(roots.positions, roots.stableIds,
            roots.rootPrim, roots.rootUV, roots.rootT, roots.rootB, roots.rootN,
            azTargets, az, &oracleAz));
        std::vector<float> got(curves * cv * 3);
        CHECK(Download(native, cand->output(), &got));
        // 16 ULP (not the single-rotation 4): the azimuth path chains two
        // Rodrigues rotations (lift, then azimuth about root N) after a
        // draw-combined angle, so libm-vs-GLSL trig plus mul+add fusion
        // noise in the unit direction (~2e-7 absolute) is amplified by the
        // per-curve target length (up to ~3 here). Measured worst case on
        // NVIDIA GB10 is 9 ULP (1 over the old budget, 5.4e-7 absolute at
        // |x|~0.55); the shader evaluates the CUDA formula op-for-op, and
        // disabling driver contraction changes no output bit, so the
        // residual is inherent cross-implementation trig noise, not a logic
        // error. FP contract: the CUDA twin (testUsdGenCudaScatterGrow.cu)
        // accepts Near 2e-5 absolute (~670 ULP at this magnitude), so 16 ULP
        // stays ~40x tighter while leaving headroom across drivers. A wrong
        // draw, salt, or rotation order would deviate by orders of
        // magnitude more (O(0.01+), i.e. 300k+ ULP).
        CHECK(FloatBytesWithinUlps(got.data(), oracleAz.points.data(),
                                   got.size() * sizeof(float), 16u));
        // Same inputs without azimuth must produce visibly different points.
        auto noAz = az; noAz.azimuth = 0.0f; noAz.azimuthRandom = 0.0f;
        auto candPlain = pipeline->Begin(native, in, curves, noAz, &status);
        CHECK(candPlain && status == VK_SUCCESS && Prove(native));
        ScatterGrowPipeline::Candidate::Semantic semPlain;
        CHECK(candPlain->Poll(&semPlain) == VK_SUCCESS &&
              semPlain == ScatterGrowPipeline::Candidate::Semantic::Ok);
        std::vector<float> plain(curves * cv * 3);
        CHECK(Download(native, candPlain->output(), &plain));
        float maxDelta = 0.0f;
        for (size_t i = 0; i < got.size(); ++i)
            maxDelta = std::max(maxDelta, std::fabs(got[i] - plain[i]));
        CHECK(maxDelta > 0.001f);
        cand.reset(); candPlain.reset(); in = ScatterGrowPipeline::Input{};
    }

    // ------------------------------------------------- quarantine path
    if (quarantine) {
        auto in = ScatterGrowPipeline::Input{};
        in.positions = Upload(context, native, roots.positions);
        in.stableIds = Upload(context, native, roots.stableIds);
        in.rootPrim = Upload(context, native, roots.rootPrim);
        in.rootUV = Upload(context, native, roots.rootUV);
        in.rootT = Upload(context, native, roots.rootT);
        in.rootB = Upload(context, native, roots.rootB);
        in.rootN = Upload(context, native, roots.rootN);
        in.targets = Upload(context, native, targets);
        auto cand = pipeline->Begin(native, in, curves, controls, &status);
        CHECK(cand && status == VK_SUCCESS);
        auto charged = pool->Snapshot();
        CHECK(charged.usedBytes > baseline.usedBytes);
        std::weak_ptr<const ChargedBuffer> weakIn = in.positions;
        CHECK(!cand->output()); // hidden pre-proof; quarantine must never publish
        cand->Quarantine(); cand.reset(); pipeline.reset(); context.reset();
        CHECK(Prove(native)); native.reset();
        CHECK(!weakNative.expired());
        CHECK(!weakIn.expired()); // quarantine pins borrowed inputs forever
        CHECK(pool->Snapshot().usedBytes == charged.usedBytes && pool->Snapshot().byKind == charged.byKind);
        return 0;
    }

    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes &&
          pool->Snapshot().byKind == baseline.byKind);
    pool.reset();  // the pool handle itself pins the context
    pipeline.reset(); context.reset(); native.reset();
    CHECK(weakNative.expired());
    return 0;
}
