// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/sourceWidthJob.h"
#include "vulkanNativeFixture.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, \
    "Vulkan source preparation lifecycle check failed: %s (%d)\n", #x, __LINE__); std::abort(); } } while (false)

namespace {

constexpr auto kTimeout = std::chrono::seconds(10);

VulkanSourceGenerationCreateInfo Source(std::shared_ptr<DeviceContext> context) {
    VulkanSourceGenerationCreateInfo info;
    info.context = std::move(context);
    auto& source = info.source;
    source.totalCurves = 1;
    source.totalCvs = 2;
    source.topologyVersion = 3;
    source.valueVersion = 4;
    source.px = {0.f, 1.f};
    source.py = {0.f, 1.f};
    source.pz = {0.f, 1.f};
    source.curveId = {7};
    source.cvOffsets = {0, 2};
    source.width = {1.f, 2.f};
    return info;
}

struct Gate final {
    std::shared_ptr<UsdGenExecutionTaskGraph::TaskCompletion> completion;
    std::atomic<bool> released{false};
    std::atomic<bool> completionError{false};

    void Release() noexcept {
        if (released.exchange(true, std::memory_order_acq_rel)) return;
        auto held = std::atomic_exchange_explicit(&completion,
            std::shared_ptr<UsdGenExecutionTaskGraph::TaskCompletion>{}, std::memory_order_acq_rel);
        if (held) (*held)([] {}, {}); // A final task must publish a nonempty action.
    }
    ~Gate() { Release(); }
};

bool StartBlocker(UsdGenExecutionPipeline& owner,
                  std::shared_ptr<UsdGenExecutionTaskGraph> const& preparer,
                  std::shared_ptr<Gate> const& gate) {
    bool accepted = false;
    owner.Await([&](auto started) {
        auto signal = std::make_shared<std::function<void()>>(std::move(started));
        auto signalled = std::make_shared<std::atomic<bool>>(false);
        UsdGenExecutionTaskGraph::Job blocker;
        blocker.tasks.resize(1);
        blocker.tasks[0].run = [gate, signal, signalled](
            UsdGenExecutionTaskGraph::Cancellation const&,
            UsdGenExecutionTaskGraph::TaskCompletion done) mutable {
            auto held = std::make_shared<UsdGenExecutionTaskGraph::TaskCompletion>(std::move(done));
            std::atomic_store_explicit(&gate->completion, std::move(held), std::memory_order_release);
            if (!signalled->exchange(true, std::memory_order_acq_rel)) (*signal)();
            // The worker is now deliberately occupied, never blocked.
        };
        blocker.completion = [gate](auto publish, std::exception_ptr error) {
            if (error || !publish) gate->completionError.store(true, std::memory_order_release);
            else publish();
        };
        accepted = preparer->Submit(std::move(blocker));
        if (!accepted && !signalled->exchange(true, std::memory_order_acq_rel)) (*signal)();
    });
    return accepted && std::atomic_load_explicit(&gate->completion, std::memory_order_acquire) != nullptr;
}

bool CopyCancellation(UsdGenExecutionPipeline& owner,
                      UsdGenExecutionPipeline::Cancellation* cancellation) {
    bool accepted = false;
    owner.Await([&](auto done) {
        auto signal = std::make_shared<std::function<void()>>(std::move(done));
        auto signalled = std::make_shared<std::atomic<bool>>(false);
        accepted = owner.Submit([cancellation, signal, signalled](
            UsdGenExecutionPipeline::Cancellation const& cancel) mutable {
            *cancellation = cancel;
            return UsdGenExecutionPipeline::Publish([signal, signalled]() mutable {
                if (!signalled->exchange(true, std::memory_order_acq_rel)) (*signal)();
            });
        }) != 0;
        if (!accepted && !signalled->exchange(true, std::memory_order_acq_rel)) (*signal)();
    });
    return accepted && cancellation->epoch != 0 && !cancellation->Superseded();
}

} // namespace

int main(int argc, char** argv) {
    CHECK(argc == 2 || argc == 3);
    const std::string mode = argc == 3 ? argv[2] : "";
    CHECK(mode.empty() || mode == "--reject" || mode == "--owner-unavailable");

    std::ifstream shader(argv[1], std::ios::binary);
    CHECK(shader);
    std::vector<char> raw((std::istreambuf_iterator<char>(shader)), {});
    CHECK(!raw.empty() && raw.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> code(raw.size() / sizeof(uint32_t));
    std::memcpy(code.data(), raw.data(), raw.size());

    bool unavailable = false;
    auto native = CreateTimelineNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    DeviceContext::CreateInfo contextInfo; contextInfo.timelineSemaphoreEnabled = true;
    contextInfo.instance = native->instance;
    contextInfo.physicalDevice = native->physical;
    contextInfo.device = native->device;
    contextInfo.computeQueue = native->queue;
    contextInfo.computeQueueFamily = native->family;
    contextInfo.physicalIndex = native->physicalIndex;
    contextInfo.resourceDeviceId = 7311;
    contextInfo.nativeLifetime = native; contextInfo.timelineSemaphoreEnabled = true;
    contextInfo.resources = {size_t{8} << 20, 0};
    VkResult status = VK_ERROR_UNKNOWN;
    auto context = DeviceContext::Create(contextInfo, &status);
    CHECK(context && status == VK_SUCCESS);
    auto widthPipeline = WidthPipeline::Create(context, code, &status);
    CHECK(widthPipeline && status == VK_SUCCESS);
    auto pool = context->resources();
    const auto baseline = pool->Snapshot().usedBytes;
    std::weak_ptr<DeviceContext> weakContext = context;
    std::weak_ptr<NativeOwner> weakNative = native;

    UsdGenExecutionRuntime runtime{2};
    UsdGenExecutionTaskGraph::Limits limits;
    limits.maxJobs = mode == "--reject" ? 1 : 2;
    limits.maxTasks = 4;
    limits.maxDependencyEdges = 4;
    limits.maxActiveTasks = 1;
    const std::string graphKey = "vulkan-source-preparation-lifecycle-" +
        (mode.empty() ? std::string("cancel") : mode.substr(2));
    auto preparer = UsdGenExecutionTaskGraph::GetOrCreate(runtime, graphKey, 7311, limits);
    CHECK(preparer);
    auto gate = std::make_shared<Gate>();
    UsdGenExecutionPipeline owner(runtime, 8, 8);
    CHECK(StartBlocker(owner, preparer, gate));

    // A real pipeline cancellation is propagated into the CPU task rather
    // than relying on the job's test-only suppression entry point.
    UsdGenExecutionPipeline::Cancellation cancellation;
    CHECK(CopyCancellation(owner, &cancellation));
    auto request = std::make_shared<unsigned>(1);
    std::weak_ptr<unsigned> weakRequest = request;
    VulkanSourceWidthJob::CreateInfo info;
    info.source = Source(context);
    info.widthPipeline = widthPipeline;
    info.width = 3.f;
    info.replace = 1;
    info.valueVersion = 5;
    info.requestLifetime = request;
    info.preparer = preparer;
    auto job = VulkanSourceWidthJob::Create(std::move(info));
    CHECK(job);
    request.reset();

    std::promise<void> completed;
    auto done = completed.get_future();
    std::atomic<unsigned> callbacks{0};
    std::atomic<bool> callbackOnOwner{false};
    std::atomic<VulkanSourceWidthJob::State> finalState{VulkanSourceWidthJob::State::Created};
    std::atomic<VkResult> finalStatus{VK_ERROR_UNKNOWN};
    CHECK(job->Start(owner, cancellation, [&](auto result, auto state, VkResult resultStatus) {
        callbackOnOwner.store(owner.IsExecutingOwner(), std::memory_order_release);
        finalState.store(state, std::memory_order_release);
        finalStatus.store(resultStatus, std::memory_order_release);
        callbacks.fetch_add(1, std::memory_order_acq_rel);
        if (result) std::abort();
        try { completed.set_value(); } catch (...) {}
    }));

    if (mode == "--reject") {
        // The blocker owns the graph's sole job admission credit.  The job's
        // owner-side submission must fail before it can begin CPU preparation.
        CHECK(done.wait_for(kTimeout) == std::future_status::ready);
        gate->Release();
        preparer->Drain();
        owner.Drain();
        CHECK(!gate->completionError.load(std::memory_order_acquire));
        CHECK(callbacks == 1 && callbackOnOwner.load(std::memory_order_acquire));
        CHECK(finalState.load(std::memory_order_acquire) == VulkanSourceWidthJob::State::Failed);
        CHECK(finalStatus.load(std::memory_order_acquire) != VK_SUCCESS);
        CHECK(pool->Snapshot().usedBytes == baseline && owner.OutstandingCommands() == 0);
        CHECK(preparer->GetAdmissionUsage().jobs == 0);
        job.reset();
        CHECK(weakRequest.expired());
        return 0;
    }

    // Run the owner ingress so preparation is queued behind the held worker.
    owner.Drain();
    CHECK(job->state() == VulkanSourceWidthJob::State::Preparing);
    CHECK(pool->Snapshot().usedBytes == baseline);

    if (mode == "--owner-unavailable") {
        owner.Shutdown();
        job.reset();
        gate->Release();
        CHECK(done.wait_for(kTimeout) == std::future_status::ready);
        preparer->Drain();
        CHECK(!gate->completionError.load(std::memory_order_acquire));
        CHECK(callbacks == 1 && !callbackOnOwner.load(std::memory_order_acquire));
        CHECK(finalState.load(std::memory_order_acquire) == VulkanSourceWidthJob::State::LostProof);
        CHECK(finalStatus.load(std::memory_order_acquire) == VK_ERROR_DEVICE_LOST);
        CHECK(!weakRequest.expired()); // Deliberately quarantined native request graph.
        CHECK(pool->Snapshot().usedBytes == baseline && owner.OutstandingCommands() == 0);
        CHECK(preparer->GetAdmissionUsage().jobs == 0);
        // This mode is process-isolated: the retained graph owns context/native lifetime.
        widthPipeline.reset();
        context.reset();
        contextInfo.nativeLifetime.reset();
        native.reset();
        CHECK(!weakContext.expired() && !weakNative.expired());
        return 0;
    }

    // Cancellation is ordered after preparation admission but before its task
    // can start.  The cancellation token copied above must suppress that task.
    CHECK(owner.CancelPending() != 0);
    owner.InvokeOwner([] {}); // Owner barrier: cancellation is visible to the queued task.
    CHECK(cancellation.Superseded());
    gate->Release();
    CHECK(done.wait_for(kTimeout) == std::future_status::ready);
    preparer->Drain();
    owner.Drain();
    CHECK(!gate->completionError.load(std::memory_order_acquire));
    CHECK(callbacks == 1 && callbackOnOwner.load(std::memory_order_acquire));
    CHECK(finalState.load(std::memory_order_acquire) == VulkanSourceWidthJob::State::Superseded);
    CHECK(finalStatus.load(std::memory_order_acquire) == VK_SUCCESS);
    CHECK(pool->Snapshot().usedBytes == baseline && owner.OutstandingCommands() == 0);
    CHECK(preparer->GetAdmissionUsage().jobs == 0);
    job.reset();
    owner.Drain();
    CHECK(weakRequest.expired());
    return 0;
}
