#include "usdGen/sessionBackendExecutor.h"

#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
#include "usdGen/vulkan/defaultProvider.h"
#endif

#include "usdGen/taskScheduler.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace usdGen {
namespace {

class SessionBackendExecutor final : public UsdGenSessionBackendExecutor,
                                     public std::enable_shared_from_this<SessionBackendExecutor> {
public:
    SessionBackendExecutor(int threadLimit, size_t cacheBytes,
                           std::shared_ptr<UsdGenExecutionCacheDomain> domain,
                           std::shared_ptr<UsdGenSessionDeviceProvider> provider)
        : _cooker(threadLimit, cacheBytes, std::move(domain)), _provider(std::move(provider)) {
        _cooker.SetDeviceProvider(_provider);
    }

    bool CanRoute(UsdGenExecutionBackend backend) const noexcept override {
        if (backend == UsdGenExecutionBackend::CpuReference) return true;
        if (backend == UsdGenExecutionBackend::Vulkan) {
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
            return true;
#else
            return bool(_provider);
#endif
        }
#ifdef USDGEN_ENABLE_CUDA
        if (backend == UsdGenExecutionBackend::Cuda) return true;
#endif
        return false;
    }

    void Submit(UsdGenSessionBackendRequest request) override {
        auto fail = [&request](std::exception_ptr error) {
            if (request.completion) request.completion(request.previous, std::move(error));
        };
        try {
            if (!request.desc) {
                fail(std::make_exception_ptr(std::runtime_error(
                    "Session backend request has no graph description")));
                return;
            }
            // Context-loss ordering is Session semantics, but belongs on this
            // executor's serial work lane with the workspace it can retire.
            for (UsdGenDeviceContextLoss const& loss : request.contextLosses) {
                if (!_cooker.ApplyDeviceContextLoss(loss)) {
                    fail(std::make_exception_ptr(std::runtime_error(
                        "device context-loss invalidation failed")));
                    return;
                }
            }

            UsdGenExecutionBackend const backend = request.desc
                ? request.desc->executionBackend : UsdGenExecutionBackend::Invalid;
            if (!CanRoute(backend)) {
                // Preserve the existing precise compiler validation and
                // retained-previous publication behavior.  Crucially, no
                // native or CPU scheduler execution starts for this backend.
                auto generation = _cooker.Cook(std::move(request.desc), request.context,
                    request.devicePublicationEnabled, std::move(request.pending),
                    request.frame, request.reason, std::move(request.previous),
                    std::move(request.publishedStats), request.invalidateValues,
                    request.previousPublishedWorkerEpoch, request.cancellation.epoch,
                    request.callerDevice, std::move(request.coalescedHooks),
                    std::move(request.tileProgress),
                    [cancel=request.cancellation] { return cancel.Superseded(); });
                if (request.completion) request.completion(std::move(generation), {});
                return;
            }

            if (backend == UsdGenExecutionBackend::Vulkan) {
                if (!request.runtime || !request.deviceReturnBinder)
                    throw std::runtime_error("device Session executor has no runtime/return route");
                auto provider = _provider;
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
                // The async work gate retains this executor and excludes any
                // later cooker mutation until completion. Native construction
                // and canonical registry resolution run externally, so neither
                // the Session owner nor its worker waits for initialization.
                if (!provider) {
                    // Context loss can arrive through Session's separate
                    // serialized loss work before the next cook request. A
                    // fresh default provider starts at epoch zero; consult
                    // its loss latch even when this request has no loss list.
                    bool contextLost = _defaultProvider &&
                        _defaultProvider->Identity().contextEpoch != 0;
                    for (auto const& loss : request.contextLosses) {
                        if (_defaultProvider && loss.serial > _defaultLossSerial &&
                            loss.backend == UsdGenDeviceBackend::Vulkan &&
                            (loss.deviceIndex < 0 || loss.deviceIndex ==
                                _defaultProvider->Identity().deviceIndex)) {
                            _defaultLossSerial = loss.serial;
                            contextLost = true;
                        }
                    }
                    if (contextLost) {
                        _defaultProvider->Shutdown();
                        _defaultProvider.reset();
                        _cooker.SetDeviceProvider({});
                        fail(std::make_exception_ptr(std::runtime_error(
                            "Vulkan device context was lost; retry creates a fresh context")));
                        return;
                    }
                    if (!_defaultProvider) {
                        // Device creation blocks, so it runs as a process
                        // scheduler task, off this lane; the task resubmits.
                        auto completion = request.completion;
                        auto previous = request.previous;
                        try {
                            auto* task = new ProviderTask;
                            task->self = shared_from_this();
                            task->request = std::move(request);
                            UsdGenTaskScheduler::Process().Post(task);
                        } catch (...) {
                            if (completion) completion(std::move(previous), std::current_exception());
                        }
                        return;
                    }
                    provider = _defaultProvider;
                }
#endif
                auto completion = request.completion;
                _cooker.CookDeviceAsync(*request.runtime, request.cancellation,
                    std::move(request.desc), request.context,
                    request.devicePublicationEnabled, std::move(request.pending),
                    request.frame, request.reason, std::move(request.previous),
                    std::move(request.publishedStats), request.invalidateValues,
                    request.previousPublishedWorkerEpoch, request.callerDevice,
                    provider, std::move(request.deviceReturnBinder),
                    std::move(completion), std::move(request.coalescedHooks));
                return;
            }

            if (backend == UsdGenExecutionBackend::Cuda) {
                if (!request.runtime)
                    throw std::runtime_error("CUDA Session executor has no runtime");
                // Keep request.completion intact for the catch path: a
                // synchronous setup exception must still release Pipeline's
                // async gate exactly once.
                auto completion = request.completion;
                _cooker.CookCudaAsync(*request.runtime, request.cancellation,
                    std::move(request.desc), request.context,
                    request.devicePublicationEnabled, std::move(request.pending),
                    request.frame, request.reason, std::move(request.previous),
                    std::move(request.publishedStats), request.invalidateValues,
                    request.previousPublishedWorkerEpoch, request.callerDevice,
                    std::move(completion), std::move(request.coalescedHooks));
                return;
            }

            auto generation = _cooker.Cook(std::move(request.desc), request.context,
                request.devicePublicationEnabled, std::move(request.pending),
                request.frame, request.reason, std::move(request.previous),
                std::move(request.publishedStats), request.invalidateValues,
                request.previousPublishedWorkerEpoch, request.cancellation.epoch,
                request.callerDevice, std::move(request.coalescedHooks),
                std::move(request.tileProgress),
                [cancel=request.cancellation] { return cancel.Superseded(); });
            if (_cooker.GetCoalescedRole() ==
                    UsdGenSessionCooker::CoalescedRole::Follower ||
                _cooker.GetCoalescedRole() ==
                    UsdGenSessionCooker::CoalescedRole::Resident)
                return;
            if (request.completion) request.completion(std::move(generation), {});
        } catch (...) {
            fail(std::current_exception());
        }
    }

    void Shutdown() noexcept override {
        // The owner guarantees that every async device completion has
        // returned before this object is destroyed. Coalesced followers are
        // cancelled by Session before pipeline shutdown; silence a leftover
        // registration without allowing a callback to reenter a dying
        // command owner. The injected provider stays caller-owned; only the
        // lazily created default is shut down here.
        _cooker.AbandonCoalesced(false);
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
        try {
            if (_defaultProvider) _defaultProvider->Shutdown();
        } catch (...) {}
        _defaultProvider.reset();
#endif
    }

    UsdGenSessionCooker &Cooker() noexcept override { return _cooker; }
    UsdGenSessionCooker const &Cooker() const noexcept override { return _cooker; }

private:
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    struct ProviderTask : UsdGenTask {
        ProviderTask() noexcept : UsdGenTask(&Run) {}
        std::shared_ptr<SessionBackendExecutor> self;
        UsdGenSessionBackendRequest request;
        static void Run(UsdGenTask* task) noexcept {
            std::unique_ptr<ProviderTask> own(static_cast<ProviderTask*>(task));
            auto& request = own->request;
            try {
                std::string reason;
                auto created = vulkan::CreateDefaultVulkanSessionProvider(&reason);
                if (!created) {
                    if (request.completion) request.completion(request.previous,
                        std::make_exception_ptr(std::runtime_error(
                            "Vulkan execution backend factory is unavailable: " + reason)));
                    return;
                }
                own->self->_defaultProvider = std::move(created);
                own->self->_cooker.SetDeviceProvider(own->self->_defaultProvider);
                own->self->Submit(std::move(request));
            } catch (...) {
                if (request.completion) request.completion(request.previous, std::current_exception());
            }
        }
    };
#endif
    UsdGenSessionCooker _cooker;
    std::shared_ptr<UsdGenSessionDeviceProvider> _provider;
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    std::shared_ptr<UsdGenSessionDeviceProvider> _defaultProvider;
    uint64_t _defaultLossSerial = 0;
#endif
};

} // namespace

std::shared_ptr<UsdGenSessionBackendExecutor> CreateUsdGenSessionBackendExecutor(
    int threadLimit, size_t executionCacheBytes,
    std::shared_ptr<UsdGenExecutionCacheDomain> executionCacheDomain,
    std::shared_ptr<UsdGenSessionDeviceProvider> deviceProvider)
{
    return std::make_shared<SessionBackendExecutor>(threadLimit,
        executionCacheBytes, std::move(executionCacheDomain), std::move(deviceProvider));
}

} // namespace usdGen
