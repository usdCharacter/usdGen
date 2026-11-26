#include "usdGen/sessionBackendExecutor.h"

#include "usdGen/registries/backend.h"
#include "usdGen/registries/backendRegistry.h"
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
        : _cooker(threadLimit, cacheBytes, std::move(domain)), _provider(std::move(provider)),
          _providerDevice(_provider ? _provider->Identity().backend : UsdGenDeviceBackend::Unknown) {
        _cooker.SetDeviceProvider(_provider);
    }

    bool CanRoute(UsdGenExecutionBackend backend) const noexcept override {
        if (backend == UsdGenExecutionBackend::CpuReference) return true;
        return Injected(backend) || Registered(backend);
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

            // A registered backend (or the injected provider): one device path.
            if (backend != UsdGenExecutionBackend::CpuReference) {
                if (!request.runtime || !request.deviceReturnBinder)
                    throw std::runtime_error("device Session executor has no runtime/return route");
                std::shared_ptr<UsdGenSessionDeviceProvider> provider =
                    Injected(backend) ? _provider : nullptr;
                // The async work gate retains this executor and excludes any
                // later cooker mutation until completion. Native construction
                // and canonical registry resolution run externally, so neither
                // the Session owner nor its worker waits for initialization.
                if (!provider) {
                    UsdGenBackendId const id = IdOf(backend);
                    UsdGenBackendDef const *def = UsdGenGetBackend(id);
                    DefaultProvider &slot = _defaults[UsdGenIndex(id)];
                    // Context loss can arrive through Session's separate
                    // serialized loss work before the next cook request. A
                    // fresh default provider starts at epoch zero; consult
                    // its loss latch even when this request has no loss list.
                    // A poisonable backend recovers in place instead: the
                    // cooker's context-loss turn retired its workspace.
                    bool const replaceOnLoss =
                        !def || !(def->flags & kUsdGenBackendPoisonable);
                    bool contextLost = replaceOnLoss && slot.provider &&
                        slot.provider->Identity().contextEpoch != 0;
                    for (auto const& loss : request.contextLosses) {
                        if (!replaceOnLoss) break;
                        if (slot.provider && loss.serial > slot.lossSerial &&
                            loss.backend == slot.provider->Identity().backend &&
                            (loss.deviceIndex < 0 || loss.deviceIndex ==
                                slot.provider->Identity().deviceIndex)) {
                            slot.lossSerial = loss.serial;
                            contextLost = true;
                        }
                    }
                    if (contextLost) {
                        slot.provider->Shutdown();
                        slot.provider.reset();
                        _cooker.SetDeviceProvider({});
                        fail(std::make_exception_ptr(std::runtime_error(
                            std::string(Label(def)) +
                            " device context was lost; retry creates a fresh context")));
                        return;
                    }
                    if (!slot.provider) {
                        // Device creation blocks, so it runs as a process
                        // scheduler task, off this lane; the task resubmits.
                        auto completion = request.completion;
                        auto previous = request.previous;
                        try {
                            auto* task = new ProviderTask;
                            task->self = shared_from_this();
                            task->backend = id;
                            task->request = std::move(request);
                            UsdGenTaskScheduler::Process().Post(task);
                        } catch (...) {
                            if (completion) completion(std::move(previous), std::current_exception());
                        }
                        return;
                    }
                    provider = slot.provider;
                }
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
        for (DefaultProvider &slot : _defaults) {
            try {
                if (slot.provider) slot.provider->Shutdown();
            } catch (...) {}
            slot.provider.reset();
        }
    }

    UsdGenSessionCooker &Cooker() noexcept override { return _cooker; }
    UsdGenSessionCooker const &Cooker() const noexcept override { return _cooker; }

private:
    // Until step 8.4 a backend handle equals its UsdGenExecutionBackend value.
    static UsdGenBackendId IdOf(UsdGenExecutionBackend backend) noexcept {
        return UsdGenBackendIdOf(uint8_t(backend));
    }
    static char const *Label(UsdGenBackendDef const *def) noexcept {
        return !def ? "device" : def->label ? def->label : def->name;
    }
    /// The injected provider serves the requests of its own device backend.
    bool Injected(UsdGenExecutionBackend backend) const noexcept {
        return _provider && _providerDevice != UsdGenDeviceBackend::Unknown &&
            GetUsdGenExecutionBackendContract(backend).deviceBackend == _providerDevice;
    }
    /// A registered backend this process can run.
    static bool Registered(UsdGenExecutionBackend backend) noexcept {
        UsdGenBackendDef const *def = UsdGenGetBackend(IdOf(backend));
        if (!def || !def->createProvider) return false;
        try {
            return !def->available || def->available(nullptr);
        } catch (...) {
            return false;
        }
    }

    struct ProviderTask : UsdGenTask {
        ProviderTask() noexcept : UsdGenTask(&Run) {}
        std::shared_ptr<SessionBackendExecutor> self;
        UsdGenBackendId backend = UsdGenBackendId::Invalid;
        UsdGenSessionBackendRequest request;
        static void Run(UsdGenTask* task) noexcept {
            std::unique_ptr<ProviderTask> own(static_cast<ProviderTask*>(task));
            auto& request = own->request;
            try {
                UsdGenBackendDef const *def = UsdGenGetBackend(own->backend);
                std::string reason = "the backend is not registered";
                UsdGenBackendProviderDesc desc;
                desc.callerDevice = request.callerDevice;
                auto created = def && def->createProvider
                    ? def->createProvider(desc, &reason) : nullptr;
                if (!created) {
                    if (request.completion) request.completion(request.previous,
                        std::make_exception_ptr(std::runtime_error(
                            std::string(Label(def)) +
                            " execution backend factory is unavailable: " + reason)));
                    return;
                }
                DefaultProvider &slot = own->self->_defaults[UsdGenIndex(own->backend)];
                slot.provider = std::move(created);
                own->self->_cooker.SetDeviceProvider(slot.provider);
                own->self->Submit(std::move(request));
            } catch (...) {
                if (request.completion) request.completion(request.previous, std::current_exception());
            }
        }
    };
    struct DefaultProvider {
        std::shared_ptr<UsdGenSessionDeviceProvider> provider;
        uint64_t lossSerial = 0;
    };
    UsdGenSessionCooker _cooker;
    std::shared_ptr<UsdGenSessionDeviceProvider> _provider;
    UsdGenDeviceBackend _providerDevice = UsdGenDeviceBackend::Unknown;
    // Lazily created providers of registered backends, by handle.
    DefaultProvider _defaults[kUsdGenMaxBackends];
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
