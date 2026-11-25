// Private Session backend routing boundary.
//
// This is intentionally separate from executionBackend.h.  A Session cook
// carries publication fences, cache-coalescing state, and a full immutable
// generation (not merely a device generation), none of which the public
// portable-adapter request/result contract can represent yet.
#ifndef USDGEN_SESSION_BACKEND_EXECUTOR_H
#define USDGEN_SESSION_BACKEND_EXECUTOR_H

#include "usdGen/sessionCooker.h"

#include <memory>
#include <vector>

namespace usdGen {

struct UsdGenSessionBackendRequest {
    UsdGenExecutionRuntime *runtime = nullptr;
    UsdGenExecutionPipeline::Cancellation cancellation;
    std::shared_ptr<const UsdGenGraphDesc> desc;
    UsdGenContext context = UsdGenContext::Interactive;
    bool devicePublicationEnabled = false;
    UsdGenPendingDirty pending;
    double frame = 0.0;
    UsdGenCommitReason reason = UsdGenCommitReason::NoticeBatchEnd;
    UsdGenGenerationConstPtr previous;
    UsdGenStats publishedStats;
    bool invalidateValues = false;
    uint64_t previousPublishedWorkerEpoch = 0;
    int callerDevice = -1;
    std::vector<UsdGenDeviceContextLoss> contextLosses;
    UsdGenSessionCooker::CudaCompletion completion;
    UsdGenSessionCooker::CoalescedHooks coalescedHooks;
    UsdGenSession::TileProgressCallback tileProgress;
    UsdGenSessionCooker::DeviceReturnBinder deviceReturnBinder;
};

/// Per-Session executor.  It owns the cooker, and therefore the compiler,
/// cache-domain state, COW publication candidates, and (for CUDA) the native
/// workspace.  The work pipeline retains this shared object for every
/// submission, so descriptor replacement cannot redirect an old completion
/// at a newly-created backend workspace.
class UsdGenSessionBackendExecutor {
public:
    virtual ~UsdGenSessionBackendExecutor() = default;
    /// A Session owns one executor/cooker across descriptor changes. Routing
    /// is selected from each immutable request rather than freezing the first
    /// descriptor's backend and losing its last-good/store/cache state.
    virtual bool CanRoute(UsdGenExecutionBackend) const noexcept = 0;
    virtual void Submit(UsdGenSessionBackendRequest) = 0;
    /// Called only after the Session pipeline has shut down and all retained
    /// async completion closures have drained.
    virtual void Shutdown() noexcept = 0;
    virtual UsdGenSessionCooker &Cooker() noexcept = 0;
    virtual UsdGenSessionCooker const &Cooker() const noexcept = 0;
};

/// Creates the one private executor owned by a Session. CPU, CUDA and (when
/// built) Vulkan retain their normal routes. An explicitly injected Vulkan
/// provider still wins; otherwise a built runtime lazily creates its default
/// provider, while an unbuilt Vulkan (and always Metal) retains the
/// unavailable compiler diagnostic, never a fabricated CPU result. Neither
/// path registers a global route.
std::shared_ptr<UsdGenSessionBackendExecutor> CreateUsdGenSessionBackendExecutor(
    int threadLimit, size_t executionCacheBytes,
    std::shared_ptr<UsdGenExecutionCacheDomain> executionCacheDomain,
    std::shared_ptr<UsdGenSessionDeviceProvider> deviceProvider = {});

} // namespace usdGen

#endif // USDGEN_SESSION_BACKEND_EXECUTOR_H
