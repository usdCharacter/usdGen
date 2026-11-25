// Private, native-header-free explicit Session device injection.
#ifndef USDGEN_SESSION_DEVICE_INTEGRATION_H
#define USDGEN_SESSION_DEVICE_INTEGRATION_H

#include "usdGen/session.h"
#include "usdGen/sessionDeviceProvider.h"

namespace usdGen {

struct UsdGenSessionDeviceObservers {
    // Thread-neutral, nonblocking admission notification after registration
    // is retained. May run on the work lane; never access mutable Session or
    // cooker state here. Primarily used for deterministic lifecycle evidence.
    std::function<void(uint64_t epoch, bool leader)> coalescedRegistered;
    // Called on the external destruction thread after Session command
    // admission closes, before the async work gate drains. Nonblocking and
    // thread-neutral: never inspect or mutate Session/cooker state here.
    std::function<void()> admissionClosed;
};

// The caller owns the shared per-device provider and exact logical-context
// cache domain obtained through ExecutionCacheDomain::AcquireShared. Equal
// labels on a separately constructed cache are rejected. Session destruction
// never closes that shared native domain.
std::unique_ptr<UsdGenSession> CreateUsdGenDeviceSession(
    int threadLimit, uint64_t commandCapacity,
    std::shared_ptr<UsdGenExecutionCacheDomain>,
    std::shared_ptr<UsdGenSessionDeviceProvider>,
    UsdGenSessionDeviceObservers const& observers = {});

} // namespace usdGen
#endif
