// Private Session-to-device-provider boundary.  This is deliberately not an
// execution-backend registration surface: a Session injects one provider for
// one explicitly owned device domain.
#ifndef USDGEN_SESSION_DEVICE_PROVIDER_H
#define USDGEN_SESSION_DEVICE_PROVIDER_H

#include "usdGen/deviceGeneration.h"
#include "usdGen/executionBackend.h"
#include "usdGen/executionValueRevisions.h"

#include <exception>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

namespace usdGen {

using UsdGenSessionDevicePlanHandle = std::shared_ptr<const UsdGenExecutionPlanHandle>;

// The identity is an exact admission key, not a device-selection hint.  The
// UUID is the factory's physical-device/pool identity; contextEpoch is bumped
// for every accepted context-loss event before a replacement context is used.
struct UsdGenSessionDeviceIdentity {
    UsdGenDeviceBackend backend = UsdGenDeviceBackend::Unknown;
    int32_t deviceIndex = -1;
    uint32_t capabilityVersion = 0;
    std::array<uint8_t, 16> physicalDeviceUuid{};
    // Stable execution-cache-domain logical context token. This prevents two
    // independently owned logical contexts on one physical device from
    // aliasing even when their epochs happen to match.
    uint64_t logicalContextIdentity = 0;
    uint64_t contextEpoch = 0;

    bool operator==(UsdGenSessionDeviceIdentity const& other) const noexcept {
        return backend == other.backend && deviceIndex == other.deviceIndex &&
            capabilityVersion == other.capabilityVersion &&
            physicalDeviceUuid == other.physicalDeviceUuid &&
            logicalContextIdentity == other.logicalContextIdentity &&
            contextEpoch == other.contextEpoch;
    }
    bool operator!=(UsdGenSessionDeviceIdentity const& other) const noexcept {
        return !(*this == other);
    }
};

// All values are authoritative when supplied.  Zero is a valid engine
// revision, so absence -- not a sentinel -- selects the plan's frozen values.
using UsdGenSessionDeviceRevisions = UsdGenExecutionValueRevisions;

struct UsdGenSessionDeviceResult {
    UsdGenSessionDeviceIdentity identity;
    UsdGenSessionDeviceRevisions revisions;
    std::shared_ptr<const UsdGenDeviceGeneration> generation;
    std::exception_ptr error;
};

// This callback is reserved by the Session command owner before Submit.  It
// must only post the immutable result to that owner (or retain it and issue a
// thread-neutral terminal failure); it must never call a cooker on a device
// owner frame.
// Return true only after the immutable result has been retained by the
// Session transport. False lets a provider quarantine the native-bearing
// result rather than releasing it on an unproved or foreign thread.
using UsdGenSessionDeviceReturn = std::function<bool(std::shared_ptr<const UsdGenSessionDeviceResult>)>;

struct UsdGenSessionDeviceRequest {
    UsdGenSessionDevicePlanHandle plan;
    UsdGenSessionDeviceIdentity identity;
    // Session-allocated neutral publication identity; distinct from tool and
    // value revisions and passed unchanged to the native publication bridge.
    uint64_t publicationGeneration = 0;
    std::optional<UsdGenSessionDeviceRevisions> authoritativeRevisions;
    UsdGenExecutionPipeline::Cancellation cancellation;
    UsdGenDeviceToolMetadata tool;
    // Optional caller-owned state retained through terminal publication.
    std::shared_ptr<const void> requestLifetime;
    UsdGenSessionDeviceReturn returnTransport;
    // Evaluation time belongs to this immutable request. Plans are cached
    // across SetTime commits and must not freeze expression clocks.
    std::optional<double> evaluationFrame;

};

class UsdGenSessionDeviceProvider {
public:
    virtual ~UsdGenSessionDeviceProvider() = default;
    virtual UsdGenSessionDevicePlanHandle Compile(UsdGenGraphDesc const&,
                                                   UsdGenDiagnostics*) const = 0;
    virtual UsdGenSessionDeviceIdentity Identity() const noexcept = 0;
    // Advances the provider's epoch only when `lost` exactly identifies its
    // current context.  Old plans/requests are thereafter rejected.
    virtual bool NotifyContextLost(UsdGenSessionDeviceIdentity const& lost) noexcept = 0;
    // False means neither device-owner ingress nor native work was admitted.
    virtual bool Submit(UsdGenSessionDeviceRequest) = 0;
    virtual void Shutdown() noexcept = 0;
};

} // namespace usdGen

#endif // USDGEN_SESSION_DEVICE_PROVIDER_H
