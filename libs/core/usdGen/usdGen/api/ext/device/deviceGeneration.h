// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The backend side of a device generation: the owner a backend implements
// and the consumer use it hands out (api/deviceLease.h is the reader side).
#ifndef USDGEN_DEVICE_GENERATION_H
#define USDGEN_DEVICE_GENERATION_H

#include "usdGen/api/deviceLease.h"

#include <cstddef>
#include <memory>

namespace usdGen {

/// A single consumer use of a backend generation.  The handle is owned by a
/// lease state, so a backend can keep all state needed for fencing locally
/// instead of maintaining a process-wide token registry.
class UsdGenDeviceConsumer
{
public:
    virtual ~UsdGenDeviceConsumer() = default;
    virtual UsdGenDeviceStream Stream() const noexcept = 0;

    /// Orders the producer's work before the consumer dereferences channels.
    /// A failed wait does not complete the handle; the owner remains alive
    /// until Complete (or destruction) fences and releases it.
    virtual UsdGenDeviceStatus WaitUntilReady() const noexcept = 0;

    /// Arranges terminal retirement of the consumer stream and releases the
    /// use. It may return before native work completes; the backend must keep
    /// the owner alive until that completion is proven. Implementations must
    /// be idempotent and must not throw.
    virtual void Complete() noexcept = 0;
};

/// Backend-specific implementation of allocation ownership and fencing.
/// Implementations may add typed accessors in a private/backend header, but
/// the generation and lease only depend on this synchronization contract.
class UsdGenDeviceOwner
{
public:
    virtual ~UsdGenDeviceOwner() = default;

    /// Return true only when all producer work needed by the generation has
    /// completed or can safely be waited on.
    virtual bool ProducerReady() const noexcept = 0;

    /// Exact bytes owned exclusively by this owner. Shared COW predecessors
    /// are excluded so a cache can charge each physical delta only once.
    /// A backend returning zero is ineligible for device-cache admission.
    virtual size_t ExclusiveRetainedBytes() const noexcept { return 0; }

    /// Conservative physical footprint reachable through this owner,
    /// including immutable COW predecessors it retains. This is the
    /// cache-charge view: it may count a shared predecessor again when more
    /// than one independent cache entry references it, but it cannot
    /// undercharge an entry whose predecessor outlives its cache record.
    /// Backends with no COW predecessor inherit the exclusive value.
    virtual size_t InclusiveRetainedBytes() const noexcept {
        return ExclusiveRetainedBytes();
    }

    /// Acquire a consumer use.  The owner may reject a stream that cannot be
    /// fenced safely.  The returned handle owns all per-use backend state and
    /// is released by UsdGenDeviceLease destruction/completion.
    virtual std::unique_ptr<UsdGenDeviceConsumer> AcquireConsumer(
        UsdGenDeviceStream stream) const noexcept = 0;

    /// The whole storage of the channel with this semantic. False: absent,
    /// or the owner exposes no device address.
    virtual bool ChannelAddress(UsdGenDeviceChannelSemantic semantic,
                                UsdGenDeviceChannelAddress *out) const noexcept
    {
        (void)semantic;
        (void)out;
        return false;
    }
};

} // namespace usdGen

#endif  // USDGEN_DEVICE_GENERATION_H
