// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "externalRetirement.h"
#include "completionService.h"
#include "usdGen/exec/executionRuntimeInternal.h"
#include "usdGen/exec/serialLane.h"
#include "usdGen/exec/parkingWord.h"
#include "usdGen/exec/sharedCell.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <corecrt_startup.h>
#endif

namespace usdGen::vulkan {
namespace {

// Test-only: runs between claiming the lane start and admitting work;
// false fails the start.
std::atomic<bool (*)()> gLaneStartHook{nullptr};

// True on a thread while it runs a retirement pass: Close from a callback
// would wait for its own pass.
thread_local bool tRetirementPass = false;

// Process exit hook. A DLL's atexit runs at DLL detach, after ExitProcess has
// killed the lane and after every exe hook; the CRT's process table runs in
// registration order with the exe's own hooks while threads still live.
bool RegisterProcessExit(void (*fn)()) noexcept {
#if defined(_WIN32)
    return _crt_atexit(fn) == 0;
#else
    return std::atexit(fn) == 0;
#endif
}

// Multi-producer, single-consumer bounded ring (per-slot sequence numbers).
// The tail word carries the closed bit, so a producer either claims a slot
// before Close or fails: Close fixes the final tail in one fetch_or.
class ExternalRetirementQueue final {
public:
    ExternalRetirementQueue() noexcept {
        for (uint64_t i = 0; i != kCapacity; ++i) slots_[i].seq.store(i, std::memory_order_relaxed);
    }

    bool Track(std::shared_ptr<VulkanExternalRetirementDomain> const& domain) noexcept {
        return Push({}, domain);
    }

    bool RegisterAtExit() noexcept {
        return !(tail_.load(std::memory_order_acquire) & kClosed) &&
               RegisterProcessExit(CloseVulkanExternalRetirement);
    }

    bool Enqueue(std::function<void()> callback) noexcept {
        return callback && Push(std::move(callback), {});
    }

    void Close() noexcept {
        tail_.fetch_or(kClosed, std::memory_order_seq_cst);
        for (uint32_t s = lane_.load(std::memory_order_acquire);;) {
            if (s == kLaneIdle) {
                // Nothing was ever admitted.
                if (lane_.compare_exchange_weak(s, kLaneDone, std::memory_order_acq_rel)) break;
            } else if (s == kLaneRunning) {
                if (tRetirementPass) std::terminate();
                Wake();
                // Shutdown parks until the lane drained and quiesced. During
                // process detach the scheduler's workers are gone: skip it.
                if (UsdGenProcessShutdownInProgress()) break;
                UsdGenParkWhileEqual(lane_, s);
                s = lane_.load(std::memory_order_acquire);
            } else if (s == kLaneDone) {
                break;
            } else {  // a start in flight: shutdown parks
                UsdGenParkWhileEqual(lane_, s);
                s = lane_.load(std::memory_order_acquire);
            }
        }
    }

private:
    static constexpr uint64_t kCapacity = 1024, kClosed = 1;
    static constexpr uint32_t kLaneIdle = 0, kLaneStarting = 1, kLaneRunning = 2,
                              kLaneDone = 3;

    struct Slot {
        std::atomic<uint64_t> seq{0};
        std::function<void()> callback;
        std::weak_ptr<VulkanExternalRetirementDomain> domain;
    };

    bool Push(std::function<void()> callback,
              std::shared_ptr<VulkanExternalRetirementDomain> const& domain) noexcept {
        // The lane starts before admission, so Close can always quiesce a
        // live service whose consumer is still held.
        if (!StartLane()) return false;
        uint64_t t = tail_.load(std::memory_order_acquire);
        Slot* slot;
        for (;;) {
            if (t & kClosed) return false;
            uint64_t const pos = t >> 1;
            slot = &slots_[pos % kCapacity];
            int64_t const diff = int64_t(slot->seq.load(std::memory_order_acquire) - pos);
            if (diff == 0) {
                if (tail_.compare_exchange_weak(t, t + 2, std::memory_order_seq_cst)) break;
            } else if (diff < 0) {
                return false;  // full: every slot holds an unfinished callback
            } else {
                t = tail_.load(std::memory_order_acquire);
            }
        }
        uint64_t const pos = t >> 1;
        // Moves into a default-constructed slot never throw.
        slot->callback = std::move(callback);
        slot->domain = domain;
        slot->seq.store(pos + 1, std::memory_order_seq_cst);
        Wake();
        return true;
    }

    void Wake() noexcept { lane.Kick(); }

    // Admission waits out a start in flight: a producer that claimed a slot
    // while another thread's start was about to fail would leave an accepted
    // callback with no lane to run it (Close takes Idle straight to Joined).
    // The wait covers one exit-hook registration, once per process.
    bool StartLane() noexcept {
        for (uint32_t s = lane_.load(std::memory_order_acquire);;) {
            if (s == kLaneStarting) {
                UsdGenParkWhileEqual(lane_, s);
                s = lane_.load(std::memory_order_acquire);
                continue;
            }
            if (s != kLaneIdle) return true;  // running, or closing (Push rejects)
            if (!lane_.compare_exchange_weak(s, kLaneStarting, std::memory_order_acq_rel)) continue;
            auto const hook = gLaneStartHook.load(std::memory_order_acquire);
            bool started = (!hook || hook()) && RegisterProcessExit(CloseVulkanExternalRetirement);
            lane_.store(started ? kLaneRunning : kLaneIdle, std::memory_order_release);
            UsdGenUnparkAll(lane_);
            return started;
        }
    }

    // One pass on the process scheduler (UsdGenSerialLane: never two at once,
    // no kick lost): runs every published callback in order, then, once
    // admission is closed and drained, quiesces the tracked domains.
    static void Pass(void* user) noexcept {
        auto& queue = *static_cast<ExternalRetirementQueue*>(user);
        tRetirementPass = true;
        queue.Drain();
        tRetirementPass = false;
    }

    void Drain() noexcept {
        if (lane_.load(std::memory_order_acquire) == kLaneDone) return;
        for (;;) {
            Slot& slot = slots_[head_ % kCapacity];
            if (slot.seq.load(std::memory_order_seq_cst) != head_ + 1) break;
            auto callback = std::move(slot.callback);
            auto domain = std::move(slot.domain);
            slot.callback = nullptr;
            slot.domain.reset();
            if (callback) {
                try { callback(); } catch (...) {}
                // Native captures are destroyed before the slot returns
                // to admission or Close reports this callback complete.
                callback = nullptr;
            } else if (!domain.expired()) {
                try {
                    domains_.erase(std::remove_if(domains_.begin(), domains_.end(),
                        [](auto const& d) { return d.expired(); }), domains_.end());
                    domains_.push_back(std::move(domain));
                } catch (...) {}
            }
            slot.seq.store(head_ + kCapacity, std::memory_order_release);
            ++head_;
        }
        // Empty, or a producer is between its claim and its publish (its
        // publish kicks another pass).
        uint64_t const t = tail_.load(std::memory_order_seq_cst);
        if ((t >> 1) != head_ || !(t & kClosed)) return;
        // Admission is closed and accepted callbacks have finished. A deferred
        // provider retains its record and every unproved resource through its
        // domain cycle.
        for (auto const& entry : domains_)
            if (auto domain = entry.lock()) domain->Quiesce();
        domains_.clear();
        lane_.store(kLaneDone, std::memory_order_release);
        UsdGenUnparkAll(lane_);
    }

    Slot slots_[kCapacity];
    alignas(64) std::atomic<uint64_t> tail_{0};  // (claimed position << 1) | closed
    std::atomic<uint32_t> lane_{kLaneIdle};
    UsdGenSerialLane lane{&ExternalRetirementQueue::Pass, this};
    uint64_t head_ = 0;  // lane only
    std::vector<std::weak_ptr<VulkanExternalRetirementDomain>> domains_;  // lane only
};

// The queue survives static destruction: only its lane is joined, and late
// domain proofs must still see closed admission, never stale storage.
std::atomic<ExternalRetirementQueue*> gQueue{nullptr};

// Read only through UsdGenLazyLeaked: while the first caller constructs the
// queue the slot holds its busy sentinel (address 1), which a raw load would
// hand back as a queue. Its own fast path already skips the sentinel.
ExternalRetirementQueue* Queue() noexcept {
    try {
        return &UsdGenLazyLeaked(gQueue);
    } catch (...) { return nullptr; }
}
} // namespace

VulkanExternalRetirementDomain::CloseEntry VulkanExternalRetirementDomain::BeginClose() noexcept {
    for (uint32_t s = state_.load(std::memory_order_acquire);;) {
        if (s & (kQuiescing | kQuiesced)) return CloseEntry::Retain;
        uint32_t const next = (s & kClosing) ? (s | kRerun) : (s | kClosing);
        if (state_.compare_exchange_weak(s, next, std::memory_order_acq_rel))
            return (s & kClosing) ? CloseEntry::Deferred : CloseEntry::Run;
    }
}

bool VulkanExternalRetirementDomain::EndClose() noexcept {
    for (uint32_t s = state_.load(std::memory_order_acquire);;) {
        // A deferred close reruns unless quiescence began meanwhile.
        bool const rerun = (s & kRerun) && !(s & (kQuiescing | kQuiesced));
        uint32_t const next = rerun ? (s & ~kRerun) : (s & ~(kClosing | kRerun));
        if (!state_.compare_exchange_weak(s, next, std::memory_order_acq_rel)) continue;
        if (!rerun) UsdGenUnparkAll(state_);
        return rerun;
    }
}

void VulkanExternalRetirementDomain::Quiesce() noexcept {
    // Claim quiescence. Only shutdown paths park here: on another quiescer,
    // then on a close in flight; neither ever waits in turn.
    uint32_t s = state_.load(std::memory_order_acquire);
    for (;;) {
        if (s & kQuiesced) return;
        if (s & kQuiescing) {
            UsdGenParkWhileEqual(state_, s);
            s = state_.load(std::memory_order_acquire);
        } else if (state_.compare_exchange_weak(s, s | kQuiescing, std::memory_order_acq_rel)) {
            s |= kQuiescing;
            break;
        }
    }
    while (s & kClosing) {
        UsdGenParkWhileEqual(state_, s);
        s = state_.load(std::memory_order_acquire);
    }
    // Close ingress and finish owner registration before signaling the
    // service's host-control semaphore. Held consumers are not awaited.
    // Shutdown closes ingress before its potentially throwing graph wait;
    // even a dispatch failure must still stop the native service waiter.
    try { owner_->Shutdown(); } catch (...) {}
    bool const quiesced = service_->CloseAndJoin();
    if (quiesced) state_.fetch_or(kQuiesced, std::memory_order_acq_rel);
    state_.fetch_and(~kQuiescing, std::memory_order_acq_rel);
    UsdGenUnparkAll(state_);
}

std::shared_ptr<VulkanExternalRetirementDomain> TrackVulkanExternalRetirementDomain(
    std::shared_ptr<UsdGenExecutionPipeline> owner,
    std::shared_ptr<VulkanCompletionService> service) noexcept {
    if (!owner || !service || service->queueOwner() != owner.get()) return {};
    try {
        auto domain = std::make_shared<VulkanExternalRetirementDomain>();
        domain->owner_ = std::move(owner);
        domain->service_ = std::move(service);
        auto* queue = Queue();
        return queue && queue->Track(domain) ? domain : nullptr;
    } catch (...) { return {}; }
}

bool RegisterVulkanExternalRetirementAtExit() noexcept {
    auto* queue = Queue();
    return queue && queue->RegisterAtExit();
}

bool EnqueueVulkanExternalRetirement(std::function<void()> callback) noexcept {
    auto* queue = Queue();
    return queue && queue->Enqueue(std::move(callback));
}

void SetVulkanExternalRetirementLaneStartHookForTesting(bool (*hook)()) noexcept {
    gLaneStartHook.store(hook, std::memory_order_release);
}

void CloseVulkanExternalRetirement() noexcept {
    if (auto* queue = Queue()) queue->Close();
}

} // namespace usdGen::vulkan
