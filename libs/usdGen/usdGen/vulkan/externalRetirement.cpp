// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "externalRetirement.h"
#include "completionService.h"
#include "usdGen/parkingWord.h"

#include <algorithm>
#include <cstdlib>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <corecrt_startup.h>
#endif

namespace usdGen::vulkan {
namespace {

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
        Wake();
        for (uint32_t s = lane_.load(std::memory_order_acquire);;) {
            if (s == kLaneIdle) {
                if (lane_.compare_exchange_weak(s, kLaneJoined, std::memory_order_acq_rel)) break;
            } else if (s == kLaneRunning) {
                if (!lane_.compare_exchange_weak(s, kLaneJoining, std::memory_order_acq_rel)) continue;
                if (worker_.get_id() == std::this_thread::get_id()) std::terminate();
                worker_.join();
                lane_.store(kLaneJoined, std::memory_order_release);
                UsdGenUnparkAll(lane_);
                break;
            } else if (s == kLaneJoined) {
                break;
            } else {  // starting or joining elsewhere: shutdown parks
                UsdGenParkWhileEqual(lane_, s);
                s = lane_.load(std::memory_order_acquire);
            }
        }
    }

private:
    static constexpr uint64_t kCapacity = 1024, kClosed = 1;
    static constexpr uint32_t kLaneIdle = 0, kLaneStarting = 1, kLaneRunning = 2,
                              kLaneJoining = 3, kLaneJoined = 4;

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

    void Wake() noexcept {
        wake_.fetch_add(1, std::memory_order_seq_cst);
        UsdGenUnparkAll(wake_);
    }

    bool StartLane() noexcept {
        for (uint32_t s = lane_.load(std::memory_order_acquire);;) {
            if (s != kLaneIdle) return true;  // started, starting, or closed (Push rejects)
            if (!lane_.compare_exchange_weak(s, kLaneStarting, std::memory_order_acq_rel)) continue;
            bool started = RegisterProcessExit(CloseVulkanExternalRetirement);
            if (started) {
                try { worker_ = std::thread([this] { Run(); }); }
                catch (...) { started = false; }
            }
            lane_.store(started ? kLaneRunning : kLaneIdle, std::memory_order_release);
            UsdGenUnparkAll(lane_);
            return started;
        }
    }

    void Run() noexcept {
        for (;;) {
            uint32_t const w = wake_.load(std::memory_order_seq_cst);
            Slot& slot = slots_[head_ % kCapacity];
            if (slot.seq.load(std::memory_order_seq_cst) == head_ + 1) {
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
                continue;
            }
            uint64_t const t = tail_.load(std::memory_order_seq_cst);
            if ((t >> 1) == head_ && (t & kClosed)) break;  // closed and drained
            // Empty, or a producer is between its claim and its publish.
            UsdGenParkWhileEqual(wake_, w);
        }
        // Admission is closed and accepted callbacks have finished. A deferred
        // provider retains its record and every unproved resource through its
        // domain cycle.
        for (auto const& entry : domains_)
            if (auto domain = entry.lock()) domain->Quiesce();
    }

    Slot slots_[kCapacity];
    alignas(64) std::atomic<uint64_t> tail_{0};  // (claimed position << 1) | closed
    alignas(64) std::atomic<uint32_t> wake_{0};
    std::atomic<uint32_t> lane_{kLaneIdle};
    std::thread worker_;
    uint64_t head_ = 0;  // lane only
    std::vector<std::weak_ptr<VulkanExternalRetirementDomain>> domains_;  // lane only
};

// The queue survives static destruction: only its lane is joined, and late
// domain proofs must still see closed admission, never stale storage.
std::atomic<ExternalRetirementQueue*> gQueue{nullptr};

ExternalRetirementQueue* Queue() noexcept {
    if (auto* q = gQueue.load(std::memory_order_acquire)) return q;
    ExternalRetirementQueue* fresh = nullptr;
    try { fresh = new ExternalRetirementQueue; } catch (...) { return nullptr; }
    ExternalRetirementQueue* expected = nullptr;
    if (gQueue.compare_exchange_strong(expected, fresh, std::memory_order_acq_rel)) return fresh;
    delete fresh;
    return expected;
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

void CloseVulkanExternalRetirement() noexcept {
    if (auto* queue = Queue()) queue->Close();
}

} // namespace usdGen::vulkan
