#include "usdGen/api/ext/device/executionRetirement.h"
#include "usdGen/exec/executionRuntimeInternal.h"
#include "usdGen/api/parkingWord.h"
#include "usdGen/api/serialLane.h"

#include <atomic>
#include <array>
#include <chrono>
#include <exception>
#include <limits>
#include <new>
#include <utility>
#include <vector>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace usdGen {
namespace {
enum SlotFlags : uint32_t {
    Reserved = 1u << 0,
    Installing = 1u << 1,
    Retired = 1u << 2,
    Success = 1u << 3,
    Failure = 1u << 4,
    Scheduled = 1u << 5,
    Quarantined = 1u << 6,
    Accounted = 1u << 7,
    Spent = 1u << 8,
};
constexpr uint64_t _Pack(uint32_t generation, uint32_t flags) noexcept {
    return (uint64_t(generation) << 32) | flags;
}
constexpr uint32_t _Generation(uint64_t state) noexcept { return uint32_t(state >> 32); }
constexpr uint32_t _Flags(uint64_t state) noexcept { return uint32_t(state); }

// Index of a backend handle; kUsdGenMaxBackends when out of range.
constexpr size_t _BackendIndex(UsdGenBackendId backend) noexcept {
    uint32_t const index = UsdGenIndex(backend);
    return index < kUsdGenMaxBackends ? index : kUsdGenMaxBackends;
}
// Constant-initialized: false means open.
std::atomic<bool> _backendClosed[kUsdGenMaxBackends] = {};
bool _BackendIsOpen(UsdGenBackendId backend) noexcept {
    size_t const index = _BackendIndex(backend);
    return index < kUsdGenMaxBackends && !_backendClosed[index].load(std::memory_order_acquire);
}

// A single guard covers every service: invoking Drain from a cleanup worker,
// including a different device service, would wait on work that cannot make
// progress until the current task returns.
thread_local bool _retirementCleanupActive = false;

// Test-only gate. Each Set publishes an immutable holder that is never freed
// (a handful per test process), so readers copy it without reclamation.
struct _GateHolder {
    std::shared_ptr<UsdGenExecutionRetirementInstallTestGate> gate;
};
std::atomic<_GateHolder const*> _installTestGate{nullptr};

// Drain credit: a count of admitted cleanups. Drain parks on it until zero.
struct _DrainCount {
    std::atomic<uint32_t> pending{0};
    void Reserve() noexcept { pending.fetch_add(1, std::memory_order_acq_rel); }
    void Release() noexcept {
        if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) UsdGenUnparkAll(pending);
    }
    void WaitForAll() noexcept {
        for (uint32_t v; (v = pending.load(std::memory_order_acquire)) != 0;)
            UsdGenParkWhileEqual(pending, v);
    }
};
} // namespace

struct UsdGenExecutionRetirementSignal::State {
    struct Slot {
        // One CAS value prevents a late callback from validating generation N
        // then mutating flags for generation N+1.
        std::atomic<uint64_t> state{0};
        std::atomic<UsdGenExecutionRetirementPayload*> payload{nullptr};
        // Retirement-lane queue link, written by the one Scheduled claimant.
        State* owner = nullptr;
        size_t index = 0;
        uint32_t laneGeneration = 0;
        Slot* laneNext = nullptr;
    };

    explicit State(size_t capacity, UsdGenBackendId backend_)
        : backend(backend_) {
        slots.reserve(capacity);
        for (size_t i = 0; i != capacity; ++i) {
            slots.emplace_back(new Slot);
            slots.back()->owner = this;
            slots.back()->index = i;
        }
    }

    bool ClaimDrain(Slot& slot, uint32_t generation) noexcept {
        uint64_t old = slot.state.load(std::memory_order_acquire);
        for (;;) {
            if (_Generation(old) != generation || (_Flags(old) & Accounted)) return false;
            if (slot.state.compare_exchange_weak(
                    old, _Pack(generation, _Flags(old) | Accounted),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return true;
            }
        }
    }
    void ReleaseDrain(Slot& slot, uint32_t generation) noexcept {
        if (ClaimDrain(slot, generation)) drain.Release();
    }

    // Exactly one caller makes a terminal claim. The object is deliberately
    // retained in its fixed slot; Quarantine must not release callback data.
    void Quarantine(Slot& slot, uint32_t generation, bool callPayload) noexcept {
        uint64_t old = slot.state.load(std::memory_order_acquire);
        for (;;) {
            if (_Generation(old) != generation || (_Flags(old) & Quarantined)) return;
            if (slot.state.compare_exchange_weak(
                    old, _Pack(generation, _Flags(old) | Quarantined),
                    std::memory_order_acq_rel, std::memory_order_acquire))
                break;
        }
        if (callPayload) {
            if (auto* payload = slot.payload.load(std::memory_order_acquire))
                payload->Quarantine();
        }
        ReleaseDrain(slot, generation);
    }

    // Closing must never steal the drain credit from an already scheduled
    // Process: it may have crossed the backend gate before the close
    // linearized, and Drain must wait for that cleanup to finish.  This CAS
    // claims only an unscheduled retired payload.
    void QuarantineUnscheduled(Slot& slot, uint32_t generation) noexcept {
        uint64_t old = slot.state.load(std::memory_order_acquire);
        for (;;) {
            if (_Generation(old) != generation) return;
            uint32_t const flags = _Flags(old);
            if (!(flags & Retired) || (flags & (Scheduled | Quarantined))) return;
            if (slot.state.compare_exchange_weak(
                    old, _Pack(generation, flags | Quarantined),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                ReleaseDrain(slot, generation);
                return;
            }
        }
    }

    void Process(size_t index, uint32_t generation) noexcept {
        _retirementCleanupActive = true;
        struct Reset { ~Reset() { _retirementCleanupActive = false; } } reset;
        Slot& slot = *slots[index];
        uint64_t state = slot.state.load(std::memory_order_acquire);
        uint32_t flags = _Flags(state);
        if (_Generation(state) != generation ||
            (flags & (Reserved | Retired | Scheduled)) !=
                (Reserved | Retired | Scheduled) ||
            (flags & Quarantined))
            return;

        // A Process which observes close before crossing this gate performs no
        // payload virtual call. A pre-close observer retains its Drain credit,
        // so QuiesceBackend cannot return until that cleanup has finished.
        if (backendQuiesced.load(std::memory_order_acquire) ||
            !_BackendIsOpen(backend)) {
            Quarantine(slot, generation, false);
            return;
        }
        auto* payload = slot.payload.load(std::memory_order_acquire);
        if ((flags & Failure) || !payload || !payload->IsComplete()) {
            Quarantine(slot, generation, true);
            return;
        }
        if (!payload->Destroy()) {
            Quarantine(slot, generation, true);
            return;
        }

        delete slot.payload.exchange(nullptr, std::memory_order_acq_rel);
        bool const releaseDrain = ClaimDrain(slot, generation);
        // Advance before exposing the empty slot. A late signal for this
        // ticket can no longer mark a free slot as signalled.
        if (generation == UINT32_MAX) {
            slot.state.store(_Pack(generation, Spent), std::memory_order_release);
        } else {
            slot.state.store(_Pack(generation + 1, 0), std::memory_order_release);
        }
        // Capacity is visible before a Drain waiter is released.
        if (releaseDrain) drain.Release();
    }

    bool ClaimAndDispatch(size_t index, uint32_t generation) noexcept {
        Slot& slot = *slots[index];
        if (backendQuiesced.load(std::memory_order_acquire) ||
            !_BackendIsOpen(backend)) {
            QuarantineUnscheduled(slot, generation);
            return true;
        }
        uint64_t old = slot.state.load(std::memory_order_acquire);
        for (;;) {
            if (_Generation(old) != generation) return false;
            uint32_t flags = _Flags(old);
            if ((flags & (Reserved | Retired)) != (Reserved | Retired) ||
                !(flags & (Success | Failure)) ||
                (flags & (Scheduled | Quarantined)))
                return true;
            if (slot.state.compare_exchange_weak(
                    old, _Pack(generation, flags | Scheduled),
                    std::memory_order_acq_rel, std::memory_order_acquire))
                break;
        }
        // Services are process-lifetime (the registry never drops one), so
        // the lane holds the slot by raw pointer.
        _LaneDispatch(&slot, generation);
        return true;
    }

    void Signal(size_t index, uint32_t generation, bool success) noexcept {
        if (index >= slots.size()) return;
        Slot& slot = *slots[index];
        uint64_t old = slot.state.load(std::memory_order_acquire);
        for (;;) {
            if (_Generation(old) != generation) return;
            uint32_t flags = _Flags(old);
            // The first terminal signal is authoritative. A duplicate or an
            // opposite late status cannot retroactively change cleanup.
            if (!(flags & Reserved) ||
                (flags & (Success | Failure | Quarantined | Spent))) return;
            uint32_t nextFlags = flags | (success ? Success : Failure);
            if (slot.state.compare_exchange_weak(
                    old, _Pack(generation, nextFlags),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                if ((nextFlags & (Reserved | Retired)) == (Reserved | Retired))
                    ClaimAndDispatch(index, generation);
                return;
            }
        }
    }

    // The retirement lane (see _LanePass).
    static std::atomic<Slot*> laneHead;
    static UsdGenSerialLane lane;
    static void _LaneDispatch(Slot*, uint32_t) noexcept;
    static void _LanePass(void*) noexcept;

    _DrainCount drain;
    std::vector<std::unique_ptr<Slot>> slots;
    std::atomic<bool> accepting{true};
    std::atomic<bool> backendQuiesced{false};
    UsdGenBackendId const backend;
};

// The retirement lane runs backend cleanup off the signalling thread (native
// completion callbacks must not free native storage inline). Producers push
// slots on a Treiber stack and kick a UsdGenSerialLane on the process
// scheduler, so at most one pass runs and nothing is held while idle. Both
// are constant-initialized and never destroyed: late callbacks at exit still
// find them. Cleanup occupies one scheduler worker while it runs.
std::atomic<UsdGenExecutionRetirementSignal::State::Slot*>
    UsdGenExecutionRetirementSignal::State::laneHead{nullptr};
UsdGenSerialLane UsdGenExecutionRetirementSignal::State::lane{
    &UsdGenExecutionRetirementSignal::State::_LanePass, nullptr};

void UsdGenExecutionRetirementSignal::State::_LanePass(void*) noexcept {
    Slot* list = laneHead.exchange(nullptr, std::memory_order_acq_rel);
    Slot* fifo = nullptr;  // pushed LIFO; run in dispatch order
    while (list) {
        Slot* next = list->laneNext;
        list->laneNext = fifo;
        fifo = list;
        list = next;
    }
    while (fifo) {
        Slot* next = fifo->laneNext;  // Process may requeue the slot
        fifo->owner->Process(fifo->index, fifo->laneGeneration);
        fifo = next;
    }
}

void UsdGenExecutionRetirementSignal::State::_LaneDispatch(Slot* slot, uint32_t generation) noexcept {
    slot->laneGeneration = generation;
    Slot* h = laneHead.load(std::memory_order_relaxed);
    do { slot->laneNext = h; }
    while (!laneHead.compare_exchange_weak(h, slot, std::memory_order_release,
                                           std::memory_order_relaxed));
    lane.Kick();
}

struct UsdGenExecutionRetirementService::Impl
    : UsdGenExecutionRetirementSignal::State {
    using UsdGenExecutionRetirementSignal::State::State;
};

UsdGenExecutionRetirementSignal::UsdGenExecutionRetirementSignal(
    std::shared_ptr<State> state, size_t slot, uint64_t generation) noexcept
    : state_(std::move(state)), slot_(slot), generation_(generation) {}
void UsdGenExecutionRetirementSignal::SignalSuccess() const noexcept {
    if (state_) state_->Signal(slot_, uint32_t(generation_), true);
}
void UsdGenExecutionRetirementSignal::SignalFailure() const noexcept {
    if (state_) state_->Signal(slot_, uint32_t(generation_), false);
}

UsdGenExecutionRetirementTicket::UsdGenExecutionRetirementTicket(
    std::shared_ptr<UsdGenExecutionRetirementSignal::State> state, size_t slot,
    uint64_t generation) noexcept
    : state_(std::move(state)), slot_(slot), generation_(generation) {}
UsdGenExecutionRetirementTicket::~UsdGenExecutionRetirementTicket() { Reset(); }
UsdGenExecutionRetirementTicket::UsdGenExecutionRetirementTicket(
    UsdGenExecutionRetirementTicket&& other) noexcept
    : state_(std::move(other.state_)), slot_(other.slot_), generation_(other.generation_) {}
UsdGenExecutionRetirementTicket& UsdGenExecutionRetirementTicket::operator=(
    UsdGenExecutionRetirementTicket&& other) noexcept {
    if (this != &other) {
        Reset(); state_ = std::move(other.state_); slot_ = other.slot_; generation_ = other.generation_;
    }
    return *this;
}
void UsdGenExecutionRetirementTicket::Reset() noexcept {
    if (!state_ || slot_ >= state_->slots.size()) { state_.reset(); return; }
    auto& slot = *state_->slots[slot_];
    uint64_t old = slot.state.load(std::memory_order_acquire);
    while (_Generation(old) == uint32_t(generation_) && !(_Flags(old) & Retired)) {
        uint32_t const nextGeneration = uint32_t(generation_) == UINT32_MAX
            ? uint32_t(generation_) : uint32_t(generation_) + 1;
        uint32_t const nextFlags = uint32_t(generation_) == UINT32_MAX
            ? uint32_t(Spent) : uint32_t{0};
        if (slot.state.compare_exchange_weak(old, _Pack(nextGeneration, nextFlags),
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) break;
    }
    state_.reset();
}
UsdGenExecutionRetirementSignal UsdGenExecutionRetirementTicket::MakeSignal() const noexcept {
    return {state_, slot_, generation_};
}
bool UsdGenExecutionRetirementTicket::Retire(
    std::unique_ptr<UsdGenExecutionRetirementPayload> payload) noexcept {
    if (!payload || !state_ || slot_ >= state_->slots.size()) return false;
    auto state = std::move(state_);
    auto& slot = *state->slots[slot_];
    uint32_t const generation = uint32_t(generation_);

    // Reserve before exposing the installing state. A concurrent backend
    // quiesce can then safely wait for this installation gap without a
    // polling protocol; losing the claim releases the unused credit below.
    state->drain.Reserve();
    _GateHolder const* holder = _installTestGate.load(std::memory_order_acquire);
    if (auto gate = holder ? holder->gate : nullptr;
        gate && !gate->claimed.exchange(true, std::memory_order_acq_rel)) {
        gate->entered.store(true, std::memory_order_release);
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!gate->release.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() >= deadline) {
                gate->timedOut.store(true, std::memory_order_release);
                break;
            }
            std::this_thread::yield();
        }
    }
    // Claim the one ticket-owned installation phase before exposing the raw
    // payload. Signal may race and set its bit, but cannot dispatch before
    // Retired is published below.
    uint64_t old = slot.state.load(std::memory_order_acquire);
    for (;;) {
        if (_Generation(old) != generation || !(_Flags(old) & Reserved) ||
            (_Flags(old) & (Installing | Retired))) {
            state->drain.Release();
            return false;
        }
        if (slot.state.compare_exchange_weak(old, _Pack(generation, _Flags(old) | Installing),
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) break;
    }
    slot.payload.store(payload.release(), std::memory_order_release);

    // Preserve a concurrently recorded signal and retry CAS; do not reserve a
    // second wait credit or turn a successful signal into quarantine.
    old = slot.state.load(std::memory_order_acquire);
    for (;;) {
        uint32_t flags = _Flags(old);
        if (_Generation(old) != generation || !(flags & Installing)) {
            // Impossible for a valid ticket; preserve the payload forever.
            state->Quarantine(slot, generation, false);
            return false;
        }
        uint32_t next = (flags | Retired) & ~Installing;
        if (slot.state.compare_exchange_weak(old, _Pack(generation, next),
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
            if (state->backendQuiesced.load(std::memory_order_acquire) ||
                !_BackendIsOpen(state->backend))
                state->QuarantineUnscheduled(slot, generation);
            else if (next & (Success | Failure)) state->ClaimAndDispatch(slot_, generation);
            return true;
        }
    }
}

UsdGenExecutionRetirementService::UsdGenExecutionRetirementService(
    size_t capacity, UsdGenBackendId backend)
    : impl_(std::make_shared<Impl>(capacity, backend)) {}
UsdGenExecutionRetirementService::~UsdGenExecutionRetirementService() = default;
std::optional<UsdGenExecutionRetirementTicket>
UsdGenExecutionRetirementService::TryReserve() noexcept {
    auto state = impl_;
    if (!state || !state->accepting.load(std::memory_order_acquire) ||
        state->backendQuiesced.load(std::memory_order_acquire) ||
        !_BackendIsOpen(state->backend)) return {};
    for (size_t i = 0; i != state->slots.size(); ++i) {
        auto& slot = *state->slots[i];
        uint64_t old = slot.state.load(std::memory_order_acquire);
        for (;;) {
            if (_Flags(old) != 0 || _Generation(old) == UINT32_MAX) break;
            uint32_t generation = _Generation(old) + 1;
            if (slot.state.compare_exchange_weak(old, _Pack(generation, Reserved),
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
                // A close which linearized before this successful recheck wins.
                if (state->accepting.load(std::memory_order_acquire) &&
                    !state->backendQuiesced.load(std::memory_order_acquire) &&
                    _BackendIsOpen(state->backend))
                    return UsdGenExecutionRetirementTicket(state, i, generation);
                uint64_t expected = _Pack(generation, Reserved);
                slot.state.compare_exchange_strong(expected, _Pack(generation, 0),
                                                   std::memory_order_acq_rel);
                return {};
            }
        }
    }
    return {};
}

void UsdGenExecutionRetirementService::Shutdown() noexcept {
    if (impl_) impl_->accepting.store(false, std::memory_order_release);
}
void UsdGenExecutionRetirementService::QuiesceBackend() {
    auto state = impl_;
    if (!state) return;
    if (_retirementCleanupActive) std::terminate();
    state->accepting.store(false, std::memory_order_release);
    state->backendQuiesced.store(true, std::memory_order_release);
    // Only unscheduled retired slots are claimed here. Scheduled work owns
    // its Retire reserve_wait until Process observes the close or completes.
    for (auto const& entry : state->slots) {
        auto& slot = *entry;
        uint64_t const observed = slot.state.load(std::memory_order_acquire);
        state->QuarantineUnscheduled(slot, _Generation(observed));
    }
    if (!UsdGenProcessShutdownInProgress()) state->drain.WaitForAll();
}
bool UsdGenExecutionRetirementService::BackendQuiesced() const noexcept {
    auto state = impl_;
    return state && state->backendQuiesced.load(std::memory_order_acquire);
}
void UsdGenExecutionRetirementService::Drain() {
    if (_retirementCleanupActive) std::terminate();
    if (impl_ && !UsdGenProcessShutdownInProgress()) impl_->drain.WaitForAll();
}

namespace {
// Process-lifetime registry: late backend callbacks retain their payload state
// through it. A fixed open-addressing table, slots claimed by CAS, entries
// never freed; constant-initialized and trivially destructible.
struct _ServiceEntry {
    UsdGenExecutionResourceDevice key;
    std::shared_ptr<UsdGenExecutionRetirementService> service;
};
constexpr size_t _kServiceSlots = 256;
std::atomic<_ServiceEntry*> _services[_kServiceSlots] = {};

size_t _ServiceSlot(UsdGenExecutionResourceDevice const& key) noexcept {
    uint64_t h = (uint64_t(uint32_t(key.device)) << 8) ^ uint64_t(key.backend);
    h *= 0x9e3779b97f4a7c15ull;
    return size_t(h >> 56) % _kServiceSlots;
}
bool _SameDevice(UsdGenExecutionResourceDevice const& a,
                 UsdGenExecutionResourceDevice const& b) noexcept {
    return a.backend == b.backend && a.device == b.device;
}
_ServiceEntry* _FindService(UsdGenExecutionResourceDevice const& key) noexcept {
    for (size_t i = 0, s = _ServiceSlot(key); i != _kServiceSlots;
         ++i, s = (s + 1) % _kServiceSlots) {
        _ServiceEntry* entry = _services[s].load(std::memory_order_acquire);
        if (!entry) return nullptr;
        if (_SameDevice(entry->key, key)) return entry;
    }
    return nullptr;
}
// The resident entry for the candidate's key (the candidate if it won), or
// null when the table is full.
_ServiceEntry* _InsertService(_ServiceEntry* candidate) noexcept {
    for (size_t i = 0, s = _ServiceSlot(candidate->key); i != _kServiceSlots;
         ++i, s = (s + 1) % _kServiceSlots) {
        _ServiceEntry* entry = _services[s].load(std::memory_order_acquire);
        if (!entry && _services[s].compare_exchange_strong(
                entry, candidate, std::memory_order_acq_rel, std::memory_order_acquire))
            return candidate;
        if (_SameDevice(entry->key, candidate->key)) return entry;
    }
    return nullptr;
}
}
std::shared_ptr<UsdGenExecutionRetirementService> GetOrCreateUsdGenExecutionRetirementService(
    UsdGenExecutionResourceDevice key, UsdGenExecutionRetirementConfig config,
    bool requireMatchingConfig) noexcept {
    try {
        if (key.device < 0 || config.capacity == 0 || !_BackendIsOpen(key.backend)) return {};
        if (_ServiceEntry const* found = _FindService(key)) {
            if (requireMatchingConfig && found->service->impl_->slots.size() != config.capacity) return {};
            return _BackendIsOpen(key.backend) ? found->service : std::shared_ptr<UsdGenExecutionRetirementService>{};
        }
        auto* candidate = new _ServiceEntry{key, std::shared_ptr<UsdGenExecutionRetirementService>(
            new UsdGenExecutionRetirementService(config.capacity, key.backend))};
        _ServiceEntry* resident = _InsertService(candidate);
        if (resident != candidate) delete candidate;  // lost the race, or full
        if (!resident) return {};
        if (resident != candidate && requireMatchingConfig &&
            resident->service->impl_->slots.size() != config.capacity) return {};
        if (!_BackendIsOpen(key.backend)) {
            // This raced a backend-wide close. No ticket escaped this call,
            // and the shared backend flag prevents Process dispatch; do not
            // add a second concurrent drain wait to the lifecycle closer.
            resident->service->Shutdown();
            return {};
        }
        return resident->service;
    } catch (...) { return {}; }
}
std::shared_ptr<UsdGenExecutionRetirementService> FindUsdGenExecutionRetirementService(
    UsdGenExecutionResourceDevice key) noexcept {
    _ServiceEntry const* found = _FindService(key);
    return found ? found->service : nullptr;
}
bool IsUsdGenExecutionRetirementBackendOpen(
    UsdGenBackendId backend) noexcept {
    return _BackendIsOpen(backend);
}
void QuiesceUsdGenExecutionRetirementBackend(
    UsdGenBackendId backend) noexcept {
    size_t const index = _BackendIndex(backend);
    if (index >= kUsdGenMaxBackends) return;
    _backendClosed[index].store(true, std::memory_order_release);
    // The registry is process-lifetime, so its service identities remain
    // stable while late static owners transfer payloads into fixed slots.
    try {
        for (auto const& slot : _services)
            if (_ServiceEntry const* entry = slot.load(std::memory_order_acquire))
                if (entry->key.backend == backend) entry->service->QuiesceBackend();
    } catch (...) { std::terminate(); }
}
void SetUsdGenExecutionRetirementInstallTestGate(
    std::shared_ptr<UsdGenExecutionRetirementInstallTestGate> gate) noexcept {
    // Leaked by design (see _GateHolder); clearing publishes null.
    _GateHolder const* holder = gate ? new (std::nothrow) _GateHolder{std::move(gate)} : nullptr;
    _installTestGate.store(holder, std::memory_order_release);
}
} // namespace usdGen
