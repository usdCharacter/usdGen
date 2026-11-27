#include "usdGen/api/ext/device/executionResources.h"

#include <atomic>
#include <limits>
#include <utility>

namespace usdGen {
namespace {
constexpr size_t KindIndex(UsdGenExecutionResourceKind kind) noexcept {
    return static_cast<size_t>(kind);
}
constexpr size_t KindCount = KindIndex(UsdGenExecutionResourceKind::Count);
constexpr bool IsAllocationKind(UsdGenExecutionResourceKind kind) noexcept {
    return KindIndex(kind) < KindIndex(UsdGenExecutionResourceKind::Pending);
}
// Process-lifetime device registry: a fixed open-addressing table whose slots
// are claimed by CAS and never freed. Device allocations can outlive shutdown
// while their contexts are being torn down, and a quarantined charge must not
// vanish. Constant-initialized and trivially destructible: nothing runs at
// DLL detach.
struct PoolEntry {
    UsdGenExecutionResourceDevice key;
    std::shared_ptr<UsdGenExecutionResourcePool> pool;
};
constexpr size_t kPoolSlots = 256;
std::atomic<PoolEntry*> gPools[kPoolSlots] = {};

size_t DeviceSlot(UsdGenExecutionResourceDevice const& key) noexcept {
    uint64_t h = (uint64_t(uint32_t(key.device)) << 8) ^ uint64_t(key.backend);
    h *= 0x9e3779b97f4a7c15ull;
    return size_t(h >> 56) % kPoolSlots;
}
bool SameDevice(UsdGenExecutionResourceDevice const& a,
                UsdGenExecutionResourceDevice const& b) noexcept {
    return a.backend == b.backend && a.device == b.device;
}
PoolEntry* FindPool(UsdGenExecutionResourceDevice const& key) noexcept {
    for (size_t i = 0, s = DeviceSlot(key); i != kPoolSlots; ++i, s = (s + 1) % kPoolSlots) {
        PoolEntry* entry = gPools[s].load(std::memory_order_acquire);
        if (!entry) return nullptr;
        if (SameDevice(entry->key, key)) return entry;
    }
    return nullptr;
}
// Returns the resident entry for the candidate's key: the candidate if it
// won, else the winner. Null when the table is full.
PoolEntry* InsertPool(PoolEntry* candidate) noexcept {
    for (size_t i = 0, s = DeviceSlot(candidate->key); i != kPoolSlots;
         ++i, s = (s + 1) % kPoolSlots) {
        PoolEntry* entry = gPools[s].load(std::memory_order_acquire);
        if (!entry && gPools[s].compare_exchange_strong(
                entry, candidate, std::memory_order_acq_rel, std::memory_order_acquire))
            return candidate;
        if (SameDevice(entry->key, candidate->key)) return entry;
    }
    return nullptr;
}
}

struct UsdGenExecutionResourcePermit::State {
    State(size_t limit, size_t headroom) : limitBytes(limit), headroomBytes(headroom) {}
    size_t const limitBytes;
    size_t const headroomBytes;
    size_t const usableBytes = limitBytes - headroomBytes;
    std::atomic<size_t> usedBytes{0};
    std::array<std::atomic<size_t>, KindCount> byKind{};
};

struct UsdGenExecutionMemoryReservationState {
    UsdGenExecutionMemoryReservationState(
        std::shared_ptr<UsdGenExecutionResourcePermit::State> pool,
        size_t initialBytes) noexcept
        : poolState(std::move(pool)), word(initialBytes) {}

    // One word makes Consume and Release one conservation transaction:
    // remaining pending bytes plus transferred permits always equal the
    // original pool charge until a permit or the reservation is released.
    // The top bit marks the reservation released; the rest is the balance.
    static constexpr size_t kReleased = size_t(1) << (sizeof(size_t) * 8 - 1);
    std::shared_ptr<UsdGenExecutionResourcePermit::State> poolState;
    std::atomic<size_t> word{0};
};

UsdGenExecutionResourcePermit::UsdGenExecutionResourcePermit(
    std::shared_ptr<State> state, size_t bytes, UsdGenExecutionResourceKind kind,
    std::shared_ptr<UsdGenExecutionMemoryReservationState> reservationCredit) noexcept
    : state_(std::move(state)), reservationCredit_(std::move(reservationCredit)),
      bytes_(bytes), kind_(kind) {}
UsdGenExecutionResourcePermit::~UsdGenExecutionResourcePermit() { Release(); }
UsdGenExecutionResourcePermit::UsdGenExecutionResourcePermit(
    UsdGenExecutionResourcePermit&& other) noexcept
    : state_(std::move(other.state_)),
      reservationCredit_(std::move(other.reservationCredit_)),
      bytes_(other.bytes_), kind_(other.kind_) {
    other.bytes_ = 0;
}
UsdGenExecutionResourcePermit& UsdGenExecutionResourcePermit::operator=(
    UsdGenExecutionResourcePermit&& other) noexcept {
    if (this != &other) {
        Release();
        state_ = std::move(other.state_);
        reservationCredit_ = std::move(other.reservationCredit_);
        bytes_ = other.bytes_;
        kind_ = other.kind_;
        other.bytes_ = 0;
    }
    return *this;
}
void UsdGenExecutionResourcePermit::Reclassify(UsdGenExecutionResourceKind kind) noexcept {
    // Pending is reservation-owned ledger state, never an allocation permit
    // lifetime class.
    if (!state_ || !IsAllocationKind(kind) || kind == kind_) return;
    state_->byKind[KindIndex(kind_)].fetch_sub(bytes_, std::memory_order_relaxed);
    state_->byKind[KindIndex(kind)].fetch_add(bytes_, std::memory_order_relaxed);
    kind_ = kind;
}
void UsdGenExecutionResourcePermit::Release() noexcept {
    if (!state_) return;
    // The reservation word linearizes child release against reservation
    // closure. A live reservation recovers the credit; a closed one lets the
    // child release its still-exact global charge normally. Pending is
    // credited before the balance so Pending never drops below the balance.
    if (auto const& credit = reservationCredit_) {
        using R = UsdGenExecutionMemoryReservationState;
        auto& pending = state_->byKind[KindIndex(UsdGenExecutionResourceKind::Pending)];
        state_->byKind[KindIndex(kind_)].fetch_sub(bytes_, std::memory_order_relaxed);
        pending.fetch_add(bytes_, std::memory_order_relaxed);
        size_t word = credit->word.load(std::memory_order_acquire);
        while (!(word & R::kReleased)) {
            if (credit->word.compare_exchange_weak(word, word + bytes_,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
                reservationCredit_.reset();
                state_.reset();
                bytes_ = 0;
                return;
            }
        }
        pending.fetch_sub(bytes_, std::memory_order_relaxed);
        state_->byKind[KindIndex(kind_)].fetch_add(bytes_, std::memory_order_relaxed);
    }
    state_->byKind[KindIndex(kind_)].fetch_sub(bytes_, std::memory_order_relaxed);
    state_->usedBytes.fetch_sub(bytes_, std::memory_order_release);
    reservationCredit_.reset();
    state_.reset();
    bytes_ = 0;
}
void UsdGenExecutionResourcePermit::Abandon() noexcept {
    // Counters intentionally remain charged. The registry keeps State alive.
    // In particular, a reserved child must not recycle credit after its
    // backend lifetime becomes unproven.
    reservationCredit_.reset();
    state_.reset();
    bytes_ = 0;
}

UsdGenExecutionMemoryReservation::UsdGenExecutionMemoryReservation(
    std::shared_ptr<UsdGenExecutionMemoryReservationState> state) noexcept
    : state_(std::move(state)) {}
UsdGenExecutionMemoryReservation::~UsdGenExecutionMemoryReservation() { Release(); }
UsdGenExecutionMemoryReservation::UsdGenExecutionMemoryReservation(
    UsdGenExecutionMemoryReservation&& other) noexcept
    : state_(std::move(other.state_)) {}
UsdGenExecutionMemoryReservation& UsdGenExecutionMemoryReservation::operator=(
    UsdGenExecutionMemoryReservation&& other) noexcept {
    if (this != &other) {
        Release();
        state_ = std::move(other.state_);
    }
    return *this;
}
UsdGenExecutionMemoryReservation::operator bool() const noexcept {
    auto const& state = state_;
    return state && !(state->word.load(std::memory_order_acquire) &
                      UsdGenExecutionMemoryReservationState::kReleased);
}
size_t UsdGenExecutionMemoryReservation::RemainingBytes() const noexcept {
    auto const& state = state_;
    if (!state) return 0;
    size_t const word = state->word.load(std::memory_order_acquire);
    return (word & UsdGenExecutionMemoryReservationState::kReleased) ? 0 : word;
}
UsdGenExecutionResourceKind UsdGenExecutionMemoryReservation::PendingKind() const noexcept {
    return UsdGenExecutionResourceKind::Pending;
}
std::optional<UsdGenExecutionResourcePermit>
UsdGenExecutionMemoryReservation::Consume(
    size_t bytes, UsdGenExecutionResourceKind targetKind) noexcept {
    auto const& state = state_;
    if (!state || !IsAllocationKind(targetKind)) return std::nullopt;
    auto const& pool = state->poolState;
    if (!pool) return std::nullopt;
    // A CAS loop that never takes the balance below zero.
    size_t word = state->word.load(std::memory_order_acquire);
    for (;;) {
        if ((word & UsdGenExecutionMemoryReservationState::kReleased) || bytes > word)
            return std::nullopt;
        if (state->word.compare_exchange_weak(word, word - bytes,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire))
            break;
    }
    // Snapshot category totals may transiently observe one side of this
    // transfer, but usedBytes is intentionally untouched throughout.
    pool->byKind[KindIndex(UsdGenExecutionResourceKind::Pending)].fetch_sub(
        bytes, std::memory_order_relaxed);
    pool->byKind[KindIndex(targetKind)].fetch_add(
        bytes, std::memory_order_relaxed);
    return UsdGenExecutionResourcePermit(pool, bytes, targetKind, state);
}
void UsdGenExecutionMemoryReservation::Release() noexcept {
    auto const& state = state_;
    if (!state) return;
    size_t const remaining = state->word.exchange(
        UsdGenExecutionMemoryReservationState::kReleased, std::memory_order_acq_rel);
    if (remaining & UsdGenExecutionMemoryReservationState::kReleased) return;
    auto const& pool = state->poolState;
    if (!pool) return;
    pool->byKind[KindIndex(UsdGenExecutionResourceKind::Pending)].fetch_sub(
        remaining, std::memory_order_relaxed);
    pool->usedBytes.fetch_sub(remaining, std::memory_order_release);
}

UsdGenExecutionResourcePool::UsdGenExecutionResourcePool(size_t limitBytes,
                                                           size_t headroomBytes)
    : state_(std::make_shared<UsdGenExecutionResourcePermit::State>(
          limitBytes, headroomBytes > limitBytes ? limitBytes : headroomBytes)) {}
UsdGenExecutionResourcePool::~UsdGenExecutionResourcePool() = default;
std::optional<UsdGenExecutionResourcePermit>
UsdGenExecutionResourcePool::TryReserve(size_t bytes,
                                        UsdGenExecutionResourceKind kind) const noexcept {
    auto const& state = state_;
    if (!state || !IsAllocationKind(kind) || bytes > state->usableBytes) return std::nullopt;
    size_t used = state->usedBytes.load(std::memory_order_acquire);
    for (;;) {
        // `used <= usable` is an invariant. The subtraction form cannot
        // overflow and makes a malformed/corrupt counter fail closed.
        if (used > state->usableBytes || bytes > state->usableBytes - used)
            return std::nullopt;
        if (state->usedBytes.compare_exchange_weak(used, used + bytes,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
            state->byKind[KindIndex(kind)].fetch_add(bytes, std::memory_order_relaxed);
            return UsdGenExecutionResourcePermit(state, bytes, kind);
        }
    }
}
std::optional<UsdGenExecutionMemoryReservation>
UsdGenExecutionResourcePool::TryReserveMemory(
    size_t bytes) const noexcept {
    auto const& state = state_;
    if (!state || bytes > state->usableBytes)
        return std::nullopt;
    std::shared_ptr<UsdGenExecutionMemoryReservationState> reservation;
    try {
        reservation = std::make_shared<UsdGenExecutionMemoryReservationState>(
            state, bytes);
    } catch (...) {
        return std::nullopt;
    }
    size_t used = state->usedBytes.load(std::memory_order_acquire);
    for (;;) {
        if (used > state->usableBytes || bytes > state->usableBytes - used)
            return std::nullopt;
        if (state->usedBytes.compare_exchange_weak(used, used + bytes,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
            state->byKind[KindIndex(UsdGenExecutionResourceKind::Pending)].fetch_add(
                bytes, std::memory_order_relaxed);
            return UsdGenExecutionMemoryReservation(std::move(reservation));
        }
    }
}
UsdGenExecutionResourceSnapshot UsdGenExecutionResourcePool::Snapshot() const noexcept {
    UsdGenExecutionResourceSnapshot result;
    auto const& state = state_;
    if (!state) return result;
    result.limitBytes = state->limitBytes;
    result.headroomBytes = state->headroomBytes;
    result.usableBytes = state->usableBytes;
    result.usedBytes = state->usedBytes.load(std::memory_order_acquire);
    for (size_t i = 0; i != KindCount; ++i)
        result.byKind[i] = state->byKind[i].load(std::memory_order_acquire);
    return result;
}
std::shared_ptr<UsdGenExecutionResourcePool> GetOrCreateUsdGenExecutionResourcePool(
    UsdGenExecutionResourceDevice key, UsdGenExecutionResourceConfig config,
    bool requireMatchingConfig) noexcept {
    try {
        if (key.device < 0) return {};
        auto matches = [&](PoolEntry const* entry) {
            if (!requireMatchingConfig) return true;
            auto const snapshot = entry->pool->Snapshot();
            return snapshot.limitBytes == config.limitBytes &&
                snapshot.headroomBytes == (config.headroomBytes > config.limitBytes ?
                                           config.limitBytes : config.headroomBytes);
        };
        if (PoolEntry const* found = FindPool(key))
            return matches(found) ? found->pool : nullptr;
        auto* candidate = new PoolEntry{key, std::make_shared<UsdGenExecutionResourcePool>(
            config.limitBytes, config.headroomBytes)};
        PoolEntry* resident = InsertPool(candidate);
        if (resident != candidate) delete candidate;  // lost the race, or full
        if (!resident || !matches(resident)) return {};
        return resident->pool;
    } catch (...) {
        return {};
    }
}
std::shared_ptr<UsdGenExecutionResourcePool> FindUsdGenExecutionResourcePool(
    UsdGenExecutionResourceDevice key) noexcept {
    PoolEntry const* found = FindPool(key);
    return found ? found->pool : nullptr;
}

} // namespace usdGen
