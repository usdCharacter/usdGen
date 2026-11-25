// Backend-neutral byte admission for GPU execution resources.
#ifndef USDGEN_EXECUTION_RESOURCES_H
#define USDGEN_EXECUTION_RESOURCES_H

#include "usdGen/executionResourcesExport.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace usdGen {

// Categories are accounting/lifetime classes, not allocator hints.  Their
// aggregate is always bounded by the same device budget.
enum class UsdGenExecutionResourceKind : uint8_t {
    Active,  // allocation owned by currently executing work
    Pinned,  // published allocation retained by a generation or lease
    Cache,   // reusable owner-local state such as an RBF binding
    Scratch, // temporary workspace which must not outlive its operation
    // Reservation-only ledger balance. Existing allocation category indices
    // remain stable; permits may never be issued in this category.
    Pending,
    Count
};

struct USDGEN_EXECUTION_RESOURCES_API UsdGenExecutionResourceSnapshot {
    size_t limitBytes = 0;
    size_t headroomBytes = 0;
    size_t usableBytes = 0;
    size_t usedBytes = 0;
    std::array<size_t, static_cast<size_t>(UsdGenExecutionResourceKind::Count)>
        byKind{};
};

class UsdGenExecutionResourcePool;
class UsdGenExecutionMemoryReservation;
struct UsdGenExecutionMemoryReservationState;

// Move-only allocation charge. It must be retained by the allocation itself,
// not by the operation which happened to allocate it.  A quarantined CUDA
// allocation deliberately retains its permit because its storage cannot be
// proven free.
class USDGEN_EXECUTION_RESOURCES_API UsdGenExecutionResourcePermit {
public:
    UsdGenExecutionResourcePermit() = default;
    ~UsdGenExecutionResourcePermit();
    UsdGenExecutionResourcePermit(UsdGenExecutionResourcePermit const&) = delete;
    UsdGenExecutionResourcePermit& operator=(UsdGenExecutionResourcePermit const&) = delete;
    UsdGenExecutionResourcePermit(UsdGenExecutionResourcePermit&&) noexcept;
    UsdGenExecutionResourcePermit& operator=(UsdGenExecutionResourcePermit&&) noexcept;

    explicit operator bool() const noexcept { return static_cast<bool>(state_); }
    size_t Bytes() const noexcept { return bytes_; }
    UsdGenExecutionResourceKind Kind() const noexcept { return kind_; }
    // Keeps the charge while changing its lifetime classification. Call only
    // from the permit owner's exclusive publication/teardown phase; this is
    // sequentially idempotent but is not concurrent with Release or Abandon.
    void Reclassify(UsdGenExecutionResourceKind) noexcept;
    // Idempotent.  Call only after the backend has actually released storage.
    void Release() noexcept;
    // For device loss/unproven backend free: keep this byte charge permanently
    // conservative while dropping the permit handle. The pool must be a
    // process-lifetime device-registry pool; standalone pools are unsuitable
    // for quarantined backend allocations.
    void Abandon() noexcept;

private:
    struct State;
    std::shared_ptr<State> state_;
    // Present only for a permit consumed from a live memory reservation. It
    // keeps that reservation's accounting state alive without owning the
    // reservation handle or creating a back-reference cycle.
    std::shared_ptr<UsdGenExecutionMemoryReservationState> reservationCredit_;
    size_t bytes_ = 0;
    UsdGenExecutionResourceKind kind_ = UsdGenExecutionResourceKind::Active;
    UsdGenExecutionResourcePermit(std::shared_ptr<State>, size_t,
                                  UsdGenExecutionResourceKind,
                                  std::shared_ptr<UsdGenExecutionMemoryReservationState> = {}) noexcept;
    friend class UsdGenExecutionResourcePool;
    friend class UsdGenExecutionMemoryReservation;
    friend struct UsdGenExecutionMemoryReservationState;
};

// Move-only precharged memory balance. Its bytes are already included in the
// pool total, so Consume transfers only category ownership (Pending to
// target permit); a child permit returns that credit to Pending when released
// while this reservation remains live. It never competes with another job.
class USDGEN_EXECUTION_RESOURCES_API UsdGenExecutionMemoryReservation {
public:
    UsdGenExecutionMemoryReservation() = default;
    ~UsdGenExecutionMemoryReservation();
    UsdGenExecutionMemoryReservation(UsdGenExecutionMemoryReservation const&) = delete;
    UsdGenExecutionMemoryReservation& operator=(UsdGenExecutionMemoryReservation const&) = delete;
    UsdGenExecutionMemoryReservation(UsdGenExecutionMemoryReservation&&) noexcept;
    UsdGenExecutionMemoryReservation& operator=(UsdGenExecutionMemoryReservation&&) noexcept;

    explicit operator bool() const noexcept;
    // The unconsumed part of this job's precharged balance. Zero is a valid,
    // deterministic reservation amount rather than an "unknown" sentinel.
    size_t RemainingBytes() const noexcept;
    UsdGenExecutionResourceKind PendingKind() const noexcept;
    // Atomically withdraws from this reservation only. On success the result
    // is a permit whose Release returns credit to this live reservation;
    // usedBytes remains unchanged while the pending category moves to target.
    std::optional<UsdGenExecutionResourcePermit> Consume(
        size_t bytes, UsdGenExecutionResourceKind targetKind) noexcept;
    // Idempotent: returns exactly the still-unconsumed charged balance.
    void Release() noexcept;

private:
    std::shared_ptr<UsdGenExecutionMemoryReservationState> state_;
    explicit UsdGenExecutionMemoryReservation(
        std::shared_ptr<UsdGenExecutionMemoryReservationState>) noexcept;
    friend class UsdGenExecutionResourcePool;
};

// A pool has immutable capacity/headroom. Backend adapters obtain its one
// process-wide instance per physical device from the registry below.
class USDGEN_EXECUTION_RESOURCES_API UsdGenExecutionResourcePool {
public:
    UsdGenExecutionResourcePool(size_t limitBytes, size_t headroomBytes = 0);
    ~UsdGenExecutionResourcePool();
    UsdGenExecutionResourcePool(UsdGenExecutionResourcePool const&) = delete;
    UsdGenExecutionResourcePool& operator=(UsdGenExecutionResourcePool const&) = delete;

    // Returns empty when bytes cannot fit without consuming headroom or when
    // the request itself overflows. No allocation has happened on failure.
    std::optional<UsdGenExecutionResourcePermit> TryReserve(
        size_t bytes, UsdGenExecutionResourceKind kind) const noexcept;
    // Atomically charges one job-owned Pending balance. Subsequent Consume
    // calls cannot exceed that balance even as other jobs allocate.
    std::optional<UsdGenExecutionMemoryReservation> TryReserveMemory(
        size_t bytes) const noexcept;
    // Diagnostic only: independent category atomics can describe a transient
    // reclassification, so byKind is not a coherent decision-making snapshot.
    UsdGenExecutionResourceSnapshot Snapshot() const noexcept;

private:
    std::shared_ptr<UsdGenExecutionResourcePermit::State> state_;
};

// Backend/device identities permit a single resource-runtime DSO to share a
// pool across workspaces, images and test executables without application
// locks. The first configuration wins; an explicit mismatched configuration
// fails instead of creating a second budget for the same physical device.
enum class UsdGenExecutionResourceBackend : uint8_t { Cuda, Metal, Vulkan };
struct UsdGenExecutionResourceDevice {
    UsdGenExecutionResourceBackend backend;
    int device = -1;
};
struct UsdGenExecutionResourceConfig {
    size_t limitBytes = 0;
    size_t headroomBytes = 0;
};
USDGEN_EXECUTION_RESOURCES_API std::shared_ptr<UsdGenExecutionResourcePool> GetOrCreateUsdGenExecutionResourcePool(
    UsdGenExecutionResourceDevice, UsdGenExecutionResourceConfig,
    bool requireMatchingConfig = true) noexcept;
USDGEN_EXECUTION_RESOURCES_API std::shared_ptr<UsdGenExecutionResourcePool> FindUsdGenExecutionResourcePool(
    UsdGenExecutionResourceDevice) noexcept;

} // namespace usdGen
#endif
