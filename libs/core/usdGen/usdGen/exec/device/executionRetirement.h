// Backend-neutral, bounded asynchronous destruction after backend completion.
#ifndef USDGEN_EXECUTION_RETIREMENT_H
#define USDGEN_EXECUTION_RETIREMENT_H

#include "usdGen/executionResources.h"

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <memory>
#include <optional>

namespace usdGen {

// Implemented by a backend-owned object.  Its lifetime, including any native
// callback userdata, is retained by the retirement slot until Destroy reports
// that storage was actually released.  Quarantine must not free backend
// storage or invalidate native callback userdata.
class USDGEN_EXECUTION_RESOURCES_API UsdGenExecutionRetirementPayload {
public:
    virtual ~UsdGenExecutionRetirementPayload() = default;
    virtual bool IsComplete() noexcept = 0;
    // Returns true after adapter cleanup has completed. A backend free failure
    // may conservatively retain its resource permit inside that cleanup.
    // False permanently quarantines this slot.
    virtual bool Destroy() noexcept = 0;
    virtual void Quarantine() noexcept = 0;
};

struct UsdGenExecutionRetirementConfig { size_t capacity = 1024; };

class UsdGenExecutionRetirementService;
class UsdGenExecutionRetirementTicket;

// Declare the exported factory before the service's friend declaration. MSVC
// diagnoses a later dllimport declaration as a different-linkage redeclaration
// when the unannotated friend is seen first.
USDGEN_EXECUTION_RESOURCES_API std::shared_ptr<UsdGenExecutionRetirementService>
GetOrCreateUsdGenExecutionRetirementService(
    UsdGenExecutionResourceDevice, UsdGenExecutionRetirementConfig,
    bool requireMatchingConfig = true) noexcept;

// Copy this while the ticket is live, then give it to the native completion
// mechanism. Signal may occur before or after Retire. The first success or
// failure wins; duplicate/opposite/late signals are ignored. A slot generation
// prevents an old callback from affecting a later ticket.
class USDGEN_EXECUTION_RESOURCES_API UsdGenExecutionRetirementSignal {
public:
    UsdGenExecutionRetirementSignal() = default;
    void SignalSuccess() const noexcept;
    void SignalFailure() const noexcept;
    explicit operator bool() const noexcept { return bool(state_); }
private:
    struct State;
    std::shared_ptr<State> state_;
    size_t slot_ = 0;
    uint64_t generation_ = 0;
    UsdGenExecutionRetirementSignal(std::shared_ptr<State>, size_t, uint64_t) noexcept;
    friend class UsdGenExecutionRetirementTicket;
    friend class UsdGenExecutionRetirementService;
};

// A fixed slot admission. Reserve this before accepting a GPU generation or
// consumer. Retire consumes a valid ticket and returns true once it has
// retained the payload, even when completion has not yet been signalled. A
// false return requires an empty payload or an invalid/previously-consumed
// ticket; callers must not transfer a live backend payload on that misuse.
class USDGEN_EXECUTION_RESOURCES_API UsdGenExecutionRetirementTicket {
public:
    UsdGenExecutionRetirementTicket() = default;
    ~UsdGenExecutionRetirementTicket();
    UsdGenExecutionRetirementTicket(UsdGenExecutionRetirementTicket const&) = delete;
    UsdGenExecutionRetirementTicket& operator=(UsdGenExecutionRetirementTicket const&) = delete;
    UsdGenExecutionRetirementTicket(UsdGenExecutionRetirementTicket&&) noexcept;
    UsdGenExecutionRetirementTicket& operator=(UsdGenExecutionRetirementTicket&&) noexcept;
    explicit operator bool() const noexcept { return bool(state_); }
    UsdGenExecutionRetirementSignal MakeSignal() const noexcept;
    bool Retire(std::unique_ptr<UsdGenExecutionRetirementPayload>) noexcept;
private:
    std::shared_ptr<UsdGenExecutionRetirementSignal::State> state_;
    size_t slot_ = 0;
    uint64_t generation_ = 0;
    UsdGenExecutionRetirementTicket(std::shared_ptr<UsdGenExecutionRetirementSignal::State>, size_t, uint64_t) noexcept;
    void Reset() noexcept;
    friend class UsdGenExecutionRetirementService;
};

class USDGEN_EXECUTION_RESOURCES_API UsdGenExecutionRetirementService {
public:
    ~UsdGenExecutionRetirementService();
    std::optional<UsdGenExecutionRetirementTicket> TryReserve() noexcept;
    // External lifecycle boundary: rejects new admissions. Existing tickets
    // remain usable and must signal their terminal backend result.
    void Shutdown() noexcept;
    // Irreversible backend teardown boundary. Unlike Shutdown(), this also
    // prevents existing tickets from entering payload IsComplete()/Destroy().
    // Retired payloads which have not begun cleanup are retained in their
    // fixed slots; this call waits for cleanup already admitted before the
    // boundary. It must run while the backend runtime is still usable.
    // QuiesceBackend/Drain are external lifecycle operations and callers must
    // serialize concurrent calls on the same service.
    void QuiesceBackend();
    bool BackendQuiesced() const noexcept;
    // External-only: cooperates with the cleanup arena while waiting for
    // retired payloads. Unsignalled retired payloads keep the wait open;
    // callers must quiesce or resolve producers.
    // Calling from any retirement cleanup worker terminates rather than
    // self-deadlocking.
    void Drain();
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    explicit UsdGenExecutionRetirementService(size_t capacity,
                                              UsdGenExecutionResourceBackend);
    friend USDGEN_EXECUTION_RESOURCES_API std::shared_ptr<UsdGenExecutionRetirementService>
    GetOrCreateUsdGenExecutionRetirementService(UsdGenExecutionResourceDevice,
                                                UsdGenExecutionRetirementConfig,
                                                bool) noexcept;
};

USDGEN_EXECUTION_RESOURCES_API std::shared_ptr<UsdGenExecutionRetirementService>
FindUsdGenExecutionRetirementService(UsdGenExecutionResourceDevice) noexcept;

// Process/DSO-wide backend lifecycle. Quiesce is irreversible and closes all
// existing services for this backend as well as future GetOrCreate admission.
// The embedding runtime must call it before backend-runtime teardown and
// serialize concurrent quiesce/drain calls for a backend.
USDGEN_EXECUTION_RESOURCES_API bool
IsUsdGenExecutionRetirementBackendOpen(UsdGenExecutionResourceBackend) noexcept;
USDGEN_EXECUTION_RESOURCES_API void
QuiesceUsdGenExecutionRetirementBackend(UsdGenExecutionResourceBackend) noexcept;

// Narrow deterministic interleaving hook for retirement installation tests.
// A claimed gate pauses once after Retire obtains its Drain credit and before
// it publishes Installing; production leaves it unset.
struct USDGEN_EXECUTION_RESOURCES_API UsdGenExecutionRetirementInstallTestGate {
    std::atomic<bool> claimed{false}, entered{false}, release{false}, timedOut{false};
};
USDGEN_EXECUTION_RESOURCES_API void
SetUsdGenExecutionRetirementInstallTestGate(
    std::shared_ptr<UsdGenExecutionRetirementInstallTestGate>) noexcept;

} // namespace usdGen
#endif
