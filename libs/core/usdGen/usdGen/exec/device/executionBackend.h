// Backend selection and availability contract for the execution graph.
//
// This interface is deliberately native-runtime-free.  It lets adapters
// select a stable backend/device cache identity and report an unavailable
// factory without inventing a CPU fallback or importing CUDA, Metal, or
// Vulkan SDK headers.
#ifndef USDGEN_EXECUTION_BACKEND_H
#define USDGEN_EXECUTION_BACKEND_H

#include "usdGen/executionCache.h"
#include "usdGen/executionPlan.h"
#include "usdGen/executionPipeline.h"

#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace usdGen {

struct UsdGenDiagnostics;

enum class UsdGenExecutionBackendAvailability : uint8_t {
    Invalid = 0,
    Available,
    Unavailable
};

/// Stable description of one selectable execution backend.  `available`
/// describes an executable path for the backend, not whether a particular
/// device is currently present.  CPU and CUDA currently expose this through
/// legacy Session-owned paths while their adapter registrations are migrated;
/// Metal and Vulkan are explicit unavailable entries until native adapters are
/// linked.
struct UsdGenExecutionBackendContract {
    UsdGenExecutionBackend backend = UsdGenExecutionBackend::Invalid;
    UsdGenDeviceBackend deviceBackend = UsdGenDeviceBackend::Unknown;
    const char *name = "invalid";
    const char *unavailabilityReason = "invalid execution backend";
    uint32_t capabilityVersion = 0;
    UsdGenExecutionBackendAvailability availability =
        UsdGenExecutionBackendAvailability::Invalid;

    bool IsValid() const noexcept {
        return availability != UsdGenExecutionBackendAvailability::Invalid;
    }
    bool Available() const noexcept {
        return availability == UsdGenExecutionBackendAvailability::Available;
    }
};

/// Immutable adapter-owned execution plan boundary.  Metadata is inspectable
/// by generic schedulers; payload is deliberately opaque so a backend can
/// retain compiled programs, command encoders, or other native state without
/// placing a CUDA, Metal, or Vulkan type in this interface.
class UsdGenExecutionPlanHandle {
public:
    static std::shared_ptr<const UsdGenExecutionPlanHandle> Create(
        UsdGenExecutionBackend backend,
        std::shared_ptr<const UsdGenExecutionPlanMetadata> metadata,
        std::shared_ptr<const void> payload = {},
        std::string *reason = nullptr);

    UsdGenExecutionBackend Backend() const noexcept { return backend_; }
    std::shared_ptr<const UsdGenExecutionPlanMetadata> Metadata() const noexcept {
        return metadata_;
    }
    std::shared_ptr<const void> Payload() const noexcept { return payload_; }

private:
    UsdGenExecutionPlanHandle(
        UsdGenExecutionBackend backend,
        std::shared_ptr<const UsdGenExecutionPlanMetadata> metadata,
        std::shared_ptr<const void> payload) noexcept
        : backend_(backend), metadata_(std::move(metadata)),
          payload_(std::move(payload)) {}

    UsdGenExecutionBackend backend_ = UsdGenExecutionBackend::Invalid;
    std::shared_ptr<const UsdGenExecutionPlanMetadata> metadata_;
    std::shared_ptr<const void> payload_;
};

/// Immutable request envelope handed from a command owner to a concrete
/// backend adapter.  Input/version identity remains neutral; native queues,
/// events, allocations, and COW storage stay inside the adapter payload.
struct UsdGenExecutionRequest {
    std::shared_ptr<const UsdGenExecutionPlanHandle> plan;
    UsdGenExecutionContext context;
    UsdGenExecutionInputVersions inputVersions;
    uint64_t requestEpoch = 0;
    double frame = 0.0;
};

/// Completion of one adapter submission.  A successful adapter must return a
/// neutral immutable generation; failure must not manufacture a CPU result.
using UsdGenExecutionCompletion = std::function<void(
    std::shared_ptr<const UsdGenDeviceGeneration>, std::exception_ptr)>;

/// Owner-private mutable execution boundary.  The registry never stores this
/// object: an adapter creates one for each Session/command owner, allowing it
/// to retain workspace, device context, previous-generation state, and native
/// reservations without making any of those lifetimes process-global.
class UsdGenExecutionBackendExecutor {
public:
    virtual ~UsdGenExecutionBackendExecutor() = default;
    virtual bool Submit(UsdGenExecutionRequest,
                        UsdGenExecutionPipeline::Cancellation const&,
                        UsdGenExecutionCompletion) = 0;
    virtual void Shutdown() noexcept = 0;
};

/// Backend implementation/factory seam.  CUDA, Metal, and Vulkan adapters
/// own their concrete plans/resources and may translate the neutral plan into
/// native work.  Compile is immutable; CreateExecutor establishes the
/// owner-private mutable boundary used for submission.  Registration does not
/// imply that a backend is available: an adapter may expose an explicit
/// unavailable contract until its real factory exists.
class UsdGenExecutionBackendAdapter {
public:
    virtual ~UsdGenExecutionBackendAdapter() = default;
    virtual UsdGenExecutionBackendContract Contract() const noexcept = 0;
    virtual std::shared_ptr<const UsdGenExecutionPlanHandle> Compile(
        UsdGenGraphDesc const&, UsdGenDiagnostics*) const = 0;
    virtual std::shared_ptr<UsdGenExecutionBackendExecutor> CreateExecutor(
        std::shared_ptr<const UsdGenExecutionPlanHandle>,
        UsdGenExecutionRuntime&, UsdGenDiagnostics*) const = 0;
};

/// Register one process-lifetime backend adapter.  The first registration for
/// a backend wins; duplicate registrations are rejected so two DSOs cannot
/// silently disagree about capability or native ownership.
bool RegisterUsdGenExecutionBackendAdapter(
    std::shared_ptr<const UsdGenExecutionBackendAdapter>,
    std::string *reason = nullptr) noexcept;

/// Close registration. Called by UsdGenFreezeRegistries; later
/// RegisterUsdGenExecutionBackendAdapter calls fail.
void UsdGenFreezeExecutionBackendRegistry() noexcept;

/// Look up the registered adapter without loading a native runtime.  A null
/// result means the backend has no linked factory; callers must fail closed.
std::shared_ptr<const UsdGenExecutionBackendAdapter>
FindUsdGenExecutionBackendAdapter(UsdGenExecutionBackend) noexcept;

/// Canonical lowercase token used by the descriptor/imaging transport.
const char *UsdGenExecutionBackendName(UsdGenExecutionBackend) noexcept;

/// Parse only canonical backend tokens. Empty/unknown tokens return Invalid;
/// callers must not silently select CpuReference on parse failure.
UsdGenExecutionBackend ParseUsdGenExecutionBackend(TfToken const&) noexcept;

/// Return the backend-neutral factory/capability contract. This function does
/// not load a native runtime and never claims that an unavailable backend can
/// execute. CPU and CUDA retain their legacy built-in contracts while their
/// current Session paths are being migrated; therefore Available does not
/// imply FindUsdGenExecutionBackendAdapter() is non-null for those two
/// transitional backends.
UsdGenExecutionBackendContract GetUsdGenExecutionBackendContract(
    UsdGenExecutionBackend) noexcept;

/// Validate a requested backend at the compiler/adapter boundary. For an
/// unavailable factory this appends a precise diagnostic and returns false.
bool ValidateUsdGenExecutionBackend(UsdGenExecutionBackend,
                                    UsdGenDiagnostics * = nullptr);

/// Build the cache context from the requested backend's stable identity. The
/// backend/device fields remain explicit even when no executable factory is
/// installed, so a future Metal/Vulkan adapter cannot collide with CPU/CUDA
/// cache entries. `deviceGeneration` is supplied by the native adapter.
UsdGenExecutionContext MakeUsdGenExecutionContext(
    UsdGenExecutionBackend backend, uint32_t capabilityVersion,
    int32_t deviceIndex, uint64_t deviceGeneration,
    UsdGenContext evaluationContext = UsdGenContext::Interactive) noexcept;

} // namespace usdGen

#endif // USDGEN_EXECUTION_BACKEND_H
