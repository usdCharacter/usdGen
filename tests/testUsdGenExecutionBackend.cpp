#include "usdGen/executionBackend.h"
#include "usdGen/sessionBackendExecutor.h"
#include "usdGen/compiler.h"
#include "usdGen/cudaExecution.h"
#include "usdGen/op.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>

using namespace usdGen;

namespace {
int failures = 0;
void Check(bool value, char const *message) {
    if (!value) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

class DisabledProbeAdapter final : public UsdGenExecutionBackendAdapter {
public:
    UsdGenExecutionBackendContract Contract() const noexcept override {
        return {UsdGenExecutionBackend::Metal, UsdGenDeviceBackend::Metal,
                "metal", "test adapter is intentionally unavailable", 77,
                UsdGenExecutionBackendAvailability::Unavailable};
    }

    std::shared_ptr<const UsdGenExecutionPlanHandle> Compile(
        UsdGenGraphDesc const&, UsdGenDiagnostics*) const override { return {}; }

    std::shared_ptr<UsdGenExecutionBackendExecutor> CreateExecutor(
        std::shared_ptr<const UsdGenExecutionPlanHandle>,
        UsdGenExecutionRuntime&, UsdGenDiagnostics*) const override { return {}; }
};

class InvalidProbeAdapter final : public UsdGenExecutionBackendAdapter {
public:
    UsdGenExecutionBackendContract Contract() const noexcept override { return {}; }
    std::shared_ptr<const UsdGenExecutionPlanHandle> Compile(
        UsdGenGraphDesc const&, UsdGenDiagnostics*) const override { return {}; }
    std::shared_ptr<UsdGenExecutionBackendExecutor> CreateExecutor(
        std::shared_ptr<const UsdGenExecutionPlanHandle>,
        UsdGenExecutionRuntime&, UsdGenDiagnostics*) const override { return {}; }
};
}

int main() {
    // These are descriptor compatibility values, not an implementation
    // detail: existing CPU/CUDA/Invalid snapshots must remain decodable.
    Check(static_cast<unsigned>(UsdGenExecutionBackend::CpuReference) == 0,
          "CpuReference value remains stable");
    Check(static_cast<unsigned>(UsdGenExecutionBackend::Cuda) == 1,
          "Cuda value remains stable");
    Check(static_cast<unsigned>(UsdGenExecutionBackend::Invalid) == 2,
          "Invalid value remains stable");

    Check(ParseUsdGenExecutionBackend(TfToken("cpu")) ==
              UsdGenExecutionBackend::CpuReference,
          "cpu token parses");
    Check(ParseUsdGenExecutionBackend(TfToken("cuda")) ==
              UsdGenExecutionBackend::Cuda,
          "cuda token parses");
    Check(ParseUsdGenExecutionBackend(TfToken("metal")) ==
              UsdGenExecutionBackend::Metal,
          "metal token parses");
    Check(ParseUsdGenExecutionBackend(TfToken("vulkan")) ==
              UsdGenExecutionBackend::Vulkan,
          "vulkan token parses");
    Check(ParseUsdGenExecutionBackend(TfToken("unknown")) ==
              UsdGenExecutionBackend::Invalid,
          "unknown token is invalid");
    Check(std::string(UsdGenExecutionBackendName(UsdGenExecutionBackend::Metal)) ==
              "metal",
          "metal name is canonical");
    Check(std::string(UsdGenExecutionBackendName(UsdGenExecutionBackend::Vulkan)) ==
              "vulkan",
          "vulkan name is canonical");

    auto const metal = GetUsdGenExecutionBackendContract(
        UsdGenExecutionBackend::Metal);
    auto const vulkan = GetUsdGenExecutionBackendContract(
        UsdGenExecutionBackend::Vulkan);
    Check(metal.IsValid() && !metal.Available() &&
              metal.deviceBackend == UsdGenDeviceBackend::Metal,
          "Metal is an explicit unavailable factory contract");
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    Check(vulkan.IsValid() && vulkan.Available() &&
              vulkan.deviceBackend == UsdGenDeviceBackend::Vulkan &&
              vulkan.capabilityVersion == 1,
          "Vulkan availability mirrors the CUDA transitional legacy path");
#else
    Check(vulkan.IsValid() && !vulkan.Available() &&
              vulkan.deviceBackend == UsdGenDeviceBackend::Vulkan,
          "Vulkan is an explicit unavailable factory contract");
#endif
    auto const cuda = GetUsdGenExecutionBackendContract(
        UsdGenExecutionBackend::Cuda);
    Check(cuda.capabilityVersion == GetCudaExecutionCapabilityMatrix().Version(),
          "CUDA contract uses the concrete capability-matrix version");
    auto const cpu = GetUsdGenExecutionBackendContract(
        UsdGenExecutionBackend::CpuReference);
    Check(cpu.Available() && !FindUsdGenExecutionBackendAdapter(
              UsdGenExecutionBackend::CpuReference),
          "CPU availability remains a transitional legacy path");
#ifdef USDGEN_ENABLE_CUDA
    Check(cuda.Available() && !FindUsdGenExecutionBackendAdapter(
              UsdGenExecutionBackend::Cuda),
          "CUDA availability remains a transitional legacy path");
#else
    Check(!cuda.Available(), "CUDA-disabled build remains unavailable");
#endif

    UsdGenDiagnostics diagnostics;
    Check(!ValidateUsdGenExecutionBackend(UsdGenExecutionBackend::Metal,
                                          &diagnostics) &&
              diagnostics.errors.size() == 1 &&
              diagnostics.errors.front().find(
                  "Metal execution backend factory is unavailable") != std::string::npos,
          "Metal reports precise unavailability without fallback");
    diagnostics.errors.clear();
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    Check(ValidateUsdGenExecutionBackend(UsdGenExecutionBackend::Vulkan,
                                         &diagnostics) &&
              diagnostics.errors.empty(),
          "Vulkan-enabled build validates the Vulkan backend");
#else
    Check(!ValidateUsdGenExecutionBackend(UsdGenExecutionBackend::Vulkan,
                                          &diagnostics) &&
              diagnostics.errors.size() == 1 &&
              diagnostics.errors.front().find(
                  "Vulkan execution backend factory is unavailable") != std::string::npos &&
              diagnostics.errors.front().find(
                  "Vulkan support was not enabled in this build") != std::string::npos,
          "Vulkan reports precise unavailability without fallback");
#endif

    auto const metalContext = MakeUsdGenExecutionContext(
        UsdGenExecutionBackend::Metal, 17, 3, 99, UsdGenContext::Render);
    Check(metalContext.backend == UsdGenDeviceBackend::Metal &&
              metalContext.capabilityVersion == 17 &&
              metalContext.deviceIndex == 3 &&
              metalContext.deviceGeneration == 99 &&
              metalContext.evaluationContext == UsdGenContext::Render,
          "Metal cache context preserves backend/device identity");
    auto const vulkanContext = MakeUsdGenExecutionContext(
        UsdGenExecutionBackend::Vulkan, 23, 5, 101);
    Check(vulkanContext.backend == UsdGenDeviceBackend::Vulkan &&
              vulkanContext.deviceIndex == 5,
          "Vulkan cache context preserves backend/device identity");

    // The registry is exercised only after the built-in unavailable checks:
    // this probe advertises no executable implementation and must not be
    // mistaken for production Metal support. It registers before the first
    // compile, which freezes the registries.
    std::string reason;
    Check(!RegisterUsdGenExecutionBackendAdapter({}, &reason) &&
              reason == "cannot register a null execution backend adapter",
          "null backend adapter registration is rejected");
    Check(!RegisterUsdGenExecutionBackendAdapter(
              std::make_shared<InvalidProbeAdapter>(), &reason) &&
              reason == "execution backend adapter contract is invalid",
          "invalid backend adapter contract is rejected");
    auto disabled = std::make_shared<DisabledProbeAdapter>();
    Check(RegisterUsdGenExecutionBackendAdapter(disabled, &reason),
          "unavailable backend adapter can register explicitly");
    auto found = FindUsdGenExecutionBackendAdapter(
        UsdGenExecutionBackend::Metal);
    Check(found && found == disabled && !GetUsdGenExecutionBackendContract(
              UsdGenExecutionBackend::Metal).Available(),
          "registered unavailable adapter remains fail-closed");
    Check(!RegisterUsdGenExecutionBackendAdapter(disabled, &reason) &&
              reason == "execution backend adapter is already registered",
          "duplicate backend adapter registration is rejected");

    // Compiler-level proof: an unavailable portable backend fails before a
    // candidate graph is committed. In particular, it must not quietly
    // produce a CpuReference graph from the same descriptor.
    UsdGenGraphDesc baseline;
    baseline.description = SdfPath("/backendContract");
    UsdGenNodeDesc source;
    source.path = SdfPath("/backendContract/source");
    source.type = TfToken("UsdGenCurveSource");
    baseline.nodes.push_back(source);
    UsdGenNodeDesc width;
    width.path = SdfPath("/backendContract/width");
    width.type = TfToken("UsdGenWidth");
    width.inputs = {source.path};
    baseline.nodes.push_back(width);
    baseline.terminal = width.path;
    {
        auto requested = baseline;
        requested.executionBackend = UsdGenExecutionBackend::Metal;
        UsdGenGraph candidate;
        UsdGenCompileResult result = UsdGenCompiler().Compile(requested, &candidate);
        bool precise = std::any_of(result.errors.begin(), result.errors.end(),
            [&](std::string const &error) {
                return error.find("Metal execution backend factory is unavailable") !=
                    std::string::npos;
            });
        Check(!result.ok && precise, "Metal compiler rejection is precise");
        Check(candidate.NodeCount() == 0,
              "Metal rejection does not produce a CPU graph");
    }
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    {
        // A built Vulkan backend validates like CUDA: direct compilation
        // succeeds and execution routes through the Session device provider.
        auto requested = baseline;
        requested.executionBackend = UsdGenExecutionBackend::Vulkan;
        UsdGenGraph candidate;
        UsdGenCompileResult result = UsdGenCompiler().Compile(requested, &candidate);
        Check(result.ok, "Vulkan-enabled build compiles the Vulkan descriptor");
    }
#else
    {
        auto requested = baseline;
        requested.executionBackend = UsdGenExecutionBackend::Vulkan;
        UsdGenGraph candidate;
        UsdGenCompileResult result = UsdGenCompiler().Compile(requested, &candidate);
        bool precise = std::any_of(result.errors.begin(), result.errors.end(),
            [&](std::string const &error) {
                return error.find("Vulkan execution backend factory is unavailable") !=
                        std::string::npos &&
                    error.find("Vulkan support was not enabled in this build") !=
                        std::string::npos;
            });
        Check(!result.ok && precise, "Vulkan compiler rejection is precise");
        Check(candidate.NodeCount() == 0,
              "Vulkan rejection does not produce a CPU graph");
    }
#endif

    auto metadata = std::make_shared<const UsdGenExecutionPlanMetadata>(
        "metal", 77, UsdGenExecutionPlanShape::SourceRootedValueDag,
        std::vector<UsdGenCompiledOperatorCapability>{},
        std::vector<UsdGenExecutionTaskMetadata>{},
        std::vector<UsdGenExecutionValueMetadata>{}, 0);
    auto payload = std::make_shared<int>(42);
    auto handle = UsdGenExecutionPlanHandle::Create(
        UsdGenExecutionBackend::Metal, metadata, payload, &reason);
    Check(handle && handle->Backend() == UsdGenExecutionBackend::Metal &&
              handle->Metadata() == metadata && handle->Payload() == payload,
          "opaque immutable plan handle retains neutral metadata and payload");
    Check(!UsdGenExecutionPlanHandle::Create(
              UsdGenExecutionBackend::Vulkan, metadata, {}, &reason) &&
              reason == "execution plan metadata backend does not match handle backend",
          "plan handle rejects backend identity collision");

    // Session routing is intentionally private to preserve the richer
    // COW/coalescing/publication contract that executionBackend.h cannot yet
    // express. One executor owns one cooker while selecting the concrete path
    // from each immutable request, so a descriptor backend switch cannot lose
    // the Session's last-good/cache state.
    auto sessionExecutor = CreateUsdGenSessionBackendExecutor(
        1, UsdGenSessionCooker::kDefaultExecutionCacheBytes, {});
    Check(sessionExecutor &&
              sessionExecutor->CanRoute(UsdGenExecutionBackend::CpuReference),
          "private Session executor routes CPU requests");
#ifdef USDGEN_ENABLE_CUDA
    Check(sessionExecutor && sessionExecutor->CanRoute(UsdGenExecutionBackend::Cuda),
          "private Session executor routes CUDA requests");
#else
    Check(sessionExecutor && !sessionExecutor->CanRoute(UsdGenExecutionBackend::Cuda),
          "CUDA-disabled private Session executor fails closed");
#endif
    Check(sessionExecutor &&
              !sessionExecutor->CanRoute(UsdGenExecutionBackend::Metal),
          "private Session executor keeps Metal fail-closed");
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    Check(sessionExecutor &&
              sessionExecutor->CanRoute(UsdGenExecutionBackend::Vulkan),
          "private Session executor routes Vulkan requests without injection");
#else
    Check(sessionExecutor &&
              !sessionExecutor->CanRoute(UsdGenExecutionBackend::Vulkan),
          "Vulkan-disabled private Session executor fails closed without injection");
#endif
    if (sessionExecutor) sessionExecutor->Shutdown();

    std::printf("testUsdGenExecutionBackend: %s\n",
                 failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
