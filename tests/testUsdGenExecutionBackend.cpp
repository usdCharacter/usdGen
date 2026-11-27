#include "utils/pxrInterop.h"
#include "usdGen/api/backend.h"
#include "usdGenUsd/lower.h"
#include "usdGen/exec/sessionBackendExecutor.h"
#include "usdGen/compile/compiler.h"
#include "usdGenCurves/cuda/cudaExecution.h"
#include "usdGen/api/ext/op.h"
#include "testBackend.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

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
        return {usdGenTestUtils::kTestBackendA, "probe",
                "test adapter is intentionally unavailable", 77,
                UsdGenExecutionBackendAvailability::Unavailable};
    }

    std::shared_ptr<const UsdGenExecutionPlanHandle> Compile(
        UsdGenGraphSpec const&, UsdGenDiagnostics*) const override { return {}; }

    std::shared_ptr<UsdGenExecutionBackendExecutor> CreateExecutor(
        std::shared_ptr<const UsdGenExecutionPlanHandle>,
        UsdGenExecutionRuntime&, UsdGenDiagnostics*) const override { return {}; }
};

// Claims the registered "cuda" handle under another name.
class MisnamedProbeAdapter final : public UsdGenExecutionBackendAdapter {
public:
    UsdGenExecutionBackendContract Contract() const noexcept override {
        return {usdGenTestUtils::Backend("cuda"), "notCuda", "", 1,
                UsdGenExecutionBackendAvailability::Unavailable};
    }
    std::shared_ptr<const UsdGenExecutionPlanHandle> Compile(
        UsdGenGraphSpec const&, UsdGenDiagnostics*) const override { return {}; }
    std::shared_ptr<UsdGenExecutionBackendExecutor> CreateExecutor(
        std::shared_ptr<const UsdGenExecutionPlanHandle>,
        UsdGenExecutionRuntime&, UsdGenDiagnostics*) const override { return {}; }
};

class InvalidProbeAdapter final : public UsdGenExecutionBackendAdapter {
public:
    UsdGenExecutionBackendContract Contract() const noexcept override { return {}; }
    std::shared_ptr<const UsdGenExecutionPlanHandle> Compile(
        UsdGenGraphSpec const&, UsdGenDiagnostics*) const override { return {}; }
    std::shared_ptr<UsdGenExecutionBackendExecutor> CreateExecutor(
        std::shared_ptr<const UsdGenExecutionPlanHandle>,
        UsdGenExecutionRuntime&, UsdGenDiagnostics*) const override { return {}; }
};

bool Contains(std::vector<std::string> const &errors, char const *text) {
    return std::any_of(errors.begin(), errors.end(), [&](std::string const &error) {
        return error.find(text) != std::string::npos;
    });
}
}

int main() {
    // Backends are registered handles: the CPU is built in as 0, Invalid is
    // distinct from it, and every other name is what a library registered.
    UsdGenBackendId const cpuId = usdGenTestUtils::Backend("cpu");
    UsdGenBackendId const cudaId = usdGenTestUtils::Backend("cuda");
    UsdGenBackendId const metalId = usdGenTestUtils::Backend("metal");
    UsdGenBackendId const vulkanId = usdGenTestUtils::Backend("vulkan");
    Check(static_cast<unsigned>(kUsdGenBackendCpu) == 0,
          "the CPU handle remains 0");
    Check(UsdGenBackendId::Invalid != kUsdGenBackendCpu,
          "Invalid is distinct from the CPU");

    Check(cpuId == kUsdGenBackendCpu, "cpu token finds the CPU");
    Check(UsdGenFindBackend(TfToken("cpuReference")) == kUsdGenBackendCpu,
          "cpuReference token finds the CPU");
    Check(UsdGenFindBackend(TfToken("cuda")) == cudaId &&
              cudaId != UsdGenBackendId::Invalid && cudaId != kUsdGenBackendCpu,
          "cuda token finds the registered CUDA handle");
    Check(metalId == UsdGenBackendId::Invalid,
          "metal names no registered backend");
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    Check(vulkanId != UsdGenBackendId::Invalid && vulkanId != cudaId,
          "vulkan token finds the registered Vulkan handle");
#endif
    Check(UsdGenFindBackend(TfToken("unknown")) == UsdGenBackendId::Invalid &&
              UsdGenFindBackend(TfToken()) == UsdGenBackendId::Invalid,
          "unknown token is invalid");
    Check(std::string(UsdGenBackendName(cudaId)) == "cuda",
          "cuda name is canonical");
    Check(std::string(UsdGenBackendName(kUsdGenBackendCpu)) == "cpu" &&
              std::string(UsdGenBackendName(UsdGenBackendId::Invalid)) == "invalid",
          "cpu and invalid names are canonical");
    Check(vulkanId == UsdGenBackendId::Invalid ||
              std::string(UsdGenBackendName(vulkanId)) == "vulkan",
          "vulkan name is canonical");

    auto const metal = GetUsdGenExecutionBackendContract(metalId);
    auto const vulkan = GetUsdGenExecutionBackendContract(vulkanId);
    Check(!metal.IsValid() && !metal.Available(),
          "an unregistered name has no contract and is fail-closed");
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    Check(vulkan.IsValid() && vulkan.Available() &&
              vulkan.backend == vulkanId &&
              vulkan.capabilityVersion == 1,
          "Vulkan availability comes from its registered def");
#else
    Check(!vulkan.Available() && (vulkanId == UsdGenBackendId::Invalid ||
              (vulkan.IsValid() && vulkan.backend == vulkanId)),
          "Vulkan is an explicit unavailable factory contract");
#endif
    auto const cuda = GetUsdGenExecutionBackendContract(cudaId);
    Check(cuda.capabilityVersion == GetCudaExecutionCapabilityMatrix().Version(),
          "CUDA contract uses the concrete capability-matrix version");
    auto const cpu = GetUsdGenExecutionBackendContract(kUsdGenBackendCpu);
    Check(cpu.Available() && !FindUsdGenExecutionBackendAdapter(kUsdGenBackendCpu),
          "CPU availability is built in");
#ifdef USDGEN_ENABLE_CUDA
    Check(cuda.Available() && !FindUsdGenExecutionBackendAdapter(cudaId),
          "CUDA availability comes from its registered def");
#else
    Check(!cuda.Available(), "CUDA-disabled build remains unavailable");
#endif

    UsdGenDiagnostics diagnostics;
    Check(!ValidateUsdGenExecutionBackend(metalId, &diagnostics) &&
              diagnostics.errors.size() == 1 &&
              diagnostics.errors.front().find(
                  "refusing implicit CPU fallback") != std::string::npos,
          "an unregistered name reports rejection without fallback");
    diagnostics.errors.clear();
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    Check(ValidateUsdGenExecutionBackend(vulkanId, &diagnostics) &&
              diagnostics.errors.empty(),
          "Vulkan-enabled build validates the Vulkan backend");
#else
    Check(!ValidateUsdGenExecutionBackend(vulkanId, &diagnostics) &&
              diagnostics.errors.size() == 1 &&
              (vulkanId == UsdGenBackendId::Invalid ||
               (diagnostics.errors.front().find(
                    "Vulkan execution backend factory is unavailable") != std::string::npos &&
                diagnostics.errors.front().find(
                    "Vulkan support was not enabled in this build") != std::string::npos)),
          "Vulkan reports precise unavailability without fallback");
#endif

    auto const cudaContext = MakeUsdGenExecutionContext(
        cudaId, 17, 3, 99, UsdGenContext::Render);
    Check(cudaContext.backend == cudaId &&
              cudaContext.capabilityVersion == 17 &&
              cudaContext.deviceIndex == 3 &&
              cudaContext.deviceGeneration == 99 &&
              cudaContext.evaluationContext == UsdGenContext::Render,
          "cache context preserves backend/device identity");
    auto const metalContext = MakeUsdGenExecutionContext(metalId, 23, 5, 101);
    Check(metalContext.backend == UsdGenBackendId::Invalid,
          "an unregistered backend has no cache context identity");
    if (vulkanId != UsdGenBackendId::Invalid) {
        auto const vulkanContext = MakeUsdGenExecutionContext(vulkanId, 23, 5, 101);
        Check(vulkanContext.backend == vulkanId && vulkanContext.deviceIndex == 5,
              "Vulkan cache context preserves backend/device identity");
    }

    // The adapter registry is exercised only after the built-in checks: this
    // probe advertises no executable implementation on a handle no library
    // registered. It registers before the first compile, which freezes the
    // registries.
    std::string reason;
    Check(!RegisterUsdGenExecutionBackendAdapter({}, &reason) &&
              reason == "cannot register a null execution backend adapter",
          "null backend adapter registration is rejected");
    Check(!RegisterUsdGenExecutionBackendAdapter(
              std::make_shared<InvalidProbeAdapter>(), &reason) &&
              reason == "execution backend adapter contract is invalid",
          "invalid backend adapter contract is rejected");
    Check(!RegisterUsdGenExecutionBackendAdapter(
              std::make_shared<MisnamedProbeAdapter>(), &reason) &&
              reason == "execution backend adapter contract name is not canonical",
          "adapter name must match the registered def");
    auto disabled = std::make_shared<DisabledProbeAdapter>();
    Check(RegisterUsdGenExecutionBackendAdapter(disabled, &reason),
          "unavailable backend adapter can register explicitly");
    auto found = FindUsdGenExecutionBackendAdapter(usdGenTestUtils::kTestBackendA);
    Check(found && found == disabled && !GetUsdGenExecutionBackendContract(
              usdGenTestUtils::kTestBackendA).Available(),
          "registered unavailable adapter remains fail-closed");
    Check(!RegisterUsdGenExecutionBackendAdapter(disabled, &reason) &&
              reason == "execution backend adapter is already registered",
          "duplicate backend adapter registration is rejected");

    // Compiler-level proof: an unknown backend fails before a candidate graph
    // is committed. In particular, it must not quietly produce a CPU graph
    // from the same descriptor.
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
        requested.executionBackend = metalId;
        UsdGenGraph candidate;
        UsdGenCompiler compiler;
        UsdGenCompileResult result = usdGenUsd::Compile(compiler, requested, &candidate);
        Check(!result.ok && Contains(result.errors, "refusing implicit CPU fallback"),
              "unknown-backend compiler rejection is precise");
        Check(candidate.NodeCount() == 0,
              "unknown-backend rejection does not produce a CPU graph");
    }
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    {
        // A built Vulkan backend validates like CUDA: direct compilation
        // succeeds and execution routes through the Session device provider.
        auto requested = baseline;
        requested.executionBackend = vulkanId;
        UsdGenGraph candidate;
        UsdGenCompiler compiler;
        UsdGenCompileResult result = usdGenUsd::Compile(compiler, requested, &candidate);
        Check(result.ok, "Vulkan-enabled build compiles the Vulkan descriptor");
    }
#else
    {
        auto requested = baseline;
        requested.executionBackend = vulkanId;
        UsdGenGraph candidate;
        UsdGenCompiler compiler;
        UsdGenCompileResult result = usdGenUsd::Compile(compiler, requested, &candidate);
        bool const precise = vulkanId == UsdGenBackendId::Invalid
            ? Contains(result.errors, "refusing implicit CPU fallback")
            : std::any_of(result.errors.begin(), result.errors.end(),
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
        "cuda", 77, UsdGenExecutionPlanShape::SourceRootedValueDag,
        std::vector<UsdGenCompiledOperatorCapability>{},
        std::vector<UsdGenExecutionTaskMetadata>{},
        std::vector<UsdGenExecutionValueMetadata>{}, 0);
    auto payload = std::make_shared<int>(42);
    auto handle = UsdGenExecutionPlanHandle::Create(cudaId, metadata, payload, &reason);
    Check(handle && handle->Backend() == cudaId &&
              handle->Metadata() == metadata && handle->Payload() == payload,
          "opaque immutable plan handle retains neutral metadata and payload");
    Check(!UsdGenExecutionPlanHandle::Create(
              usdGenTestUtils::kTestBackendA, metadata, {}, &reason) &&
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
              sessionExecutor->CanRoute(kUsdGenBackendCpu),
          "private Session executor routes CPU requests");
#ifdef USDGEN_ENABLE_CUDA
    Check(sessionExecutor && sessionExecutor->CanRoute(cudaId),
          "private Session executor routes CUDA requests");
#else
    Check(sessionExecutor && !sessionExecutor->CanRoute(cudaId),
          "CUDA-disabled private Session executor fails closed");
#endif
    Check(sessionExecutor &&
              !sessionExecutor->CanRoute(metalId),
          "private Session executor keeps an unregistered backend fail-closed");
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    Check(sessionExecutor &&
              sessionExecutor->CanRoute(vulkanId),
          "private Session executor routes Vulkan requests without injection");
#else
    Check(sessionExecutor &&
              !sessionExecutor->CanRoute(vulkanId),
          "Vulkan-disabled private Session executor fails closed without injection");
#endif
    if (sessionExecutor) sessionExecutor->Shutdown();

    std::printf("testUsdGenExecutionBackend: %s\n",
                 failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
