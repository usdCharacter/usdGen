// Compile/link seam for consumers which only use the backend-neutral API.
#include "usdGen/api/ext/device/deviceGeneration.h"
#include "api/testBackend.h"
#include "usdGen/api/backend.h"
#include "usdGen/api/ext/device/executionPlan.h"
#include "usdGen/api/ext/device/executionResources.h"
#include "usdGen/api/ext/device/executionRetirement.h"
#include "usdGen/api/ext/device/executionTaskGraph.h"
#include "usdGen/api/ext/device/executionValueRevisions.h"
#include "usdGen/exec/executionCache.h"

#include <cstdio>
#include <type_traits>

using namespace usdGen;

int main()
{
    static_assert(std::is_same<decltype(UsdGenExecutionRequest::requestEpoch),
                               uint64_t>::value,
                  "request identity stays backend-neutral");
    static_assert(std::is_same<decltype(UsdGenExecutionResourceDevice::device),
                               int>::value,
                  "resource device identity has no native handle");
    static_assert(std::is_same<UsdGenDeviceStream, uintptr_t>::value,
                  "device stream remains an opaque neutral token");

    UsdGenExecutionValueRevisions revisions{0, 0, 2, {1}};
    if (!revisions.IsStrictlyOrdered(1) || revisions.IsStrictlyOrdered(0)) return 2;
    revisions.intermediateValueVersions = {0};
    if (revisions.IsStrictlyOrdered(1)) return 3;
    revisions.intermediateValueVersions = {2};
    if (revisions.IsStrictlyOrdered(1)) return 4;
    revisions = {0, 1, 5, {3, 2}};
    if (revisions.IsStrictlyOrdered(2)) return 5;
    revisions = {0, UINT64_MAX - 1, UINT64_MAX, {}};
    if (!revisions.IsStrictlyOrdered(0)) return 6;

    // Interleaved chains reserve every actual stage from the shared domain.
    // A valid intermediate is not necessarily source+1 or final-1.
    auto domain = UsdGenExecutionCacheDomain::AcquireShared(
        {usdGenTestUtils::kTestBackendA, 0, 0x5354414745524556ull}, 1024);
    auto sourceA = domain->AllocatePublicationGeneration(0);
    auto sourceB = domain->AllocatePublicationGeneration(sourceA);
    auto middleA = domain->AllocatePublicationGeneration(sourceA);
    auto middleB = domain->AllocatePublicationGeneration(sourceB);
    auto finalA = domain->AllocatePublicationGeneration(middleA);
    auto finalB = domain->AllocatePublicationGeneration(middleB);
    UsdGenExecutionValueRevisions chainA{sourceA, sourceA, finalA, {middleA}};
    UsdGenExecutionValueRevisions chainB{sourceB, sourceB, finalB, {middleB}};
    if (!chainA.IsStrictlyOrdered(1) || !chainB.IsStrictlyOrdered(1) ||
        middleA <= sourceA + 1 || finalA <= middleA + 1) return 7;

    auto const contract = GetUsdGenExecutionBackendContract(
        usdGenTestUtils::Backend("vulkan"));
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    if (!contract.IsValid() || contract.backend != usdGenTestUtils::Backend("vulkan") ||
        !contract.Available() || contract.capabilityVersion != 1) {
        std::fprintf(stderr, "neutral backend contract check failed\n");
        return 1;
    }
#else
    // Without the runtime "vulkan" registers unavailable; without usdGenVulkan
    // nothing registers it. Either way it cannot run.
    UsdGenBackendId const vulkan = usdGenTestUtils::Backend("vulkan");
    if (contract.Available() || (vulkan != UsdGenBackendId::Invalid &&
        (!contract.IsValid() || contract.backend != vulkan))) {
        std::fprintf(stderr, "neutral backend contract check failed\n");
        return 1;
    }
#endif
    std::puts("testUsdGenExecutionBackendNeutral: PASS");
    return 0;
}
