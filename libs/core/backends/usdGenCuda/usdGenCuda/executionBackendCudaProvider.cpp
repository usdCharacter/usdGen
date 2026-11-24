// Small capability-version bridge for the neutral backend contract.
//
// The backend registry must not include cudaExecution.h: future portable
// adapters should be able to include and link the neutral contract without
// importing a concrete native adapter.  This TU is the only bridge needed by
// the current built-in CUDA entry.
#include "usdGen/cudaExecution.h"

#include <cstdint>

namespace usdGen {

uint32_t UsdGenCudaCapabilityVersionForBackendContract() noexcept
{
    return GetCudaExecutionCapabilityMatrix().Version();
}

} // namespace usdGen
