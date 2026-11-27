#ifndef USDGEN_EXECUTION_VALUE_REVISIONS_H
#define USDGEN_EXECUTION_VALUE_REVISIONS_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace usdGen {

// Authoritative immutable-value revisions for an admitted device plan.
// Intermediate modifiers follow the plan's topological operator order; the terminal
// modifier receives finalValueVersion. Zero is a valid supplied revision.
// These are allocated identities, never source-relative arithmetic guesses.
struct UsdGenExecutionValueRevisions {
    uint64_t topologyVersion = 0;
    uint64_t sourceValueVersion = 0;
    uint64_t finalValueVersion = 0;
    std::vector<uint64_t> intermediateValueVersions{};

    bool IsStrictlyOrdered(size_t expectedIntermediates) const noexcept {
        if (intermediateValueVersions.size() != expectedIntermediates) return false;
        uint64_t previous = sourceValueVersion;
        for (uint64_t value : intermediateValueVersions) {
            if (value <= previous) return false;
            previous = value;
        }
        return finalValueVersion > previous;
    }
};

} // namespace usdGen
#endif
