// Immutable, backend-neutral decoded ImageMap cache.  This deliberately has
// no UsdStage or Hydra dependency so both graph-description builders can use
// the same pixel ownership contract.
#ifndef USDGEN_IMAGING_IMAGE_MAP_CACHE_H
#define USDGEN_IMAGING_IMAGE_MAP_CACHE_H

#include "usdGenUsd/graphDesc.h"
#include "usdGen/imagePayload.h"

#include <cstdint>
#include <memory>
#include <string>

namespace usdGenImaging {

struct UsdGenDecodedImageMap {
    std::shared_ptr<const usdGen::UsdGenImagePayload> payload;
    // Monotonic cache epoch identifying the exact decoded-pixel generation.
    uint64_t textureGeneration = 0;
    std::string error;

    explicit operator bool() const noexcept {
        return static_cast<bool>(payload);
    }
};

// Decodes `resolvedAssetPath` once per current global generation. Concurrent
// callers for the same key join the same load and receive the same immutable
// payload object. The path must already have been resolved by the caller's
// source-specific asset layer (UsdStage or Hydra); this API never opens a
// stage or reads a scene-index data source.
UsdGenDecodedImageMap ResolveUsdGenImageMap(
    std::string const& resolvedAssetPath);

// Bumps the global image generation and drops cache-owned references. Existing
// plans/jobs retain their shared immutable payloads and therefore remain safe.
// Returns the new generation, which is never zero.
uint64_t InvalidateUsdGenImageMapCache() noexcept;

// The generation that a newly resolved image will carry. Primarily useful to
// coordinate resolver-change handling before graph descriptor capture.
uint64_t CurrentUsdGenImageMapGeneration() noexcept;

// Re-resolves every ImageMap in one descriptor against a single coherent
// cache generation and stamps every map kind with that reload epoch. Decode
// diagnostics owned by this helper are replaced, not accumulated.
void ResolveUsdGenImageMaps(usdGen::UsdGenGraphDesc* desc);

} // namespace usdGenImaging

#endif // USDGEN_IMAGING_IMAGE_MAP_CACHE_H
