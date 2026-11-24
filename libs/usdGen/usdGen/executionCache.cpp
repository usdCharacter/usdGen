#include "usdGen/executionCache.h"

#include <algorithm>
#include <array>

#include <limits>

namespace usdGen {
namespace {

uint64_t Mix(uint64_t hash, void const* data, size_t size) noexcept
{
    auto const* bytes = static_cast<unsigned char const*>(data);
    for (size_t i = 0; i != size; ++i) {
        hash ^= bytes[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

uint64_t MixU64(uint64_t hash, uint64_t value) noexcept
{
    // Fix the byte order explicitly so cache identities remain stable across
    // little- and big-endian hosts.
    for (unsigned shift = 0; shift != 64; shift += 8) {
        unsigned char const byte =
            static_cast<unsigned char>((value >> shift) & 0xffu);
        hash = Mix(hash, &byte, sizeof(byte));
    }
    return hash;
}

uint64_t MixPath(uint64_t hash, SdfPath const& path) noexcept
{
    std::string const text = path.GetString();
    hash = MixU64(hash, static_cast<uint64_t>(text.size()));
    return Mix(hash, text.data(), text.size());
}

void Canonicalize(std::vector<UsdGenExecutionInputGeneration>* values)
{
    std::sort(values->begin(), values->end(),
        [](auto const& a, auto const& b) { return a.identity < b.identity; });
}

void Coalesce(std::vector<UsdGenExecutionInputGeneration>* values, bool* valid)
{
    Canonicalize(values);
    std::vector<UsdGenExecutionInputGeneration> unique;
    unique.reserve(values->size());
    for (auto const& value : *values) {
        if (unique.empty() || unique.back().identity != value.identity) {
            unique.push_back(value);
            continue;
        }
        if (unique.back().generation != 0 && value.generation != 0 &&
            unique.back().generation != value.generation)
            *valid = false;
        // A resolved payload generation is stronger than a relationship-only
        // zero. On contradictory nonzero records, retain the newest value so
        // an accidental stale alias cannot be treated as current.
        unique.back().generation = std::max(unique.back().generation,
                                            value.generation);
    }
    *values = std::move(unique);
}

uint64_t HashValues(uint64_t hash,
                    std::vector<UsdGenExecutionInputGeneration> const& values,
                    uint64_t tag) noexcept
{
    hash = MixU64(hash, tag);
    hash = MixU64(hash, static_cast<uint64_t>(values.size()));
    for (auto const& value : values) {
        hash = MixPath(hash, value.identity);
        hash = MixU64(hash, value.generation);
    }
    return hash;
}

bool Update(std::vector<UsdGenExecutionInputGeneration>* values,
            SdfPath const& identity, uint64_t generation) noexcept
{
    bool found = false;
    for (auto& value : *values) {
        if (value.identity == identity) {
            value.generation = generation;
            found = true;
        }
    }
    return found;
}

} // namespace

UsdGenExecutionInputVersions UsdGenExecutionInputVersions::FromDescription(
    UsdGenGraphDesc const& desc)
{
    UsdGenExecutionInputVersions result;
    for (auto const& curve : desc.curveSets) {
        auto& bucket = curve.role == UsdGenRole::Reference
            ? result.references : result.sources;
        bucket.push_back({curve.path, curve.curveGeneration});
    }
    for (auto const& node : desc.nodes)
        for (auto const& reference : node.references)
            result.references.push_back({reference, 0});
    for (auto const& map : desc.maps)
        result.maps.push_back({map.path, map.textureGeneration});
    for (auto const& surface : desc.surfaces)
        result.surfaces.push_back({surface.path, surface.surfaceGeneration});
    Coalesce(&result.sources, &result.valid);
    Coalesce(&result.references, &result.valid);
    Coalesce(&result.maps, &result.valid);
    Coalesce(&result.surfaces, &result.valid);
    return result;
}

uint64_t UsdGenExecutionInputVersions::Hash() const noexcept
{
    uint64_t hash = 1469598103934665603ULL;
    hash = HashValues(hash, sources, 1);
    hash = HashValues(hash, references, 2);
    hash = HashValues(hash, maps, 3);
    hash = HashValues(hash, surfaces, 4);
    return MixU64(hash, valid ? 1 : 0);
}

bool UsdGenExecutionInputVersions::UpdateCurve(
    SdfPath const& identity, uint64_t generation) noexcept
{
    return Update(&sources, identity, generation) |
        Update(&references, identity, generation);
}

bool UsdGenExecutionInputVersions::UpdateMap(
    SdfPath const& identity, uint64_t generation) noexcept
{
    return Update(&maps, identity, generation);
}

bool UsdGenExecutionInputVersions::UpdateSurface(
    SdfPath const& identity, uint64_t generation) noexcept
{
    return Update(&surfaces, identity, generation);
}

UsdGenExecutionCacheKey UsdGenExecutionCacheKey::Make(
    SdfPath description, UsdGenEpoch planDigest,
    UsdGenExecutionInputVersions inputs,
    UsdGenExecutionContext context,
    uint64_t layoutDigest, double frame) noexcept
{
    UsdGenExecutionCacheKey result;
    result.description = std::move(description);
    result.planDigest = planDigest;
    result.inputs = std::move(inputs);
    result.context = context;
    result.layoutDigest = layoutDigest;
    std::memcpy(&result.frameBits, &frame, sizeof(frame));
    return result;
}

uint64_t UsdGenExecutionCacheKey::Hash() const noexcept
{
    uint64_t hash = MixPath(1469598103934665603ULL, description);
    hash = MixU64(hash, planDigest[0]);
    hash = MixU64(hash, planDigest[1]);
    hash = MixU64(hash, layoutDigest);
    hash = MixU64(hash, frameBits);
    hash = MixU64(hash, static_cast<uint64_t>(context.backend));
    hash = MixU64(hash, context.capabilityVersion);
    hash = MixU64(hash, static_cast<uint64_t>(context.deviceIndex));
    hash = MixU64(hash, context.deviceGeneration);
    hash = MixU64(hash, static_cast<uint64_t>(context.evaluationContext));
    return MixU64(hash, inputs.Hash());
}

UsdGenExecutionCacheDomain::SharedRegistry*
UsdGenExecutionCacheDomain::_SharedRegistry() noexcept
{
    static SharedRegistry* registry = new SharedRegistry;
    return registry;
}

} // namespace usdGen

// The cache store is header-only so callers can select a payload type and an
// injectable hasher without adding backend-specific explicit instantiations.
