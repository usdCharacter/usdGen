#include "usdGenImaging/imageMapCache.h"
#include "usdGen/sharedCell.h"

#include "usdGen/parkingWord.h"

#include "pxr/imaging/hio/image.h"
#include "pxr/imaging/hio/types.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {
namespace {

struct Key {
    std::string path;
    uint64_t generation = 0;
    bool operator==(Key const& rhs) const noexcept {
        return generation == rhs.generation && path == rhs.path;
    }
};
// A loading entry is inserted by CAS; the caller that inserted it decodes,
// and later callers for the key park on its state word until it completes.
struct Entry {
    enum : uint32_t { kLoading = 0, kReady = 1, kFailed = 2 };
    std::atomic<uint32_t> state{kLoading};
    // Written once, before the release store that leaves kLoading.
    std::shared_ptr<const UsdGenDecodedImageMap> result;
};
// Immutable once published. The generation lives in the table, so swapping
// the table is the invalidation. Image maps are few: linear lookup.
struct Table {
    uint64_t generation = 1;
    std::vector<std::pair<Key, std::shared_ptr<Entry>>> entries;
};
// Null until the first insert or invalidation: generation 1, no entries.
usdGen::UsdGenSharedCell<Table const> g_table;

uint64_t GenerationOf(std::shared_ptr<Table const> const& table) noexcept {
    return table ? table->generation : 1;
}

constexpr char kDecodePrefix[] = "[ImageMapDecode] ";

float HalfToFloat(uint16_t bits) noexcept {
    uint32_t const sign = uint32_t(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x3ffu;
    uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa) {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0) { mantissa <<= 1; --exponent; }
            mantissa &= 0x3ffu;
            result = sign | (exponent << 23) | (mantissa << 13);
        } else result = sign;
    } else if (exponent == 31) {
        result = sign | 0x7f800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float value;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

UsdGenDecodedImageMap Decode(std::string const& path, uint64_t generation) {
    UsdGenDecodedImageMap result;
    result.textureGeneration = generation;
    if (path.empty()) { result.error = "ImageMap has an empty resolved asset path"; return result; }
    HioImageSharedPtr image = HioImage::OpenForReading(
        path, 0, 0, HioImage::Raw, true);
    if (!image) { result.error = "HioImage could not open '" + path + "'"; return result; }
    int const width = image->GetWidth(), height = image->GetHeight();
    HioFormat const format = image->GetFormat();
    int const channels = HioGetComponentCount(format);
    HioType const type = HioGetHioType(format);
    if (width <= 0 || height <= 0 || channels < 1 || channels > 4 ||
        (type != HioTypeUnsignedByte && type != HioTypeUnsignedByteSRGB &&
         type != HioTypeHalfFloat &&
         type != HioTypeFloat)) {
        result.error = "ImageMap uses an unsupported Hio image format";
        return result;
    }
    size_t pixels = size_t(width);
    if (pixels > std::numeric_limits<size_t>::max() / size_t(height) ||
        (pixels *= size_t(height)) > std::numeric_limits<size_t>::max() / size_t(channels)) {
        result.error = "ImageMap dimensions overflow host storage";
        return result;
    }
    size_t const componentBytes = HioGetDataSizeOfType(type);
    if (!componentBytes || pixels > std::numeric_limits<size_t>::max() /
            (size_t(channels) * componentBytes)) {
        result.error = "ImageMap storage size overflows host storage";
        return result;
    }
    std::vector<uint8_t> source(pixels * size_t(channels) * componentBytes);
    HioImage::StorageSpec storage;
    storage.width = width; storage.height = height; storage.depth = 1;
    storage.format = format; storage.flipped = false; storage.data = source.data();
    if (!image->Read(storage)) { result.error = "HioImage read failed for '" + path + "'"; return result; }
    std::vector<float> texels(pixels * size_t(channels));
    for (size_t i = 0; i != texels.size(); ++i) {
        uint8_t const* value = source.data() + i * componentBytes;
        if (type == HioTypeUnsignedByte || type == HioTypeUnsignedByteSRGB)
            texels[i] = float(*value) / 255.0f;
        else if (type == HioTypeHalfFloat) {
            uint16_t raw; std::memcpy(&raw, value, sizeof(raw)); texels[i] = HalfToFloat(raw);
        } else std::memcpy(&texels[i], value, sizeof(float));
    }
    std::string error;
    // Hio's non-flipped storage is retained as top-down rows. Sampling owns
    // the explicit UV conversion through ImagePayload::Orientation.
    result.payload = usdGen::UsdGenImagePayload::Create(uint32_t(width), uint32_t(height),
        uint32_t(channels), std::move(texels),
        usdGen::UsdGenImageRowOrientation::TopDown, &error);
    if (!result.payload) result.error = error.empty() ? "ImageMap payload creation failed" : error;
    return result;
}

} // namespace

uint64_t CurrentUsdGenImageMapGeneration() noexcept {
    return GenerationOf(g_table.Load());
}

void ResolveUsdGenImageMaps(usdGen::UsdGenGraphDesc* desc) {
    if (!desc) return;
    std::string const prefix = kDecodePrefix;
    desc->validationErrors.erase(std::remove_if(
        desc->validationErrors.begin(), desc->validationErrors.end(),
        [&](std::string const& error) { return error.rfind(prefix, 0) == 0; }),
        desc->validationErrors.end());
    std::vector<UsdGenDecodedImageMap> decoded(desc->maps.size());
    uint64_t generation = 0;
    for (;;) {
        generation = CurrentUsdGenImageMapGeneration();
        bool coherent = true;
        for (size_t i = 0; i != desc->maps.size(); ++i) {
            auto const& map = desc->maps[i];
            decoded[i] = {};
            if (map.type != TfToken("UsdGenImageMap")) continue;
            decoded[i] = ResolveUsdGenImageMap(map.resolvedAssetPath);
            if (decoded[i].textureGeneration != generation) {
                coherent = false;
                break;
            }
        }
        if (coherent && generation == CurrentUsdGenImageMapGeneration()) break;
    }
    for (size_t i = 0; i != desc->maps.size(); ++i) {
        auto& map = desc->maps[i];
        map.textureGeneration = generation;
        if (map.type != TfToken("UsdGenImageMap")) continue;
        map.imagePayload = std::move(decoded[i].payload);
        if (!decoded[i].error.empty())
            desc->validationErrors.push_back(prefix + map.path.GetString() +
                ": " + decoded[i].error);
    }
}

uint64_t InvalidateUsdGenImageMapCache() noexcept {
    // Swap in an empty table of the next generation. The old table, its
    // entries and payloads retire through the epochs; in-flight plans retain
    // their own shared references.
    for (;;) {
        std::shared_ptr<Table const> current = g_table.Load();
        uint64_t const generation = GenerationOf(current);
        // Wrapping would make a stale decoded payload current again.
        if (generation == std::numeric_limits<uint64_t>::max()) std::terminate();
        auto next = std::make_shared<Table>();
        next->generation = generation + 1;
        if (g_table.CompareExchange(current, std::move(next))) return generation + 1;
    }
}

UsdGenDecodedImageMap ResolveUsdGenImageMap(std::string const& path) {
    for (;;) {
        std::shared_ptr<Table const> table = g_table.Load();
        uint64_t const generation = GenerationOf(table);
        Key const key{path, generation};
        std::shared_ptr<Entry> entry;
        if (table) {
            for (auto const& item : table->entries)
                if (item.first == key) { entry = item.second; break; }
        }
        if (!entry) {
            // Insert a loading entry; a lost CAS (another insert or an
            // invalidation) retries against the new table.
            auto candidate = std::make_shared<Entry>();
            auto next = std::make_shared<Table>();
            next->generation = generation;
            if (table) next->entries = table->entries;
            next->entries.push_back({key, candidate});
            if (!g_table.CompareExchange(table, std::move(next))) continue;
            entry = std::move(candidate);
        } else {
            uint32_t state;
            while ((state = entry->state.load(std::memory_order_acquire)) == Entry::kLoading)
                usdGen::UsdGenParkWhileEqual(entry->state, Entry::kLoading);
            if (generation != CurrentUsdGenImageMapGeneration()) continue;
            auto const& result = entry->result;
            return result ? *result : UsdGenDecodedImageMap{
                {}, generation, "ImageMap cache entry completed without a result"};
        }
        // This caller inserted the entry: decode outside any shared state.
        // A throwing decode must still complete the entry, or every waiter
        // on it parks forever.
        UsdGenDecodedImageMap decoded;
        std::shared_ptr<const UsdGenDecodedImageMap> completed;
        try {
            decoded = Decode(path, generation);
            completed = std::make_shared<const UsdGenDecodedImageMap>(decoded);
        } catch (std::exception const& error) {
            decoded = UsdGenDecodedImageMap{{}, generation,
                std::string("ImageMap decode threw: ") + error.what()};
        } catch (...) {
            decoded = UsdGenDecodedImageMap{{}, generation, "ImageMap decode threw"};
        }
        if (!completed) {
            try { completed = std::make_shared<const UsdGenDecodedImageMap>(decoded); }
            catch (...) {}   // waiters then report a missing result
        }
        entry->result = std::move(completed);
        entry->state.store(decoded ? Entry::kReady : Entry::kFailed, std::memory_order_release);
        usdGen::UsdGenUnparkAll(entry->state);
        if (generation != CurrentUsdGenImageMapGeneration()) continue;
        return decoded;
    }
}

} // namespace usdGenImaging
