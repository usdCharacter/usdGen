#include "usdGenImaging/imageMapCache.h"

#include "pxr/imaging/hio/image.h"
#include "pxr/imaging/hio/types.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
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
struct Entry {
    enum class State : uint8_t { Loading, Ready, Failed };
    State state = State::Loading;
    std::shared_ptr<const UsdGenDecodedImageMap> result;
    std::condition_variable complete;
};
using Table = std::vector<std::pair<Key, std::shared_ptr<Entry>>>;
struct Cache {
    std::mutex mutex;
    // A copy-on-write table keeps all entry/table allocation and retirement
    // out of the mutex. Image maps are few, so linear lookup is intentional.
    std::shared_ptr<const Table> entries = std::make_shared<Table>();
    std::atomic<uint64_t> generation{1};
};
Cache& GetCache() {
    static auto* cache = new Cache;
    return *cache;
}

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
    return GetCache().generation.load(std::memory_order_acquire);
}

void ResolveUsdGenImageMaps(usdGen::UsdGenGraphDesc* desc) {
    if (!desc) return;
    static std::string const prefix = "[ImageMapDecode] ";
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
    Cache& cache = GetCache();
    // Move all owners out while locked, but let every payload/entry destructor
    // run after unlock. In-flight plans retain their own shared references.
    auto empty = std::make_shared<Table>();
    std::shared_ptr<const Table> retired;
    uint64_t next = 0;
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        uint64_t const current = cache.generation.load(std::memory_order_relaxed);
        // Wrapping would make a stale decoded payload current again.
        if (current == std::numeric_limits<uint64_t>::max()) std::terminate();
        next = current + 1;
        cache.generation.store(next, std::memory_order_release);
        retired = std::move(cache.entries);
        cache.entries = std::move(empty);
    }
    return next;
}

UsdGenDecodedImageMap ResolveUsdGenImageMap(std::string const& path) {
    Cache& cache = GetCache();
    auto find = [](std::shared_ptr<const Table> const& table, Key const& key) {
        auto it = std::find_if(table->begin(), table->end(), [&](auto const& item) {
            return item.first == key;
        });
        return it == table->end() ? std::shared_ptr<Entry>() : it->second;
    };
    for (;;) {
        uint64_t const generation = CurrentUsdGenImageMapGeneration();
        Key const key{path, generation};
        std::shared_ptr<Entry> entry;
        bool decode = false;
        {
            std::unique_lock<std::mutex> lock(cache.mutex);
            if (generation != cache.generation.load(std::memory_order_acquire))
                continue;
            auto table = cache.entries;
            entry = find(table, key);
            if (!entry) {
                lock.unlock();
                auto candidate = std::make_shared<Entry>();
                auto replacement = std::make_shared<Table>(*table);
                replacement->push_back({key, candidate});
                lock.lock();
                if (generation != cache.generation.load(std::memory_order_acquire) ||
                    cache.entries != table)
                    continue;
                cache.entries = std::move(replacement);
                entry = std::move(candidate); decode = true;
            }
            if (!decode && entry->state == Entry::State::Loading)
                entry->complete.wait(lock, [&] { return entry->state != Entry::State::Loading; });
            if (!decode && generation != CurrentUsdGenImageMapGeneration()) {
                lock.unlock();
                continue;
            }
            if (!decode) {
                auto result = entry->result;
                lock.unlock();
                return result ? *result : UsdGenDecodedImageMap{
                    {}, generation, "ImageMap cache entry completed without a result"};
            }
            lock.unlock();
        }
        // Decode, image allocation, and ImagePayload construction occur with
        // no cache mutex held.
        // A throwing decode must still complete the entry, or every waiter
        // on it blocks forever.
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
        {
            std::lock_guard<std::mutex> lock(cache.mutex);
            entry->result = std::move(completed);
            entry->state = decoded ? Entry::State::Ready : Entry::State::Failed;
            entry->complete.notify_all();
        }
        if (generation != CurrentUsdGenImageMapGeneration()) continue;
        return decoded;
    }
}

} // namespace usdGenImaging
