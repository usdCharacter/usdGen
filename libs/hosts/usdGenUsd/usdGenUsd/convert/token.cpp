// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenUsd/convert/token.h"
#include "usdGen/registries/token.h"

#include <atomic>

namespace usdGenUsd {

namespace {
// Id -> TfToken, two levels, chunks published by CAS; never freed.
constexpr uint32_t kChunkBits = 10;
constexpr uint32_t kChunkSize = uint32_t(1) << kChunkBits;
constexpr uint32_t kChunks = 4096;
using Entry = std::atomic<PXR_NS::TfToken const *>;
std::atomic<Entry *> gChunks[kChunks];

Entry *ChunkOf(uint32_t index)
{
    Entry *chunk = gChunks[index].load(std::memory_order_acquire);
    if (chunk) return chunk;
    Entry *fresh = new Entry[kChunkSize]();
    if (gChunks[index].compare_exchange_strong(chunk, fresh, std::memory_order_acq_rel,
                                               std::memory_order_acquire))
        return fresh;
    delete[] fresh;
    return chunk;
}

PXR_NS::TfToken const gEmpty;
}  // namespace

usdGen::UsdGenToken ToToken(std::string_view text)
{
    if (text.empty()) return usdGen::UsdGenToken();
    usdGen::UsdGenToken found = usdGen::UsdGenFindToken(text);
    if (!found.IsEmpty()) return found;
    usdGen::UsdGenCompileTokenScope scope;   // host data entering the engine
    return usdGen::UsdGenInternToken(text);
}

usdGen::UsdGenToken ToToken(PXR_NS::TfToken const &t)
{
    return ToToken(std::string_view(t.GetText(), t.size()));
}

PXR_NS::TfToken const &ToTf(usdGen::UsdGenToken t)
{
    if (t.IsEmpty()) return gEmpty;
    uint32_t const id = t.Id();
    uint32_t const chunk = id >> kChunkBits;
    if (chunk >= kChunks) return *new PXR_NS::TfToken(t.GetString());   // past 4M ids: leaks
    Entry &entry = ChunkOf(chunk)[id & (kChunkSize - 1)];
    PXR_NS::TfToken const *tf = entry.load(std::memory_order_acquire);
    if (tf) return *tf;
    auto *fresh = new PXR_NS::TfToken(t.GetString());
    if (entry.compare_exchange_strong(tf, fresh, std::memory_order_acq_rel, std::memory_order_acquire))
        return *fresh;
    delete fresh;
    return *tf;
}

usdGen::UsdGenTokenVector ToTokens(std::vector<PXR_NS::TfToken> const &t)
{
    usdGen::UsdGenTokenVector out;
    out.reserve(t.size());
    for (PXR_NS::TfToken const &x : t) out.push_back(ToToken(x));
    return out;
}

std::vector<PXR_NS::TfToken> ToTfs(usdGen::UsdGenTokenVector const &t)
{
    std::vector<PXR_NS::TfToken> out;
    out.reserve(t.size());
    for (usdGen::UsdGenToken x : t) out.push_back(ToTf(x));
    return out;
}

usdGen::UsdGenTokenArray ToTokens(PXR_NS::VtArray<PXR_NS::TfToken> const &t)
{
    usdGen::UsdGenTokenArray out(t.size());
    usdGen::UsdGenToken *dst = out.data();
    for (size_t i = 0; i < t.size(); ++i) dst[i] = ToToken(t.cdata()[i]);
    return out;
}

PXR_NS::VtArray<PXR_NS::TfToken> ToTfs(usdGen::UsdGenTokenArray const &t)
{
    PXR_NS::VtArray<PXR_NS::TfToken> out(t.size());
    PXR_NS::TfToken *dst = out.data();
    for (size_t i = 0; i < t.size(); ++i) dst[i] = ToTf(t.cdata()[i]);
    return out;
}

usdGen::UsdGenTokenArray ToCow(PXR_NS::VtArray<PXR_NS::TfToken> const &a) { return ToTokens(a); }

PXR_NS::VtArray<PXR_NS::TfToken> ToVt(usdGen::UsdGenTokenArray const &a) { return ToTfs(a); }

}  // namespace usdGenUsd
