// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenUsd/nodeKeyMap.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenUsd {

using usdGen::UsdGenNodeKey;

namespace {
constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x100000001b3ull;

inline uint64_t FnvByte(uint64_t h, unsigned char c) noexcept { return (h ^ c) * kFnvPrime; }
}  // namespace

NodeKeyMap::NodeKeyMap() noexcept : _hash(&NodeKeyMap::Hash) {}

NodeKeyMap::NodeKeyMap(HashFn hash) noexcept : _hash(hash ? hash : &NodeKeyMap::Hash) {}

uint64_t NodeKeyMap::Hash(std::string const &text, uint32_t salt) noexcept
{
    uint64_t h = kFnvBasis;
    if (salt != 0) {
        for (int i = 0; i < 4; ++i) {
            h = FnvByte(h, static_cast<unsigned char>((salt >> (8 * i)) & 0xFFu));
        }
    }
    for (char c : text) {
        h = FnvByte(h, static_cast<unsigned char>(c));
    }
    return h;
}

UsdGenNodeKey NodeKeyMap::Key(SdfPath const &path)
{
    if (path.IsEmpty()) {
        return UsdGenNodeKey{};
    }
    auto const found = _byPath.find(path);
    if (found != _byPath.end()) {
        return found->second;
    }
    std::string const &text = path.GetString();
    UsdGenNodeKey key;
    for (uint32_t salt = 0;; ++salt) {
        key.value = _hash(text, salt);
        if (key.IsValid() && _byKey.find(key) == _byKey.end()) {
            break;
        }
    }
    _byKey.emplace(key, path);
    _byPath.emplace(path, key);
    return key;
}

UsdGenNodeKey NodeKeyMap::Find(SdfPath const &path) const
{
    auto const it = _byPath.find(path);
    return it == _byPath.end() ? UsdGenNodeKey{} : it->second;
}

SdfPath const &NodeKeyMap::Path(UsdGenNodeKey key) const
{
    auto const it = _byKey.find(key);
    return it == _byKey.end() ? SdfPath::EmptyPath() : it->second;
}

void NodeKeyMap::KeysUnder(SdfPath const &prim, std::vector<UsdGenNodeKey> *out) const
{
    if (!out || prim.IsEmpty()) {
        return;
    }
    for (auto it = _byPath.lower_bound(prim); it != _byPath.end() && it->first.HasPrefix(prim); ++it) {
        out->push_back(it->second);
    }
}

void NodeKeyMap::Clear() noexcept
{
    _byPath.clear();
    _byKey.clear();
}

}  // namespace usdGenUsd
