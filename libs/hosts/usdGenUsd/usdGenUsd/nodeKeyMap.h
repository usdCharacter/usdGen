// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// SdfPath <-> UsdGenNodeKey for one host graph builder (step 7.1). The only
// place a node key meets a path.
#pragma once
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/nodeKey.h"

#include "pxr/usd/sdf/path.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace usdGenUsd {

/// One per groom scene index / stage builder. Not thread-safe; owned by the
/// builder that keys its paths.
class NodeKeyMap {
public:
    /// Key hash of a path's text. Salt 0 is plain FNV-1a64 of the text; a
    /// salt > 0 folds its four bytes in first (collision re-probe).
    using HashFn = uint64_t (*)(std::string const &text, uint32_t salt);

    NodeKeyMap() noexcept;
    /// Tests pass a degenerate hash to force collisions.
    explicit NodeKeyMap(HashFn hash) noexcept;

    static uint64_t Hash(std::string const &text, uint32_t salt) noexcept;

    /// The key for 'path', assigned on first use: the first salt whose hash
    /// is non-zero and not taken. Stable across runs for the same paths keyed
    /// in the same order; without a collision it is Hash(text, 0). Invalid
    /// for the empty path.
    usdGen::UsdGenNodeKey Key(PXR_NS::SdfPath const &path);
    /// The key 'path' already has, else invalid.
    usdGen::UsdGenNodeKey Find(PXR_NS::SdfPath const &path) const;
    /// The path of 'key', else the empty path.
    PXR_NS::SdfPath const &Path(usdGen::UsdGenNodeKey key) const;
    /// Appends the keys of 'prim' and every keyed path under it (HasPrefix),
    /// in path order.
    void KeysUnder(PXR_NS::SdfPath const &prim, std::vector<usdGen::UsdGenNodeKey> *out) const;

    size_t Size() const noexcept { return _byPath.size(); }
    void Clear() noexcept;
    /// The input columns Lower packed with these keys (unchanged arrays are
    /// shared, not copied, by the next Lower).
    usdGen::UsdGenInputColumnCache &Columns() noexcept { return _columns; }

private:
    HashFn _hash;
    std::map<PXR_NS::SdfPath, usdGen::UsdGenNodeKey> _byPath;  // sorted: a prefix's descendants are contiguous
    std::unordered_map<usdGen::UsdGenNodeKey, PXR_NS::SdfPath, usdGen::UsdGenNodeKeyHash> _byKey;
    usdGen::UsdGenInputColumnCache _columns;
};

}  // namespace usdGenUsd
