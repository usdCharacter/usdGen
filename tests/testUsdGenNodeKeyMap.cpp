// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenNodeKeyMap: host node keys (step 7.1).
//   - keys are FNV-1a64 of the path text: fixed values, stable across runs
//     and independent of keying order when nothing collides
//   - Key is idempotent; Find/Path round-trip; unknown and empty map to invalid
//   - a collision (or a zero hash) re-probes with a salt, deterministically
//   - KeysUnder returns the prim and its descendants, not text-prefix siblings

#include "usdGenUsd/nodeKeyMap.h"
#include "usdGen/api/nodeKey.h"

#include "pxr/usd/sdf/path.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using usdGen::UsdGenNodeKey;
using usdGenUsd::NodeKeyMap;

namespace {

int g_failures = 0;
void Check(bool ok, char const *what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

// Every path collides at salt 0; salt s hashes to s.
uint64_t AllCollide(std::string const &, uint32_t salt) { return salt; }
// Two paths share a hash; everything else is FNV.
uint64_t PairCollides(std::string const &text, uint32_t salt)
{
    if (salt == 0 && (text == "/x" || text == "/y")) {
        return 42u;
    }
    return NodeKeyMap::Hash(text, salt);
}

std::vector<SdfPath> Paths(std::vector<char const *> const &texts)
{
    std::vector<SdfPath> out;
    for (char const *t : texts) {
        out.emplace_back(t);
    }
    return out;
}

void TestKeyType()
{
    static_assert(!UsdGenNodeKey{}.IsValid(), "default key is invalid");
    constexpr UsdGenNodeKey a{1}, b{2};
    static_assert(a != b && a < b && a == UsdGenNodeKey{1}, "key comparisons");
    Check(usdGen::UsdGenNodeKeyHash()(UsdGenNodeKey{0x100000001ull}) == size_t(0x100000001ull ^ 1u),
          "key hash folds the high word");
}

void TestStableValues()
{
    // FNV-1a64 reference vectors; a change here breaks key stability.
    Check(NodeKeyMap::Hash("", 0) == 0xcbf29ce484222325ull, "FNV basis");
    Check(NodeKeyMap::Hash("a", 0) == 0xaf63dc4c8601ec8cull, "FNV-1a64('a')");
    Check(NodeKeyMap::Hash("a", 1) != NodeKeyMap::Hash("a", 0), "salt changes the hash");

    NodeKeyMap map;
    UsdGenNodeKey const desc = map.Key(SdfPath("/groom/Description"));
    UsdGenNodeKey const scatter = map.Key(SdfPath("/groom/Description/Scatter"));
    Check(desc.value == 0xfd0c4eed5861553dull, "description key is FNV-1a64 of its text");
    Check(scatter.value == 0x2e9ff9830bfa3cc4ull, "scatter key is FNV-1a64 of its text");

    // Same keys whatever the order, with no collision.
    std::vector<SdfPath> const paths =
        Paths({"/groom", "/groom/Description", "/groom/Description/Scatter",
               "/groom/Description/Scatter.points", "/groom/Mesh", "/groom/Mesh/Subset", "/other"});
    NodeKeyMap forward, backward;
    std::vector<UsdGenNodeKey> f, b;
    for (SdfPath const &p : paths) {
        f.push_back(forward.Key(p));
    }
    for (auto it = paths.rbegin(); it != paths.rend(); ++it) {
        b.push_back(backward.Key(*it));
    }
    std::reverse(b.begin(), b.end());
    Check(f == b, "keys do not depend on keying order");
    for (size_t i = 0; i < paths.size(); ++i) {
        Check(f[i].value == NodeKeyMap::Hash(paths[i].GetString(), 0), "uncollided key is the salt-0 hash");
    }
}

void TestRoundTrip()
{
    NodeKeyMap map;
    SdfPath const p("/groom/Description/Noise");
    UsdGenNodeKey const k = map.Key(p);
    Check(k.IsValid(), "key is valid");
    Check(map.Key(p) == k, "Key is idempotent");
    Check(map.Size() == 1, "one entry");
    Check(map.Find(p) == k, "Find returns the key");
    Check(map.Path(k) == p, "Path returns the path");
    Check(!map.Find(SdfPath("/never")).IsValid(), "unknown path has no key");
    Check(map.Path(UsdGenNodeKey{12345}).IsEmpty(), "unknown key has no path");
    Check(map.Path(UsdGenNodeKey{}).IsEmpty(), "invalid key has no path");
    Check(!map.Key(SdfPath()).IsValid(), "empty path keys invalid");
    Check(map.Size() == 1, "empty path is not stored");
    map.Clear();
    Check(map.Size() == 0 && map.Path(k).IsEmpty(), "Clear forgets every key");
    Check(map.Key(p) == k, "a cleared map keys the same path the same way");
}

void TestCollision()
{
    std::vector<SdfPath> const paths = Paths({"/a", "/b", "/c", "/d"});
    NodeKeyMap map(&AllCollide);
    std::unordered_set<uint64_t> seen;
    for (SdfPath const &p : paths) {
        UsdGenNodeKey const k = map.Key(p);
        Check(k.IsValid(), "re-probe skips the zero hash");
        Check(seen.insert(k.value).second, "re-probed keys are distinct");
    }
    Check(map.Key(paths[0]).value == 1 && map.Key(paths[3]).value == 4, "re-probe walks salts in order");
    for (SdfPath const &p : paths) {
        Check(map.Path(map.Find(p)) == p, "collided keys round-trip");
    }
    NodeKeyMap again(&AllCollide);
    for (SdfPath const &p : paths) {
        Check(again.Key(p) == map.Find(p), "re-probe is deterministic");
    }

    NodeKeyMap pair(&PairCollides);
    UsdGenNodeKey const x = pair.Key(SdfPath("/x"));
    UsdGenNodeKey const y = pair.Key(SdfPath("/y"));
    Check(x.value == 42u, "first of a colliding pair keeps the plain hash");
    Check(y.value == NodeKeyMap::Hash("/y", 1), "second re-probes with salt 1");
    Check(pair.Path(x) == SdfPath("/x") && pair.Path(y) == SdfPath("/y"), "colliding pair round-trips");
}

void TestKeysUnder()
{
    NodeKeyMap map;
    std::vector<SdfPath> const paths =
        Paths({"/ab", "/a/b/c", "/a", "/a/b.points", "/a/b", "/a/bc", "/b", "/a/b/c/d"});
    for (SdfPath const &p : paths) {
        map.Key(p);
    }
    auto under = [&](char const *prim) {
        std::vector<UsdGenNodeKey> keys;
        map.KeysUnder(SdfPath(prim), &keys);
        std::vector<std::string> texts;
        for (UsdGenNodeKey k : keys) {
            texts.push_back(map.Path(k).GetString());
        }
        return texts;
    };
    Check(under("/a/b") == std::vector<std::string>({"/a/b", "/a/b.points", "/a/b/c", "/a/b/c/d"}),
          "KeysUnder(/a/b): the prim, its property and its descendants in path order");
    std::vector<std::string> const a = under("/a");
    Check(a.size() == 6 && std::find(a.begin(), a.end(), "/ab") == a.end(),
          "KeysUnder(/a) excludes the text-prefix sibling /ab");
    Check(under("/").size() == paths.size(), "KeysUnder(/) returns every key");
    Check(under("/a/b/c/d") == std::vector<std::string>({"/a/b/c/d"}), "a leaf returns itself");
    Check(under("/zzz").empty(), "an unkeyed prim with no descendants returns nothing");
    Check(under("/a/b/x").empty(), "an unkeyed prim inside the range returns nothing");

    std::vector<UsdGenNodeKey> keys{UsdGenNodeKey{7}};
    map.KeysUnder(SdfPath("/b"), &keys);
    Check(keys.size() == 2 && keys[0].value == 7 && keys[1] == map.Find(SdfPath("/b")), "KeysUnder appends");
    map.KeysUnder(SdfPath("/b"), nullptr);
}

}  // namespace

int main()
{
    TestKeyType();
    TestStableValues();
    TestRoundTrip();
    TestCollision();
    TestKeysUnder();
    if (g_failures) {
        std::printf("testUsdGenNodeKeyMap: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("testUsdGenNodeKeyMap: ok\n");
    return 0;
}
