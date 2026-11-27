// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenToken (step 11.5): the engine's interner. Order parity with
// TfToken over 100k random texts (empty, ASCII, bytes >= 0x80, prefixes
// longer than 8; Tf's 8-byte compare code widens plain char, so a high byte
// in the first 8 is not plain unsigned order), deterministic FNV-1a hashes, identity equality, lookup
// without interning, the host conversions, and a 64-thread intern stress.

#include "usdGen/api/token.h"
#include "usdGen/api/ext/tokenScope.h"
#include "usdGenUsd/convert/token.h"

#include "pxr/base/tf/token.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, char const *what)
{
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_failures;
}

uint64_t Fnv1a(std::string const &s)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ull;
    }
    return h;
}

std::string RandomText(std::mt19937_64 &rng, std::vector<std::string> const &prefixes)
{
    std::uniform_int_distribution<int> kind(0, 9), len(0, 24), ascii(32, 126), byte(1, 255),
        pick(0, int(prefixes.size()) - 1);
    std::string s;
    int const k = kind(rng);
    if (k == 0) return s;                                      // empty
    if (k <= 3) s = prefixes[size_t(pick(rng))];               // shared prefix > 8 bytes
    int const n = len(rng);
    bool const high = k >= 7;                                  // bytes >= 0x80 included
    for (int i = 0; i < n; ++i) s.push_back(char(high ? byte(rng) : ascii(rng)));
    return s;
}

void BasicLaws()
{
    UsdGenToken const empty;
    Check(empty.IsEmpty() && empty.GetString().empty() && std::string(empty.GetText()).empty() &&
              empty.Hash() == 0 && empty.Id() == 0,
          "default token is empty");
    Check(UsdGenInternToken("").IsEmpty(), "interning empty text gives the empty token");

    uint64_t const before = UsdGenTotalInternCount();
    UsdGenToken const missing = UsdGenFindToken("usdGenTokenTest:neverInterned");
    Check(missing.IsEmpty() && UsdGenTotalInternCount() == before, "Find never interns");

    UsdGenToken const a = UsdGenInternToken("usdGenTokenTest:alpha");
    UsdGenToken const b = UsdGenInternToken(std::string("usdGenTokenTest:alpha"));
    Check(a == b && a.Id() == b.Id() && a.GetText() == b.GetText(), "same text, same rep");
    Check(UsdGenTotalInternCount() == before + 2, "each intern counts");
    Check(UsdGenFindToken("usdGenTokenTest:alpha") == a, "Find returns the interned token");
    Check(a.Hash() == size_t(Fnv1a("usdGenTokenTest:alpha")), "Hash is FNV-1a 64 of the bytes");
    Check(a.GetString() == "usdGenTokenTest:alpha" && a.size() == 21 &&
              a.View() == "usdGenTokenTest:alpha",
          "text accessors");
    Check(a == "usdGenTokenTest:alpha" && a != "usdGenTokenTest:beta" &&
              a == std::string("usdGenTokenTest:alpha"),
          "text comparisons");
    UsdGenToken const c = UsdGenInternToken("usdGenTokenTest:beta");
    Check(a != c && a.Id() != c.Id() && a < c && !(c < a), "distinct texts, distinct reps");

    std::string embedded("x\0y", 3);
    UsdGenToken const e = UsdGenInternToken(embedded);
    Check(e.size() == 3 && e.GetString() == embedded && e != UsdGenInternToken("x"),
          "embedded NUL keeps its bytes");
}

void OrderParity()
{
    std::mt19937_64 rng(0x5eed11u);
    std::vector<std::string> prefixes;
    for (int i = 0; i < 16; ++i) {
        std::string p = "sharedPrefix";
        p.push_back(char('a' + i));
        if (i & 1) p.push_back(char(0xC3));
        prefixes.push_back(p);
    }
    size_t const n = 100000;
    std::vector<UsdGenToken> ours(n);
    std::vector<PXR_NS::TfToken> theirs(n);
    for (size_t i = 0; i < n; ++i) {
        std::string const s = RandomText(rng, prefixes);
        ours[i] = UsdGenInternToken(s);
        theirs[i] = PXR_NS::TfToken(s);
    }
    std::vector<size_t> a(n), b(n);
    std::iota(a.begin(), a.end(), size_t(0));
    std::iota(b.begin(), b.end(), size_t(0));
    std::stable_sort(a.begin(), a.end(), [&](size_t x, size_t y) { return ours[x] < ours[y]; });
    std::stable_sort(b.begin(), b.end(), [&](size_t x, size_t y) { return theirs[x] < theirs[y]; });
    bool same = true;
    for (size_t i = 0; i < n && same; ++i)
        same = ours[a[i]].GetString() == theirs[b[i]].GetString();
    Check(same, "sorted order matches TfToken over 100k texts");

    bool pairwise = true;
    std::uniform_int_distribution<size_t> pick(0, n - 1);
    for (int i = 0; i < 200000 && pairwise; ++i) {
        size_t const x = pick(rng), y = pick(rng);
        pairwise = (ours[x] < ours[y]) == (theirs[x] < theirs[y]) &&
                   (ours[x] == ours[y]) == (theirs[x] == theirs[y]);
    }
    Check(pairwise, "pairwise < and == match TfToken");

    std::set<UsdGenToken> ourSet(ours.begin(), ours.end());
    std::set<PXR_NS::TfToken> theirSet(theirs.begin(), theirs.end());
    bool setSame = ourSet.size() == theirSet.size();
    auto it = theirSet.begin();
    for (UsdGenToken t : ourSet) {
        if (!setSame) break;
        setSame = t.GetString() == it->GetString();
        ++it;
    }
    Check(setSame, "std::set walks in TfToken's order");

    bool hashes = true;
    for (size_t i = 0; i < n && hashes; i += 7)
        hashes = ours[i].IsEmpty() ? ours[i].Hash() == 0
                                   : ours[i].Hash() == size_t(Fnv1a(ours[i].GetString()));
    Check(hashes, "hashes are FNV-1a of the bytes (deterministic across runs)");
}

void Conversions()
{
    PXR_NS::TfToken const tf("usdGenTokenTest:convert");
    UsdGenToken const t = usdGenUsd::ToToken(tf);
    Check(t.GetString() == tf.GetString(), "ToToken keeps the text");
    uint64_t const before = UsdGenTotalInternCount();
    uint64_t const execBefore = UsdGenExecInternCount();
    Check(usdGenUsd::ToToken(tf) == t && UsdGenTotalInternCount() == before,
          "ToToken of a known text does not intern");
    PXR_NS::TfToken const &back = usdGenUsd::ToTf(t);
    Check(back == tf && &back == &usdGenUsd::ToTf(t), "ToTf round-trips and is cached per id");
    Check(usdGenUsd::ToTf(UsdGenToken()).IsEmpty() && usdGenUsd::ToToken(PXR_NS::TfToken()).IsEmpty(),
          "empty converts to empty");
    (void)usdGenUsd::ToToken(PXR_NS::TfToken("usdGenTokenTest:hostNew"));
    Check(UsdGenExecInternCount() == execBefore, "host conversions are compile-phase interns");
}

void Stress()
{
    size_t const texts = 20000;
    std::vector<std::string> words(texts);
    for (size_t i = 0; i < texts; ++i) words[i] = "stress:" + std::to_string(i * 2654435761u % 1000003u);
    unsigned const threads = 64;
    std::vector<std::vector<UsdGenToken>> got(threads, std::vector<UsdGenToken>(texts));
    std::atomic<unsigned> ready{0};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            ready.fetch_add(1);
            while (ready.load() < threads) std::this_thread::yield();
            // Each thread walks the texts from its own offset and stride.
            size_t const stride = (t % 2) ? texts - 1 : 1;
            size_t i = (size_t(t) * 977u) % texts;
            for (size_t k = 0; k < texts; ++k, i = (i + stride) % texts) {
                got[t][i] = UsdGenInternToken(words[i]);
                UsdGenToken const f = UsdGenFindToken(words[i]);
                if (f != got[t][i]) got[t][i] = UsdGenToken();   // Find must see the intern
            }
        });
    }
    for (std::thread &th : pool) th.join();
    bool agree = true, text = true;
    for (size_t i = 0; i < texts; ++i) {
        for (unsigned t = 1; t < threads && agree; ++t) agree = got[t][i] == got[0][i];
        text = text && !got[0][i].IsEmpty() && got[0][i].GetString() == words[i];
    }
    Check(agree, "64 threads get one rep per text");
    Check(text, "every rep holds its text and Find sees it");
    std::set<uint32_t> ids;
    std::set<std::string> distinct(words.begin(), words.end());
    for (size_t i = 0; i < texts; ++i) ids.insert(got[0][i].Id());
    Check(ids.size() == distinct.size() && !ids.count(0), "ids are unique and non-zero");
}

}  // namespace

int main()
{
    BasicLaws();
    OrderParity();
    Conversions();
    Stress();
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
