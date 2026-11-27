// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenCowArray (step 11.6): UsdGenCowArray detaches exactly where
// VtArray does (each operation runs on both and the address moves on both or
// on neither), identity (IsIdentical, Arena), external arenas (never unique,
// released once), and the host conversions (ToCow, ToVt) are zero copy,
// proven by address.

#include "usdGen/api/arena.h"
#include "usdGen/api/cowArray.h"
#include "usdGen/api/text.h"
#include "usdGenUsd/convert/array.h"
#include "usdGenUsd/convert/math.h"
#include "usdGenUsd/convert/token.h"

#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, char const *what)
{
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_failures;
}

struct CountedExternal final : UsdGenExternal {
    std::atomic<int> *released;
    explicit CountedExternal(std::atomic<int> *r) : released(r) {}
    void Release() noexcept override
    {
        released->fetch_add(1);
        delete this;
    }
};

/// Runs 'op' on a VtArray and a UsdGenCowArray set up alike by 'setup' and
/// reports whether each one's data address moved.
template <class Setup, class Op>
void SameDetach(char const *what, Setup setup, Op op)
{
    PXR_NS::VtArray<int> vt;
    UsdGenCowArray<int> cow;
    PXR_NS::VtArray<int> vtKeep;
    UsdGenCowArray<int> cowKeep;
    setup(vt, vtKeep);
    setup(cow, cowKeep);
    int const *vtBefore = vt.cdata();
    int const *cowBefore = cow.cdata();
    op(vt);
    op(cow);
    bool const vtMoved = vt.cdata() != vtBefore;
    bool const cowMoved = cow.cdata() != cowBefore;
    bool const same = vtMoved == cowMoved && vt.size() == cow.size() &&
        std::equal(vt.cbegin(), vt.cend(), cow.cbegin()) && vtKeep.size() == cowKeep.size() &&
        std::equal(vtKeep.cbegin(), vtKeep.cend(), cowKeep.cbegin());
    std::string const text = std::string(what) + (vtMoved ? " (detaches)" : " (in place)");
    Check(same, text.c_str());
}

void DetachParity()
{
    auto unique = [](auto &a, auto &) { a.assign({1, 2, 3, 4}); };
    auto shared = [](auto &a, auto &keep) {
        a.assign({1, 2, 3, 4});
        keep = a;
    };
    SameDetach("unique: non-const data()", unique, [](auto &a) { a.data()[0] = 9; });
    SameDetach("shared: non-const data()", shared, [](auto &a) { a.data()[0] = 9; });
    SameDetach("unique: operator[]", unique, [](auto &a) { a[1] = 7; });
    SameDetach("shared: operator[]", shared, [](auto &a) { a[1] = 7; });
    SameDetach("shared: const read", shared, [](auto &a) {
        auto const &c = a;
        (void)c[0];
        (void)c.cdata();
        (void)c.begin();
    });
    SameDetach("unique: shrink", unique, [](auto &a) { a.resize(2); });
    SameDetach("shared: shrink", shared, [](auto &a) { a.resize(2); });
    SameDetach("unique: same size resize", unique, [](auto &a) { a.resize(4); });
    SameDetach("shared: same size resize", shared, [](auto &a) { a.resize(4); });
    SameDetach("shared: pop_back", shared, [](auto &a) { a.pop_back(); });
    SameDetach("unique: pop_back", unique, [](auto &a) { a.pop_back(); });
}

void Basics()
{
    UsdGenCowArray<float> empty;
    Check(empty.empty() && !empty.cdata() && !empty.Arena() && empty.capacity() == 0, "default is empty");
    UsdGenCowArray<UsdGenVec3f> zeros(3);
    Check(zeros.size() == 3 && zeros[2] == UsdGenVec3f(0, 0, 0), "size constructor value-initialises");
    UsdGenCowArray<int> filled(4, 7);
    Check(filled.size() == 4 && filled[3] == 7, "fill constructor");
    UsdGenCowArray<int> list{1, 2, 3};
    std::vector<int> const v{4, 5};
    UsdGenCowArray<int> range(v.begin(), v.end());
    Check(list.size() == 3 && list[1] == 2 && range.size() == 2 && range[1] == 5, "list and range constructors");

    UsdGenCowArray<int> copy = list;
    Check(copy.IsIdentical(list) && copy.cdata() == list.cdata() && copy.Arena() == list.Arena(),
          "a copy shares the arena");
    Check(!list.IsUnique() && !copy.IsUnique(), "shared arrays are not unique");
    copy.push_back(4);
    Check(!copy.IsIdentical(list) && list.size() == 3 && copy.size() == 4 && copy[3] == 4,
          "push_back on a shared array detaches");
    Check(copy.IsUnique() && list.IsUnique(), "both are unique after the detach");
    Check(copy == UsdGenCowArray<int>({1, 2, 3, 4}) && copy != list, "elementwise equality");

    UsdGenCowArray<int> grow;
    for (int i = 0; i < 1000; ++i) grow.push_back(i);
    bool seq = grow.size() == 1000;
    for (int i = 0; seq && i < 1000; ++i) seq = grow[size_t(i)] == i;
    Check(seq && grow.capacity() >= 1000, "push_back grows");
    int const *before = grow.cdata();
    grow.clear();
    Check(grow.empty() && grow.cdata() == before && grow.capacity() >= 1000, "clear keeps a unique buffer");
    grow.resize(10, 3);
    Check(grow.cdata() == before && grow[9] == 3, "resize within capacity stays in place");
    grow.insert(grow.begin() + 1, {8, 9});
    grow.erase(grow.begin());
    Check(grow.size() == 11 && grow[0] == 8 && grow[1] == 9 && grow[2] == 3, "insert and erase");

    UsdGenCowArray<float> a{0.0f, 1.0f}, b{-0.0f, 1.0f};
    Check(a == b && hash_value(a) == hash_value(b), "+0 and -0 are equal and hash alike");
    UsdGenCowArray<float> c{1.0f};
    Check(hash_value(a) != hash_value(c), "different arrays hash apart");

    UsdGenText const text("hello");
    UsdGenText const same = text;
    Check(text.View() == "hello" && std::string(text.c_str()) == "hello" && same == text, "text");
}

void External()
{
    std::atomic<int> released{0};
    std::vector<int> host{1, 2, 3};
    {
        UsdGenCowArray<int> a = UsdGenCowArray<int>::Adopt(host.data(), host.size(), new CountedExternal(&released));
        Check(a.cdata() == host.data() && a.Arena() && a.Arena()->IsExternal(), "Adopt is zero copy");
        Check(!a.IsUnique(), "an external array is never unique");
        UsdGenCowArray<int> b = a;
        b[0] = 5;
        Check(b.cdata() != host.data() && host[0] == 1 && b[0] == 5 && a.cdata() == host.data(),
              "writing detaches from host memory");
        Check(released.load() == 0, "the owner lives while a copy does");
    }
    Check(released.load() == 1, "the owner is released once");
    UsdGenCowArray<int> none = UsdGenCowArray<int>::Adopt(host.data(), 0, new CountedExternal(&released));
    Check(none.empty() && released.load() == 2, "an empty adoption releases at once");
}

void Share()
{
    UsdGenRef<UsdGenArena> arena = UsdGenArena::Acquire(64 * sizeof(float));
    float *payload = reinterpret_cast<float *>(arena->Payload());
    for (int i = 0; i < 64; ++i) payload[i] = float(i);
    UsdGenCowArray<float> view = UsdGenCowArray<float>::Share(UsdGenRef<const UsdGenArena>(arena), payload + 8, 4);
    UsdGenCowArray<float> const &read = view;   // const reads never detach
    Check(read.size() == 4 && read.cdata() == payload + 8 && read[0] == 8.0f && read.Arena() == arena.Get(),
          "Share views a column");
    arena->MarkShared();
    arena.Reset();
    float const *at = read.cdata();
    Check(!view.IsUnique(), "a view of a published arena is not unique");
    view.data()[0] = 1.0f;
    Check(read.cdata() != at && read[0] == 1.0f && read[1] == 9.0f,
          "a published arena is never written in place");
}

void Conversions()
{
    PXR_NS::VtArray<PXR_NS::GfVec3f> host{PXR_NS::GfVec3f(1, 2, 3), PXR_NS::GfVec3f(4, 5, 6)};
    UsdGenVec3fArray const engine = usdGenUsd::ToCow(host);
    Check(static_cast<void const *>(engine.cdata()) == static_cast<void const *>(host.cdata()) &&
              engine[1] == UsdGenVec3f(4, 5, 6),
          "ToCow adopts the host buffer");
    UsdGenVec3fArray const again = usdGenUsd::ToCow(host);
    Check(again.cdata() == engine.cdata(), "an unchanged host array keeps its address");
    PXR_NS::VtArray<PXR_NS::GfVec3f> back = usdGenUsd::ToVt(engine);
    Check(static_cast<void const *>(back.cdata()) == static_cast<void const *>(engine.cdata()) && back == host,
          "ToVt hands the engine buffer to a VtArray");

    PXR_NS::VtArray<PXR_NS::GfVec3f> outlives;
    {
        UsdGenVec3fArray made(2, UsdGenVec3f(7, 8, 9));
        outlives = usdGenUsd::ToVt(made);
    }
    Check(outlives.size() == 2 && outlives[1] == PXR_NS::GfVec3f(7, 8, 9), "the VtArray pins the arena");
    outlives[0] = PXR_NS::GfVec3f(0, 0, 0);
    Check(outlives[0] == PXR_NS::GfVec3f(0, 0, 0) && outlives[1] == PXR_NS::GfVec3f(7, 8, 9),
          "a foreign VtArray detaches on write");

    PXR_NS::VtArray<float> floats{1.5f, 2.5f};
    UsdGenFloatArray const f = usdGenUsd::ToCow(floats);
    Check(f.cdata() == floats.cdata() && f[1] == 2.5f, "scalar arrays adopt too");

    PXR_NS::VtArray<PXR_NS::TfToken> tokens{PXR_NS::TfToken("a"), PXR_NS::TfToken("bc")};
    UsdGenTokenArray const t = usdGenUsd::ToCow(tokens);
    PXR_NS::VtArray<PXR_NS::TfToken> const tokensBack = usdGenUsd::ToVt(t);
    Check(t.size() == 2 && t[1] == "bc" && tokensBack == tokens, "token arrays convert per element");

    usdGen::UsdGenValue const value = usdGenUsd::ToEngineValue(PXR_NS::VtValue(floats));
    Check(value.IsHolding<UsdGenFloatArray>() && value.UncheckedGet<UsdGenFloatArray>().cdata() == floats.cdata(),
          "ToEngineValue holds the engine array, zero copy");
    PXR_NS::VtValue const hostValue = usdGenUsd::ToHostValue(value);
    Check(hostValue.IsHolding<PXR_NS::VtArray<float>>() && hostValue.UncheckedGet<PXR_NS::VtArray<float>>() == floats,
          "ToHostValue gives the VtArray back");
    usdGen::UsdGenValue const vec = usdGenUsd::ToEngineValue(PXR_NS::VtValue(host));
    Check(vec.IsHolding<UsdGenVec3fArray>() && usdGenUsd::ToHostValue(vec).IsHolding<PXR_NS::VtArray<PXR_NS::GfVec3f>>(),
          "math arrays round trip through values");
}

}  // namespace

int main()
{
    DetachParity();
    Basics();
    External();
    Share();
    Conversions();
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
