// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenElementSet: the generic data model in isolation (step 5.1).
//   - span, channel descriptors and the channel table's run order
//   - arenas: 256-aligned payload, refcount on its own line, unique serials
//   - Fresh with offsets, Derive sharing unwritten columns by pointer
//   - Recycle reuses the delta arena in place, refuses once exported or held
//   - compaction past kUsdGenMaxArenasPerSet arenas keeps every byte
//   - the second Derive/Recycle/Plan/Seal cycle allocates nothing
//   - primitive layouts register through the type registry

#include "usdGen/api/elementSet.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/span.h"
#include "usdGen/api/writeSet.h"
#include "usdGen/exec/allocCounter.h"
#include "usdGen/registries/registrar.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, char const *what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

bool Aligned(void const *p) { return reinterpret_cast<uintptr_t>(p) % UsdGenArena::kColumnAlign == 0; }

UsdGenChannelDesc Desc(char const *name, UsdGenScalar scalar, uint8_t arity, uint8_t level,
                       UsdGenChannelRole role = UsdGenChannelRole::Other, uint8_t flags = 0)
{
    UsdGenChannelDesc d;
    d.name = UsdGenToken(name);
    d.scalar = scalar;
    d.arity = arity;
    d.level = level;
    d.role = role;
    d.flags = flags;
    return d;
}

enum : UsdGenSlot { kPx, kWidth, kId, kUv, kStatics };

struct TestLayout {
    UsdGenChannelDesc statics[kStatics];
    UsdGenElementLayout layout;
    TestLayout()
    {
        statics[kPx] = Desc("px", UsdGenScalar::F32, 1, 2, UsdGenChannelRole::Position);
        statics[kWidth] = Desc("width", UsdGenScalar::F32, 1, 2, UsdGenChannelRole::Width);
        statics[kId] = Desc("id", UsdGenScalar::U64, 1, 1, UsdGenChannelRole::Id);
        statics[kUv] = Desc("uv", UsdGenScalar::F32, 2, 1, UsdGenChannelRole::Uv, kUsdGenChannelPacked);
        layout.levels = 2;
        layout.channels = UsdGenSpan<const UsdGenChannelDesc>(statics, kStatics);
    }
};

TestLayout const &Layout()
{
    static TestLayout const *t = new TestLayout;   // test only
    return *t;
}

void TestSpan()
{
    int v[3] = {1, 2, 3};
    UsdGenSpan<int> s(v, 3);
    UsdGenSpan<const int> c = s;
    Check(c.size() == 3 && c.front() == 1 && c.back() == 3 && c.cdata() == v, "span conversion");
    int sum = 0;
    for (int x : c) sum += x;
    Check(sum == 6 && !c.empty() && UsdGenSpan<int>().empty(), "span iteration");
    Check(UsdGenScalarBytes(UsdGenScalar::U64) == 8 && UsdGenScalarBytes(UsdGenScalar::F32) == 4,
          "scalar bytes");
    UsdGenChunkRange a, b;
    b.first[1] = 4;
    Check(a == a && a != b, "chunk range equality");
}

void TestChannelTable()
{
    UsdGenElementLayout const &layout = Layout().layout;
    UsdGenChannelDesc const dyn[] = {
        Desc("zeta", UsdGenScalar::F32, 1, 2),
        Desc("alpha", UsdGenScalar::I32, 1, 1),
        Desc("beta", UsdGenScalar::F32, 3, 0),
        Desc("gamma", UsdGenScalar::F32, 1, 2),
    };
    auto t = UsdGenChannelTable::Make(layout, UsdGenSpan<const UsdGenChannelDesc>(dyn, 4));
    Check(t->Size() == kStatics + 4 && t->StaticCount() == kStatics, "table size");
    // Runs: (level <= 1) alpha, beta; (level 2) gamma, zeta.
    Check((*t)[kStatics + 0].name == UsdGenToken("alpha") && (*t)[kStatics + 1].name == UsdGenToken("beta") &&
              (*t)[kStatics + 2].name == UsdGenToken("gamma") && (*t)[kStatics + 3].name == UsdGenToken("zeta"),
          "dynamic order (max(level,1), name)");
    Check(t->DynamicRun(0).size() == 2 && t->DynamicRun(1).size() == 2 && t->DynamicRun(1)[0] == kStatics &&
              t->DynamicRun(2).size() == 2 && t->DynamicRun(2)[1] == kStatics + 3 && t->DynamicRun(3).empty(),
          "dynamic runs");
    Check(t->Find(UsdGenToken("width")) == kWidth && t->Find(UsdGenToken("zeta")) == kStatics + 3 &&
              t->Find(UsdGenToken("beta")) == kStatics + 1 && t->Find(UsdGenToken("nope")) == kUsdGenNoSlot,
          "find");

    auto same = t->With(UsdGenSpan<const UsdGenChannelDesc>(dyn + 1, 2));
    Check(same.Get() == t.Get(), "With returns this when nothing changes");
    UsdGenChannelDesc const more[] = {Desc("beta", UsdGenScalar::F32, 2, 0), Desc("delta", UsdGenScalar::F32, 1, 2)};
    auto u = t->With(UsdGenSpan<const UsdGenChannelDesc>(more, 2));
    Check(u.Get() != t.Get() && u->Size() == t->Size() + 1, "With adds");
    Check((*u)[u->Find(UsdGenToken("beta"))].arity == 2, "With replaces same-name entries");
    Check(u->Find(UsdGenToken("delta")) == kStatics + 2 && u->Find(UsdGenToken("zeta")) == kStatics + 4,
          "With keeps run order");
    auto again = UsdGenChannelTable::Make(layout, UsdGenSpan<const UsdGenChannelDesc>(dyn, 4));
    Check(again->Digest() == t->Digest() && u->Digest() != t->Digest(), "digest is content identity");
}

void TestArena()
{
    auto a = UsdGenArena::Acquire(1000);
    auto b = UsdGenArena::Acquire(0);
    Check(Aligned(a->Payload()) && Aligned(b->Payload()), "arena payload 256-aligned");
    Check(a->Capacity() >= 1000, "arena capacity");
    uintptr_t const header = reinterpret_cast<uintptr_t>(a.Get()) / 64;
    uintptr_t const payload = reinterpret_cast<uintptr_t>(a->Payload()) / 64;
    Check(header != payload, "refcount line separate from the payload");
    Check(a->Serial() != b->Serial(), "unique serials");
    Check(!a->IsShared(), "fresh arena unshared");
    a->MarkShared();
    Check(a->IsShared(), "MarkShared");
    std::memset(a->Payload(), 0xAB, a->Capacity());   // the whole capacity is writable
    auto big = UsdGenArena::Acquire(size_t(80) << 20);    // past the pool: heap fallback
    Check(Aligned(big->Payload()) && big->Capacity() >= (size_t(80) << 20), "large arena");
}

/// 3 elements with 2, 0, 3 sub-items; px = 10*i, width = i, id = 100+e, uv = (e, -e).
UsdGenRef<const UsdGenElementSet> MakeFresh(UsdGenRef<const UsdGenChannelTable> channels)
{
    uint32_t const counts[2] = {3, 5};
    UsdGenWriteSet w = UsdGenWriteSet::Fresh(std::move(channels), UsdGenSpan<const uint32_t>(counts, 2));
    UsdGenSlot const written[] = {kPx, kWidth, kId, kUv};
    w.Plan(UsdGenSpan<const UsdGenSlot>(written, 4));
    UsdGenSpan<uint32_t> off = w.WriteOffsets(2);
    Check(off.size() == 4, "offsets size count[1]+1");
    uint32_t const o[] = {0, 2, 2, 5};
    for (size_t i = 0; i < off.size(); ++i) off[i] = o[i];
    Check(w.WriteOffsets(1).empty() && w.WriteOffsets(3).empty(), "offsets only for levels 2..levels");
    UsdGenSpan<float> px = w.Write<float>(kPx);
    UsdGenSpan<float> width = w.Write<float>(kWidth);
    UsdGenSpan<uint64_t> id = w.Write<uint64_t>(kId);
    UsdGenSpan<float> uv = w.Write<float>(kUv);
    Check(px.size() == 5 && width.size() == 5 && id.size() == 3 && uv.size() == 6, "write sizes");
    Check(Aligned(px.data()) && Aligned(width.data()) && Aligned(id.data()) && Aligned(uv.data()),
          "written columns 256-aligned");
    for (uint32_t i = 0; i < 5; ++i) { px[i] = 10.0f * float(i); width[i] = float(i); }
    for (uint32_t e = 0; e < 3; ++e) { id[e] = 100 + e; uv[2 * e] = float(e); uv[2 * e + 1] = -float(e); }
    Check(w.Write<float>(kUv, 1).data() == nullptr, "only sample 0");
    return std::move(w).Seal();
}

void TestFreshAndDerive()
{
    auto table = UsdGenChannelTable::Make(Layout().layout, {});
    auto s = MakeFresh(table);
    Check(s->Count(0) == 1 && s->Count(1) == 3 && s->Count(2) == 5, "fresh counts");
    UsdGenSpan<const uint32_t> off = s->Offsets(2);
    Check(off.size() == 4 && off[3] == 5 && off[1] == 2 && s->Offsets(1).empty(), "fresh offsets");
    Check(s->Column<float>(kPx).size() == 5 && s->Column<float>(kPx)[4] == 40.0f, "fresh column");
    Check(s->Column<uint64_t>(kId)[2] == 102, "fresh u64 column");
    struct V2 { float x, y; };
    Check(s->Column<V2>(kUv).size() == 3 && s->Column<V2>(kUv)[2].y == -2.0f, "packed column as vectors");
    Check(s->ArenaCount() == 1, "fresh set: one arena");
    for (UsdGenSlot k = 0; k < kStatics; ++k) {
        Check(Aligned(s->Column<std::byte>(k).data()), "every column 256-aligned");
        Check(s->ColumnArena(k) == 0 &&
                  s->ArenaAt(0).Payload() + s->ColumnOffset(k) == s->Column<std::byte>(k).data(),
              "column arena and offset");
    }
    UsdGenElementSetView v = s->View();
    Check(v.channels == table.Get() && v.count[2] == 5 && v.offsets[2] == off.data() &&
              v.Column<float>(kWidth) == s->Column<float>(kWidth).data() && v.Values(kUv) == 6 &&
              v.Has(kId) && v.Stride(2) == 0 && v.Stride(1) == 3,
          "view");
    double const *times = nullptr;
    UsdGenSpan<const double> ts(times, 0);
    Check(!s->SampleTimes(kPx, 0.0, 1.0, &ts), "one sample");

    // Derive: write px, share the rest.
    UsdGenWriteSet w = UsdGenWriteSet::Derive(s, table);
    UsdGenSlot const written[] = {kPx};
    w.Plan(UsdGenSpan<const UsdGenSlot>(written, 1));
    UsdGenSpan<float> px = w.Write<float>(kPx);
    Check(px.size() == 5, "derived write size");
    for (uint32_t i = 0; i < 5; ++i) px[i] = -float(i);
    Check(w.Write<float>(kWidth).data() == nullptr, "unplanned slot is not writable");
    auto d = std::move(w).Seal();
    Check(d->Column<float>(kWidth).data() == s->Column<float>(kWidth).data() &&
              d->Column<uint64_t>(kId).data() == s->Column<uint64_t>(kId).data() &&
              d->Column<float>(kUv).data() == s->Column<float>(kUv).data(),
          "Derive shares unwritten columns");
    Check(d->Offsets(2).data() == s->Offsets(2).data(), "Derive shares offsets");
    Check(d->Column<float>(kPx).data() != s->Column<float>(kPx).data() && d->Column<float>(kPx)[3] == -3.0f &&
              s->Column<float>(kPx)[3] == 30.0f,
          "written column is new");
    Check(d->ArenaCount() == 2, "derived set: base arena plus delta");
    Check(d->TopologyVersion() == s->TopologyVersion() && d->ValueVersion() != s->ValueVersion() &&
              d->ValueVersion(kWidth) == s->ValueVersion(kWidth) && d->ValueVersion(kPx) == d->ValueVersion(),
          "versions");

    // Derive with nothing written: same value version, no new arena.
    UsdGenWriteSet n = UsdGenWriteSet::Derive(d, table);
    n.Plan({});
    auto nd = std::move(n).Seal();
    Check(nd->ValueVersion() == d->ValueVersion() && nd->ArenaCount() == 2, "unwritten derive");

    // Share and Drop.
    UsdGenWriteSet sh = UsdGenWriteSet::Derive(s, table);
    sh.Share(kPx, *d, kPx);
    sh.Drop(kWidth);
    sh.Share(kId, *d, kPx);   // layout mismatch: absent
    auto shd = std::move(sh).Seal();
    Check(shd->Column<float>(kPx).data() == d->Column<float>(kPx).data() && !shd->Has(kWidth) && !shd->Has(kId),
          "Share and Drop");

    // Derive onto an extended table: shared by the With() parent map and by name.
    UsdGenChannelDesc const extra[] = {Desc("curl", UsdGenScalar::F32, 1, 2)};
    auto ext = table->With(UsdGenSpan<const UsdGenChannelDesc>(extra, 1));
    UsdGenWriteSet we = UsdGenWriteSet::Derive(s, ext);
    UsdGenSlot const curl = ext->Find(UsdGenToken("curl"));
    we.Plan(UsdGenSpan<const UsdGenSlot>(&curl, 1));
    UsdGenSpan<float> c = we.Write<float>(curl);
    for (float &x : c) x = 7.0f;
    auto es = std::move(we).Seal();
    Check(c.size() == 5 && es->Column<float>(curl)[4] == 7.0f &&
              es->Column<float>(kWidth).data() == s->Column<float>(kWidth).data(),
          "derive onto With() table");
    UsdGenWriteSet back = UsdGenWriteSet::Derive(es, table);   // unrelated table: by name
    auto bs = std::move(back).Seal();
    Check(bs->Column<float>(kPx).data() == s->Column<float>(kPx).data() && bs->Channels().Size() == kStatics,
          "derive onto a narrower table");

    auto empty = UsdGenElementSet::Empty(Layout().layout);
    Check(empty->Count(0) == 1 && empty->Count(1) == 0 && empty->ArenaCount() == 0 && !empty->Has(kPx) &&
              empty->Channels().Size() == kStatics,
          "Empty");
}

void TestRecycle()
{
    auto table = UsdGenChannelTable::Make(Layout().layout, {});
    auto s = MakeFresh(table);
    UsdGenSlot const written[] = {kPx, kWidth};
    auto derive = [&](UsdGenRef<const UsdGenElementSet> spare) {
        UsdGenWriteSet w = UsdGenWriteSet::Derive(s, table);
        w.Recycle(std::move(spare));
        w.Plan(UsdGenSpan<const UsdGenSlot>(written, 2));
        for (float &x : w.Write<float>(kPx)) x = 1.0f;
        for (float &x : w.Write<float>(kWidth)) x = 2.0f;
        return std::move(w).Seal();
    };
    auto first = derive({});
    uint64_t const serial = first->ArenaAt(1).Serial();
    float const *px = first->Column<float>(kPx).data();
    auto second = derive(std::move(first));
    Check(second->ArenaAt(1).Serial() == serial && second->Column<float>(kPx).data() == px,
          "Recycle reuses the delta arena in place");

    second->Export();
    auto third = derive(std::move(second));
    Check(third->ArenaAt(1).Serial() != serial, "Recycle refuses an exported set");

    auto held = third;   // a second owner
    auto fourth = derive(third);
    Check(fourth->ArenaAt(1).Serial() != held->ArenaAt(1).Serial() && held->Column<float>(kPx)[0] == 1.0f,
          "Recycle refuses a set held elsewhere");
    held.Reset();
    third.Reset();

    // A set deriving from the spare holds its arena: refused.
    auto spare = derive({});
    UsdGenWriteSet child = UsdGenWriteSet::Derive(spare, table);
    auto kid = std::move(child).Seal();
    uint64_t const spareSerial = spare->ArenaAt(1).Serial();
    auto fifth = derive(std::move(spare));
    Check(fifth->ArenaAt(1).Serial() != spareSerial && kid->Column<float>(kPx)[0] == 1.0f,
          "Recycle refuses an arena another set shares");
}

void TestCompaction()
{
    UsdGenChannelDesc const dyn[] = {
        Desc("a", UsdGenScalar::F32, 1, 2), Desc("b", UsdGenScalar::F32, 1, 2),
        Desc("c", UsdGenScalar::F32, 1, 2), Desc("d", UsdGenScalar::F32, 1, 2),
        Desc("e", UsdGenScalar::F32, 1, 2),
    };
    auto table = UsdGenChannelTable::Make(Layout().layout, UsdGenSpan<const UsdGenChannelDesc>(dyn, 5));
    auto set = MakeFresh(table);
    for (UsdGenSlot k = 0; k < 5; ++k) {
        UsdGenSlot const slot = UsdGenSlot(kStatics + k);
        UsdGenWriteSet w = UsdGenWriteSet::Derive(set, table);
        w.Plan(UsdGenSpan<const UsdGenSlot>(&slot, 1));
        UsdGenSpan<float> col = w.Write<float>(slot);
        for (size_t i = 0; i < col.size(); ++i) col[i] = float(k * 100 + i);
        set = std::move(w).Seal();
        Check(set->ArenaCount() <= kUsdGenMaxArenasPerSet, "arena count stays within the limit");
    }
    Check(set->ArenaCount() == 2, "compacted after the fourth arena");
    bool bytes = set->Offsets(2).size() == 4 && set->Offsets(2)[3] == 5 && set->Column<float>(kPx)[2] == 20.0f &&
                 set->Column<uint64_t>(kId)[1] == 101 && set->Column<float>(kUv)[5] == -2.0f;
    for (UsdGenSlot k = 0; k < 5; ++k) {
        UsdGenSpan<const float> col = set->Column<float>(UsdGenSlot(kStatics + k));
        for (size_t i = 0; i < col.size(); ++i) bytes = bytes && col[i] == float(k * 100 + i);
        bytes = bytes && col.size() == 5 && Aligned(col.data());
    }
    Check(bytes, "compaction preserves bytes");
}

void TestWarmAllocations()
{
    auto table = UsdGenChannelTable::Make(Layout().layout, {});
    auto base = MakeFresh(table);
    UsdGenSlot const written[] = {kPx, kWidth};
    UsdGenRef<const UsdGenElementSet> spare;
    auto cycle = [&]() {
        UsdGenWriteSet w = UsdGenWriteSet::Derive(base, table);
        w.Recycle(std::move(spare));
        w.Plan(UsdGenSpan<const UsdGenSlot>(written, 2));
        for (float &x : w.Write<float>(kPx)) x = 3.0f;
        for (float &x : w.Write<float>(kWidth)) x = 4.0f;
        spare = std::move(w).Seal();
    };
    cycle();
    cycle();
    UsdGenEngineAllocCounting(true);
    cycle();
    uint64_t const n = UsdGenEngineAllocCount();
    UsdGenEngineAllocCounting(false);
    if (n) std::printf("warm cycle allocations: %llu\n", static_cast<unsigned long long>(n));
    Check(n == 0, "0 allocations on a warm Derive/Recycle/Plan/Seal cycle");
    Check(spare->Column<float>(kWidth)[4] == 4.0f, "warm cycle output");
}

UsdGenTypeId g_registered = UsdGenTypeId::Invalid;
UsdGenTypeId g_rejected = UsdGenTypeId::Invalid;
UsdGenElementLayout g_badLayout;

void RegisterTestType(UsdGenRegistrar &r)
{
    UsdGenTypeDef def;
    def.name = "UsdGenTestElementSet";
    def.layout = &Layout().layout;
    def.limits[1] = 4096;
    g_registered = r.RegisterType(def);
    g_badLayout.levels = kUsdGenMaxLevels + 1;
    UsdGenTypeDef bad;
    bad.name = "UsdGenTestBadLayout";
    bad.layout = &g_badLayout;
    g_rejected = r.RegisterType(bad);
}

void TestTypeRegistry()
{
    Check(UsdGenRegisterStatic(&RegisterTestType), "register");
    Check(g_registered != UsdGenTypeId::Invalid && g_rejected == UsdGenTypeId::Invalid, "layout validated");
    UsdGenTypeRegistry const &types = UsdGenTypeRegistry::Get();
    UsdGenTypeId const id = types.Find(UsdGenToken("UsdGenTestElementSet"));
    Check(id == g_registered && types.Def(id).layout == &Layout().layout && types.Def(id).limits[1] == 4096,
          "type def carries the layout");
}

}  // namespace

int main()
{
    TestSpan();
    TestChannelTable();
    TestArena();
    TestFreshAndDerive();
    TestRecycle();
    TestCompaction();
    TestWarmAllocations();
    TestTypeRegistry();
    if (g_failures) {
        std::printf("testUsdGenElementSet: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("testUsdGenElementSet: ok\n");
    return 0;
}
