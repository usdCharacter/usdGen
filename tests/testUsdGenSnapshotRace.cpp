// testUsdGenSnapshotRace — gate SI-4 (T1 list, engine form; 03 §6.1, I7):
// lock-free generation reads never observe a torn snapshot while the commit
// thread publishes.
//
// Form: 1 commit thread publishes 100 generations (value-class dirty per
// cycle) while 8 reader threads hammer Session::Generation() (the exact
// atomic_load a Hydra GetPrim does). Zero torn reads required, where a
// snapshot is internally consistent iff:
//   * per reader, observed generation ids are non-decreasing,
//   * tiles.size() == signature.tileCount,
//   * every tile: points.size() == sum(curveVertexCounts) (SI-1 invariant)
//     and widths.size() == hairT.size() == points.size(),
//     hairId.size() == curveVertexCounts.size(),
//   * curveVertexCounts are all >= 1 and their sum > 0.
// After the join: exactly 101 published generations, ids 0..100.
//
// Tier T1 (registered in CMake); engine-only body (gate B-1).

#include "utils/pxrInterop.h"
#include "usdGenHydraCurves/publication.h"
#include "usdGenTestUtils/hydraPublishers.h"
#include "usdGen/op.h"
#include "usdGenUsd/lower.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/session.h"
#include "usdGen/types.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;

namespace {

std::atomic<int> g_torn{0};
std::atomic<int> g_regression{0};
std::atomic<uint64_t> g_reads{0};

void ReportBad(std::string const &what)
{
    ++g_torn;
    if (g_torn <= 5) std::printf("TORN READ: %s\n", what.c_str());
}

// The internal-consistency check one reader performs per load.
bool SnapshotConsistent(UsdGenGenerationConstPtr const &g)
{
    if (!g) { ReportBad("null generation pointer"); return false; }
    if (g->id < 0) { ReportBad("negative generation id"); return false; }
    if (usdGen::UsdGenCurvePublicationOf(*g).tiles.size() != size_t(usdGen::UsdGenCurvePublicationOf(*g).signature.tileCount)) {
        ReportBad("tileCount != tiles.size() at gen " + std::to_string(g->id));
        return false;
    }
    for (UsdGenTilePublication const &t : usdGen::UsdGenCurvePublicationOf(*g).tiles) {
        size_t sum = 0;
        for (int c : t.curveVertexCounts) {
            if (c <= 0) { ReportBad("non-positive curveVertexCount"); return false; }
            sum += size_t(c);
        }
        if (sum == 0 || t.points.size() != sum || t.widths.size() != sum ||
            t.hairT.size() != sum || t.hairId.size() != t.curveVertexCounts.size()) {
            std::string s = "gen " + std::to_string(g->id) + " tile " +
                            std::to_string(t.tile) + " payload split (points " +
                            std::to_string(t.points.size()) + ", sum " +
                            std::to_string(sum) + ")";
            ReportBad(s);
            return false;
        }
    }
    return true;
}

UsdGenGraphDesc MakeG3(int NX = 250, int NY = 10)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/width");
    d.time = 0.0;

    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.id = 0;
    const int np = (NX + 1) * (NY + 1);
    s.restPoints = usdGen::UsdGenVec3fArray(np);
    s.uv = usdGen::UsdGenVec2fArray(np);
    for (int j = 0; j <= NY; ++j)
        for (int i = 0; i <= NX; ++i) {
            const int k = j * (NX + 1) + i;
            s.restPoints[k] = usdGen::UsdGenVec3f(i * 0.1f, j * 0.1f, 0.0f);
            s.uv[k] = usdGen::UsdGenVec2f(float(i) / NX, float(j) / NY);
        }
    s.faceVertexCounts = VtIntArray(NX * NY, 4);
    s.faceVertexIndices = VtIntArray(NX * NY * 4);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            const int a = j * (NX + 1) + i;
            const int o = (j * NX + i) * 4;
            s.faceVertexIndices[o + 0] = a;
            s.faceVertexIndices[o + 1] = a + 1;
            s.faceVertexIndices[o + 2] = a + NX + 2;
            s.faceVertexIndices[o + 3] = a + NX + 1;
        }
    d.surfaces.push_back(std::move(s));

    auto addNode = [&](std::string const &name, TfToken type,
                       std::string const &input, int seed) {
        UsdGenNodeDesc n;
        n.path = SdfPath("/groom/" + name);
        n.type = type;
        n.enabled = true;
        n.seed = seed;
        if (!input.empty()) n.inputs.push_back(SdfPath("/groom/" + input));
        if (type == TfToken("UsdGenScatter"))
            n.surfaces.push_back(SdfPath("/groom/surface"));
        d.nodes.push_back(std::move(n));
    };
    addNode("scatter", TfToken("UsdGenScatter"), "", 42);
    addNode("grow", TfToken("UsdGenGrow"), "scatter", 43);
    addNode("noise", TfToken("UsdGenNoise"), "grow", 44);
    addNode("length", TfToken("UsdGenLength"), "noise", 45);
    addNode("width", TfToken("UsdGenWidth"), "length", 46);

    auto setp = [&](std::string const &name, TfToken p, VtValue v) {
        for (auto &n : d.nodes)
            if (n.path == SdfPath("/groom/" + name))
                n.params.push_back(UsdGenParamValue{p, v, false});
    };
    setp("grow", TfToken("segments"), VtValue(8));
    setp("grow", TfToken("length"), VtValue(1.0));
    setp("noise", TfToken("noise:magnitude"), VtValue(0.05));
    setp("length", TfToken("length:mode"), VtValue(TfToken("scale")));
    setp("length", TfToken("length:value"), VtValue(1.2));
    setp("width", TfToken("width"), VtValue(0.02));
    return d;
}

}  // namespace

int main()
{
    usdGenTest::EnsureHydraPublishers();

    constexpr int kReaders = 8;
    constexpr int kGenerations = 100;

    UsdGenSession session(8);
    usdGenUsd::SetGraphDesc(session, MakeG3());
    UsdGenGenerationConstPtr const g0 =
        session.Commit(0.0, UsdGenCommitReason::NoticeBatchEnd);
    if (!g0 || g0->id != 0) {
        std::printf("FAIL: seed commit did not publish generation 0\n");
        return 1;
    }
    UsdGenNodeId const noise = usdGenUsd::NodeIdForPath(session.Graph(), SdfPath("/groom/noise"));

    std::atomic<bool> stop{false};

    std::vector<std::thread> readers;
    readers.reserve(kReaders);
    for (int r = 0; r < kReaders; ++r) {
        readers.emplace_back([&, r] {
            int64_t lastId = -1;
            uint64_t localReads = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                UsdGenGenerationConstPtr g = session.Generation();
                ++localReads;
                if (!SnapshotConsistent(g)) continue;
                if (g->id < lastId) {
                    ++g_regression;
                    if (g_regression <= 5)
                        std::printf("TORN READ: reader %d saw id %lld < %lld\n", r,
                                    static_cast<long long>(g->id),
                                    static_cast<long long>(lastId));
                }
                lastId = g->id;
            }
            g_reads.fetch_add(localReads, std::memory_order_relaxed);
        });
    }

    for (int i = 1; i <= kGenerations; ++i) {
        UsdGenPendingDirty p;
        p.nodeBits[noise] = UsdGenDirtyParameter;
        session.AccumulateDirty(std::move(p));
        session.Commit(double(i), UsdGenCommitReason::NoticeBatchEnd);
    }

    stop.store(true, std::memory_order_relaxed);
    for (std::thread &t : readers) t.join();

    int failures = 0;
    auto Check = [&](bool ok, std::string const &what) {
        if (!ok) { ++failures; std::printf("FAIL: %s\n", what.c_str()); }
        else std::printf("ok:   %s\n", what.c_str());
    };

    Check(g_reads.load() > 0, "readers performed loads during the storm");
    std::printf("      (%llu concurrent reads over %d published generations)\n",
                static_cast<unsigned long long>(g_reads.load()), kGenerations + 1);
    Check(g_torn.load() == 0 && g_regression.load() == 0,
          "SI-4: zero torn reads / zero id regressions across 8 readers x 100 commits");
    Check(session.Generation()->id == kGenerations,
          "commit thread published exactly the 100 requested generations (no loss)");
    Check(!session.NeedsCommit(), "session clean after the commit storm");

    std::printf(failures ? "testUsdGenSnapshotRace: FAILED (%d)\n"
                         : "testUsdGenSnapshotRace: PASS (SI-4 engine form)\n",
                failures);
    return failures ? 1 : 0;
}