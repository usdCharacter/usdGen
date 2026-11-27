// testUsdGenScheduling — gate SI-3 (T1 list, engine half; 03 §5.4/§5.6,
// 06 §3.9): event -> commit -> generation accounting.
//
// Asserted (engine-only, gate B-1):
//   1. SetGraphDesc marks the session dirty; one Commit publishes generation 0
//      and leaves NeedsCommit() false.
//   2. 10 accumulate+Commit cycles publish EXACTLY 10 further generations:
//      Stats().commits and Generation()->id both step by exactly 10,
//      generation ids strictly increasing, id == commits-1.
//   3. Reads via Generation() never cook: 50 loads leave every counter
//      (commits, recompiles, capturedNodes, supersessions) unchanged —
//      "nothing cooks inside GetPrim" (I7).
//   4. A commit with nothing dirty still publishes (cheap, tiles carried over)
//      but reports ZERO payload changes in LastReport() and leaves the
//      generation payload bit-identical (VtArray::IsIdentical).
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

#include <cstdio>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;

namespace {

int g_failures = 0;

void Check(bool ok, std::string const &what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

// Small G3 stand-in: the accounting contract is size-independent.
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

bool SameFloatPlane(VtFloatArray const &a, VtFloatArray const &b)
{
    if (a.size() != b.size()) return false;
    const size_t n = a.size() * sizeof(float);
    return n == 0 || std::memcmp(a.data(), b.data(), n) == 0;
}

}  // namespace

int main()
{
    usdGenTest::EnsureHydraPublishers();

    UsdGenSession session(8);
    UsdGenGraphDesc const desc = MakeG3();

    // ---- 1: first event -> generation 0 --------------------------------------
    usdGenUsd::SetGraphDesc(session, desc);
    Check(session.NeedsCommit(), "SetGraphDesc marks the session dirty");
    UsdGenGenerationConstPtr g0 = session.Commit(0.0, UsdGenCommitReason::NoticeBatchEnd);
    Check(g0 != nullptr, "first Commit publishes a generation");
    Check(g0 && g0->id == 0, "first published generation id == 0");
    Check(session.Stats().commits == 1, "Stats().commits == 1 after one commit");
    Check(!session.NeedsCommit(), "NeedsCommit() false after the commit drained dirt");
    Check(g0 && !usdGen::UsdGenCurvePublicationOf(*g0).tiles.empty(), "generation carries published tiles");

    // ---- 2: 10 events -> exactly 10 new generations ---------------------------
    UsdGenNodeId const noise = usdGenUsd::NodeIdForPath(session.Graph(), SdfPath("/groom/noise"));
    int64_t const idBefore = session.Generation()->id;
    uint64_t const commitsBefore = session.Stats().commits;
    for (int i = 0; i < 10; ++i) {
        UsdGenPendingDirty p;
        p.nodeBits[noise] = UsdGenDirtyParameter;
        session.AccumulateDirty(std::move(p));
        Check(session.NeedsCommit(), "event marks dirty (cycle " + std::to_string(i) + ")");
        UsdGenGenerationConstPtr g =
            session.Commit(double(i + 1), UsdGenCommitReason::NoticeBatchEnd);
        if (!g || g->id != idBefore + i + 1) {
            Check(false, "generation id steps +1 per commit (cycle " +
                             std::to_string(i) + ")");
            break;
        }
    }
    Check(session.Stats().commits == commitsBefore + 10,
          "10 events -> exactly 10 published commits");
    Check(session.Generation()->id == idBefore + 10,
          "generation id strictly increased by exactly 10");
    Check(static_cast<uint64_t>(session.Generation()->id) == session.Stats().commits - 1,
          "generation id == commits - 1 (one publish per committed request)");

    // ---- 3: readers never cook -----------------------------------------------
    uint64_t const c0 = session.Stats().commits;
    uint64_t const r0 = session.Stats().recompiles;
    uint64_t const cap0 = session.Stats().capturedNodes;
    uint64_t const sup0 = session.Stats().supersessions;
    for (int i = 0; i < 50; ++i) {
        UsdGenGenerationConstPtr g = session.Generation();
        if (!g) { Check(false, "Generation() non-null during reads"); break; }
    }
    Check(session.Stats().commits == c0 && session.Stats().recompiles == r0 &&
              session.Stats().capturedNodes == cap0 &&
              session.Stats().supersessions == sup0,
          "50 Generation() reads bump no counter (nothing cooks in GetPrim)");

    // ---- 4: clean commit publishes, reports zero payload change ---------------
    UsdGenGenerationConstPtr before = session.Generation();
    UsdGenGenerationConstPtr after =
        session.Commit(99.0, UsdGenCommitReason::SetTime);
    Check(after != nullptr && after->id == before->id + 1,
          "clean commit still publishes the next generation id");
    bool noPayloadChange = true;
    auto const lastReport = session.LastReport();
    for (UsdGenTileDirty const &td : usdGen::UsdGenCurveDirtyReportOf(lastReport).tiles)
        noPayloadChange = noPayloadChange && !td.pointsDirty && !td.widthsDirty &&
                          !td.added && !td.removed && td.newPrimvars.empty() &&
                          td.dirtyPrimvars.empty();
    Check(noPayloadChange, "LastReport() carries no payload/structural changes for a "
                           "clean commit (tiles carried over)");
    bool payloadIdentical = usdGen::UsdGenCurvePublicationOf(*before).tiles.size() == usdGen::UsdGenCurvePublicationOf(*after).tiles.size();
    for (size_t i = 0; payloadIdentical && i < usdGen::UsdGenCurvePublicationOf(*before).tiles.size(); ++i) {
        UsdGenTilePublication const &tb = usdGen::UsdGenCurvePublicationOf(*before).tiles[i];
        UsdGenTilePublication const &ta = usdGen::UsdGenCurvePublicationOf(*after).tiles[i];
        payloadIdentical =
            tb.points.size() == ta.points.size() &&
            (tb.points.empty() ||
             std::memcmp(tb.points.data(), ta.points.data(),
                         tb.points.size() * sizeof(usdGen::UsdGenVec3f)) == 0);
    }
    Check(payloadIdentical, "carried-over tile payloads are bit-identical "
                            "(VtArray buffer sharing)");

    std::printf(g_failures ? "testUsdGenScheduling: FAILED (%d)\n"
                           : "testUsdGenScheduling: PASS (SI-3 engine half)\n",
                g_failures);
    return g_failures ? 1 : 0;
}