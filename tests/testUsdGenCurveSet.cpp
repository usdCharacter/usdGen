// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenCurveSet: the CurveSet layout reproduces today's buffer and partition (5.2).
//   - the type registers with its layout, statics in slot order
//   - buffer -> set -> buffer is bitwise for uniform, ragged, empty and extra-plane buffers
//   - set and buffer views agree field by field; FindPlane; inconsistent buffers are refused
//   - the partition and group hooks equal UsdGenGraph::Repartition on 10 fixtures
//   - UsdGenCurveChunk over a generic chunk view (testUsdGenCurveSetChunk.cpp)

#include "usdGenCurves/types/curveBuffer.h"
#include "nodeSets.h"
#include "usdGenCurves/types/curveSet.h"

#include "usdGen/compile/compiler.h"
#include "usdGen/compile/graph.h"
#include "usdGenUsd/lower.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/registrar.h"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/value.h"
#include "pxr/usd/sdf/path.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include "utils/pxrInterop.h"

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

int g_failures = 0;
void Check(bool ok, char const *what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

int RunCurveChunkTests();   // testUsdGenCurveSetChunk.cpp

namespace {

template <class T>
bool Same(VtArray<T> const &a, VtArray<T> const &b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.cdata(), b.cdata(), a.size() * sizeof(T)) == 0);
}

template <class T>
bool Same(UsdGenSpan<const T> a, UsdGenSpan<const T> b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

bool SamePlanes(std::vector<UsdGenPlane> const &a, std::vector<UsdGenPlane> const &b)
{
    if (a.size() != b.size()) return false;
    for (size_t k = 0; k < a.size(); ++k)
        if (a[k].name != b[k].name || a[k].interpolation != b[k].interpolation || a[k].type != b[k].type ||
            a[k].arity != b[k].arity || !Same(a[k].f, b[k].f) || !Same(a[k].i, b[k].i))
            return false;
    return true;
}

bool SameBuffer(UsdGenCurveBuffer const &a, UsdGenCurveBuffer const &b)
{
    return Same(a.px, b.px) && Same(a.py, b.py) && Same(a.pz, b.pz) && Same(a.rest, b.rest) &&
           Same(a.width, b.width) && Same(a.hairT, b.hairT) && SamePlanes(a.extraCv, b.extraCv) &&
           Same(a.curveId, b.curveId) && Same(a.rootPrim, b.rootPrim) && Same(a.rootUV, b.rootUV) &&
           Same(a.rootT, b.rootT) && Same(a.rootN, b.rootN) && Same(a.rootB, b.rootB) &&
           Same(a.cvOffsets, b.cvOffsets) && SamePlanes(a.extraCurve, b.extraCurve) &&
           a.chunks.size() == b.chunks.size() && a.totalCurves == b.totalCurves && a.totalCvs == b.totalCvs &&
           a.topologyVersion == b.topologyVersion && a.valueVersion == b.valueVersion;
}

uint32_t RaggedCvs(uint32_t c) { return 1u + (c * 7u) % 5u; }

/// cvp 0 = ragged (RaggedCvs). full: every static channel; else px/py/pz/hairT only.
UsdGenCurveBuffer MakeBuffer(uint32_t curves, uint32_t cvp, bool full, bool extras, uint64_t version = 7)
{
    UsdGenCurveBuffer b;
    b.totalCurves = curves;
    uint32_t cvs = 0;
    if (cvp) {
        cvs = curves * cvp;
    } else {
        b.cvOffsets.resize(size_t(curves) + 1);
        b.cvOffsets[0] = 0;
        for (uint32_t c = 0; c < curves; ++c) b.cvOffsets[c + 1] = b.cvOffsets[c] + int(RaggedCvs(c));
        cvs = uint32_t(b.cvOffsets[curves]);
    }
    b.totalCvs = cvs;
    b.topologyVersion = version;
    b.valueVersion = version * 3 + 1;
    if (!cvs && !curves) {
        full = false;
    } else {
    b.px.resize(cvs); b.py.resize(cvs); b.pz.resize(cvs); b.hairT.resize(cvs);
    for (uint32_t i = 0; i < cvs; ++i) {
        b.px[i] = 0.25f * float(i);
        b.py[i] = -1.5f * float(i % 13);
        b.pz[i] = float(i) / 3.0f;
        b.hairT[i] = float(i % 9) / 8.0f;
    }
    if (cvs > 2) {
        b.px[1] = std::numeric_limits<float>::quiet_NaN();   // bits, not values
        b.py[2] = -0.0f;
    }
    }
    if (full) {
        b.rest.resize(cvs); b.width.resize(cvs);
        for (uint32_t i = 0; i < cvs; ++i) {
            b.rest[i] = usdGen::UsdGenVec3f(float(i), 2.0f * float(i), -float(i));
            b.width[i] = 0.01f + 0.001f * float(i % 7);
        }
        b.curveId.resize(curves); b.rootPrim.resize(curves); b.rootUV.resize(curves);
        b.rootT.resize(curves); b.rootN.resize(curves); b.rootB.resize(curves);
        for (uint32_t c = 0; c < curves; ++c) {
            b.curveId[c] = 0x9e3779b97f4a7c15ull * (c + 1);
            b.rootPrim[c] = int(c % 17) - 3;
            b.rootUV[c] = usdGen::UsdGenVec2f(float(c) * 0.5f, 1.0f - float(c));
            b.rootT[c] = usdGen::UsdGenVec3f(1, 0, float(c));
            b.rootN[c] = usdGen::UsdGenVec3f(0, 1, float(c));
            b.rootB[c] = usdGen::UsdGenVec3f(0, 0, 1.0f + float(c));
        }
    }
    if (extras) {
        UsdGenEngineTokens const &tk = UsdGenTokens();
        auto plane = [&](char const *name, TfToken const &interp, bool isInt, uint8_t arity, size_t elements) {
            UsdGenPlane p;
            p.name = TfToken(name);
            p.interpolation = interp;
            p.type = isInt ? tk.int_ : tk.float_;
            p.arity = arity;
            size_t const n = elements * arity;
            if (isInt) {
                p.i.resize(n);
                for (size_t k = 0; k < n; ++k) p.i[k] = int(k * 31 % 1000) - 500;
            } else {
                p.f.resize(n);
                for (size_t k = 0; k < n; ++k) p.f[k] = float(k) * 0.125f - 2.0f;
            }
            return p;
        };
        // Sorted by name per vector (C2).
        b.extraCurve.push_back(plane("clumpCenterId_0", tk.uniform, true, 2, curves));
        b.extraCurve.push_back(plane("clumpCenter_0", tk.uniform, false, 3, curves));
        b.extraCurve.push_back(plane("clumpId_0", tk.uniform, true, 1, curves));
        b.extraCurve.push_back(plane("furTint", tk.constant, false, 3, 1));
        b.extraCv.push_back(plane("clumpWeight_0", tk.vertex, false, 1, cvs));
        b.extraCv.push_back(plane("strandMask", tk.vertex, true, 2, cvs));
    }
    return b;
}

void TestRegistration()
{
    UsdGenTypeId const id = UsdGenCurveSetType();
    Check(id != UsdGenTypeId::Invalid, "UsdGenCurveSet registers");
    UsdGenElementLayout const &layout = UsdGenCurveSetLayout();
    Check(UsdGenTypeRegistry::Get().Find(TfToken("UsdGenCurveSet")) == id, "type found by name");
    Check(UsdGenTypeRegistry::Get().Def(id).layout == &layout, "type def points at the layout");
    Check(UsdGenTypeRegistry::Get().Def(id).limits[1] == kUsdGenMaxCvsPerCurve, "CV limit");
    Check(layout.primitive == id && layout.levels == 2 && layout.channels.size() == kUsdGenCurveStaticSlots &&
              layout.partition && layout.groups,
          "layout shape");
    UsdGenChannelDesc const *d = layout.channels.data();
    Check(d[kUsdGenCurvePx].level == 2 && d[kUsdGenCurvePz].component == 2 &&
              d[kUsdGenCurvePy].role == UsdGenChannelRole::Position,
          "points planar L2");
    Check(d[kUsdGenCurveRest].arity == 3 && (d[kUsdGenCurveRest].flags & kUsdGenChannelPassThrough) &&
              (d[kUsdGenCurveRest].flags & kUsdGenChannelPacked),
          "rest packed pass-through");
    Check(d[kUsdGenCurveId].scalar == UsdGenScalar::U64 && d[kUsdGenCurveId].level == 1, "curveId L1 U64");
    Check(d[kUsdGenCurveRootUV].arity == 2 && d[kUsdGenCurveRootB].arity == 3 &&
              d[kUsdGenCurveRootB].role == UsdGenChannelRole::Frame,
          "root frame channels");
    for (UsdGenSlot s = 0; s < kUsdGenCurveStaticSlots; ++s) Check(!d[s].name.IsEmpty(), "static named");
}

void CheckViewsAgree(UsdGenCurveBuffer const &b, UsdGenElementSet const &set, char const *what)
{
    UsdGenCurveSetView const a = UsdGenCurveSetView::Of(b), s = UsdGenCurveSetView::Of(set);
    bool ok = Same(a.px, s.px) && Same(a.py, s.py) && Same(a.pz, s.pz) && Same(a.width, s.width) &&
              Same(a.hairT, s.hairT) && Same(a.rest, s.rest) && Same(a.rootT, s.rootT) &&
              Same(a.rootN, s.rootN) && Same(a.rootB, s.rootB) && Same(a.curveId, s.curveId) &&
              Same(a.rootPrim, s.rootPrim) && Same(a.rootUV, s.rootUV) && Same(a.cvOffsets, s.cvOffsets) &&
              a.totalCurves == s.totalCurves && a.totalCvs == s.totalCvs &&
              a.topologyVersion == s.topologyVersion && a.valueVersion == s.valueVersion &&
              a.ExtraCurveCount() == s.ExtraCurveCount() && a.ExtraCvCount() == s.ExtraCvCount();
    auto samePlane = [](UsdGenPlaneView const &x, UsdGenPlaneView const &y) {
        return x.name == y.name && x.interpolation == y.interpolation && x.type == y.type &&
               x.arity == y.arity && Same(x.f, y.f) && Same(x.i, y.i);
    };
    for (uint32_t k = 0; ok && k < a.ExtraCurveCount(); ++k) ok = samePlane(a.ExtraCurve(k), s.ExtraCurve(k));
    for (uint32_t k = 0; ok && k < a.ExtraCvCount(); ++k) ok = samePlane(a.ExtraCv(k), s.ExtraCv(k));
    for (uint32_t k = 0; ok && k < a.ExtraCvCount(); ++k) {
        UsdGenPlaneView pa, ps;
        bool ca = false, cs = false;
        ok = a.FindPlane(a.ExtraCv(k).name, &pa, &ca) && s.FindPlane(a.ExtraCv(k).name, &ps, &cs) && ca && cs &&
             samePlane(pa, ps);
    }
    for (uint32_t k = 0; ok && k < a.ExtraCurveCount(); ++k) {
        UsdGenPlaneView pa, ps;
        bool ca = true, cs = true;
        ok = a.FindPlane(a.ExtraCurve(k).name, &pa, &ca) && s.FindPlane(a.ExtraCurve(k).name, &ps, &cs) &&
             !ca && !cs && samePlane(pa, ps);
    }
    Check(ok, what);
}

void RoundTrip(UsdGenCurveBuffer const &b, char const *what)
{
    UsdGenRef<const UsdGenElementSet> set = UsdGenCurveSetFromBuffer(b, b.topologyVersion, b.valueVersion);
    Check(bool(set), what);
    if (!set) return;
    Check(set->Count(1) == b.totalCurves && set->Count(2) == b.totalCvs, "set counts");
    Check(set->Offsets(2).empty() == b.cvOffsets.empty(), "ragged iff offsets");
    Check(set->Has(kUsdGenCurvePx) == !b.px.empty() && set->Has(kUsdGenCurveRest) == !b.rest.empty() &&
              set->Has(kUsdGenCurveId) == !b.curveId.empty(),
          "absent static columns stay absent");
    Check(set->ArenaCount() <= 1, "one arena");
    for (UsdGenSlot s = 0; s < set->Channels().Size(); ++s)
        Check(!set->Has(s) || set->ColumnOffset(s) % UsdGenArena::kColumnAlign == 0, "columns aligned");
    UsdGenCurveBuffer const back = UsdGenCurveBufferFromSet(*set);
    Check(SameBuffer(b, back), what);
    CheckViewsAgree(b, *set, what);
}

void TestRoundTrip()
{
    RoundTrip(MakeBuffer(37, 5, true, true), "uniform round trip is bitwise");
    RoundTrip(MakeBuffer(41, 0, true, true), "ragged round trip is bitwise");
    RoundTrip(MakeBuffer(0, 0, false, false), "empty round trip is bitwise");
    RoundTrip(MakeBuffer(0, 4, false, true), "empty with planes round trip is bitwise");
    RoundTrip(MakeBuffer(600, 3, false, true), "extra planes on partial statics round trip");
    RoundTrip(MakeBuffer(19, 0, false, false), "ragged partial statics round trip");
    UsdGenCurveBuffer rootsOnly = MakeBuffer(12, 1, true, false);
    RoundTrip(rootsOnly, "one CV per curve round trip");

    UsdGenCurveBuffer const b = MakeBuffer(9, 4, true, true);
    auto set = UsdGenCurveSetFromBuffer(b, 0, 0);
    Check(set && set->TopologyVersion() != 0 && set->ValueVersion() != 0, "zero versions take fresh ones");
    UsdGenCurveSetView const v = UsdGenCurveSetView::Of(*set);
    Check(v.ExtraCurveCount() == 4 && v.ExtraCvCount() == 2, "plane runs");
    UsdGenPlaneView const tint = v.ExtraCurve(3);
    Check(tint.name == TfToken("furTint") && tint.interpolation == UsdGenTokens().constant && tint.f.size() == 3,
          "constant plane at level 0");
    Check(!v.FindPlane(TfToken("__px"), nullptr, nullptr) && !v.FindPlane(TfToken("nope"), nullptr, nullptr),
          "FindPlane skips statics and misses");
}

void TestRefused()
{
    UsdGenCurveBuffer b = MakeBuffer(8, 3, true, true);
    b.width.resize(5);
    Check(!UsdGenCurveSetFromBuffer(b, 1, 1), "short width refused");
    b = MakeBuffer(8, 0, false, false);
    b.cvOffsets.resize(4);
    Check(!UsdGenCurveSetFromBuffer(b, 1, 1), "short offsets refused");
    b = MakeBuffer(8, 3, false, true);
    b.extraCv[0].interpolation = UsdGenTokens().uniform;
    Check(!UsdGenCurveSetFromBuffer(b, 1, 1), "CV plane at the wrong rate refused");
    b = MakeBuffer(8, 3, false, true);
    b.extraCurve[0].type = TfToken("double");
    Check(!UsdGenCurveSetFromBuffer(b, 1, 1), "unknown plane type refused");
    b = MakeBuffer(8, 3, false, true);
    b.extraCv[1].name = b.extraCurve[0].name;
    Check(!UsdGenCurveSetFromBuffer(b, 1, 1), "a name in both rates refused");
    b = MakeBuffer(8, 3, false, true);
    b.extraCurve[2].i.resize(3);
    Check(!UsdGenCurveSetFromBuffer(b, 1, 1), "short plane refused");
}

// ---- partition ------------------------------------------------------------------

UsdGenGraphDesc MakeDesc(int tileTarget)
{
    SdfPath const root("/curveSet");
    UsdGenGraphDesc d;
    d.description = root;
    d.terminal = root.AppendChild(TfToken("grow"));
    d.tileTarget = tileTarget;
    d.curveBasis = TfToken("bspline");
    d.xformMatrix = GfMatrix4d(1.0);
    UsdGenSurfaceDesc surf;
    surf.path = root.AppendChild(TfToken("surface"));
    surf.id = 0;
    surf.faceVertexCounts = VtIntArray(1, 4);
    surf.faceVertexIndices = VtIntArray{0, 1, 2, 3};
    surf.restPoints = usdGen::UsdGenVec3fArray{usdGen::UsdGenVec3f(0, 0, 0), usdGen::UsdGenVec3f(1, 0, 0), usdGen::UsdGenVec3f(1, 1, 0), usdGen::UsdGenVec3f(0, 1, 0)};
    d.surfaces.push_back(surf);
    UsdGenNodeDesc scatter;
    scatter.path = root.AppendChild(TfToken("scatter"));
    scatter.type = TfToken("UsdGenScatter");
    scatter.enabled = true;
    scatter.seed = 3;
    scatter.surfaces.push_back(surf.path);
    d.nodes.push_back(scatter);
    UsdGenNodeDesc grow;
    grow.path = d.terminal;
    grow.type = TfToken("UsdGenGrow");
    grow.enabled = true;
    grow.inputs = {scatter.path};
    d.nodes.push_back(grow);
    return d;
}

struct PartitionCase {
    uint32_t curves;
    uint32_t cvp;        // 0 = ragged
    int tileTarget;
    bool cooked;         // node buffers hold the fixture; else the fallback cvp applies
    int fallbackCvp;
    char const *what;
};

bool SameChunks(UsdGenGraph const &graph, UsdGenGraph::NodeId id, std::vector<UsdGenChunkRange> const &ranges)
{
    auto const chunks = graph.Chunks(id);
    if (chunks.size() != ranges.size()) return false;
    for (size_t c = 0; c < chunks.size(); ++c) {
        UsdGenChunkRange const &a = chunks[c];
        UsdGenChunkRange const &r = ranges[c];
        if (a.first[0] != r.first[0] || a.count[0] != r.count[0] || a.live != r.live ||
            a.first[1] != r.first[1] || a.stride[1] != r.stride[1] || a.group != r.group || a.domain != r.domain)
            return false;
    }
    return true;
}

void PartitionCaseRun(PartitionCase const &k)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const result = usdGenUsd::Compile(compiler, MakeDesc(k.tileTarget), &graph);
    Check(result.ok && graph.NodeCount() == 2, "partition fixture compiles");
    if (!result.ok) return;
    UsdGenCurveBuffer const fixture = MakeBuffer(k.curves, k.cvp, false, false);
    if (k.cooked)
        for (int n = 0; n < graph.NodeCount(); ++n)
            Check(SetNodeBuffer(graph, UsdGenGraph::NodeId(n), fixture), "partition fixture installs");
    UsdGenRef<const UsdGenElementSet> set = UsdGenCurveSetFromBuffer(fixture, 1, 1);
    Check(bool(set), "partition fixture converts");
    if (!set) return;
    // Uncooked: the engine passes the partition override as the set's counts.
    UsdGenElementSetView view = set->View();
    if (!k.cooked) {
        view = UsdGenElementSetView{};
        view.count[1] = k.curves;
        view.count[2] = k.curves * uint32_t(std::max(0, k.fallbackCvp));
    }
    UsdGenPartitionParams params;
    params.chunkSize = uint32_t(graph.ChunkSize());
    params.groupTarget = uint32_t(graph.TileTarget());
    UsdGenElementLayout const &layout = UsdGenCurveSetLayout();

    for (int pass = 0; pass < 2; ++pass) {   // the second pass keeps every chunk
        graph.Repartition(k.curves, uint32_t(std::max(0, k.fallbackCvp)));
        uint32_t const n = layout.partition(k.curves, view, params, {});
        std::vector<UsdGenChunkRange> ranges(n);
        Check(layout.partition(k.curves, view, params, {ranges.data(), ranges.size()}) == n, "partition count");
        bool same = true;
        for (int id = 0; id < graph.NodeCount(); ++id) same = same && SameChunks(graph, UsdGenGraph::NodeId(id), ranges);
        Check(same, k.what);
        uint64_t cvs = 0;
        for (UsdGenChunkRange const &r : ranges) cvs += r.count[1];
        Check(cvs == uint64_t(view.count[2]) || !view.count[1], "chunk CV counts cover the set");

        auto const tiles = graph.Groups();
        uint32_t const g = layout.groups(n, params, {});
        std::vector<UsdGenChunkGroup> groups(g);
        layout.groups(n, params, {groups.data(), groups.size()});
        bool sameTiles = tiles.size() == groups.size();
        for (size_t t = 0; sameTiles && t < tiles.size(); ++t)
            sameTiles = tiles[t].id == groups[t].id && tiles[t].firstChunk == groups[t].firstChunk &&
                        tiles[t].chunkCount == groups[t].chunkCount;
        Check(sameTiles, k.what);
        Check(int(g) == graph.NumGroups(), "group count");
    }
}

void TestPartition()
{
    PartitionCase const cases[] = {
        {1000, 4, 64, true, 1, "uniform 1000x4"},
        {100000, 4, 64, true, 1, "uniform 100k x4 (49 tiles)"},
        {1500, 0, 64, true, 1, "ragged 1500"},
        {33 * 512, 2, 32, true, 1, "33 chunks, 32 tiles"},
        {34 * 512, 0, 32, true, 1, "ragged 34 chunks"},
        {65 * 512, 3, 32, true, 1, "65 chunks, [16x3,2,15x1]"},
        {0, 4, 64, true, 0, "empty"},
        {5000, 4, 64, false, 6, "uncooked: fallback CVs"},
        {7, 0, 64, true, 1, "ragged single chunk"},
        {300 * 512 + 17, 1, 300, true, 2, "tileTarget clamps to 256"},
    };
    for (PartitionCase const &k : cases) PartitionCaseRun(k);
}

}  // namespace

int main()
{
    TestRegistration();
    TestRoundTrip();
    TestRefused();
    TestPartition();
    g_failures += RunCurveChunkTests();
    if (g_failures) {
        std::printf("testUsdGenCurveSet: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("testUsdGenCurveSet: ok\n");
    return 0;
}
