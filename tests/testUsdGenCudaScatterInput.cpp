#include "usdGen/cudaScatterInput.h"
#include "usdGen/opRegistry.h"

#include "tbb/task_arena.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"CHECK failed: %s (%d)\n",#x,__LINE__); return 1; } } while (false)

static UsdGenGraphDesc Desc(double density = 8.0) {
    UsdGenGraphDesc d;
    UsdGenSurfaceDesc s; s.path=SdfPath("/surface");
    s.restPoints={GfVec3f(0,0,0),GfVec3f(1,0,0),GfVec3f(1,1,0),GfVec3f(0,1,0)};
    s.uv={GfVec2f(0,0),GfVec2f(1,0),GfVec2f(1,1),GfVec2f(0,1)};
    s.faceVertexCounts={4}; s.faceVertexIndices={0,1,2,3}; d.surfaces.push_back(s);
    UsdGenNodeDesc n; n.path=SdfPath("/scatter"); n.type=TfToken("UsdGenScatter"); n.seed=41;
    n.surfaces={s.path}; n.params={{TfToken("density"),VtValue(density),false}};
    d.nodes.push_back(n); return d;
}

static bool Direct(UsdGenGraphDesc const& d, UsdGenCurveBuffer* out) {
    UsdGenParamView p{&d,&d.nodes.front()}; UsdGenDiagnostics diag;
    auto op=UsdGenOpRegistry::Get().Create(TfToken("UsdGenScatter"));
    if (!op || !op->Bind(p,&diag)) return false;
    auto capture=op->CreateCapture(); UsdGenCurveBuffer empty; UsdGenCaptureContext c;
    c.desc=&d; c.params=&p; c.surface=0; c.seed=uint32_t(d.nodes.front().seed); c.diag=&diag;
    if (!op->Capture(c,empty,capture.get(),&diag)) return false;
    *out=capture->Buffer();
    return true;
}

static bool CheckThreadedValidation();

int main() {
    auto d=Desc(); UsdGenCurveBuffer direct; CHECK(Direct(d,&direct));
    std::shared_ptr<const gpu::ScatterGrowRoots> roots; std::string reason;
    CHECK(PrepareCudaScatterInput(d,SdfPath("/scatter"),&roots,&reason)==CudaScatterInputStatus::Ok && roots);
    CHECK(roots->positions.size()==direct.totalCurves && roots->stableIds.size()==direct.totalCurves);
    for(size_t i=0;i<roots->positions.size();++i) {
        CHECK(roots->positions[i].x==direct.px[i]&&roots->positions[i].y==direct.py[i]&&roots->positions[i].z==direct.pz[i]);
        CHECK(roots->stableIds[i]==direct.curveId[i]&&roots->rootPrim[i]==direct.rootPrim[i]);
        CHECK(roots->rootUV[i].x==direct.rootUV[i][0]&&roots->rootUV[i].y==direct.rootUV[i][1]);
        CHECK(roots->rootT[i].x==direct.rootT[i][0]&&roots->rootT[i].y==direct.rootT[i][1]&&roots->rootT[i].z==direct.rootT[i][2]);
        CHECK(roots->rootB[i].x==direct.rootB[i][0]&&roots->rootB[i].y==direct.rootB[i][1]&&roots->rootB[i].z==direct.rootB[i][2]);
        CHECK(roots->rootN[i].x==direct.rootN[i][0]&&roots->rootN[i].y==direct.rootN[i][1]&&roots->rootN[i].z==direct.rootN[i][2]);
    }
    // 02 §6.3: a disabled generator publishes an EMPTY curve set on this lane
    // too, rather than being refused admission.
    {
        auto disabled=d; disabled.nodes[0].enabled=false;
        std::shared_ptr<const gpu::ScatterGrowRoots> empty; std::string why;
        CHECK(PrepareCudaScatterInput(disabled,SdfPath("/scatter"),&empty,&why)==CudaScatterInputStatus::Ok);
        CHECK(empty && empty->positions.empty() && empty->stableIds.empty());
    }
    auto preserved=roots; auto malformed=d; malformed.surfaces[0].faceVertexIndices.pop_back();
    CHECK(PrepareCudaScatterInput(malformed,SdfPath("/scatter"),&roots,&reason)==CudaScatterInputStatus::InvalidSurface && roots==preserved);
    auto badDensity=d; badDensity.nodes[0].params[0].value=VtValue(8);
    CHECK(PrepareCudaScatterInput(badDensity,SdfPath("/scatter"),&roots,&reason)==CudaScatterInputStatus::Unsupported && roots==preserved);
    auto badUv=d; badUv.surfaces[0].uv[2]=GfVec2f(std::numeric_limits<float>::quiet_NaN(),0);
    CHECK(PrepareCudaScatterInput(badUv,SdfPath("/scatter"),&roots,&reason)==CudaScatterInputStatus::InvalidSurface && roots==preserved);
    auto duplicateSubset=d; duplicateSubset.surfaces[0].subsetFaces={0,0};
    CHECK(PrepareCudaScatterInput(duplicateSubset,SdfPath("/scatter"),&roots,&reason)==CudaScatterInputStatus::InvalidSurface && roots==preserved);
    auto zero=Desc(0.0); roots.reset();
    CHECK(PrepareCudaScatterInput(zero,SdfPath("/scatter"),&roots,&reason)==CudaScatterInputStatus::Ok && roots && roots->positions.empty());
    // Pool recycle pin: a full convert, released, then an empty convert
    // must read empty on EVERY plane. The n==0 path skips the guarded
    // assigns, so the recycled shell must be cleared on release; without
    // the clear this returns 1M stale positions alongside 0 stableIds.
    {
        std::shared_ptr<const gpu::ScatterGrowRoots> full; std::string why2;
        CHECK(PrepareCudaScatterInput(d,SdfPath("/scatter"),&full,&why2)==CudaScatterInputStatus::Ok && full && !full->positions.empty());
        full.reset();
        std::shared_ptr<const gpu::ScatterGrowRoots> empty2; std::string why3;
        CHECK(PrepareCudaScatterInput(zero,SdfPath("/scatter"),&empty2,&why3)==CudaScatterInputStatus::Ok && empty2);
        CHECK(empty2->positions.empty() && empty2->stableIds.empty() && empty2->rootPrim.empty() && empty2->rootUV.empty() && empty2->rootT.empty() && empty2->rootB.empty() && empty2->rootN.empty());
    }
    // Capture-shell recycle pin: a full capture, recycled, then an
    // empty-topology capture must read empty on EVERY plane. The empty
    // early return sets only the totals, so the recycled shell must be
    // cleared on release; without the clear the convert's topology
    // validation trips on the stale sizes (CaptureFailed, not Ok).
    {
        std::shared_ptr<const gpu::ScatterGrowRoots> fullCap; std::string whyCap;
        CHECK(PrepareCudaScatterInput(d,SdfPath("/scatter"),&fullCap,&whyCap)==CudaScatterInputStatus::Ok && fullCap && !fullCap->positions.empty());
        fullCap.reset();
        auto emptySurf=d;
        emptySurf.surfaces[0].faceVertexCounts.clear();
        emptySurf.surfaces[0].faceVertexIndices.clear();
        emptySurf.surfaces[0].restPoints.clear();
        emptySurf.surfaces[0].uv.clear();
        std::shared_ptr<const gpu::ScatterGrowRoots> emptyCap; std::string whyEmpty;
        CHECK(PrepareCudaScatterInput(emptySurf,SdfPath("/scatter"),&emptyCap,&whyEmpty)==CudaScatterInputStatus::Ok && emptyCap);
        CHECK(emptyCap->positions.empty() && emptyCap->stableIds.empty() && emptyCap->rootPrim.empty() && emptyCap->rootUV.empty() && emptyCap->rootT.empty() && emptyCap->rootB.empty() && emptyCap->rootN.empty());
    }
    // Scatter is a generator: it declares only density and flip, so any other
    // authored parameter (usdGen:mask included) is refused, not ignored.
    auto undeclared = d;
    undeclared.nodes[0].params.push_back({TfToken("mask"), VtValue(0.0f), false});
    auto emptyPreserved = roots;
    CHECK(PrepareCudaScatterInput(undeclared, SdfPath("/scatter"), &roots, &reason) ==
          CudaScatterInputStatus::Unsupported && roots == emptyPreserved);
    auto mapped=d; mapped.nodes[0].maps.push_back(SdfPath("/ignored"));
    CHECK(PrepareCudaScatterInput(mapped,SdfPath("/scatter"),&roots,&reason)==CudaScatterInputStatus::Unsupported);
    auto level=d;level.nodes[0].params.push_back({TfToken("subdivisionLevel"),VtValue(0),false});
    CHECK(PrepareCudaScatterInput(level,SdfPath("/scatter"),&roots,&reason)==CudaScatterInputStatus::Ok);
    level.nodes[0].params.back().value=VtValue(3);
    CHECK(PrepareCudaScatterInput(level,SdfPath("/scatter"),&roots,&reason)==CudaScatterInputStatus::Unsupported);
    CHECK(CheckThreadedValidation());
    std::puts("testUsdGenCudaScatterInput: PASS"); return 0;
}

// Threaded validation equivalence (r26): past 32768 elements every
// ValidateSurface group scans over workers. Verdicts (status + reason)
// must match the serial spelling exactly, so each case runs in a
// 1-worker arena and the default arena with identical results. Faults
// sit in first/middle/last chunks of every group, so a chunking bug
// that skipped elements fails loudly. Precedence cases pin the group
// order under threading (counts before rest before uv before indices).
static UsdGenGraphDesc BigDesc(int gx, int gy) {
    UsdGenGraphDesc d;
    UsdGenSurfaceDesc s; s.path=SdfPath("/surface");
    for (int y = 0; y <= gy; ++y)
        for (int x = 0; x <= gx; ++x) {
            s.restPoints.push_back(GfVec3f(float(x), float(y), 0.0f));
            s.uv.push_back(GfVec2f(float(x) / float(gx), float(y) / float(gy)));
        }
    for (int y = 0; y < gy; ++y)
        for (int x = 0; x < gx; ++x) {
            int v = y * (gx + 1) + x;
            s.faceVertexCounts.push_back(4);
            s.faceVertexIndices.push_back(v);
            s.faceVertexIndices.push_back(v + 1);
            s.faceVertexIndices.push_back(v + gx + 2);
            s.faceVertexIndices.push_back(v + gx + 1);
        }
    d.surfaces.push_back(s);
    UsdGenNodeDesc n; n.path=SdfPath("/scatter"); n.type=TfToken("UsdGenScatter"); n.seed=7;
    n.surfaces={s.path}; n.params={{TfToken("density"),VtValue(8.0),false}};
    d.nodes.push_back(n); return d;
}

struct ValidateVerdict {
    CudaScatterInputStatus status = CudaScatterInputStatus::Ok;
    std::string reason;
    size_t roots = 0;
};

static ValidateVerdict RunVerdict(UsdGenGraphDesc const& d,
                                 tbb::task_arena* arena) {
    ValidateVerdict v;
    auto run = [&] {
        std::shared_ptr<const gpu::ScatterGrowRoots> roots;
        v.status = PrepareCudaScatterInput(d, SdfPath("/scatter"), &roots,
                                           &v.reason);
        if (v.status == CudaScatterInputStatus::Ok && roots)
            v.roots = roots->positions.size();
    };
    if (arena) arena->execute(run);
    else run();
    return v;
}

static bool VerdictMatches(UsdGenGraphDesc const& d,
                           tbb::task_arena& serialArena,
                           CudaScatterInputStatus want,
                           char const* wantReason) {
    ValidateVerdict s = RunVerdict(d, &serialArena);
    ValidateVerdict t = RunVerdict(d, nullptr);
    if (s.status != want || t.status != want) return false;
    if (want == CudaScatterInputStatus::Ok)
        return s.roots == t.roots;
    return s.reason == wantReason && t.reason == wantReason;
}

static bool CheckThreadedValidation() {
    tbb::task_arena serialArena(1);
    auto big = BigDesc(200, 200);
    // Every validation group sits past the 32768 threading threshold.
    if (big.surfaces[0].restPoints.size() <= 32768 ||
        big.surfaces[0].uv.size() <= 32768 ||
        big.surfaces[0].faceVertexCounts.size() <= 32768 ||
        big.surfaces[0].faceVertexIndices.size() <= 32768)
        return false;
    float const qnan = std::numeric_limits<float>::quiet_NaN();
    if (!VerdictMatches(big, serialArena, CudaScatterInputStatus::Ok, ""))
        return false;
    if (RunVerdict(big, nullptr).roots == 0)
        return false;
    auto badRest = big;
    badRest.surfaces[0].restPoints[0] = GfVec3f(qnan, 0, 0);
    if (!VerdictMatches(badRest, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has non-finite rest position"))
        return false;
    auto badRestLast = big;
    badRestLast.surfaces[0].restPoints.back() = GfVec3f(0, 0, qnan);
    if (!VerdictMatches(badRestLast, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has non-finite rest position"))
        return false;
    auto badUv = big;
    badUv.surfaces[0].uv[badUv.surfaces[0].uv.size() / 2] =
        GfVec2f(0, qnan);
    if (!VerdictMatches(badUv, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has non-finite UV"))
        return false;
    // Infinities in every lane position of both float groups: the
    // exponent-OR scan flags any 0xFF exponent, not just NaN patterns.
    float const inf = std::numeric_limits<float>::infinity();
    auto infRest = big;
    infRest.surfaces[0].restPoints[12345] = GfVec3f(0, -inf, 0);
    if (!VerdictMatches(infRest, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has non-finite rest position"))
        return false;
    auto infUv = big;
    infUv.surfaces[0].uv[23456] = GfVec2f(inf, 0);
    if (!VerdictMatches(infUv, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has non-finite UV"))
        return false;
    // Signaling NaN (bit-built: a float literal would quiet it). The
    // integer OR flags it without touching the FP units, so it can
    // never signal during the scan; isfinite agrees it is non-finite.
    uint32_t const snanBits = 0x7F800001u;
    float snan = 0;
    std::memcpy(&snan, &snanBits, sizeof(snan));
    auto snanRest = big;
    snanRest.surfaces[0].restPoints[34567] = GfVec3f(0, 0, snan);
    if (!VerdictMatches(snanRest, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has non-finite rest position"))
        return false;
    // Subnormals and signed zeros are clean: only the 0xFF exponent
    // fails. (Extreme finite magnitudes live on the empty-faces desc
    // below: on a real mesh they would trip capture's own area/root
    // guards, which is a different verdict.)
    auto tinyLanes = big;
    tinyLanes.surfaces[0].restPoints[111] = GfVec3f(
        std::numeric_limits<float>::denorm_min(), -0.0f, 0.0f);
    tinyLanes.surfaces[0].uv[222] = GfVec2f(
        -std::numeric_limits<float>::denorm_min(), 0.0f);
    if (!VerdictMatches(tinyLanes, serialArena, CudaScatterInputStatus::Ok,
                        ""))
        return false;
    // Index extremes: INT_MIN/INT_MAX fail, points-1 is the top valid
    // slot. The min/max reduction must not mistake its own init
    // values (INT_MAX/INT_MIN) for data.
    auto minIndex = big;
    minIndex.surfaces[0].faceVertexIndices[60000] =
        std::numeric_limits<int>::min();
    if (!VerdictMatches(minIndex, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has out-of-range face index"))
        return false;
    auto maxIndex = big;
    maxIndex.surfaces[0].faceVertexIndices[70000] =
        std::numeric_limits<int>::max();
    if (!VerdictMatches(maxIndex, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has out-of-range face index"))
        return false;
    auto edgeIndex = big;
    edgeIndex.surfaces[0].faceVertexIndices[80000] =
        int(edgeIndex.surfaces[0].restPoints.size() - 1);
    if (!VerdictMatches(edgeIndex, serialArena, CudaScatterInputStatus::Ok,
                        ""))
        return false;
    auto badCount = big;
    badCount.surfaces[0].faceVertexCounts[35000] = 2;
    if (!VerdictMatches(badCount, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has invalid face cardinality"))
        return false;
    auto badCorners = big;
    badCorners.surfaces[0].faceVertexIndices.pop_back();
    if (!VerdictMatches(badCorners, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface face-index cardinality differs from face counts"))
        return false;
    auto badIndex = big;
    badIndex.surfaces[0].faceVertexIndices[100000] = 100000000;
    if (!VerdictMatches(badIndex, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has out-of-range face index"))
        return false;
    auto badIndexLast = big;
    badIndexLast.surfaces[0].faceVertexIndices.back() = -1;
    if (!VerdictMatches(badIndexLast, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has out-of-range face index"))
        return false;
    // Group precedence under threading: the earliest failing group in
    // serial order names the message, wherever the faults sit.
    auto countsAndRest = big;
    countsAndRest.surfaces[0].faceVertexCounts[0] = 1;
    countsAndRest.surfaces[0].restPoints[5] = GfVec3f(qnan, 0, 0);
    if (!VerdictMatches(countsAndRest, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has invalid face cardinality"))
        return false;
    auto restAndIndex = big;
    restAndIndex.surfaces[0].restPoints[5] = GfVec3f(qnan, 0, 0);
    restAndIndex.surfaces[0].faceVertexIndices[7] = -1;
    if (!VerdictMatches(restAndIndex, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has non-finite rest position"))
        return false;
    auto uvAndIndex = big;
    uvAndIndex.surfaces[0].uv[9] = GfVec2f(qnan, 0);
    uvAndIndex.surfaces[0].faceVertexIndices[7] = -1;
    if (!VerdictMatches(uvAndIndex, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has non-finite UV"))
        return false;
    // The empty-faces path scans the same helpers past the threshold.
    auto noFaces = big;
    noFaces.surfaces[0].faceVertexCounts.clear();
    noFaces.surfaces[0].faceVertexIndices.clear();
    if (!VerdictMatches(noFaces, serialArena, CudaScatterInputStatus::Ok, ""))
        return false;
    auto noFacesBad = noFaces;
    noFacesBad.surfaces[0].restPoints.back() = GfVec3f(qnan, 0, 0);
    if (!VerdictMatches(noFacesBad, serialArena,
                        CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has non-finite rest position"))
        return false;
    // Extreme-but-finite lanes (exponent 0xFE) are clean. Empty faces
    // so capture's area guards never see them; the validation scans
    // still stream every lane.
    auto noFacesHuge = noFaces;
    noFacesHuge.surfaces[0].restPoints[333] = GfVec3f(
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(), 1.0f);
    noFacesHuge.surfaces[0].uv[444] =
        GfVec2f(std::numeric_limits<float>::max(), -1.0f);
    if (!VerdictMatches(noFacesHuge, serialArena,
                        CudaScatterInputStatus::Ok, ""))
        return false;
    return true;
}
