#include "usdGen/surfaceRootBindings.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "usdGen/curveRootCapture.h"
#include "usdGen/curveLoader.h"
#ifdef USDGEN_TEST_CUDA_BINDING_CACHE
#include "usdGen/cudaExecution.h"
#include "usdGen/gpu/generation.h"
#endif

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include <thread>
#include <vector>

using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#c); return 1; } } while (false)

static UsdGenGraphDesc Desc() {
    UsdGenGraphDesc d;
    d.description = SdfPath("/Groom");
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Scalp");
    surface.restPoints = {{0,0,0},{1,0,0},{0,1,0}};
    surface.points = surface.restPoints;
    surface.faceVertexCounts = {3}; surface.faceVertexIndices = {0,1,2};
    d.surfaces.push_back(surface);
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/Hair"); curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair"); curves.curveVertexCounts = {2};
    curves.points = {{.2f,.3f,2},{.2f,.3f,3}}; curves.rest = curves.points;
    curves.curveId = {42}; d.curveSets.push_back(curves);
    UsdGenNodeDesc source;
    source.path = SdfPath("/Source"); source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path}; source.surfaces = {surface.path};
    source.params = {{TfToken("rebind"),VtValue(TfToken("always")),false}};
    d.nodes.push_back(source); d.terminal = source.path;
    return d;
}

int main() {
    auto d = Desc();
    auto surface = d.surfaces[0];
    std::string error;
    auto cache = UsdGenRestSurfaceBindingCache::Create(usdGenUsd::Lower(surface),&error);
    CHECK(cache && cache->Matches(usdGenUsd::Lower(surface)) && cache->BytesOwned() > 0);
    auto posed = surface;
    posed.points[0][2] = 10;
    posed.restNormalDomain = UsdGenSurfaceNormalDomain::Constant;
    posed.restNormals = {{0,1,1}};
    CHECK(cache->Matches(usdGenUsd::Lower(posed))); // neither field affects spatial search.
    auto generationChanged = posed;
    ++generationChanged.surfaceGeneration;
    CHECK(cache->Matches(usdGenUsd::Lower(generationChanged))); // broad posed dirty hint is not identity.
    auto changed = surface;
    changed.restPoints[1][0] = 2;
    CHECK(!cache->Matches(usdGenUsd::Lower(changed))); // no generation bump needed.
    changed = surface; ++changed.surfaceGeneration;
    CHECK(cache->Matches(usdGenUsd::Lower(changed))); // no generation-only invalidation.
    changed = surface; changed.faceVertexIndices[0] = 1;
    CHECK(!cache->Matches(usdGenUsd::Lower(changed)));
    changed = surface; changed.worldMatrix.SetTranslate(GfVec3d(1,0,0));
    CHECK(!cache->Matches(usdGenUsd::Lower(changed)));
    changed = surface; changed.restFromCurrentPoints = true;
    CHECK(!cache->Matches(usdGenUsd::Lower(changed)));

    std::atomic<bool> good{true};
    std::array<std::thread,4> readers;
    for (auto& thread : readers) thread = std::thread([&] {
        for (int i=0; i!=32; ++i) {
            UsdGenSurfaceRootBindingResult result;
            std::string failure;
            if (!cache->Bind({{.2f,.3f,2}},GfMatrix4d(1.0),&result,&failure) ||
                result.unresolvedCount || result.rootUV.size()!=1 ||
                std::fabs(result.rootUV[0][0]-.2f)>1.e-6f)
                good.store(false);
        }
    });
    for (auto& thread : readers) thread.join();
    CHECK(good.load());

    // 127 roots remain serial; the repeated 508-root request crosses the
    // cache's parallel threshold. Every lane must reproduce its serial
    // reference exactly, including array order and unresolved accounting.
    VtVec3fArray serialRoots;
    for (int i = 0; i != 127; ++i)
        serialRoots.push_back(GfVec3f(float(i % 17) / 16.0f,
                                      float((i * 7) % 17) / 16.0f,
                                      float(i % 3) - 1.0f));
    UsdGenSurfaceRootBindingResult serial;
    CHECK(cache->Bind(serialRoots, GfMatrix4d(1.0), &serial, &error));
    VtVec3fArray parallelRoots;
    parallelRoots.resize(serialRoots.size() * 4);
    for (size_t i = 0; i != parallelRoots.size(); ++i)
        parallelRoots[i] = serialRoots[i % serialRoots.size()];
    UsdGenSurfaceRootBindingResult parallel;
    CHECK(cache->Bind(parallelRoots, GfMatrix4d(1.0), &parallel, &error));
    CHECK(parallel.unresolvedCount == serial.unresolvedCount * 4);
    for (size_t i = 0; i != parallelRoots.size(); ++i) {
        size_t const lane = i % serialRoots.size();
        CHECK(parallel.rootPrim[i] == serial.rootPrim[lane] &&
              parallel.rootUV[i] == serial.rootUV[lane] &&
              parallel.valid[i] == serial.valid[lane] &&
              parallel.unresolved[i] == serial.unresolved[lane]);
    }
    // Concurrent large queries also share only the immutable tree; each
    // result owns fresh output planes.
    std::array<UsdGenSurfaceRootBindingResult, 4> parallelReaders;
    std::array<std::thread, 4> parallelThreads;
    for (size_t i = 0; i != parallelThreads.size(); ++i)
        parallelThreads[i] = std::thread([&, i] {
            std::string failure;
            if (!cache->Bind(parallelRoots, GfMatrix4d(1.0), &parallelReaders[i], &failure))
                good.store(false);
        });
    for (auto &thread : parallelThreads) thread.join();
    CHECK(good.load());
    for (UsdGenSurfaceRootBindingResult const &reader : parallelReaders)
        CHECK(reader.rootPrim == parallel.rootPrim && reader.rootUV == parallel.rootUV &&
              reader.valid == parallel.valid && reader.unresolved == parallel.unresolved &&
              reader.rootPrim.cdata() != parallel.rootPrim.cdata());
    // A root transform overflow happens inside a worker, but must retain the
    // previous public result because candidate publication is transactional.
    auto retainedParallel = parallel;
    GfMatrix4d overflow(1.0);
    overflow[0][0] = std::numeric_limits<double>::max();
    overflow[3][0] = std::numeric_limits<double>::max();
    CHECK(!cache->Bind(parallelRoots, overflow, &parallel, &error) && !error.empty());
    CHECK(parallel.rootPrim.cdata() == retainedParallel.rootPrim.cdata() &&
          parallel.rootUV.cdata() == retainedParallel.rootUV.cdata() &&
          parallel.valid.cdata() == retainedParallel.valid.cdata());

    auto degenerate = surface;
    degenerate.restPoints.assign(3, GfVec3f(0));
    auto degenerateCache = UsdGenRestSurfaceBindingCache::Create(usdGenUsd::Lower(degenerate), &error);
    CHECK(degenerateCache);
    UsdGenSurfaceRootBindingResult unresolved;
    CHECK(degenerateCache->Bind(parallelRoots, GfMatrix4d(1.0), &unresolved, &error));
    CHECK(unresolved.unresolvedCount == parallelRoots.size());
    for (size_t i = 0; i != parallelRoots.size(); ++i)
        CHECK(unresolved.rootPrim[i] == -1 && unresolved.rootUV[i] == GfVec2f(0) &&
              unresolved.valid[i] == 0 && unresolved.unresolved[i] == 1);

    std::shared_ptr<const UsdGenRestSurfaceBindingCache> slot;
    UsdGenCurveRootCaptureResult roots;
    CHECK(UsdGenCaptureCurveRoots(usdGenUsd::Lower(d.curveSets[0]),usdGenUsd::Addr(usdGenUsd::View(surface)),TfToken("always"),&roots,&error,&slot));
    auto old = slot;
    CHECK(old && old->Matches(usdGenUsd::Lower(surface)));
    CHECK(UsdGenCaptureCurveRoots(usdGenUsd::Lower(d.curveSets[0]),usdGenUsd::Addr(usdGenUsd::View(posed)),TfToken("always"),&roots,&error,&slot));
    CHECK(slot == old);
    changed = surface; changed.restPoints[1][0] = 2;
    CHECK(UsdGenCaptureCurveRoots(usdGenUsd::Lower(d.curveSets[0]),usdGenUsd::Addr(usdGenUsd::View(changed)),TfToken("always"),&roots,&error,&slot));
    CHECK(slot != old && std::fabs(roots.rootUV[0][0]-.1f)<1.e-6f);
    auto current = slot;
    auto oldRoots = roots;
    changed.faceVertexIndices[0] = 99;
    CHECK(!UsdGenCaptureCurveRoots(usdGenUsd::Lower(d.curveSets[0]),usdGenUsd::Addr(usdGenUsd::View(changed)),TfToken("always"),&roots,&error,&slot));
    CHECK(slot == current && roots.rootUV.cdata() == oldRoots.rootUV.cdata());
    UsdGenSurfaceRootBindingResult retained;
    CHECK(old->Bind({{.2f,.3f,2}},GfMatrix4d(1.0),&retained,&error));
    CHECK(std::fabs(retained.rootUV[0][0]-.2f)<1.e-6f);

    // A failure after successful root capture must not replace the loader's
    // prior cache or prior geometry (named-plane validation happens later).
    usdGen::UsdGenGraphSpec const lowered168 = usdGenUsd::Lower(d);
    UsdGenParamView params; params.desc = &lowered168; params.node = &lowered168.nodes[0];
    UsdGenCaptureContext ctx; ctx.desc = &lowered168; ctx.params = &params;
    UsdGenCurveBuffer output;
    UsdGenDiagnostics diagnostics;
    slot.reset();
    CHECK(UsdGenCurveLoader::Load(ctx,usdGenUsd::Lower(d.curveSets[0]),&output,&diagnostics,&slot));
    old = slot;
    auto priorOutput = output;
    d.surfaces[0].restPoints[1][0] = 2;
    UsdGenAuthoredPlaneDesc malformed;
    malformed.name = TfToken("bad"); malformed.domain = UsdGenAuthoredPlaneDomain::Point;
    malformed.type = UsdGenAuthoredPlaneType::Float32; malformed.arity = 1;
    d.curveSets[0].authoredPlanes = {malformed};
    CHECK(!UsdGenCurveLoader::Load(ctx,usdGenUsd::Lower(d.curveSets[0]),&output,&diagnostics,&slot));
    CHECK(slot == old && output.px.cdata() == priorOutput.px.cdata());

#ifdef USDGEN_TEST_CUDA_BINDING_CACHE
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    d = Desc(); d.executionBackend = usdGenTestUtils::Backend("cuda");
    diagnostics = {};
    auto workspace = CreateCudaExecutionWorkspace(-1,&diagnostics);
    auto other = CreateCudaExecutionWorkspace(-1,&diagnostics);
    CHECK(workspace && other);
    auto plan = CompileCudaGraph(usdGenUsd::Lower(d),&diagnostics);
    CHECK(plan);
    auto first = ExecuteCudaGraph(*plan,*workspace,1,1,&diagnostics);
    CHECK(first && !diagnostics.HasErrors());
    auto identity = getCudaRestSurfaceBindingCacheIdentityForTesting(*workspace);
    CHECK(identity && !getCudaRestSurfaceBindingCacheIdentityForTesting(*other));
    auto second = ExecuteCudaGraph(*plan,*workspace,2,2,&diagnostics);
    CHECK(second && identity == getCudaRestSurfaceBindingCacheIdentityForTesting(*workspace));
    auto isolated = ExecuteCudaGraph(*plan,*other,1,1,&diagnostics);
    CHECK(isolated && identity != getCudaRestSurfaceBindingCacheIdentityForTesting(*other));
    auto lease = gpu::AcquireGeometry(first,nullptr);
    CHECK(lease && lease.RootUV().size == 1);
    d.surfaces[0].restPoints[1][0] = 2;
    plan = CompileCudaGraph(usdGenUsd::Lower(d),&diagnostics);
    CHECK(plan);
    auto refreshed = ExecuteCudaGraph(*plan,*workspace,3,3,&diagnostics);
    CHECK(refreshed && identity != getCudaRestSurfaceBindingCacheIdentityForTesting(*workspace));
    float2 oldUv{};
    CHECK(cudaMemcpy(&oldUv,lease.RootUV().data,sizeof(oldUv),cudaMemcpyDeviceToHost)==cudaSuccess);
    CHECK(std::fabs(oldUv.x-.2f)<1.e-6f);
#endif
    std::puts("surface binding cache tests passed");
    return 0;
}
