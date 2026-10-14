// benchUsdGenInstanceScatter — per-stage timing + bit-identity checksums for
// the instancing/scattering path (autoresearch instscatter).
//
// Stages (median of --reps runs, taskset-pinned by the caller):
//   scatter_capture_ms   UsdGenScatterOp::Capture on a 1M-face grid (~1M roots)
//   scatter_digest_ms    UsdGenScatterOp::CaptureDigest on the same surface
//   bake_cards_ms        UsdGenInstancer::Bake, cards+surfaceFrame+twist stress
//   bake_notwist_ms      Bake, cards+surfaceFrame at schema defaults (twist 0)
//   bake_twist_ms        Bake, cards+surfaceFrame uniform twist (no random)
//   bake_tangent_ms      UsdGenInstancer::Bake, cards+curveTangent
//   bake_spheres_ms      UsdGenInstancer::Bake, spheres
//   bake_strided_ms      Bake, cards stress with per-CV displayColor (root-CV gather)
//   instancer_draw_ms    BuildInstancerDataSource + primvar readback
//   draw_vary2_ms        draw assembly over displayColor + an arity-2 uniform plane
//   attr_cook_ms         UsdGenAttributeCookInstances over the scattered roots
//   session_cook_ms      UsdGenSessionCooker::Cook end to end (cache key +
//                        run operators + tile publication + occlusion + diff)
//   cuda_input_ms        PrepareCudaScatterInput (CPU capture + convert)
//   cuda_validate_ms     CudaScatterGrow::ValidateRoots (CPU-only preflight)
//   cuda_grow_begin_ms   CudaScatterGrow::BeginFresh (validate+alloc+H2D+launch)
//   cuda_grow_finish_ms  FinishFreshAsync + sync + CommitFreshFinish
//   vk_build_targets_ms  ScatterGrowPipeline::BuildTargets (host)
//   vk_build_cpu_ms      ScatterGrowPipeline::BuildCpu (host fallback)
//   vk_dispatch_ms       Begin + Poll + queue proof + download (needs --vk-spv)
//
// Every stage prints `result <stage> <median ms> (best .. worst)` plus
// `checksum <stage> <fnv1a64 hex>` over its full output, so A/B runs of
// swapped libraries verify bit-identity. GPU stages print SKIP (and exit 0)
// when no device / SPIR-V is available. Exit is 1 only on a stage error.
//
// Scatter Evaluate is intentionally absent: it is an empty identity hook
// (ops/scatter.cpp), so there is nothing to time.
#include "usdGen/curveBuffer.h"
#include "usdGen/graphDesc.h"
#include "usdGen/maps/attributeInstance.h"
#include "usdGen/maps/attributeMap.h"
#include "usdGen/opRegistry.h"
#include "usdGen/ops/scatter.h"
#include "usdGen/sessionCooker.h"
#include "usdGen/types.h"
#include "usdGenImaging/usdGenInstancer.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/quatf.h"
#include "pxr/base/gf/quath.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"
#include "pxr/imaging/hd/dataSource.h"
#include "pxr/usd/sdf/path.h"

#ifdef USDGEN_ENABLE_CUDA
#include "usdGen/cudaScatterInput.h"
#include "usdGen/executionResources.h"
#include "usdGen/gpu/deviceResources.h"
#include "usdGen/gpu/scatterGrow.h"
#endif
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
#include "usdGen/vulkan/scatterGrowPipeline.h"
#include "vulkanNativeFixture.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;
using namespace usdGenImaging;

namespace {

struct Options {
    int reps = 9;
    int grid = 1000;      // grid x grid quads; --quick sets 200
    int bakeCvs = 4;
    int cudaCv = 8;
    int vkCurves = 16384; // --quick sets 1024
    std::string stage;    // empty = all
    std::string vkSpv;
    bool csv = false;
};

double Median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

// FNV-1a over raw bytes; the bit-identity oracle for A/B library swaps.
struct Fnv {
    uint64_t h = 14695981039346656037ull;
    void Add(void const *p, size_t n)
    {
        auto const *b = static_cast<unsigned char const *>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= uint64_t(b[i]);
            h *= 1099511628211ull;
        }
    }
    void AddSize(size_t n) { Add(&n, sizeof(n)); }
};

template <class A>
void HashArray(Fnv *fnv, A const &a)
{
    fnv->AddSize(a.size());
    if (!a.empty())
        fnv->Add(a.cdata(), a.size() * sizeof(typename A::value_type));
}

template <class T>
void HashVector(Fnv *fnv, std::vector<T> const &v)
{
    fnv->AddSize(v.size());
    if (!v.empty())
        fnv->Add(v.data(), v.size() * sizeof(T));
}

// Same size+bytes fold as HashVector, for the VtArray-backed roots planes
// (identical checksums for identical bytes).
template <class T>
void HashSpan(Fnv *fnv, T const *d, size_t n)
{
    fnv->AddSize(n);
    if (n)
        fnv->Add(d, n * sizeof(T));
}

// Order-sensitive rep fold (XOR zeroes out on even rep counts when every rep
// is identical, as it must be).
void Fold(uint64_t *acc, uint64_t v)
{
    *acc ^= v + 0x9E3779B97F4A7C15ull + (*acc << 6) + (*acc >> 2);
    *acc *= 1099511628211ull;
}

void Report(Options const &opts, char const *stage, std::vector<double> const &ms,
            uint64_t checksum, char const *extra = "")
{
    double const med = Median(ms);
    double const best = *std::min_element(ms.begin(), ms.end());
    double const worst = *std::max_element(ms.begin(), ms.end());
    if (opts.csv) {
        std::printf("%s,%.4f,%.4f,%.4f,%016llx%s%s\n", stage, med, best, worst,
                    (unsigned long long)checksum, extra[0] ? "," : "", extra);
    } else {
        std::printf("result %s_ms %.3f (best %.3f worst %.3f)%s%s\n", stage,
                    med, best, worst, extra[0] ? " " : "", extra);
        std::printf("checksum %s %016llx\n", stage,
                    (unsigned long long)checksum);
    }
    std::fflush(stdout);
}

void ReportSkip(Options const &opts, char const *stage, char const *why)
{
    if (opts.csv)
        std::printf("%s,SKIP,SKIP,SKIP,0,%s\n", stage, why);
    else
        std::printf("result %s_ms SKIP (%s)\n", stage, why);
    std::fflush(stdout);
}

bool WantStage(Options const &opts, char const *stage)
{
    return opts.stage.empty() || opts.stage == stage;
}

// NX x NY unit-grid surface (benchUsdGenChain form): 0.1-unit quads, density
// 100/ft^2 emits ~1 root/face, so curves ~= faces.
UsdGenGraphDesc MakeDesc(int nx, int ny, int seed = 42)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/scatter");
    d.time = 0.0;
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.id = 0;
    s.restPoints = VtVec3fArray((nx + 1) * (ny + 1));
    s.uv = VtVec2fArray((nx + 1) * (ny + 1));
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            int const k = j * (nx + 1) + i;
            s.restPoints[k] = GfVec3f(i * 0.1f, j * 0.1f, 0.0f);
            s.uv[k] = GfVec2f(float(i) / nx, float(j) / ny);
        }
    s.faceVertexCounts = VtIntArray(nx * ny, 4);
    s.faceVertexIndices = VtIntArray(nx * ny * 4);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int const a = j * (nx + 1) + i;
            int const o = (j * nx + i) * 4;
            s.faceVertexIndices[o + 0] = a;
            s.faceVertexIndices[o + 1] = a + 1;
            s.faceVertexIndices[o + 2] = a + nx + 2;
            s.faceVertexIndices[o + 3] = a + nx + 1;
        }
    d.surfaces.push_back(std::move(s));
    UsdGenNodeDesc n;
    n.path = SdfPath("/groom/scatter");
    n.type = TfToken("UsdGenScatter");
    n.enabled = true;
    n.seed = seed;
    n.surfaces.push_back(SdfPath("/groom/surface"));
    d.nodes.push_back(std::move(n));
    return d;
}

uint64_t HashCurves(UsdGenCurveBuffer const &b)
{
    Fnv f;
    HashArray(&f, b.px);
    HashArray(&f, b.py);
    HashArray(&f, b.pz);
    HashArray(&f, b.curveId);
    HashArray(&f, b.rootPrim);
    HashArray(&f, b.rootUV);
    HashArray(&f, b.rootT);
    HashArray(&f, b.rootN);
    HashArray(&f, b.rootB);
    HashArray(&f, b.hairT);
    return f.h;
}

// One authoritative capture; reused as bake/cook/vk input so those stages
// time only themselves.
bool RunCaptureOnce(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *roots,
                    std::string *error)
{
    usdGenRegisterM1Operators();
    UsdGenNodeDesc const &node = desc.nodes[0];
    UsdGenParamView params{&desc, &node};
    UsdGenScatterOp op;
    UsdGenDiagnostics diag;
    if (!op.Bind(params, &diag)) {
        *error = diag.errors.empty() ? "scatter Bind failed" : diag.errors[0];
        return false;
    }
    std::unique_ptr<UsdGenCapture> cap = op.CreateCapture();
    UsdGenCurveBuffer empty;
    UsdGenCaptureContext ctx;
    ctx.desc = &desc;
    ctx.params = &params;
    ctx.surface = 0;
    ctx.seed = uint32_t(node.seed);
    ctx.diag = &diag;
    if (!op.Capture(ctx, empty, cap.get(), &diag)) {
        *error = diag.errors.empty() ? "scatter Capture failed" : diag.errors[0];
        return false;
    }
    *roots = cap->Buffer();
    return true;
}

int RunScatterCapture(Options const &opts, UsdGenGraphDesc const &desc)
{
    UsdGenNodeDesc const &node = desc.nodes[0];
    UsdGenParamView params{&desc, &node};
    UsdGenScatterOp op;
    UsdGenDiagnostics diag;
    if (!op.Bind(params, &diag)) {
        std::printf("scatter_capture error: bind failed\n");
        return 1;
    }
    std::vector<double> ms;
    ms.reserve(size_t(opts.reps));
    uint64_t sum = 0;
    uint32_t ncurves = 0;
    for (int r = 0; r < opts.reps; ++r) {
        std::unique_ptr<UsdGenCapture> cap = op.CreateCapture();
        UsdGenCurveBuffer empty;
        UsdGenCaptureContext ctx;
        ctx.desc = &desc;
        ctx.params = &params;
        ctx.surface = 0;
        ctx.seed = uint32_t(node.seed);
        ctx.diag = &diag;
        auto const t0 = std::chrono::steady_clock::now();
        bool const ok = op.Capture(ctx, empty, cap.get(), &diag);
        auto const t1 = std::chrono::steady_clock::now();
        if (!ok) {
            std::printf("scatter_capture error: %s\n",
                        diag.errors.empty() ? "?" : diag.errors[0].c_str());
            return 1;
        }
        ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        Fold(&sum, HashCurves(cap->Buffer()));
        ncurves = cap->Buffer().totalCurves;
    }
    char extra[64];
    std::snprintf(extra, sizeof(extra), "curves=%u", ncurves);
    Report(opts, "scatter_capture", ms, sum, extra);
    return 0;
}

int RunScatterDigest(Options const &opts, UsdGenGraphDesc const &desc)
{
    UsdGenNodeDesc const &node = desc.nodes[0];
    UsdGenParamView params{&desc, &node};
    UsdGenScatterOp op;
    UsdGenCaptureContext ctx;
    ctx.desc = &desc;
    ctx.params = &params;
    ctx.surface = 0;
    ctx.seed = uint32_t(node.seed);
    std::vector<double> ms;
    ms.reserve(size_t(opts.reps));
    uint64_t sum = 0;
    for (int r = 0; r < opts.reps; ++r) {
        auto const t0 = std::chrono::steady_clock::now();
        UsdGenEpoch const e = op.CaptureDigest(ctx);
        auto const t1 = std::chrono::steady_clock::now();
        ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        Fold(&sum, e[0]);
        Fold(&sum, e[1]);
    }
    Report(opts, "scatter_digest", ms, sum);
    return 0;
}

// Multi-CV bake input grown from real scattered roots along rootN: uniform
// topology (no cvOffsets), real frames, deterministic curveIds.
UsdGenCurveBuffer GrowForBake(UsdGenCurveBuffer const &roots, int cvs)
{
    UsdGenCurveBuffer b;
    uint32_t const n = roots.totalCurves;
    b.totalCurves = n;
    b.totalCvs = n * uint32_t(cvs);
    b.px.resize(b.totalCvs);
    b.py.resize(b.totalCvs);
    b.pz.resize(b.totalCvs);
    for (uint32_t c = 0; c < n; ++c) {
        float const ox = roots.px[c], oy = roots.py[c], oz = roots.pz[c];
        GfVec3f const d = roots.rootN[c];
        for (int i = 0; i < cvs; ++i) {
            float const t = float(i) / float(cvs - 1);
            uint32_t const o = c * uint32_t(cvs) + uint32_t(i);
            b.px[o] = ox + d[0] * t;
            b.py[o] = oy + d[1] * t;
            b.pz[o] = oz + d[2] * t;
        }
    }
    b.curveId = roots.curveId;
    b.rootPrim = roots.rootPrim;
    b.rootUV = roots.rootUV;
    b.rootT = roots.rootT;
    b.rootN = roots.rootN;
    b.rootB = roots.rootB;
    return b;
}

uint64_t HashBake(UsdGenInstanceResult const &r)
{
    Fnv f;
    HashArray(&f, r.translations);
    HashArray(&f, r.rotations);
    HashArray(&f, r.scales);
    HashArray(&f, r.prototypeIndex);
    for (VtIntArray const &a : r.instanceIndices) HashArray(&f, a);
    for (UsdGenPlane const &p : r.varyings) {
        HashArray(&f, p.f);
        HashArray(&f, p.i);
    }
    return f.h;
}

UsdGenInstanceParams BakeParams(char const *primitive, char const *orient,
                              float twist, float twistRandom)
{
    UsdGenInstanceParams p;
    p.primitive = TfToken(primitive);
    p.orient = TfToken(orient);
    p.prototypes = {SdfPath("/groom/Prototypes/cardA"),
                    SdfPath("/groom/Prototypes/cardB")};
    p.scaleRandom = GfVec2f(0.8f, 1.2f);
    p.twist = twist;
    p.twistRandom = twistRandom;
    p.width = 0.02f;
    p.length = 0.09f;
    p.seed = 7;
    return p;
}

int RunBake(Options const &opts, char const *stage, char const *primitive,
            char const *orient, float twist, float twistRandom,
            UsdGenCurveBuffer const &curves, VtVec3fArray const &color)
{
    UsdGenInstanceParams const params =
        BakeParams(primitive, orient, twist, twistRandom);
    UsdGenInstanceCurves input;
    input.curves = &curves;
    input.displayColor = color;
    SdfPath const path("/groom/__usdGenRender/inst_op");
    std::vector<double> ms;
    ms.reserve(size_t(opts.reps));
    uint64_t sum = 0;
    for (int r = 0; r < opts.reps; ++r) {
        UsdGenInstanceResult result;
        std::string error;
        auto const t0 = std::chrono::steady_clock::now();
        bool const ok = UsdGenInstancer::Bake(params, input, path, &result,
                                              &error);
        auto const t1 = std::chrono::steady_clock::now();
        if (!ok) {
            std::printf("%s error: %s\n", stage, error.c_str());
            return 1;
        }
        ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        Fold(&sum, HashBake(result));
    }
    Report(opts, stage, ms, sum);
    return 0;
}

HdContainerDataSourceHandle Child(HdContainerDataSourceHandle const &c,
                                  char const *name)
{
    if (!c)
        return nullptr;
    return HdContainerDataSource::Cast(c->Get(TfToken(name)));
}

int RunDraw(Options const &opts, UsdGenCurveBuffer const &curves,
            VtVec3fArray const &color)
{
    UsdGenInstanceParams const params =
        BakeParams("cards", "surfaceFrame", 15.0f, 30.0f);
    UsdGenInstanceCurves input;
    input.curves = &curves;
    input.displayColor = color;
    UsdGenInstanceResult baked;
    std::string error;
    if (!UsdGenInstancer::Bake(params, input,
                               SdfPath("/groom/__usdGenRender/inst_op"),
                               &baked, &error)) {
        std::printf("instancer_draw error: bake failed: %s\n", error.c_str());
        return 1;
    }
    std::vector<double> ms;
    ms.reserve(size_t(opts.reps));
    uint64_t sum = 0;
    for (int r = 0; r < opts.reps; ++r) {
        auto const t0 = std::chrono::steady_clock::now();
        HdContainerDataSourceHandle const ds =
            UsdGenInstancer::BuildInstancerDataSource(baked,
                                                      SdfPath("/groom"));
        HdContainerDataSourceHandle const by =
            UsdGenInstancer::BuildInstancedByDataSource(
                baked.instancerPath, baked.prototypePaths[0]);
        // Read every published value back, as a Hydra consumer would: this
        // forces retained-source materialization into the timed region.
        // Checksumming stays outside the timer.
        VtVec3fArray tv, sv, cv;
        VtQuathArray qv;
        HdContainerDataSourceHandle const pv = Child(ds, "primvars");
        if (pv) {
            HdDataSourceBaseHandle const hs[4] = {
                Child(pv, "hydra:instanceTranslations"),
                Child(pv, "hydra:instanceRotations"),
                Child(pv, "hydra:instanceScales"),
                Child(pv, "displayColor")};
            for (int k = 0; k < 4; ++k) {
                HdContainerDataSourceHandle const prim =
                    HdContainerDataSource::Cast(hs[k]);
                if (!prim)
                    continue;
                if (HdSampledDataSourceHandle s =
                        HdSampledDataSource::Cast(
                            prim->Get(TfToken("primvarValue")))) {
                    VtValue const val = s->GetValue(0.0);
                    if (val.IsHolding<VtVec3fArray>()) {
                        VtVec3fArray const &a =
                            val.UncheckedGet<VtVec3fArray>();
                        if (k == 0)
                            tv = a;
                        else if (k == 2)
                            sv = a;
                        else if (k == 3)
                            cv = a;
                    } else if (val.IsHolding<VtQuathArray>())
                        qv = val.UncheckedGet<VtQuathArray>();
                }
            }
        }
        (void)by;
        auto const t1 = std::chrono::steady_clock::now();
        if (!ds) {
            std::printf("instancer_draw error: null data source\n");
            return 1;
        }
        ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        Fnv f;
        HashArray(&f, tv);
        HashArray(&f, qv);
        HashArray(&f, sv);
        HashArray(&f, cv);
        Fold(&sum, f.h);
    }
    Report(opts, "instancer_draw", ms, sum);
    return 0;
}

// Draw assembly over an arity-2 uniform varying plus displayColor: covers
// the VtVec2fArray pack leg that instancer_draw never reaches. Baked once
// outside the timer; the timed region mirrors RunDraw.
int RunDrawVary2(Options const &opts, UsdGenCurveBuffer const &curves,
                 VtVec3fArray const &color)
{
    UsdGenCurveBuffer vary = curves;
    UsdGenPlane pair;
    pair.name = TfToken("usdGen:pair");
    pair.interpolation = TfToken("uniform");
    pair.type = TfToken("float");
    pair.arity = 2;
    pair.f.resize(size_t(vary.totalCurves) * 2);
    for (size_t i = 0; i < pair.f.size(); ++i)
        pair.f[i] = float(i % 1024) / 1024.0f - 0.5f;
    vary.extraCurve.push_back(pair);
    UsdGenInstanceParams params =
        BakeParams("cards", "surfaceFrame", 15.0f, 30.0f);
    params.variationPrimvars =
        VtArray<TfToken>{TfToken("displayColor"), TfToken("usdGen:pair")};
    UsdGenInstanceCurves input;
    input.curves = &vary;
    input.displayColor = color;
    UsdGenInstanceResult baked;
    std::string error;
    if (!UsdGenInstancer::Bake(params, input,
                               SdfPath("/groom/__usdGenRender/inst_op"),
                               &baked, &error)) {
        std::printf("draw_vary2 error: bake failed: %s\n", error.c_str());
        return 1;
    }
    std::vector<double> ms;
    ms.reserve(size_t(opts.reps));
    uint64_t sum = 0;
    for (int r = 0; r < opts.reps; ++r) {
        auto const t0 = std::chrono::steady_clock::now();
        HdContainerDataSourceHandle const ds =
            UsdGenInstancer::BuildInstancerDataSource(baked,
                                                      SdfPath("/groom"));
        VtVec3fArray tv, sv, cv;
        VtQuathArray qv;
        VtVec2fArray pv2;
        HdContainerDataSourceHandle const pv = Child(ds, "primvars");
        if (pv) {
            char const *names[5] = {"hydra:instanceTranslations",
                                    "hydra:instanceRotations",
                                    "hydra:instanceScales", "displayColor",
                                    "usdGen:pair"};
            for (int k = 0; k < 5; ++k) {
                HdContainerDataSourceHandle const prim =
                    Child(pv, names[k]);
                if (!prim)
                    continue;
                if (HdSampledDataSourceHandle s =
                        HdSampledDataSource::Cast(
                            prim->Get(TfToken("primvarValue")))) {
                    VtValue const val = s->GetValue(0.0);
                    if (val.IsHolding<VtVec3fArray>()) {
                        VtVec3fArray const &a =
                            val.UncheckedGet<VtVec3fArray>();
                        if (k == 0)
                            tv = a;
                        else if (k == 2)
                            sv = a;
                        else if (k == 3)
                            cv = a;
                    } else if (val.IsHolding<VtQuathArray>()) {
                        qv = val.UncheckedGet<VtQuathArray>();
                    } else if (val.IsHolding<VtVec2fArray>()) {
                        pv2 = val.UncheckedGet<VtVec2fArray>();
                    }
                }
            }
        }
        auto const t1 = std::chrono::steady_clock::now();
        if (!ds) {
            std::printf("draw_vary2 error: null data source\n");
            return 1;
        }
        ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        Fnv f;
        HashArray(&f, tv);
        HashArray(&f, qv);
        HashArray(&f, sv);
        HashArray(&f, cv);
        HashArray(&f, pv2);
        Fold(&sum, f.h);
    }
    Report(opts, "draw_vary2", ms, sum);
    return 0;
}

int RunAttrCook(Options const &opts, UsdGenCurveBuffer const &roots,
                size_t faceCount)
{
    UsdGenAttributeMapSpec spec;
    spec.numFaces = int(faceCount);
    spec.resolution = 4;
    spec.channels = 1;
    std::string error;
    std::shared_ptr<UsdGenAttributeMap> map =
        UsdGenAttributeMap::Create(spec, &error);
    if (!map) {
        std::printf("attr_cook error: %s\n", error.c_str());
        return 1;
    }
    // A value gradient over faces so both keep and drop paths run.
    for (size_t f = 0; f < faceCount; ++f) {
        float const v = float(f % 100) / 99.0f;
        for (int t = 0; t < 4; ++t)
            for (int s = 0; s < 4; ++s)
                map->SetTexel(int(f), s, t, 0, v);
    }
    UsdGenAttributeInstanceInput input;
    input.map = map;
    input.roots.resize(roots.totalCurves);
    for (uint32_t c = 0; c < roots.totalCurves; ++c) {
        input.roots[c].face = roots.rootPrim[c];
        input.roots[c].u = roots.rootUV[c][0];
        input.roots[c].v = roots.rootUV[c][1];
    }
    input.threshold = 0.5f;
    input.numPrototypes = 4;
    std::vector<double> ms;
    ms.reserve(size_t(opts.reps));
    uint64_t sum = 0;
    size_t kept = 0;
    for (int r = 0; r < opts.reps; ++r) {
        UsdGenAttributeInstanceResult result;
        auto const t0 = std::chrono::steady_clock::now();
        bool const ok = UsdGenAttributeCookInstances(input, &result, &error);
        auto const t1 = std::chrono::steady_clock::now();
        if (!ok) {
            std::printf("attr_cook error: %s\n", error.c_str());
            return 1;
        }
        ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        Fnv f;
        HashVector(&f, result.values);
        HashVector(&f, result.keep);
        HashVector(&f, result.prototype);
        f.Add(&result.digest, sizeof(result.digest));
        Fold(&sum, f.h);
        kept = result.kept;
    }
    char extra[64];
    std::snprintf(extra, sizeof(extra), "kept=%zu", kept);
    Report(opts, "attr_cook", ms, sum, extra);
    return 0;
}

uint64_t HashTiles(UsdGenGenerationConstPtr const &gen)
{
    Fnv f;
    for (UsdGenTilePublication const &t : gen->tiles) {
        uint32_t const id = t.tile;
        f.Add(&id, sizeof(id));
        HashArray(&f, t.curveVertexCounts);
        HashArray(&f, t.points);
        HashArray(&f, t.widths);
        HashArray(&f, t.hairT);
        HashArray(&f, t.hairId);
        HashArray(&f, t.st);
        HashArray(&f, t.displayColor);
        f.Add(&t.extentMin, sizeof(t.extentMin));
        f.Add(&t.extentMax, sizeof(t.extentMax));
    }
    return f.h;
}

// Full session cook at bench scale: the per-update end-to-end number the
// op-level stages compose into (cache key, run operators, tile
// publication, occlusion, diff). One persistent cooker, production cache
// budget; a fresh fixed seed per rep defeats cross-rep memoization (every
// production cook carries a new desc) while keeping the Fold deterministic.
int RunSessionCook(Options const &opts, UsdGenGraphDesc const &desc)
{
    UsdGenSessionCooker cooker(
        0, UsdGenSessionCooker::kDefaultExecutionCacheBytes);
    UsdGenGenerationConstPtr prev;
    UsdGenStats stats;
    uint64_t epoch = 0;
    std::vector<double> ms;
    ms.reserve(size_t(opts.reps));
    uint64_t sum = 0;
    size_t tiles = 0, curves = 0, cvs = 0;
    for (int r = 0; r < opts.reps; ++r) {
        UsdGenGraphDesc d = desc;
        d.nodes[0].seed = 42 + r;
        auto shared = std::make_shared<const UsdGenGraphDesc>(d);
        auto const t0 = std::chrono::steady_clock::now();
        UsdGenGenerationConstPtr gen = cooker.Cook(
            shared, UsdGenContext::Interactive, false, {}, 0.0,
            UsdGenCommitReason::NoticeBatchEnd, prev, stats, false, epoch,
            epoch + 1, -1);
        auto const t1 = std::chrono::steady_clock::now();
        if (!gen) {
            std::printf("session_cook error: cook failed\n");
            return 1;
        }
        ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        Fold(&sum, HashTiles(gen));
        tiles = gen->tiles.size();
        curves = 0;
        cvs = 0;
        for (UsdGenTilePublication const &t : gen->tiles) {
            curves += t.curveVertexCounts.size();
            cvs += t.points.size();
        }
        auto candidate = cooker.TakeCacheCandidate();
        if (candidate)
            cooker.CommitCacheCandidate(*candidate);
        prev = gen;
        stats = cooker.Stats();
        ++epoch;
    }
    char extra[96];
    std::snprintf(extra, sizeof(extra), "tiles=%zu curves=%zu cvs=%zu",
                  tiles, curves, cvs);
    Report(opts, "session_cook", ms, sum, extra);
    return 0;
}

#ifdef USDGEN_ENABLE_CUDA
namespace {

struct CudaRelay {
    std::atomic<int> status{-1};
};

void CudaDone(cudaStream_t, cudaError_t status, void *p) noexcept
{
    static_cast<CudaRelay *>(p)->status.store(int(status));
}

int RunCuda(Options const &opts, UsdGenGraphDesc const &desc)
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 1) {
        ReportSkip(opts, "cuda_input", "no CUDA device");
        ReportSkip(opts, "cuda_grow_begin", "no CUDA device");
        ReportSkip(opts, "cuda_grow_finish", "no CUDA device");
        return 0;
    }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        device = 0;

    // Input prep: CPU scatter capture + conversion to device input.
    std::vector<double> inMs;
    inMs.reserve(size_t(opts.reps));
    uint64_t inSum = 0;
    size_t nroots = 0;
    std::shared_ptr<const gpu::ScatterGrowRoots> roots;
    for (int r = 0; r < opts.reps; ++r) {
        std::string reason;
        std::shared_ptr<const gpu::ScatterGrowRoots> out;
        auto const t0 = std::chrono::steady_clock::now();
        CudaScatterInputStatus const st = PrepareCudaScatterInput(
            desc, SdfPath("/groom/scatter"), &out, &reason);
        auto const t1 = std::chrono::steady_clock::now();
        if (st != CudaScatterInputStatus::Ok) {
            std::printf("cuda_input error: %s\n", reason.c_str());
            return 1;
        }
        inMs.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        Fnv f;
        HashSpan(&f, out->positions.data(), out->positions.size());
        HashSpan(&f, out->stableIds.data(), out->stableIds.size());
        Fold(&inSum, f.h);
        nroots = out->positions.size();
        if (r == 0)
            roots = out;
    }
    char extra[64];
    std::snprintf(extra, sizeof(extra), "roots=%zu", nroots);
    Report(opts, "cuda_input", inMs, inSum, extra);

    gpu::ScatterGrowControls controls;
    controls.cvCount = uint32_t(opts.cudaCv);
    controls.seed = 42;
    controls.length = 1.0;
    controls.randomLo = 0.8;
    controls.randomHi = 1.2;
    gpu::ScatterGrowRequirements req;
    if (gpu::GetScatterGrowRequirements(nroots, controls.cvCount, &req) !=
        gpu::ScatterGrowStatus::Ok) {
        std::printf("cuda_grow error: requirements rejected\n");
        return 1;
    }
    if (!gpu::ConfigureCudaExecutionResources(
            device, {req.peakBytes + (size_t{64} << 20), 0})) {
        ReportSkip(opts, "cuda_grow_begin", "resource configure failed");
        ReportSkip(opts, "cuda_grow_finish", "resource configure failed");
        return 0;
    }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        // Box-wide GPU memory pressure (other tenants) can refuse even a
        // stream: skip the leg, keep the rest of the bench green.
        ReportSkip(opts, "cuda_grow_begin", "cuda setup unavailable");
        ReportSkip(opts, "cuda_grow_finish", "cuda setup unavailable");
        return 0;
    }
    std::vector<double> beginMs, finishMs;
    beginMs.reserve(size_t(opts.reps));
    finishMs.reserve(size_t(opts.reps));
    uint64_t sum = 0;
    for (int r = 0; r < opts.reps; ++r) {
        gpu::CudaScatterGrow grow;
        auto const t0 = std::chrono::steady_clock::now();
        gpu::ScatterGrowStatus st =
            grow.BeginFresh(roots, controls, stream);
        auto const t1 = std::chrono::steady_clock::now();
        if (st != gpu::ScatterGrowStatus::Ok) {
            std::printf("cuda_grow_begin error: status %d\n", int(st));
            return 1;
        }
        beginMs.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        CudaRelay relay;
        auto const t2 = std::chrono::steady_clock::now();
        st = grow.FinishFreshAsync(stream, CudaDone, &relay);
        bool ok = st == gpu::ScatterGrowStatus::Ok &&
                  cudaStreamSynchronize(stream) == cudaSuccess &&
                  relay.status.load() == int(cudaSuccess) &&
                  grow.CommitFreshFinish() == gpu::ScatterGrowStatus::Ok;
        auto const t3 = std::chrono::steady_clock::now();
        if (!ok) {
            std::printf("cuda_grow_finish error: status %d\n", int(st));
            return 1;
        }
        finishMs.push_back(
            std::chrono::duration<double, std::milli>(t3 - t2).count());
        // Download the point plane for the checksum (outside the timer).
        auto const view = grow.view();
        std::vector<float3> pts(view.pointCount);
        if (!pts.empty() &&
            (cudaMemcpy(pts.data(), view.points.data,
                        pts.size() * sizeof(float3),
                        cudaMemcpyDeviceToHost) != cudaSuccess)) {
            std::printf("cuda_grow error: download failed\n");
            return 1;
        }
        Fnv f;
        HashVector(&f, pts);
        Fold(&sum, f.h);
    }
    cudaStreamDestroy(stream);
    std::snprintf(extra, sizeof(extra), "roots=%zu cv=%d", nroots,
                  opts.cudaCv);
    Report(opts, "cuda_grow_begin", beginMs, sum, extra);
    Report(opts, "cuda_grow_finish", finishMs, sum, extra);
    return 0;
}

// CPU-only grow preflight: needs no device, so it runs (and gates
// validate optimizations) even when the grow legs SKIP on memory
// pressure. Roots prep once outside the timer; the grow controls match
// RunCuda's.
int RunCudaValidate(Options const &opts, UsdGenGraphDesc const &desc)
{
    std::string reason;
    std::shared_ptr<const gpu::ScatterGrowRoots> roots;
    if (PrepareCudaScatterInput(desc, SdfPath("/groom/scatter"), &roots,
                                &reason) != CudaScatterInputStatus::Ok) {
        std::printf("cuda_validate error: %s\n", reason.c_str());
        return 1;
    }
    gpu::ScatterGrowControls controls;
    controls.cvCount = uint32_t(opts.cudaCv);
    controls.seed = 42;
    controls.length = 1.0;
    controls.randomLo = 0.8;
    controls.randomHi = 1.2;
    std::vector<double> ms;
    ms.reserve(size_t(opts.reps));
    uint64_t sum = 0;
    size_t total = 0;
    for (int r = 0; r < opts.reps; ++r) {
        auto const t0 = std::chrono::steady_clock::now();
        gpu::ScatterGrowStatus const st =
            gpu::CudaScatterGrow::ValidateRoots(roots, controls, &total);
        auto const t1 = std::chrono::steady_clock::now();
        if (st != gpu::ScatterGrowStatus::Ok) {
            std::printf("cuda_validate error: status %d\n", int(st));
            return 1;
        }
        ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        Fnv f;
        int const s = int(st);
        f.Add(&s, sizeof(s));
        f.Add(&total, sizeof(total));
        Fold(&sum, f.h);
    }
    char extra[64];
    std::snprintf(extra, sizeof(extra), "roots=%zu cv=%d",
                  roots->positions.size(), opts.cudaCv);
    Report(opts, "cuda_validate", ms, sum, extra);
    return 0;
}

}  // namespace
#endif  // USDGEN_ENABLE_CUDA

#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
namespace {

int RunVkHost(Options const &opts, UsdGenCurveBuffer const &roots)
{
    using vulkan::ScatterGrowPipeline;
    size_t const n = std::min<size_t>(size_t(opts.vkCurves),
                                      roots.totalCurves);
    std::vector<float> positions(3 * n);
    std::vector<uint64_t> ids(n);
    std::vector<int32_t> prim(n);
    std::vector<float> uv(2 * n), t(3 * n), b(3 * n), nn(3 * n);
    for (size_t i = 0; i < n; ++i) {
        positions[3 * i + 0] = roots.px[i];
        positions[3 * i + 1] = roots.py[i];
        positions[3 * i + 2] = roots.pz[i];
        ids[i] = roots.curveId[i];
        prim[i] = roots.rootPrim[i];
        uv[2 * i + 0] = roots.rootUV[i][0];
        uv[2 * i + 1] = roots.rootUV[i][1];
        t[3 * i + 0] = roots.rootT[i][0];
        t[3 * i + 1] = roots.rootT[i][1];
        t[3 * i + 2] = roots.rootT[i][2];
        b[3 * i + 0] = roots.rootB[i][0];
        b[3 * i + 1] = roots.rootB[i][1];
        b[3 * i + 2] = roots.rootB[i][2];
        nn[3 * i + 0] = roots.rootN[i][0];
        nn[3 * i + 1] = roots.rootN[i][1];
        nn[3 * i + 2] = roots.rootN[i][2];
    }
    ScatterGrowPipeline::Controls controls;
    controls.cvCount = 8;
    controls.seed = 42;
    controls.length = 1.0;
    controls.randomLo = 0.8;
    controls.randomHi = 1.2;

    std::vector<double> tgtMs;
    tgtMs.reserve(size_t(opts.reps));
    std::vector<float> targets;
    uint64_t tgtSum = 0;
    for (int r = 0; r < opts.reps; ++r) {
        auto const t0 = std::chrono::steady_clock::now();
        targets = ScatterGrowPipeline::BuildTargets(ids, controls);
        auto const t1 = std::chrono::steady_clock::now();
        tgtMs.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        Fnv f;
        HashVector(&f, targets);
        Fold(&tgtSum, f.h);
    }
    char extra[64];
    std::snprintf(extra, sizeof(extra), "roots=%zu", n);
    Report(opts, "vk_build_targets", tgtMs, tgtSum, extra);

    std::vector<double> cpuMs;
    cpuMs.reserve(size_t(opts.reps));
    uint64_t cpuSum = 0;
    for (int r = 0; r < opts.reps; ++r) {
        ScatterGrowPipeline::Outputs out;
        auto const t0 = std::chrono::steady_clock::now();
        bool const ok = ScatterGrowPipeline::BuildCpu(
            positions, ids, prim, uv, t, b, nn, targets, controls, &out);
        auto const t1 = std::chrono::steady_clock::now();
        if (!ok) {
            std::printf("vk_build_cpu error: BuildCpu rejected\n");
            return 1;
        }
        cpuMs.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        Fnv f;
        HashVector(&f, out.points);
        HashVector(&f, out.widths);
        Fold(&cpuSum, f.h);
    }
    Report(opts, "vk_build_cpu", cpuMs, cpuSum, extra);
    return 0;
}

// Test-only queue proof + upload/download, mirroring
// testUsdGenVulkanScatterGrowPipeline.
bool VkProve(std::shared_ptr<NativeOwner> const &native)
{
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS)
        return false;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) ==
               VK_SUCCESS &&
           vkWaitForFences(native->device, 1, &native->fence, VK_TRUE,
                           10000000000ull) == VK_SUCCESS;
}

template <typename T>
std::shared_ptr<vulkan::ChargedBuffer> VkUpload(
    std::shared_ptr<vulkan::DeviceContext> const &context,
    std::shared_ptr<NativeOwner> const &native, std::vector<T> const &values)
{
    using vulkan::ChargedBuffer;
    VkDeviceSize bytes = values.empty()
                             ? 4
                             : static_cast<VkDeviceSize>(values.size() *
                                                         sizeof(T));
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto buffer = ChargedBuffer::Create(
        context, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Active);
    if (!buffer)
        return buffer;
    void *mapped = nullptr;
    if (vkMapMemory(native->device, buffer->memory(), 0, bytes, 0, &mapped) !=
        VK_SUCCESS)
        return std::shared_ptr<ChargedBuffer>{};
    if (!values.empty())
        std::memcpy(mapped, values.data(), values.size() * sizeof(T));
    vkUnmapMemory(native->device, buffer->memory());
    return buffer;
}

template <typename T>
bool VkDownload(std::shared_ptr<NativeOwner> const &native,
                std::shared_ptr<const vulkan::ChargedBuffer> const &source,
                std::vector<T> *values)
{
    using vulkan::ChargedBuffer;
    VkDeviceSize bytes = values->size() * sizeof(T);
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto staging = ChargedBuffer::Create(
        source->context(), bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!staging)
        return false;
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = native->commands;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(native->device, &ai, &command) != VK_SUCCESS)
        return false;
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) {
        vkFreeCommandBuffers(native->device, native->commands, 1, &command);
        return false;
    }
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(command, source->buffer(), staging->buffer(), 1, &region);
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
    if (vkEndCommandBuffer(command) != VK_SUCCESS ||
        vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(native->device, native->commands, 1, &command);
        return false;
    }
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    if (vkQueueSubmit(native->queue, 1, &submit, native->fence) != VK_SUCCESS ||
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE,
                        10000000000ull) != VK_SUCCESS) {
        vkFreeCommandBuffers(native->device, native->commands, 1, &command);
        return false;
    }
    void *mapped = nullptr;
    if (vkMapMemory(native->device, staging->memory(), 0, bytes, 0, &mapped) !=
        VK_SUCCESS) {
        vkFreeCommandBuffers(native->device, native->commands, 1, &command);
        return false;
    }
    std::memcpy(values->data(), mapped, bytes);
    vkUnmapMemory(native->device, staging->memory());
    vkFreeCommandBuffers(native->device, native->commands, 1, &command);
    return true;
}

int RunVkDispatch(Options const &opts, UsdGenCurveBuffer const &roots)
{
    using vulkan::DeviceContext;
    using vulkan::ScatterGrowPipeline;
    if (opts.vkSpv.empty()) {
        ReportSkip(opts, "vk_dispatch", "no --vk-spv");
        return 0;
    }
    std::ifstream shader(opts.vkSpv, std::ios::binary);
    if (!shader) {
        ReportSkip(opts, "vk_dispatch", "SPIR-V unreadable");
        return 0;
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(shader)), {});
    if (bytes.empty() || bytes.size() % sizeof(uint32_t) != 0) {
        ReportSkip(opts, "vk_dispatch", "SPIR-V malformed");
        return 0;
    }
    std::vector<uint32_t> code(bytes.size() / sizeof(uint32_t));
    std::memcpy(code.data(), bytes.data(), bytes.size());
    bool unavailable = false;
    auto native = CreateNative(&unavailable);
    if (unavailable || !native) {
        ReportSkip(opts, "vk_dispatch", "no Vulkan device");
        return 0;
    }
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(native->physical, &props);
        if (props.limits.maxPerStageDescriptorStorageBuffers < 20 ||
            props.limits.maxDescriptorSetStorageBuffers < 20) {
            ReportSkip(opts, "vk_dispatch", "weak device");
            return 0;
        }
    }
    DeviceContext::CreateInfo info;
    info.instance = native->instance;
    info.physicalDevice = native->physical;
    info.device = native->device;
    info.computeQueue = native->queue;
    info.computeQueueFamily = native->family;
    info.physicalIndex = native->physicalIndex;
    info.resourceDeviceId = 7004;
    info.nativeLifetime = native;
    info.resources = {size_t{256} << 20, 0};
    auto context = DeviceContext::Create(info);
    if (!context) {
        ReportSkip(opts, "vk_dispatch", "context create failed");
        return 0;
    }
    info.nativeLifetime.reset();
    VkResult status = VK_SUCCESS;
    auto pipeline = ScatterGrowPipeline::Create(context, code, &status);
    if (!pipeline || status != VK_SUCCESS) {
        ReportSkip(opts, "vk_dispatch", "pipeline create failed");
        return 0;
    }

    uint32_t const curves = uint32_t(std::min<size_t>(
        size_t(opts.vkCurves), roots.totalCurves));
    std::vector<float> uv(2 * curves), t(3 * curves), b(3 * curves),
        nn(3 * curves);
    std::vector<uint64_t> ids(curves);
    std::vector<int32_t> prim(curves);
    std::vector<float> pos3(3 * curves);
    for (uint32_t i = 0; i < curves; ++i) {
        pos3[3 * i + 0] = roots.px[i];
        pos3[3 * i + 1] = roots.py[i];
        pos3[3 * i + 2] = roots.pz[i];
        ids[i] = roots.curveId[i];
        prim[i] = roots.rootPrim[i];
        uv[2 * i + 0] = roots.rootUV[i][0];
        uv[2 * i + 1] = roots.rootUV[i][1];
        t[3 * i + 0] = roots.rootT[i][0];
        t[3 * i + 1] = roots.rootT[i][1];
        t[3 * i + 2] = roots.rootT[i][2];
        b[3 * i + 0] = roots.rootB[i][0];
        b[3 * i + 1] = roots.rootB[i][1];
        b[3 * i + 2] = roots.rootB[i][2];
        nn[3 * i + 0] = roots.rootN[i][0];
        nn[3 * i + 1] = roots.rootN[i][1];
        nn[3 * i + 2] = roots.rootN[i][2];
    }
    ScatterGrowPipeline::Controls controls;
    controls.cvCount = 8;
    controls.seed = 42;
    controls.length = 1.0;
    controls.randomLo = 0.8;
    controls.randomHi = 1.2;
    std::vector<float> const targets =
        ScatterGrowPipeline::BuildTargets(ids, controls);

    struct Owner {
        std::shared_ptr<vulkan::ChargedBuffer> positions, stableIds, rootPrim,
            rootUV, rootT, rootB, rootN, targets;
    };
    auto owner = std::make_shared<Owner>();
    owner->positions = VkUpload(context, native, pos3);
    owner->stableIds = VkUpload(context, native, ids);
    owner->rootPrim = VkUpload(context, native, prim);
    owner->rootUV = VkUpload(context, native, uv);
    owner->rootT = VkUpload(context, native, t);
    owner->rootB = VkUpload(context, native, b);
    owner->rootN = VkUpload(context, native, nn);
    owner->targets = VkUpload(context, native, targets);
    if (!owner->positions || !owner->stableIds || !owner->rootPrim ||
        !owner->rootUV || !owner->rootT || !owner->rootB || !owner->rootN ||
        !owner->targets) {
        ReportSkip(opts, "vk_dispatch", "upload failed");
        return 0;
    }
    ScatterGrowPipeline::Input in;
    in.positions = owner->positions;
    in.stableIds = owner->stableIds;
    in.rootPrim = owner->rootPrim;
    in.rootUV = owner->rootUV;
    in.rootT = owner->rootT;
    in.rootB = owner->rootB;
    in.rootN = owner->rootN;
    in.targets = owner->targets;

    std::vector<double> ms;
    ms.reserve(size_t(opts.reps));
    uint64_t sum = 0;
    for (int r = 0; r < opts.reps; ++r) {
        auto const t0 = std::chrono::steady_clock::now();
        std::unique_ptr<ScatterGrowPipeline::Candidate> const cand =
            pipeline->Begin(owner, in, curves, controls, &status);
        ScatterGrowPipeline::Candidate::Semantic sem =
            ScatterGrowPipeline::Candidate::Semantic::NonFinite;
        bool ok = cand && status == VK_SUCCESS;
        if (ok) {
            // Poll until device proof; then a queue fence proves completion.
            for (;;) {
                VkResult const pr = cand->Poll(&sem);
                if (pr == VK_NOT_READY)
                    continue;
                ok = pr == VK_SUCCESS &&
                     sem == ScatterGrowPipeline::Candidate::Semantic::Ok;
                break;
            }
        }
        if (ok)
            ok = VkProve(native);
        std::vector<float> pts;
        if (ok) {
            pts.resize(size_t(curves) * controls.cvCount * 3);
            ok = VkDownload(native, cand->output(), &pts);
        }
        auto const t1 = std::chrono::steady_clock::now();
        if (!ok) {
            std::printf("vk_dispatch error: submit/proof/download failed\n");
            return 1;
        }
        ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        Fnv f;
        HashVector(&f, pts);
        Fold(&sum, f.h);
    }
    char extra[64];
    std::snprintf(extra, sizeof(extra), "roots=%u", curves);
    Report(opts, "vk_dispatch", ms, sum, extra);
    return 0;
}

}  // namespace
#endif  // USDGEN_ENABLE_VULKAN_RUNTIME

}  // namespace

int main(int argc, char **argv)
{
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string const a(argv[i]);
        if (a == "--reps" && i + 1 < argc)
            opts.reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--grid" && i + 1 < argc)
            opts.grid = std::max(2, std::atoi(argv[++i]));
        else if (a == "--stage" && i + 1 < argc)
            opts.stage = argv[++i];
        else if (a == "--vk-spv" && i + 1 < argc)
            opts.vkSpv = argv[++i];
        else if (a == "--vk-curves" && i + 1 < argc)
            opts.vkCurves = std::max(1, std::atoi(argv[++i]));
        else if (a == "--quick") {
            opts.grid = 200;
            opts.vkCurves = 1024;
        } else if (a == "--csv")
            opts.csv = true;
        else {
            std::printf("unknown arg: %s\n", a.c_str());
            return 1;
        }
    }
    if (opts.csv)
        std::printf("stage,median_ms,best_ms,worst_ms,checksum,extra\n");

    usdGenRegisterM1Operators();
    UsdGenGraphDesc const desc = MakeDesc(opts.grid, opts.grid);
    size_t const faceCount = desc.surfaces[0].faceVertexCounts.size();

    if (WantStage(opts, "scatter_capture"))
        if (int rc = RunScatterCapture(opts, desc))
            return rc;
    if (WantStage(opts, "scatter_digest"))
        if (int rc = RunScatterDigest(opts, desc))
            return rc;

    bool const needRoots =
        WantStage(opts, "") || WantStage(opts, "bake_cards") ||
        WantStage(opts, "bake_tangent") || WantStage(opts, "bake_spheres") ||
        WantStage(opts, "bake_notwist") || WantStage(opts, "bake_twist") ||
        WantStage(opts, "bake_strided") || WantStage(opts, "instancer_draw") ||
        WantStage(opts, "draw_vary2") || WantStage(opts, "attr_cook") ||
        WantStage(opts, "vk_build_targets") || WantStage(opts, "vk_build_cpu") ||
        WantStage(opts, "vk_dispatch");
    UsdGenCurveBuffer roots;
    if (needRoots) {
        std::string error;
        if (!RunCaptureOnce(desc, &roots, &error)) {
            std::printf("setup error: %s\n", error.c_str());
            return 1;
        }
        if (!opts.csv)
            std::printf("setup: %u roots on %zu faces\n", roots.totalCurves,
                        faceCount);
    }
    if (WantStage(opts, "bake_cards") || WantStage(opts, "bake_tangent") ||
        WantStage(opts, "bake_spheres") || WantStage(opts, "bake_notwist") ||
        WantStage(opts, "bake_twist") || WantStage(opts, "bake_strided") ||
        WantStage(opts, "instancer_draw") || WantStage(opts, "draw_vary2")) {
        UsdGenCurveBuffer const curves = GrowForBake(roots, opts.bakeCvs);
        VtVec3fArray color(curves.totalCurves);
        for (uint32_t c = 0; c < curves.totalCurves; ++c)
            color[c] = GfVec3f(float(c % 256) / 255.0f, 0.5f, 0.25f);
        // bake_cards is the twist stress case; bake_notwist is the schema
        // default (twist 0/0, the common path); bake_twist is uniform twist
        // (constant angle, hoistable trig).
        if (WantStage(opts, "bake_cards"))
            if (int rc = RunBake(opts, "bake_cards", "cards", "surfaceFrame",
                                 15.0f, 30.0f, curves, color))
                return rc;
        if (WantStage(opts, "bake_notwist"))
            if (int rc = RunBake(opts, "bake_notwist", "cards", "surfaceFrame",
                                 0.0f, 0.0f, curves, color))
                return rc;
        if (WantStage(opts, "bake_twist"))
            if (int rc = RunBake(opts, "bake_twist", "cards", "surfaceFrame",
                                 15.0f, 0.0f, curves, color))
                return rc;
        if (WantStage(opts, "bake_tangent"))
            if (int rc = RunBake(opts, "bake_tangent", "cards",
                                 "curveTangent", 15.0f, 30.0f, curves, color))
                return rc;
        if (WantStage(opts, "bake_spheres"))
            if (int rc = RunBake(opts, "bake_spheres", "spheres",
                                 "surfaceFrame", 15.0f, 30.0f, curves, color))
                return rc;
        // bake_strided re-runs the cards stress case with a per-CV color
        // source (one color per CV, sampled at each curve's root CV).
        if (WantStage(opts, "bake_strided")) {
            VtVec3fArray colorCv(curves.totalCvs);
            for (uint32_t i = 0; i < curves.totalCvs; ++i)
                colorCv[i] =
                    GfVec3f(float(i % 256) / 255.0f, 0.5f, 0.25f);
            if (int rc = RunBake(opts, "bake_strided", "cards",
                                 "surfaceFrame", 15.0f, 30.0f, curves,
                                 colorCv))
                return rc;
        }
        if (WantStage(opts, "instancer_draw"))
            if (int rc = RunDraw(opts, curves, color))
                return rc;
        if (WantStage(opts, "draw_vary2"))
            if (int rc = RunDrawVary2(opts, curves, color))
                return rc;
    }
    if (WantStage(opts, "attr_cook"))
        if (int rc = RunAttrCook(opts, roots, faceCount))
            return rc;
    if (WantStage(opts, "session_cook"))
        if (int rc = RunSessionCook(opts, desc))
            return rc;

#ifdef USDGEN_ENABLE_CUDA
    if (WantStage(opts, "cuda_validate"))
        if (int rc = RunCudaValidate(opts, desc))
            return rc;
    if (WantStage(opts, "cuda_input") || WantStage(opts, "cuda_grow_begin") ||
        WantStage(opts, "cuda_grow_finish") || WantStage(opts, "")) {
        if (int rc = RunCuda(opts, desc))
            return rc;
    }
#else
    if (!opts.stage.empty() &&
        (opts.stage == "cuda_input" || opts.stage == "cuda_validate" ||
         opts.stage == "cuda_grow_begin" || opts.stage == "cuda_grow_finish"))
        ReportSkip(opts, opts.stage.c_str(), "CUDA build off");
#endif
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
    if (WantStage(opts, "vk_build_targets") || WantStage(opts, "vk_build_cpu"))
        if (int rc = RunVkHost(opts, roots))
            return rc;
    if (WantStage(opts, "vk_dispatch"))
        if (int rc = RunVkDispatch(opts, roots))
            return rc;
#else
    if (!opts.stage.empty() &&
        (opts.stage == "vk_build_targets" || opts.stage == "vk_build_cpu" ||
         opts.stage == "vk_dispatch"))
        ReportSkip(opts, opts.stage.c_str(), "Vulkan build off");
#endif
    return 0;
}
