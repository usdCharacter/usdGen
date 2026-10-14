// SPDX-License-Identifier: MIT
// benchUsdGenShaderRender - shader/viewport/render benchmark driver.
//
// Times the cost of shading, drawing, and presenting instanced groom
// geometry (not generating it): per configuration it reports the cold first
// frame, CPU-side draw submission (Render without glFinish), total frame
// time (Render + glFinish), and a color-AOV checksum for bit-identity.
//
// Scenes: usd/fixtures/{sr,ss,gr,gs}{1m,10m}.usda from
// make_shader_scenes.py (1M/10M hairs x scatter/guides x RBF/surface).
// --deform (default for animated scenes) cycles time like benchUsdGenStorm
// --deform, then measures the static pose in the same invocation; the
// static-vs-deform delta is the per-frame publish+re-upload+redraw cost.
// --static measures the static pose only. --deform-only skips the trailing
// static baseline (for memory-capped boxes; run --static separately).
// --dump-ppm PATH writes the last frame's color AOV as a binary PPM
// (debug only). USDGEN_BENCH_STATS=1 prints every numeric GetRenderStats
// counter to stderr after each run.
//
// The bench renders the stage pseudo-root with an explicit full-buffer
// framing: the Key/Fill lights are siblings of the groom (a Groom-rooted
// render shades exact-black), and without SetFraming the engine draws an
// empty data window (all-clear image) even with a valid camera.
//
// Novel vs cached deform: run with warmup+frames == scene span (RBF scenes
// span 12: --warmup 2 --frames 10; surface scenes span 4: --warmup 1
// --frames 3). Both repeats then cook every frame of the span: cold
// repeat 0 is the first loop iteration (cold allocator/scratch/topology
// caches) and kept repeats are subsequent iterations (warm). There are
// no execution-cache HITS on repeats (trace-verified: displace strands
// and BuildFurOcclusion evaluate on every repeat; Scheduler::Run is
// cheaper on kept repeats only because static upstream nodes hit). The
// 64MB execution cache (sessionCooker.h) cannot retain deform frames
// (~96MB+ of positions alone at 1M), and the drain compares consecutive
// snapshots, so every repeat frame is a novel transition anyway (t11->t0
// across the repeat boundary, then t0->t1...). novelFrameMs reports the
// cold-iteration median, frameMs the warm-iteration median; their gap is
// warmth (allocation, scratch, driver caches), not skipped evaluation.
//
// Counter honesty: HdPerfLog drawCalls/drawBatches/itemsDrawn are reset by
// every HdStCommandBuffer::Commit (hdSt/commandBuffer.cpp), so a final read
// reflects only the LAST render pass of the last frame, not frame totals.
// This bench therefore reports cumulative GetRenderStats() bytes
// (primvar/topology/gpuMemoryUsed/copyBufferCpuToGpu) instead, plus timings
// and checksums. The hair tiles batch into a single indirect draw batch
// (verified via TF_DEBUG=HDST_DRAW), so batch counts carry no signal.
//
// Timing (benchUsdGenStorm §4.5 convention): per repeat, `warmup` frames
// then `frames` timed; the first (cold) repeat is discarded; median is the
// median of per-repeat medians, p90/min over retained frame samples.
// firstFrameMs is the very first Render of the invocation (cold: first cook
// + Storm shader compile + first sync + draw). shaderCompileMs is NOT
// directly separable (no engine-side usdGen stats hook yet); estimate it
// offline as firstFrameMs - steadyDeformMs - cookMs, with cookMs from a
// TF_DEBUG=USDGEN_COMMIT instrumented run.
#include "eglctx.h"

#include "pxr/base/gf/half.h"
#include "pxr/base/gf/vec2i.h"
#include "pxr/base/tf/stringUtils.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/trace/collector.h"
#include "pxr/base/trace/reporter.h"
#include "pxr/base/trace/reporterDataSourceCollector.h"
#include "pxr/base/vt/dictionary.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/cameraUtil/framing.h"
#include "pxr/imaging/hd/driver.h"
#include "pxr/imaging/hd/renderBuffer.h"
#include "pxr/imaging/hd/rendererPluginRegistry.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/timeCode.h"
#include "pxr/usd/usdGeom/basisCurves.h"
#include "pxr/usd/usdGeom/camera.h"
#include "pxr/usdImaging/usdImagingGL/engine.h"
#include "pxr/usdImaging/usdImagingGL/renderParams.h"

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

typedef void (*PFNGLFINISH)();

PFNGLFINISH ResolveGLFinish()
{
    PFNGLFINISH p = (PFNGLFINISH)dlsym(RTLD_DEFAULT, "glFinish");
    if (!p) {
        void* gl = dlopen("libGL.so.1", RTLD_LAZY | RTLD_GLOBAL);
        if (gl) p = (PFNGLFINISH)dlsym(gl, "glFinish");
    }
    if (!p) {
        void* egl = dlopen("libEGL.so.1", RTLD_NOW);
        typedef void* (*PFNGPA)(const char*);
        PFNGPA gpa = egl ? (PFNGPA)dlsym(egl, "eglGetProcAddress") : nullptr;
        if (gpa) p = (PFNGLFINISH)gpa("glFinish");
    }
    return p;
}

double MedianOf(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
double P90Of(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    size_t i = (size_t)std::ceil(0.9 * (v.size() - 1));
    return v[std::min(i, v.size() - 1)];
}
double MinOf(const std::vector<double>& v)
{
    return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
}

std::string Num(double v)
{
    char b[64];
    std::snprintf(b, sizeof(b), "%.6g", v);
    return b;
}

uint64_t Fnv1a(const void* data, size_t n)
{
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

std::string Hex16(uint64_t v)
{
    char b[17];
    std::snprintf(b, sizeof(b), "%016llx", (unsigned long long)v);
    return b;
}

// ---- stats flattening -------------------------------------------------
struct FlatStats {
    std::vector<std::pair<std::string, double>> numeric;  // lowercased keys
};

bool ValueToNumber(const VtValue& v, double* out)
{
    if (v.IsHolding<size_t>())    { *out = (double)v.Get<size_t>(); return true; }
    if (v.IsHolding<long long>()) { *out = (double)v.Get<long long>(); return true; }
    if (v.IsHolding<unsigned>())  { *out = (double)v.Get<unsigned>(); return true; }
    if (v.IsHolding<int>())       { *out = (double)v.Get<int>(); return true; }
    if (v.IsHolding<double>())    { *out = v.Get<double>(); return true; }
    if (v.IsHolding<float>())     { *out = (double)v.Get<float>(); return true; }
    if (v.IsHolding<bool>())      { *out = v.Get<bool>() ? 1 : 0; return true; }
    return false;
}

std::string ToLower(std::string s)
{
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

void Flatten(const VtDictionary& dict, const std::string& prefix,
             FlatStats* out)
{
    for (const auto& kv : dict) {
        std::string key = prefix.empty() ? kv.first : prefix + "/" + kv.first;
        if (kv.second.IsHolding<VtDictionary>()) {
            Flatten(kv.second.Get<VtDictionary>(), key, out);
            continue;
        }
        double d = 0.0;
        if (ValueToNumber(kv.second, &d))
            out->numeric.emplace_back(ToLower(key), d);
    }
}

bool FindCounter(const FlatStats& fs, const char* sub, double* val)
{
    for (const auto& kv : fs.numeric)
        if (kv.first.find(sub) != std::string::npos) {
            *val = kv.second;
            return true;
        }
    return false;
}

// Cumulative byte counters from GetRenderStats (honest across passes).
const char* kByteCounters[] = {
    "primvar", "topology", "gpumemory", "glresourcemem", "cputogpu",
};
const char* kByteNames[] = {
    "primvarBytes", "topologyBytes", "gpuMemoryUsed", "glResourceMem",
    "copyBufferCpuToGpu",
};
const size_t kNumByteCounters =
    sizeof(kByteCounters) / sizeof(kByteCounters[0]);

std::string BaseName(const std::string& path)
{
    size_t s = path.find_last_of('/');
    std::string b = (s == std::string::npos) ? path : path.substr(s + 1);
    size_t d = b.find_last_of('.');
    return d == std::string::npos ? b : b.substr(0, d);
}

std::string HostName()
{
    char b[256] = { 0 };
    return gethostname(b, sizeof(b) - 1) == 0 ? std::string(b) : "unknown";
}

int ThreadCount()
{
    const char* e = std::getenv("PXR_WORK_THREAD_LIMIT");
    if (e && *e) { int n = std::atoi(e); if (n > 0) return n; }
    unsigned n = std::thread::hardware_concurrency();
    return n ? (int)n : 1;
}

// ---- scene + measurement -------------------------------------------------
struct Scene {
    UsdStageRefPtr stage;
    UsdPrim root;
    SdfPath camPath;
    double t0 = 0, t1 = 0;
    bool animated = false;
};

bool OpenScene(const std::string& path, Scene* out)
{
    out->stage = UsdStage::Open(path);
    if (!out->stage) return false;
    // Render the whole stage, not the Groom default prim: the Key/Fill
    // DistantLights are siblings of the groom, and a Groom-rooted render
    // excludes them, so the hair shades exact-black (NUM_LIGHTS == 0) and
    // the checksum verifies a single artifact pixel. The Guides sibling is
    // purpose=guide so it stays out of the render; the Drivers live inside
    // the groom and were always rendered (black, now lit).
    out->root = out->stage->GetPseudoRoot();
    for (const UsdPrim& p : UsdPrimRange(out->stage->GetPseudoRoot())) {
        if (p.IsA<UsdGeomCamera>() && out->camPath.IsEmpty())
            out->camPath = p.GetPath();
    }
    out->t0 = out->stage->GetStartTimeCode();
    out->t1 = out->stage->GetEndTimeCode();
    out->animated = out->t1 > out->t0;
    return true;
}

struct RunResult {
    std::vector<double> submitKept;  // Render() w/o finish, cold rep dropped
    std::vector<double> frameKept;   // Render() + finish
    std::vector<double> submitMedians;
    std::vector<double> frameMedians;
    // Cold repeat 0 timed samples (kept separately, never pooled): for
    // --deform with warmup+frames == span these are all execution-cache
    // MISSES (novel motion), while kept repeats are all HITS (looped
    // motion, tiles rebuilt from cache). Report both.
    std::vector<double> coldSubmit;
    std::vector<double> coldFrame;
    double firstFrameMs = 0.0;  // cold first Render incl. cook+compile
    bool haveFirst = false;
    FlatStats stats;
    std::string checksum;  // color AOV FNV after the last timed frame
};

// Renders warmup+timed frames; `cycleTime` steps params.frame per frame
// (deform). Engine must already have GL current + buffer size set. When
// `wantChecksum`, the color AOV is checksummed after the last timed frame
// (glFinish first so no in-flight GPU work races the Map).
RunResult Measure(UsdImagingGLEngine* engine, const Scene& scene, int frames,
                  int warmup, int repeats, int refineLevel, bool cycleTime,
                  bool wantChecksum, PFNGLFINISH glFinish)
{
    RunResult rr;
    UsdImagingGLRenderParams params;
    params.complexity = 1.0f + 0.5f * (float)refineLevel;
    const int span = scene.animated ? (int)(scene.t1 - scene.t0) + 1 : 1;
    long long frameNo = 0;

    auto renderOne = [&](bool first, bool timed,
                         std::vector<double>* submitOut,
                         std::vector<double>* frameOut) {
        if (cycleTime && scene.animated)
            params.frame = UsdTimeCode(scene.t0 + (double)(frameNo % span));
        ++frameNo;
        auto t = std::chrono::steady_clock::now();
        engine->Render(scene.root, params);
        auto tSubmit = std::chrono::steady_clock::now();
        if (glFinish) glFinish();
        auto tFrame = std::chrono::steady_clock::now();
        double submitMs = std::chrono::duration<double, std::milli>(
            tSubmit - t).count();
        double frameMs = std::chrono::duration<double, std::milli>(
            tFrame - t).count();
        if (first && !rr.haveFirst) {
            rr.firstFrameMs = frameMs;
            rr.haveFirst = true;
        }
        if (timed) {
            submitOut->push_back(submitMs);
            frameOut->push_back(frameMs);
        }
    };

    bool first = true;
    for (int i = 0; i < warmup; ++i) {
        renderOne(first, false, nullptr, nullptr);
        first = false;
    }
    // camera path after population (usdview order)
    if (!scene.camPath.IsEmpty()) engine->SetCameraPath(scene.camPath);

    for (int r = 0; r < std::max(repeats, 1); ++r) {
        for (int i = 0; i < warmup; ++i) {
            renderOne(first, false, nullptr, nullptr);
            first = false;
        }
        std::vector<double> submit, frame;
        submit.reserve(frames);
        frame.reserve(frames);
        for (int i = 0; i < frames; ++i) {
            renderOne(first, true, &submit, &frame);
            first = false;
        }
        if (r > 0) {
            rr.submitKept.insert(rr.submitKept.end(),
                                 submit.begin(), submit.end());
            rr.frameKept.insert(rr.frameKept.end(), frame.begin(), frame.end());
            rr.submitMedians.push_back(MedianOf(submit));
            rr.frameMedians.push_back(MedianOf(frame));
        } else if (repeats <= 1) {  // nothing else kept: salvage single run
            rr.submitKept = submit;
            rr.frameKept = frame;
            rr.submitMedians.push_back(MedianOf(submit));
            rr.frameMedians.push_back(MedianOf(frame));
        } else {  // cold repeat: stash (novel-motion), never pool
            rr.coldSubmit = submit;
            rr.coldFrame = frame;
        }
    }
    Flatten(engine->GetRenderStats(), "", &rr.stats);
    if (std::getenv("USDGEN_BENCH_STATS")) {
        for (const auto& kv : rr.stats.numeric)
            std::fprintf(stderr, "stat: %s = %.6g\n", kv.first.c_str(),
                         kv.second);
    }
    if (wantChecksum) {
        if (glFinish) glFinish();
        HdRenderBuffer* rb = engine->GetAovRenderBuffer(HdAovTokens->color);
        if (rb) {
            if (void* mapped = rb->Map()) {
                size_t px = 0;
                const HdFormat fmt = rb->GetFormat();
                if (fmt == HdFormatFloat16Vec4) px = 8;
                else if (fmt == HdFormatFloat32Vec4) px = 16;
                else if (fmt == HdFormatUNorm8Vec4) px = 4;
                if (px) {
                    rr.checksum = Hex16(Fnv1a(
                        mapped, (size_t)rb->GetWidth() * rb->GetHeight() * px));
                }
                rb->Unmap();
            }
        }
    }
    return rr;
}

// Dumps the current color AOV to a binary PPM (structure/debug only).
bool DumpPpm(UsdImagingGLEngine* engine, const std::string& path,
             PFNGLFINISH glFinish)
{
    if (glFinish) glFinish();
    HdRenderBuffer* rb = engine->GetAovRenderBuffer(HdAovTokens->color);
    if (!rb) return false;
    void* mapped = rb->Map();
    if (!mapped) return false;
    const int W = rb->GetWidth(), H = rb->GetHeight();
    const HdFormat fmt = rb->GetFormat();
    std::vector<uint8_t> rgb((size_t)W * H * 3, 0);
    if (fmt == HdFormatUNorm8Vec4) {
        const uint8_t* s = (const uint8_t*)mapped;
        for (int i = 0; i < W * H; ++i) {
            rgb[3 * i] = s[4 * i]; rgb[3 * i + 1] = s[4 * i + 1];
            rgb[3 * i + 2] = s[4 * i + 2];
        }
    } else if (fmt == HdFormatFloat16Vec4 || fmt == HdFormatFloat32Vec4) {
        const bool half = fmt == HdFormatFloat16Vec4;
        for (int i = 0; i < W * H; ++i) {
            for (int c = 0; c < 3; ++c) {
                float v;
                if (half)
                    v = ((const GfHalf*)mapped)[4 * i + c];
                else
                    v = ((const float*)mapped)[4 * i + c];
                v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                rgb[3 * i + c] = (uint8_t)(v * 255.0f + 0.5f);
            }
        }
    } else {
        rb->Unmap();
        return false;
    }
    rb->Unmap();
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs << "P6\n" << W << " " << H << "\n255\n";
    ofs.write((const char*)rgb.data(), rgb.size());
    return !!ofs;
}

void AppendMs(std::string* j, const char* name,
              const std::vector<double>& kept,
              const std::vector<double>& medians, const char* indent)
{
    *j += std::string(indent) + "\"" + name + "\": ";
    if (kept.empty()) { *j += "null,\n"; return; }
    const std::vector<double>& m = medians.empty() ? kept : medians;
    *j += "{ \"median\": " + Num(MedianOf(m)) +
          ", \"p90\": " + Num(P90Of(kept)) +
          ", \"min\": " + Num(MinOf(kept)) + " },\n";
}

void AppendRun(std::string* j, const char* name, const RunResult& rr,
               const char* indent)
{
    std::string in = std::string(indent) + "  ";
    *j += std::string(indent) + "\"" + name + "\": {\n";
    *j += in + "\"firstFrameMs\": " +
          (rr.haveFirst ? Num(rr.firstFrameMs) : std::string("null")) + ",\n";
    AppendMs(j, "submitMs", rr.submitKept, rr.submitMedians, in.c_str());
    AppendMs(j, "frameMs", rr.frameKept, rr.frameMedians, in.c_str());
    // gpuMs: median-of-medians difference (finish cost above submission).
    double gpu = 0.0;
    if (!rr.submitMedians.empty() && !rr.frameMedians.empty())
        gpu = MedianOf(rr.frameMedians) - MedianOf(rr.submitMedians);
    *j += in + "\"gpuMs\": " + Num(gpu) + ",\n";
    *j += in + "\"checksum\": " +
          (rr.checksum.empty() ? std::string("null")
                               : "\"" + rr.checksum + "\"") + ",\n";
    // Cold-repeat medians (novel motion for --deform; null when repeats<2).
    *j += in + "\"novelSubmitMs\": " +
          (rr.coldSubmit.empty() ? std::string("null")
                                 : Num(MedianOf(rr.coldSubmit))) + ",\n";
    *j += in + "\"novelFrameMs\": " +
          (rr.coldFrame.empty() ? std::string("null")
                                : Num(MedianOf(rr.coldFrame))) + ",\n";
    *j += in + "\"bytes\": {";
    for (size_t i = 0; i < kNumByteCounters; ++i) {
        double v = 0.0;
        bool have = FindCounter(rr.stats, kByteCounters[i], &v);
        *j += (i ? ", " : " ") + std::string("\"") + kByteNames[i] + "\": " +
              (have ? Num(std::llround(v)) : std::string("null"));
    }
    *j += " }\n";
    *j += std::string(indent) + "},\n";
}

}  // namespace

int main(int argc, char** argv)
{
    std::string scenePath, jsonOut = "usdGenShaderRender.json";
    std::string traceOut;  // --trace PATH: chrome trace of the deform run only
    std::string dumpPpm;   // --dump-ppm PATH: last-frame color AOV as PPM
    const char* mode = nullptr;  // null = deform when animated, else static
    // --deform-only: skip the trailing same-invocation static baseline, so a
    // memory-capped box can afford the deform cooks alone (each cook retains
    // ~0.7GB at 10M; base is ~18GB). Run --static separately for statics.
    bool deformOnly = false;
    int w = 1280, h = 720, refineLevel = 2, frames = 6, warmup = 2, repeats = 3;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--scene") scenePath = next("--scene");
        else if (a == "--static") mode = "static";
        else if (a == "--deform") mode = "deform";
        else if (a == "--res") {
            if (std::sscanf(next("--res"), "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0) {
                std::fprintf(stderr, "bad --res (want WxH)\n");
                return 2;
            }
        }
        else if (a == "--refine") refineLevel = std::atoi(next("--refine"));
        else if (a == "--frames") frames = std::atoi(next("--frames"));
        else if (a == "--warmup") warmup = std::atoi(next("--warmup"));
        else if (a == "--repeats") repeats = std::atoi(next("--repeats"));
        else if (a == "--json") jsonOut = next("--json");
        else if (a == "--trace") traceOut = next("--trace");
        else if (a == "--deform-only") deformOnly = true;
        else if (a == "--dump-ppm") dumpPpm = next("--dump-ppm");
        else {
            std::fprintf(stderr,
                "usage: benchUsdGenShaderRender --scene PATH [--static|--deform]"
                " [--res WxH] [--refine N] [--frames N] [--warmup N]"
                " [--repeats N] [--json OUT] [--trace PATH] [--deform-only]"
                " [--dump-ppm PATH]\n");
            return 2;
        }
    }
    if (scenePath.empty()) return 2;
    if (refineLevel < 0 || refineLevel > 2) refineLevel = 2;

    {
        if (!HdRendererPluginRegistry::GetInstance().GetOrCreateRendererPlugin(
                TfToken("HdStormRendererPlugin"))) {
            std::fprintf(stderr, "FATAL: HdStormRendererPlugin not registered\n");
            return 1;
        }
    }

    if (!eglctx::MakeHeadlessGLContext()) {
        std::fprintf(stderr, "FATAL: no headless EGL device context available\n");
        return 1;
    }
    PFNGLFINISH glFinish = ResolveGLFinish();
    if (!glFinish)
        std::fprintf(stderr, "WARN: glFinish unresolved; timings CPU-side only\n");

    Scene sc;
    if (!OpenScene(scenePath, &sc)) {
        std::fprintf(stderr, "FATAL: cannot open %s\n", scenePath.c_str());
        return 1;
    }
    const bool deform = mode ? std::strcmp(mode, "deform") == 0
                             : sc.animated;

    UsdImagingGLEngine engine(HdDriver(), TfToken("HdStormRendererPlugin"), true);
    engine.SetEnablePresentation(false);
    // Framing is mandatory: without it the engine renders an empty data
    // window (all-clear image) even with a valid camera and populated stage.
    engine.SetFraming(CameraUtilFraming(GfRect2i(GfVec2i(0), w, h)));
    engine.SetRenderBufferSize(GfVec2i(w, h));
    engine.SetRendererSetting(TfToken("collectStats"), VtValue(true));
    engine.SetRendererAov(HdAovTokens->color);

    std::string json;
    json += "{\n";
    json += "  \"scene\": \"" + BaseName(scenePath) + "\",\n";
    json += "  \"host\": \"" + HostName() + "\", \"resolution\": [" + Num(w) +
            ", " + Num(h) + "],\n";
    json += "  \"refineLevel\": " + Num(refineLevel) +
            ", \"threads\": " + Num(ThreadCount()) + ",\n";
    json += "  \"frames\": " + Num(frames) + ", \"warmup\": " + Num(warmup) +
            ", \"repeats\": " + Num(repeats) + ",\n";

    if (deform && sc.animated) {
        TraceReporterRefPtr reporter;
        if (!traceOut.empty()) {
            reporter = TraceReporter::New(
                "benchUsdGenShaderRender",
                TraceReporterDataSourceCollector::New());
            TraceCollector::GetInstance().SetEnabled(true);
        }
        RunResult rrDeform = Measure(&engine, sc, frames, warmup, repeats,
                                     refineLevel, true, true, glFinish);
        if (reporter) {
            // Trace timings taint this run's medians; use --trace runs for
            // structure only, never for numbers.
            TraceCollector::GetInstance().SetEnabled(false);
            TraceCollector::GetInstance().CreateCollection();
            reporter->UpdateTraceTrees();
            std::ofstream tos(traceOut);
            if (tos) reporter->ReportChromeTracing(tos);
            else std::fprintf(stderr, "WARN: cannot write %s\n",
                              traceOut.c_str());
        }
        AppendRun(&json, "deform", rrDeform, "  ");
        double dMed = rrDeform.frameMedians.empty() ? 0.0
            : MedianOf(rrDeform.frameMedians);
        double dSub = rrDeform.submitMedians.empty() ? 0.0
            : MedianOf(rrDeform.submitMedians);
        double nMed = rrDeform.coldFrame.empty() ? 0.0
            : MedianOf(rrDeform.coldFrame);
        if (deformOnly) {
            std::printf("deform novel %s ms cached %s ms (submit %s gpu %s) "
                        "checksum %s\n",
                        Num(nMed).c_str(), Num(dMed).c_str(),
                        Num(dSub).c_str(), Num(dMed - dSub).c_str(),
                        rrDeform.checksum.empty() ? "-"
                                                 : rrDeform.checksum.c_str());
            json += "  \"deformOnly\": true,\n";
        } else {
        // Static baseline in the same invocation (same-process comparison).
        RunResult rrStatic = Measure(&engine, sc, frames, warmup, repeats,
                                     refineLevel, false, true, glFinish);
        AppendRun(&json, "static", rrStatic, "  ");
        double sMed = rrStatic.frameMedians.empty() ? 0.0
            : MedianOf(rrStatic.frameMedians);
        double sSub = rrStatic.submitMedians.empty() ? 0.0
            : MedianOf(rrStatic.submitMedians);
        json += "  \"deformDeltaFrameMs\": " + Num(dMed - sMed) + ",\n";
        json += "  \"deformDeltaSubmitMs\": " + Num(dSub - sSub) + ",\n";
        std::printf("deform novel %s ms cached %s ms (submit %s gpu %s) "
                    "static %s ms delta %s ms checksum %s/%s\n",
                    Num(nMed).c_str(), Num(dMed).c_str(), Num(dSub).c_str(),
                    Num(dMed - dSub).c_str(), Num(sMed).c_str(),
                    Num(dMed - sMed).c_str(),
                    rrDeform.checksum.empty() ? "-" : rrDeform.checksum.c_str(),
                    rrStatic.checksum.empty() ? "-" : rrStatic.checksum.c_str());
        }
    } else {
        if (deform && !sc.animated)
            std::fprintf(stderr, "NOTE: --deform on a static scene; "
                         "measuring static\n");
        RunResult rrStatic = Measure(&engine, sc, frames, warmup, repeats,
                                     refineLevel, false, true, glFinish);
        AppendRun(&json, "static", rrStatic, "  ");
        double sMed = rrStatic.frameMedians.empty() ? 0.0
            : MedianOf(rrStatic.frameMedians);
        double sSub = rrStatic.submitMedians.empty() ? 0.0
            : MedianOf(rrStatic.submitMedians);
        std::printf("static frame %s ms (submit %s gpu %s) checksum %s\n",
                    Num(sMed).c_str(), Num(sSub).c_str(),
                    Num(sMed - sSub).c_str(),
                    rrStatic.checksum.empty() ? "-" : rrStatic.checksum.c_str());
    }

    if (!dumpPpm.empty()) {
        if (DumpPpm(&engine, dumpPpm, glFinish))
            std::printf("dumped %s\n", dumpPpm.c_str());
        else
            std::fprintf(stderr, "WARN: cannot dump %s\n", dumpPpm.c_str());
    }

    while (!json.empty() && (json.back() == '\n' || json.back() == ',' ||
                             json.back() == ' '))
        json.pop_back();
    json += "\n}\n";

    std::ofstream ofs(jsonOut);
    if (!ofs) {
        std::fprintf(stderr, "FATAL: cannot write %s\n", jsonOut.c_str());
        return 1;
    }
    ofs << json;
    ofs.close();
    return 0;
}
