// usdGenTracePlayback — plays a usdGen scene through the groom scene index
// frame by frame, the way usdview does without --allow-async (every frame's
// dirty notice captures, cooks and publishes before the frame returns), and
// reports where the time goes.
//
//   usdGenTracePlayback <scene.usda> [--frames first:last] [--loops N]
//       [--chrome trace.json] [--report report.txt] [--pull] [--quiet]
//
//   --frames  default: the stage's start/end time codes
//   --loops   play the range N times (default 2): the second pass shows what
//             the execution cache and incremental compile save
//   --chrome  write a Chrome trace (chrome://tracing, ui.perfetto.dev)
//   --report  write OpenUSD's aggregate trace report
//   --pull    read every published tile back as a renderer sync would, and
//             hash the points (the table's last column) so two builds can be
//             compared for identical output
//   --quiet   no per-cook / per-operator lines (TF_DEBUG USDGEN_*) and no
//             per-frame tally of the notices sent to Hydra
//
// bin/trace_playback.ps1 sets up the plugin environment and runs this.
#include "usdGen/registries/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGenHydraCurves/register.h"

#include "pxr/base/tf/debug.h"
#include "pxr/base/trace/collector.h"
#include "pxr/base/trace/reporter.h"
#include "pxr/base/trace/trace.h"
#include "pxr/base/vt/types.h"
#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/primvarsSchema.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"
#include "pxr/imaging/hdsi/sceneGlobalsSceneIndex.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

using Clock = std::chrono::steady_clock;

double Ms(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Pulled {
    size_t tiles = 0, points = 0;
    uint64_t pointsHash = 1469598103934665603ull;   // FNV-1a over every tile's points
};

/// Reads every primvar value and the topology of every published tile, the
/// data a renderer's sync would fetch.
Pulled Pull(HdSceneIndexBaseRefPtr const &index, SdfPathVector const &renderScopes)
{
    TRACE_FUNCTION();
    Pulled out;
    for (SdfPath const &render : renderScopes) {
        for (SdfPath const &child : index->GetChildPrimPaths(render)) {
            HdSceneIndexPrim const prim = index->GetPrim(child);
            if (prim.primType != TfToken("basisCurves")) continue;
            ++out.tiles;
            HdPrimvarsSchema const primvars = HdPrimvarsSchema::GetFromParent(prim.dataSource);
            for (TfToken const &name : primvars.GetPrimvarNames()) {
                HdSampledDataSourceHandle const value = primvars.GetPrimvar(name).GetPrimvarValue();
                if (!value) continue;
                VtValue const v = value->GetValue(0.0f);
                if (name != TfToken("points")) continue;
                out.points += v.GetArraySize();
                if (v.IsHolding<VtVec3fArray>()) {
                    VtVec3fArray const &p = v.UncheckedGet<VtVec3fArray>();
                    auto const *bytes = static_cast<unsigned char const *>(
                        static_cast<void const *>(p.cdata()));
                    for (size_t i = 0; i < p.size() * sizeof(GfVec3f); ++i)
                        out.pointsHash = (out.pointsHash ^ bytes[i]) * 1099511628211ull;
                }
            }
            if (auto counts = HdSampledDataSource::Cast(HdContainerDataSource::Get(
                    prim.dataSource, HdDataSourceLocator(TfToken("basisCurves"),
                                                         TfToken("topology"),
                                                         TfToken("curveVertexCounts")))))
                counts->GetValue(0.0f);
        }
    }
    return out;
}

/// Tallies the notices the groom index sends Hydra: how many prims each
/// dirtied locator reaches, which tells whether a frame re-syncs only the
/// primvars that moved.
class NoticeTally : public HdSceneIndexObserver
{
public:
    void PrimsAdded(HdSceneIndexBase const &, AddedPrimEntries const &entries) override
    {
        added += entries.size();
    }
    void PrimsRemoved(HdSceneIndexBase const &, RemovedPrimEntries const &entries) override
    {
        removed += entries.size();
    }
    void PrimsDirtied(HdSceneIndexBase const &, DirtiedPrimEntries const &entries) override
    {
        for (DirtiedPrimEntry const &entry : entries)
            for (HdDataSourceLocator const &locator : entry.dirtyLocators)
                ++dirtied[locator.IsEmpty() ? "<universal " + entry.primPath.GetString() + ">"
                                            : locator.GetString()];
    }
    void PrimsRenamed(HdSceneIndexBase const &, RenamedPrimEntries const &) override {}

    void PrintAndClear()
    {
        std::printf("  hydra notices: %zu added, %zu removed, dirtied:", added, removed);
        if (dirtied.empty()) std::printf(" none");
        for (auto const &entry : dirtied)
            std::printf(" %s x%zu", entry.first.c_str(), entry.second);
        std::printf("\n");
        added = removed = 0;
        dirtied.clear();
    }

private:
    size_t added = 0, removed = 0;
    std::map<std::string, size_t> dirtied;
};

void Usage()
{
    std::fprintf(stderr,
        "usage: usdGenTracePlayback <scene.usda> [--frames first:last] [--loops N]\n"
        "           [--chrome trace.json] [--report report.txt] [--pull] [--quiet]\n");
}

}  // namespace

int main(int argc, char **argv)
{
    // The curve publisher registers before the first compile.
    (void)usdGenHydra::UsdGenHydraCurvesEnsureRegistered();
    std::string scene, chrome, report;
    double first = 0.0, last = -1.0;
    int loops = 2;
    bool pull = false, quiet = false, framesGiven = false;
    for (int i = 1; i < argc; ++i) {
        std::string const arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { Usage(); std::exit(2); }
            return argv[++i];
        };
        if (arg == "--frames") {
            std::string const range = next();
            size_t const colon = range.find(':');
            first = std::atof(range.substr(0, colon).c_str());
            last = colon == std::string::npos ? first : std::atof(range.substr(colon + 1).c_str());
            framesGiven = true;
        } else if (arg == "--loops") loops = std::max(1, std::atoi(next().c_str()));
        else if (arg == "--chrome") chrome = next();
        else if (arg == "--report") report = next();
        else if (arg == "--pull") pull = true;
        else if (arg == "--quiet") quiet = true;
        else if (!arg.empty() && arg[0] == '-') { Usage(); return 2; }
        else scene = arg;
    }
    if (scene.empty()) { Usage(); return 2; }

    if (!quiet) TfDebug::SetDebugSymbolsByName("USDGEN_*", true);

    auto const openStart = Clock::now();
    UsdStageRefPtr const stage = UsdStage::Open(scene, UsdStage::LoadAll);
    if (!stage) { std::fprintf(stderr, "cannot open %s\n", scene.c_str()); return 1; }
    if (!framesGiven) {
        first = stage->GetStartTimeCode();
        last = stage->GetEndTimeCode();
    }
    if (last < first) last = first;
    SdfPathVector renderScopes;
    for (UsdPrim const &prim : stage->Traverse())
        if (prim.GetTypeName() == TfToken("UsdGenDescription"))
            renderScopes.push_back(prim.GetPath().AppendChild(TfToken("__usdGenRender")));
    std::printf("usdGenTracePlayback: %s, frames %g-%g, %d loop(s), %zu description(s)\n",
                scene.c_str(), first, last, loops, renderScopes.size());

    TraceCollector::GetInstance().SetEnabled(true);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    UsdImagingSceneIndices indices = UsdImagingCreateSceneIndices(info);
    // UsdImagingGLEngine's order: the app scene indices (scene globals among
    // them) are inserted before the scene index plugins, so the groom index
    // sees the current frame; each frame sets the stage time first and the
    // scene globals' current frame second.
    HdsiSceneGlobalsSceneIndexRefPtr const globals =
        HdsiSceneGlobalsSceneIndex::New(indices.finalSceneIndex);
    indices.stageSceneIndex->SetTime(UsdTimeCode(first));
    globals->SetCurrentFrame(first);
    HdSceneIndexBaseRefPtr groom;
    UsdGenGroomSceneIndex *owner = nullptr;
    {
        TRACE_SCOPE("usdGen playback: populate");
        groom = UsdGenGroomSceneIndex::New(globals);
        owner = dynamic_cast<UsdGenGroomSceneIndex *>(groom.operator->());
        if (!owner) { std::fprintf(stderr, "no groom scene index\n"); return 1; }
        owner->Synchronize();
        if (pull) Pull(groom, renderScopes);
    }
    std::printf("populated in %.1f ms\n\n", Ms(openStart, Clock::now()));
    NoticeTally tally;
    if (!quiet) groom->AddObserver(HdSceneIndexObserverPtr(&tally));

    struct Row { int loop; double frame, setTime, setFrame, sync, pull, total; Pulled data; };
    std::vector<Row> rows;
    for (int loop = 0; loop < loops; ++loop) {
        for (double frame = first; frame <= last + 1e-9; frame += 1.0) {
            TRACE_SCOPE_DYNAMIC("usdGen playback: loop " + std::to_string(loop + 1) +
                                " frame " + std::to_string(int(frame)));
            if (!quiet) std::printf("---- loop %d frame %g\n", loop + 1, frame);
            Row row{loop + 1, frame, 0, 0, 0, 0, 0, {}};
            auto const t0 = Clock::now();
            {
                TRACE_SCOPE("usdGen playback: SetTime (notices, capture, cook, publish)");
                indices.stageSceneIndex->SetTime(UsdTimeCode(frame));
            }
            auto const t1 = Clock::now();
            {
                TRACE_SCOPE("usdGen playback: SetCurrentFrame (scene globals)");
                globals->SetCurrentFrame(frame);
                indices.stageSceneIndex->ApplyPendingUpdates();
            }
            auto const t2 = Clock::now();
            {
                TRACE_SCOPE("usdGen playback: Synchronize");
                owner->Synchronize();
            }
            auto const t3 = Clock::now();
            if (pull) row.data = Pull(groom, renderScopes);
            auto const t4 = Clock::now();
            if (!quiet) tally.PrintAndClear();
            row.setTime = Ms(t0, t1);
            row.setFrame = Ms(t1, t2);
            row.sync = Ms(t2, t3);
            row.pull = Ms(t3, t4);
            row.total = Ms(t0, t4);
            rows.push_back(row);
        }
    }
    TraceCollector::GetInstance().SetEnabled(false);

    std::printf("\n loop  frame   SetTime  SetFrame      Sync      Pull     Total   tiles    points"
                "  points hash\n");
    for (Row const &r : rows)
        std::printf("%5d %6g %9.2f %9.2f %9.2f %9.2f %9.2f %7zu %9zu  %016llx\n", r.loop, r.frame,
                    r.setTime, r.setFrame, r.sync, r.pull, r.total, r.data.tiles, r.data.points,
                    pull ? (unsigned long long)r.data.pointsHash : 0ull);
    for (int loop = 1; loop <= loops; ++loop) {
        double sum = 0.0, worst = 0.0;
        size_t n = 0;
        for (Row const &r : rows)
            if (r.loop == loop) { sum += r.total; worst = std::max(worst, r.total); ++n; }
        if (n)
            std::printf("loop %d: mean %.2f ms/frame (%.1f fps), worst %.2f ms\n", loop,
                        sum / double(n), sum > 0.0 ? 1000.0 * double(n) / sum : 0.0, worst);
    }

    TraceReporterPtr const reporter = TraceReporter::GetGlobalReporter();
    reporter->UpdateTraceTrees();
    if (!chrome.empty()) {
        std::ofstream out(chrome);
        reporter->ReportChromeTracing(out);
        std::printf("chrome trace: %s\n", chrome.c_str());
    }
    if (!report.empty()) {
        std::ofstream out(report);
        reporter->Report(out);
        std::printf("aggregate report: %s\n", report.c_str());
    }
    groom.Reset();
    UsdGenGroomSceneIndex::DrainRetired();
    return 0;
}
