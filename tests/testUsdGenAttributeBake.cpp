// Painted attribute maps: the synchronous .ptx bake and the asynchronous
// bake-to-stage worker.
//
//   * Sync validation fails closed (null map, face-count mismatch, non-quad
//     mesh, bad indices, empty out path) and writes nothing.
//   * Sync round-trip: a baked map reopens through UsdGenPtexTexture with
//     identical texels (1- and 3-channel), and the shared quad edge blends
//     across faces, which proves the adjacency was written.
//   * The worker never bakes on the calling thread: with the worker held,
//     Enqueue stores without writing, strokes still preview from the base,
//     and release-when-resumed bakes only the latest version (coalescing).
//   * TakeCompleted/NoteSwapped retire older versioned files; no .tmp files
//     survive a bake; a failed bake reports a diagnostic, not a completion.
//   * Stress: producers on many threads, a UI thread taking completions and
//     diagnostics, and a pausing thread race the lock-free lane; every
//     request is baked, dropped or diagnosed exactly once.
#include "usdGen/maps/attributeBake.h"
#include "usdGen/maps/attributeMap.h"
#include "usdGen/maps/ptexMap.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using namespace usdGen;
namespace fs = std::filesystem;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what, int line)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL (line %d): %s\n", line, what.c_str());
    }
}
#define CHECK(cond, what) Check(static_cast<bool>(cond), what, __LINE__)

bool Near(double a, double b, double tolerance) { return std::fabs(a - b) <= tolerance; }

// Two quads sharing the edge (1, 4): face0 = [0,1,4,3], face1 = [1,2,5,4].
// Face0's u=1 edge meets face1's u=0 edge.
std::vector<int> QuadCounts() { return {4, 4}; }
std::vector<int> QuadIndices() { return {0, 1, 4, 3, 1, 2, 5, 4}; }

std::shared_ptr<UsdGenAttributeMap> TwoFaceMap(int res = 8, int channels = 1)
{
    UsdGenAttributeMapSpec spec;
    spec.numFaces = 2;
    spec.resolution = res;
    spec.channels = channels;
    std::string error;
    return UsdGenAttributeMap::Create(spec, &error);
}

fs::path TestDir()
{
    auto const nanos =
        std::chrono::steady_clock::now().time_since_epoch().count();
    size_t const tid =
        std::hash<std::thread::id>{}(std::this_thread::get_id());
    fs::path dir =
        fs::temp_directory_path() /
        ("usdGenAttrBake." + std::to_string(size_t(nanos) ^ (tid * 0x9e3779b9)));
    std::error_code ec;
    fs::remove_all(dir, ec);
    return dir;
}

UsdGenAttributeBakeInput BakeInput(std::shared_ptr<const UsdGenAttributeMap> map,
                                   fs::path const &dir)
{
    UsdGenAttributeBakeInput input;
    input.map = std::move(map);
    input.faceVertexCounts = QuadCounts();
    input.faceVertexIndices = QuadIndices();
    input.outDir = dir.string();
    return input;
}

void CheckFileName()
{
    CHECK(UsdGenAttributeBakeFileName("paintMap", 7) == "paintMap.v7.ptx",
          "versioned filename");
    CHECK(UsdGenAttributeBakeFileName("density", 0) == "density.v0.ptx",
          "version 0 filename");
}

void CheckValidation()
{
    fs::path const dir = TestDir();
    fs::path const out = dir / "invalid.ptx";
    UsdGenAttributeBakeStats stats;
    std::string error;

    UsdGenAttributeBakeInput nullMap = BakeInput(TwoFaceMap(), dir);
    nullMap.map.reset();
    CHECK(!UsdGenAttributeBakePtex(nullMap, out.string(), &stats, &error) &&
              !error.empty(),
          "null map rejected");

    UsdGenAttributeBakeInput shortCounts = BakeInput(TwoFaceMap(), dir);
    shortCounts.faceVertexCounts = {4};
    CHECK(!UsdGenAttributeBakePtex(shortCounts, out.string(), &stats, &error),
          "face-count mismatch rejected");

    UsdGenAttributeBakeInput ngon = BakeInput(TwoFaceMap(), dir);
    ngon.faceVertexCounts = {4, 3};
    CHECK(!UsdGenAttributeBakePtex(ngon, out.string(), &stats, &error),
          "non-quad mesh rejected");

    UsdGenAttributeBakeInput shortIndices = BakeInput(TwoFaceMap(), dir);
    shortIndices.faceVertexIndices = {0, 1, 2, 3};
    CHECK(!UsdGenAttributeBakePtex(shortIndices, out.string(), &stats, &error),
          "index-count mismatch rejected");

    UsdGenAttributeBakeInput negative = BakeInput(TwoFaceMap(), dir);
    negative.faceVertexIndices[0] = -1;
    CHECK(!UsdGenAttributeBakePtex(negative, out.string(), &stats, &error),
          "negative index rejected");

    CHECK(!UsdGenAttributeBakePtex(BakeInput(TwoFaceMap(), dir), "", &stats, &error),
          "empty out path rejected");
    CHECK(!UsdGenAttributeBakePtex(BakeInput(TwoFaceMap(), dir), "", &stats, nullptr),
          "null error sink still fails");
    CHECK(!fs::exists(out), "a rejected bake writes nothing");

    std::error_code ec;
    fs::remove_all(dir, ec);
}

void CheckRoundTrip(int channels)
{
    fs::path const dir = TestDir();
    fs::path const out = dir / "roundtrip.ptx";
    auto map = TwoFaceMap(8, channels);
    // Face 0: a per-texel ramp; face 1: uniform per channel.
    for (int t = 0; t < 8; ++t) {
        for (int s = 0; s < 8; ++s) {
            for (int c = 0; c < channels; ++c) {
                map->SetTexel(0, s, t, c, float(s + t * 8) / 64.0f + 0.1f * c);
                map->SetTexel(1, s, t, c, 0.5f + 0.1f * c);
            }
        }
    }
    UsdGenAttributeBakeStats stats;
    std::string error;
    CHECK(UsdGenAttributeBakePtex(BakeInput(map, dir), out.string(), &stats, &error),
          "bake writes");
    CHECK(stats.faces == 2 && stats.texels == 128 && stats.channels == channels,
          "bake stats");
    CHECK(fs::exists(out), "bake output exists");
    CHECK(!fs::exists(fs::path(out.string() + ".tmp")), "no .tmp survives a bake");

    UsdGenPtexMapOptions options;
    options.filter = "nearest";
    options.channelCount = channels;
    auto texture = UsdGenPtexTexture::Open(out.string(), options, &error);
    CHECK(texture && error.empty(), "baked file reopens");
    if (!texture) return;
    CHECK(texture->NumFaces() == 2, "ptex face per quad");
    CHECK(texture->SampleChannels() == channels, "channel window");
    auto sampler = texture->MakeSampler();
    CHECK(sampler != nullptr, "sampler builds");
    bool exact = true;
    std::vector<float> got;
    got.resize(size_t(channels));
    for (int f = 0; f < 2 && exact; ++f) {
        for (int t = 0; t < 8 && exact; ++t) {
            for (int s = 0; s < 8 && exact; ++s) {
                if (!sampler->Sample(f, (float(s) + 0.5f) / 8.0f,
                                     (float(t) + 0.5f) / 8.0f, got.data())) {
                    exact = false;
                    break;
                }
                for (int c = 0; c < channels; ++c) {
                    float want = 0.0f;
                    map->GetTexel(f, s, t, c, &want);
                    if (!Near(got[size_t(c)], want, 1e-6)) exact = false;
                }
            }
        }
    }
    CHECK(exact, channels == 1 ? "1-channel texels round-trip exactly"
                               : "3-channel texels round-trip exactly");

    std::error_code ec;
    fs::remove_all(dir, ec);
}

void CheckAdjacencyBlends()
{
    // Uniform faces, 0.0 and 1.0: a bilinear sample just inside face 0's
    // shared edge can only lift off 0.0 by reading face 1 through the
    // adjacency the bake wrote.
    fs::path const dir = TestDir();
    fs::path const out = dir / "adjacent.ptx";
    auto map = TwoFaceMap(8, 1);
    for (int t = 0; t < 8; ++t)
        for (int s = 0; s < 8; ++s) map->SetTexel(1, s, t, 0, 1.0f);
    std::string error;
    CHECK(UsdGenAttributeBakePtex(BakeInput(map, dir), out.string(), nullptr, &error),
          "uniform bake writes");
    UsdGenPtexMapOptions options;
    options.filter = "bilinear";
    auto texture = UsdGenPtexTexture::Open(out.string(), options, &error);
    CHECK(texture != nullptr, "uniform bake reopens");
    if (!texture) return;
    auto sampler = texture->MakeSampler();
    float centre = -1.0f, edge = -1.0f;
    CHECK(sampler->Sample(0, 0.5f, 0.5f, &centre) && Near(centre, 0.0, 1e-6),
          "face centre reads the face value");
    CHECK(sampler->Sample(0, 0.97f, 0.5f, &edge), "edge sample reads");
    CHECK(edge > 0.05 && edge < 0.95,
          "the shared edge blends across faces (adjacency written)");

    std::error_code ec;
    fs::remove_all(dir, ec);
}

void CheckWorker()
{
    fs::path const dir = TestDir();
    UsdGenAttributeBakeWorker worker;
    worker.PauseWorker(true);

    auto mapFor = [&](float value) {
        auto map = TwoFaceMap(8, 1);
        map->Fill(value);
        return map;
    };
    // Three releases before the worker wakes: nothing may bake on the
    // calling thread, and the stroke that owns the base still previews.
    UsdGenBrushStroke stroke(mapFor(0.0f));
    UsdGenBrushDab dab;
    dab.face = 0;
    dab.radius = 1.0f;
    dab.strength = 1.0f;
    dab.value = 1.0f;
    dab.falloff = UsdGenBrushFalloff::Constant;
    std::string error;
    CHECK(stroke.AddDab(dab, &error), "stroke records while bakes pend");
    worker.Enqueue(1, BakeInput(mapFor(0.1f), dir));
    worker.Enqueue(2, BakeInput(mapFor(0.2f), dir));
    worker.Enqueue(3, BakeInput(stroke.Preview(), dir));
    CHECK(worker.PendingVersion() == 3, "pending tracks the latest enqueue");
    CHECK(worker.BakeCount() == 0, "Enqueue bakes nothing on the caller");
    CHECK(!fs::exists(dir / UsdGenAttributeBakeFileName("paintMap", 1)) &&
              !fs::exists(dir / UsdGenAttributeBakeFileName("paintMap", 2)) &&
              !fs::exists(dir / UsdGenAttributeBakeFileName("paintMap", 3)),
          "no file appears while the worker is held");
    float painted = 0.0f;
    CHECK(stroke.Preview()->GetTexel(0, 0, 0, 0, &painted) && painted == 1.0f,
          "the stroke previews from its base while bakes pend");

    worker.PauseWorker(false);
    CHECK(worker.WaitCompleted(3), "the latest version completes");
    CHECK(worker.BakeCount() == 1, "coalesced releases bake once");
    CHECK(worker.DropCount() == 2, "superseded inputs are dropped, never baked");
    uint64_t version = 0;
    std::string path;
    CHECK(worker.TakeCompleted(&version, &path) && version == 3,
          "TakeCompleted reports the latest version");
    CHECK(path == (dir / UsdGenAttributeBakeFileName("paintMap", 3)).string(),
          "TakeCompleted reports the versioned path");
    CHECK(!worker.TakeCompleted(&version, &path), "completion drains once");
    CHECK(!fs::exists(dir / UsdGenAttributeBakeFileName("paintMap", 1)) &&
              !fs::exists(dir / UsdGenAttributeBakeFileName("paintMap", 2)),
          "superseded versions never reach disk");
    // The baked file carries the stroke's preview, not the base.
    {
        UsdGenPtexMapOptions options;
        options.filter = "nearest";
        auto texture = UsdGenPtexTexture::Open(path, options, &error);
        CHECK(texture != nullptr, "coalesced bake reopens");
        if (texture) {
            auto sampler = texture->MakeSampler();
            float v = 0.0f;
            CHECK(sampler->Sample(0, 0.0625f, 0.0625f, &v) && Near(v, 1.0, 1e-6),
                  "the coalesced bake carries the stroke preview");
        }
    }
    // The sweep below deletes the retired file; on Windows that fails while
    // the Ptex cache holds it open, so release it first (as the app would
    // when it drops the old map revision).
    UsdGenPtexTexture::PurgeCache();

    // A second bake, then the stage swap retires the first file.
    worker.Enqueue(4, BakeInput(mapFor(0.4f), dir));
    CHECK(worker.WaitCompleted(4), "the next version completes");
    CHECK(worker.TakeCompleted(&version, &path) && version == 4, "v4 completes");
    size_t const sweeps = worker.SweepCount();
    worker.NoteSwapped(4);
    bool swept = false;
    for (int i = 0; i < 200 && !swept; ++i) {
        swept = worker.SweepCount() > sweeps &&
                !fs::exists(dir / UsdGenAttributeBakeFileName("paintMap", 3));
        if (!swept) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(swept, "NoteSwapped sweeps the retired version");
    CHECK(fs::exists(dir / UsdGenAttributeBakeFileName("paintMap", 4)),
          "the swapped version stays");

    // A failed bake reports a diagnostic, not a completion.
    UsdGenAttributeBakeInput bad = BakeInput(mapFor(0.0f), dir);
    bad.map.reset();
    worker.Enqueue(5, bad);
    std::string diagnostic;
    for (int i = 0; i < 200 && diagnostic.empty(); ++i) {
        diagnostic = worker.TakeDiagnostic();
        if (diagnostic.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(!diagnostic.empty(), "a failed bake leaves a diagnostic");
    CHECK(!worker.WaitCompleted(5, 200), "a failed bake never completes");

    // An empty out directory fails closed instead of writing into the
    // process working directory.
    UsdGenAttributeBakeInput noDir = BakeInput(mapFor(0.0f), dir);
    noDir.outDir.clear();
    worker.Enqueue(6, noDir);
    diagnostic.clear();
    for (int i = 0; i < 200 && diagnostic.empty(); ++i) {
        diagnostic = worker.TakeDiagnostic();
        if (diagnostic.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(!diagnostic.empty(), "an empty out directory is diagnosed");
    CHECK(!worker.WaitCompleted(6, 200), "an empty out directory never completes");
    CHECK(!fs::exists(UsdGenAttributeBakeFileName("paintMap", 6)),
          "nothing lands in the working directory");

    bool tmpLeft = false;
    for (auto const &entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".tmp") tmpLeft = true;
    }
    CHECK(!tmpLeft, "no .tmp files survive the worker");

    worker.PauseWorker(false);  // the destructor joins; never hold it paused
    std::error_code ec;
    fs::remove_all(dir, ec);
}

void CheckWorkerStress()
{
    fs::path const dir = TestDir();
    constexpr int kProducers = 8;
    constexpr int kPerProducer = 40;
    std::atomic<uint64_t> nextVersion{1};
    std::atomic<int> noDirSent{0};
    std::atomic<bool> producing{true};
    std::atomic<int> badCompletion{0};
    std::atomic<int> diagnosed{0};
    {
        UsdGenAttributeBakeWorker worker;
        std::thread ui([&] {
            uint64_t last = 0;
            auto drain = [&] {
                uint64_t version = 0;
                std::string path;
                while (worker.TakeCompleted(&version, &path)) {
                    if (path != (dir / UsdGenAttributeBakeFileName("paintMap", version)).string())
                        badCompletion.fetch_add(1);
                    if (version > last) last = version;
                    worker.NoteSwapped(last);
                }
                std::string const diag = worker.TakeDiagnostic();
                for (size_t at = diag.find("needs an out directory"); at != std::string::npos;
                     at = diag.find("needs an out directory", at + 1))
                    diagnosed.fetch_add(1);
            };
            while (producing.load()) {
                drain();
                std::this_thread::yield();
            }
            drain();
        });
        std::thread pauser([&] {
            for (int i = 0; i < 50 && producing.load(); ++i) {
                worker.PauseWorker(true);
                std::this_thread::yield();
                worker.PauseWorker(false);
            }
        });
        std::vector<std::thread> producers;
        for (int p = 0; p < kProducers; ++p) {
            producers.emplace_back([&, p] {
                for (int i = 0; i < kPerProducer; ++i) {
                    auto map = TwoFaceMap(4, 1);
                    map->Fill(float(p) * 0.1f);
                    UsdGenAttributeBakeInput input = BakeInput(map, dir);
                    if ((i + p) % 8 == 0) {
                        input.outDir.clear();
                        noDirSent.fetch_add(1);
                    }
                    worker.Enqueue(nextVersion.fetch_add(1), std::move(input));
                }
            });
        }
        for (std::thread &t : producers) t.join();
        pauser.join();
        worker.PauseWorker(false);
        // A last request newer than all of them: once it completes, the lane
        // has consumed everything before it.
        uint64_t const sentinel = nextVersion.fetch_add(1);
        auto map = TwoFaceMap(4, 1);
        worker.Enqueue(sentinel, BakeInput(map, dir));
        CHECK(worker.WaitCompleted(sentinel, 60000), "stress: the sentinel completes");
        producing.store(false);
        ui.join();
        size_t const requests = size_t(kProducers * kPerProducer + 1);
        size_t const bakes = worker.BakeCount(), drops = worker.DropCount();
        // An empty-directory request is either dropped or diagnosed.
        CHECK(bakes + drops + size_t(diagnosed.load()) == requests,
              "stress: every request is baked, dropped or diagnosed once (" +
                  std::to_string(bakes) + " + " + std::to_string(drops) + " + " +
                  std::to_string(diagnosed.load()) + " of " + std::to_string(requests) + ")");
        CHECK(diagnosed.load() <= noDirSent.load(), "stress: no diagnostic without a cause");
        CHECK(badCompletion.load() == 0, "stress: completions name their versioned file");
        CHECK(worker.CompletedVersion() == sentinel, "stress: the newest request completes last");
    }
    UsdGenPtexTexture::PurgeCache();
    std::error_code ec;
    fs::remove_all(dir, ec);
}

}  // namespace

int main()
{
    CheckFileName();
    CheckValidation();
    CheckRoundTrip(1);
    CheckRoundTrip(3);
    CheckAdjacencyBlends();
    CheckWorker();
    CheckWorkerStress();

    // The round-trip opens share the process Ptex cache by path; a bake the
    // worker overwrote under an observed path would read stale without this.
    // (Each test uses a fresh directory, so this is hygiene, not load-bearing.)
    UsdGenPtexTexture::PurgeCache();

    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("testUsdGenAttributeBake: OK\n");
    return 0;
}
