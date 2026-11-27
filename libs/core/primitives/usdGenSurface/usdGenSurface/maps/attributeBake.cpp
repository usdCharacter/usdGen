// usdGen — painted attribute maps: the .ptx bake and its I/O lane.
// See attributeBake.h for the handoff contract.
#include "usdGenSurface/maps/attributeBake.h"
#include "usdGen/exec/parkingWord.h"

// Ptexture.h comes from the private usdGen_ptex archive (PTEX_STATIC,
// PTEX_VENDOR=usdGen); only this translation unit (and ptexMap.cpp) sees it.
// On ELF/Mach-O its declarations are made hidden, so the Ptex code
// instantiated here stays out of libusdGen.so's dynamic table (gate B-1,
// cmake/CheckNoThirdPartyExports.cmake).
#include <cstdint>
#include <cstdio>
#if defined(__GNUC__) && !defined(_WIN32)
#  pragma GCC visibility push(hidden)
#endif
#include <Ptexture.h>
#if defined(__GNUC__) && !defined(_WIN32)
#  pragma GCC visibility pop
#endif

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace usdGen {

namespace {

namespace fs = std::filesystem;

int Log2Int(int value)
{
    int bits = 0;
    while ((1 << (bits + 1)) <= value) ++bits;
    return bits;
}

// Quad-edge adjacency, the bakePtex.cpp derivation reduced to the quad/quad
// case: manifold edges only, boundary and non-manifold edges stay (-1, 0).
// Edge e of face f runs from corner e to corner (e+1)%4.
class QuadAdjacency {
public:
    QuadAdjacency(std::vector<int> const &counts, std::vector<int> const &indices)
    {
        for (int f = 0; f < int(counts.size()); ++f) {
            for (int e = 0; e < 4; ++e) {
                uint64_t const a = uint32_t(indices[size_t(f) * 4 + size_t(e)]);
                uint64_t const b = uint32_t(indices[size_t(f) * 4 + size_t((e + 1) % 4)]);
                _uses[a < b ? (a << 32) | b : (b << 32) | a].push_back({f, e});
            }
        }
    }

    void Resolve(std::vector<int> const &indices, int face, int adjFaces[4],
                 int adjEdges[4]) const
    {
        for (int e = 0; e < 4; ++e) {
            adjFaces[e] = -1;
            adjEdges[e] = 0;
            uint64_t const a = uint32_t(indices[size_t(face) * 4 + size_t(e)]);
            uint64_t const b = uint32_t(indices[size_t(face) * 4 + size_t((e + 1) % 4)]);
            auto const found = _uses.find(a < b ? (a << 32) | b : (b << 32) | a);
            // Exactly two uses (this face plus one neighbour) is manifold.
            if (found == _uses.end() || found->second.size() != 2) continue;
            auto const &uses = found->second;
            auto const other = (uses[0].first == face && uses[0].second == e)
                                   ? uses[1]
                                   : uses[0];
            if (other.first == face) continue;  // a folded edge is no neighbour
            adjFaces[e] = other.first;
            adjEdges[e] = other.second;
        }
    }

private:
    std::unordered_map<uint64_t, std::vector<std::pair<int, int>>> _uses;
};

bool ParseVersionedName(std::string const &baseName, std::string const &fileName,
                        uint64_t *version)
{
    // baseName.v<digits>.ptx, nothing else.
    std::string const prefix = baseName + ".v";
    std::string const suffix = ".ptx";
    if (fileName.size() <= prefix.size() + suffix.size()) return false;
    if (fileName.compare(0, prefix.size(), prefix) != 0) return false;
    if (fileName.compare(fileName.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;
    std::string const digits =
        fileName.substr(prefix.size(), fileName.size() - prefix.size() - suffix.size());
    if (digits.empty() ||
        !std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; }))
        return false;
    try {
        *version = std::stoull(digits);
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace

std::string UsdGenAttributeBakeFileName(std::string const &baseName, uint64_t version)
{
    return baseName + ".v" + std::to_string(version) + ".ptx";
}

bool UsdGenAttributeBakePtex(UsdGenAttributeBakeInput const &input,
                             std::string const &outPath,
                             UsdGenAttributeBakeStats *stats, std::string *err)
{
    auto fail = [&](std::string const &message) {
        if (err) *err = message;
        return false;
    };
    if (!input.map) return fail("attribute bake needs a map");
    int const faces = input.map->NumFaces();
    if (input.faceVertexCounts.size() != size_t(faces))
        return fail("attribute bake topology has " +
                    std::to_string(input.faceVertexCounts.size()) + " faces for a " +
                    std::to_string(faces) + "-face map");
    for (int c : input.faceVertexCounts) {
        if (c != 4)
            return fail("attribute bake v1 needs a quad mesh (found a non-quad face)");
    }
    if (input.faceVertexIndices.size() != size_t(faces) * 4)
        return fail("attribute bake indices do not match the quad topology");
    for (int index : input.faceVertexIndices) {
        if (index < 0) return fail("attribute bake indices must be >= 0");
    }
    if (outPath.empty()) return fail("attribute bake needs an out path");

    int const res = input.map->Resolution();
    int const channels = input.map->Channels();
    // Ptex::Res takes the log2 as int8_t. A functional cast in the FaceInfo
    // initializer below is parsed as a parameter declaration (int8_t resLog2),
    // so the second argument redefines it. Store the value in that type and
    // pass the name.
    int8_t const resLog2 = int8_t(Log2Int(res));  // res is a validated power of two
    float const *src = input.map->Data();

    // The out directory is the worker's to own: create it rather than fail a
    // bake on a missing folder.
    {
        std::error_code ec;
        fs::path const parent = fs::path(outPath).parent_path();
        if (!parent.empty()) fs::create_directories(parent, ec);
        if (ec) return fail("attribute bake cannot create " + parent.string());
    }

    std::string const tmp = outPath + ".tmp";
    {
        Ptex::String error;
        PtexPtr<PtexWriter> writer(PtexWriter::open(
            tmp.c_str(), Ptex::mt_quad, Ptex::dt_float, channels, /*alphachan*/ -1,
            faces, error, /*genmipmaps*/ true));
        if (!writer)
            return fail("attribute bake cannot open " + tmp + ": " + error.c_str());
        bool ok = true;
        writer->writeMeta("usdGen:attributeBake:baseName", input.baseName.c_str());
        QuadAdjacency const adjacency(input.faceVertexCounts, input.faceVertexIndices);
        for (int f = 0; f < faces && ok; ++f) {
            int adjFaces[4], adjEdges[4];
            adjacency.Resolve(input.faceVertexIndices, f, adjFaces, adjEdges);
            Ptex::FaceInfo const info(Ptex::Res(resLog2, resLog2),
                                      adjFaces, adjEdges, /*isSubface*/ false);
            // Ptex face id == coarse face id on a quad mesh, and the texel
            // order matches the map's ((t * res) + s) grid, so each face
            // copies verbatim: texel (s, t) sits at u = (s + 0.5) / res on
            // both sides.
            ok = writer->writeFace(
                f, info, src + size_t(f) * size_t(res) * size_t(res) * size_t(channels));
        }
        ok = writer->close(error) && ok;
        if (!ok) {
            std::error_code ec;
            fs::remove(tmp, ec);
            return fail("attribute bake write failed: " + std::string(error.c_str()));
        }
    }
    std::error_code ec;
    fs::rename(tmp, outPath, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return fail("attribute bake cannot rename to " + outPath);
    }
    if (stats) {
        stats->faces = size_t(faces);
        stats->texels = size_t(faces) * size_t(res) * size_t(res);
        stats->channels = channels;
    }
    return true;
}

UsdGenAttributeBakeWorker::UsdGenAttributeBakeWorker()
{
    // Start the lane once every member is live: _worker is declared before
    // the words and mailboxes it reads.
    _worker = std::thread(&UsdGenAttributeBakeWorker::_WorkerLoop, this);
}

UsdGenAttributeBakeWorker::~UsdGenAttributeBakeWorker()
{
    _stop.store(true, std::memory_order_release);
    _Signal();
    if (_worker.joinable()) _worker.join();
    // The lane is gone: whatever is left in the mailboxes is ours.
    delete _pending.exchange(nullptr, std::memory_order_acquire);
    delete _spareRequest.exchange(nullptr, std::memory_order_acquire);
    delete _completed.exchange(nullptr, std::memory_order_acquire);
    delete _spareCompletion.exchange(nullptr, std::memory_order_acquire);
    for (Diagnostic *d = _diagnostics.exchange(nullptr, std::memory_order_acquire); d;) {
        Diagnostic *next = d->next;
        delete d;
        d = next;
    }
}

void UsdGenAttributeBakeWorker::_Signal()
{
    // After the state change: the lane reads the word before the state, so a
    // bump it has not seen makes its park return at once.
    _wakeWord.fetch_add(1, std::memory_order_acq_rel);
    UsdGenUnparkAll(_wakeWord);
}

void UsdGenAttributeBakeWorker::Enqueue(uint64_t version, UsdGenAttributeBakeInput input)
{
    Request *request = _spareRequest.exchange(nullptr, std::memory_order_acq_rel);
    if (!request) request = new Request;
    request->version = version;
    request->input = std::move(input);
    _pendingVersionAtomic.store(version);
    if (Request *dropped = _pending.exchange(request, std::memory_order_acq_rel)) {
        _dropCount.fetch_add(1);  // coalesced, never baked
        dropped->input = UsdGenAttributeBakeInput();
        delete _spareRequest.exchange(dropped, std::memory_order_acq_rel);
    }
    _Signal();
}

bool UsdGenAttributeBakeWorker::TakeCompleted(uint64_t *version, std::string *path)
{
    Completion *done = _completed.exchange(nullptr, std::memory_order_acq_rel);
    if (!done) return false;
    if (version) *version = done->version;
    if (path) *path = done->path;
    delete _spareCompletion.exchange(done, std::memory_order_acq_rel);
    return true;
}

void UsdGenAttributeBakeWorker::NoteSwapped(uint64_t version)
{
    uint64_t seen = _swappedVersion.load(std::memory_order_relaxed);
    while (version > seen &&
           !_swappedVersion.compare_exchange_weak(seen, version, std::memory_order_acq_rel)) {
    }
    _sweepRequested.store(true, std::memory_order_release);
    _Signal();
}

void UsdGenAttributeBakeWorker::_Diagnose(std::string text)
{
    Diagnostic *d = new Diagnostic;
    d->text = std::move(text);
    d->next = _diagnostics.load(std::memory_order_relaxed);
    while (!_diagnostics.compare_exchange_weak(d->next, d, std::memory_order_release,
                                               std::memory_order_relaxed)) {
    }
}

std::string UsdGenAttributeBakeWorker::TakeDiagnostic()
{
    // Newest first on the stack; reverse to report in the order seen.
    Diagnostic *d = _diagnostics.exchange(nullptr, std::memory_order_acquire);
    Diagnostic *ordered = nullptr;
    while (d) {
        Diagnostic *next = d->next;
        d->next = ordered;
        ordered = d;
        d = next;
    }
    std::string out;
    while (ordered) {
        if (!out.empty()) out += "; ";
        out += ordered->text;
        Diagnostic *next = ordered->next;
        delete ordered;
        ordered = next;
    }
    return out;
}

void UsdGenAttributeBakeWorker::PauseWorker(bool pause)
{
    _paused.store(pause, std::memory_order_release);
    _Signal();
}

bool UsdGenAttributeBakeWorker::WaitCompleted(uint64_t version, int timeoutMs)
{
    auto const deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    for (;;) {
        uint32_t const seen = _completedWord.load(std::memory_order_acquire);
        if (_completedVersionAtomic.load() >= version) return true;
        auto const now = std::chrono::steady_clock::now();
        if (now >= deadline) return false;
        auto const left =
            std::chrono::ceil<std::chrono::milliseconds>(deadline - now).count();
        UsdGenParkWhileEqualFor(_completedWord, seen, uint32_t(left));
    }
}

void UsdGenAttributeBakeWorker::_WorkerLoop()
{
    for (;;) {
        // Read the word before the state (see _Signal).
        uint32_t const seen = _wakeWord.load(std::memory_order_acquire);
        if (_stop.load(std::memory_order_acquire)) return;
        if (!_paused.load(std::memory_order_acquire)) {
            if (Request *request = _pending.exchange(nullptr, std::memory_order_acq_rel)) {
                _Bake(request);
                request->input = UsdGenAttributeBakeInput();
                delete _spareRequest.exchange(request, std::memory_order_acq_rel);
                continue;
            }
            if (_sweepRequested.exchange(false, std::memory_order_acq_rel)) {
                uint64_t const keepFrom = _swappedVersion.load(std::memory_order_acquire);
                if (!_lastOutDir.empty()) _SweepStale(_lastOutDir, _lastBaseName, keepFrom);
                _sweepCount.fetch_add(1);
                continue;
            }
        }
        UsdGenParkWhileEqual(_wakeWord, seen);
    }
}

void UsdGenAttributeBakeWorker::_Bake(Request *request)
{
    UsdGenAttributeBakeInput const &input = request->input;
    uint64_t const version = request->version;
    if (input.outDir.empty()) {
        _Diagnose("attribute bake needs an out directory");
        return;
    }
    std::string const outPath =
        (fs::path(input.outDir) / UsdGenAttributeBakeFileName(input.baseName, version))
            .string();
    UsdGenAttributeBakeStats stats;
    std::string err;
    bool ok = false;
    try {
        ok = UsdGenAttributeBakePtex(input, outPath, &stats, &err);
    } catch (std::exception const &e) {
        err = std::string("attribute bake threw: ") + e.what();
    } catch (...) {
        err = "attribute bake threw";
    }
    _lastOutDir = input.outDir;
    _lastBaseName = input.baseName;
    if (!ok) {
        _Diagnose(std::move(err));
        return;
    }
    Completion *done = _spareCompletion.exchange(nullptr, std::memory_order_acq_rel);
    if (!done) done = new Completion;
    done->version = version;
    done->path = outPath;
    delete _completed.exchange(done, std::memory_order_acq_rel);  // only the latest
    _bakeCount.fetch_add(1);
    _completedVersionAtomic.store(version);
    _completedWord.fetch_add(1, std::memory_order_acq_rel);
    UsdGenUnparkAll(_completedWord);
}

void UsdGenAttributeBakeWorker::_SweepStale(std::string const &outDir,
                                            std::string const &baseName, uint64_t keepFrom)
{
    // Older versioned files the stage no longer references go away; the
    // swapped file and anything newer stay. Anything that is not a versioned
    // bake file is left alone.
    std::error_code ec;
    fs::directory_iterator it(outDir, ec);
    if (ec) return;
    for (fs::directory_entry const &entry : it) {
        uint64_t version = 0;
        if (!ParseVersionedName(baseName, entry.path().filename().string(), &version))
            continue;
        if (version < keepFrom) fs::remove(entry.path(), ec);
    }
}

}  // namespace usdGen
