// Painted attribute maps: the synchronous .ptx bake and the asynchronous
// bake-to-stage worker.
//
// A UsdGenBrushStroke mutates a UsdGenAttributeMap (maps/attributeMap.h) on
// the interaction thread; the groom's capture loop and the stage's
// UsdGenPtexMap prim read .ptx files through UsdGenPtexTexture (maps/ptexMap.h).
// This module is the bridge between the two: UsdGenAttributeBakePtex writes
// one map to one .ptx file, and UsdGenAttributeBakeWorker runs those bakes on
// its own I/O lane so a stroke never blocks on file I/O.
//
// The handoff contract, per stroke:
//
//   press/move (interaction thread): stroke.Preview() -> live Hydra overlay
//       (UsdGenAttributePreviewSceneIndex); Enqueue() the preview at release
//   I/O lane:      write <base>.v<n>.ptx (tmp -> rename)
//   UI idle:       TakeCompleted() -> author usdGen:map:file on the stage prim
//       -> NoteSwapped() so older versioned files are swept
//
// Versioned filenames keep the stage swap atomic (the swap never points at a
// half-written file) and keep the process Ptex cache, keyed by path, from
// serving stale texels.
//
// v1 limits, documented rather than silent:
//   - Quad meshes only: every faceVertexCounts entry must be 4, so ptex face
//     id == coarse face id and each face's res x res grid copies verbatim.
//     An n-gon mesh fails closed; sub-face replication is future work.
//   - The worker coalesces: only the latest enqueued input bakes. A stroke
//     that releases twice before the worker wakes pays for one bake.
//
// Threading: no locks. Requests and completions are newest-wins mailboxes
// (one atomic pointer each), diagnostics a lock-free stack. The lane parks on
// a word (parkingWord.h) and every producer bumps it after publishing, so no
// wake is lost and nothing polls.
//
// This header has no USD dependency, like attributeMap.h and ptexMap.h. Only
// attributeBake.cpp includes <Ptexture.h>.
#ifndef USDGEN_MAPS_ATTRIBUTE_BAKE_H
#define USDGEN_MAPS_ATTRIBUTE_BAKE_H

#include "usdGenSurface/export.h"
#include "usdGen/maps/attributeMap.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace usdGen {

// Everything one bake needs: plain data, safe to move to the worker. The map
// is shared (a per-enqueue clone of a 256-res scalp map would be pure waste);
// the caller must not mutate it while the bake is in flight — pass the
// stroke's Preview()/Commit() snapshot, which the stroke never touches again.
struct UsdGenAttributeBakeInput {
    std::shared_ptr<const UsdGenAttributeMap> map;
    // Parent mesh topology, for the Ptex adjacency. counts.size() must equal
    // map->NumFaces(); every entry must be 4 (v1: quads only).
    std::vector<int> faceVertexCounts;
    std::vector<int> faceVertexIndices;
    std::string outDir;  // versioned .ptx files land here (created as needed)
    std::string baseName = "paintMap";  // paintMap.v<n>.ptx
};

struct UsdGenAttributeBakeStats {
    size_t faces = 0;
    size_t texels = 0;
    int channels = 0;
};

// Writes the map to `outPath` synchronously (the worker's unit of work, also
// the direct entry point). Writes to a temp file and renames, so `outPath`
// never names a half-written file. Returns false with *err set (when
// non-null) on any failure — a null map, a face-count mismatch, a non-quad
// face, inconsistent indices, an empty out path — and writes nothing.
USDGEN_SURFACE_API bool UsdGenAttributeBakePtex(
    UsdGenAttributeBakeInput const &input, std::string const &outPath,
    UsdGenAttributeBakeStats *stats, std::string *err);

// Versioned filename for a bake version (no directory join).
USDGEN_SURFACE_API std::string UsdGenAttributeBakeFileName(
    std::string const &baseName, uint64_t version);

class USDGEN_SURFACE_API UsdGenAttributeBakeWorker {
public:
    UsdGenAttributeBakeWorker();
    ~UsdGenAttributeBakeWorker();
    UsdGenAttributeBakeWorker(UsdGenAttributeBakeWorker const &) = delete;
    UsdGenAttributeBakeWorker &operator=(UsdGenAttributeBakeWorker const &) = delete;

    // Request a bake of `input` as version `version` (interaction thread, at
    // release). Returns immediately: the bake runs on the worker thread, so a
    // stroke never waits for file I/O. Stores only the latest input; the
    // worker takes it when it wakes (coalescing: an unconsumed pending input
    // is dropped and counted, never baked).
    void Enqueue(uint64_t version, UsdGenAttributeBakeInput input);

    // Latest bake finished by the worker (UI thread, at idle). Returns false
    // when no bake has completed since the last TakeCompleted.
    bool TakeCompleted(uint64_t *version, std::string *path);

    // Tell the worker which version the UI swapped into the stage (UI thread,
    // after authoring usdGen:map:file). Older versioned files the stage no
    // longer references are deleted by the worker.
    void NoteSwapped(uint64_t version);

    uint64_t PendingVersion() const { return _pendingVersionAtomic.load(); }
    uint64_t CompletedVersion() const { return _completedVersionAtomic.load(); }
    // Test introspection: bakes finished, enqueued inputs dropped by
    // coalescing, stale-file sweeps run, and the last error the worker saw.
    size_t BakeCount() const { return _bakeCount.load(); }
    size_t DropCount() const { return _dropCount.load(); }
    size_t SweepCount() const { return _sweepCount.load(); }
    std::string TakeDiagnostic();
    // Test hook: hold the worker between bakes so coalescing is
    // deterministic (enqueue N versions, resume, expect one bake).
    void PauseWorker(bool pause);
    // Test hook: wait for a version to complete (parks; no polling).
    bool WaitCompleted(uint64_t version, int timeoutMs = 10000);

private:
    // Mailbox payloads. Whoever exchanges one out of its slot owns it.
    struct Request {
        uint64_t version = 0;
        UsdGenAttributeBakeInput input;
    };
    struct Completion {
        uint64_t version = 0;
        std::string path;
    };
    struct Diagnostic {
        Diagnostic *next = nullptr;
        std::string text;
    };

    void _WorkerLoop();
    void _Bake(Request *request);
    void _SweepStale(std::string const &outDir, std::string const &baseName,
                     uint64_t keepFrom);
    void _Diagnose(std::string text);
    void _Signal();

    std::thread _worker;  // the I/O lane
    std::atomic<uint32_t> _wakeWord{0};       // bumped after every request
    std::atomic<uint32_t> _completedWord{0};  // bumped after every bake
    std::atomic<bool> _stop{false};
    std::atomic<bool> _paused{false};
    std::atomic<bool> _sweepRequested{false};
    std::atomic<uint64_t> _swappedVersion{0};
    // Newest wins: an unconsumed request is dropped (counted) when replaced.
    std::atomic<Request *> _pending{nullptr};
    std::atomic<Request *> _spareRequest{nullptr};
    // Only the latest completion is kept: the UI swaps the newest file.
    std::atomic<Completion *> _completed{nullptr};
    std::atomic<Completion *> _spareCompletion{nullptr};
    std::atomic<Diagnostic *> _diagnostics{nullptr};  // newest first
    // Lane-owned: the last bake's directory and base name, for the sweep.
    std::string _lastOutDir;
    std::string _lastBaseName;

    std::atomic<uint64_t> _pendingVersionAtomic{0};
    std::atomic<uint64_t> _completedVersionAtomic{0};
    std::atomic<size_t> _bakeCount{0};
    std::atomic<size_t> _dropCount{0};
    std::atomic<size_t> _sweepCount{0};
};

}  // namespace usdGen

#endif  // USDGEN_MAPS_ATTRIBUTE_BAKE_H
