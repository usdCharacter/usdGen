// usdGenPomade — the asynchronous region-map bake implementation (P2).
#include "usdGenPomade/pomadeBake.h"

#include "usdGenPomade/pomadeHierarchy.h"
#include "usdGenPomade/pomadeModel.h"

#include "usdGen/exec/epoch.h"
#include "usdGenSurface/maps/ptexMap.h"
#include "usdGen/exec/parkingWord.h"

#ifdef USDGEN_POMADE_HAS_CUDA
#include "usdGenPomade/pomadeKernels.h"
#include "usdGenCuda/gpu/deviceBuffer.h"
#include <cuda_runtime.h>
#endif

// Last: <Ptexture.h> brings in <windows.h>.
#include <Ptexture.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <thread>

namespace usdGenPomade {

namespace {

namespace fs = std::filesystem;

// Cache entries contain classifier output, rather than just a coarse-face
// label.  Hash every input that can change a texel so an edit which stays
// within one face (and therefore preserves its centroid label) cannot reuse
// stale texels.  Exact repeats retain the incremental no-work fast path.
void _HashBytes(uint64_t *hash, void const *data, size_t size)
{
    unsigned char const *bytes = static_cast<unsigned char const *>(data);
    for (size_t i = 0; i < size; ++i) {
        *hash ^= uint64_t(bytes[i]);
        *hash *= 1099511628211ull;
    }
}

template <class T>
void _HashValue(uint64_t *hash, T const &value)
{
    _HashBytes(hash, &value, sizeof(value));
}

template <class T>
void _HashVector(uint64_t *hash, std::vector<T> const &values)
{
    size_t const size = values.size();
    _HashValue(hash, size);
    if (!values.empty()) {
        _HashBytes(hash, values.data(), values.size() * sizeof(T));
    }
}

uint64_t _ClassifierKey(PomadeBakeInput const &input,
                        PomadeRegionLoops const &loops)
{
    uint64_t hash = 1469598103934665603ull;
    PomadeScalpMesh const &mesh = *input.scalp;
    _HashVector(&hash, mesh.points);
    _HashVector(&hash, mesh.faceVertexCounts);
    _HashVector(&hash, mesh.faceVertexIndices);
    if (!mesh.activeFaces.empty()) {
        _HashVector(&hash, mesh.activeFaces);  // a face subset (§2.20)
    }
    _HashVector(&hash, loops.points);
    _HashVector(&hash, loops.loopBegin);
    _HashVector(&hash, loops.loopCount);
    _HashVector(&hash, loops.planeN);
    _HashVector(&hash, loops.planeP);
    _HashVector(&hash, loops.basisU);
    _HashVector(&hash, loops.basisV);
    _HashVector(&hash, loops.interpIds);
    _HashValue(&hash, input.levelCount);
    _HashValue(&hash, input.resOverride);
    _HashValue(&hash, input.outputOwnerMap);
    size_t const tubeCount = input.tubes.size();
    _HashValue(&hash, tubeCount);
    for (PomadeBakeTube const &tube : input.tubes) {
        _HashValue(&hash, tube.tubeId);
        _HashValue(&hash, tube.parentTubeId);
        _HashValue(&hash, tube.level);
        _HashValue(&hash, tube.regionId);
        _HashValue(&hash, tube.childIndex);
        _HashBytes(&hash, tube.rootCenter, sizeof(tube.rootCenter));
        _HashValue(&hash, tube.rootFrame.tx);
        _HashValue(&hash, tube.rootFrame.ty);
        _HashValue(&hash, tube.rootFrame.tz);
        _HashValue(&hash, tube.rootFrame.nx);
        _HashValue(&hash, tube.rootFrame.ny);
        _HashValue(&hash, tube.rootFrame.nz);
        _HashValue(&hash, tube.rootFrame.bx);
        _HashValue(&hash, tube.rootFrame.by);
        _HashValue(&hash, tube.rootFrame.bz);
    }
    return hash;
}

std::string _CacheIdentity(std::string const &outDir, uint64_t classifierKey)
{
    // Kept in the existing worker cache-directory string so PomadeBakeWorker
    // remains ABI-stable for clients that construct it across the DLL.
    return outDir + "\x1f" + std::to_string(classifierKey);
}

struct _BakeFace {
    int coarse = -1;
    int sub = 0;  // n-gon sub-face, else 0
    int resU = 1;
    int resV = 1;
    // False outside a bound face subset: the ptex keeps every parent face
    // (ids are parent-derived), but such a face is never claimed.
    bool active = true;
    float corners[12];
};

struct _BakePlan {
    std::vector<_BakeFace> faces;  // ptex-face order
    std::vector<int> firstIds;  // per coarse face (UsdGenPtexFirstFaceIds)
    int channels = 1;
    bool outputOwnerMap = false;
};

bool _PlanFaces(PomadeBakeInput const &input, PomadeRegionMaps const &maps,
                std::vector<int> const &resLog2, _BakePlan *plan,
                std::string *err)
{
    PomadeScalpMesh const &mesh = *input.scalp;
    size_t const faceCount = mesh.faceVertexCounts.size();
    int total = 0;
    plan->firstIds = usdGen::UsdGenPtexFirstFaceIds(
        mesh.faceVertexCounts.data(), faceCount, &total);
    if (plan->firstIds.empty() || total < 0) {
        if (err) {
            *err = "PomadeBakePtex: ptex face ids overflow";
        }
        return false;
    }
    plan->channels = input.outputOwnerMap ? 1 : std::max(input.levelCount, 1);
    plan->outputOwnerMap = input.outputOwnerMap;
    plan->faces.clear();
    plan->faces.reserve(size_t(total));
    for (size_t f = 0; f < faceCount; ++f) {
        int const nv = mesh.faceVertexCounts[f];
        int const subs = (nv == 4) ? 1 : nv;
        int const r = resLog2[f];
        for (int s = 0; s < subs; ++s) {
            _BakeFace bf;
            bf.coarse = int(f);
            bf.sub = s;
            bf.active = PomadeScalpFaceActive(mesh, int(f));
            int rr = (subs == 1) ? r : std::max(r - 1, 0);
            bf.resU = bf.resV = 1 << rr;
            float const *pts = mesh.points.data();
            if (!usdGen::UsdGenPtexFaceCorners(
                    pts, mesh.points.size() / 3,
                    mesh.faceVertexCounts.data(),
                    mesh.faceVertexIndices.data(), mesh.faceOffsets.data(),
                    faceCount, int(f), s, bf.corners)) {
                if (err) {
                    *err = "PomadeBakePtex: bad face corners";
                }
                return false;
            }
            plan->faces.push_back(bf);
        }
    }
    return true;
}

void _TexelPosition(_BakeFace const &bf, int tu, int tv, float out[3])
{
    float const u = (float(tu) + 0.5f) / float(bf.resU);
    float const v = (float(tv) + 0.5f) / float(bf.resV);
    float const w0 = (1.0f - u) * (1.0f - v);
    float const w1 = u * (1.0f - v);
    float const w2 = u * v;
    float const w3 = (1.0f - u) * v;
    for (int a = 0; a < 3; ++a) {
        out[a] = w0 * bf.corners[a] + w1 * bf.corners[3 + a] +
                 w2 * bf.corners[6 + a] + w3 * bf.corners[9 + a];
    }
}

bool _WritePtexFile(_BakePlan const &plan,
                    std::vector<std::vector<float>> const &texelCache,
                    std::string const &outPath, std::string *err)
{
    auto fail = [&](std::string const &what) {
        if (err) {
            *err = "PomadeBakePtex: " + what;
        }
        return false;
    };
    if (texelCache.size() != plan.faces.size()) {
        return fail("texel cache does not cover every ptex face");
    }
    std::string const tmp = outPath + ".tmp";
    {
        Ptex::String error;
        PtexPtr<PtexWriter> writer(PtexWriter::open(
            tmp.c_str(), Ptex::mt_quad, Ptex::dt_float, plan.channels, -1,
            int(plan.faces.size()), error, true));
        if (!writer) {
            return fail("cannot open " + tmp + ": " + error.c_str());
        }
        bool ok = true;
        for (size_t id = 0; id < plan.faces.size() && ok; ++id) {
            _BakeFace const &bf = plan.faces[id];
            int adjfaces[4] = {-1, -1, -1, -1};
            int adjedges[4] = {0, 0, 0, 0};
            auto log2i = [](int n) {
                int r = 0;
                while ((1 << (r + 1)) <= n) {
                    ++r;
                }
                return r;
            };
            Ptex::FaceInfo info(Ptex::Res(log2i(bf.resU), log2i(bf.resV)),
                                adjfaces, adjedges);
            size_t const want =
                size_t(bf.resU) * size_t(bf.resV) * size_t(plan.channels);
            if (texelCache[id].size() != want) {
                return fail("ptex face " + std::to_string(id) + " caches " +
                            std::to_string(texelCache[id].size()) +
                            " floats for " + std::to_string(want));
            }
            ok = writer->writeFace(int(id), info, texelCache[id].data());
        }
        ok = writer->close(error) && ok;
        if (!ok) {
            std::error_code ec;
            fs::remove(tmp, ec);
            return fail("write failed: " + std::string(error.c_str()));
        }
    }
    // Tmp -> rename: the versioned path never names a half-written file.
    std::error_code ec;
    fs::rename(tmp, outPath, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return fail("rename to " + outPath + " failed");
    }
    return true;
}

// A face whose id is unchanged can still need re-classification: the
// res-0 collapse keys on corner uniformity, so a graph edit can flip a
// face's resolution (and a level/res-override change flips every face's
// texel count) without touching its id. Any cache entry whose size misses
// the plan is stale by construction and joins the dirty set.
void _HealStaleCache(_BakePlan const &plan,
                     std::vector<std::vector<float>> const &texelCache,
                     std::vector<char> *isDirty)
{
    for (size_t id = 0; id < plan.faces.size(); ++id) {
        _BakeFace const &bf = plan.faces[id];
        size_t const want = size_t(bf.resU) * size_t(bf.resV) *
                            size_t(plan.channels);
        if (id < texelCache.size() && texelCache[id].size() != want) {
            (*isDirty)[size_t(bf.coarse)] = 1;
        }
    }
}

// The hierarchy side of the channel layout (plan/17 §4.5, plan/18 §7 G4):
// channel 0 is the L1 interpolation id, channel k the id of the level-(k+1)
// tube the texel falls in. Built once per bake from PomadeBakeInput::tubes.
struct _LevelIndex {
    // interp id -> L1 tube id.
    std::map<int, int> l1OfRegion;
    std::map<int, PomadeBakeTube> byId;
    std::map<int, std::vector<int>> childrenOf;  // childIndex order

    bool empty() const { return byId.empty(); }
};

_LevelIndex _BuildLevelIndex(PomadeBakeInput const &input,
                             PomadeRegionLoops const &loops)
{
    _LevelIndex index;
    for (PomadeBakeTube const &tube : input.tubes) {
        index.byId[tube.tubeId] = tube;
        if (tube.parentTubeId != -1) {
            index.childrenOf[tube.parentTubeId].push_back(tube.tubeId);
        } else if (tube.level == 1) {
            int interp = tube.regionId;
            if (tube.regionId >= 0 &&
                size_t(tube.regionId) < loops.interpIds.size()) {
                interp = loops.interpIds[size_t(tube.regionId)];
            }
            index.l1OfRegion.emplace(interp, tube.tubeId);
        }
    }
    for (auto &kv : index.childrenOf) {
        std::sort(kv.second.begin(), kv.second.end(), [&](int a, int b) {
            return index.byId[a].childIndex < index.byId[b].childIndex;
        });
    }
    return index;
}

// The normal RegionMap stores linked interpolation ids in channel zero.  An
// Output owner map instead needs the source graph-region identity: two
// linked regions may interpolate together but still root distinct L1 tubes.
// Keeping a private loop copy avoids changing the ordinary RegionMap path.
PomadeRegionLoops
_OutputOwnerLoops(PomadeRegionLoops loops)
{
    for (size_t i = 0; i < loops.interpIds.size(); ++i) {
        loops.interpIds[i] = int(i);
    }
    return loops;
}

// Fill texels[1 .. channels-1] for one point. `region` is the channel-0 id.
void _ClassifyLevels(_LevelIndex const &index, int region, float const p[3],
                     int channels, float *texels)
{
    for (int c = 1; c < channels; ++c) {
        texels[c] = 0.0f;
    }
    if (channels <= 1 || index.empty()) {
        return;
    }
    auto const l1 = index.l1OfRegion.find(region);
    if (l1 == index.l1OfRegion.end()) {
        return;
    }
    int current = l1->second;
    for (int c = 1; c < channels; ++c) {
        auto const kids = index.childrenOf.find(current);
        if (kids == index.childrenOf.end() || kids->second.empty()) {
            return;  // this level has no tube here: 0, per §2.2
        }
        auto const parent = index.byId.find(current);
        if (parent == index.byId.end()) {
            return;
        }
        std::vector<float> centers;
        centers.reserve(kids->second.size() * 3);
        for (int kid : kids->second) {
            auto const it = index.byId.find(kid);
            if (it == index.byId.end()) {
                return;
            }
            centers.push_back(it->second.rootCenter[0]);
            centers.push_back(it->second.rootCenter[1]);
            centers.push_back(it->second.rootCenter[2]);
        }
        int const cell = PomadeOwningChildCell(
            parent->second.rootCenter, parent->second.rootFrame,
            centers.data(), int(kids->second.size()), p);
        if (cell < 0) {
            return;
        }
        current = kids->second[size_t(cell)];
        texels[c] = float(current);
    }
}

// Output's categorical map names the deepest live tube which owns a scalp
// point.  It walks precisely the same root-frame K14 cells as the hierarchy
// channels, but stores tubeId + 1 so zero remains the unowned sentinel.
int _ClassifyOutputOwner(_LevelIndex const &index, int region,
                         float const p[3])
{
    if (index.empty()) {
        return 0;
    }
    auto const l1 = index.l1OfRegion.find(region);
    if (l1 == index.l1OfRegion.end()) {
        return 0;
    }
    int current = l1->second;
    for (;;) {
        auto const kids = index.childrenOf.find(current);
        if (kids == index.childrenOf.end() || kids->second.empty()) {
            break;
        }
        auto const parent = index.byId.find(current);
        if (parent == index.byId.end()) {
            return 0;
        }
        std::vector<float> centers;
        centers.reserve(kids->second.size() * 3);
        for (int kid : kids->second) {
            auto const child = index.byId.find(kid);
            if (child == index.byId.end()) {
                return 0;
            }
            centers.push_back(child->second.rootCenter[0]);
            centers.push_back(child->second.rootCenter[1]);
            centers.push_back(child->second.rootCenter[2]);
        }
        int const cell = PomadeOwningChildCell(
            parent->second.rootCenter, parent->second.rootFrame,
            centers.data(), int(kids->second.size()), p);
        if (cell < 0) {
            break;
        }
        current = kids->second[size_t(cell)];
    }
    return current >= 0 ? current + 1 : 0;
}

// The texels of a face no region claims: channel 0 is -1 (an Output owner
// map's unowned sentinel is 0), every level channel 0 (plan/17 §2.2).
std::vector<float> _UnclaimedTexels(_BakePlan const &plan,
                                    _BakeFace const &bf)
{
    std::vector<float> texels(size_t(bf.resU) * size_t(bf.resV) *
                              size_t(plan.channels));
    for (size_t t = 0; t < size_t(bf.resU) * size_t(bf.resV); ++t) {
        texels[t * size_t(plan.channels)] =
            plan.outputOwnerMap ? 0.0f : -1.0f;
    }
    return texels;
}

// Classify every texel of the plan's dirty coarse faces on the CPU.
void _ClassifyDirtyCpu(PomadeRegionLoops const &loops, _BakePlan const &plan,
                       _LevelIndex const &levels,
                       std::vector<char> const &isDirty,
                       std::vector<std::vector<float>> *texelCache,
                       PomadeBakeStats *stats)
{
    for (size_t id = 0; id < plan.faces.size(); ++id) {
        _BakeFace const &bf = plan.faces[id];
        if (!isDirty[size_t(bf.coarse)]) {
            continue;
        }
        if (!bf.active) {
            // Outside the face subset: unclaimed whatever a loop spans,
            // exactly as the live primvar reads it.
            (*texelCache)[id] = _UnclaimedTexels(plan, bf);
            if (stats) {
                ++stats->facesClassified;
            }
            continue;
        }
        std::vector<float> texels(size_t(bf.resU) * size_t(bf.resV) *
                                  size_t(plan.channels));
        for (int tv = 0; tv < bf.resV; ++tv) {
            for (int tu = 0; tu < bf.resU; ++tu) {
                float p[3];
                _TexelPosition(bf, tu, tv, p);
                int const region = PomadeClassifyPointCpu(loops, p);
                size_t const o = (size_t(tv) * size_t(bf.resU) + size_t(tu)) *
                                 size_t(plan.channels);
                if (plan.outputOwnerMap) {
                    texels[o] = float(_ClassifyOutputOwner(levels, region, p));
                } else {
                    texels[o] = float(region);
                    _ClassifyLevels(levels, region, p, plan.channels,
                                    &texels[o]);
                }
                if (stats) {
                    ++stats->texelsClassified;
                }
            }
        }
        (*texelCache)[id] = std::move(texels);
        if (stats) {
            ++stats->facesClassified;
        }
    }
}

#ifdef USDGEN_POMADE_HAS_CUDA

// Lowest-priority stream for the bake (plan/17 §3.1a): a bake in flight
// cannot delay a move on pomadeStream.
cudaStream_t _BakeStream()
{
    int least = 0, greatest = 0;
    if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess) {
        return nullptr;
    }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, least) !=
        cudaSuccess) {
        return nullptr;
    }
    return stream;
}

// A pinned host block for the classify download: cudaMemcpyAsync on the
// bake stream only overlaps (and only stays off the default stream) when
// the destination is page-locked.
struct _PinnedInts {
    int *data = nullptr;
    size_t count = 0;
    ~_PinnedInts()
    {
        if (data) {
            cudaFreeHost(data);
        }
    }
    bool reset(size_t n)
    {
        if (data) {
            cudaFreeHost(data);
            data = nullptr;
            count = 0;
        }
        if (n == 0) {
            return true;
        }
        void *p = nullptr;
        if (cudaHostAlloc(&p, n * sizeof(int), cudaHostAllocDefault) !=
            cudaSuccess) {
            return false;
        }
        data = static_cast<int *>(p);
        count = n;
        return true;
    }
};

bool _ClassifyDirtyGpu(PomadeRegionLoops const &loops, _BakePlan const &plan,
                       _LevelIndex const &levels,
                       std::vector<char> const &isDirty,
                       std::vector<std::vector<float>> *texelCache,
                       cudaStream_t stream, PomadeBakeStats *stats,
                       std::string *err)
{
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    // Gather dirty texel positions on the host (bilinear: cheap), classify
    // on the device in ~1M-texel launches, scatter back per ptex face.
    struct _Span {
        size_t id;
        size_t begin;
        size_t count;
    };
    std::vector<_Span> spans;
    std::vector<float> positions;
    for (size_t id = 0; id < plan.faces.size(); ++id) {
        _BakeFace const &bf = plan.faces[id];
        if (!isDirty[size_t(bf.coarse)]) {
            continue;
        }
        if (!bf.active) {
            // Outside the face subset: no kernel work (the CPU twin agrees).
            (*texelCache)[id] = _UnclaimedTexels(plan, bf);
            if (stats) {
                ++stats->facesClassified;
            }
            continue;
        }
        _Span span{id, positions.size() / 3, 0};
        for (int tv = 0; tv < bf.resV; ++tv) {
            for (int tu = 0; tu < bf.resU; ++tu) {
                float p[3];
                _TexelPosition(bf, tu, tv, p);
                positions.push_back(p[0]);
                positions.push_back(p[1]);
                positions.push_back(p[2]);
                ++span.count;
            }
        }
        spans.push_back(span);
    }
    if (spans.empty()) {
        return true;
    }
    int const regionCount = int(loops.loopCount.size());
    if (regionCount == 0) {
        // No regions: every dirty texel is -1, no kernel needed.
        for (auto const &span : spans) {
            _BakeFace const &bf = plan.faces[span.id];
            // No regions means no L1 tube either, so every level channel
            // stays 0; nothing to classify.
            (*texelCache)[span.id] = _UnclaimedTexels(plan, bf);
            if (stats) {
                ++stats->facesClassified;
                stats->texelsClassified +=
                    size_t(bf.resU) * size_t(bf.resV);
            }
        }
        return true;
    }
    usdGen::gpu::DeviceBuffer<float> dPos, dLoop, dN, dP, dU, dV;
    usdGen::gpu::DeviceBuffer<int> dBegin, dCount, dInterp, dOut;
    size_t const nPoints = positions.size() / 3;
    if (dPos.reset(positions.size()) != cudaSuccess ||
        dLoop.reset(loops.points.size()) != cudaSuccess ||
        dBegin.reset(loops.loopBegin.size()) != cudaSuccess ||
        dCount.reset(loops.loopCount.size()) != cudaSuccess ||
        dN.reset(loops.planeN.size()) != cudaSuccess ||
        dP.reset(loops.planeP.size()) != cudaSuccess ||
        dU.reset(loops.basisU.size()) != cudaSuccess ||
        dV.reset(loops.basisV.size()) != cudaSuccess ||
        dInterp.reset(loops.interpIds.size()) != cudaSuccess ||
        dOut.reset(nPoints) != cudaSuccess) {
        return fail("PomadeBakePtex: device allocation failed");
    }
    auto upload = [&](auto &buf, auto const &host) {
        return cudaMemcpyAsync(buf.data(), host.data(),
                               host.size() * sizeof(host[0]),
                               cudaMemcpyHostToDevice, stream) == cudaSuccess;
    };
    if (!upload(dPos, positions) || !upload(dLoop, loops.points) ||
        !upload(dBegin, loops.loopBegin) || !upload(dCount, loops.loopCount) ||
        !upload(dN, loops.planeN) || !upload(dP, loops.planeP) ||
        !upload(dU, loops.basisU) || !upload(dV, loops.basisV) ||
        !upload(dInterp, loops.interpIds)) {
        return fail("PomadeBakePtex: device upload failed");
    }
    char launchErr[256] = {0};
    if (!PomadeLaunchClassifyPoints(dPos.data(), int(nPoints), dLoop.data(),
                                   dBegin.data(), dCount.data(), dN.data(),
                                   dP.data(), dU.data(), dV.data(),
                                   dInterp.data(), regionCount, dOut.data(),
                                   stream, launchErr, sizeof(launchErr))) {
        if (err) {
            *err = std::string("PomadeBakePtex: classify failed: ") + launchErr;
        }
        return false;
    }
    // D2H on the BAKE stream into pinned memory, with one event the worker
    // waits on (§3.1a). Nothing here touches the default stream, so a bake
    // in flight cannot serialise against an interaction launch, and the UI
    // thread never synchronises with either stream.
    _PinnedInts ids;
    if (!ids.reset(nPoints)) {
        return fail("PomadeBakePtex: pinned host allocation failed");
    }
    if (cudaMemcpyAsync(ids.data, dOut.data(), nPoints * sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        return fail("PomadeBakePtex: device download failed");
    }
    cudaEvent_t done = nullptr;
    if (cudaEventCreate(&done) != cudaSuccess) {
        return fail("PomadeBakePtex: event creation failed");
    }
    cudaEventRecord(done, stream);
    cudaError_t const wait = cudaEventSynchronize(done);
    cudaEventDestroy(done);
    if (wait != cudaSuccess) {
        return fail("PomadeBakePtex: classify wait failed");
    }
    for (auto const &span : spans) {
        _BakeFace const &bf = plan.faces[span.id];
        std::vector<float> texels(size_t(bf.resU) * size_t(bf.resV) *
                                  size_t(plan.channels), 0.0f);
        for (size_t t = 0; t < span.count; ++t) {
            int const region = ids.data[span.begin + t];
            float *texel = &texels[t * size_t(plan.channels)];
            if (plan.outputOwnerMap) {
                float p[3];
                _TexelPosition(bf, int(t % size_t(bf.resU)),
                               int(t / size_t(bf.resU)), p);
                texel[0] = float(_ClassifyOutputOwner(levels, region, p));
            } else {
                texel[0] = float(region);
            }
            if (!plan.outputOwnerMap && plan.channels > 1) {
                float p[3];
                _TexelPosition(bf, int(t % size_t(bf.resU)),
                               int(t / size_t(bf.resU)), p);
                _ClassifyLevels(levels, region, p, plan.channels, texel);
            }
        }
        (*texelCache)[span.id] = std::move(texels);
        if (stats) {
            ++stats->facesClassified;
            stats->texelsClassified += span.count;
        }
    }
    return true;
}

#endif  // USDGEN_POMADE_HAS_CUDA

}  // namespace

void PomadeCollectBakeTubes(PomadeModel const &model,
                           std::vector<PomadeBakeTube> *out)
{
    if (!out) {
        return;
    }
    out->clear();
    for (int id : model.TubeIds()) {
        if (model.IsTubeImported(id)) {
            continue;  // a bridge import is not a subdivision cell
        }
        PomadeTubeDesc desc;
        if (!model.GetTubeDesc(id, &desc) || desc.centerX.size() < 2) {
            continue;
        }
        std::vector<PomadeFrame> frames;
        std::string err;
        if (!PomadeTubeFramesCpu(desc, &frames, &err) ||
            frames.empty()) {
            continue;
        }
        PomadeBakeTube tube;
        tube.tubeId = id;
        tube.parentTubeId = desc.parentTubeId;
        tube.level = desc.level;
        tube.regionId = desc.regionId;
        tube.childIndex = desc.childIndex;
        tube.rootCenter[0] = desc.centerX[0];
        tube.rootCenter[1] = desc.centerY[0];
        tube.rootCenter[2] = desc.centerZ[0];
        tube.rootFrame = frames[0];
        out->push_back(tube);
    }
}

std::string PomadeBakeFileName(std::string const &baseName, uint64_t mapVersion)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), ".v%llu.ptx",
                  (unsigned long long)mapVersion);
    return baseName + buf;
}

bool PomadeBakePtex(PomadeBakeInput const &input, std::string const &outPath,
                   std::vector<int> const *dirtyFaces,
                   std::vector<std::vector<float>> *texelCache,
                   PomadeBakeStats *stats, std::string *err)
{
    auto fail = [&](std::string const &what) {
        if (err) {
            *err = "PomadeBakePtex: " + what;
        }
        return false;
    };
    if (!input.scalp || !input.scalp->finalized || !texelCache) {
        return fail("bad scalp or null texel cache");
    }
    PomadeScalpMesh const &mesh = *input.scalp;
    size_t const faceCount = mesh.faceVertexCounts.size();
    PomadeRegionLoops loops;
    if (!PomadeFlattenLoops(input.graph, &loops, err)) {
        return false;
    }
    PomadeRegionMaps maps;
    if (!PomadeRasteriseRegionsCpu(mesh, input.graph, &maps, err)) {
        return false;
    }
    PomadeRegionLoops const ownerLoops = input.outputOwnerMap
        ? _OutputOwnerLoops(loops) : PomadeRegionLoops();
    PomadeRegionLoops const &classifierLoops = input.outputOwnerMap
        ? ownerLoops : loops;
    std::vector<int> const resLog2 =
        PomadeFaceResLog2(mesh, maps, loops, input.resOverride);
    _BakePlan plan;
    if (!_PlanFaces(input, maps, resLog2, &plan, err)) {
        return false;
    }
    if (texelCache->size() != plan.faces.size()) {
        texelCache->assign(plan.faces.size(), {});
    }
    std::vector<char> isDirty(faceCount, dirtyFaces ? 0 : 1);
    if (dirtyFaces) {
        for (int f : *dirtyFaces) {
            if (f >= 0 && size_t(f) < faceCount) {
                isDirty[size_t(f)] = 1;
            }
        }
    }
    _HealStaleCache(plan, *texelCache, &isDirty);
    if (stats) {
        *stats = PomadeBakeStats();
        stats->ptexFaces = plan.faces.size();
        stats->channels = plan.channels;
    }
    // The synchronous entry classifies on the CPU (deterministic, no stream
    // plumbing); the worker's chunked path takes the GPU lane when one
    // exists. Both call the same inside-test and the same level walk.
    _LevelIndex const levels = _BuildLevelIndex(input, classifierLoops);
    _ClassifyDirtyCpu(classifierLoops, plan, levels, isDirty, texelCache,
                      stats);
    return _WritePtexFile(plan, *texelCache, outPath, err);
}

// The lane's shared state. Requests, completions, swap notes and
// diagnostics are newest-wins handoffs (usdGen::UsdGenNewestWins); the
// worker parks on `wake`, and WaitCompleted parks on `completedWord`. A
// waiter reads its word BEFORE it checks for work and parks against that
// value, so a notify that lands in between changes the word and the park
// returns at once.
struct PomadeBakeWorker::_Lane {
    struct Request : usdGen::UsdGenRefCounted {
        uint64_t version = 0;
        PomadeBakeInput input;
    };
    struct Completion : usdGen::UsdGenRefCounted {
        uint64_t version = 0;
        std::string path;
    };
    struct Diagnostic : usdGen::UsdGenRefCounted {
        std::string text;
    };
    usdGen::UsdGenNewestWins<Request> pending;
    usdGen::UsdGenNewestWins<Completion> completed;
    usdGen::UsdGenNewestWins<Completion> swapped;  // NoteSwapped, for sweeps
    usdGen::UsdGenNewestWins<Diagnostic> diagnostic;
    std::atomic<uint32_t> wake{0};
    std::atomic<uint32_t> completedWord{0};

    static void Notify(std::atomic<uint32_t> &word)
    {
        word.fetch_add(1u, std::memory_order_seq_cst);
        usdGen::UsdGenUnparkAll(word);
    }
};

PomadeBakeWorker::PomadeBakeWorker()
    : _lane(new _Lane())
{
    // Every member is live before the lane starts (see the header).
    _worker = std::thread(&PomadeBakeWorker::_WorkerLoop, this);
}

PomadeBakeWorker::~PomadeBakeWorker()
{
    _stop.store(true, std::memory_order_seq_cst);
    _Lane::Notify(_lane->wake);
    if (_worker.joinable()) {
        _worker.join();
    }
}

void PomadeBakeWorker::Enqueue(uint64_t mapVersion, PomadeBakeInput input)
{
    auto request = usdGen::UsdGenMakeRef<_Lane::Request>();
    request->version = mapVersion;
    request->input = std::move(input);
    _pendingVersionAtomic.store(mapVersion);
    _lane->pending.Publish(std::move(request));
    _Lane::Notify(_lane->wake);
}

bool PomadeBakeWorker::TakeCompleted(uint64_t *mapVersion, std::string *path)
{
    usdGen::UsdGenRef<_Lane::Completion> done = _lane->completed.Take();
    if (!done) {
        return false;
    }
    if (mapVersion) {
        *mapVersion = done->version;
    }
    if (path) {
        *path = std::move(done->path);
    }
    return true;
}

void PomadeBakeWorker::NoteSwapped(uint64_t mapVersion, std::string const &path)
{
    // Its own wake reason: with no pending bake the worker would otherwise
    // go straight back to sleep and the file the previous swap pinned
    // would survive forever.
    auto note = usdGen::UsdGenMakeRef<_Lane::Completion>();
    note->version = mapVersion;
    note->path = path;
    _lane->swapped.Publish(std::move(note));
    _Lane::Notify(_lane->wake);
}

void PomadeBakeWorker::PauseWorker(bool pause)
{
    _paused.store(pause, std::memory_order_seq_cst);
    _Lane::Notify(_lane->wake);
}

bool PomadeBakeWorker::WaitCompleted(uint64_t mapVersion, int timeoutMs)
{
    auto const deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeoutMs);
    for (;;) {
        uint32_t const seen =
            _lane->completedWord.load(std::memory_order_seq_cst);
        if (_completedVersionAtomic.load() >= mapVersion) {
            return true;
        }
        auto const now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return false;
        }
        auto const left = std::chrono::ceil<std::chrono::milliseconds>(
                              deadline - now).count();
        usdGen::UsdGenParkWhileEqualFor(
            _lane->completedWord, seen,
            uint32_t(left > 0x7fffffff ? 0x7fffffff : left));
    }
}

std::string PomadeBakeWorker::TakeDiagnostic()
{
    usdGen::UsdGenRef<_Lane::Diagnostic> d = _lane->diagnostic.Take();
    return d ? std::move(d->text) : std::string();
}

void PomadeBakeWorker::_WorkerLoop()
{
#ifdef USDGEN_POMADE_HAS_CUDA
    cudaStream_t bakeStream = _BakeStream();  // lowest priority, may be null
#else
    void *bakeStream = nullptr;
#endif
    auto takeSwapNote = [&] {
        if (usdGen::UsdGenRef<_Lane::Completion> note = _lane->swapped.Take()) {
            _swappedVersion = note->version;
            _swappedPath = std::move(note->path);
            return true;
        }
        return false;
    };
    for (;;) {
        uint32_t const seen = _lane->wake.load(std::memory_order_seq_cst);
        if (_stop.load(std::memory_order_seq_cst)) {
            break;
        }
        usdGen::UsdGenRef<_Lane::Request> request;
        if (!_paused.load(std::memory_order_seq_cst)) {
            request = _lane->pending.Take();
            if (!request) {
                if (takeSwapNote()) {
                    // A swap freed an older versioned file and no bake
                    // follows: sweep and go back to sleep (plan/18 §7 G5).
                    if (!_lastOutDir.empty()) {
                        _SweepStale(_lastOutDir, _lastBaseName);
                    }
                    continue;
                }
            }
        }
        if (!request) {
            usdGen::UsdGenParkWhileEqual(_lane->wake, seen);
            continue;
        }
        // Coalescing: the mailbox holds only the latest enqueue. A swap
        // note taken here is served by the sweep after this bake.
        takeSwapNote();
        uint64_t const version = request->version;
        PomadeBakeInput const &input = request->input;
        _lastOutDir = input.outDir;
        _lastBaseName = input.baseName;
        std::string outPath;
        std::string err;
        bool ok = false;
        try {
            if (int const pending = _throwBakes.load(); pending > 0) {
                _throwBakes.store(pending - 1);
                throw std::runtime_error(
                    "PomadeBakeWorker: injected bake failure (test hook)");
            }
            ok = _Bake(version, input, &outPath, &err,
#ifdef USDGEN_POMADE_HAS_CUDA
                       bakeStream
#else
                       nullptr
#endif
            );
        } catch (std::exception const &e) {
            // §3.4: the map keeps its previous file, the model is untouched
            // and this thread survives to serve the next enqueue.
            _workerThrowCount.fetch_add(1);
            ok = false;
            err = std::string("PomadeBakeWorker: worker threw: ") + e.what();
        } catch (...) {
            _workerThrowCount.fetch_add(1);
            ok = false;
            err = "PomadeBakeWorker: worker threw an unknown exception";
        }
        if (!ok) {
            if (err.empty()) {
                // Cancelled at a tile boundary: no file, no completion.
                _cancelCount.fetch_add(1);
            } else {
                auto d = usdGen::UsdGenMakeRef<_Lane::Diagnostic>();
                d->text = std::move(err);
                _lane->diagnostic.Publish(std::move(d));
            }
            continue;  // a newer version may be waiting in the mailbox
        }
        _bakeCount.fetch_add(1);
        // The previous completion is still in the mailbox (never handed
        // out; this publish would displace it) or the UI took it and may
        // not have swapped to it yet: then the sweep must keep its file.
        // The exchange decides which, so the UI cannot take it after this.
        if (!_lane->completed.Take()) {
            _takenVersion = _completedVersion;
            _takenPath = std::move(_completedPath);
        }
        _completedVersion = version;
        _completedPath = outPath;
        auto done = usdGen::UsdGenMakeRef<_Lane::Completion>();
        done->version = version;
        done->path = outPath;
        _lane->completed.Publish(std::move(done));
        _completedVersionAtomic.store(version);
        _Lane::Notify(_lane->completedWord);
        takeSwapNote();
        _SweepStale(input.outDir, input.baseName);
    }
#ifdef USDGEN_POMADE_HAS_CUDA
    if (bakeStream) {
        cudaStreamDestroy(bakeStream);
    }
#else
    (void)bakeStream;
#endif
}

void PomadeBakeWorker::_SweepStale(std::string const &outDir,
                                  std::string const &baseName)
{
    // Keep the file the live layer references, the latest completion, and
    // the completion before it when the UI took it after that swap (it may
    // be swapping to it right now); delete older versioned files. Runs on
    // the worker only, so the kept paths are the worker's own copies.
    std::error_code ec;
    if (outDir.empty() || !fs::is_directory(outDir, ec)) {
        return;
    }
    bool const keepTaken =
        !_takenPath.empty() && _takenVersion > _swappedVersion;
    std::string const prefix = baseName + ".v";
    for (auto const &entry : fs::directory_iterator(outDir, ec)) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        std::string const name = entry.path().filename().string();
        if (name.size() <= prefix.size() + 4 ||
            name.compare(0, prefix.size(), prefix) != 0 ||
            name.compare(name.size() - 4, 4, ".ptx") != 0) {
            continue;
        }
        std::string const full = entry.path().string();
        if (full == _swappedPath || full == _completedPath ||
            (keepTaken && full == _takenPath)) {
            continue;
        }
        unsigned long long v = 0;
        if (std::sscanf(name.c_str() + prefix.size(), "%llu", &v) == 1 &&
            v < _completedVersionAtomic.load()) {
            fs::remove(entry.path(), ec);
        }
    }
    _sweepCount.fetch_add(1);
}

void PomadeBakeWorker::ThrowOnNextBakesForTest(int count)
{
    _throwBakes.store(count < 0 ? 0 : count);
}

bool PomadeBakeWorker::_Bake(uint64_t mapVersion, PomadeBakeInput const &input,
                            std::string *outPath, std::string *err,
                            void *bakeStream)
{
    auto fail = [&](std::string const &what) {
        if (err) {
            *err = "PomadeBakeWorker: " + what;
        }
        return false;
    };
    // Cancelled bakes report success=false with an EMPTY error (the loop
    // counts them as cancellations, not failures).
    auto cancelled = [&] {
        return _stop.load(std::memory_order_acquire) ||
               (_lane->pending.HasValue() &&
                _pendingVersionAtomic.load() > mapVersion);
    };
    if (!input.scalp || !input.scalp->finalized) {
        return fail("bad scalp input");
    }
    if (input.outDir.empty()) {
        return fail("no output directory");
    }
    std::error_code ec;
    fs::create_directories(input.outDir, ec);
    if (ec) {
        return fail("cannot create " + input.outDir);
    }
    PomadeScalpMesh const &mesh = *input.scalp;
    size_t const faceCount = mesh.faceVertexCounts.size();
    PomadeRegionLoops loops;
    if (!PomadeFlattenLoops(input.graph, &loops, err)) {
        return false;
    }
    PomadeRegionMaps maps;
    if (!PomadeRasteriseRegionsCpu(mesh, input.graph, &maps, err)) {
        return false;
    }
    PomadeRegionLoops const ownerLoops = input.outputOwnerMap
        ? _OutputOwnerLoops(loops) : PomadeRegionLoops();
    PomadeRegionLoops const &classifierLoops = input.outputOwnerMap
        ? ownerLoops : loops;
    // Incremental dirty set: labels alone are not enough because a boundary
    // can move within a face while its centroid classification stays fixed.
    // A changed classifier key invalidates all faces; exact repeats reuse the
    // texel cache without work.
    std::vector<int> const resLog2 =
        PomadeFaceResLog2(mesh, maps, loops, input.resOverride);
    _BakePlan plan;
    if (!_PlanFaces(input, maps, resLog2, &plan, err)) {
        return false;
    }
    uint64_t const classifierKey = _ClassifierKey(input, loops);
    std::string const cacheIdentity =
        _CacheIdentity(input.outDir, classifierKey);
    bool full = _cachedFaceRegion.size() != faceCount ||
                _cacheDir != cacheIdentity ||
                _texelCache.size() != plan.faces.size();
    std::vector<char> isDirty(faceCount, full ? 1 : 0);
    if (!full) {
        for (size_t f = 0; f < faceCount; ++f) {
            if (_cachedFaceRegion[f] != maps.faceRegion[f]) {
                isDirty[f] = 1;
            }
        }
    } else if (_texelCache.size() != plan.faces.size()) {
        _texelCache.assign(plan.faces.size(), {});
    }
    _HealStaleCache(plan, _texelCache, &isDirty);
    if (cancelled()) {
        if (err) {
            err->clear();
        }
        return false;
    }
    // Chunked classification at ~4K-face tile boundaries with a supersede
    // check between chunks (plan/17 §3.1a cancellation).
    _LevelIndex const levels = _BuildLevelIndex(input, classifierLoops);
    size_t const kChunkFaces = 4096;
    PomadeBakeStats stats;
    stats.ptexFaces = plan.faces.size();
    stats.channels = plan.channels;
    for (size_t begin = 0; begin < faceCount; begin += kChunkFaces) {
        size_t const end = std::min(begin + kChunkFaces, faceCount);
        std::vector<char> chunkDirty(faceCount, 0);
        bool anyDirty = false;
        for (size_t f = begin; f < end; ++f) {
            if (isDirty[f]) {
                chunkDirty[f] = 1;
                anyDirty = true;
            }
        }
        if (anyDirty) {
#ifdef USDGEN_POMADE_HAS_CUDA
            // The GPU lane classifies this chunk on the bake stream; any
            // failure falls back to the CPU twin for the chunk (never a
            // failed bake: §3.4 owns the same fallback for interaction).
            std::string gpuErr;
            if (bakeStream == nullptr ||
                !_ClassifyDirtyGpu(classifierLoops, plan, levels, chunkDirty,
                                   &_texelCache, (cudaStream_t)bakeStream,
                                   &stats, &gpuErr)) {
                _ClassifyDirtyCpu(classifierLoops, plan, levels, chunkDirty,
                                  &_texelCache, &stats);
            }
#else
            (void)bakeStream;
            _ClassifyDirtyCpu(classifierLoops, plan, levels, chunkDirty, &_texelCache,
                              &stats);
#endif
        }
        if (cancelled()) {
            if (err) {
                err->clear();
            }
            return false;
        }
    }
    std::string const finalPath =
        (fs::path(input.outDir) / PomadeBakeFileName(input.baseName, mapVersion))
            .string();
    if (!_WritePtexFile(plan, _texelCache, finalPath, err)) {
        return false;
    }
    if (cancelled()) {
        // A newer version arrived during the file write: the versioned file
        // is complete and harmless on disk, but it is NOT reported, so the
        // UI never swaps a superseded bake (a stale file is never referenced).
        fs::remove(finalPath, ec);
        if (err) {
            err->clear();
        }
        return false;
    }
    _cachedFaceRegion = maps.faceRegion;
    _cacheDir = cacheIdentity;
    _lastBakedFaces.store(stats.facesClassified);
    if (outPath) {
        *outPath = finalPath;
    }
    return true;
}

}  // namespace usdGenPomade
