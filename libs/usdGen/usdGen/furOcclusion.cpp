#include "usdGen/furOcclusion.h"

#include "usdGen/debugCodes.h"
#include "usdGen/digest.h"
#include "usdGen/scheduler.h"

#include "pxr/base/gf/matrix4f.h"
#include "pxr/base/trace/trace.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace usdGen {
namespace {
constexpr char const* names[] = {"furTauP", "furTauN"};
// Samples per voxel along a curve; the trilinear splat carries the rest, and
// the deposited area is the sub-sample length, so this changes where density
// lands, never how much of it there is.
constexpr float kSamplesPerVoxel = 1.0f;
constexpr int   kMaxSegmentSteps = 512;
// Published depth is gathered from a fixed-point grid, as a host renderer's voxel pages
// are (24-bit hair count at x1000): half the bytes of the eight cells every CV
// touches, and the same quantisation the engine it is matched to uses.
constexpr float kDepthScale = 1000.f;
// The legacy cube resolution, kept as the coarsest voxel the derived grid may
// pick so a small swatch never loses detail to the world-size target.
constexpr int   kLegacyResolution = 48;
// One tile is far too coarse and far too uneven a unit of work for a groom
// whose tiles differ by an order of magnitude in curve count.
constexpr uint32_t kJobCurves = 4096;
constexpr size_t   kJobCvs    = 16384;

bool Reserved(TfToken const& name) {
    for (auto n : names) if (name == n) return true;
    return false;
}

struct Grid {
    int n[3] = {0, 0, 0};
    GfVec3f origin;
    float h = 1.f, invH = 1.f;
    size_t Index(int x, int y, int z) const { return (size_t(z)*n[1]+y)*n[0]+x; }
    size_t Cells() const { return size_t(n[0])*n[1]*n[2]; }
    // The eight cells and trilinear weights around p. Gather and scatter use
    // precisely the same cell-centre convention.
    void Corners(GfVec3f const& p, int base[3], float frac[3]) const {
        for (int k=0;k<3;++k) {
            float q=std::clamp((p[k]-origin[k])*invH-0.5f,0.f,float(n[k]-1));
            base[k]=std::min(int(q),n[k]-2); frac[k]=q-base[k];
        }
    }
    bool Contains(GfVec3f const& p) const {
        for (int k=0;k<3;++k) {
            float const q=(p[k]-origin[k])*invH;
            if (!(q >= 0.f && q <= float(n[k]))) return false;
        }
        return true;
    }
};

// Uniform cubic basis weights. Both bases are evaluated with Hydra's pinned
// convention: the end control points are extrapolated (2*P0 - P1) so the curve
// interpolates the first and last CV, which is the geometry Storm draws.
inline void CubicWeights(bool catmullRom, float u, float w[4]) {
    float const u2 = u*u, u3 = u2*u;
    if (catmullRom) {
        w[0] = 0.5f*(-u3 + 2*u2 - u);
        w[1] = 0.5f*(3*u3 - 5*u2 + 2);
        w[2] = 0.5f*(-3*u3 + 4*u2 + u);
        w[3] = 0.5f*(u3 - u2);
    } else {
        float const v = 1.f-u;
        w[0] = v*v*v*(1.f/6.f);
        w[1] = (3*u3 - 6*u2 + 4)*(1.f/6.f);
        w[2] = (-3*u3 + 3*u2 + 3*u + 1)*(1.f/6.f);
        w[3] = u3*(1.f/6.f);
    }
}

// Runs body(i) for i in [0, count) on the dispatcher's arena, or serially.
template <class F>
void ForEach(UsdGenWorkDispatcher* dispatcher, size_t count, F const& body) {
    if (!dispatcher || count < 2) {
        for (size_t i = 0; i < count; ++i) body(i);
        return;
    }
    struct Payload { F const* body; } payload{&body};
    dispatcher->ParallelFor(count, [](size_t i, void* p) {
        (*static_cast<Payload*>(p)->body)(i);
    }, &payload);
}

/// A contiguous run of curves of one tile: the unit of parallel work for both
/// the splat and the gather.
struct Job {
    uint32_t tile = 0;
    uint32_t firstCurve = 0, curveCount = 0;
    size_t   firstCv = 0, cvCount = 0;
};

// A job's contribution (UsdGenSplatJobDensity, furOcclusion.h) covers the
// sub-box of cells its samples touch; jobs are splatted in parallel and
// summed over disjoint slabs.

// The world-space mesh lives in UsdGenFurOccluderTriangle (furOcclusion.h)
// so a caller-owned build cache can carry it across cooks.

// Triangles with less hair than this over them are dropped, so bare skin is
// untouched and the translucent pass stays small. It has to be small: dropping
// a triangle is a step in the shadow, and transmittance falls steeply from
// zero crossings (a_f is around 0.2 for a dark coat, so 0.05 crossings is
// already 6% of the light), which at a face's size reads as a hard polygonal
// edge across the skin. At 0.002 the step is a quarter of one 8-bit level.
constexpr float kScalpShadowFloor = 0.002f;
// The taps the scalp shader weights the sky with: the normal plus a ring at
// 60 degrees. The cull below has to ask the same question the shader will.
constexpr int   kScalpRing = 4;
constexpr float kScalpRingCos = 0.5f;

/// The consumer's reconstruction, mirrored: a normalised cosine-power-4 blend
/// of the six stored depths. `values` is +X,-X,+Y,-Y,+Z,-Z.
inline float DirectionalDepth(float const values[6], GfVec3f const& dir)
{
    float sum=0, weighted=0;
    for(int k=0;k<3;++k) {
        float w=dir[k]*dir[k]; w*=w;
        weighted+=w*(dir[k]>=0.f?values[k*2]:values[k*2+1]);
        sum+=w;
    }
    return sum>0.f ? weighted/sum : 0.f;
}

/// How much hair a cosine-weighted sky sees from a point with normal `n`. This
/// is what decides whether a patch of skin is under hair at all: the depth
/// toward any one axis is not, because for a point anywhere on a head the
/// horizontal axes run through the hair mass on the far side.
inline float HemisphereDepth(float const values[6], GfVec3f const& n)
{
    GfVec3f up(0,0,1);
    if(std::fabs(n[2])>0.9f) up=GfVec3f(1,0,0);
    GfVec3f t=GfCross(n,up);
    float length=t.GetLength();
    t = length>1e-6f ? t/length : GfVec3f(1,0,0);
    GfVec3f const b=GfCross(n,t);
    float const sinRing=std::sqrt(std::max(0.f,1.f-kScalpRingCos*kScalpRingCos));
    float weighted=DirectionalDepth(values,n), sum=1.f;
    for(int tap=0;tap<kScalpRing;++tap) {
        float const phi=float(tap)*(6.2831853f/float(kScalpRing));
        GfVec3f const dir=n*kScalpRingCos+
            (t*std::cos(phi)+b*std::sin(phi))*sinRing;
        weighted+=kScalpRingCos*DirectionalDepth(values,dir);
        sum+=kScalpRingCos;
    }
    return weighted/sum;
}

/// Trilinear gather of the hair-only sweep's six depths at one point, with
/// same-cell reuse across a triangle's corners and across consecutive
/// triangles: scalp triangles are far smaller than a voxel, so the corners
/// usually share one cell and consecutive triangles usually share it too, and
/// the eight cell x six depth loads happen once per run of same-cell corners
/// instead of once per corner. Each corner still runs its own Corners() for
/// its weights, and the accumulation visits the same cells in the same order,
/// so reused corners are bit-identical. One instance per chunk, carried across
/// its triangles (the grid and sweep are fixed for the whole build, so a
/// stale entry is impossible); never shared across threads.
struct ScalpDepthGather {
    Grid const* grid = nullptr;
    std::array<std::vector<float>,6> const* tau = nullptr;
    int cell[3] = {0, 0, 0};
    float cached[8][6] = {};
    bool have = false;
    void operator()(GfVec3f const& at, float values[6]) {
        int c[3]; float w[3];
        grid->Corners(at,c,w);
        if(!have || c[0]!=cell[0] || c[1]!=cell[1] || c[2]!=cell[2]) {
            cell[0]=c[0]; cell[1]=c[1]; cell[2]=c[2];
            for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
                size_t const idx=grid->Index(c[0]+x,c[1]+y,c[2]+z);
                float* const slot=cached[(z*2+y)*2+x];
                for(int d=0;d<6;++d) slot[d]=(*tau)[d][idx];
            }
            have=true;
        }
        for(int d=0;d<6;++d) values[d]=0.f;
        for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
            float const weight=(x?w[0]:1-w[0])*(y?w[1]:1-w[1])*(z?w[2]:1-w[2]);
            float const* const slot=cached[(z*2+y)*2+x];
            for(int d=0;d<6;++d) values[d]+=weight*slot[d];
        }
    }
};

/// Incremental 4-lane word feed: digest.h's UsdGenDigestBytes over bytes
/// that arrive in pieces (a merged slab concatenated from chunk outputs).
/// Words group from the slab's first byte exactly as the contiguous call
/// groups them (a partial word carries into the next piece), lanes assign
/// from the running word count, and the trailing bytes feed lane-rotated as
/// the contiguous call's tail does, so one FeedCopy per piece plus Finish
/// equals UsdGenDigestBytes over the concatenated slab, bit for bit.
struct _SlabHash {
    uint64_t h[4];
    uint64_t nWords = 0;
    unsigned char carry[8];
    size_t nCarry = 0;
    explicit _SlabHash(uint64_t seed) {
        h[0] = seed;
        h[1] = seed ^ 0x9E3779B97F4A7C15ull;
        h[2] = seed ^ 0xBF58476D1CE4E5B9ull;
        h[3] = seed ^ 0x94D049BB133111EBull;
    }
    // Copy n bytes while feeding them (dst null skips the copy: a carried
    // array hashes from its chunks without touching the shared merged array).
    void FeedCopy(unsigned char* dst, unsigned char const* src, size_t n) {
        size_t i = 0;
        if (nCarry) {
            while (nCarry < 8 && i < n) {
                unsigned char const v = src[i];
                if (dst) dst[i] = v;
                carry[nCarry++] = v;
                ++i;
            }
            if (nCarry == 8) {
                uint64_t w;
                std::memcpy(&w, carry, 8);
                h[nWords & 3] ^= w;
                h[nWords & 3] *= UsdGenDigestPrime;
                ++nWords;
                nCarry = 0;
            } else return;
        }
        for (; i + 8 <= n; i += 8) {
            uint64_t w;
            std::memcpy(&w, src + i, 8);
            if (dst) std::memcpy(dst + i, &w, 8);
            h[nWords & 3] ^= w;
            h[nWords & 3] *= UsdGenDigestPrime;
            ++nWords;
        }
        // Fewer than 8 bytes remain and the carry is empty (the head
        // flushed a whole word or returned), so the nCarry < 8 bound never
        // binds: every remaining byte lands in the carry.
        for (; i < n && nCarry < 8; ++i) {
            unsigned char const v = src[i];
            if (dst) dst[i] = v;
            carry[nCarry++] = v;
        }
    }
    uint64_t Finish() {
        size_t lane = size_t(nWords & 3);
        for (size_t i = 0; i < nCarry; ++i) {
            h[lane] ^= uint64_t(carry[i]);
            h[lane] *= UsdGenDigestPrime;
            lane = (lane + 1) & 3;
        }
        nCarry = 0;
        return UsdGenDigestCombine4(h[0], h[1], h[2], h[3]);
    }
};

/// The scalp-shadow cap. Each occluder triangle is refined k x k, every
/// sub-vertex takes the hair-only depth one voxel out along the surface normal
/// (far enough that the opaque shell two voxels IN cannot reach it), and
/// sub-triangles with no hair over them are dropped.
///
/// One k for the whole mesh, so shared edges are split identically and the cap
/// has no T-junctions: adaptive per face would crack.
template <class Gather>
void BuildScalpShadow(std::vector<UsdGenFurOccluderTriangle> const& triangles,
                      std::vector<GfVec3f> const& vertexNormals,
                      std::vector<float> const& extents, float inward,
                      Grid const& grid, int maxTriangles, Gather const& gather,
                      UsdGenWorkDispatcher* dispatcher,
                      UsdGenScalpShadowPublication* out,
                      uint64_t occluderKey,
                      UsdGenScalpShadowScratch* scratch = nullptr)
{
    // The caller seeds `out` with the previous generation's cap, so a rebuild
    // whose lit set and occluder match can share the previous merged arrays
    // (VtArray CoW) instead of merging fresh ones: the drain and the renderer
    // then see IsIdentical instead of a ~320MB memcmp and re-upload. Snapshot
    // the seed now; every carry below is size-checked and fails closed.
    bool const havePrev = !out->IsEmpty();
    uint64_t const prevTopo = out->topologyKey;
    uint64_t const prevOcc = out->occluderKey;
    float const prevLift = out->lift;
    float const prevInward = out->inward;
    int const prevLevel = out->tessLevel;
    VtVec3fArray const prevPoints = out->points;
    VtVec3fArray const prevNormals = out->normals;
    VtIntArray const prevCounts = out->faceVertexCounts;
    VtIntArray const prevIndices = out->faceVertexIndices;
    // The 90th-percentile edge decides the level, so one oversized face cannot
    // refine the whole mesh; the triangle ceiling then bounds it outright.
    // The ceiling forces k==1 outright whenever the mesh alone exceeds it
    // (every benchmark groom: millions of triangles against a 200k ceiling),
    // whatever the edge distribution says, so the percentile pass is dead
    // there: skip the copy and the nth_element. Small meshes run it as
    // before. Either way k is exactly what the old code computed.
    int k = 1;
    int const ceiling=std::max(1,maxTriangles);
    { TRACE_SCOPE("usdGen scalp: edges");
    if(triangles.size() <= size_t(ceiling)) {
        std::vector<float> edges;
        edges.reserve(triangles.size());
        for(size_t i=0;i<triangles.size();++i)
            edges.push_back(extents[i]);
        auto at=edges.begin()+std::min(edges.size()-1,size_t(0.9*double(edges.size())));
        std::nth_element(edges.begin(),at,edges.end());
        k=std::max(1,int(std::ceil(*at*grid.invH)));
        while(k>1 && triangles.size()*size_t(k)*size_t(k) > size_t(ceiling)) --k;
    }
    }

    float const outward=-inward;
    // Sample where the cap is DRAWN, not a voxel out along the normal. The
    // outward probe this used to take was guarding against the opaque shell
    // two voxels in -- but the cap is gathered from the HAIR-ONLY sweep, run
    // before the shell is injected, so there was nothing there to avoid. What
    // it did instead was skip the voxel the roots sit in, which is the densest
    // one, and cost the cap almost exactly one crossing: measured in-render
    // against a strand CV at the same point and direction, the cap read 1.53
    // where the strand read 2.51, i.e. the strand's count after a host renderer's
    // self-shift -- the very shift a skin receiver must not apply.
    float const probe=0.f;
    // A hair's breadth of separation, so the cap wins the depth test against
    // the skin it sits on. glslfx has no polygon offset, and at this scale the
    // margin is still an order of magnitude above the depth buffer's
    // resolution. Kept small because the lift also pushes the cap's silhouette
    // outside the skin's; the shader fades it at grazing angles for the rest.
    float const lift=0.06f*grid.h;
    size_t const perTriangle=size_t(k+1)*size_t(k+2)/2;

    // Triangles are independent work: shared edges are split identically but
    // vertices are never shared across triangles, so the build fans out over
    // contiguous triangle chunks with thread-local outputs, merged in chunk
    // order. The merged arrays match the serial emission order exactly, and
    // no floating-point value crosses a chunk boundary, so the cap, extent,
    // and digest are bit-identical to the serial build.
    //
    // With caller-owned scratch the chunk outputs persist across cooks: the
    // same capacities are refilled every frame instead of re-faulted, and
    // the emission below is unchanged so the merged cap is bit-identical.
    // Without scratch the outputs are function-local, as historically.
    using ChunkOut = UsdGenScalpShadowScratch::Chunk;
    size_t const nChunks = triangles.empty() ? 1 : std::max<size_t>(1,
        std::min<size_t>(triangles.size(), size_t(dispatcher ?
            std::max(1, dispatcher->MaxConcurrency()) : 1)));
    std::vector<ChunkOut> localChunks;
    if (scratch) {
        scratch->chunks.resize(nChunks);
        for (ChunkOut& c : scratch->chunks) {
            c.points.clear(); c.normals.clear();
            c.tauP.clear(); c.tauN.clear();
            c.counts.clear(); c.indices.clear();
        }
    } else {
        localChunks.resize(nChunks);
    }
    std::vector<ChunkOut>& chunks = scratch ? scratch->chunks : localChunks;
    // The lit-set key: every emitted face mixes its triangle (and, past k=1,
    // its candidate ordinal) in emission order, combined in chunk order after
    // the refine. One integer multiply per face; the merged topology is a pure
    // function of this key and k, so a match carries counts and indices.
    std::vector<uint64_t> chunkTopo(nChunks, UsdGenDigestOffset);
    // Per-chunk cap extrema, tracked during emission below and reduced in
    // chunk order for the published extent, so the extent needs no second
    // pass over the merged points.
    float const kInf = std::numeric_limits<float>::max();
    std::vector<GfVec3f> chunkLo(nChunks, GfVec3f(kInf)),
        chunkHi(nChunks, GfVec3f(-kInf));
    { TRACE_SCOPE("usdGen scalp: refine");
    ForEach(dispatcher, nChunks, [&](size_t chunk) {
        size_t const first=(chunk*triangles.size())/nChunks;
        size_t const last=((chunk+1)*triangles.size())/nChunks;
        ChunkOut& c=chunks[chunk];
        uint64_t topo = UsdGenDigestOffset;
        GfVec3f lo(kInf), hi(-kInf);
        // The gather carries same-cell state, so each chunk works on its own
        // copy, carried across its triangles; the merged cap is independent
        // of how the mesh was chunked.
        Gather chunkGather=gather;
        // Every chunk output has a tight upper bound: one point and normal
        // per lattice vertex, three tau floats per point, and at most k*k
        // sub-faces (one count, three indices each) per triangle. Without
        // these the chunk arrays re-copy ~1GB per cook through push_back
        // growth; reserve() changes capacity only, so the cap is
        // bit-identical.
        size_t const nTri=last-first;
        c.points.reserve(nTri*perTriangle);
        c.normals.reserve(nTri*perTriangle);
        c.tauP.reserve(nTri*perTriangle*3);
        c.tauN.reserve(nTri*perTriangle*3);
        c.counts.reserve(nTri*size_t(k)*size_t(k));
        c.indices.reserve(nTri*size_t(k)*size_t(k)*3);
        // At k==1 the emission below writes through raw pointers into the
        // reserved capacity and publishes the sizes once per chunk after
        // the triangle loop: six VtArray::resize calls per lit triangle
        // (size check, uniqueness check, size store each) cost more than
        // the ~20 stores they frame. data() detaches when shared, so the
        // pointers are uniquely owned; nothing between here and the
        // closing resizes can reallocate (the k>1 push_back path never
        // runs in a k==1 build). Same values in the same order as the
        // per-triangle resizes: bit-identical.
        GfVec3f* emitPts=nullptr; GfVec3f* emitNrm=nullptr;
        float* emitTauP=nullptr; float* emitTauN=nullptr;
        int* emitCounts=nullptr; int* emitIndices=nullptr;
        size_t nEmitPts=0, nEmitTau=0, nEmitCounts=0, nEmitIndices=0;
        if(k==1) {
            emitPts=c.points.data(); emitNrm=c.normals.data();
            emitTauP=c.tauP.data(); emitTauN=c.tauN.data();
            emitCounts=c.counts.data(); emitIndices=c.indices.data();
        }
        std::vector<GfVec3f> local(perTriangle), localN(perTriangle);
        std::vector<float> depth(perTriangle*6), shaded(perTriangle);
        std::vector<int> remap(perTriangle);
        // Memo epoch for the lazy coverage below: one int per lattice slot,
        // stamped per triangle, so skipped corners cost no fill.
        std::vector<int> shadedEpoch(perTriangle, 0);
        int shadedClock = 0;
        for(size_t ti=first;ti<last;++ti) {
            UsdGenFurOccluderTriangle const& tri=triangles[ti];
            GfVec3f const na=vertexNormals[tri.ia]*outward;
            GfVec3f const nb=vertexNormals[tri.ib]*outward;
            GfVec3f const nc=vertexNormals[tri.ic]*outward;
            GfVec3f const flat=tri.normal*outward;
            GfVec3f const e1=tri.b-tri.a, e2=tri.c-tri.a;
            size_t index=0;
            for(int i=0;i<=k;++i) for(int j=0;i+j<=k;++j,++index) {
                float const u=float(i)/float(k), v=float(j)/float(k);
                GfVec3f const on=tri.a+e1*u+e2*v;
                GfVec3f n=na*(1.f-u-v)+nb*u+nc*v;
                float const length=n.GetLength();
                n = length>1e-6f ? n/length : flat;
                local[index]=on; localN[index]=n;
                chunkGather(on+n*(probe+lift),&depth[index*6]);
            }
            // Coverage is read only through OR-reductions (the triangle-wide
            // `any` and each face's lit check), so corners are evaluated
            // lazily, at most once each, with short-circuit. HemisphereDepth
            // is a pure function of the corner's gathered depths and normal,
            // and skipped corners are never read, so the cap is
            // bit-identical; the lazy loop never evaluates more corners than
            // the eager one. With a lit first corner at k=1 this is one
            // evaluation per triangle instead of three.
            ++shadedClock;
            auto coverage=[&](int s)->float {
                if(shadedEpoch[size_t(s)]!=shadedClock) {
                    shadedEpoch[size_t(s)]=shadedClock;
                    shaded[size_t(s)]=HemisphereDepth(&depth[size_t(s)*6],
                                                     localN[size_t(s)]);
                }
                return shaded[size_t(s)];
            };
            bool any=false;
            for(size_t s=0;s<perTriangle;++s)
                if(coverage(int(s))>kScalpShadowFloor) { any=true; break; }
            if(!any) continue;
            if(k==1) {
                // The ceiling forces k==1 whenever the mesh exceeds it (every
                // benchmark groom), and then the general machinery below has
                // exactly one outcome: one face over slots {0,2,1}, lit iff
                // `any`, emitting its three corners in visit order with
                // indices base, base+1, base+2 and one count of 3. The slot
                // arithmetic, remap fill, per-corner memo checks, and 28
                // capacity-checked push_backs per triangle collapse into
                // direct stores of the same expressions in the same order
                // through the chunk's raw pointers; the sizes publish once
                // per chunk after the loop: bit-identical.
                int const order[3]={0,2,1};
                for(int e=0;e<3;++e) {
                    int const s=order[e];
                    GfVec3f const pt=local[size_t(s)]+localN[size_t(s)]*lift;
                    emitPts[nEmitPts+size_t(e)]=pt;
                    for(int d=0;d<3;++d) {
                        lo[d]=std::min(lo[d],pt[d]); hi[d]=std::max(hi[d],pt[d]);
                    }
                    emitNrm[nEmitPts+size_t(e)]=localN[size_t(s)];
                    for(int d=0;d<3;++d) {
                        emitTauP[nEmitTau+size_t(e)*3+size_t(d)]=
                            depth[size_t(s)*6+size_t(d)*2];
                        emitTauN[nEmitTau+size_t(e)*3+size_t(d)]=
                            depth[size_t(s)*6+size_t(d)*2+1];
                    }
                    emitIndices[nEmitIndices+size_t(e)]=int(nEmitPts+size_t(e));
                }
                emitCounts[nEmitCounts]=3;
                nEmitPts+=3; nEmitTau+=9; nEmitCounts+=1; nEmitIndices+=3;
                UsdGenDigestMixWord(topo, uint64_t(ti));
                continue;
            }
            // (i, j) -> the lattice slot, walking i in rows of decreasing length.
            auto slot=[&](int i,int j) {
                int base=0;
                for(int r=0;r<i;++r) base+=k+1-r;
                return base+j;
            };
            std::fill(remap.begin(),remap.end(),-1);
            auto emit=[&](int i,int j)->int {
                int const s=slot(i,j);
                if(remap[size_t(s)]<0) {
                    remap[size_t(s)]=int(c.points.size());
                    GfVec3f const pt=local[size_t(s)]+localN[size_t(s)]*lift;
                    c.points.push_back(pt);
                    for(int d=0;d<3;++d) {
                        lo[d]=std::min(lo[d],pt[d]); hi[d]=std::max(hi[d],pt[d]);
                    }
                    c.normals.push_back(localN[size_t(s)]);
                    for(int d=0;d<3;++d) {
                        c.tauP.push_back(depth[size_t(s)*6+d*2]);
                        c.tauN.push_back(depth[size_t(s)*6+d*2+1]);
                    }
                }
                return remap[size_t(s)];
            };
            // The candidate ordinal, not the emitted one: two different lit
            // sets can emit the same number of faces per triangle.
            int faceCand = 0;
            auto face=[&](int i0,int j0,int i1,int j1,int i2,int j2) {
                int const cand = faceCand++;
                int const t0=slot(i0,j0), t1=slot(i1,j1), t2=slot(i2,j2);
                bool lit=false;
                for(int s:{t0,t1,t2})
                    if(coverage(s)>kScalpShadowFloor) { lit=true; break; }
                if(!lit) return;
                UsdGenDigestMixWord(topo, (uint64_t(ti) << 32) | uint64_t(cand));
                c.counts.push_back(3);
                c.indices.push_back(emit(i0,j0));
                c.indices.push_back(emit(i1,j1));
                c.indices.push_back(emit(i2,j2));
            };
            for(int i=0;i<k;++i) for(int j=0;i+j<k;++j) {
                face(i,j, i+1,j, i,j+1);
                if(i+j+2<=k) face(i+1,j, i+1,j+1, i,j+1);
            }
        }
        if(k==1) {
            // Publish the raw-pointer emission above: capacity was reserved
            // up front and uniquely owned throughout, so these only set
            // sizes over the already-written elements.
            auto noInit=[](auto* b,auto* e) { (void)b; (void)e; };
            c.points.resize(nEmitPts,noInit);
            c.normals.resize(nEmitPts,noInit);
            c.tauP.resize(nEmitTau,noInit);
            c.tauN.resize(nEmitTau,noInit);
            c.counts.resize(nEmitCounts,noInit);
            c.indices.resize(nEmitIndices,noInit);
        }
        chunkTopo[chunk] = topo;
        chunkLo[chunk] = lo; chunkHi[chunk] = hi;
    });
    } /* scalp: refine */
    uint64_t const topoKey = UsdGenDigestBytes(chunkTopo.data(),
        chunkTopo.size()*sizeof(uint64_t), UsdGenDigestOffset);
    VtVec3fArray points, normals;
    VtFloatArray tauP, tauN;
    VtIntArray counts, indices;
    // The digest hashes five merged arrays in 16 fixed slabs each; four of
    // those slabs fill during the merge below (fused copy+hash), the index
    // slabs fill in the digest scope. The slab count is a constant, never
    // the worker count, so the same bytes hash identically under any
    // dispatcher or none.
    constexpr size_t kSlabsPerArray = 16;
    std::vector<uint64_t> sub(5*kSlabsPerArray);
    { TRACE_SCOPE("usdGen scalp: merge");
        size_t nPoints=0, nTau=0, nCounts=0, nIndices=0;
        for(ChunkOut const& c:chunks) {
            nPoints+=c.points.size(); nTau+=c.tauP.size();
            nCounts+=c.counts.size(); nIndices+=c.indices.size();
        }
        // The merged topology is a pure function of the lit set and k; the
        // normals add the occluder mesh and the solid side, and the points
        // add the lift (which drifts with the hair bounds even when the
        // scalp holds still). Matching keys share the seeded arrays; the
        // depths always merge fresh. Every carry is size-checked: a key
        // collision (or any gate skew) falls back to the fresh merge.
        bool const topoSame = havePrev && topoKey == prevTopo && k == prevLevel;
        bool const occSame = topoSame && occluderKey == prevOcc &&
            inward == prevInward;
        bool const carryTopo = topoSame &&
            nCounts == prevCounts.size() && nIndices == prevIndices.size();
        bool const carryNormals = occSame && nPoints == prevNormals.size();
        bool const carryPoints = occSame && lift == prevLift &&
            nPoints == prevPoints.size();
        // Every element of every merged array is overwritten below --
        // points/normals/tauP/tauN by the fused slab loop, counts by chunk
        // memcpys, indices by the rebase loop -- so value-filling ~320MB
        // first is pure waste. The no-op fill leaves the storage
        // uninitialized (all six element types are trivially
        // copyable/destructible); the merged bytes are bit-identical.
        auto noInit = [](auto* b, auto* e) { (void)b; (void)e; };
        if (carryPoints) points = prevPoints;
        else points.resize(nPoints, noInit);
        if (carryNormals) normals = prevNormals;
        else normals.resize(nPoints, noInit);
        tauP.resize(nTau, noInit); tauN.resize(nTau, noInit);
        if (carryTopo) { counts = prevCounts; indices = prevIndices; }
        else { counts.resize(nCounts, noInit); indices.resize(nIndices, noInit); }
        // Prefix offsets are serial and cheap; the copies then run over the
        // pool. Each worker writes disjoint ranges with the same bytes,
        // indices, and rebase arithmetic: bit-identical.
        size_t const nMerge=chunks.size();
        std::vector<size_t> mergePoints(nMerge),
            mergeCounts(nMerge), mergeIndices(nMerge);
        { size_t oP=0, oC=0, oI=0;
          for(size_t i=0;i<nMerge;++i) {
              mergePoints[i]=oP;
              mergeCounts[i]=oC; mergeIndices[i]=oI;
              oP+=chunks[i].points.size();
              oC+=chunks[i].counts.size(); oI+=chunks[i].indices.size();
          } }
        // Points, normals, and both depth planes copy from the chunk
        // outputs AND hash into the digest's slabs in one pass over the
        // chunk bytes: the digest's ~290MB re-read of the merged arrays
        // becomes a few ALU ops per loaded word, hidden under the copy's
        // own DRAM stalls. Slab boundaries, seeds, and the final
        // combination are exactly the digest's, so the digest value is
        // unchanged. Carried arrays skip the copy (their merged array is
        // the shared previous one; writing it would detach the CoW) but
        // still hash from their chunks, whose bytes are the bytes the
        // carried merge shares: same lit set, same lift, same chunking.
        // Counts are not hashed and stay on chunk memcpys; indices keep
        // the rebase loop plus a hash from the merged array, because a
        // slab cut can split one index's four bytes across two slabs.
        struct FusedPiece { size_t chunk, src, n; };  // byte offsets
        struct FusedSlab {
            int array; size_t sub; uint64_t seed; size_t dst;
            std::vector<FusedPiece> pieces;
        };
        unsigned char* const fusedDst[4] = {
            carryPoints ? nullptr : (unsigned char*)points.data(),
            carryNormals ? nullptr : (unsigned char*)normals.data(),
            (unsigned char*)tauP.data(), (unsigned char*)tauN.data() };
        size_t const fusedTotal[4] = {
            nPoints*sizeof(GfVec3f), nPoints*sizeof(GfVec3f),
            nTau*sizeof(float), nTau*sizeof(float) };
        size_t const fusedWhole[4] = { 0, 1, 3, 4 };  // digest ordinals
        auto chunkSpan = [&](size_t i, int which,
                             unsigned char const** b, size_t* n) {
            ChunkOut const& c = chunks[i];
            if (which == 0) {
                *b = (unsigned char const*)c.points.cdata();
                *n = c.points.size()*sizeof(GfVec3f);
            } else if (which == 1) {
                *b = (unsigned char const*)c.normals.cdata();
                *n = c.normals.size()*sizeof(GfVec3f);
            } else if (which == 2) {
                *b = (unsigned char const*)c.tauP.cdata();
                *n = c.tauP.size()*sizeof(float);
            } else {
                *b = (unsigned char const*)c.tauN.cdata();
                *n = c.tauN.size()*sizeof(float);
            }
            if (*n == 0) *b = nullptr;
        };
        std::vector<std::vector<unsigned char const*>> fusedBytes(
            4, std::vector<unsigned char const*>(nMerge, nullptr));
        std::vector<FusedSlab> fusedSlabs;
        fusedSlabs.reserve(4*kSlabsPerArray);
        for (int w = 0; w < 4; ++w) {
            std::vector<size_t> pre(nMerge+1, 0);
            for (size_t i = 0; i < nMerge; ++i) {
                unsigned char const* b; size_t n;
                chunkSpan(i, w, &b, &n);
                fusedBytes[w][i] = b;
                pre[i+1] = pre[i] + n;
            }
            for (size_t s = 0; s < kSlabsPerArray; ++s) {
                size_t const a = (s*fusedTotal[w])/kSlabsPerArray;
                size_t const b = ((s+1)*fusedTotal[w])/kSlabsPerArray;
                FusedSlab slab;
                slab.array = w;
                slab.sub = fusedWhole[w]*kSlabsPerArray + s;
                slab.seed = UsdGenDigestOffset ^
                    (uint64_t(slab.sub)*0x9E3779B97F4A7C15ull);
                slab.dst = a;
                for (size_t i = 0; i < nMerge && pre[i] < b; ++i) {
                    size_t const lo = std::max(a, pre[i]);
                    size_t const hi = std::min(b, pre[i+1]);
                    if (hi > lo)
                        slab.pieces.push_back({i, lo-pre[i], hi-lo});
                }
                fusedSlabs.push_back(std::move(slab));
            }
        }
        ForEach(dispatcher, fusedSlabs.size(), [&](size_t u) {
            FusedSlab const& s = fusedSlabs[u];
            unsigned char* const dstBase = fusedDst[s.array];
            unsigned char* dst = dstBase ? dstBase + s.dst : nullptr;
            _SlabHash h(s.seed);
            for (FusedPiece const& p : s.pieces) {
                unsigned char const* const src =
                    fusedBytes[s.array][p.chunk] + p.src;
                h.FeedCopy(dst, src, p.n);
                if (dst) dst += p.n;
            }
            sub[s.sub] = h.Finish();
        });
        int* const mergeCnt=carryTopo ? nullptr : counts.data();
        int* const mergeIdx=carryTopo ? nullptr : indices.data();
        ForEach(dispatcher, nMerge, [&](size_t i) {
            ChunkOut& c=chunks[i];
            if (carryTopo) return;
            if(!c.counts.empty())
                std::memcpy(mergeCnt+mergeCounts[i],c.counts.cdata(),
                            c.counts.size()*sizeof(int));
            int const base=int(mergePoints[i]);
            size_t const o=mergeIndices[i];
            for(size_t j=0;j<c.indices.size();++j)
                mergeIdx[o+j]=c.indices[j]+base;
        });
        // Without scratch the chunk outputs are freed here, before the
        // extent and digest, as historically; with scratch they persist for
        // the next cook (cleared on entry above).
        if (!scratch) for(ChunkOut& c:chunks) c=ChunkOut();
    }
    out->points=points; out->normals=normals;
    out->faceVertexCounts=counts; out->faceVertexIndices=indices;
    out->topologyKey=topoKey; out->occluderKey=occluderKey;
    out->lift=lift; out->inward=inward; out->tessLevel=k;
    out->extraUniform.clear();
    if(!points.empty()) {
        for(int which=0;which<2;++which) {
            UsdGenPlane plane;
            plane.name=TfToken(names[which]);
            plane.interpolation=TfToken("vertex");
            plane.type=TfToken("float");
            plane.arity=3;
            plane.f=which?tauN:tauP;
            out->extraUniform.push_back(std::move(plane));
        }
        { TRACE_SCOPE("usdGen scalp: extent");
        // Per-chunk extrema tracked during emission, merged in chunk order:
        // min/max are exact over the finite cap points, so the ordered
        // reduction matches the merged-points scan, with no second pass
        // over the ~72MB of points. Non-emitting chunks hold +/-max, which
        // never wins; the guard above guarantees at least one emitted point.
        GfVec3f lo=chunkLo[0], hi=chunkHi[0];
        for(size_t i=1;i<chunkLo.size();++i) for(int d=0;d<3;++d) {
            lo[d]=std::min(lo[d],chunkLo[i][d]);
            hi[d]=std::max(hi[d],chunkHi[i][d]);
        }
        out->extentMin=GfVec3d(lo); out->extentMax=GfVec3d(hi);
        }
    }
    // The cap digest is presentation identity (the scene index dirties the
    // prim on it alone). Points, normals, and both depth planes hashed
    // during the merge above, in the same slabs with the same seeds; only
    // the rebased indices hash here, from the merged array exactly as
    // before. Every input byte still flips its slab's lanes, so
    // sensitivity is kept, and the value is unchanged. The digest is
    // in-memory equality-only (stability/sensitivity tested, never golden).
    uint64_t digest = UsdGenDigestOffset;
    { TRACE_SCOPE("usdGen scalp: digest");
    struct Span { void const* data; size_t bytes; };
    Span const whole[1] = {
        {indices.cdata(), indices.size()*sizeof(int)},
    };
    struct Slab { void const* data; size_t bytes; uint64_t seed; };
    std::vector<Slab> slabs;
    slabs.reserve(kSlabsPerArray);
    for (Span const& a : whole) {
        for (size_t s = 0; s < kSlabsPerArray; ++s) {
            size_t const first = (s*a.bytes)/kSlabsPerArray;
            size_t const last = ((s+1)*a.bytes)/kSlabsPerArray;
            auto const* bytes = static_cast<unsigned char const*>(a.data);
            size_t const ordinal = 2*kSlabsPerArray + slabs.size();
            slabs.push_back({bytes ? bytes+first : nullptr, last-first,
                UsdGenDigestOffset ^
                    (uint64_t(ordinal)*0x9E3779B97F4A7C15ull)});
        }
    }
    ForEach(dispatcher, slabs.size(), [&](size_t i) {
        sub[2*kSlabsPerArray+i] =
            UsdGenDigestBytes(slabs[i].data, slabs[i].bytes, slabs[i].seed);
    });
    digest = UsdGenDigestBytes(sub.data(), sub.size()*sizeof(uint64_t),
                               UsdGenDigestOffset);
    }
    out->digest=digest;
    TF_DEBUG(USDGEN_FUR).Msg(
        "usdGen fur: scalp shadow %d x refinement, %zu points, %zu triangles\n",
        k, points.size(), counts.size());
}
}

bool UsdGenBuildFurOcclusion(std::vector<UsdGenTilePublication>* tiles,
    std::vector<UsdGenTilePublication> const* previous,
    UsdGenFurOcclusionParams const& params, uint64_t* volumeKey,
    UsdGenScalpShadowPublication* scalpShadow)
{
    TRACE_FUNCTION();
    if (!tiles) return false;

    // --- volume identity: everything `previous` cannot show ------------------
    // Same coverage and feed order as the byte-at-a-time FNV-1a this
    // replaces, hashed with the shared 4-lane word mixer whose contract
    // names cook digests explicitly. Values change; the key is in-memory
    // equality-only (stability/sensitivity tested, never golden).
    uint64_t key = UsdGenDigestOffset;
    uint64_t occluderKey = UsdGenDigestOffset;
    UsdGenFurOccluderBuild* const occluderCache = params.occluderCache;
    {
        TRACE_SCOPE("usdGen occlusion: identity");
        auto word = [&](void const* p, size_t n) {
            key = UsdGenDigestBytes(p, n, key);
        };
        auto oword = [&](void const* p, size_t n) {
            occluderKey = UsdGenDigestBytes(p, n, occluderKey);
        };
        word(&params.voxelSize, sizeof(params.voxelSize));
        word(&params.maxDimension, sizeof(params.maxDimension));
        word(&params.resolution, sizeof(params.resolution));
        word(&params.opaqueBiasVoxels, sizeof(params.opaqueBiasVoxels));
        word(&params.opaqueMarkVoxels, sizeof(params.opaqueMarkVoxels));
        word(&params.scalpShadow, sizeof(params.scalpShadow));
        word(&params.scalpMaxTriangles, sizeof(params.scalpMaxTriangles));
        // The occluder bytes feed their own digest first, which the build
        // cache below keys on; folding that digest into the volume key keeps
        // one pass over the mesh while covering the same inputs. Key values
        // change; the key is in-memory equality-only (never golden). A
        // deform timeline re-cooks every frame while its emitting surfaces
        // sit still, so the memoized digest skips the rehash when the
        // arrays are the same shares; identical bytes hash identically, so a
        // hit returns exactly the recomputed value.
        auto sameShare = [](auto const& a, auto const& b) {
            return a.size() == b.size() &&
                (a.empty() || a.cdata() == b.cdata());
        };
        bool digestHit = occluderCache && occluderCache->digestValid &&
            occluderCache->digestArrays.size() == params.occluders.size();
        for (size_t i = 0; digestHit && i < params.occluders.size(); ++i) {
            UsdGenFurOccluder const& a = occluderCache->digestArrays[i];
            UsdGenFurOccluder const& b = params.occluders[i];
            digestHit = a.worldMatrix == b.worldMatrix &&
                sameShare(a.points, b.points) &&
                sameShare(a.faceVertexCounts, b.faceVertexCounts) &&
                sameShare(a.faceVertexIndices, b.faceVertexIndices);
        }
        if (digestHit) {
            occluderKey = occluderCache->digest;
        } else {
            for (UsdGenFurOccluder const& occluder : params.occluders) {
                oword(&occluder.worldMatrix, sizeof(occluder.worldMatrix));
                oword(occluder.points.cdata(), occluder.points.size()*sizeof(GfVec3f));
                oword(occluder.faceVertexCounts.cdata(),
                      occluder.faceVertexCounts.size()*sizeof(int));
                oword(occluder.faceVertexIndices.cdata(),
                      occluder.faceVertexIndices.size()*sizeof(int));
            }
            if (occluderCache) {
                occluderCache->digestArrays = params.occluders;
                occluderCache->digest = occluderKey;
                occluderCache->digestValid = true;
            }
        }
        key = UsdGenDigestBytes(&occluderKey, sizeof(occluderKey), key);
    }
    bool same = previous && previous->size()==tiles->size() &&
                (!volumeKey || *volumeKey == key);
    if (volumeKey) *volumeKey = key;
    if (same) for(size_t i=0;i<tiles->size();++i) {
        auto const& a=(*tiles)[i]; auto const& b=(*previous)[i];
        if(a.tile!=b.tile || !a.points.IsIdentical(b.points) ||
           !a.widths.IsIdentical(b.widths) || a.curveVertexCounts!=b.curveVertexCounts ||
           a.basis!=b.basis || a.xformMatrix!=b.xformMatrix) { same=false; break; }
        for(auto name:names) {
            auto it=std::find_if(b.extraUniform.begin(),b.extraUniform.end(),
                [&](auto const& plane){return plane.name==name;});
            if(it==b.extraUniform.end() || it->arity!=3 || it->f.size()!=b.points.size()*3)
                same=false;
        }
    }
    if (same) {
        for(size_t i=0;i<tiles->size();++i) {
            auto& dst=(*tiles)[i].extraUniform;
            dst.erase(std::remove_if(dst.begin(),dst.end(),
                [](auto const& p){return Reserved(p.name);}),dst.end());
            for(auto const& p:(*previous)[i].extraUniform)
                if(Reserved(p.name)) dst.push_back(p);
            std::sort(dst.begin(),dst.end(),[](auto const& a,auto const& b){return a.name<b.name;});
        }
        return false;
    }
    UsdGenWorkDispatcher* const dispatcher = params.dispatcher;
    size_t const tileCount = tiles->size();

    // --- world-space positions and bounds, per tile ---------------------------
    // Identity tiles (every benchmark tile) alias the published points
    // instead of copying them: the splat and gather only read, the tile
    // outlives the bake, and nothing below mutates points, so the view is
    // the same floats with no 96MB copy, alloc, or page faults. The
    // validation and bounds scan still read every point exactly as before
    // (same order, same throws); only transformed tiles fill `owned`.
    struct TilePositions {
        GfVec3f const* data = nullptr;
        size_t count = 0;
        std::vector<GfVec3f> owned;
    };
    std::vector<TilePositions> positions(tileCount);
    std::vector<GfVec3f> tileLo(tileCount, GfVec3f(std::numeric_limits<float>::max()));
    std::vector<GfVec3f> tileHi(tileCount, GfVec3f(-std::numeric_limits<float>::max()));
    std::vector<char const*> invalid(tileCount, nullptr);
    {
        TRACE_SCOPE("usdGen occlusion: positions");
        ForEach(dispatcher, tileCount, [&](size_t t) {
            auto const& tile=(*tiles)[t];
            size_t count=0;
            for(int c:tile.curveVertexCounts) {
                if(c<0) { invalid[t]="negative fur curve vertex count"; return; }
                count+=size_t(c);
            }
            if(count!=tile.points.size() || (!tile.widths.empty() &&
                tile.widths.size()!=1 && tile.widths.size()!=count)) {
                invalid[t]="invalid fur points/widths cardinality";
                return;
            }
            GfMatrix4f const m(tile.xformMatrix);
            bool const identity = tile.xformMatrix == GfMatrix4d(1.0);
            auto& p=positions[t]; p.count=count;
            if(identity) {
                p.data=tile.points.cdata();
            } else {
                p.owned.resize(count);
                p.data=p.owned.data();
            }
            GfVec3f const* const src=tile.points.cdata();
            GfVec3f lo=tileLo[t], hi=tileHi[t];
            for(size_t i=0;i<count;++i) {
                GfVec3f const w = identity ? src[i] : m.Transform(src[i]);
                for(int k=0;k<3;++k) {
                    if(!std::isfinite(w[k])) { invalid[t]="non-finite fur point"; return; }
                    lo[k]=std::min(lo[k],w[k]); hi[k]=std::max(hi[k],w[k]);
                }
                if(!identity) p.owned[i]=w;
            }
            tileLo[t]=lo; tileHi[t]=hi;
        });
    }
    GfVec3f lo(std::numeric_limits<float>::max()), hi(-std::numeric_limits<float>::max());
    size_t pointCount=0;
    for(size_t t=0;t<tileCount;++t) {
        if(invalid[t]) throw std::invalid_argument(invalid[t]);
        pointCount+=positions[t].count;
        if(positions[t].count==0) continue;
        for(int k=0;k<3;++k) { lo[k]=std::min(lo[k],tileLo[t][k]); hi[k]=std::max(hi[k],tileHi[t][k]); }
    }
    if(!pointCount) return true;
    auto span=hi-lo;
    for(int k=0;k<3;++k) if(!std::isfinite(span[k]))
        throw std::invalid_argument("fur bounds exceed finite grid range");

    // --- grid: anisotropic dims at a target world voxel size -----------------
    // The longest axis alone used to decide a cube, which wastes resolution on
    // a flat groom and under-resolves a head. Here every axis gets the voxels
    // its own extent needs, at a shared edge length.
    int const maxDimension = std::clamp(params.maxDimension, 8, 512);
    int const opaqueBias = std::max(0, params.opaqueBiasVoxels);
    int const opaqueMark = std::max(1, params.opaqueMarkVoxels);
    float const groomLongest = std::max({span[0],span[1],span[2],1e-6f});
    // Voxel edge: keyed to the groom itself, so adding an occluder never
    // coarsens the hair.
    float voxel;
    if (params.resolution > 0) {
        voxel = groomLongest/float(std::clamp(params.resolution,8,maxDimension)-4);
    } else {
        float const target = params.voxelSize > 0.f ? params.voxelSize : 0.3f;
        // Never coarser than the legacy cube, never finer than maxDimension.
        voxel = std::min(target, groomLongest/float(kLegacyResolution-4));
        for (int k=0;k<3;++k) voxel = std::max(voxel, span[k]/float(maxDimension-4));
    }
    if (!(voxel > 0.f) || !std::isfinite(voxel))
        throw std::invalid_argument("fur bounds exceed finite grid range");

    // The opaque shell lives up to (bias + mark + 1) voxels behind a surface.
    // Grow the volume so a scalp sitting on the groom's own boundary -- a
    // ground-plane emitter is the pure case -- still has its shell inside it,
    // but only as far as the occluder actually reaches, so a room-sized floor
    // cannot cost the groom its resolution.
    if (!params.occluders.empty()) {
        float const reach = float(opaqueBias+opaqueMark+1)*voxel;
        GfVec3f occLo(std::numeric_limits<float>::max()), occHi(-occLo[0]);
        for (UsdGenFurOccluder const& occluder : params.occluders) {
            GfMatrix4f const m(occluder.worldMatrix);
            bool const identity = occluder.worldMatrix == GfMatrix4d(1.0);
            for (GfVec3f const& p : occluder.points) {
                GfVec3f const w = identity ? p : m.Transform(p);
                for (int k=0;k<3;++k) {
                    if (!std::isfinite(w[k]))
                        throw std::invalid_argument("non-finite occluder point");
                    occLo[k]=std::min(occLo[k],w[k]); occHi[k]=std::max(occHi[k],w[k]);
                }
            }
        }
        if (occLo[0] <= occHi[0]) for (int k=0;k<3;++k) {
            lo[k]=std::min(lo[k],std::max(occLo[k]-reach,lo[k]-reach));
            hi[k]=std::max(hi[k],std::min(occHi[k]+reach,hi[k]+reach));
        }
        span=hi-lo;
    }

    Grid grid;
    grid.h = voxel;
    if (params.resolution > 0) {
        int const resolution = std::clamp(params.resolution, 8, maxDimension);
        grid.h = std::max({span[0],span[1],span[2],1e-6f})/float(resolution-4);
        grid.n[0]=grid.n[1]=grid.n[2]=resolution;
    } else {
        for (int k=0;k<3;++k) grid.h = std::max(grid.h, span[k]/float(maxDimension-4));
        for (int k=0;k<3;++k)
            grid.n[k] = std::clamp(int(std::ceil(span[k]/grid.h))+4, 8, maxDimension);
    }
    if (!(grid.h > 0.f) || !std::isfinite(grid.h))
        throw std::invalid_argument("fur bounds exceed finite grid range");
    grid.origin = lo-GfVec3f(2*grid.h);
    grid.invH = 1.f/grid.h;
    float const h = grid.h;
    size_t const cells = grid.Cells();

    // --- work decomposition ---------------------------------------------------
    std::vector<Job> jobs;
    for(size_t t=0;t<tileCount;++t) {
        auto const& counts=(*tiles)[t].curveVertexCounts;
        size_t cv=0;
        for(size_t c=0;c<counts.size();) {
            Job job; job.tile=uint32_t(t); job.firstCurve=uint32_t(c); job.firstCv=cv;
            while(c<counts.size() && job.curveCount<kJobCurves && job.cvCount<kJobCvs) {
                job.cvCount+=size_t(counts[c]); ++c; ++job.curveCount;
            }
            cv=job.firstCv+job.cvCount;
            jobs.push_back(job);
        }
    }

    // --- splat projected fibre area, per job into its own sub-box -------------
    // With caller-owned bake scratch the density grid keeps its capacity
    // across cooks; the zero fill stays either way (cells outside every job
    // sub-box must read 0), it just lands on already-faulted storage.
    UsdGenFurBakeScratch* const bakeScratch = params.bakeScratch;
    std::vector<GfVec3f> localDensity;
    std::vector<GfVec3f>& density =
        bakeScratch ? bakeScratch->density : localDensity;
    density.assign(cells,GfVec3f(0));
    size_t const slabs=std::max<size_t>(1,std::min<size_t>(size_t(grid.n[2]),64));
    {
        TRACE_SCOPE("usdGen occlusion: splat");
        auto const clock0=std::chrono::steady_clock::now();
        // Carried sub-boxes keep their capacities across cooks (hundreds of
        // per-cook mallocs otherwise); the per-job zero fill stays, landing
        // on already-faulted storage. A job with no CVs leaves no
        // contribution, so it clears its (possibly carried) box: without
        // this a retopologized cook could merge a stale carried box.
        std::vector<UsdGenSplatJobDensity> localJobs;
        std::vector<UsdGenSplatJobDensity>& local =
            bakeScratch ? bakeScratch->jobDensity : localJobs;
        local.resize(jobs.size());
        ForEach(dispatcher, jobs.size(), [&](size_t index) {
            Job const& job=jobs[index];
            auto const& tile=(*tiles)[job.tile]; GfVec3f const* const p=positions[job.tile].data;
            if(!job.cvCount) { local[index].density.clear(); return; }
            GfVec3f jobLo(std::numeric_limits<float>::max()), jobHi(-jobLo[0]);
            for(size_t i=job.firstCv;i<job.firstCv+job.cvCount;++i)
                for(int k=0;k<3;++k) {
                    jobLo[k]=std::min(jobLo[k],p[i][k]); jobHi[k]=std::max(jobHi[k],p[i][k]);
                }
            UsdGenSplatJobDensity& out=local[index];
            int b0[3], b1[3]; float f[3];
            grid.Corners(jobLo,b0,f);
            grid.Corners(jobHi,b1,f);
            // Two voxels of slack: a pinned end segment and a catmullRom
            // overshoot can leave the control points' own box.
            for(int k=0;k<3;++k) {
                int const first=std::max(0,b0[k]-2);
                int const last=std::min(grid.n[k]-1,b1[k]+3);
                out.lo[k]=first; out.dims[k]=last-first+1;
            }
            out.density.assign(size_t(out.dims[0])*out.dims[1]*out.dims[2],GfVec3f(0));
            size_t const strideY=size_t(out.dims[0]), strideZ=strideY*out.dims[1];
            auto deposit=[&](GfVec3f const& at, GfVec3f const& value) {
                int c[3]; float w[3];
                grid.Corners(at,c,w);
                for(int k=0;k<3;++k)
                    c[k]=std::clamp(c[k],out.lo[k],out.lo[k]+out.dims[k]-2);
                size_t const cell=(size_t(c[2]-out.lo[2])*out.dims[1]+(c[1]-out.lo[1]))
                                  *out.dims[0]+(c[0]-out.lo[0]);
                float const wx[2]={1-w[0],w[0]}, wy[2]={1-w[1],w[1]}, wz[2]={1-w[2],w[2]};
                for(int z=0;z<2;++z) {
                    GfVec3f const vz=value*wz[z];
                    for(int y=0;y<2;++y) {
                        GfVec3f const vy=vz*wy[y];
                        GfVec3f* const row=&out.density[cell+y*strideY+z*strideZ];
                        row[0]+=vy*wx[0]; row[1]+=vy*wx[1];
                    }
                }
            };
            // Conservative cross-section scale under nonuniform transforms.
            float scale=0;
            for(int k=0;k<3;++k) { GfVec3d axis(0);axis[k]=1;
                scale=std::max(scale,float(tile.xformMatrix.TransformDir(axis).GetLength())); }
            bool const catmullRom = tile.basis == "catmullRom";
            // Single-substep segments (chord under one voxel: every segment
            // of a groom whose CV spacing is sub-voxel) evaluate the basis
            // at exactly u=0 and u=1, so both weight sets hoist out of the
            // segment loop. Same function, same inputs (float(1)/float(1)
            // is exactly 1): bit-identical.
            float sw0[4], sw1[4];
            CubicWeights(catmullRom,0.f,sw0);
            CubicWeights(catmullRom,1.f,sw1);
            bool const constantWidth = tile.widths.size()==1;
            float const oneWidth = tile.widths.empty() ? 0.f :
                (std::isfinite(tile.widths[0]) ? std::max(0.f,tile.widths[0])*scale : 0.f);
            size_t base=job.firstCv;
            for(uint32_t c=job.firstCurve;c<job.firstCurve+job.curveCount;++c) {
                int const count=tile.curveVertexCounts[c];
                if(count<2) { base+=size_t(count>0?count:0); continue; }
                auto point=[&](int i)->GfVec3f {
                    if(i<0) return 2.f*p[base]-p[base+1];
                    if(i>=count) return 2.f*p[base+count-1]-p[base+count-2];
                    return p[base+i];
                };
                auto width=[&](int i)->float {
                    if(tile.widths.empty() || constantWidth) return oneWidth;
                    auto at=[&](int j)->float {
                        float const w=tile.widths[base+size_t(j)];
                        return std::isfinite(w)?std::max(0.f,w):0.f;
                    };
                    float w;
                    if(i<0) w=2.f*at(0)-at(1);
                    else if(i>=count) w=2.f*at(count-1)-at(count-2);
                    else w=at(i);
                    return std::max(0.f,w)*scale;
                };
                // A pinned cubic has one segment per CV interval; splat the
                // curve Storm draws, not the control polygon.
                for(int j=0;j<count-1;++j) {
                    GfVec3f const q[4]={point(j-1),point(j),point(j+1),point(j+2)};
                    float const qw[4]={width(j-1),width(j),width(j+1),width(j+2)};
                    float const chord=(q[2]-q[1]).GetLength();
                    if(!(chord>0.f)) continue;
                    int const steps=std::clamp(
                        int(std::ceil(kSamplesPerVoxel*chord*grid.invH)),1,kMaxSegmentSteps);
                    if(steps==1) {
                        // The general loop below unrolled for its common
                        // case: one substep at u=1 with hoisted weights. The
                        // lerps and the deposit tail are the same
                        // expressions in the same order: bit-identical.
                        GfVec3f const a=q[0]*sw0[0]+q[1]*sw0[1]+q[2]*sw0[2]+q[3]*sw0[3];
                        float const aw=qw[0]*sw0[0]+qw[1]*sw0[1]+qw[2]*sw0[2]+qw[3]*sw0[3];
                        GfVec3f const b=q[0]*sw1[0]+q[1]*sw1[1]+q[2]*sw1[2]+q[3]*sw1[3];
                        float const bw=qw[0]*sw1[0]+qw[1]*sw1[1]+qw[2]*sw1[2]+qw[3]*sw1[3];
                        GfVec3f const d=b-a;
                        float const len=d.GetLength();
                        if(len>1e-12f) {
                            GfVec3f const tangent=d/len;
                            float const area=std::max(0.f,0.5f*(aw+bw))*len*grid.invH*grid.invH;
                            GfVec3f projection;
                            for(int k=0;k<3;++k)
                                projection[k]=std::sqrt(std::max(0.f,1-tangent[k]*tangent[k]));
                            deposit(a+0.5f*d,projection*area);
                        }
                        continue;
                    }
                    float w4[4];
                    CubicWeights(catmullRom,0.f,w4);
                    GfVec3f a=q[0]*w4[0]+q[1]*w4[1]+q[2]*w4[2]+q[3]*w4[3];
                    float aw=qw[0]*w4[0]+qw[1]*w4[1]+qw[2]*w4[2]+qw[3]*w4[3];
                    for(int s=0;s<steps;++s) {
                        float const u=float(s+1)/float(steps);
                        CubicWeights(catmullRom,u,w4);
                        GfVec3f const b=q[0]*w4[0]+q[1]*w4[1]+q[2]*w4[2]+q[3]*w4[3];
                        float const bw=qw[0]*w4[0]+qw[1]*w4[1]+qw[2]*w4[2]+qw[3]*w4[3];
                        GfVec3f const d=b-a;
                        float const len=d.GetLength();
                        if(len>1e-12f) {
                            GfVec3f const tangent=d/len;
                            // Projected fibre area over the cell cross-section:
                            // the expected crossings a ray along each axis sees.
                            float const area=std::max(0.f,0.5f*(aw+bw))*len*grid.invH*grid.invH;
                            GfVec3f projection;
                            for(int k=0;k<3;++k)
                                projection[k]=std::sqrt(std::max(0.f,1-tangent[k]*tangent[k]));
                            deposit(a+0.5f*d,projection*area);
                        }
                        a=b; aw=bw;
                    }
                }
                base+=size_t(count);
            }
        });
        if(TfDebug::IsEnabled(USDGEN_FUR)) {
            size_t sub=0; for(auto const& job:local)
                sub+=size_t(job.dims[0])*job.dims[1]*job.dims[2];
            TF_DEBUG(USDGEN_FUR).Msg(
                "usdGen fur: grid %dx%dx%d voxel %.4f, %zu cells, %zu tiles, "
                "%zu jobs, %zu CVs, %zu sub-box cells, %zu occluder mesh(es), "
                "scatter %.1f ms\n",
                grid.n[0],grid.n[1],grid.n[2],double(h),cells,tileCount,jobs.size(),
                pointCount,sub,params.occluders.size(),
                std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-clock0).count());
        }
        // Merge over disjoint z slabs: with the grid at a world voxel size this
        // sum is O(jobs * sub-box) and dominated the splat on one thread.
        ForEach(dispatcher, slabs, [&](size_t slab) {
            int const z0=int((slab*size_t(grid.n[2]))/slabs);
            int const z1=int(((slab+1)*size_t(grid.n[2]))/slabs);
            for(UsdGenSplatJobDensity const& job:local) {
                if(job.density.empty()) continue;
                int const first=std::max(z0,job.lo[2]);
                int const last=std::min(z1,job.lo[2]+job.dims[2]);
                for(int z=first;z<last;++z) for(int y=0;y<job.dims[1];++y) {
                    size_t const src=(size_t(z-job.lo[2])*job.dims[1]+y)*job.dims[0];
                    size_t const dst=grid.Index(job.lo[0],job.lo[1]+y,z);
                    for(int x=0;x<job.dims[0];++x) density[dst+x]+=job.density[src+x];
                }
            }
        });
    }

    // --- occluder triangles, and which side of them the solid is on ----------
    // Built before the shell is injected, because the scalp-shadow cap needs
    // hair-only depth and the side vote needs hair-only density.
    //
    // The mesh is a pure function of the occluder bytes, so a caller-owned
    // cache carries it across cooks: a deform timeline re-cooks every frame
    // while its emitting surfaces sit still. On a miss the build lands
    // directly in the cache's vectors, overwritten in place from per-array
    // write cursors (the key published only after the whole mesh built
    // without throwing); on a hit the cached mesh is used as-is. In-place
    // overwrite matters: a deforming surface rebuilds every cook with the
    // same counts, so the steady-state rebuild resizes nothing and the
    // ~120MB triangle fill plus the vertex transform land on already-faulted
    // storage instead of re-zeroing fresh pages. The side vote below always
    // re-runs: it reads the fresh hair density, so `inward` is identical to
    // a from-scratch build either way. The world-space vertices are
    // build-only temporaries (every later stage reads the triangles) but
    // ride the same cache for their capacity.
    // (occluderCache was hoisted to the digest block above.)
    bool const occluderHit = occluderCache && occluderCache->valid &&
                             occluderCache->key == occluderKey;
    std::vector<UsdGenFurOccluderTriangle> missTriangles;
    std::vector<GfVec3f> missVertices;
    std::vector<GfVec3f> missNormals;
    std::vector<float> missExtents;
    if (occluderCache && !occluderHit) {
        occluderCache->valid = false;
        if (params.occluders.empty()) {
            // No mesh this cook: drop the carried arrays exactly as the
            // historical clear() did, so the emptiness checks below agree.
            occluderCache->triangles.clear();
            occluderCache->vertices.clear();
            occluderCache->vertexNormals.clear();
        }
    }
    std::vector<UsdGenFurOccluderTriangle>& triangles =
        occluderCache ? occluderCache->triangles : missTriangles;
    std::vector<GfVec3f>& vertices =
        occluderCache ? occluderCache->vertices : missVertices;
    std::vector<GfVec3f>& vertexNormals =
        occluderCache ? occluderCache->vertexNormals : missNormals;
    std::vector<float>& extents =
        occluderCache ? occluderCache->extents : missExtents;
    float inward = -1.f;
    if(!params.occluders.empty()) {
        TRACE_SCOPE("usdGen occlusion: occluder triangles");
        TF_DEBUG(USDGEN_FUR).Msg("usdGen fur: occluder mesh %s\n",
                                 occluderHit ? "reused" : "rebuilt");
        if (!occluderHit) {
        if (occluderCache &&
            occluderCache->topo.size() != params.occluders.size()) {
            // Structural change (an occluder added or removed): the fan
            // entries are per-ordinal, so drop them all rather than skew.
            occluderCache->topo.clear();
            occluderCache->topo.resize(params.occluders.size());
        }
        size_t vCursor=0, tCursor=0;
        for(size_t oi=0;oi<params.occluders.size();++oi) {
            UsdGenFurOccluder const& occluder=params.occluders[oi];
            GfMatrix4f const m(occluder.worldMatrix);
            bool const identity = occluder.worldMatrix == GfMatrix4d(1.0);
            size_t const vbase=vCursor;
            uint32_t const base=uint32_t(vbase);
            // Vertices: every output slot is an independent function of its
            // input point, so chunks write disjoint ranges of the resized
            // array directly: no merge, identical order. The serial build
            // throws on the first non-finite point; chunks report their
            // lowest bad index and the serial min reproduces that throw.
            { TRACE_SCOPE("usdGen tris: vertices");
            size_t const nPoints=occluder.points.size();
            size_t const nVChunks=std::max<size_t>(1,std::min(nPoints,
                size_t(dispatcher?std::max(1,dispatcher->MaxConcurrency()):1)));
            // Every slot below is written, so only a shortfall resizes (and
            // only its delta zero-fills): the steady-state rebuild with
            // carried capacity writes straight into place.
            if(vertices.size()<vbase+nPoints) vertices.resize(vbase+nPoints);
            std::vector<size_t> vFirstBad(nVChunks,nPoints);
            ForEach(dispatcher,nVChunks,[&](size_t chunk) {
                size_t const first=(chunk*nPoints)/nVChunks;
                size_t const last=((chunk+1)*nPoints)/nVChunks;
                size_t bad=nPoints;
                for(size_t i=first;i<last;++i) {
                    GfVec3f const& p=occluder.points[i];
                    GfVec3f const w=identity?p:m.Transform(p);
                    if(bad==nPoints && !(std::isfinite(w[0])&&
                                         std::isfinite(w[1])&&std::isfinite(w[2])))
                        bad=i;
                    vertices[vbase+i]=w;
                }
                vFirstBad[chunk]=bad;
            });
            size_t badPoint=nPoints;
            for(size_t b:vFirstBad) badPoint=std::min(badPoint,b);
            if(badPoint!=nPoints)
                throw std::invalid_argument("non-finite occluder point");
            vCursor=vbase+nPoints;
            }
            // Faces: a deforming surface moves its points every cook but never
            // its topology, so the fan expansion and its validation run once
            // and later rebuilds emit straight from the cached triples (the
            // degenerate skip still re-evaluates per cook: it reads
            // positions). On a topology hit chunks emit disjoint ranges of
            // the pre-sized output directly: no thread-local vectors, no
            // merge. When no chunk skipped (every benchmark cook) that is
            // already exactly the merged sequence; otherwise the rare serial
            // compact below restores it in chunk order. The fill is the old
            // validated expansion verbatim (same throws, same triangles)
            // plus the triple record.
            { TRACE_SCOPE("usdGen tris: faces");
            bool topoHit=false;
            if(occluderCache) {
                auto const& topo=occluderCache->topo[oi];
                topoHit=topo.valid
                    && topo.counts.size()==occluder.faceVertexCounts.size()
                    && topo.counts.cdata()==occluder.faceVertexCounts.cdata()
                    && topo.indices.size()==occluder.faceVertexIndices.size()
                    && topo.indices.cdata()==occluder.faceVertexIndices.cdata()
                    && topo.pointsSize==occluder.points.size();
            }
            if(topoHit) {
                auto const& fan=occluderCache->topo[oi].fan;
                size_t const nFan=fan.size();
                size_t const nHChunks=std::max<size_t>(1,std::min(nFan,
                    size_t(dispatcher?std::max(1,dispatcher->MaxConcurrency()):1)));
                size_t const tbase=tCursor;
                if(triangles.size()<tbase+nFan) triangles.resize(tbase+nFan);
                std::vector<size_t> hCounts(nHChunks,0);
                ForEach(dispatcher,nHChunks,[&](size_t chunk) {
                    size_t const first=(chunk*nFan)/nHChunks;
                    size_t const last=((chunk+1)*nFan)/nHChunks;
                    size_t w=tbase+first, kept=0;
                    for(size_t t=first;t<last;++t) {
                        auto const& r=fan[t];
                        uint32_t const ia=base+r.a, ib=base+r.b, ic=base+r.c;
                        UsdGenFurOccluderTriangle tri{vertices[ia],vertices[ib],vertices[ic],
                                         GfVec3f(0),ia,ib,ic};
                        GfVec3f const n=GfCross(tri.b-tri.a,tri.c-tri.a);
                        float const length=n.GetLength();
                        if(!(length>0.f)) continue;
                        tri.normal=n/length;
                        triangles[w++]=tri; ++kept;
                    }
                    hCounts[chunk]=kept;
                });
                size_t emitted=0;
                for(size_t c:hCounts) emitted+=c;
                if(emitted!=nFan) {
                    size_t w=tbase;
                    for(size_t chunk=0;chunk<nHChunks;++chunk) {
                        size_t const first=(chunk*nFan)/nHChunks;
                        size_t const n=hCounts[chunk];
                        for(size_t k=0;k<n;++k,++w)
                            triangles[w]=triangles[tbase+first+k];
                    }
                    triangles.resize(tbase+emitted);
                }
                tCursor=tbase+emitted;
            } else {
            size_t const nFaces=occluder.faceVertexCounts.size();
            size_t const nFChunks=std::max<size_t>(1,std::min(nFaces,
                size_t(dispatcher?std::max(1,dispatcher->MaxConcurrency()):1)));
            std::vector<size_t> chunkCursor(nFChunks+1,0);
            for(size_t chunk=0;chunk<nFChunks;++chunk) {
                size_t const first=(chunk*nFaces)/nFChunks;
                size_t const last=((chunk+1)*nFaces)/nFChunks;
                size_t cursor=chunkCursor[chunk];
                for(size_t f=first;f<last;++f)
                    cursor+=size_t(occluder.faceVertexCounts[f]);
                chunkCursor[chunk+1]=cursor;
            }
            struct FaceError {
                size_t face=0; int step=0; size_t corner=0; int code=0;
                bool has=false;
            };
            std::vector<FaceError> chunkErr(nFChunks);
            std::vector<std::vector<UsdGenFurOccluderTriangle>> fchunks(nFChunks);
            using FanTri=UsdGenFurOccluderBuild::TopoEntry::Tri;
            std::vector<std::vector<FanTri>> tchunks(nFChunks);
            bool const record=occluderCache!=nullptr;
            ForEach(dispatcher,nFChunks,[&](size_t chunk) {
                size_t const first=(chunk*nFaces)/nFChunks;
                size_t const last=((chunk+1)*nFaces)/nFChunks;
                auto& out=fchunks[chunk]; out.reserve(last-first);
                auto& tout=tchunks[chunk]; if(record) tout.reserve(last-first);
                FaceError err;
                size_t cursor=chunkCursor[chunk];
                for(size_t f=first;f<last && !err.has;++f) {
                    int const face=occluder.faceVertexCounts[f];
                    if(face<0) {
                        err={f,1,0,1,true}; break;
                    }
                    if(cursor+size_t(face)>occluder.faceVertexIndices.size()) {
                        err={f,2,0,2,true}; break;
                    }
                    if(face<3) { cursor+=size_t(face); continue; }
                    size_t seq=0;
                    bool faceBad=false;
                    auto vertex=[&](size_t at)->uint32_t {
                        int const index=occluder.faceVertexIndices[cursor+at];
                        size_t const here=seq++;
                        if(index<0 || size_t(index)>=occluder.points.size()) {
                            err={f,3,here,3,true}; faceBad=true; return uint32_t(0);
                        }
                        return uint32_t(index);
                    };
                    uint32_t const afirst=vertex(0);
                    for(size_t k=1;k+1<size_t(face) && !faceBad;++k) {
                        uint32_t const ab=vertex(k);
                        if(faceBad) break;
                        uint32_t const ac=vertex(k+1);
                        if(faceBad) break;
                        if(record) tout.push_back({afirst,ab,ac});
                        uint32_t const vfirst=base+afirst, ib=base+ab, ic=base+ac;
                        UsdGenFurOccluderTriangle tri{vertices[vfirst],vertices[ib],vertices[ic],
                                     GfVec3f(0),vfirst,ib,ic};
                        GfVec3f const n=GfCross(tri.b-tri.a,tri.c-tri.a);
                        float const length=n.GetLength();
                        if(!(length>0.f)) continue;
                        tri.normal=n/length;
                        out.push_back(tri);
                    }
                    cursor+=size_t(face);
                }
                chunkErr[chunk]=err;
            });
            size_t const tbase=tCursor;
            // Truncate any carried tail (a shrink, never a fill), then append
            // this occluder's expansion exactly as the cleared build did.
            triangles.resize(tbase);
            size_t total=0;
            for(auto const& c:fchunks) total+=c.size();
            triangles.reserve(triangles.size()+total);
            for(auto const& c:fchunks)
                triangles.insert(triangles.end(),c.begin(),c.end());
            tCursor=tbase+total;
            FaceError first;
            for(auto const& e:chunkErr) {
                if(!e.has) continue;
                if(!first.has || e.face<first.face ||
                   (e.face==first.face && (e.step<first.step ||
                    (e.step==first.step && e.corner<first.corner))))
                    first=e;
            }
            if(first.has) {
                if(first.code==1)
                    throw std::invalid_argument("negative occluder face vertex count");
                if(first.code==2)
                    throw std::invalid_argument("invalid occluder face cardinality");
                throw std::invalid_argument("occluder face index out of range");
            }
            if(occluderCache) {
                auto& topo=occluderCache->topo[oi];
                topo.counts=occluder.faceVertexCounts;
                topo.indices=occluder.faceVertexIndices;
                topo.pointsSize=occluder.points.size();
                topo.fan.clear();
                size_t totalT=0;
                for(auto const& c:tchunks) totalT+=c.size();
                topo.fan.reserve(totalT);
                for(auto const& c:tchunks)
                    topo.fan.insert(topo.fan.end(),c.begin(),c.end());
                topo.valid=true;
            }
            }
            }
        }
        // Size the carried arrays exactly: steady-state cursors already match,
        // so these are no-ops; a shrunk mesh truncates without filling.
        triangles.resize(tCursor);
        vertices.resize(vCursor);
        // Area-weighted vertex normals: the cross product's length is twice the
        // triangle's area, so accumulating it unnormalised is the weighting.
        // The max edge length fills in the same pass (see the header): the
        // triangle is already in hand, so the edges, vote, and shell loops'
        // per-cook lengths cost one sqrt per triangle per mesh build, never
        // a second sweep over the mesh.
        { TRACE_SCOPE("usdGen tris: normals");
        vertexNormals.assign(vertices.size(),GfVec3f(0));
        extents.resize(triangles.size());
        for(size_t i=0;i<triangles.size();++i) {
            UsdGenFurOccluderTriangle const& tri=triangles[i];
            GfVec3f const weighted=GfCross(tri.b-tri.a,tri.c-tri.a);
            vertexNormals[tri.ia]+=weighted;
            vertexNormals[tri.ib]+=weighted;
            vertexNormals[tri.ic]+=weighted;
            GfVec3f const e1=tri.b-tri.a, e2=tri.c-tri.a, e3=tri.c-tri.b;
            // One sqrt, not three: sqrt is monotone on [0,inf), so the max
            // of the square roots is the square root of the max, bit for
            // bit (GetLength is exactly GfSqrt(GetLengthSq())).
            float const m2=std::max({e1.GetLengthSq(),e2.GetLengthSq(),
                                     e3.GetLengthSq()});
            extents[i]=std::sqrt(m2);
        }
        }
        for(size_t i=0;i<vertexNormals.size();++i) {
            float const length=vertexNormals[i].GetLength();
            if(length>1e-12f) vertexNormals[i]/=length;
        }
        if (occluderCache) {
            occluderCache->key = occluderKey;
            occluderCache->valid = true;
        }
        } /* if (!occluderHit): the vote below re-runs on the fresh density */
        // Which side of the winding the solid is on: the hair grows out of the
        // surface, so the side carrying less fur density is the inside. Voting
        // over the triangles keeps this right for either winding order and for
        // a flat emitter, where a centroid test would be degenerate.
        float vote=0;
        { TRACE_SCOPE("usdGen tris: vote");
            size_t const stride=std::max<size_t>(1,triangles.size()/4096);
            float const probe=3.f*h;
            auto fur=[&](GfVec3f const& at)->float {
                if(!grid.Contains(at)) return 0.f;
                int c[3]; float w[3];
                grid.Corners(at,c,w);
                float sum=0;
                for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
                    float const weight=(x?w[0]:1-w[0])*(y?w[1]:1-w[1])*(z?w[2]:1-w[2]);
                    GfVec3f const d=density[grid.Index(c[0]+x,c[1]+y,c[2]+z)];
                    sum+=weight*(d[0]+d[1]+d[2]);
                }
                return sum;
            };
            for(size_t i=0;i<triangles.size();i+=stride) {
                UsdGenFurOccluderTriangle const& tri=triangles[i];
                GfVec3f const e1=tri.b-tri.a, e2=tri.c-tri.a;
                // A lattice, not the centroid: one triangle can be larger than
                // the whole groom, and then its centroid says nothing.
                float const extent=extents[i];
                int const n=std::clamp(int(std::ceil(extent*grid.invH)),1,4);
                for(int a=0;a<=n;++a) for(int b=0;a+b<=n;++b) {
                    GfVec3f const on=tri.a+e1*(float(a)/float(n))+e2*(float(b)/float(n));
                    vote+=fur(on+tri.normal*probe)-fur(on-tri.normal*probe);
                }
            }
        }
        // vote > 0: more fur on the +normal side, so the solid is at -normal.
        inward = vote >= 0.f ? -1.f : 1.f;
        TF_DEBUG(USDGEN_FUR).Msg(
            "usdGen fur: %zu occluder triangle(s), fur vote %+.3f, solid side %s\n",
            triangles.size(), double(vote), inward < 0.f ? "-normal" : "+normal");
    }

    // --- the six axis sweeps, run once over hair alone and once with the
    // opaque shell ------------------------------------------------------------
    // Prefix integration is O(voxels), independent of hair count and number of
    // lights. Half the receiver's own cell is the correct discretisation of an
    // integral that starts at the CV, and is what a host renderer's HairCount - 1 shift
    // expects to find.
    // The sweep writes every cell exactly once, so the planes need sizing,
    // never filling: carried planes skip the ~7MB zero fill per sweep.
    std::array<std::vector<float>,6> localTau;
    std::array<std::vector<float>,6>& tau =
        bakeScratch ? bakeScratch->tau : localTau;
    auto sweep=[&] {
        ForEach(dispatcher, 6, [&](size_t direction) {
            int const axis=int(direction)/2;
            int const na=grid.n[axis];
            int const nu=grid.n[(axis+1)%3], nv=grid.n[(axis+2)%3];
            auto& out=tau[direction];
            if(out.size()!=cells) out.resize(cells);
            if (axis != 0) {
                // The recurrence runs along a strided axis (stride nx for
                // axis 1, nx*ny for axis 2: one 4-byte element per 64-byte
                // line touched). Step outermost instead: the inner loop
                // walks x consecutively while per-line running sums hold the
                // recurrence state. Each (u,v) line still accumulates in
                // stepping order with the same expressions, so the bytes
                // are bit-identical; only the interleave across lines
                // changes. The sums plane (nu*nv floats) stays L1/L2-hot
                // across steps.
                int const nx=grid.n[0], ny=grid.n[1];
                std::vector<float> sums(size_t(nu)*size_t(nv), 0.f);
                for(int step=0;step<na;++step) {
                    int const s=(direction%2)?step:na-1-step;
                    if(axis==1) {
                        // u over z, v over x: idx=v+nx*(s+ny*u); x innermost.
                        for(int u=0;u<nu;++u) {
                            size_t const base=size_t(nx)*(size_t(s)+
                                size_t(ny)*size_t(u));
                            float* const row=sums.data()+size_t(nv)*size_t(u);
                            for(int v=0;v<nv;++v) {
                                size_t const idx=size_t(v)+base;
                                float const depth=density[idx][axis];
                                float const sum=row[v];
                                out[idx]=std::min(UsdGenFurTauClamp,
                                    sum+0.5f*depth);
                                row[v]=sum+depth;
                            }
                        }
                    } else {
                        // u over x, v over y: idx=u+nx*(v+ny*s); x innermost.
                        size_t const plane=size_t(nx)*size_t(ny)*size_t(s);
                        for(int v=0;v<nv;++v) {
                            size_t const base=size_t(nx)*size_t(v)+plane;
                            float* const row=sums.data()+size_t(nu)*size_t(v);
                            for(int u=0;u<nu;++u) {
                                size_t const idx=size_t(u)+base;
                                float const depth=density[idx][axis];
                                float const sum=row[u];
                                out[idx]=std::min(UsdGenFurTauClamp,
                                    sum+0.5f*depth);
                                row[u]=sum+depth;
                            }
                        }
                    }
                }
                return;
            }
            for(int v=0;v<nv;++v) for(int u=0;u<nu;++u) {
                float sum=0;
                for(int step=0;step<na;++step) {
                    int c[3]; c[axis]=(direction%2)?step:na-1-step;
                    c[(axis+1)%3]=u;c[(axis+2)%3]=v;
                    size_t idx=grid.Index(c[0],c[1],c[2]);
                    float depth=density[idx][axis];
                    out[idx]=std::min(UsdGenFurTauClamp,sum+0.5f*depth);sum+=depth;
                }
            }
        });
    };
    // Trilinear gather of the six depths at one point.
    ScalpDepthGather gather{&grid, &tau};

    if(params.scalpShadow && scalpShadow && !triangles.empty()) {
        TRACE_SCOPE("usdGen occlusion: scalp shadow");
        sweep();   // hair only: the head shadowing itself is the skin shader's job
        BuildScalpShadow(triangles, vertexNormals, extents, inward, grid,
                         params.scalpMaxTriangles, gather, dispatcher,
                         scalpShadow, occluderKey, params.scalpScratch);
    }

    // --- opaque occluders: a saturated shell just under the surface -----------
    // a host renderer injects the opaque depth buffer with InjectOpaque.BiasCount /
    // MarkCount; with real triangles the same shell is marked along the inward
    // normal, so light is blocked through the scalp while roots resting on it
    // are not shadowed by it.
    if(!triangles.empty()) {
        TRACE_SCOPE("usdGen occlusion: occluders");
        int const bias=opaqueBias, mark=opaqueMark;
        // One crossing of the shell must always reach the published ceiling.
        // Marked layers land at whatever sub-voxel phase the surface happens
        // to have, and max() over trilinear weights then sums to between
        // mark/2 and mark; twice the naive share covers the worst phase, and
        // the ceiling absorbs the rest.
        float const perVoxel=2.f*UsdGenFurTauClamp/float(mark);

        // A shell taken with max(), so it does not depend on how densely the
        // triangles happen to be sampled. Each triangle is rasterized once,
        // into its chunk's own mask; the chunk masks then merge with max(),
        // which over non-negative trilinear weights is order-independent
        // bit-for-bit, so the merged shell matches the old walk exactly.
        // (The slab-bucketed walk this replaces visited every triangle once
        // per overlapping slab and re-iterated the full lattice per visit.)
        size_t const nShellChunks = std::max<size_t>(1, std::min<size_t>(
            triangles.size(), size_t(dispatcher ?
                std::max(1, dispatcher->MaxConcurrency()) : 1)));
        // Carried chunk masks keep their capacities (the per-chunk zero fill
        // stays: each chunk's stencil writes are sparse, so max() needs the
        // zero start); only a changed chunk count resizes the outer vector.
        std::vector<std::vector<float>> localChunkMasks;
        std::vector<std::vector<float>>& chunkMasks =
            bakeScratch ? bakeScratch->chunkMasks : localChunkMasks;
        if(chunkMasks.size()!=nShellChunks) chunkMasks.resize(nShellChunks);
        ForEach(dispatcher,nShellChunks,[&](size_t chunk) {
            size_t const first=(chunk*triangles.size())/nShellChunks;
            size_t const last=((chunk+1)*triangles.size())/nShellChunks;
            std::vector<float>& mask=chunkMasks[chunk];
            mask.assign(cells,0.f);
            for(size_t ti=first;ti<last;++ti) {
                UsdGenFurOccluderTriangle const& tri=triangles[ti];
                GfVec3f const e1=tri.b-tri.a, e2=tri.c-tri.a;
                // Barycentric lattice fine enough that adjacent samples'
                // trilinear stencils overlap, so the shell has no pinholes.
                float const extent=extents[ti];
                int const n=std::clamp(int(std::ceil(1.5f*extent*grid.invH)),1,512);
                GfVec3f const step=tri.normal*(inward*h);
                if(n==1 && mark<=8) {
                    // Sub-voxel triangles (every benchmark triangle): the
                    // lattice is exactly the three corners in (i,j) order
                    // (0,0),(0,1),(1,0), and float(i)/float(1) is exactly
                    // {0,1}, so the points are a, a+e2, a+e1 with no
                    // divisions: x*1 and x+0 are exact, so a+e1*1+e2*0 is
                    // bit-identical to a+e1 (likewise a+e2 and a). The
                    // per-layer offsets hoist out of the point loop (the
                    // same expressions, evaluated once). The stencil below
                    // is the general one, call for call.
                    GfVec3f off[8];
                    for(int layer=0;layer<mark;++layer)
                        off[layer]=step*(float(bias+layer)+0.5f);
                    GfVec3f const pts[3]={tri.a, tri.a+e2, tri.a+e1};
                    for(int q=0;q<3;++q) {
                        GfVec3f const& on=pts[q];
                        for(int layer=0;layer<mark;++layer) {
                            GfVec3f const at=on+off[layer];
                            if(!grid.Contains(at)) continue;
                            int c[3]; float w[3];
                            grid.Corners(at,c,w);
                            for(int z=0;z<2;++z) {
                                for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
                                    float const weight=(x?w[0]:1-w[0])*(y?w[1]:1-w[1])*(z?w[2]:1-w[2]);
                                    float& cell=mask[grid.Index(c[0]+x,c[1]+y,c[2]+z)];
                                    cell=std::max(cell,weight);
                                }
                            }
                        }
                    }
                    continue;
                }
                for(int i=0;i<=n;++i) for(int j=0;i+j<=n;++j) {
                    GfVec3f const on=tri.a+e1*(float(i)/float(n))+e2*(float(j)/float(n));
                    for(int layer=0;layer<mark;++layer) {
                        GfVec3f const at=on+step*(float(bias+layer)+0.5f);
                        if(!grid.Contains(at)) continue;
                        int c[3]; float w[3];
                        grid.Corners(at,c,w);
                        for(int z=0;z<2;++z) {
                            for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
                                float const weight=(x?w[0]:1-w[0])*(y?w[1]:1-w[1])*(z?w[2]:1-w[2]);
                                float& cell=mask[grid.Index(c[0]+x,c[1]+y,c[2]+z)];
                                cell=std::max(cell,weight);
                            }
                        }
                    }
                }
            }
        });
        // The merge writes every cell, so the carried mask skips the fill.
        std::vector<float> localMask;
        std::vector<float>& mask = bakeScratch ? bakeScratch->mask : localMask;
        if(mask.size()!=cells) mask.resize(cells);
        ForEach(dispatcher,slabs,[&](size_t slab) {
            size_t const first=(slab*cells)/slabs, last=((slab+1)*cells)/slabs;
            for(size_t i=first;i<last;++i) {
                float m=0.f;
                for(auto const& cm:chunkMasks) m=std::max(m,cm[i]);
                mask[i]=m;
            }
        });
        if(!bakeScratch) { chunkMasks.clear(); chunkMasks.shrink_to_fit(); }
        ForEach(dispatcher,slabs,[&](size_t slab) {
            size_t const first=(slab*cells)/slabs, last=((slab+1)*cells)/slabs;
            for(size_t i=first;i<last;++i)
                if(mask[i]>0.f) density[i]+=GfVec3f(mask[i]*perVoxel);
        });
    }

    // --- the sweep the strands read: hair plus the opaque shell --------------
    {
        TRACE_SCOPE("usdGen occlusion: sweep");
        sweep();
    }
    // Interleave the six sweeps: the gather reads all six depths of eight
    // neighbouring cells, which is two cache lines per cell instead of six
    // separate strided arrays. The interleave writes every element, so the
    // carried array skips the fill.
    std::vector<uint16_t> localDepths;
    std::vector<uint16_t>& depths =
        bakeScratch ? bakeScratch->depths : localDepths;
    if(depths.size()!=cells*6) depths.resize(cells*6);
    {
        TRACE_SCOPE("usdGen occlusion: interleave");
        ForEach(dispatcher, slabs, [&](size_t slab) {
            size_t const first=(slab*cells)/slabs, last=((slab+1)*cells)/slabs;
            for(size_t i=first;i<last;++i)
                for(int d=0;d<6;++d)
                    depths[i*6+d]=uint16_t(tau[d][i]*kDepthScale+0.5f);
        });
        if(!bakeScratch) for(auto& plane:tau) std::vector<float>().swap(plane);
    }

    // --- gather: every receiver reads its six depths once --------------------
    {
        TRACE_SCOPE("usdGen occlusion: gather");
        std::vector<std::array<float*,2>> planes(tileCount, {nullptr,nullptr});
        for(size_t t=0;t<tileCount;++t) {
            auto& out=(*tiles)[t].extraUniform;
            out.erase(std::remove_if(out.begin(),out.end(),
                [](auto const& p){return Reserved(p.name);}),out.end());
            for(auto name:names) {
                UsdGenPlane plane;plane.name=TfToken(name);plane.interpolation=TfToken("vertex");
                plane.type=TfToken("float");plane.arity=3;
                // Every element below is written; zero-filling six floats per
                // CV first costs more than the gather itself on a full groom.
                plane.f.resize((*tiles)[t].points.size()*3, [](float* b, float* e) {
                    std::uninitialized_default_construct(b, e);
                });
                out.push_back(std::move(plane));
            }
            std::sort(out.begin(),out.end(),[](auto const& a,auto const& b){return a.name<b.name;});
            for(int which=0;which<2;++which) {
                auto it=std::find_if(out.begin(),out.end(),
                    [&](auto const& p){return p.name==names[which];});
                planes[t][which]=it->f.data();   // detaches this tile's own new plane
            }
        }
        size_t const strideY=size_t(grid.n[0])*6, strideZ=size_t(grid.n[0])*grid.n[1]*6;
        ForEach(dispatcher, jobs.size(), [&](size_t index) {
            Job const& job=jobs[index];
            // names[0] (P) holds the even directions, names[1] (N) the odd.
            float* const p0=planes[job.tile][0];
            float* const p1=planes[job.tile][1];
            GfVec3f const* const p=positions[job.tile].data;
            // Consecutive CVs of a hair are usually sub-voxel apart (a
            // single-substep splat segment is shorter than a voxel), so they
            // usually share one cell: the eight cell x six depth loads happen
            // once per run of same-cell CVs instead of once per CV, converted
            // to float once on the miss so hit CVs skip the 48
            // integer-to-float conversions as well (the conversion is exact
            // and deterministic). Each CV still runs its own Corners() for its
            // weights, and the accumulation visits the same cells in the same
            // order, so reused CVs are bit-identical. Job-local (the sweep is
            // fixed for the whole gather, so a stale entry is impossible).
            int pc[3]={0,0,0};
            float cached[8][6];
            bool have=false;
            for(size_t i=job.firstCv;i<job.firstCv+job.cvCount;++i) {
                int c[3]; float w[3];
                grid.Corners(p[i],c,w);
                if(!have || c[0]!=pc[0] || c[1]!=pc[1] || c[2]!=pc[2]) {
                    pc[0]=c[0]; pc[1]=c[1]; pc[2]=c[2];
                    uint16_t const* const cell=&depths[grid.Index(c[0],c[1],c[2])*6];
                    for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
                        uint16_t const* const at=cell+x*6+y*strideY+z*strideZ;
                        float* const slot=cached[(z*2+y)*2+x];
                        for(int d=0;d<6;++d) slot[d]=float(at[d]);
                    }
                    have=true;
                }
                float values[6]={0,0,0,0,0,0};
                for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
                    float const weight=(x?w[0]:1-w[0])*(y?w[1]:1-w[1])*(z?w[2]:1-w[2]);
                    float const* const at=cached[(z*2+y)*2+x];
                    for(int d=0;d<6;++d) values[d]+=weight*at[d];
                }
                for(int d=0;d<3;++d) {
                    p0[i*3+d]=values[d*2]*(1.f/kDepthScale);
                    p1[i*3+d]=values[d*2+1]*(1.f/kDepthScale);
                }
            }
        });
    }
    return true;
}
}
