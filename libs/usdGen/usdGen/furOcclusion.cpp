#include "usdGen/furOcclusion.h"

#include "usdGen/debugCodes.h"
#include "usdGen/scheduler.h"

#include "pxr/base/gf/matrix4f.h"
#include "pxr/base/trace/trace.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <memory>
#include <stdexcept>
#include <string>

namespace usdGen {
namespace {
constexpr char const* names[] = {"furTauP", "furTauN"};
// The optional local-density plane: vertex float, arity 1, managed here like
// the two depth planes but kept out of names[], whose loops assume arity 3.
constexpr char const* kDensityName = "furDensity";
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
    return name == kDensityName;
}

// FNV-1a over raw bytes: the volume key only has to change when the inputs do.
struct Hash {
    uint64_t h = 1469598103934665603ull;
    void Bytes(void const* p, size_t n) {
        auto b = static_cast<unsigned char const*>(p);
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    }
    template <class T> void Value(T const& v) { Bytes(&v, sizeof(T)); }
};

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
    // Corners without the clamp, for points inside the [lo,hi] the grid was
    // grown from: every strand-gather CV qualifies, because the bounds ARE
    // the min/max over these same positions (non-finite ones were rejected
    // upstream) and occluder growth only expands them. Then q lands in
    // [1.5, n-2.5] up to float rounding (n >= 8 on every axis), so neither
    // the clamp nor the min can trigger and this is bit-identical to
    // Corners(). Every other caller keeps the clamped form: splat samples
    // overshoot the CV box by construction.
    void CornersInner(GfVec3f const& p, int base[3], float frac[3]) const {
        for (int k=0;k<3;++k) {
            float q=(p[k]-origin[k])*invH-0.5f;
            base[k]=int(q); frac[k]=q-base[k];
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

// A job's contribution to the density grid, over the sub-box of cells its
// samples touch; jobs are splatted in parallel and summed over disjoint slabs.
struct JobDensity {
    int lo[3] = {0, 0, 0}, dims[3] = {0, 0, 0};
    std::vector<GfVec3f> density;
};

// One world-space occluder triangle. The direction that points into the solid
// is resolved once for the whole mesh set. The corner ids index a per-bake
// vertex array, which is what lets the scalp cap carry smooth normals: a face
// normal would band the shadow along every edge of the scalp mesh.
struct Triangle { GfVec3f a, b, c, normal; uint32_t ia, ib, ic; };

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

/// How much solid lies behind the occluder surface: for every sample of the
/// opaque shell's barycentric lattice, the distance along the triangle's
/// inward normal to the next surface crossing, capped at `reach`.
///
/// The opaque shell is marked from two to six voxels UNDER each surface. Where
/// the solid is thinner than that -- an ear, an eyelid, a lip, a thin sheet
/// with two sides -- the shell under one side comes out through the other
/// side into open air, which is where hair lies over an ear. A CV inside that
/// stray shell reads a saturated count toward every light and renders black
/// in the outline of the ear (measured on an imported head: the ear is 5-10 mm
/// thick and the shell starts 11 mm down). The marking stops each sample's
/// layers at this thickness, so the shell never leaves the solid it stands for.
///
/// The rays leave the same lattice points, along the same face normal, that
/// the layers are marked at: a ray from a mesh vertex along a smoothed normal
/// can slip past the edge of the opposite face, which is exactly where a
/// coarse thin solid has all its vertices. Triangles are binned in a uniform
/// grid of cells about a quarter of the reach, each over its bounds grown by
/// half a cell, and every ray samples the cells along it every half cell: a
/// crossing inside an unsampled cell is still within half a cell of a sample,
/// so its triangle is in that sample's bin. The hit test takes a sliver of
/// barycentric slack, so a ray through a shared edge cannot fall between the
/// two faces. A ray that crosses nothing within reach keeps `reach`.
struct SolidThickness
{
    std::vector<size_t> offset;     // per triangle: first lattice sample
    std::vector<int> lattice;       // per triangle: the lattice's n
    std::vector<float> thickness;   // per lattice sample

    float At(size_t triangle, int i, int j) const {
        int const n=lattice[triangle];
        int base=0;
        for(int r=0;r<i;++r) base+=n+1-r;
        return thickness[offset[triangle]+size_t(base+j)];
    }
};

template <class LatticeOf>
SolidThickness MeasureSolidThickness(std::vector<Triangle> const& triangles,
                                     float inward, float reach,
                                     LatticeOf const& latticeOf,
                                     UsdGenWorkDispatcher* dispatcher)
{
    SolidThickness out;
    out.offset.resize(triangles.size());
    out.lattice.resize(triangles.size());
    size_t total=0;
    for(size_t i=0;i<triangles.size();++i) {
        int const n=latticeOf(triangles[i]);
        out.lattice[i]=n;
        out.offset[i]=total;
        total+=size_t(n+1)*size_t(n+2)/2;
    }
    out.thickness.assign(total,reach);
    if(triangles.empty() || !(reach>0.f)) return out;

    GfVec3f lo(std::numeric_limits<float>::max()), hi(-lo[0]);
    for(Triangle const& t:triangles)
        for(GfVec3f const& p:{t.a,t.b,t.c}) for(int k=0;k<3;++k) {
            lo[k]=std::min(lo[k],p[k]); hi[k]=std::max(hi[k],p[k]);
        }
    float cell=0.25f*reach;
    // A surface far larger than the reach (a ground plane) coarsens the bins
    // rather than allocating millions of them.
    for(int k=0;k<3;++k) cell=std::max(cell,(hi[k]-lo[k])/128.f);
    cell=std::max(cell,1e-6f);
    float const grow=0.5f*cell;
    lo-=GfVec3f(grow); hi+=GfVec3f(grow);
    int dims[3];
    for(int k=0;k<3;++k) dims[k]=std::max(1,int(std::ceil((hi[k]-lo[k])/cell)));
    auto bin=[&](int x,int y,int z) {
        return (size_t(z)*size_t(dims[1])+size_t(y))*size_t(dims[0])+size_t(x);
    };
    std::vector<std::vector<uint32_t>> bins(size_t(dims[0])*size_t(dims[1])*size_t(dims[2]));
    for(size_t i=0;i<triangles.size();++i) {
        Triangle const& t=triangles[i];
        int a[3], b[3];
        for(int k=0;k<3;++k) {
            float const mn=std::min({t.a[k],t.b[k],t.c[k]})-grow;
            float const mx=std::max({t.a[k],t.b[k],t.c[k]})+grow;
            a[k]=std::clamp(int((mn-lo[k])/cell),0,dims[k]-1);
            b[k]=std::clamp(int((mx-lo[k])/cell),0,dims[k]-1);
        }
        for(int z=a[2];z<=b[2];++z) for(int y=a[1];y<=b[1];++y) for(int x=a[0];x<=b[0];++x)
            bins[bin(x,y,z)].push_back(uint32_t(i));
    }

    float const epsilon=1e-3f*reach;
    float const slack=1e-4f;
    auto cast=[&](GfVec3f const& o,GfVec3f const& d,size_t self) {
        float best=reach;
        size_t last=std::numeric_limits<size_t>::max();
        for(float s=0.f;s<=reach+0.5f*cell;s+=0.5f*cell) {
            GfVec3f const p=o+d*std::min(s,reach);
            int q[3];
            for(int k=0;k<3;++k) q[k]=std::clamp(int((p[k]-lo[k])/cell),0,dims[k]-1);
            size_t const index=bin(q[0],q[1],q[2]);
            if(index==last) continue;
            last=index;
            for(uint32_t ti:bins[index]) {
                if(ti==self) continue;
                Triangle const& t=triangles[ti];
                GfVec3f const e1=t.b-t.a, e2=t.c-t.a;
                GfVec3f const pv=GfCross(d,e2);
                float const det=GfDot(e1,pv);
                if(std::fabs(det)<1e-20f) continue;
                float const inv=1.f/det;
                GfVec3f const tv=o-t.a;
                float const u=GfDot(tv,pv)*inv;
                if(u<-slack || u>1.f+slack) continue;
                GfVec3f const qv=GfCross(tv,e1);
                float const w=GfDot(d,qv)*inv;
                if(w<-slack || u+w>1.f+slack) continue;
                // Faces sharing the sample's edge or vertex meet the ray at
                // its origin; anything nearer than epsilon is that, not solid.
                float const hit=GfDot(e2,qv)*inv;
                if(hit>epsilon && hit<best) best=hit;
            }
        }
        return best;
    };
    ForEach(dispatcher,triangles.size(),[&](size_t i) {
        Triangle const& t=triangles[i];
        GfVec3f const e1=t.b-t.a, e2=t.c-t.a;
        GfVec3f const d=t.normal*inward;
        int const n=out.lattice[i];
        size_t k=out.offset[i];
        for(int a=0;a<=n;++a) for(int b=0;a+b<=n;++b,++k)
            out.thickness[k]=cast(t.a+e1*(float(a)/float(n))+e2*(float(b)/float(n)),d,i);
    });
    return out;
}

/// The scalp-shadow cap. Each occluder triangle is refined k x k, every
/// sub-vertex takes the hair-only depth where the cap is drawn, and
/// sub-triangles with no hair over them are dropped.
///
/// One k for the whole mesh, so shared edges are split identically and the cap
/// has no T-junctions: adaptive per face would crack.
///
/// The lattice (k, the lift, which side is out) is fixed by the occluders
/// alone, never by this description's voxel grid: every description over the
/// same surface then samples the same points, and the scene index can sum
/// their caps into one (UsdGenMergeScalpShadows). One cap per surface is what
/// lets the cap be drawn partly as an opaque sample mask, which is the only
/// thing in stock Storm that darkens the skin between a strand's samples.
template <class Gather>
void BuildScalpShadow(std::vector<Triangle> const& triangles,
                      std::vector<GfVec3f> const& vertexNormals, float inward,
                      int maxTriangles, Gather const& gather,
                      UsdGenScalpShadowPublication* out)
{
    // The 90th-percentile edge decides the level, so one oversized face cannot
    // refine the whole mesh; the triangle ceiling then bounds it outright.
    std::vector<float> edges;
    edges.reserve(triangles.size());
    GfVec3f boxLo(std::numeric_limits<float>::max()), boxHi(-boxLo[0]);
    for(Triangle const& t:triangles) {
        edges.push_back(std::max({(t.b-t.a).GetLength(),(t.c-t.b).GetLength(),
                                  (t.a-t.c).GetLength()}));
        for(GfVec3f const& p:{t.a,t.b,t.c}) for(int d=0;d<3;++d) {
            boxLo[d]=std::min(boxLo[d],p[d]); boxHi[d]=std::max(boxHi[d],p[d]);
        }
    }
    auto at=edges.begin()+std::min(edges.size()-1,size_t(0.9*double(edges.size())));
    std::nth_element(edges.begin(),at,edges.end());
    // A target edge of 1/128 of the surface's bounding diagonal: about a
    // voxel for a head, and unit-free.
    float const target=std::max((boxHi-boxLo).GetLength()/128.f,1e-12f);
    int k=std::max(1,int(std::ceil(*at/target)));
    int const ceiling=std::max(1,maxTriangles);
    while(k>1 && triangles.size()*size_t(k)*size_t(k) > size_t(ceiling)) --k;

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
    // The cap's sample mask pushes its depth back by half this, so the
    // translucent remainder always wins the depth test against it.
    float const lift=0.08f*target;
    size_t const perTriangle=size_t(k+1)*size_t(k+2)/2;

    // The lattice's identity: everything that decides where its points are.
    // A strided sample of the triangles is enough to tell two surfaces (or
    // two frames of one) apart; the counts catch a surface set that grew.
    Hash key;
    key.Value(k); key.Value(lift); key.Value(inward);
    key.Value(triangles.size()); key.Value(vertexNormals.size());
    {
        size_t const stride=std::max<size_t>(1,triangles.size()/4096);
        for(size_t i=0;i<triangles.size();i+=stride) {
            Triangle const& t=triangles[i];
            key.Value(t.a); key.Value(t.b); key.Value(t.c);
            key.Value(t.ia); key.Value(t.ib); key.Value(t.ic);
        }
    }

    VtVec3fArray points, normals;
    VtFloatArray tauP, tauN;
    VtIntArray counts, indices, ids;
    points.reserve(triangles.size()*perTriangle);
    std::vector<GfVec3f> local(perTriangle), localN(perTriangle);
    std::vector<float> depth(perTriangle*6), shaded(perTriangle);
    std::vector<int> remap(perTriangle);
    for(size_t triIndex=0;triIndex<triangles.size();++triIndex) {
        Triangle const& tri=triangles[triIndex];
        GfVec3f const na=vertexNormals[tri.ia]*outward;
        GfVec3f const nb=vertexNormals[tri.ib]*outward;
        GfVec3f const nc=vertexNormals[tri.ic]*outward;
        GfVec3f const flat=tri.normal*outward;
        GfVec3f const e1=tri.b-tri.a, e2=tri.c-tri.a;
        size_t index=0;
        bool any=false;
        for(int i=0;i<=k;++i) for(int j=0;i+j<=k;++j,++index) {
            float const u=float(i)/float(k), v=float(j)/float(k);
            GfVec3f const on=tri.a+e1*u+e2*v;
            GfVec3f n=na*(1.f-u-v)+nb*u+nc*v;
            float const length=n.GetLength();
            n = length>1e-6f ? n/length : flat;
            local[index]=on; localN[index]=n;
            gather(on+n*(probe+lift),&depth[index*6]);
            shaded[index]=HemisphereDepth(&depth[index*6],n);
            if(shaded[index]>kScalpShadowFloor) any=true;
        }
        if(!any) continue;
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
                remap[size_t(s)]=int(points.size());
                points.push_back(local[size_t(s)]+localN[size_t(s)]*lift);
                normals.push_back(localN[size_t(s)]);
                ids.push_back(int(triIndex*perTriangle+size_t(s)));
                for(int d=0;d<3;++d) {
                    tauP.push_back(depth[size_t(s)*6+d*2]);
                    tauN.push_back(depth[size_t(s)*6+d*2+1]);
                }
            }
            return remap[size_t(s)];
        };
        auto face=[&](int i0,int j0,int i1,int j1,int i2,int j2) {
            int const t0=slot(i0,j0), t1=slot(i1,j1), t2=slot(i2,j2);
            bool lit=false;
            for(int s:{t0,t1,t2})
                if(shaded[size_t(s)]>kScalpShadowFloor) lit=true;
            if(!lit) return;
            counts.push_back(3);
            indices.push_back(emit(i0,j0));
            indices.push_back(emit(i1,j1));
            indices.push_back(emit(i2,j2));
        };
        for(int i=0;i<k;++i) for(int j=0;i+j<k;++j) {
            face(i,j, i+1,j, i,j+1);
            if(i+j+2<=k) face(i+1,j, i+1,j+1, i,j+1);
        }
    }
    out->points=points; out->normals=normals;
    out->faceVertexCounts=counts; out->faceVertexIndices=indices;
    out->latticeIds=ids;
    out->lift=lift;
    // A lattice id has to fit an int; a surface too large for that publishes
    // a cap that is simply never merged.
    out->latticeKey = triangles.size()*perTriangle <= size_t(std::numeric_limits<int>::max())
        ? (key.h ? key.h : 1) : 0;
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
        GfVec3f lo=points[0], hi=points[0];
        for(GfVec3f const& p:points) for(int d=0;d<3;++d) {
            lo[d]=std::min(lo[d],p[d]); hi[d]=std::max(hi[d],p[d]);
        }
        out->extentMin=GfVec3d(lo); out->extentMax=GfVec3d(hi);
    }
    Hash hash;
    hash.Bytes(points.cdata(),points.size()*sizeof(GfVec3f));
    hash.Bytes(normals.cdata(),normals.size()*sizeof(GfVec3f));
    hash.Bytes(indices.cdata(),indices.size()*sizeof(int));
    hash.Bytes(tauP.cdata(),tauP.size()*sizeof(float));
    hash.Bytes(tauN.cdata(),tauN.size()*sizeof(float));
    hash.Bytes(ids.cdata(),ids.size()*sizeof(int));
    hash.Value(out->latticeKey);
    out->digest=hash.h;
    TF_DEBUG(USDGEN_FUR).Msg(
        "usdGen fur: scalp shadow %d x refinement, %zu points, %zu triangles, "
        "lift %.3g\n", k, points.size(), counts.size(), double(lift));
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
    uint64_t key;
    {
        Hash hash;
        hash.Value(params.voxelSize);
        hash.Value(params.maxDimension);
        hash.Value(params.resolution);
        hash.Value(params.opaqueBiasVoxels);
        hash.Value(params.opaqueMarkVoxels);
        hash.Value(params.scalpShadow);
        hash.Value(params.scalpMaxTriangles);
        hash.Value(params.localDensity);
        for (UsdGenFurOccluder const& occluder : params.occluders) {
            hash.Value(occluder.worldMatrix);
            hash.Bytes(occluder.points.cdata(), occluder.points.size()*sizeof(GfVec3f));
            hash.Bytes(occluder.faceVertexCounts.cdata(),
                       occluder.faceVertexCounts.size()*sizeof(int));
            hash.Bytes(occluder.faceVertexIndices.cdata(),
                       occluder.faceVertexIndices.size()*sizeof(int));
        }
        key = hash.h;
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
        // The density plane must be present exactly when it is asked for, so
        // a caller without a volume key still sees the switch flip.
        auto density=std::find_if(b.extraUniform.begin(),b.extraUniform.end(),
            [](auto const& plane){return plane.name==kDensityName;});
        bool const present=density!=b.extraUniform.end();
        if(params.localDensity ?
               !present || density->arity!=1 || density->f.size()!=b.points.size() :
               present)
            same=false;
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
    std::vector<std::vector<GfVec3f>> positions(tileCount);
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
            auto& p=positions[t]; p.resize(count);
            GfVec3f lo=tileLo[t], hi=tileHi[t];
#if defined(__aarch64__)
            // Identity fast path: copy + bounds + finiteness in 4-wide
            // deinterleaved quads. min/max over finite values (signed zeros
            // included) select lane-wise with no rounding, so splitting the
            // set across lanes is exact; the exponent-255 test flags NaN and
            // Inf alike, deferred to the end, and the partial positions are
            // discarded by the same throw the scalar loop reports. Bit-
            // identical bounds, same invalid[] verdict.
            static_assert(sizeof(GfVec3f)==3*sizeof(float),
                          "vec3 is 3 contiguous floats");
            if(identity) {
                float const* const src=
                    reinterpret_cast<float const*>(tile.points.cdata());
                float* const dst=reinterpret_cast<float*>(p.data());
                float32x4_t lox=vdupq_n_f32(lo[0]), loy=vdupq_n_f32(lo[1]),
                              loz=vdupq_n_f32(lo[2]);
                float32x4_t hix=vdupq_n_f32(hi[0]), hiy=vdupq_n_f32(hi[1]),
                              hiz=vdupq_n_f32(hi[2]);
                uint32x4_t bad=vdupq_n_u32(0);
                uint32x4_t const e255=vdupq_n_u32(255);
                size_t i=0;
                size_t const m4=count&~size_t(3);
                for(;i<m4;i+=4) {
                    float32x4x3_t const v=vld3q_f32(src+i*3);
                    lox=vminq_f32(lox,v.val[0]); loy=vminq_f32(loy,v.val[1]);
                    loz=vminq_f32(loz,v.val[2]);
                    hix=vmaxq_f32(hix,v.val[0]); hiy=vmaxq_f32(hiy,v.val[1]);
                    hiz=vmaxq_f32(hiz,v.val[2]);
                    bad=vorrq_u32(bad,vceqq_u32(vshrq_n_u32(vshlq_n_u32(
                        vreinterpretq_u32_f32(v.val[0]),1),24),e255));
                    bad=vorrq_u32(bad,vceqq_u32(vshrq_n_u32(vshlq_n_u32(
                        vreinterpretq_u32_f32(v.val[1]),1),24),e255));
                    bad=vorrq_u32(bad,vceqq_u32(vshrq_n_u32(vshlq_n_u32(
                        vreinterpretq_u32_f32(v.val[2]),1),24),e255));
                    vst3q_f32(dst+i*3,v);
                }
                lo[0]=vminvq_f32(lox); lo[1]=vminvq_f32(loy);
                lo[2]=vminvq_f32(loz);
                hi[0]=vmaxvq_f32(hix); hi[1]=vmaxvq_f32(hiy);
                hi[2]=vmaxvq_f32(hiz);
                if(vmaxvq_u32(bad)!=0) {
                    invalid[t]="non-finite fur point"; return;
                }
                for(;i<count;++i) {
                    GfVec3f const w=tile.points[i];
                    for(int k=0;k<3;++k) {
                        if(!std::isfinite(w[k])) {
                            invalid[t]="non-finite fur point"; return;
                        }
                        lo[k]=std::min(lo[k],w[k]); hi[k]=std::max(hi[k],w[k]);
                    }
                    p[i]=w;
                }
                tileLo[t]=lo; tileHi[t]=hi;
                return;
            }
#endif
            for(size_t i=0;i<count;++i) {
                GfVec3f const w = identity ? tile.points[i] : m.Transform(tile.points[i]);
                for(int k=0;k<3;++k) {
                    if(!std::isfinite(w[k])) { invalid[t]="non-finite fur point"; return; }
                    lo[k]=std::min(lo[k],w[k]); hi[k]=std::max(hi[k],w[k]);
                }
                p[i]=w;
            }
            tileLo[t]=lo; tileHi[t]=hi;
        });
    }
    GfVec3f lo(std::numeric_limits<float>::max()), hi(-std::numeric_limits<float>::max());
    size_t pointCount=0;
    for(size_t t=0;t<tileCount;++t) {
        if(invalid[t]) throw std::invalid_argument(invalid[t]);
        pointCount+=positions[t].size();
        if(positions[t].empty()) continue;
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
    std::vector<GfVec3f> density(cells,GfVec3f(0));
    size_t const slabs=std::max<size_t>(1,std::min<size_t>(size_t(grid.n[2]),64));
    {
        TRACE_SCOPE("usdGen occlusion: splat");
        auto const clock0=std::chrono::steady_clock::now();
        std::vector<JobDensity> local(jobs.size());
        ForEach(dispatcher, jobs.size(), [&](size_t index) {
            Job const& job=jobs[index];
            auto const& tile=(*tiles)[job.tile]; auto const& p=positions[job.tile];
            if(!job.cvCount) return;
            GfVec3f jobLo(std::numeric_limits<float>::max()), jobHi(-jobLo[0]);
            for(size_t i=job.firstCv;i<job.firstCv+job.cvCount;++i)
                for(int k=0;k<3;++k) {
                    jobLo[k]=std::min(jobLo[k],p[i][k]); jobHi[k]=std::max(jobHi[k],p[i][k]);
                }
            JobDensity& out=local[index];
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
            for(JobDensity const& job:local) {
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

    // --- local hair density, taken before the opaque shell is injected -------
    // density[cell][k] is the expected crossings of a ray traversing that one
    // cell along axis k (the sweeps below integrate exactly this into tau), so
    // the axis mean is a non-directional "how much hair is here" per voxel.
    std::vector<float> hairDensity;
    if(params.localDensity) {
        TRACE_SCOPE("usdGen occlusion: local density");
        hairDensity.resize(cells);
        ForEach(dispatcher, slabs, [&](size_t slab) {
            size_t const first=(slab*cells)/slabs, last=((slab+1)*cells)/slabs;
            for(size_t i=first;i<last;++i) {
                GfVec3f const& d=density[i];
                hairDensity[i]=std::min(UsdGenFurTauClamp,(d[0]+d[1]+d[2])*(1.f/3.f));
            }
        });
    }

    // --- occluder triangles, and which side of them the solid is on ----------
    // Built before the shell is injected, because the scalp-shadow cap needs
    // hair-only depth and the side vote needs hair-only density.
    std::vector<Triangle> triangles;
    std::vector<GfVec3f> vertices, vertexNormals;
    float inward = -1.f;
    if(!params.occluders.empty()) {
        TRACE_SCOPE("usdGen occlusion: occluder triangles");
        for(UsdGenFurOccluder const& occluder:params.occluders) {
            GfMatrix4f const m(occluder.worldMatrix);
            bool const identity = occluder.worldMatrix == GfMatrix4d(1.0);
            uint32_t const base=uint32_t(vertices.size());
            for(GfVec3f const& p:occluder.points) {
                GfVec3f const w=identity?p:m.Transform(p);
                for(int k=0;k<3;++k) if(!std::isfinite(w[k]))
                    throw std::invalid_argument("non-finite occluder point");
                vertices.push_back(w);
            }
            size_t cursor=0;
            for(int face:occluder.faceVertexCounts) {
                if(face<0) throw std::invalid_argument("negative occluder face vertex count");
                if(cursor+size_t(face)>occluder.faceVertexIndices.size())
                    throw std::invalid_argument("invalid occluder face cardinality");
                if(face<3) { cursor+=size_t(face); continue; }
                auto vertex=[&](size_t at)->uint32_t {
                    int const index=occluder.faceVertexIndices[cursor+at];
                    if(index<0 || size_t(index)>=occluder.points.size())
                        throw std::invalid_argument("occluder face index out of range");
                    return base+uint32_t(index);
                };
                uint32_t const first=vertex(0);
                for(size_t k=1;k+1<size_t(face);++k) {
                    uint32_t const ib=vertex(k), ic=vertex(k+1);
                    Triangle tri{vertices[first],vertices[ib],vertices[ic],
                                 GfVec3f(0),first,ib,ic};
                    GfVec3f const n=GfCross(tri.b-tri.a,tri.c-tri.a);
                    float const length=n.GetLength();
                    if(!(length>0.f)) continue;
                    tri.normal=n/length;
                    triangles.push_back(tri);
                }
                cursor+=size_t(face);
            }
        }
        // Area-weighted vertex normals: the cross product's length is twice the
        // triangle's area, so accumulating it unnormalised is the weighting.
        vertexNormals.assign(vertices.size(),GfVec3f(0));
        for(Triangle const& tri:triangles) {
            GfVec3f const weighted=GfCross(tri.b-tri.a,tri.c-tri.a);
            vertexNormals[tri.ia]+=weighted;
            vertexNormals[tri.ib]+=weighted;
            vertexNormals[tri.ic]+=weighted;
        }
        for(size_t i=0;i<vertexNormals.size();++i) {
            float const length=vertexNormals[i].GetLength();
            if(length>1e-12f) vertexNormals[i]/=length;
        }
        // Which side of the winding the solid is on: the hair grows out of the
        // surface, so the side carrying less fur density is the inside. Voting
        // over the triangles keeps this right for either winding order and for
        // a flat emitter, where a centroid test would be degenerate.
        float vote=0;
        {
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
                Triangle const& tri=triangles[i];
                GfVec3f const e1=tri.b-tri.a, e2=tri.c-tri.a;
                // A lattice, not the centroid: one triangle can be larger than
                // the whole groom, and then its centroid says nothing.
                float const extent=std::max({e1.GetLength(),e2.GetLength(),
                                             (tri.c-tri.b).GetLength()});
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
    std::array<std::vector<float>,6> tau;
    auto sweep=[&] {
        // One task per direction leaves workers idle and recomputes the cell
        // index per step; worse, the strided axes walk one column at a time,
        // so every store misses a line its neighbour column just touched.
        // Flatten (direction, column-block) over the dispatcher instead: each
        // block walks 8 adjacent columns of the unit-stride dimension in
        // lockstep with a strided index, sharing cache lines across the
        // block. The accumulation order within a column is unchanged, so the
        // sweep is bit-identical; every cell is still written exactly once.
        size_t const stride[3]={
            1, size_t(grid.n[0]), size_t(grid.n[0])*size_t(grid.n[1])};
        struct SweepDir {
            int axis, na, nInner, nOuter, innerBlocks;
            size_t strideA, strideU, strideV;
            bool innerIsU;
        };
        SweepDir dirs[6];
        size_t first[7];
        first[0]=0;
        for(int d=0;d<6;++d) {
            int const axis=d/2;
            int const au=(axis+1)%3, av=(axis+2)%3;
            SweepDir& sd=dirs[d];
            sd.axis=axis;
            sd.na=grid.n[axis];
            // Axis 1 steps along y, so its unit-stride column dimension is
            // v (x); the other axes block over u (axis 0's columns are
            // sequential rows either way).
            sd.innerIsU=(axis!=1);
            sd.nInner=sd.innerIsU?grid.n[au]:grid.n[av];
            sd.nOuter=sd.innerIsU?grid.n[av]:grid.n[au];
            sd.strideA=stride[axis];
            sd.strideU=stride[au];
            sd.strideV=stride[av];
            sd.innerBlocks=(sd.nInner+7)/8;
            first[d+1]=first[d]+size_t(sd.nOuter)*size_t(sd.innerBlocks);
        }
        for(int d=0;d<6;++d) tau[d].assign(cells,0.f);
        ForEach(dispatcher, first[6], [&](size_t block) {
            int d=0;
            while(block>=first[d+1]) ++d;
            SweepDir const& sd=dirs[d];
            size_t const local=block-first[d];
            int const outer=int(local/size_t(sd.innerBlocks));
            int const inner0=int(local%size_t(sd.innerBlocks))*8;
            int const nb=std::min(8,sd.nInner-inner0);
            float* const out=tau[d].data();
            int const last=sd.na-1;
            // Direction parity is the walk sense: odd starts at index 0 and
            // ascends, even starts at the far side and descends.
            bool const ascending=(d%2)!=0;
            ptrdiff_t const dStride=ascending ?
                ptrdiff_t(sd.strideA) : -ptrdiff_t(sd.strideA);
            auto columnBase=[&](int b) {
                int const inner=inner0+b;
                int const u=sd.innerIsU?inner:outer;
                int const v=sd.innerIsU?outer:inner;
                return size_t(u)*sd.strideU+size_t(v)*sd.strideV;
            };
            if(nb==8) {
                float sums[8]={0,0,0,0,0,0,0,0};
                size_t idx[8];
                for(int b=0;b<8;++b)
                    idx[b]=ascending ? columnBase(b)
                                     : columnBase(b)+size_t(last)*sd.strideA;
                for(int step=0;step<sd.na;++step) {
                    for(int b=0;b<8;++b) {
                        float const depth=density[idx[b]][sd.axis];
                        float const value=std::min(
                            UsdGenFurTauClamp,sums[b]+0.5f*depth);
                        sums[b]+=depth;
                        out[idx[b]]=value;
                        idx[b]=size_t(ptrdiff_t(idx[b])+dStride);
                    }
                }
            } else {
                for(int b=0;b<nb;++b) {
                    size_t idx=ascending ? columnBase(b)
                                         : columnBase(b)+size_t(last)*sd.strideA;
                    float sum=0;
                    for(int step=0;step<sd.na;++step) {
                        float const depth=density[idx][sd.axis];
                        float const value=std::min(
                            UsdGenFurTauClamp,sum+0.5f*depth);
                        sum+=depth;
                        out[idx]=value;
                        idx=size_t(ptrdiff_t(idx)+dStride);
                    }
                }
            }
        });
    };
    // Trilinear gather of the six depths at one point.
    auto gather=[&](GfVec3f const& at, float values[6]) {
        int c[3]; float w[3];
        grid.Corners(at,c,w);
        for(int d=0;d<6;++d) values[d]=0.f;
        for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
            float const weight=(x?w[0]:1-w[0])*(y?w[1]:1-w[1])*(z?w[2]:1-w[2]);
            size_t const idx=grid.Index(c[0]+x,c[1]+y,c[2]+z);
            for(int d=0;d<6;++d) values[d]+=weight*tau[d][idx];
        }
    };

    if(params.scalpShadow && scalpShadow && !triangles.empty()) {
        TRACE_SCOPE("usdGen occlusion: scalp shadow");
        sweep();   // hair only: the head shadowing itself is the skin shader's job
        BuildScalpShadow(triangles, vertexNormals, inward,
                         params.scalpMaxTriangles, gather, scalpShadow);
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

        // How much solid lies behind each vertex, so a surface's shell stops
        // at the far side of a solid thinner than the shell is deep (an ear)
        // instead of coming out into the hair beyond it.
        // The marking's own lattice, so each sample knows its thickness.
        auto latticeOf=[&](Triangle const& tri) {
            float const extent=std::max({(tri.b-tri.a).GetLength(),
                                         (tri.c-tri.a).GetLength(),
                                         (tri.c-tri.b).GetLength()});
            return std::clamp(int(std::ceil(1.5f*extent*grid.invH)),1,512);
        };
        SolidThickness const thickness=MeasureSolidThickness(
            triangles,inward,float(bias+mark+1)*h,latticeOf,dispatcher);

        // A shell taken with max(), so it does not depend on how densely the
        // triangles happen to be sampled. Slabs own disjoint cells, and each
        // slab only walks the triangles whose shell can reach it.
        std::vector<float> mask(cells,0.f);
        std::vector<std::vector<int>> buckets(slabs);
        for(size_t i=0;i<triangles.size();++i) {
            Triangle const& tri=triangles[i];
            GfVec3f const step=tri.normal*(inward*h);
            float low=std::numeric_limits<float>::max(), high=-low;
            for(GfVec3f const& corner:{tri.a,tri.b,tri.c})
                for(int layer=0;layer<mark;++layer) {
                    float const z=(corner+step*(float(bias+layer)+0.5f))[2];
                    low=std::min(low,z); high=std::max(high,z);
                }
            int const first=int(std::floor((low-grid.origin[2])*grid.invH))-1;
            int const last=int(std::floor((high-grid.origin[2])*grid.invH))+1;
            if(last<0 || first>=grid.n[2]) continue;
            for(size_t slab=0;slab<slabs;++slab) {
                int const z0=int((slab*size_t(grid.n[2]))/slabs);
                int const z1=int(((slab+1)*size_t(grid.n[2]))/slabs);
                if(last>=z0 && first<z1) buckets[slab].push_back(int(i));
            }
        }
        ForEach(dispatcher,slabs,[&](size_t slab) {
            int const z0=int((slab*size_t(grid.n[2]))/slabs);
            int const z1=int(((slab+1)*size_t(grid.n[2]))/slabs);
            if(z0>=z1) return;
            for(int index:buckets[slab]) {
                Triangle const& tri=triangles[size_t(index)];
                GfVec3f const e1=tri.b-tri.a, e2=tri.c-tri.a;
                // Barycentric lattice fine enough that adjacent samples'
                // trilinear stencils overlap, so the shell has no pinholes
                // (chosen by latticeOf above, with each sample's thickness).
                int const n=thickness.lattice[size_t(index)];
                GfVec3f const step=tri.normal*(inward*h);
                for(int i=0;i<=n;++i) for(int j=0;i+j<=n;++j) {
                    float const u=float(i)/float(n), v=float(j)/float(n);
                    GfVec3f const on=tri.a+e1*u+e2*v;
                    float const solid=thickness.At(size_t(index),i,j);
                    for(int layer=0;layer<mark;++layer) {
                        // The far edge of this layer's splat must still be
                        // inside the solid.
                        if((float(bias+layer)+1.f)*h>solid) break;
                        GfVec3f const at=on+step*(float(bias+layer)+0.5f);
                        if(!grid.Contains(at)) continue;
                        int c[3]; float w[3];
                        grid.Corners(at,c,w);
                        for(int z=0;z<2;++z) {
                            if(c[2]+z<z0 || c[2]+z>=z1) continue;
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
    // separate strided arrays. The 8-deep pad keeps the aarch64 gather's
    // 8-lane tuple loads in bounds at the last cell (lanes 6-7 discarded).
    std::vector<uint16_t> depths(cells*6+8);
    {
        TRACE_SCOPE("usdGen occlusion: interleave");
        ForEach(dispatcher, slabs, [&](size_t slab) {
            size_t const first=(slab*cells)/slabs, last=((slab+1)*cells)/slabs;
            for(size_t i=first;i<last;++i)
                for(int d=0;d<6;++d)
                    depths[i*6+d]=uint16_t(tau[d][i]*kDepthScale+0.5f);
        });
        for(auto& plane:tau) std::vector<float>().swap(plane);
    }

    // --- gather: every receiver reads its six depths once --------------------
    {
        TRACE_SCOPE("usdGen occlusion: gather");
        std::vector<std::array<float*,2>> planes(tileCount, {nullptr,nullptr});
        std::vector<float*> densityPlanes(tileCount, nullptr);
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
            if(params.localDensity) {
                UsdGenPlane plane;plane.name=TfToken(kDensityName);
                plane.interpolation=TfToken("vertex");plane.type=TfToken("float");plane.arity=1;
                plane.f.resize((*tiles)[t].points.size(), [](float* b, float* e) {
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
            if(params.localDensity) {
                auto it=std::find_if(out.begin(),out.end(),
                    [](auto const& p){return p.name==kDensityName;});
                densityPlanes[t]=it->f.data();
            }
        }
        size_t const densityY=size_t(grid.n[0]), densityZ=size_t(grid.n[0])*grid.n[1];
        size_t const strideY=size_t(grid.n[0])*6, strideZ=size_t(grid.n[0])*grid.n[1]*6;
        ForEach(dispatcher, jobs.size(), [&](size_t index) {
            Job const& job=jobs[index];
            // names[0] (P) holds the even directions, names[1] (N) the odd.
            float* const p0=planes[job.tile][0];
            float* const p1=planes[job.tile][1];
            float* const pd=densityPlanes[job.tile];
            auto const& p=positions[job.tile];
            for(size_t i=job.firstCv;i<job.firstCv+job.cvCount;++i) {
                int c[3]; float w[3];
                grid.CornersInner(p[i],c,w);
                uint16_t const* const cell=&depths[grid.Index(c[0],c[1],c[2])*6];
#if defined(__aarch64__)
                // The six depth lanes accumulate in two NEON quads, corners
                // in the same order, each lane the same fused
                // acc + weight * float(at[d]) the scalar loop compiles to
                // (u16->f32 is exact; multiply commutes exactly), so lanes
                // 0-5 are bit-identical and 6-7 are never stored. The weight
                // is the same scalar expression; scalar and NEON obey the
                // same FPCR, so denormal weights round the same lane-wise.
                float values[6];
                float32x4_t acc0=vdupq_n_f32(0.f), acc1=vdupq_n_f32(0.f);
                for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
                    float const weight=(x?w[0]:1-w[0])*(y?w[1]:1-w[1])*(z?w[2]:1-w[2]);
                    uint16_t const* const at=cell+x*6+y*strideY+z*strideZ;
                    float32x4_t const vw=vdupq_n_f32(weight);
                    uint16x8_t const u=vld1q_u16(at);
                    float32x4_t const v0=vcvtq_f32_u32(vmovl_u16(vget_low_u16(u)));
                    float32x4_t const v1=vcvtq_f32_u32(vmovl_u16(vget_high_u16(u)));
                    acc0=vfmaq_f32(acc0,v0,vw);
                    acc1=vfmaq_f32(acc1,v1,vw);
                }
                vst1q_f32(values,acc0);
                vst1_f32(values+4,vget_low_f32(acc1));
#else
                float values[6]={0,0,0,0,0,0};
                for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
                    float const weight=(x?w[0]:1-w[0])*(y?w[1]:1-w[1])*(z?w[2]:1-w[2]);
                    uint16_t const* const at=cell+x*6+y*strideY+z*strideZ;
                    for(int d=0;d<6;++d) values[d]+=weight*float(at[d]);
                }
#endif
                for(int d=0;d<3;++d) {
                    p0[i*3+d]=values[d*2]*(1.f/kDepthScale);
                    p1[i*3+d]=values[d*2+1]*(1.f/kDepthScale);
                }
                if(pd) {
                    // Same corners and weights as the depths, over the
                    // hair-only scalar grid (plain float: one value per cell).
                    float const* const at=&hairDensity[grid.Index(c[0],c[1],c[2])];
                    float sum=0.f;
                    for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
                        float const weight=(x?w[0]:1-w[0])*(y?w[1]:1-w[1])*(z?w[2]:1-w[2]);
                        sum+=weight*at[x+y*densityY+z*densityZ];
                    }
                    pd[i]=sum;
                }
            }
        });
    }
    return true;
}

UsdGenScalpShadowPublication UsdGenMergeScalpShadows(
    std::vector<UsdGenScalpShadowContribution> const& parts)
{
    TRACE_FUNCTION();
    UsdGenScalpShadowPublication out;
    // The first usable part fixes the lattice; the rest must share it.
    UsdGenScalpShadowContribution const* first=nullptr;
    std::vector<UsdGenScalpShadowContribution const*> used;
    for(UsdGenScalpShadowContribution const& part:parts) {
        UsdGenScalpShadowPublication const* cap=part.cap;
        if(!cap || cap->IsEmpty() || !cap->latticeKey) continue;
        if(cap->latticeIds.size()!=cap->points.size() ||
           cap->normals.size()!=cap->points.size()) continue;
        if(first && cap->latticeKey!=first->cap->latticeKey) continue;
        bool planes=true;
        for(char const* name:names) {
            bool found=false;
            for(UsdGenPlane const& plane:cap->extraUniform)
                if(plane.name==TfToken(name)) {
                    found = plane.arity==3 && plane.f.size()==cap->points.size()*3;
                    break;
                }
            planes = planes && found;
        }
        if(!planes) continue;
        if(!first) first=&part;
        used.push_back(&part);
    }
    if(!first) return out;
    UsdGenScalpShadowPublication const& lead=*first->cap;
    out.primPath=lead.primPath;
    out.purpose=lead.purpose;
    out.visibility=lead.visibility;
    out.materialPath=lead.materialPath;
    out.latticeKey=lead.latticeKey;
    out.lift=lead.lift;
    if(used.size()==1) {
        // Nothing to sum: share the part's arrays outright.
        out.points=lead.points; out.normals=lead.normals;
        out.faceVertexCounts=lead.faceVertexCounts;
        out.faceVertexIndices=lead.faceVertexIndices;
        out.extraUniform=lead.extraUniform;
        out.latticeIds=lead.latticeIds;
        out.extentMin=lead.extentMin; out.extentMax=lead.extentMax;
        out.digest=lead.digest;
        return out;
    }

    auto plane=[](UsdGenScalpShadowPublication const& cap,char const* name)
        -> VtFloatArray const& {
        for(UsdGenPlane const& p:cap.extraUniform)
            if(p.name==TfToken(name)) return p.f;
        static VtFloatArray const none;
        return none;
    };
    bool sameAlbedo=true;
    for(auto const* part:used)
        sameAlbedo = sameAlbedo && part->albedo==first->albedo;

    // Lattice id -> merged point. Ids are dense per surface (source triangle
    // times slots per triangle), so a flat table beats hashing.
    int maxId=-1;
    for(auto const* part:used)
        for(int id:part->cap->latticeIds) maxId=std::max(maxId,id);
    std::vector<int> slot(size_t(maxId)+1,-1);
    VtVec3fArray points, normals;
    VtIntArray ids;
    std::vector<float> tau[2];
    std::vector<GfVec3f> albedoSum;
    std::vector<float> albedoWeight;
    for(auto const* part:used) {
        UsdGenScalpShadowPublication const& cap=*part->cap;
        VtFloatArray const& p=plane(cap,names[0]);
        VtFloatArray const& n=plane(cap,names[1]);
        for(size_t i=0;i<cap.points.size();++i) {
            int const id=cap.latticeIds[i];
            if(id<0) continue;
            int& at=slot[size_t(id)];
            if(at<0) {
                at=int(points.size());
                points.push_back(cap.points[i]);
                normals.push_back(cap.normals[i]);
                ids.push_back(id);
                for(auto& t:tau) t.insert(t.end(),3,0.f);
                if(!sameAlbedo) { albedoSum.emplace_back(0.f); albedoWeight.push_back(0.f); }
            }
            size_t const o=size_t(at);
            float values[6];
            for(int d=0;d<3;++d) {
                tau[0][o*3+d]+=p[i*3+d];
                tau[1][o*3+d]+=n[i*3+d];
                values[d*2]=p[i*3+d]; values[d*2+1]=n[i*3+d];
            }
            if(!sameAlbedo) {
                // Weight each colour by how much of that hair this patch of
                // skin sees, the quantity the cull and the shader both use.
                float const w=HemisphereDepth(values,cap.normals[i]);
                albedoSum[o]+=part->albedo*w;
                albedoWeight[o]+=w;
            }
        }
    }
    // Every triangle any part kept, once. A part's triangles are emitted in
    // lattice order, so the same lattice triangle has the same id triple in
    // every part.
    VtIntArray counts, indices;
    std::set<std::array<int,3>> seen;
    for(auto const* part:used) {
        UsdGenScalpShadowPublication const& cap=*part->cap;
        for(size_t f=0;f+2<cap.faceVertexIndices.size();f+=3) {
            std::array<int,3> key;
            bool ok=true;
            for(int c=0;c<3;++c) {
                int const v=cap.faceVertexIndices[f+size_t(c)];
                ok = ok && v>=0 && size_t(v)<cap.latticeIds.size();
                key[size_t(c)] = ok ? cap.latticeIds[size_t(v)] : -1;
            }
            if(!ok || !seen.insert(key).second) continue;
            counts.push_back(3);
            for(int id:key) indices.push_back(slot[size_t(id)]);
        }
    }

    out.points=points; out.normals=normals; out.latticeIds=ids;
    out.faceVertexCounts=counts; out.faceVertexIndices=indices;
    for(int which=0;which<2;++which) {
        UsdGenPlane p;
        p.name=TfToken(names[which]);
        p.interpolation=TfToken("vertex");
        p.type=TfToken("float");
        p.arity=3;
        p.f.assign(tau[which].begin(),tau[which].end());
        out.extraUniform.push_back(std::move(p));
    }
    if(!sameAlbedo) {
        UsdGenPlane p;
        p.name=TfToken("furCapAlbedo");
        p.interpolation=TfToken("vertex");
        p.type=TfToken("float");
        p.arity=3;
        p.f.resize(points.size()*3);
        for(size_t i=0;i<points.size();++i) {
            // A point no part sees hair from keeps the lead colour; it is
            // not darkened anyway.
            GfVec3f const c = albedoWeight[i]>0.f
                ? albedoSum[i]/albedoWeight[i] : first->albedo;
            for(int d=0;d<3;++d) p.f[i*3+d]=c[d];
        }
        out.extraUniform.push_back(std::move(p));
    }
    GfVec3f lo=points[0], hi=points[0];
    for(GfVec3f const& q:points) for(int d=0;d<3;++d) {
        lo[d]=std::min(lo[d],q[d]); hi[d]=std::max(hi[d],q[d]);
    }
    out.extentMin=GfVec3d(lo); out.extentMax=GfVec3d(hi);
    Hash hash;
    for(auto const* part:used) {
        hash.Value(part->cap->digest);
        hash.Value(part->albedo);
    }
    out.digest=hash.h;
    return out;
}
}
