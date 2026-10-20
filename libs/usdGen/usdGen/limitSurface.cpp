#include "usdGen/limitSurface.h"
#include "usdGen/digest.h"
#include "pxr/imaging/pxOsd/meshTopology.h"
#include "pxr/imaging/pxOsd/refinerFactory.h"
#include "pxr/base/gf/vec2d.h"
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/patchMap.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/ptexIndices.h>
#include <algorithm>
#include <cmath>
#include <vector>
PXR_NAMESPACE_USING_DIRECTIVE
namespace usdGen {
namespace {
double Cross2(GfVec2d const& a, GfVec2d const& b) { return a[0]*b[1]-a[1]*b[0]; }
// (u,v) of p in the bilinear quad a(0,0) b(1,0) c(1,1) d(0,1), clamped to
// [0,1]^2. The quads used here are convex kites, so the root inside the unit
// square is the one wanted.
GfVec2d InverseBilinear(GfVec2d const& p, GfVec2d const& a, GfVec2d const& b,
                        GfVec2d const& c, GfVec2d const& d) {
    GfVec2d const e=b-a, f=d-a, g=a-b+c-d, h=p-a;
    double const k2=Cross2(g,f), k1=Cross2(e,f)+Cross2(h,g), k0=Cross2(h,e);
    auto uFor=[&](double v) {
        GfVec2d const den=e+g*v;
        return std::abs(den[0])>=std::abs(den[1]) ? (h[0]-f[0]*v)/den[0] : (h[1]-f[1]*v)/den[1];
    };
    double u=0, v=0;
    if(std::abs(k2)<1e-12) {
        v=std::abs(k1)>1e-300 ? -k0/k1 : 0; u=uFor(v);
    } else {
        double const w=std::sqrt(std::max(0.0,k1*k1-4*k0*k2));
        v=(-k1-w)/(2*k2); u=uFor(v);
        if(u<-1e-6||u>1+1e-6||v<-1e-6||v>1+1e-6) {v=(-k1+w)/(2*k2); u=uFor(v);}
    }
    return GfVec2d(std::clamp(u,0.0,1.0),std::clamp(v,0.0,1.0));
}
PxOsdMeshTopology Topology(UsdGenSurfaceDesc const& s) {
    PxOsdSubdivTags tags(s.interpolateBoundary, s.faceVaryingLinearInterpolation,
        s.creaseMethod, s.triangleSubdivisionRule, s.creaseIndices, s.creaseLengths,
        s.creaseSharpnesses, s.cornerIndices, s.cornerSharpnesses);
    // Evaluate in authored corner order. Left-handedness changes the normal,
    // not Ptex's corner coordinate convention.
    return PxOsdMeshTopology(s.subdivisionScheme, TfToken("rightHanded"),
        s.faceVertexCounts, s.faceVertexIndices, s.holeIndices, tags);
}
struct Vertex {
    GfVec3f p{0};
    void Clear() { p=GfVec3f(0); }
    void AddWithWeight(Vertex const& v, float w) { p+=v.p*w; }
};
}
struct UsdGenLimitSurface::State {
    std::unique_ptr<OpenSubdiv::Far::PatchTable> patches;
    std::unique_ptr<OpenSubdiv::Far::PatchMap> map;
    std::vector<Vertex> values;
    VtIntArray holes;
    std::vector<int> faceSize;   // corners per coarse face
    std::vector<int> ptexBase;   // first Ptex face of each coarse face (n-gons own n)
};
UsdGenLimitSurface::UsdGenLimitSurface() = default;
UsdGenLimitSurface::~UsdGenLimitSurface() = default;
namespace {
void FeedSubdivText(uint64_t &h, TfToken const& tok) {
    std::string const& str = tok.GetString();
    h = UsdGenDigestBytes(str.data(), str.size(), h);
    UsdGenDigestMixWord(h, uint64_t(str.size()));
}
template <class Array>
void FeedSubdivArray(uint64_t &h, Array const& values) {
    h = UsdGenDigestBytes(values.cdata(),
        values.size() * sizeof(*values.cdata()), h);
    UsdGenDigestMixWord(h, uint64_t(values.size()));
}
// Tags, holes, creases, corners: the subdivision inputs OUTSIDE the two
// face arrays. Shared by both entry points so the coverage cannot drift.
void FeedSubdivTags(uint64_t &h, UsdGenSurfaceDesc const& s) {
    FeedSubdivText(h, s.subdivisionScheme);
    FeedSubdivText(h, s.orientation);
    FeedSubdivText(h, s.interpolateBoundary);
    FeedSubdivText(h, s.faceVaryingLinearInterpolation);
    FeedSubdivText(h, s.triangleSubdivisionRule);
    FeedSubdivText(h, s.creaseMethod);
    FeedSubdivArray(h, s.holeIndices);
    FeedSubdivArray(h, s.creaseIndices);
    FeedSubdivArray(h, s.creaseLengths);
    FeedSubdivArray(h, s.creaseSharpnesses);
    FeedSubdivArray(h, s.cornerIndices);
    FeedSubdivArray(h, s.cornerSharpnesses);
}
void FeedSubdivTopology(uint64_t &h, UsdGenSurfaceDesc const& s) {
    FeedSubdivArray(h, s.faceVertexCounts);
    FeedSubdivArray(h, s.faceVertexIndices);
}
} // namespace
// Same logical coverage as Topology(s).ComputeHash() plus s.orientation,
// hashed with the 4-lane word-wise FNV instead of SpookyHash (which
// re-streamed ~20MB of topology arrays at ~half the per-byte throughput).
// Tokens feed as string bytes rather than TfToken rep bytes, so the digest
// is deterministic across processes (values stay internal keys, never
// persisted). Sizes mix between fields so no two distinct field sequences
// concatenate ambiguously.
uint64_t UsdGenSubdivisionDigest(UsdGenSurfaceDesc const& s) {
    uint64_t h = UsdGenDigestOffset;
    FeedSubdivTags(h, s);
    FeedSubdivTopology(h, s);
    return h;
}
uint64_t UsdGenSubdivisionTagsDigest(UsdGenSurfaceDesc const& s) {
    uint64_t h = UsdGenDigestOffset;
    FeedSubdivTags(h, s);
    return h;
}
bool UsdGenLimitSurface::Build(UsdGenSurfaceDesc const& s, int level, std::string* error) {
    _state.reset();
    auto fail=[&](char const* e){if(error)*error=e;return false;};
    if(level<1 || level>6) return fail("subdivisionLevel must be in [1,6]");
    if(s.subdivisionScheme!=TfToken("catmullClark"))
        return fail("limit scatter requires subdivisionScheme=catmullClark");
    if(s.faceVertexCounts.empty() || s.restPoints.empty()) return fail("empty limit surface");
    size_t corners=0;
    for(int n:s.faceVertexCounts) {if(n<3) return fail("limit surface faces need at least 3 corners"); corners+=size_t(n);}
    if(s.faceVertexIndices.size()!=corners) return fail("inconsistent limit topology");
    for(int i:s.faceVertexIndices) if(i<0 || size_t(i)>=s.restPoints.size()) return fail("invalid limit vertex index");
    for(auto const& p:s.restPoints) for(int j=0;j<3;++j)
        if(!std::isfinite(p[j])) return fail("non-finite limit point");
    for(int f:s.holeIndices) if(f<0 || size_t(f)>=s.faceVertexCounts.size()) return fail("invalid hole face");
    size_t nc=0,ne=0;
    for(int n:s.creaseLengths) {if(n<2)return fail("invalid crease length");nc+=n;ne+=n-1;}
    if(nc!=s.creaseIndices.size() ||
       (s.creaseSharpnesses.size()!=s.creaseLengths.size() && s.creaseSharpnesses.size()!=ne))
        return fail("inconsistent crease tags");
    if(s.cornerIndices.size()!=s.cornerSharpnesses.size()) return fail("inconsistent corner tags");
    for(auto const* ids:{&s.creaseIndices,&s.cornerIndices})
        for(int i:*ids) if(i<0 || size_t(i)>=s.restPoints.size()) return fail("invalid sharpness vertex");
    for(auto const* weights:{&s.creaseSharpnesses,&s.cornerSharpnesses})
        for(float w:*weights) if(!std::isfinite(w)||w<0) return fail("invalid sharpness");
    auto refiner=PxOsdRefinerFactory::Create(Topology(s));
    if(!refiner) return fail("OpenSubdiv could not create the topology refiner");
    OpenSubdiv::Far::PatchTableFactory::Options opts(level);
    opts.SetEndCapType(OpenSubdiv::Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);
    refiner->RefineAdaptive(opts.GetRefineAdaptiveOptions());
    auto state=std::make_unique<State>();
    state->patches.reset(OpenSubdiv::Far::PatchTableFactory::Create(*refiner,opts));
    if(!state->patches) return fail("OpenSubdiv could not create limit patches");
    int total=refiner->GetNumVerticesTotal();
    state->values.resize(total+state->patches->GetNumLocalPoints());
    int offset=refiner->GetLevel(0).GetNumVertices();
    for(int i=0;i<offset;++i) state->values[i].p=s.restPoints[i];
    OpenSubdiv::Far::PrimvarRefiner pv(*refiner);
    int previous=0;
    for(int l=1;l<=refiner->GetMaxLevel();++l) {
        auto* src=state->values.data()+previous;auto* dst=state->values.data()+offset;
        pv.Interpolate(l,src,dst);
        previous=offset;offset+=refiner->GetLevel(l).GetNumVertices();
    }
    state->patches->ComputeLocalPointValues(state->values.data(),state->values.data()+total);
    state->map=std::make_unique<OpenSubdiv::Far::PatchMap>(*state->patches);
    state->holes=s.holeIndices;
    OpenSubdiv::Far::PtexIndices const ptex(*refiner);
    state->faceSize.assign(s.faceVertexCounts.begin(),s.faceVertexCounts.end());
    state->ptexBase.resize(s.faceVertexCounts.size());
    for(size_t f=0;f<state->ptexBase.size();++f) state->ptexBase[f]=ptex.GetFaceId(int(f));
    _state=std::move(state);
    return true;
}
namespace {
bool EvaluatePtex(OpenSubdiv::Far::PatchTable const& patches, OpenSubdiv::Far::PatchMap const& map,
                  std::vector<Vertex> const& values, int ptexFace, float u, float v,
                  GfVec3f* p, GfVec3f* du, GfVec3f* dv) {
    auto const* h=map.FindPatch(ptexFace,u,v);
    if(!h) return false;
    float w[20],wu[20],wv[20];
    patches.EvaluateBasis(*h,u,v,w,wu,wv);
    auto ids=patches.GetPatchVertices(*h);
    *p=*du=*dv=GfVec3f(0);
    for(int i=0;i<ids.size();++i){auto const& q=values[ids[i]].p;*p+=q*w[i];*du+=q*wu[i];*dv+=q*wv[i];}
    for(auto const* value:{p,du,dv})for(int j=0;j<3;++j)if(!std::isfinite((*value)[j]))return false;
    return true;
}
}
bool UsdGenLimitSurface::EvaluateCagePoint(int face, int const corners[3], GfVec3f const& bary,
                                           GfVec3f* p, GfVec3f* du, GfVec3f* dv) const {
    if(!_state || !p || !du || !dv || !corners || face<0 || size_t(face)>=_state->faceSize.size() || IsHole(face))
        return false;
    int const n=_state->faceSize[size_t(face)];
    for(int k=0;k<3;++k) if(corners[k]<0||corners[k]>=n||!std::isfinite(bary[k])) return false;
    double const sum=double(bary[0])+bary[1]+bary[2];
    if(!(sum>1e-12)) return false;
    // The face in a canonical layout: a quad as its unit (u,v) square, any
    // other polygon as the regular n-gon whose corner k sits at angle 2*pi*k/n.
    auto layout=[n](int k) {
        if(n==4) {
            static GfVec2d const square[4]={GfVec2d(0,0),GfVec2d(1,0),GfVec2d(1,1),GfVec2d(0,1)};
            return square[k];
        }
        double const a=2*3.14159265358979323846*double(k)/double(n);
        return GfVec2d(std::cos(a),std::sin(a));
    };
    GfVec2d x(0);
    for(int k=0;k<3;++k) x+=layout(corners[k])*(double(bary[k])/sum);
    int const base=_state->ptexBase[size_t(face)];
    if(n==4) {
        float const u=float(std::clamp(x[0],0.0,1.0)), v=float(std::clamp(x[1],0.0,1.0));
        return EvaluatePtex(*_state->patches,*_state->map,_state->values,base,u,v,p,du,dv);
    }
    // Ptex sub-face k of an n-gon spans corner k (0,0), the midpoint of the
    // edge to corner k+1 (1,0), the face centre (1,1) and the midpoint of the
    // edge from corner k-1 (0,1); in the regular layout that is the corner's
    // kite, the points nearer corner k than any other.
    int sub=0;double best=-1e300;
    for(int k=0;k<n;++k){double const d=GfDot(x,layout(k));if(d>best){best=d;sub=k;}}
    GfVec2d const a=layout(sub), next=layout((sub+1)%n), prev=layout((sub+n-1)%n);
    GfVec2d const uv=InverseBilinear(x,a,(a+next)*0.5,GfVec2d(0),(a+prev)*0.5);
    return EvaluatePtex(*_state->patches,*_state->map,_state->values,base+sub,
                        float(uv[0]),float(uv[1]),p,du,dv);
}
bool UsdGenLimitSurface::IsHole(int face) const {
    return _state && std::find(_state->holes.begin(),_state->holes.end(),face)!=_state->holes.end();
}
bool UsdGenLimitSurface::Evaluate(int face,float u,float v,GfVec3f* p,GfVec3f* du,GfVec3f* dv) const {
    if(!_state || !p || !du || !dv || !std::isfinite(u)||!std::isfinite(v)||u<0||v<0||u>1||v>1||IsHole(face)) return false;
    if(face<0 || size_t(face)>=_state->faceSize.size() || _state->faceSize[size_t(face)]!=4) return false;
    return EvaluatePtex(*_state->patches,*_state->map,_state->values,_state->ptexBase[size_t(face)],u,v,p,du,dv);
}
}
