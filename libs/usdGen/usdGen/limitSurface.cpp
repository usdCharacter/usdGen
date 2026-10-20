#include "usdGen/limitSurface.h"
#include "usdGen/digest.h"
#include "pxr/imaging/pxOsd/meshTopology.h"
#include "pxr/imaging/pxOsd/refinerFactory.h"
#include "pxr/base/gf/vec2d.h"
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/patchMap.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/ptexIndices.h>
#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/surface.h>
#include <opensubdiv/bfr/tessellation.h>
#include <algorithm>
#include <cmath>
#include <limits>
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

struct UsdGenCollisionLimitSurface::State {
    struct Face {
        std::unique_ptr<OpenSubdiv::Bfr::Surface<double>> surface;
        std::vector<double> patchPoints;
    };
    PxOsdTopologyRefinerSharedPtr refiner;
    std::unique_ptr<OpenSubdiv::Bfr::RefinerSurfaceFactory<>> factory;
    std::vector<Face> faces;
};

UsdGenCollisionLimitSurface::UsdGenCollisionLimitSurface() = default;
UsdGenCollisionLimitSurface::~UsdGenCollisionLimitSurface() = default;
UsdGenCollisionLimitSurface::UsdGenCollisionLimitSurface(
    UsdGenCollisionLimitSurface&&) noexcept = default;
UsdGenCollisionLimitSurface& UsdGenCollisionLimitSurface::operator=(
    UsdGenCollisionLimitSurface&&) noexcept = default;

bool UsdGenCollisionLimitSurface::Build(UsdGenSurfaceDesc const& s,
                                        std::string* error) {
    _state.reset();
    auto fail = [&](std::string const& why) {
        if (error) *error = why;
        return false;
    };
    if (s.subdivisionScheme != TfToken("catmullClark") &&
        s.subdivisionScheme != TfToken("loop") &&
        s.subdivisionScheme != TfToken("bilinear"))
        return fail("unsupported subdivision scheme '" +
                    s.subdivisionScheme.GetString() + "'");
    if (s.points.empty() || s.faceVertexCounts.empty())
        return fail("empty posed subdivision mesh");
    size_t cornerCount = 0;
    for (int n : s.faceVertexCounts) {
        if (n < 3 || (s.subdivisionScheme == TfToken("loop") && n != 3))
            return fail("invalid face size for subdivision scheme");
        cornerCount += size_t(n);
    }
    if (cornerCount != s.faceVertexIndices.size())
        return fail("faceVertexCounts/faceVertexIndices mismatch");
    for (int i : s.faceVertexIndices)
        if (i < 0 || size_t(i) >= s.points.size())
            return fail("subdivision vertex index out of range");
    for (GfVec3f const& p : s.points)
        for (int k = 0; k < 3; ++k)
            if (!std::isfinite(p[k])) return fail("non-finite posed subdivision point");
    for (int f : s.holeIndices)
        if (f < 0 || size_t(f) >= s.faceVertexCounts.size())
            return fail("subdivision hole face out of range");
    size_t creaseCorners = 0, creaseEdges = 0;
    for (int n : s.creaseLengths) {
        if (n < 2) return fail("invalid crease length");
        creaseCorners += size_t(n);
        creaseEdges += size_t(n - 1);
    }
    if (creaseCorners != s.creaseIndices.size() ||
        (s.creaseSharpnesses.size() != s.creaseLengths.size() &&
         s.creaseSharpnesses.size() != creaseEdges) ||
        s.cornerIndices.size() != s.cornerSharpnesses.size())
        return fail("inconsistent subdivision crease/corner tags");
    for (auto const* ids : {&s.creaseIndices, &s.cornerIndices})
        for (int i : *ids)
            if (i < 0 || size_t(i) >= s.points.size())
                return fail("subdivision sharpness vertex out of range");
    for (auto const* weights : {&s.creaseSharpnesses, &s.cornerSharpnesses})
        for (float w : *weights)
            if (!std::isfinite(w) || w < 0.0f)
                return fail("invalid subdivision sharpness");

    auto state = std::make_unique<State>();
    state->refiner = PxOsdRefinerFactory::Create(Topology(s));
    if (!state->refiner)
        return fail("OpenSubdiv rejected the subdivision topology");
    OpenSubdiv::Bfr::SurfaceFactory::Options options;
    options.SetApproxLevelSmooth(4).SetApproxLevelSharp(6);
    state->factory = std::make_unique<OpenSubdiv::Bfr::RefinerSurfaceFactory<>>(
        *state->refiner, options);
    std::vector<double> posed;
    posed.reserve(s.points.size() * 3);
    for (GfVec3f const& p : s.points)
        for (int k = 0; k < 3; ++k) posed.push_back(double(p[k]));
    state->faces.resize(s.faceVertexCounts.size());
    size_t liveFaces = 0;
    for (size_t f = 0; f < state->faces.size(); ++f) {
        if (!state->factory->FaceHasLimitSurface(int(f))) {
            if (std::find(s.holeIndices.begin(),s.holeIndices.end(),int(f)) ==
                s.holeIndices.end())
                return fail("OpenSubdiv rejected non-hole limit face " +
                            std::to_string(f));
            continue;
        }
        auto& face = state->faces[f];
        face.surface.reset(state->factory->CreateVertexSurface<double>(int(f)));
        if (!face.surface)
            return fail("OpenSubdiv failed to create a face limit surface");
        face.patchPoints.resize(size_t(face.surface->GetNumPatchPoints()) * 3);
        face.surface->PreparePatchPoints(posed.data(), 3,
                                         face.patchPoints.data(), 3);
        for (double v : face.patchPoints)
            if (!std::isfinite(v))
                return fail("non-finite subdivision patch point");
        ++liveFaces;
    }
    // A mesh may intentionally have every face holed.  Collision treats it
    // as empty; callers must not replace it with its coarse cage.
    _state = std::move(state);
    return true;
}

bool UsdGenCollisionLimitSurface::HasFace(int face) const {
    return _state && face >= 0 && size_t(face) < _state->faces.size() &&
        bool(_state->faces[size_t(face)].surface);
}

bool UsdGenCollisionLimitSurface::FaceControlBounds(
        int face, GfVec3d* lo, GfVec3d* hi) const {
    if (!HasFace(face) || !lo || !hi) return false;
    auto const& data = _state->faces[size_t(face)];
    auto const& values = data.patchPoints;
    if (values.size() < 3) return false;
    // A regular nonlinear quad is one bicubic patch. Recover its Bernstein
    // controls from the four corners and their derivatives, then use the
    // convex-hull property for a tighter bound than the supporting mesh.
    // Other surfaces may contain several patches, so retain their support
    // hull below.
    if (data.surface->GetFaceSize() == 4 &&
        data.surface->IsRegular() && !data.surface->IsLinear() &&
        !data.surface->GetParameterization().HasSubFaces()) {
        GfVec3d controls[4][4];
        bool valid = true;
        for (int u = 0; u <= 1 && valid; ++u) {
            for (int v = 0; v <= 1; ++v) {
                double uv[2] = {double(u), double(v)};
                double p[3], du[3], dv[3], duu[3], duv[3], dvv[3];
                data.surface->Evaluate(uv, values.data(), 3,
                                       p, du, dv, duu, duv, dvv);
                for (int k = 0; k < 3; ++k)
                    valid = valid && std::isfinite(p[k]) &&
                        std::isfinite(du[k]) && std::isfinite(dv[k]) &&
                        std::isfinite(duv[k]);
                if (!valid) break;
                int const i = u ? 3 : 0, j = v ? 3 : 0;
                int const ii = u ? 2 : 1, jj = v ? 2 : 1;
                double const su = u ? -1.0 : 1.0;
                double const sv = v ? -1.0 : 1.0;
                for (int k = 0; k < 3; ++k) {
                    controls[i][j][k] = p[k];
                    controls[ii][j][k] = p[k] + su * du[k] / 3.0;
                    controls[i][jj][k] = p[k] + sv * dv[k] / 3.0;
                    controls[ii][jj][k] = p[k] + su * du[k] / 3.0 +
                        sv * dv[k] / 3.0 + su * sv * duv[k] / 9.0;
                }
            }
        }
        if (valid) {
            GfVec3d lower = controls[0][0], upper = controls[0][0];
            for (auto const& row : controls)
                for (GfVec3d const& control : row)
                    for (int k = 0; k < 3; ++k) {
                        valid = valid && std::isfinite(control[k]);
                        lower[k] = std::min(lower[k], control[k]);
                        upper[k] = std::max(upper[k], control[k]);
                    }
            if (valid) {
                for (int k = 0; k < 3; ++k) {
                    double const scale = 1.0 +
                        std::max(std::abs(lower[k]), std::abs(upper[k])) +
                        (upper[k] - lower[k]);
                    double const pad = 1024.0 *
                        std::numeric_limits<double>::epsilon() * scale;
                    if (!std::isfinite(pad)) { valid = false; break; }
                    lower[k] = std::nextafter(lower[k] - pad,
                        -std::numeric_limits<double>::infinity());
                    upper[k] = std::nextafter(upper[k] + pad,
                        std::numeric_limits<double>::infinity());
                }
                if (valid) { *lo = lower; *hi = upper; return true; }
            }
        }
    }
    *lo = *hi = GfVec3d(values[0], values[1], values[2]);
    for (size_t i = 3; i < values.size(); i += 3)
        for (int k = 0; k < 3; ++k) {
            (*lo)[k] = std::min((*lo)[k], values[i + size_t(k)]);
            (*hi)[k] = std::max((*hi)[k], values[i + size_t(k)]);
        }
    return true;
}

bool UsdGenCollisionLimitSurface::FaceIsTriangle(int face) const {
    return HasFace(face) &&
        _state->faces[size_t(face)].surface->GetParameterization().GetType() ==
        OpenSubdiv::Bfr::Parameterization::TRI;
}
bool UsdGenCollisionLimitSurface::FaceHasSubFaces(int face) const {
    return HasFace(face) &&
        _state->faces[size_t(face)].surface->GetParameterization().HasSubFaces();
}

bool UsdGenCollisionLimitSurface::Evaluate(int face, double u, double v,
                                            GfVec3d* p, GfVec3d* du,
                                            GfVec3d* dv) const {
    if (!HasFace(face) || !p || !du || !dv ||
        !std::isfinite(u) || !std::isfinite(v)) return false;
    auto const& data = _state->faces[size_t(face)];
    double const uv[2] = {u, v};
    double out[3], outDu[3], outDv[3];
    data.surface->Evaluate(uv, data.patchPoints.data(), 3,
                           out, outDu, outDv);
    for (int k = 0; k < 3; ++k)
        if (!std::isfinite(out[k]) || !std::isfinite(outDu[k]) ||
            !std::isfinite(outDv[k])) return false;
    *p = GfVec3d(out[0], out[1], out[2]);
    *du = GfVec3d(outDu[0], outDu[1], outDu[2]);
    *dv = GfVec3d(outDv[0], outDv[1], outDv[2]);
    return true;
}

bool UsdGenCollisionLimitSurface::TessellateFace(int face, int rate,
        std::vector<GfVec2d>* uv,
        std::vector<std::array<int, 3>>* facets) const {
    if (!HasFace(face) || !uv || !facets || rate < 1) return false;
    auto const& parameterization =
        _state->faces[size_t(face)].surface->GetParameterization();
    if (parameterization.HasSubFaces()) {
        // QUAD_SUBFACES is a tiled, discontinuous face domain.  Tessellating
        // it as one grid can create facets whose UV corners belong to
        // different normalized sub-faces; interpolating such a facet gives
        // invalid error samples and Newton seeds.  Build each Ptex-like
        // normalized sub-face independently so every triangle stays within
        // one Bfr tile.
        uv->clear(); facets->clear();
        int const subFaces = parameterization.GetFaceSize();
        size_t const side = size_t(rate + 1);
        uv->reserve(size_t(subFaces) * side * side);
        facets->reserve(size_t(subFaces) * size_t(rate) * size_t(rate) * 2);
        for (int subFace = 0; subFace < subFaces; ++subFace) {
            int const base = int(uv->size());
            for (int j = 0; j <= rate; ++j) {
                for (int i = 0; i <= rate; ++i) {
                    double const local[2] = {double(i) / double(rate),
                                             double(j) / double(rate)};
                    double coord[2];
                    parameterization.ConvertNormalizedSubFaceToCoord(
                        subFace, local, coord);
                    uv->emplace_back(coord[0], coord[1]);
                }
            }
            for (int j = 0; j < rate; ++j) {
                for (int i = 0; i < rate; ++i) {
                    int const a = base + j * (rate + 1) + i;
                    int const b = a + 1;
                    int const d = a + rate + 1;
                    int const c = d + 1;
                    facets->push_back({a, b, c});
                    facets->push_back({a, c, d});
                }
            }
        }
        return true;
    }
    OpenSubdiv::Bfr::Tessellation tess(
        parameterization, rate);
    if (!tess.IsValid() || tess.GetFacetSize() != 3) return false;
    std::vector<double> rawUv(size_t(tess.GetNumCoords()) * 2);
    std::vector<int> rawFacets(size_t(tess.GetNumFacets()) * 3);
    tess.GetCoords(rawUv.data());
    tess.GetFacets(rawFacets.data());
    uv->clear(); facets->clear();
    uv->reserve(size_t(tess.GetNumCoords()));
    facets->reserve(size_t(tess.GetNumFacets()));
    for (size_t i = 0; i < rawUv.size(); i += 2)
        uv->emplace_back(rawUv[i], rawUv[i + 1]);
    for (size_t i = 0; i < rawFacets.size(); i += 3) {
        for (size_t k = 0; k < 3; ++k)
            if (rawFacets[i + k] < 0 ||
                size_t(rawFacets[i + k]) >= uv->size()) return false;
        facets->push_back({rawFacets[i], rawFacets[i + 1], rawFacets[i + 2]});
    }
    return true;
}
}
