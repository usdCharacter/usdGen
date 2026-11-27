#ifndef USDGEN_OPS_CURVE_WRAP_H
#define USDGEN_OPS_CURVE_WRAP_H

#include "usdGenMath/math.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace usdGen::curveWrap {

// A rotation-minimizing frame carries transverse offsets along one open
// polyline. Rest segment + fraction is the material binding; it never slides
// to a different segment in the pose. A centerline alone supplies no roll.
struct Frame { UsdGenVec3d t, n, b; };
inline UsdGenVec3d Rotate(UsdGenVec3d const &v, UsdGenVec3d const &a, UsdGenVec3d const &b) {
    return UsdGenRotateInto(a, b, v);
}
inline Frame Orthonormal(UsdGenVec3d t, UsdGenVec3d n) {
    t.Normalize(); n -= t*UsdGenDot(n,t); n.Normalize();
    return {t,n,UsdGenCross(t,n)};
}
inline bool Frames(std::vector<UsdGenVec3d> const &p, UsdGenVec3d const *initial,
                   std::vector<Frame> *frames, std::string *error) {
    if (p.size()<2) { *error="curveWrap needs at least two driver CVs"; return false; }
    std::vector<UsdGenVec3d> edges;
    for (auto const &v:p) for (int d=0;d<3;++d)
        if (!std::isfinite(v[d])) { *error="curveWrap driver has non-finite points"; return false; }
    for (size_t i=1;i<p.size();++i) {
        auto e=p[i]-p[i-1];
        if (e.Normalize()<1e-10) { *error="curveWrap driver has a zero-length segment"; return false; }
        edges.push_back(e);
    }
    frames->resize(p.size());
    for (size_t i=0;i<p.size();++i) {
        auto t=i==0 ? edges.front() : (i+1==p.size() ? edges.back() : edges[i-1]+edges[i]);
        if (t.Normalize()<1e-8) { *error="curveWrap driver has a reversing cusp"; return false; }
        UsdGenVec3d n;
        if (i==0) {
            if (initial) n=*initial;
            else {
                int axis=0;
                for (int d=1;d<3;++d) if (std::abs(t[d])<std::abs(t[axis])) axis=d;
                n=UsdGenVec3d(0); n[axis]=1;
            }
        } else n=Rotate((*frames)[i-1].n,(*frames)[i-1].t,t);
        (*frames)[i]=Orthonormal(t,n);
    }
    return true;
}
inline Frame Interpolate(Frame const &a,Frame const &b,double u) {
    return Orthonormal(a.t*(1-u)+b.t*u,a.n*(1-u)+b.n*u);
}

class Field {
public:
    bool Bind(std::vector<UsdGenVec3d> const &rest,std::vector<UsdGenVec3d> const &pose,std::string *error) {
        if (rest.size()!=pose.size()) { *error="curveWrap rest/pose topology differs"; return false; }
        if (!Frames(rest,nullptr,&restFrames_,error)) return false;
        std::vector<Frame> temporary;
        if (!Frames(pose,nullptr,&temporary,error)) return false;
        auto n=Rotate(restFrames_[0].n,restFrames_[0].t,temporary[0].t);
        if (!Frames(pose,&n,&poseFrames_,error)) return false;
        rest_=rest;pose_=pose;return true;
    }
    UsdGenVec3d Map(UsdGenVec3d const &x) const {
        size_t segment=0;double fraction=0,best=std::numeric_limits<double>::infinity();
        for (size_t i=0;i+1<rest_.size();++i) {
            auto e=rest_[i+1]-rest_[i];
            double u=std::clamp(UsdGenDot(x-rest_[i],e)/e.GetLengthSq(),0.0,1.0);
            double distance=(x-rest_[i]-u*e).GetLengthSq();
            if (distance<best) { best=distance;segment=i;fraction=u; }
        }
        auto const r=Interpolate(restFrames_[segment],restFrames_[segment+1],fraction);
        auto const p=Interpolate(poseFrames_[segment],poseFrames_[segment+1],fraction);
        auto offset=x-(rest_[segment]*(1-fraction)+rest_[segment+1]*fraction);
        return pose_[segment]*(1-fraction)+pose_[segment+1]*fraction+
            UsdGenDot(offset,r.t)*p.t+UsdGenDot(offset,r.n)*p.n+UsdGenDot(offset,r.b)*p.b;
    }
private:
    std::vector<UsdGenVec3d> rest_,pose_;
    std::vector<Frame> restFrames_,poseFrames_;
};
} // namespace usdGen::curveWrap
#endif
