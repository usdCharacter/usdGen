#ifndef USDGEN_OPS_REGION_MAP_H
#define USDGEN_OPS_REGION_MAP_H

#include "usdGen/op.h"
#include "usdGen/maps/ptexMap.h"
#include "usdGen/ops/opUtil.h"
#include <algorithm>
#include <cmath>

namespace usdGen {

// Direct categorical region-map transport shared by growth and deformation.
// This samples actual Ptex texels at REST roots, never face display colors.
template <class Geometry>
inline bool UsdGenReadRootRegions(UsdGenCaptureContext const &ctx,
                                 Geometry const &geometry,
                                 std::vector<int> *regions, bool *mapped,
                                 std::string *error) {
    *mapped=false; regions->clear();
    if (!ctx.params || !ctx.params->node || !ctx.desc) return true;
    auto const &node=*ctx.params->node;
    SdfPath target;
    for (auto const &binding:node.mapBindings) if (binding.relationship.GetString()=="usdGen:regionMap") {
        if (!target.IsEmpty()) { *error="regionMap requires exactly one target"; return false; }
        target=binding.map;
    }
    if (target.IsEmpty()) return true;
    *mapped=true;
    auto fail=[&](std::string const &s) { *error="regionMap: "+s;return false; };
    auto map=std::find_if(ctx.desc->maps.begin(),ctx.desc->maps.end(),
        [&](UsdGenMapDesc const &m) { return m.path==target; });
    if (map==ctx.desc->maps.end() || !UsdGenMapHas(*map,kUsdGenTypeMapPtex) || map->resolvedAssetPath.empty())
        return fail("target must be a resolved UsdGenPtexMap");
    UsdGenPtexMapOptions options;options.filter="nearest";options.channelCount=1;
    for (auto const &param:map->params) {
        std::string const &name=param.name.GetString();
        if (name=="map:firstChannel") {
            if (!param.value.IsHolding<int>()) return fail("firstChannel must be an integer");
            options.firstChannel=param.value.UncheckedGet<int>();
        }
        if (name=="map:filter" &&
            (!param.value.IsHolding<TfToken>() || param.value.UncheckedGet<TfToken>().GetString()!="nearest"))
            return fail("categorical maps require filter=nearest");
        if (name=="map:blur" &&
            ((param.value.IsHolding<float>() && param.value.UncheckedGet<float>()!=0) ||
             (param.value.IsHolding<double>() && param.value.UncheckedGet<double>()!=0)))
            return fail("categorical maps require blur=0");
    }
    auto texture=UsdGenPtexTexture::Open(map->resolvedAssetPath,options,error);
    if (!texture) return false;
    if (ctx.surface>=ctx.desc->surfaces.size() || geometry.rootPrim.size()!=geometry.totalCurves)
        return fail("requires a bound rest surface and a face binding for every strand");
    auto const &surface=ctx.desc->surfaces[ctx.surface];
    if (surface.restFromCurrentPoints || surface.restPoints.empty())
        return fail("requires Default-time surface rest points");
    size_t const count=surface.faceVertexCounts.size();
    std::vector<int> offsets(count+1,0);
    for (size_t i=0;i<count;++i) offsets[i+1]=offsets[i]+surface.faceVertexCounts[i];
    if (offsets.back()<0 || size_t(offsets.back())!=surface.faceVertexIndices.size())
        return fail("invalid rest surface topology");
    auto ids=UsdGenPtexFirstFaceIds(surface.faceVertexCounts.cdata(),count);
    if (ids.size()!=count) return fail("invalid Ptex face topology");
    bool limitUV=false;
    for (auto const &n:ctx.desc->nodes)
        limitUV=limitUV || UsdGenNodeEmitsLimitSurfaceUv(*ctx.desc,n);
    limitUV=limitUV && geometry.rootUV.size()==geometry.totalCurves;
    std::vector<uint32_t> spans;
    if (!opUtil::CurveSpans(geometry,&spans,error)) return false;
    auto sampler=texture->MakeSampler();
    regions->resize(geometry.totalCurves);
    for (size_t c=0;c<geometry.totalCurves;++c) {
        int const face=geometry.rootPrim[c];int pf=-1;float u=0,v=0;
        bool located=false;
        if (limitUV && !texture->IsTriangleMesh() && face>=0 && size_t(face)<count) {
            pf=ids[face];u=geometry.rootUV[c][0];v=geometry.rootUV[c][1];located=true;
        } else {
            auto const p=opUtil::RestPoint(geometry,spans[c]);
            located=UsdGenPtexFaceCoordinate(reinterpret_cast<float const *>(surface.restPoints.cdata()),
                surface.restPoints.size(),surface.faceVertexCounts.cdata(),surface.faceVertexIndices.cdata(),
                offsets.data(),count,ids.data(),texture->IsTriangleMesh(),face,p[0],p[1],p[2],&pf,&u,&v);
        }
        float value=0;
        if (!located || !sampler->Sample(pf,u,v,&value))
            return fail("cannot sample strand "+std::to_string(c)+" at its rest root");
        if (!std::isfinite(value) || value<0 || value>16777215 || std::abs(value-std::round(value))>1e-4)
            return fail("texels must contain nonnegative integer region IDs");
        (*regions)[c]=int(std::round(value));
    }
    return true;
}
} // namespace usdGen
#endif
