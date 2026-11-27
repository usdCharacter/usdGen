// Offline density bake for native PointInstancer strand prototypes. The
// viewport stays instanced: four float3 instance primvars are added.
#include "usdGenHydraCurves/furOcclusion.h"
#include "usdGenHydraCurves/register.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usdGeom/basisCurves.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/pointInstancer.h"
#include "pxr/usd/usdGeom/primvarsAPI.h"
#include "pxr/usd/usdGeom/xformCache.h"
#include <chrono>
#include <cstdio>
#include <stdexcept>
PXR_NAMESPACE_USING_DIRECTIVE
int main(int argc,char** argv) try {
    (void)usdGenHydra::UsdGenHydraCurvesEnsureRegistered();
    if(argc<3) {std::fprintf(stderr,"Usage: usdGenBakeFur input.usd output.usda [resolution]\n"
        "  resolution: voxels per axis; omit to derive an anisotropic grid at ~0.3 world units\n");return 2;}
    auto stage=UsdStage::Open(argv[1]);
    if(!stage) throw std::runtime_error("cannot open input stage");
    auto start=std::chrono::steady_clock::now();
    std::vector<usdGen::UsdGenTilePublication> tiles;
    struct Instance {size_t root,tip;};
    struct Group {UsdGeomPointInstancer prim;std::vector<Instance> instances;};
    std::vector<Group> groups;
    UsdGeomXformCache cache;
    for(auto const& prim:stage->Traverse()) {
        UsdGeomPointInstancer inst(prim);if(!inst) continue;
        SdfPathVector paths;inst.GetPrototypesRel().GetTargets(&paths);
        VtIntArray indices;inst.GetProtoIndicesAttr().Get(&indices);
        VtMatrix4dArray transforms;
        if(!inst.ComputeInstanceTransformsAtTime(&transforms,UsdTimeCode::Default(),UsdTimeCode::Default(),
            UsdGeomPointInstancer::IncludeProtoXform,UsdGeomPointInstancer::IgnoreMask))
            throw std::runtime_error("invalid instance transforms");
        auto mask=inst.ComputeMaskAtTime(UsdTimeCode::Default());
        usdGen::UsdGenTilePublication tile;tile.tile=static_cast<usdGen::UsdGenTileId>(tiles.size());tile.xformMatrix.SetIdentity();
        Group group{inst,{}};
        auto world=cache.GetLocalToWorldTransform(prim);
        for(size_t i=0;i<indices.size();++i) {
            if(indices[i]<0 || size_t(indices[i])>=paths.size()) throw std::runtime_error("bad prototype index");
            UsdGeomBasisCurves curve(stage->GetPrimAtPath(paths[indices[i]]));
            VtVec3fArray points;VtFloatArray widths;VtIntArray counts;
            if(!curve || !curve.GetPointsAttr().Get(&points) || !curve.GetWidthsAttr().Get(&widths) ||
               !curve.GetCurveVertexCountsAttr().Get(&counts) || counts.size()!=1 || points.size()<2 ||
               size_t(counts[0])!=points.size() || (widths.size()!=1&&widths.size()!=points.size()))
                throw std::runtime_error("each prototype must be one BasisCurves strand with constant/vertex widths");
            GfMatrix4d xf=transforms[i]*world;
            float scale=0;
            for(int k=0;k<3;++k){GfVec3d a(0);a[k]=1;scale=std::max(scale,float(xf.TransformDir(a).GetLength()));}
            group.instances.push_back({tile.points.size(),tile.points.size()+points.size()-1});
            bool visible=mask.empty() || mask[i];
            for(size_t j=0;j<points.size();++j) {
                tile.points.push_back(GfVec3f(xf.Transform(GfVec3d(points[j]))));
                tile.widths.push_back(visible?widths[widths.size()==1?0:j]*scale:0.f);
            }
            tile.curveVertexCounts.push_back(static_cast<int>(points.size()));
        }
        tiles.push_back(std::move(tile));groups.push_back(std::move(group));
    }
    if(tiles.empty()) throw std::runtime_error("no PointInstancer found");
    // Every Mesh in the stage is an opaque blocker, the way the procedural
    // lane treats the groom's emitting surfaces.
    usdGen::UsdGenFurOcclusionParams occlusion;
    if(argc>3) occlusion.resolution=std::stoi(argv[3]);
    for(auto const& prim:stage->Traverse()) {
        UsdGeomMesh mesh(prim);if(!mesh) continue;
        usdGen::UsdGenFurOccluder occluder;
        if(!mesh.GetPointsAttr().Get(&occluder.points) ||
           !mesh.GetFaceVertexCountsAttr().Get(&occluder.faceVertexCounts) ||
           !mesh.GetFaceVertexIndicesAttr().Get(&occluder.faceVertexIndices) ||
           occluder.points.empty() || occluder.faceVertexIndices.empty()) continue;
        occluder.worldMatrix=cache.GetLocalToWorldTransform(prim);
        occlusion.occluders.push_back(std::move(occluder));
    }
    usdGen::UsdGenBuildFurOcclusion(&tiles,nullptr,occlusion);
    for(size_t g=0;g<groups.size();++g) {
        UsdGeomPrimvarsAPI pv(groups[g].prim);
        for(auto const& plane:tiles[g].extraUniform) {
            // Only the per-CV vec3 optical-depth planes are instanced here:
            // this loop indexes f[cv*3..+2] directly, and the cooker also
            // publishes constant look planes (hairTipColor and friends) that
            // would read off the end of a 1- or 3-element buffer.
            if(plane.arity!=3||plane.interpolation!=TfToken("vertex")||
               plane.f.size()<tiles[g].points.size()*3) continue;
            VtVec3fArray root,tip;root.reserve(groups[g].instances.size());tip.reserve(root.capacity());
            for(auto const& i:groups[g].instances){
                root.push_back(GfVec3f(plane.f[i.root*3],plane.f[i.root*3+1],plane.f[i.root*3+2]));
                tip.push_back(GfVec3f(plane.f[i.tip*3],plane.f[i.tip*3+1],plane.f[i.tip*3+2]));
            }
            pv.CreatePrimvar(TfToken(plane.name.GetString()),SdfValueTypeNames->Float3Array,UsdGeomTokens->varying).Set(root);
            std::string tipName=plane.name.GetString();tipName.insert(3,"Tip");
            pv.CreatePrimvar(TfToken(tipName),SdfValueTypeNames->Float3Array,UsdGeomTokens->varying).Set(tip);
        }
    }
    if(!stage->Export(argv[2])) throw std::runtime_error("cannot export output stage");
    std::printf("Baked %zu instancer(s) in %.2f ms (including USD export)\n",groups.size(),
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    return 0;
} catch(std::exception const& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
