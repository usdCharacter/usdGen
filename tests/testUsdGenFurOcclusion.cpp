#include "usdGen/furOcclusion.h"
#include "usdGen/generationStore.h"
#include "usdGen/scheduler.h"
#include "usdGenImaging/usdGenTilePublisher.h"
#include "pxr/base/gf/rotation.h"
#include "pxr/imaging/hd/primvarsSchema.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE
namespace {
void Require(bool b,char const* msg) { if(!b) throw std::runtime_error(msg); }
UsdGenPlane const& Plane(std::vector<UsdGenPlane> const& planes,char const* name) {
    for(auto const& p:planes) if(p.name==name) return p;
    throw std::runtime_error("missing optical depth plane");
}
UsdGenPlane const& Plane(UsdGenTilePublication const& tile,char const* name) {
    return Plane(tile.extraUniform,name);
}
GfVec3f Tau(UsdGenTilePublication const& tile,char const* name,size_t cv) {
    auto const& p=Plane(tile,name);
    return GfVec3f(p.f[cv*3],p.f[cv*3+1],p.f[cv*3+2]);
}
// The reconstruction the shader must use: a normalised cosine-power-4 blend of
// the six stored depths. Documented in furOcclusion.h and mirrored here so a
// change to one without the other fails the suite.
float Reconstruct(GfVec3f const& positive,GfVec3f const& negative,GfVec3f const& L,
                  float power=4.f) {
    float sum=0,weighted=0;
    for(int k=0;k<3;++k) {
        float const w=std::pow(L[k]*L[k],power*0.5f);
        weighted+=w*(L[k]>=0?positive[k]:negative[k]); sum+=w;
    }
    return sum>0?weighted/sum:0.f;
}
UsdGenTilePublication Make(int id,float x,int strands) {
    UsdGenTilePublication t;t.tile=static_cast<UsdGenTileId>(id);t.xformMatrix.SetIdentity();
    t.primPath=SdfPath("/fur/tile_"+std::to_string(id));
    for(int i=0;i<strands;++i) {
        float z=float(i%20)/100-0.1f;
        float dx=float(i/20)/100;
        t.points.push_back(GfVec3f(x+dx,0,z));
        t.points.push_back(GfVec3f(x+dx,1,z));
        t.curveVertexCounts.push_back(2);
    }
    t.widths=VtFloatArray{0.01f};return t;
}
// A two-CV probe with zero width: a receiver that adds no density of its own.
UsdGenTilePublication Probe(int id,GfVec3f const& at,GfVec3f const& to) {
    UsdGenTilePublication t;t.tile=static_cast<UsdGenTileId>(id);t.xformMatrix.SetIdentity();
    t.primPath=SdfPath("/fur/probe_"+std::to_string(id));
    t.points.push_back(at);t.points.push_back(to);
    t.curveVertexCounts.push_back(2);t.widths=VtFloatArray{0.f};return t;
}
// layers x-running fibres at spacing `spacing` in z, stacked in y: a slab whose
// expected fibre crossings along +y is exactly layers * width / spacing.
UsdGenTilePublication Slab(int id,int layers,float width,float spacing,
                           float y0,float dy,float length,float depth) {
    UsdGenTilePublication t;t.tile=static_cast<UsdGenTileId>(id);t.xformMatrix.SetIdentity();
    t.primPath=SdfPath("/fur/slab_"+std::to_string(id));
    int const perLayer=int(depth/spacing);
    for(int j=0;j<layers;++j) for(int i=0;i<perLayer;++i) {
        float const z=-0.5f*depth+float(i)*spacing;
        t.points.push_back(GfVec3f(-0.5f*length,y0+float(j)*dy,z));
        t.points.push_back(GfVec3f( 0.5f*length,y0+float(j)*dy,z));
        t.curveVertexCounts.push_back(2);
    }
    t.widths=VtFloatArray{width};return t;
}
// A combed patch: strands rooted on a small grid and leaning over, so they
// have projected area against a ray rising off the surface. Make()'s vertical
// strands deliberately have none -- that is the first assertion in this file --
// which makes them the wrong probe for a shadow cast straight down.
UsdGenTilePublication Comb(int id,float x,int strands,float lean) {
    UsdGenTilePublication t;t.tile=static_cast<UsdGenTileId>(id);t.xformMatrix.SetIdentity();
    t.primPath=SdfPath("/fur/comb_"+std::to_string(id));
    for(int i=0;i<strands;++i) {
        float const z=float(i%20)/100-0.1f;
        float const dx=float(i/20)/100;
        GfVec3f const root(x+dx,0,z);
        t.points.push_back(root);
        t.points.push_back(root+GfVec3f(lean,lean,0));
        t.curveVertexCounts.push_back(2);
    }
    t.widths=VtFloatArray{0.01f};return t;
}
UsdGenFurOccluder Quad(float y,float half,bool flip) {
    UsdGenFurOccluder q;
    q.points={GfVec3f(-half,y,-half),GfVec3f(half,y,-half),
              GfVec3f(half,y,half),GfVec3f(-half,y,half)};
    q.faceVertexCounts={4};
    // Right-handed winding around +y, or reversed.
    q.faceVertexIndices = flip ? VtIntArray{3,2,1,0} : VtIntArray{0,1,2,3};
    return q;
}
// A groom shaped like the thing being shaded: a shell of hair rooted on a
// sphere, over that sphere as an opaque blocker. Its bounds are a cube and its
// density field is rigid, so rotating the whole scene and baking again gives
// the true optical depth along the rotated axis.
UsdGenTilePublication HairShell(int id,int strands,unsigned seed) {
    UsdGenTilePublication t;t.tile=static_cast<UsdGenTileId>(id);t.xformMatrix.SetIdentity();
    t.primPath=SdfPath("/fur/shell_"+std::to_string(id));
    unsigned state=seed;
    auto next=[&]{ state=state*1664525u+1013904223u; return float(state>>8)/float(1<<24); };
    for(int i=0;i<strands;++i) {
        GfVec3f n,lean;
        do { n=GfVec3f(next()*2-1,next()*2-1,next()*2-1); } while(n.GetLength()<1e-3f);
        n.Normalize();
        do { lean=GfVec3f(next()*2-1,next()*2-1,next()*2-1); } while(lean.GetLength()<1e-3f);
        lean.Normalize();
        GfVec3f dir=n*0.45f+lean*0.55f;
        if(dir.GetLength()<1e-3f) dir=n; else dir.Normalize();
        GfVec3f const root=n*0.5f;
        for(int j=0;j<4;++j) t.points.push_back(root+dir*(0.1f*float(j)));
        t.curveVertexCounts.push_back(4);
    }
    t.widths=VtFloatArray{0.02f};return t;
}
UsdGenFurOccluder SphereMesh(float radius,int rings,int segments) {
    UsdGenFurOccluder s;
    float const pi=3.14159265358979323846f;
    VtVec3fArray points; VtIntArray counts, indices;
    for(int j=0;j<=rings;++j) {
        float const theta=pi*float(j)/float(rings);
        for(int i=0;i<segments;++i) {
            float const phi=2.f*pi*float(i)/float(segments);
            points.push_back(radius*GfVec3f(std::sin(theta)*std::cos(phi),
                                            std::cos(theta),
                                            std::sin(theta)*std::sin(phi)));
        }
    }
    auto at=[&](int j,int i){ return j*segments+(i%segments); };
    for(int j=0;j<rings;++j) for(int i=0;i<segments;++i) {
        counts.push_back(4);
        // Counter-clockwise seen from outside: the geometric normal is outward.
        indices.push_back(at(j,i));   indices.push_back(at(j+1,i));
        indices.push_back(at(j+1,i+1)); indices.push_back(at(j,i+1));
    }
    s.points=points;s.faceVertexCounts=counts;s.faceVertexIndices=indices;return s;
}
std::vector<UsdGenTilePublication> Rotate(std::vector<UsdGenTilePublication> const& in,
                                          GfMatrix4d const& m) {
    auto out=in;
    for(auto& tile:out) {
        VtVec3fArray points(tile.points.size());
        for(size_t i=0;i<points.size();++i)
            points[i]=GfVec3f(m.Transform(GfVec3d(tile.points[i])));
        tile.points=points;
        tile.extraUniform.clear();
    }
    return out;
}
UsdGenFurOccluder Rotate(UsdGenFurOccluder const& in,GfMatrix4d const& m) {
    auto out=in;
    VtVec3fArray points(in.points.size());
    for(size_t i=0;i<points.size();++i)
        points[i]=GfVec3f(m.Transform(GfVec3d(in.points[i])));
    out.points=points;return out;
}
}
int main() try {
    // ---- cross-tile occlusion, COW reuse, dirty notices, transport ----------
    std::vector<UsdGenTilePublication> tiles{Make(0,0,1),Make(1,0.2f,400)};
    Require(UsdGenBuildFurOcclusion(&tiles),"initial bake");
    auto const px=Plane(tiles[0],"furTauP").f[0];
    auto const nx=Plane(tiles[0],"furTauN").f[0];
    Require(px>nx+0.2f,"neighboring tile shadows receiver from +X");
    Require(Plane(tiles[0],"furTauP").f[1]<0.001f,"parallel fibers have zero projected area");
    for(auto const& tile:tiles) for(auto const& plane:tile.extraUniform) {
        Require(plane.f.size()==tile.points.size()*3,"vertex cardinality");
        for(float x:plane.f)
            Require(std::isfinite(x)&&x>=0&&x<=UsdGenFurTauClamp,"finite bounded depth");
    }
    {   // Determinism: the same input bakes to the same bits.
        auto again=std::vector<UsdGenTilePublication>{Make(0,0,1),Make(1,0.2f,400)};
        UsdGenBuildFurOcclusion(&again);
        for(size_t t=0;t<tiles.size();++t) for(auto name:{"furTauP","furTauN"})
            for(size_t i=0;i<Plane(tiles[t],name).f.size();++i)
                Require(Plane(tiles[t],name).f[i]==Plane(again[t],name).f[i],
                        "deterministic bake");
    }
    auto prior=tiles;
    Require(!UsdGenBuildFurOcclusion(&tiles,&prior),"no-op uses cached bake");
    Require(Plane(tiles[0],"furTauP").f.IsIdentical(Plane(prior[0],"furTauP").f),"COW cache sharing");
    tiles[1].widths=VtFloatArray{0.02f};
    Require(UsdGenBuildFurOcclusion(&tiles,&prior),"width edit invalidates all receivers");
    Require(Plane(tiles[0],"furTauP").f[0]>px*1.5f,"thicker neighbors increase depth");
    auto moved=prior;
    moved[1].xformMatrix.SetTranslate(GfVec3d(-1,0,0));
    Require(UsdGenBuildFurOcclusion(&moved,&prior),"transform edit invalidates bake");
    Require(Plane(moved[0],"furTauN").f[0]>Plane(moved[0],"furTauP").f[0],
        "moving occluder to -X reverses shadow direction");
    UsdGenGeneration a,b;a.tiles=prior;b.tiles=tiles;
    auto report=UsdGenGenerationStore().Diff(a,b);
    Require(!report.tiles[0].pointsDirty,"shadow update does not dirty receiver points");
    Require(std::find(report.tiles[0].dirtyPrimvars.begin(),report.tiles[0].dirtyPrimvars.end(),TfToken("furTauP"))!=report.tiles[0].dirtyPrimvars.end(),"neighbor update dirties receiver optical depth");
    auto ds=usdGenImaging::UsdGenTilePublisher::BuildTileDataSource(tiles[0]);
    auto pv=HdPrimvarsSchema::GetFromParent(ds).GetPrimvar(TfToken("furTauP"));
    Require(pv.IsDefined()&&pv.GetInterpolation()->GetTypedValue(0)==TfToken("vertex"),"Hydra vertex transport");
    Require(pv.GetPrimvarValue()->GetValue(0).IsHolding<VtVec3fArray>(),"packed vec3 GPU transport");
    tiles.resize(1);
    UsdGenBuildFurOcclusion(&tiles,&prior);
    Require(Plane(tiles[0],"furTauP").f[0]<px,"removed occluder clears shadow");
    tiles[0].points[0][0]=std::numeric_limits<float>::quiet_NaN();
    bool rejected=false;try { UsdGenBuildFurOcclusion(&tiles); } catch(std::invalid_argument const&) {rejected=true;}
    Require(rejected,"reject nonfinite geometry");
    {   // Diff extras verdict is buffer-identity, never content (r49): two
        // independent bakes of identical geometry hold bitwise-identical tau
        // values in FRESH buffers, and Diff must report them dirty
        // (conservative: re-upload identical bytes -- Storm re-uploads every
        // non-points primvar on any primvar dirty anyway, so identical
        // recomputes, which always co-dirty displayColor or the topology,
        // cost zero extra traffic). Shared-buffer carries stay clean.
        std::vector<UsdGenTilePublication> left{Make(0,0,1),Make(1,0.2f,400)};
        std::vector<UsdGenTilePublication> right{Make(0,0,1),Make(1,0.2f,400)};
        Require(UsdGenBuildFurOcclusion(&left),"left bake");
        Require(UsdGenBuildFurOcclusion(&right),"right bake");
        for(size_t t=0;t<left.size();++t) for(auto name:{"furTauP","furTauN"}) {
            VtFloatArray const& a=Plane(left[t],name).f;
            VtFloatArray const& b=Plane(right[t],name).f;
            Require(a.size()==b.size(),"recompute preserves tau cardinality");
            for(size_t i=0;i<a.size();++i)
                Require(a[i]==b[i],"recompute preserves tau values");
            Require(!a.IsIdentical(b),"recompute takes fresh buffers");
        }
        UsdGenGeneration gL,gR,gS; gL.tiles=left; gR.tiles=right; gS.tiles=left;
        auto rep=UsdGenGenerationStore().Diff(gL,gR);
        for(size_t t=0;t<rep.tiles.size();++t) for(auto name:{"furTauP","furTauN"}) {
            auto const& dirty=rep.tiles[t].dirtyPrimvars;
            Require(std::find(dirty.begin(),dirty.end(),TfToken(name))!=dirty.end(),
                    "fresh-buffer recompute dirties receiver optical depth");
        }
        auto rep2=UsdGenGenerationStore().Diff(gL,gS);
        for(size_t t=0;t<rep2.tiles.size();++t) for(auto name:{"furTauP","furTauN"}) {
            auto const& dirty=rep2.tiles[t].dirtyPrimvars;
            Require(std::find(dirty.begin(),dirty.end(),TfToken(name))==dirty.end(),
                    "shared-buffer extras stay clean");
        }
    }

    // ---- hair count calibration --------------------------------------------
    // N layers of parallel fibres of width w spaced s apart must read exactly
    // N*w/s crossings from below: that is a host renderer's HairCount, not a strand
    // count, and the shader's HairCount-1 shift depends on the scale.
    {
        int const layers=5; float const width=0.01f, spacing=0.05f;
        float const expected=float(layers)*width/spacing;   // 1.0
        float lowest=std::numeric_limits<float>::max(), highest=0;
        for(float voxel : {0.3f, 0.12f, 0.06f}) {
            std::vector<UsdGenTilePublication> slab{
                Slab(0,layers,width,spacing,1.0f,0.2f,4.0f,2.0f),
                Probe(1,GfVec3f(0,0.10f,0),GfVec3f(0,0.15f,0))};
            UsdGenFurOcclusionParams params; params.voxelSize=voxel;
            UsdGenBuildFurOcclusion(&slab,nullptr,params);
            float const up=Tau(slab[1],"furTauP",0)[1];
            Require(std::fabs(up-expected)<0.08f*expected,
                    "slab reports layers * coverage crossings upward");
            Require(Tau(slab[1],"furTauN",0)[1]<0.02f*expected,"nothing below the slab");
            lowest=std::min(lowest,up); highest=std::max(highest,up);
        }
        // Hair count is a physical quantity: the voxel size must not move it.
        Require(highest-lowest<0.05f*expected,"hair count is resolution independent");
    }

    // ---- grid selection ------------------------------------------------------
    {
        std::vector<UsdGenTilePublication> base{Make(0,0,1),Make(1,0.2f,400)};
        auto derived=base, coarse=base, pinned=base, pinnedAgain=base;
        UsdGenFurOcclusionParams fine; fine.voxelSize=0.01f;
        UsdGenFurOcclusionParams blunt; blunt.voxelSize=4.f;
        UsdGenFurOcclusionParams legacy; legacy.resolution=48;
        UsdGenBuildFurOcclusion(&derived,nullptr,fine);
        UsdGenBuildFurOcclusion(&coarse,nullptr,blunt);
        UsdGenBuildFurOcclusion(&pinned,nullptr,legacy);
        UsdGenBuildFurOcclusion(&pinnedAgain,nullptr,legacy);
        Require(Plane(derived[0],"furTauP").f[0]!=Plane(coarse[0],"furTauP").f[0],
                "voxelSize changes the grid");
        for(size_t i=0;i<Plane(pinned[0],"furTauP").f.size();++i)
            Require(Plane(pinned[0],"furTauP").f[i]==Plane(pinnedAgain[0],"furTauP").f[i],
                    "resolution override is reproducible");
        // A groom far longer than it is thick must not be forced into a cube:
        // the flat axes keep their own (smaller) dimension.
        std::vector<UsdGenTilePublication> flat{Make(0,0,4000)};
        UsdGenFurOcclusionParams derive;
        UsdGenBuildFurOcclusion(&flat,nullptr,derive);
        for(float x:Plane(flat[0],"furTauP").f)
            Require(std::isfinite(x)&&x>=0&&x<=UsdGenFurTauClamp,"anisotropic grid stays bounded");
    }

    // ---- opaque occluders ----------------------------------------------------
    {
        auto build=[&](bool withOccluder,bool flipWinding) {
            std::vector<UsdGenTilePublication> scene{
                Make(0,-0.1f,400), Probe(1,GfVec3f(0,-0.3f,0),GfVec3f(0,-0.25f,0))};
            UsdGenFurOcclusionParams params;
            if(withOccluder) params.occluders.push_back(Quad(0.f,0.2f,flipWinding));
            UsdGenBuildFurOcclusion(&scene,nullptr,params);
            return scene;
        };
        auto open=build(false,false), blocked=build(true,false), flipped=build(true,true);
        float const openUp=Tau(open[1],"furTauP",0)[1];
        float const blockedUp=Tau(blocked[1],"furTauP",0)[1];
        Require(openUp<8.f,"without an occluder the probe only sees hair");
        Require(blockedUp>0.75f*UsdGenFurTauClamp,
                "the emitting surface blocks light to geometry behind it");
        for(size_t i=0;i<Plane(blocked[1],"furTauP").f.size();++i)
            Require(std::fabs(Plane(blocked[1],"furTauP").f[i]-
                              Plane(flipped[1],"furTauP").f[i])<1e-3f,
                    "the shell is placed by the fur, not by the winding order");
        // Roots resting on the surface keep their sideways and upward light:
        // that is what the bias in voxels is for.
        float worstSide=0, worstUp=0;
        for(size_t i=0;i<blocked[0].points.size();i+=2) {   // even CVs are roots at y=0
            GfVec3f const p=Tau(blocked[0],"furTauP",i), n=Tau(blocked[0],"furTauN",i);
            worstSide=std::max({worstSide,p[0],n[0],p[2],n[2]});
            worstUp=std::max(worstUp,p[1]);
        }
        Require(worstSide<0.25f*UsdGenFurTauClamp,"roots are not self-shadowed sideways");
        Require(worstUp<0.25f*UsdGenFurTauClamp,"roots are not self-shadowed upward");
        Require(Tau(blocked[0],"furTauN",0)[1]>0.75f*UsdGenFurTauClamp,
                "roots are shadowed downward, through the surface");
        // Occluder edits are invisible in the tiles, so the volume key carries
        // them: with it the bake rebuilds, and an unchanged one still reuses.
        uint64_t volumeKey=0;
        std::vector<UsdGenTilePublication> live{
            Make(0,-0.1f,400), Probe(1,GfVec3f(0,-0.3f,0),GfVec3f(0,-0.25f,0))};
        UsdGenFurOcclusionParams none, some;
        some.occluders.push_back(Quad(0.f,0.2f,false));
        Require(UsdGenBuildFurOcclusion(&live,nullptr,none,&volumeKey),"first bake");
        auto before=live;
        Require(!UsdGenBuildFurOcclusion(&live,&before,none,&volumeKey),
                "an unchanged volume is still reused");
        Require(UsdGenBuildFurOcclusion(&live,&before,some,&volumeKey),
                "adding an occluder rebuilds the volume");
        Require(Tau(live[1],"furTauP",0)[1]>0.75f*UsdGenFurTauClamp,"rebuild used the occluder");
    }

    // ---- occluder-mesh reuse across cooks -----------------------------------
    // The world-space mesh is keyed on the occluder bytes alone: moving the
    // hair reuses it bit-for-bit (including the scalp cap, which reads the
    // cached normals), while editing the occluder rebuilds it.
    {
        auto checkSame = [&](std::vector<UsdGenTilePublication> const& got,
                             std::vector<UsdGenTilePublication> const& want,
                             UsdGenScalpShadowPublication const& gotShadow,
                             UsdGenScalpShadowPublication const& wantShadow,
                             char const* msg) {
            Require(got.size()==want.size(),msg);
            for(size_t t=0;t<got.size();++t) for(auto name:{"furTauP","furTauN"}) {
                auto const& g=Plane(got[t],name).f, &w=Plane(want[t],name).f;
                Require(g.size()==w.size(),msg);
                for(size_t i=0;i<g.size();++i) Require(g[i]==w[i],msg);
            }
            Require(gotShadow.digest==wantShadow.digest,msg);
            Require(gotShadow.points.size()==wantShadow.points.size(),msg);
            for(size_t i=0;i<gotShadow.points.size();++i) {
                Require(gotShadow.points[i]==wantShadow.points[i],msg);
                Require(gotShadow.normals[i]==wantShadow.normals[i],msg);
            }
            Require(gotShadow.faceVertexCounts==wantShadow.faceVertexCounts,msg);
            Require(gotShadow.faceVertexIndices==wantShadow.faceVertexIndices,msg);
            for(auto name:{"furTauP","furTauN"}) {
                auto const& g=Plane(gotShadow.extraUniform,name).f;
                auto const& w=Plane(wantShadow.extraUniform,name).f;
                Require(g.size()==w.size(),msg);
                for(size_t i=0;i<g.size();++i) Require(g[i]==w[i],msg);
            }
        };
        UsdGenFurOcclusionParams params;
        params.occluders.push_back(Quad(0.f,0.2f,false));
        params.scalpShadow=true;
        UsdGenFurOccluderBuild cache;
        params.occluderCache=&cache;
        // The same bakes also reuse the cap-build scratch below: every
        // comparison after the first runs on retained chunk capacity.
        UsdGenScalpShadowScratch scratch;
        params.scalpScratch=&scratch;
        // And the volume-bake temporaries: the moved/edited bakes below run
        // on retained sweep/density/shell capacity at changed grid sizes.
        UsdGenFurBakeScratch bake;
        params.bakeScratch=&bake;
        std::vector<UsdGenTilePublication> live{
            Make(0,-0.1f,400), Probe(1,GfVec3f(0,-0.3f,0),GfVec3f(0,-0.25f,0))};
        UsdGenScalpShadowPublication first;
        Require(UsdGenBuildFurOcclusion(&live,nullptr,params,nullptr,&first),
                "first cached bake");
        Require(cache.valid,"a miss publishes the mesh");
        Require(!cache.triangles.empty(),"the cache holds the triangulation");
        // Move the hair: the volume rebakes, the mesh reuses.
        auto moved=live;
        moved[0].xformMatrix.SetTranslate(GfVec3d(0.05,0,0));
        UsdGenFurOcclusionParams bare=params;
        bare.occluderCache=nullptr;
        bare.scalpScratch=nullptr;
        bare.bakeScratch=nullptr;
        auto expect=moved;
        UsdGenScalpShadowPublication expectShadow;
        Require(UsdGenBuildFurOcclusion(&expect,nullptr,bare,nullptr,&expectShadow),
                "uncached reference bake");
        UsdGenScalpShadowPublication hit;
        Require(UsdGenBuildFurOcclusion(&moved,nullptr,params,nullptr,&hit),
                "moved hair rebakes through the cache");
        checkSame(moved,expect,hit,expectShadow,"cache hit is bit-identical");
        // Edit the occluder: the mesh rebuilds and still matches no-cache.
        uint64_t const before=cache.key;
        params.occluders[0].points[0][1]+=0.01f;
        bare.occluders=params.occluders;
        auto edited=moved;
        UsdGenScalpShadowPublication editedShadow;
        Require(UsdGenBuildFurOcclusion(&edited,nullptr,params,nullptr,&editedShadow),
                "occluder edit rebakes through the cache");
        Require(cache.key!=before,"an occluder edit re-keys the mesh");
        auto expectEdited=moved;
        UsdGenScalpShadowPublication expectEditedShadow;
        Require(UsdGenBuildFurOcclusion(&expectEdited,nullptr,bare,nullptr,
                                       &expectEditedShadow),
                "edited uncached reference bake");
        checkSame(edited,expectEdited,editedShadow,expectEditedShadow,
                  "occluder edit rebuilds bit-identically");
        // Bake once more unchanged: the memoized occluder digest holds the
        // mesh key stable (a hit returns exactly the recomputed value).
        uint64_t const afterEdit=cache.key;
        auto still=edited;
        UsdGenScalpShadowPublication stillShadow;
        Require(UsdGenBuildFurOcclusion(&still,nullptr,params,nullptr,
                                       &stillShadow),
                "repeat bake through the cache");
        Require(cache.key==afterEdit,"an unchanged rebake holds the mesh key");
        checkSame(still,expectEdited,stillShadow,expectEditedShadow,
                  "the repeat bake matches bit-identically");
        // Retopologize one tile to degenerate (zero-CV) curves through the
        // same scratch: its jobs now carry no CVs, so the splat clears
        // their retained sub-boxes instead of merging them. Without the
        // clear the stale boxes corrupt the rebake.
        std::vector<UsdGenTilePublication> retopo{
            Make(0,-0.1f,400), Make(1,0.2f,400)};
        UsdGenScalpShadowPublication retopoSeedShadow;
        Require(UsdGenBuildFurOcclusion(&retopo,nullptr,params,nullptr,
                                       &retopoSeedShadow),
                "seed bake populates the carried sub-boxes");
        retopo[1].points.clear();
        retopo[1].curveVertexCounts.clear();
        for(int i=0;i<400;++i) retopo[1].curveVertexCounts.push_back(0);
        auto expectRetopo = retopo;
        UsdGenScalpShadowPublication retopoShadow, expectRetopoShadow;
        Require(UsdGenBuildFurOcclusion(&expectRetopo,nullptr,bare,nullptr,
                                       &expectRetopoShadow),
                "retopologized uncached reference bake");
        Require(UsdGenBuildFurOcclusion(&retopo,nullptr,params,nullptr,
                                       &retopoShadow),
                "retopologized bake through retained sub-boxes");
        checkSame(retopo,expectRetopo,retopoShadow,expectRetopoShadow,
                  "cleared sub-boxes match the uncached bake");
    }

    // ---- scalp-cap array carry across deform frames -------------------------
    // A rebuild with the same lit set and occluder shares the seeded cap's
    // topology (and the normals/points when their own inputs match) instead
    // of merging them fresh; the depths always rebake. Every carry below is
    // also checked bit-for-bit against an unseeded reference bake.
    {
        auto capSame = [&](UsdGenScalpShadowPublication const& got,
                           UsdGenScalpShadowPublication const& want,
                           char const* msg) {
            Require(got.IsEmpty()==want.IsEmpty(),msg);
            Require(got.digest==want.digest,msg);
            Require(got.points==want.points,msg);
            Require(got.normals==want.normals,msg);
            Require(got.faceVertexCounts==want.faceVertexCounts,msg);
            Require(got.faceVertexIndices==want.faceVertexIndices,msg);
            Require(got.topologyKey==want.topologyKey,msg);
            Require(got.occluderKey==want.occluderKey,msg);
            Require(got.lift==want.lift && got.inward==want.inward,msg);
            Require(got.tessLevel==want.tessLevel,msg);
            // An emptied cap publishes no prim, so the bake leaves the
            // seeded extent and depth planes as they were; only compare
            // them when the cap is non-empty.
            if (!got.points.empty()) {
                Require(got.extentMin==want.extentMin &&
                        got.extentMax==want.extentMax,msg);
                for(auto name:{"furTauP","furTauN"})
                    Require(Plane(got.extraUniform,name).f==
                            Plane(want.extraUniform,name).f,msg);
            }
        };
        // A micro-quad buried in dense hair, like a benchmark scalp patch:
        // every triangle is deeply lit, so small moves hold the full set.
        // The far zero-width probe stretches the span without adding
        // density; moving it drifts the grid (and the lift) while the
        // coverage over the quad stays far above the floor.
        GfVec3f const probeAt(1.5f,0.f,0.f);
        std::vector<UsdGenTilePublication> live{
            Comb(0,-0.3f,1200,0.3f), Probe(1,probeAt,probeAt+GfVec3f(0,0.01f,0))};
        UsdGenFurOcclusionParams params;
        params.occluders.push_back(Quad(0.f,0.01f,false));
        params.scalpShadow=true;
        UsdGenFurOccluderBuild cache;
        params.occluderCache=&cache;
        UsdGenScalpShadowScratch scratch;
        params.scalpScratch=&scratch;
        uint64_t volumeKey=0;
        UsdGenScalpShadowPublication first;
        Require(UsdGenBuildFurOcclusion(&live,nullptr,params,&volumeKey,&first),
                "first cap bake");
        Require(!first.IsEmpty(),"the cap is non-empty");
        Require(first.tessLevel==1,"the micro-quad refines exactly 1x1");
        Require(first.topologyKey!=0 && first.occluderKey!=0,
                "the bake publishes its carry keys");
        // The ceiling-skip path: the same micro-quad against a ceiling of 1
        // takes the percentile skip (2 triangles > 1) and must bake the
        // identical k==1 cap the percentile path above computed.
        {
            std::vector<UsdGenTilePublication> fresh{
                Comb(0,-0.3f,1200,0.3f),
                Probe(1,probeAt,probeAt+GfVec3f(0,0.01f,0))};
            UsdGenFurOcclusionParams skipped;
            skipped.occluders.push_back(Quad(0.f,0.01f,false));
            skipped.scalpShadow=true;
            skipped.scalpMaxTriangles=1;
            uint64_t skipKey=0;
            UsdGenScalpShadowPublication skippedCap;
            Require(UsdGenBuildFurOcclusion(&fresh,nullptr,skipped,&skipKey,
                                           &skippedCap),
                    "ceiling-skipped cap bake");
            Require(skippedCap.tessLevel==1,
                    "the skipped bake refines exactly 1x1");
            capSame(skippedCap,first,
                    "the ceiling skip matches the percentile cap");
        }
        // Move the hair within its margin and the probe far: the span
        // (hence the grid and the lift) drifts, but the occluder and the
        // lit set hold.
        auto moved=live;
        moved[0].xformMatrix.SetTranslate(GfVec3d(0.02,0,0));
        moved[1].xformMatrix.SetTranslate(GfVec3d(0.3,0,0));
        UsdGenScalpShadowPublication carry=first;
        Require(UsdGenBuildFurOcclusion(&moved,&live,params,&volumeKey,&carry),
                "moved hair rebuilds");
        Require(carry.topologyKey==first.topologyKey,"the lit set holds");
        Require(carry.occluderKey==first.occluderKey,"the occluder holds");
        Require(carry.tessLevel==first.tessLevel,"the level holds");
        Require(carry.inward==first.inward,"the solid side holds");
        Require(carry.lift!=first.lift,"the span drift moves the lift");
        Require(carry.faceVertexCounts.IsIdentical(first.faceVertexCounts),
                "counts carry across the move");
        Require(carry.faceVertexIndices.IsIdentical(first.faceVertexIndices),
                "indices carry across the move");
        Require(carry.normals.IsIdentical(first.normals),
                "normals carry across the move");
        Require(!carry.points.IsIdentical(first.points),
                "points go fresh when the lift drifts");
        Require(carry.digest!=first.digest,"fresh depths re-key the digest");
        auto expect=moved;
        UsdGenScalpShadowPublication want;
        {
            UsdGenFurOcclusionParams bare=params;
            bare.occluderCache=nullptr; bare.scalpScratch=nullptr;
            uint64_t freshKey=0;
            Require(UsdGenBuildFurOcclusion(&expect,nullptr,bare,&freshKey,
                                           &want),
                    "unseeded reference bake");
        }
        capSame(carry,want,"the carried cap matches the fresh build");
        // Edit the occluder a hair's breadth: the lit set (the hair's, not
        // the mesh's) still carries while points and normals go fresh.
        float const savedY=params.occluders[0].points[0][1];
        params.occluders[0].points[0][1]+=1e-5f;
        auto edited=moved;
        UsdGenScalpShadowPublication editedCap=carry;
        Require(UsdGenBuildFurOcclusion(&edited,&moved,params,&volumeKey,
                                       &editedCap),
                "occluder edit rebuilds");
        Require(editedCap.occluderKey!=carry.occluderKey,
                "the edit re-keys the occluder");
        Require(editedCap.topologyKey==carry.topologyKey,
                "the lit set holds across the edit");
        Require(editedCap.faceVertexCounts.IsIdentical(carry.faceVertexCounts),
                "counts carry across the edit");
        Require(editedCap.faceVertexIndices.IsIdentical(carry.faceVertexIndices),
                "indices carry across the edit");
        Require(!editedCap.normals.IsIdentical(carry.normals),
                "normals go fresh on an occluder edit");
        Require(!editedCap.points.IsIdentical(carry.points),
                "points go fresh on an occluder edit");
        auto expectEdited=edited;
        UsdGenScalpShadowPublication wantEdited;
        {
            UsdGenFurOcclusionParams bare=params;
            bare.occluderCache=nullptr; bare.scalpScratch=nullptr;
            uint64_t freshKey=0;
            Require(UsdGenBuildFurOcclusion(&expectEdited,nullptr,bare,&freshKey,
                                           &wantEdited),
                    "edited unseeded reference bake");
        }
        capSame(editedCap,wantEdited,"the edited carry matches the fresh build");
        // A move that parks the hair far to the side empties the lit set:
        // the vote only flips the cap's facing, so dropping the hair below
        // would re-orient onto it and stay lit, and sliding along one axis
        // would keep it on that axis' ray. Nothing carries, and the cap
        // still matches the fresh build. The occluder is restored
        // bit-for-bit so only the lit set differs.
        params.occluders[0].points[0][1]=savedY;
        auto uncovered=live;
        uncovered[0].xformMatrix.SetTranslate(GfVec3d(5,0,5));
        UsdGenScalpShadowPublication uncoveredCap=first;
        Require(UsdGenBuildFurOcclusion(&uncovered,&live,params,&volumeKey,
                                       &uncoveredCap),
                "uncovering move rebuilds");
        Require(uncoveredCap.IsEmpty(),"the uncovered cap is empty");
        Require(uncoveredCap.topologyKey!=first.topologyKey,
                "the lit set moves with the hair");
        Require(!uncoveredCap.faceVertexCounts.IsIdentical(first.faceVertexCounts),
                "counts go fresh when the lit set changes");
        Require(!uncoveredCap.faceVertexIndices.IsIdentical(first.faceVertexIndices),
                "indices go fresh when the lit set changes");
        Require(!uncoveredCap.normals.IsIdentical(first.normals),
                "normals go fresh when the lit set changes");
        Require(!uncoveredCap.points.IsIdentical(first.points),
                "points go fresh when the lit set changes");
        auto expectUncovered=uncovered;
        UsdGenScalpShadowPublication wantUncovered;
        {
            UsdGenFurOcclusionParams bare=params;
            bare.occluderCache=nullptr; bare.scalpScratch=nullptr;
            uint64_t freshKey=0;
            Require(UsdGenBuildFurOcclusion(&expectUncovered,nullptr,bare,
                                           &freshKey,&wantUncovered),
                    "uncovered unseeded reference bake");
        }
        capSame(uncoveredCap,wantUncovered,
                "the uncovered cap matches the fresh build");
        // A refined level exercises the general sub-face path's share of the
        // lit-set key: two triangles at ceiling 8 refine exactly 2x2.
        {
            std::vector<UsdGenTilePublication> scene{
                Comb(0,-0.55f,400,0.3f),
                Probe(1,probeAt,probeAt+GfVec3f(0,0.01f,0))};
            UsdGenFurOcclusionParams fine;
            fine.occluders.push_back(Quad(0.f,4.f,false));
            fine.scalpShadow=true;
            fine.scalpMaxTriangles=8;
            uint64_t fineKey=0;
            UsdGenScalpShadowPublication coarse;
            Require(UsdGenBuildFurOcclusion(&scene,nullptr,fine,&fineKey,
                                           &coarse),
                    "refined first bake");
            Require(coarse.tessLevel==2,"the ceiling refines exactly 2x2");
            Require(!coarse.IsEmpty(),"the refined cap is non-empty");
            auto shifted=scene;
            shifted[0].xformMatrix.SetTranslate(GfVec3d(0.05,0,0));
            UsdGenScalpShadowPublication refined=coarse;
            Require(UsdGenBuildFurOcclusion(&shifted,&scene,fine,&fineKey,
                                           &refined),
                    "shifted hair rebuilds refined");
            Require(refined.topologyKey==coarse.topologyKey,
                    "the refined lit set holds");
            Require(refined.faceVertexCounts.IsIdentical(coarse.faceVertexCounts),
                    "refined counts carry");
            Require(refined.faceVertexIndices.IsIdentical(coarse.faceVertexIndices),
                    "refined indices carry");
            Require(refined.normals.IsIdentical(coarse.normals),
                    "refined normals carry");
            Require(refined.points.IsIdentical(coarse.points)==
                    (refined.lift==coarse.lift),
                    "refined points carry exactly when the lift holds");
            Require(refined.digest!=coarse.digest,
                    "refined fresh depths re-key the digest");
            auto expectShifted=shifted;
            UsdGenScalpShadowPublication wantShifted;
            uint64_t shiftedKey=0;
            Require(UsdGenBuildFurOcclusion(&expectShifted,nullptr,fine,
                                           &shiftedKey,&wantShifted),
                    "refined unseeded reference bake");
            capSame(refined,wantShifted,
                    "the refined carry matches the fresh build");
        }
    }

    // ---- angular reconstruction ---------------------------------------------
    // Ground truth for direction d: rotate the groom so d maps to +Y and bake.
    // The axis sweeps are exact along the axes, and a ball of fibres keeps its
    // bounds and its density field under rotation, so the rotated bake's +Y
    // depth is the true optical depth along d in the original frame.
    {
        std::vector<UsdGenTilePublication> ball{HairShell(0,6000,12345u)};
        UsdGenFurOccluder const core=SphereMesh(0.5f,20,28);
        UsdGenFurOcclusionParams params; params.voxelSize=0.04f;
        params.occluders.push_back(core);
        UsdGenBuildFurOcclusion(&ball,nullptr,params);
        GfVec3f const directions[]={
            GfVec3f(1,1,0),GfVec3f(1,0,1),GfVec3f(0,1,1),GfVec3f(1,1,1),
            GfVec3f(-1,1,0),GfVec3f(1,-1,1),GfVec3f(2,1,0),GfVec3f(1,2,3),
            GfVec3f(-3,1,2),GfVec3f(1,-2,-1)};
        // Two metrics: optical depth itself, and the quantity the shader
        // actually uses, Tf = A_front ^ HairCount. The shift to power 4 is
        // chosen for the second -- a sharper blend keeps lit and shadowed
        // directions apart instead of averaging them -- and costs a little on
        // the first, so both are measured.
        double square4=0,square2=0,squareRef=0;
        double transmit4=0,transmit2=0; size_t samples=0;
        double const absorption=0.35;
        auto tf=[&](double tau){ return std::pow(absorption,std::max(0.0,tau-1.0)); };
        for(GfVec3f d:directions) {
            d.Normalize();
            GfMatrix4d toY; toY.SetRotate(GfRotation(GfVec3d(d),GfVec3d(0,1,0)));
            auto rotated=Rotate(ball,toY);
            UsdGenFurOcclusionParams turned=params;
            turned.occluders={Rotate(core,toY)};
            UsdGenBuildFurOcclusion(&rotated,nullptr,turned);
            for(size_t i=0;i<ball[0].points.size();i+=7) {
                double const reference=Tau(rotated[0],"furTauP",i)[1];
                GfVec3f const p=Tau(ball[0],"furTauP",i), n=Tau(ball[0],"furTauN",i);
                double const t4=Reconstruct(p,n,d,4.f), t2=Reconstruct(p,n,d,2.f);
                square4+=(t4-reference)*(t4-reference);
                square2+=(t2-reference)*(t2-reference);
                squareRef+=reference*reference;
                transmit4+=(tf(t4)-tf(reference))*(tf(t4)-tf(reference));
                transmit2+=(tf(t2)-tf(reference))*(tf(t2)-tf(reference));
                ++samples;
            }
        }
        double const rms4=std::sqrt(square4/double(samples));
        double const rms2=std::sqrt(square2/double(samples));
        double const rmsRef=std::sqrt(squareRef/double(samples));
        double const tone4=std::sqrt(transmit4/double(samples));
        double const tone2=std::sqrt(transmit2/double(samples));
        std::printf("  angular reconstruction over %zu samples: reference tau RMS %.3f;"
                    " tau error cos^4 %.3f (%.0f%%) cos^2 %.3f (%.0f%%);"
                    " Tf error cos^4 %.4f cos^2 %.4f\n",
                    samples,rmsRef,rms4,100*rms4/rmsRef,rms2,100*rms2/rmsRef,tone4,tone2);
        Require(tone4<tone2,
                "the cosine-power-4 blend beats power 2 on transmittance, which is what is shaded");
        Require(rms4<0.6*rmsRef,"oblique optical depth stays within the documented bound");
    }

    // ---- the opaque shell must not shadow the hair standing on it ------------
    // A sparse groom on a large plane, lit from the side: nothing is between
    // any CV and a horizontal light, so the hair count along the whole strand
    // must be about zero -- while a receiver BELOW the plane must still be
    // fully blocked looking up through it. Getting this wrong turns the lower
    // part of every strand black and is invisible in a dense groom.
    {
        // Sized like examples/styled-fur-plane.usda: ~1200 strands on a unit
        // plane, 0.004 wide, so the numbers below are the ones that scene sees.
        std::vector<UsdGenTilePublication> scene;
        UsdGenTilePublication grass;
        grass.tile=0;grass.xformMatrix.SetIdentity();
        grass.primPath=SdfPath("/fur/tile_0");
        int const side=35, cvs=8; float const length=0.35f;
        for(int i=0;i<side;++i) for(int j=0;j<side;++j) {
            GfVec3f const root(-0.5f+float(i)/float(side-1),0.f,
                               -0.5f+float(j)/float(side-1));
            for(int k=0;k<cvs;++k)
                grass.points.push_back(root+GfVec3f(0,length*float(k)/float(cvs-1),0));
            grass.curveVertexCounts.push_back(cvs);
        }
        grass.widths=VtFloatArray{0.004f};
        scene.push_back(grass);
        scene.push_back(Probe(1,GfVec3f(0.02f,-0.25f,0.02f),GfVec3f(0.02f,-0.2f,0.02f)));
        auto bakeGrass=[&](bool withSurface) {
            auto copy=scene;
            UsdGenFurOcclusionParams params;
            if(withSurface) params.occluders.push_back(Quad(0.f,4.f,false));
            UsdGenBuildFurOcclusion(&copy,nullptr,params);
            return copy;
        };
        auto bare=bakeGrass(false), floored=bakeGrass(true);

        GfVec3f const sideways[]={GfVec3f(1,0,0),GfVec3f(-1,0,0),
                                  GfVec3f(0,0,1),GfVec3f(0,0,-1),
                                  GfVec3f(1,0.15f,0),GfVec3f(-1,0.15f,1)};
        // The surface's own shell is the only difference between the two
        // bakes, so anything it adds to a CV's sideways depth is light the
        // hair loses to the ground it stands on.
        float worst=0; size_t worstCv=0;
        for(size_t i=0;i<scene[0].points.size();++i) {
            GfVec3f const bp=Tau(bare[0],"furTauP",i), bn=Tau(bare[0],"furTauN",i);
            GfVec3f const fp=Tau(floored[0],"furTauP",i), fn=Tau(floored[0],"furTauN",i);
            for(GfVec3f d:sideways) {
                d.Normalize();
                float const added=Reconstruct(fp,fn,d)-Reconstruct(bp,bn,d);
                if(added>worst) { worst=added; worstCv=i; }
            }
        }
        float const under=Reconstruct(Tau(floored[1],"furTauP",0),
                                      Tau(floored[1],"furTauN",0),GfVec3f(0,1,0));
        // The profile is printed because it is the thing that looks like a bug
        // and is not one: a horizontal path through a unit-wide mat of upright
        // fibres crosses many of them near the ground and few near the tips,
        // so a grazing light legitimately leaves the lower strand dark.
        std::printf("  grazing light, hair count by height (x = fraction of "
                    "strand length):");
        for(int bucket=0;bucket<4;++bucket) {
            double sum=0; size_t count=0;
            for(size_t i=0;i<scene[0].points.size();++i) {
                float const t=scene[0].points[i][1]/length;
                if(t<float(bucket)/4.f || t>=float(bucket+1)/4.f) continue;
                sum+=Reconstruct(Tau(floored[0],"furTauP",i),
                                 Tau(floored[0],"furTauN",i),GfVec3f(1,0,0));
                ++count;
            }
            if(count) std::printf(" %.2f:%.1f",0.125+0.25*bucket,sum/double(count));
        }
        std::printf("\n  the surface adds %.3f crossings at worst, at CV %zu "
                    "(height %.3f of %.2f); a receiver below it reads %.1f\n",
                    worst,worstCv,scene[0].points[worstCv][1],length,under);
        Require(worst<0.05f,
                "the surface a groom stands on does not shadow the groom sideways");
        Require(under>0.75f*UsdGenFurTauClamp,
                "the surface still blocks light to whatever is behind it");
    }

    // ---- the scalp-shadow cap ------------------------------------------------
    // Hair over half a plane: the cap must darken the half under it and leave
    // the bare half alone, and must sit just off the surface without being
    // pickable, doubled or mis-wound.
    {
        auto bake=[&](bool cap,int strands) {
            std::vector<UsdGenTilePublication> scene{Comb(0,-0.55f,strands,0.3f)};
            UsdGenFurOcclusionParams params;
            params.occluders.push_back(Quad(0.f,0.6f,false));
            params.scalpShadow=cap;
            UsdGenScalpShadowPublication shadow;
            UsdGenBuildFurOcclusion(&scene,nullptr,params,nullptr,&shadow);
            return shadow;
        };
        Require(bake(false,400).IsEmpty(),"no cap unless it is asked for");
        UsdGenScalpShadowPublication shadow=bake(true,400);
        Require(!shadow.IsEmpty(),"the cap is built over an occluder with hair");
        size_t const points=shadow.points.size();
        Require(shadow.normals.size()==points,"one normal per cap point");
        Require(!shadow.faceVertexCounts.empty(),"the cap has faces");
        for(int count:shadow.faceVertexCounts) Require(count==3,"cap faces are triangles");
        Require(shadow.faceVertexIndices.size()==shadow.faceVertexCounts.size()*3,
                "cap face cardinality");
        for(int index:shadow.faceVertexIndices)
            Require(index>=0 && size_t(index)<points,"cap index in range");
        for(size_t i=0;i<points;++i) {
            Require(std::fabs(shadow.normals[i].GetLength()-1.f)<1e-3f,"unit cap normal");
            Require(shadow.normals[i][1]>0.9f,"the cap faces away from the solid");
            // Lifted off the surface, but by far less than a strand's length.
            Require(shadow.points[i][1]>0.f && shadow.points[i][1]<0.05f,
                    "the cap is lifted a hair's breadth off the surface");
        }
        UsdGenPlane const& capP=Plane(shadow.extraUniform,"furTauP");
        UsdGenPlane const& capN=Plane(shadow.extraUniform,"furTauN");
        Require(capP.interpolation==TfToken("vertex") && capP.arity==3 &&
                capP.f.size()==points*3,"cap depth is vertex float3");
        Require(capN.f.size()==points*3,"cap depth is vertex float3");
        // Hair sits over x in [-0.5, -0.31]; the far half of the quad is bare.
        // Straight up is what "how much hair is over this patch of skin" means.
        // A horizontal axis is not: from anywhere on a surface it runs along
        // the surface and through whatever hair stands on the far side, which
        // is the same trap the cap's own cull has to avoid.
        double under=0, bare=0; size_t nUnder=0, nBare=0;
        for(size_t i=0;i<points;++i) {
            float const depth=capP.f[i*3+1];
            if(shadow.points[i][0]<-0.35f) { under+=depth; ++nUnder; }
            else if(shadow.points[i][0]>0.2f) { bare+=depth; ++nBare; }
        }
        Require(nUnder>16 && nBare>16,"the cap spans both halves of the quad");
        under/=double(nUnder); bare/=double(nBare);
        std::printf("  scalp cap: %zu points, %zu triangles, mean depth %.3f "
                    "under hair vs %.3f on bare surface\n",
                    points,shadow.faceVertexCounts.size(),under,bare);
        Require(under>10.0*bare,"the cap darkens skin under hair, not bare skin");
        // Like for like against a CV at the same place: the cap is a receiver
        // sitting on the surface, so its count toward a direction must be the
        // count a zero-width strand there would read -- NOT that count minus
        // one. Sampling the cap a voxel out along the normal, which an earlier
        // version did to clear an opaque shell that its hair-only sweep never
        // contains, skipped the voxel the roots sit in and silently produced
        // exactly a host renderer's self-shifted value.
        {
            GfVec3f const at(-0.45f,0.f,0.f), up(0,1,0);
            std::vector<UsdGenTilePublication> withProbe{
                Comb(0,-0.55f,400,0.3f), Probe(1,at,at+GfVec3f(0,0.01f,0))};
            UsdGenFurOcclusionParams params;
            params.occluders.push_back(Quad(0.f,0.6f,false));
            params.scalpShadow=true;
            UsdGenScalpShadowPublication cap;
            UsdGenBuildFurOcclusion(&withProbe,nullptr,params,nullptr,&cap);
            float const atProbe=Reconstruct(Tau(withProbe[1],"furTauP",0),
                                            Tau(withProbe[1],"furTauN",0),up);
            // Nearest cap vertex to the probe.
            size_t nearest=0; float best=std::numeric_limits<float>::max();
            UsdGenPlane const& p=Plane(cap.extraUniform,"furTauP");
            UsdGenPlane const& n=Plane(cap.extraUniform,"furTauN");
            for(size_t i=0;i<cap.points.size();++i) {
                float const d=(cap.points[i]-at).GetLength();
                if(d<best) { best=d; nearest=i; }
            }
            GfVec3f const capP(p.f[nearest*3],p.f[nearest*3+1],p.f[nearest*3+2]);
            GfVec3f const capN(n.f[nearest*3],n.f[nearest*3+1],n.f[nearest*3+2]);
            float const atCap=Reconstruct(capP,capN,up);
            std::printf("  cap vs a CV at the same point: %.3f vs %.3f "
                        "crossings (%.3f away)\n",atCap,atProbe,best);
            Require(best<0.05f,"the cap has a vertex near the probe");
            Require(atProbe>0.5f,"the probe sees the hair above it");
            Require(std::fabs(atCap-atProbe)<0.25f*atProbe,
                    "the cap reads the same crossings a receiver there does");
        }
        Require(bare<0.2,"bare skin keeps its light");
        // Identity: a rebake of the same groom must reuse, a changed groom must
        // not -- the scene index dirties the prim on this digest alone.
        Require(shadow.digest==bake(true,400).digest,"the cap digest is stable");
        Require(shadow.digest!=bake(true,800).digest,"more hair changes the digest");
        // The reuse path leaves the caller's cap untouched, so a COW share of
        // the previous generation survives.
        std::vector<UsdGenTilePublication> live{Comb(0,-0.55f,400,0.3f)};
        UsdGenFurOcclusionParams params;
        params.occluders.push_back(Quad(0.f,0.6f,false));
        params.scalpShadow=true;
        uint64_t volumeKey=0;
        UsdGenScalpShadowPublication kept;
        Require(UsdGenBuildFurOcclusion(&live,nullptr,params,&volumeKey,&kept),
                "first cap bake");
        auto before=live;
        UsdGenScalpShadowPublication carried=kept;
        Require(!UsdGenBuildFurOcclusion(&live,&before,params,&volumeKey,&carried),
                "unchanged groom reuses");
        Require(carried.points.IsIdentical(kept.points),"the cap shares its arrays");
    }

    // ---- parallel mesh build: dispatcher identity + error equivalence ----
    // The occluder mesh build fans vertices/faces over the dispatcher pool
    // and merges in chunk order; the parallel bake must match the serial
    // bake bit-for-bit, and every malformed occluder must throw the same
    // message in both modes (the serial scan order's first error).
    {
        UsdGenScheduler sched(4);
        UsdGenWorkDispatcher disp=sched.MakeWorkDispatcher();
        UsdGenFurOcclusionParams params;
        params.occluders.push_back(SphereMesh(0.5f,12,16));
        params.occluders.push_back(Quad(0.4f,0.6f,true));
        params.scalpShadow=true;
        auto bake=[&](UsdGenWorkDispatcher* d,
                      std::vector<UsdGenTilePublication>* tiles,
                      UsdGenScalpShadowPublication* shadow) {
            UsdGenFurOcclusionParams q=params; q.dispatcher=d;
            return UsdGenBuildFurOcclusion(tiles,nullptr,q,nullptr,shadow);
        };
        std::vector<UsdGenTilePublication> a{HairShell(0,200,777u)}, b=a;
        UsdGenScalpShadowPublication sa, sb;
        Require(bake(nullptr,&a,&sa),"serial reference bake");
        Require(bake(&disp,&b,&sb),"parallel bake");
        Require(a.size()==b.size(),"tile count");
        for(size_t t=0;t<a.size();++t) for(auto name:{"furTauP","furTauN"}) {
            auto const& g=Plane(a[t],name).f, &w=Plane(b[t],name).f;
            Require(g.size()==w.size(),"plane size");
            for(size_t i=0;i<g.size();++i) Require(g[i]==w[i],"plane bits");
        }
        Require(sa.digest==sb.digest,"cap digest");
        // The fused merge hashes each digest slab from its chunk pieces, so
        // slab bytes routinely span piece (and word) boundaries; a third
        // dispatcher width re-chunks every slab boundary and must hash the
        // same digest over the same merged bytes.
        UsdGenScheduler schedW(7);
        UsdGenWorkDispatcher dispW=schedW.MakeWorkDispatcher();
        std::vector<UsdGenTilePublication> e{HairShell(0,200,777u)};
        UsdGenScalpShadowPublication se;
        Require(bake(&dispW,&e,&se),"wide parallel bake");
        Require(se.digest==sa.digest,"cap digest across dispatcher widths");
        Require(se.faceVertexCounts==sa.faceVertexCounts,"wide cap counts");
        Require(se.points.size()==sa.points.size(),"wide cap points");
        Require(sa.points.size()==sb.points.size(),"cap points");
        // The published extent reduces per-chunk emission extrema, so it
        // must match across chunkings exactly like the cap arrays do.
        Require(sb.extentMin==sa.extentMin && sb.extentMax==sa.extentMax,
                "parallel cap extent");
        Require(se.extentMin==sa.extentMin && se.extentMax==sa.extentMax,
                "wide cap extent");
        for(size_t i=0;i<sa.points.size();++i) {
            Require(sa.points[i]==sb.points[i],"cap point bits");
            Require(sa.normals[i]==sb.normals[i],"cap normal bits");
        }
        Require(sa.faceVertexCounts==sb.faceVertexCounts,"cap counts");
        Require(sa.faceVertexIndices==sb.faceVertexIndices,"cap indices");
        // The k==1 chunk emission writes through raw pointers into
        // scratch capacity and publishes sizes once per chunk: parallel
        // bakes reusing one scratch (cleared, re-reserved, rewritten)
        // must match the serial k==1 cap bit-for-bit. The ceiling of 8
        // forces the skip path (386 triangles > 8) and k==1.
        {
            std::vector<UsdGenTilePublication> ref{HairShell(0,200,777u)};
            UsdGenFurOcclusionParams serialP=params;
            serialP.scalpMaxTriangles=8;
            UsdGenScalpShadowPublication want;
            Require(UsdGenBuildFurOcclusion(&ref,nullptr,serialP,nullptr,
                                           &want),
                    "serial k==1 reference bake");
            Require(want.tessLevel==1,"the reference refines exactly 1x1");
            UsdGenScalpShadowScratch scratch;
            for(int round=0;round<2;++round) {
                std::vector<UsdGenTilePublication> t{HairShell(0,200,777u)};
                UsdGenFurOcclusionParams q=params;
                q.dispatcher=&disp; q.scalpScratch=&scratch;
                q.scalpMaxTriangles=8;
                UsdGenScalpShadowPublication s;
                Require(UsdGenBuildFurOcclusion(&t,nullptr,q,nullptr,&s),
                        "parallel scratch k==1 bake");
                Require(s.tessLevel==1,"the scratch bake refines exactly 1x1");
                Require(s.digest==want.digest,"scratch cap digest");
                Require(s.points==want.points,"scratch cap points");
                Require(s.normals==want.normals,"scratch cap normals");
                Require(s.faceVertexCounts==want.faceVertexCounts,
                        "scratch cap counts");
                Require(s.faceVertexIndices==want.faceVertexIndices,
                        "scratch cap indices");
            }
        }
        auto message=[&](UsdGenFurOccluder const& bad,UsdGenWorkDispatcher* d) {
            std::vector<UsdGenTilePublication> t{HairShell(0,8,1u)};
            UsdGenFurOcclusionParams q; q.occluders.push_back(bad);
            q.dispatcher=d;
            try { UsdGenBuildFurOcclusion(&t,nullptr,q); }
            catch(std::invalid_argument const& e) { return std::string(e.what()); }
            return std::string("<no throw>");
        };
        UsdGenFurOccluder const good=SphereMesh(0.5f,4,6);
        auto bad=good; bad.points[3][1]=std::numeric_limits<float>::quiet_NaN();
        Require(message(bad,nullptr)==message(bad,&disp),"NaN point message");
        Require(message(bad,nullptr)=="non-finite occluder point","NaN point throws");
        bad=good; bad.faceVertexCounts[1]=-2;
        Require(message(bad,nullptr)==message(bad,&disp),"negative count message");
        Require(message(bad,nullptr)=="negative occluder face vertex count",
                "negative count throws");
        bad=good; bad.faceVertexCounts[1]=1000000;
        Require(message(bad,nullptr)==message(bad,&disp),"cardinality message");
        Require(message(bad,nullptr)=="invalid occluder face cardinality",
                "cardinality throws");
        bad=good; bad.faceVertexIndices[2]=999999;
        Require(message(bad,nullptr)==message(bad,&disp),"index range message");
        Require(message(bad,nullptr)=="occluder face index out of range",
                "index range throws");
        bad=good; bad.faceVertexIndices[0]=-1;
        Require(message(bad,nullptr)==message(bad,&disp),"negative index message");
        Require(message(bad,nullptr)=="occluder face index out of range",
                "negative index throws");
        // First error in scan order wins, whichever check it is.
        bad=good; bad.faceVertexCounts[1]=1000000; bad.faceVertexCounts[3]=-2;
        Require(message(bad,nullptr)=="invalid occluder face cardinality",
                "earlier cardinality shadows later negative");
        Require(message(bad,&disp)==message(bad,nullptr),"parallel keeps scan order");
        bad=good; bad.faceVertexCounts[1]=-2; bad.faceVertexCounts[3]=1000000;
        Require(message(bad,nullptr)=="negative occluder face vertex count",
                "earlier negative shadows later cardinality");
        Require(message(bad,&disp)==message(bad,nullptr),"parallel keeps scan order");
    }

    std::vector<UsdGenTilePublication> bench{Make(0,0,50000)};
    auto start=std::chrono::steady_clock::now();
    UsdGenBuildFurOcclusion(&bench);
    auto end=std::chrono::steady_clock::now();
    std::printf("PASS fur optical depth; 50k strands / 100k CV: %.2f ms bake\n",
        std::chrono::duration<double,std::milli>(end-start).count());
    return 0;
} catch(std::exception const& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
