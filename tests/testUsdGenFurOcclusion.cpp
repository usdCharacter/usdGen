#include "usdGen/furOcclusion.h"
#include "usdGen/generationStore.h"
#include "usdGenImaging/usdGenTilePublisher.h"
#include "pxr/base/gf/rotation.h"
#include "pxr/imaging/hd/primvarsSchema.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
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

    // ---- a solid thinner than the shell -----------------------------------
    // The shell is marked two to six voxels under every surface. Under one
    // side of a thin solid -- an ear -- that depth is already past the other
    // side, and the shell came out into the air beyond it: hair lying over an
    // imported head's ear read saturated toward every light and rendered black
    // in the ear's outline. A thin closed fin beside a haired sphere (which
    // keeps the solid-side vote unambiguous), and a zero-width probe just
    // above the fin, where that stray shell used to be.
    {
        auto box=[](GfVec3f const& lo,GfVec3f const& hi) {
            UsdGenFurOccluder b;
            b.points={GfVec3f(lo[0],lo[1],lo[2]),GfVec3f(hi[0],lo[1],lo[2]),
                      GfVec3f(hi[0],hi[1],lo[2]),GfVec3f(lo[0],hi[1],lo[2]),
                      GfVec3f(lo[0],lo[1],hi[2]),GfVec3f(hi[0],lo[1],hi[2]),
                      GfVec3f(hi[0],hi[1],hi[2]),GfVec3f(lo[0],hi[1],hi[2])};
            // Wound like SphereMesh as the bake sees it (its vote puts the
            // solid on the +normal side): the solid side is one decision per
            // bake, so every occluder in it must agree.
            b.faceVertexCounts={4,4,4,4,4,4};
            b.faceVertexIndices={1,2,3,0, 7,6,5,4, 4,5,1,0, 2,6,7,3, 3,7,4,0, 5,6,2,1};
            return b;
        };
        GfVec3f const above(1.15f,0.07f,0.f);
        std::vector<UsdGenTilePublication> scene{
            HairShell(0,1500,7u), Probe(1,above,above+GfVec3f(0.f,0.001f,0.f))};
        UsdGenFurOcclusionParams params;
        params.occluders.push_back(SphereMesh(0.5f,24,48));
        params.occluders.push_back(box(GfVec3f(1.0f,-0.002f,-0.1f),GfVec3f(1.3f,0.002f,0.1f)));
        UsdGenBuildFurOcclusion(&scene,nullptr,params);
        float const sideways=Reconstruct(Tau(scene[1],"furTauP",0),
                                         Tau(scene[1],"furTauN",0),GfVec3f(0,0,-1));
        float const up=Reconstruct(Tau(scene[1],"furTauP",0),
                                   Tau(scene[1],"furTauN",0),GfVec3f(0,1,0));
        float const down=Reconstruct(Tau(scene[1],"furTauP",0),
                                     Tau(scene[1],"furTauN",0),GfVec3f(0,-1,0));
        std::printf("  above a 4 mm fin: %.2f crossings sideways, %.2f up, %.2f "
                    "down through the fin\n",sideways,up,down);
        Require(sideways<1.f && up<1.f,
                "the shell under a thin solid does not come out the other side");
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

    // ---- one cap per surface -------------------------------------------------
    // Descriptions over the same surface bake caps on the same lattice, so the
    // scene index can sum them into the single cap it publishes: the sum keeps
    // every triangle either kept, adds the depths point for point, and carries
    // a per-point hair colour only when the colours differ.
    {
        auto capOver=[&](UsdGenTilePublication const& hair,float half) {
            std::vector<UsdGenTilePublication> scene{hair};
            UsdGenFurOcclusionParams params;
            params.occluders.push_back(Quad(0.f,half,false));
            params.scalpShadow=true;
            UsdGenScalpShadowPublication cap;
            UsdGenBuildFurOcclusion(&scene,nullptr,params,nullptr,&cap);
            return cap;
        };
        UsdGenScalpShadowPublication const left=capOver(Comb(0,-0.55f,400,0.3f),0.6f);
        UsdGenScalpShadowPublication const right=capOver(Comb(1,0.15f,400,0.3f),0.6f);
        Require(!left.IsEmpty() && !right.IsEmpty(),"both descriptions bake a cap");
        Require(left.latticeKey!=0 && left.latticeKey==right.latticeKey,
                "caps over one surface share a lattice, whatever their hair");
        Require(left.latticeIds.size()==left.points.size(),"one lattice id per cap point");
        Require(left.lift>0.f && left.lift<0.05f,"the cap records its lift");
        UsdGenScalpShadowPublication const elsewhere=capOver(Comb(0,-0.55f,400,0.3f),0.7f);
        Require(elsewhere.latticeKey!=left.latticeKey,"another surface is another lattice");

        GfVec3f const dark(0.02f,0.01f,0.005f), pale(0.8f,0.7f,0.5f);
        UsdGenScalpShadowPublication const ignored=
            UsdGenMergeScalpShadows({{&left,dark},{&elsewhere,dark}});
        Require(ignored.digest==left.digest && ignored.points.IsIdentical(left.points),
                "a part on another lattice is left out");
        UsdGenScalpShadowPublication const alone=UsdGenMergeScalpShadows({{&left,dark}});
        Require(alone.points.IsIdentical(left.points) && alone.digest==left.digest,
                "one part passes through, sharing its arrays");
        Require(UsdGenMergeScalpShadows({}).IsEmpty(),"nothing to merge, no cap");

        UsdGenScalpShadowPublication const both=
            UsdGenMergeScalpShadows({{&left,dark},{&right,dark}});
        size_t const tl=left.faceVertexCounts.size(), tr=right.faceVertexCounts.size();
        size_t const tb=both.faceVertexCounts.size();
        std::printf("  merged caps: %zu + %zu triangles -> %zu\n",tl,tr,tb);
        Require(tb>=std::max(tl,tr) && tb<=tl+tr,
                "the sum keeps every triangle either part kept, once");
        Require(both.faceVertexIndices.size()==tb*3,"merged cap face cardinality");
        for(int index:both.faceVertexIndices)
            Require(index>=0 && size_t(index)<both.points.size(),"merged cap index in range");
        Require(both.normals.size()==both.points.size() &&
                both.latticeIds.size()==both.points.size(),"merged cap per-point arrays");
        auto byId=[](UsdGenScalpShadowPublication const& cap) {
            std::map<int,std::pair<GfVec3f,GfVec3f>> m;
            UsdGenPlane const& p=Plane(cap.extraUniform,"furTauP");
            for(size_t i=0;i<cap.points.size();++i)
                m[cap.latticeIds[i]]={cap.points[i],
                                      GfVec3f(p.f[i*3],p.f[i*3+1],p.f[i*3+2])};
            return m;
        };
        auto const inLeft=byId(left), inRight=byId(right), inBoth=byId(both);
        float worst=0;
        bool samePoints=true;
        for(auto const& [id,value]:inBoth) {
            GfVec3f want(0.f);
            for(auto const* part:{&inLeft,&inRight}) {
                auto const found=part->find(id);
                if(found==part->end()) continue;
                want+=found->second.second;
                samePoints = samePoints && found->second.first==value.first;
            }
            worst=std::max(worst,(value.second-want).GetLength());
        }
        Require(samePoints,"both parts put a lattice id at the same point");
        Require(worst<1e-5f,"merged depths are the sums of the parts' depths");
        bool leftSide=false, rightSide=false;
        for(GfVec3f const& p:both.points) { leftSide|=p[0]<-0.35f; rightSide|=p[0]>0.2f; }
        Require(leftSide && rightSide,"the merged cap covers both patches");

        bool colour=false;
        for(auto const& p:both.extraUniform) colour|=p.name==TfToken("furCapAlbedo");
        Require(!colour,"one hair colour needs no per-point colour");
        UsdGenScalpShadowPublication const mixed=
            UsdGenMergeScalpShadows({{&left,dark},{&right,pale}});
        UsdGenPlane const& albedo=Plane(mixed.extraUniform,"furCapAlbedo");
        Require(albedo.interpolation==TfToken("vertex") && albedo.arity==3 &&
                albedo.f.size()==mixed.points.size()*3,"per-point colour is vertex float3");
        // Straight under each patch, the colour is that patch's.
        float underLeft=-1, underRight=-1;
        for(size_t i=0;i<mixed.points.size();++i) {
            if(mixed.points[i][0]<-0.45f && mixed.points[i][0]>-0.5f) underLeft=albedo.f[i*3];
            if(mixed.points[i][0]>0.25f && mixed.points[i][0]<0.3f) underRight=albedo.f[i*3];
        }
        std::printf("  merged colour: %.3f under the dark patch, %.3f under the pale one\n",
                    underLeft,underRight);
        Require(underLeft>=0.f && underLeft<0.3f,"skin under dark hair takes the dark colour");
        Require(underRight>0.5f,"skin under pale hair takes the pale colour");
        Require(mixed.digest!=both.digest,"the hair colours are part of the merged identity");
    }

    // ---- local hair density --------------------------------------------------
    // furDensity is a gather of the HAIR-ONLY density grid: how crowded a CV's
    // own neighbourhood is, averaged over the three axes, in crossings per
    // voxel. It is published only when asked for, never reads the opaque
    // shell, and follows the same reuse rules as the depths.
    {
        auto has=[](UsdGenTilePublication const& tile) {
            for(auto const& p:tile.extraUniform) if(p.name==TfToken("furDensity")) return true;
            return false;
        };
        auto density=[](UsdGenTilePublication const& tile,size_t cv) {
            return Plane(tile,"furDensity").f[cv];
        };
        auto cleared=[](std::vector<UsdGenTilePublication> tiles) {
            for(auto& tile:tiles) tile.extraUniform.clear();   // a cooker's new tiles
            return tiles;
        };
        // A lone strand, a dense bundle (x in [0.2, 0.39]), and two zero-width
        // probes: one in the middle of the bundle, one far from any hair.
        auto scene=[] { return std::vector<UsdGenTilePublication>{
            Make(0,0,1),Make(1,0.2f,400),
            Probe(2,GfVec3f(0.3f,0.5f,0),GfVec3f(0.3f,0.55f,0)),
            Probe(3,GfVec3f(-0.4f,0.5f,0),GfVec3f(-0.4f,0.55f,0))}; };
        UsdGenFurOcclusionParams on; on.localDensity=true;
        UsdGenFurOcclusionParams const off;
        auto plain=scene(), lit=scene();
        Require(UsdGenBuildFurOcclusion(&plain,nullptr,off),"plain bake");
        Require(UsdGenBuildFurOcclusion(&lit,nullptr,on),"local density bake");
        for(auto const& tile:plain) Require(!has(tile),"no furDensity unless it is asked for");
        for(size_t t=0;t<lit.size();++t) {
            UsdGenPlane const& plane=Plane(lit[t],"furDensity");
            Require(plane.interpolation==TfToken("vertex") && plane.type==TfToken("float") &&
                    plane.arity==1,"furDensity is a vertex float");
            Require(plane.f.size()==lit[t].points.size(),"one furDensity value per CV");
            for(float x:plane.f)
                Require(std::isfinite(x)&&x>=0&&x<=UsdGenFurTauClamp,"finite bounded density");
            // Turning it on adds one plane and changes nothing else.
            Require(lit[t].extraUniform.size()==plain[t].extraUniform.size()+1,
                    "local density adds exactly one plane");
            for(size_t i=1;i<lit[t].extraUniform.size();++i)
                Require(lit[t].extraUniform[i-1].name<lit[t].extraUniform[i].name,
                        "planes stay sorted by name");
            for(auto name:{"furTauP","furTauN"}) {
                VtFloatArray const& withDensity=Plane(lit[t],name).f;
                VtFloatArray const& without=Plane(plain[t],name).f;
                Require(withDensity.size()==without.size(),"local density keeps tau cardinality");
                for(size_t i=0;i<without.size();++i)
                    Require(withDensity[i]==without[i],"local density leaves the depths alone");
            }
        }
        float const inside=density(lit[2],0), open=density(lit[3],0);
        float const lone=density(lit[0],0);
        std::printf("  local density: %.3f crossings per voxel inside a bundle, %.3f at a "
                    "lone strand's root, %.4f in the open\n",inside,lone,open);
        Require(inside>0.5f,"a CV inside a dense bundle reads the hair around it");
        Require(inside>10.f*open,"a crowded CV reads far more than an isolated one");
        Require(lone<inside,"a lone strand is less crowded than a bundle");
        {   // Diff sees the new plane generically: first appearance is new.
            UsdGenGeneration gPlain,gLit; gPlain.tiles=plain; gLit.tiles=lit;
            auto rep=UsdGenGenerationStore().Diff(gPlain,gLit);
            for(auto const& td:rep.tiles)
                Require(std::find(td.newPrimvars.begin(),td.newPrimvars.end(),
                                  TfToken("furDensity"))!=td.newPrimvars.end(),
                        "furDensity appears as a new primvar");
        }
        {   // Hydra transport: a plain vertex float primvar, one value per CV.
            auto litDs=usdGenImaging::UsdGenTilePublisher::BuildTileDataSource(lit[1]);
            auto litPv=HdPrimvarsSchema::GetFromParent(litDs).GetPrimvar(TfToken("furDensity"));
            Require(litPv.IsDefined()&&litPv.GetInterpolation()->GetTypedValue(0)==TfToken("vertex"),
                    "furDensity Hydra vertex transport");
            VtValue const value=litPv.GetPrimvarValue()->GetValue(0);
            Require(value.IsHolding<VtFloatArray>() &&
                    value.UncheckedGet<VtFloatArray>().size()==lit[1].points.size(),
                    "furDensity reaches Hydra as one float per CV");
            auto plainDs=usdGenImaging::UsdGenTilePublisher::BuildTileDataSource(plain[1]);
            Require(!HdPrimvarsSchema::GetFromParent(plainDs).GetPrimvar(
                        TfToken("furDensity")).IsDefined(),"no furDensity primvar when off");
        }

        // Hair only: sparse hair standing on an opaque quad, and a zero-width
        // probe INSIDE the shell the quad injects (2.5 to 5.5 voxels under the
        // surface, about 4 voxels at y = -0.1 here). The shell is ~32 crossings
        // per voxel; with it, furDensity there would sit near the clamp.
        auto sparse=[&](bool withOccluder) {
            std::vector<UsdGenTilePublication> s{
                Make(0,-0.1f,20), Probe(1,GfVec3f(0.1f,-0.1f,0.1f),GfVec3f(0.1f,-0.09f,0.1f))};
            UsdGenFurOcclusionParams params; params.localDensity=true;
            if(withOccluder) params.occluders.push_back(Quad(0.f,0.2f,false));
            UsdGenBuildFurOcclusion(&s,nullptr,params);
            return s;
        };
        auto bare=sparse(false), floored=sparse(true);
        GfVec3f const up=Tau(floored[1],"furTauP",0), down=Tau(floored[1],"furTauN",0);
        float worst=0, peak=0;
        for(size_t i=0;i<floored[0].points.size();++i) {
            worst=std::max(worst,std::fabs(density(floored[0],i)-density(bare[0],i)));
            peak=std::max(peak,density(floored[0],i));
        }
        std::printf("  inside the shell: tau up %.1f down %.1f, furDensity %.4f; "
                    "sparse hair peaks at %.3f, the occluder moves it by %.5f\n",
                    up[1],down[1],density(floored[1],0),peak,worst);
        Require(std::max(up[1],down[1])>0.25f*UsdGenFurTauClamp,
                "the probe sits inside the opaque shell, and the depths see it");
        Require(density(floored[1],0)<0.05f,"the opaque shell is not hair");
        Require(peak<2.f,"sparse hair reads as sparse next to an occluder");
        Require(worst<0.01f,"the occluder does not change the hair's own density");

        // Reuse: unchanged geometry and parameters share the density buffer;
        // the switch itself rebakes, with or without a volume key.
        uint64_t volumeKey=0;
        auto live=scene();
        Require(UsdGenBuildFurOcclusion(&live,nullptr,on,&volumeKey),"first density bake");
        auto const before=live;
        auto carried=cleared(before);
        Require(!UsdGenBuildFurOcclusion(&carried,&before,on,&volumeKey),
                "an unchanged density bake is reused");
        for(size_t t=0;t<carried.size();++t) {
            Require(Plane(carried[t],"furDensity").f.IsIdentical(Plane(before[t],"furDensity").f),
                    "the reuse path shares furDensity");
            Require(Plane(carried[t],"furTauP").f.IsIdentical(Plane(before[t],"furTauP").f),
                    "the reuse path still shares the depths");
        }
        auto dropped=cleared(before);
        Require(UsdGenBuildFurOcclusion(&dropped,&before,off,&volumeKey),
                "turning local density off rebakes");
        for(auto const& tile:dropped) Require(!has(tile),"turning it off drops furDensity");
        auto restored=cleared(before);
        Require(UsdGenBuildFurOcclusion(&restored,&dropped,on,&volumeKey),
                "turning local density back on rebakes");
        for(size_t t=0;t<restored.size();++t)
            for(size_t i=0;i<restored[t].points.size();++i)
                Require(density(restored[t],i)==density(before[t],i),"the rebake is deterministic");
        // Without a key, the previous planes alone must reveal the switch.
        auto keylessOn=cleared(before);
        Require(UsdGenBuildFurOcclusion(&keylessOn,&dropped,on),
                "a previous bake without furDensity is not reused for one that wants it");
        Require(has(keylessOn[0]),"the keyless rebake publishes furDensity");
        auto keylessOff=cleared(before);
        Require(UsdGenBuildFurOcclusion(&keylessOff,&before,off),
                "a previous bake with furDensity is not reused for one that does not");
        Require(!has(keylessOff[0]),"the keyless rebake drops furDensity");
        auto keylessSame=cleared(before);
        Require(!UsdGenBuildFurOcclusion(&keylessSame,&before,on),
                "a matching previous bake is reused without a key");
        Require(Plane(keylessSame[1],"furDensity").f.IsIdentical(Plane(before[1],"furDensity").f),
                "the keyless reuse shares furDensity");
    }

    std::vector<UsdGenTilePublication> bench{Make(0,0,50000)};
    auto start=std::chrono::steady_clock::now();
    UsdGenBuildFurOcclusion(&bench);
    auto end=std::chrono::steady_clock::now();
    std::printf("PASS fur optical depth; 50k strands / 100k CV: %.2f ms bake\n",
        std::chrono::duration<double,std::milli>(end-start).count());
    return 0;
} catch(std::exception const& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
