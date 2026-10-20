#include "usdGen/limitSurface.h"
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/ops/scatter.h"
#include "usdGen/scheduler.h"
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>
using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE
#define CHECK(c) do {if(!(c)){std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#c);return 1;}}while(false)
static UsdGenSurfaceDesc Surface() {
    UsdGenSurfaceDesc s;s.path=SdfPath("/Scalp");s.subdivisionScheme=TfToken("catmullClark");
    for(int y=-1;y<=2;++y)for(int x=-1;x<=2;++x)s.restPoints.push_back(GfVec3f(x,y,x*x+y*y));
    for(int y=0;y<3;++y)for(int x=0;x<3;++x){s.faceVertexCounts.push_back(4);for(int i:{y*4+x,y*4+x+1,(y+1)*4+x+1,(y+1)*4+x})s.faceVertexIndices.push_back(i);}
    s.points=s.restPoints;s.subsetFaces={4};return s;
}
static UsdGenGraphDesc Desc(UsdGenSurfaceDesc const& s,int level) {
    UsdGenGraphDesc d;d.description=SdfPath("/Groom");d.surfaces={s};
    UsdGenNodeDesc n;n.path=SdfPath("/Scatter");n.type=TfToken("UsdGenScatter");n.surfaces={s.path};
    n.params={{TfToken("density"),VtValue(100.0f),false},{TfToken("subdivisionLevel"),VtValue(level),false}};
    d.nodes={n};d.terminal=n.path;return d;
}
static bool Run(UsdGenGraphDesc const& d,UsdGenCurveBuffer* out,int threads=2) {
    UsdGenCompiler compiler;UsdGenGraph graph;if(!compiler.Compile(d,&graph).ok)return false;
    UsdGenScheduler scheduler(threads);UsdGenEvalContext ctx;ctx.desc=&graph.Desc();
    auto run=scheduler.Run(graph,ctx,1);if(run.diagnostics.HasErrors())return false;
    *out=graph.Output();return true;
}
static UsdGenSurfaceDesc BigGrid(int w,int h) {
    UsdGenSurfaceDesc s;s.path=SdfPath("/Scalp");s.subdivisionScheme=TfToken("catmullClark");
    for(int y=0;y<=h;++y)for(int x=0;x<=w;++x)s.restPoints.push_back(GfVec3f(float(x),float(y),0));
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){s.faceVertexCounts.push_back(4);int i0=y*(w+1)+x;for(int i:{i0,i0+1,i0+w+2,i0+w+1})s.faceVertexIndices.push_back(i);}
    s.points=s.restPoints;return s;
}
static UsdGenGraphDesc BigDesc(UsdGenSurfaceDesc const& s) {
    UsdGenGraphDesc d;d.description=SdfPath("/Groom");d.surfaces={s};
    UsdGenNodeDesc n;n.path=SdfPath("/Scatter");n.type=TfToken("UsdGenScatter");n.surfaces={s.path};
    n.params={{TfToken("density"),VtValue(4.0f),false},{TfToken("subdivisionLevel"),VtValue(0),false}};
    d.nodes={n};d.terminal=n.path;return d;
}
static bool RunDirect(UsdGenGraphDesc const& d,UsdGenCurveBuffer* out) {
    UsdGenNodeDesc const& node=d.nodes[0];
    UsdGenParamView params{&d,&node};
    UsdGenScatterOp op;UsdGenDiagnostics diag;
    if(!op.Bind(params,&diag))return false;
    auto cap=op.CreateCapture();if(!cap)return false;
    UsdGenCaptureContext ctx;ctx.desc=&d;ctx.params=&params;ctx.surface=0;
    ctx.seed=uint32_t(node.seed);ctx.diag=&diag;
    UsdGenCurveBuffer empty;
    if(!op.Capture(ctx,empty,cap.get(),&diag))return false;
    *out=cap->Buffer();return true;
}
// z = x^2 + y^2 over [-3,4]^2, quads except the four cells nearest (-3,-3),
// each split into two triangles (alternating diagonals). Vertex UVs are the
// cage (x, y), so a root's rootUV is the cage point it was drawn at.
static UsdGenSurfaceDesc MixedSurface() {
    UsdGenSurfaceDesc s;s.path=SdfPath("/Scalp");s.subdivisionScheme=TfToken("catmullClark");
    for(int y=-3;y<=4;++y)for(int x=-3;x<=4;++x){s.restPoints.push_back(GfVec3f(float(x),float(y),float(x*x+y*y)));s.uv.push_back(GfVec2f(float(x),float(y)));}
    auto id=[](int x,int y){return (y+3)*8+(x+3);};
    for(int y=-3;y<4;++y)for(int x=-3;x<4;++x){
        int const a=id(x,y),b=id(x+1,y),c=id(x+1,y+1),d=id(x,y+1);
        if(x<=-2&&y<=-2) {
            if((x+y)%2==0){for(int i:{a,b,c})s.faceVertexIndices.push_back(i);for(int i:{a,c,d})s.faceVertexIndices.push_back(i);}
            else{for(int i:{a,b,d})s.faceVertexIndices.push_back(i);for(int i:{b,c,d})s.faceVertexIndices.push_back(i);}
            s.faceVertexCounts.push_back(3);s.faceVertexCounts.push_back(3);
        } else {for(int i:{a,b,c,d})s.faceVertexIndices.push_back(i);s.faceVertexCounts.push_back(4);}
    }
    s.points=s.restPoints;return s;
}
static UsdGenGraphDesc GenDesc(UsdGenSurfaceDesc const& s,int level,char const* distribution,double density) {
    auto d=Desc(s,level);auto& n=d.nodes[0];
    n.params={{TfToken("distribution"),VtValue(TfToken(distribution)),false},
              {TfToken("density"),VtValue(density),false},{TfToken("subdivisionLevel"),VtValue(level),false}};
    return d;
}
// Limit-surface checks on mixed triangle/quad and n-gon topology; 0 on pass.
static int MixedTopology() {
    auto s=MixedSurface();std::string error;
    std::vector<int> start(s.faceVertexCounts.size()+1,0);
    for(size_t f=0;f<s.faceVertexCounts.size();++f)start[f+1]=start[f]+s.faceVertexCounts[f];
    UsdGenLimitSurface limit;CHECK(limit.Build(s,3,&error));
    GfVec3f p,du,dv,q,qu,qv;
    int const anyTri[3]={0,1,2};
    CHECK(!limit.Evaluate(0,.5f,.5f,&p,&du,&dv));   // (u,v) addressing is for quads
    CHECK(limit.EvaluateCagePoint(0,anyTri,GfVec3f(1,1,1),&p,&du,&dv));
    // Every shared edge and vertex is the same limit point from both faces.
    struct Use{int face,local;};
    std::vector<std::vector<Use>> uses(s.restPoints.size());
    for(size_t f=0;f<s.faceVertexCounts.size();++f)
        for(int k=0;k<s.faceVertexCounts[f];++k)uses[size_t(s.faceVertexIndices[size_t(start[f]+k)])].push_back({int(f),k});
    auto localOf=[&](int f,int v){for(auto const& u:uses[size_t(v)])if(u.face==f)return u.local;return -1;};
    auto third=[](int a,int b,int n){for(int k=0;k<n;++k)if(k!=a&&k!=b)return k;return -1;};
    int edges=0;
    for(size_t f=0;f<s.faceVertexCounts.size();++f){
        int const n=s.faceVertexCounts[f];
        for(int k=0;k<n;++k){
            int const a=s.faceVertexIndices[size_t(start[f]+k)],b=s.faceVertexIndices[size_t(start[f]+(k+1)%n)];
            for(auto const& other:uses[size_t(a)]){
                if(other.face<=int(f))continue;
                int const no=s.faceVertexCounts[size_t(other.face)];
                int const lb=localOf(other.face,b);
                if(lb<0||((other.local+1)%no!=lb&&(lb+1)%no!=other.local))continue;   // not an edge of the other face
                int const c0[3]={k,(k+1)%n,third(k,(k+1)%n,n)};
                int const c1[3]={other.local,lb,third(other.local,lb,no)};
                for(float t:{0.f,.2f,.5f,.8f,1.f}){
                    CHECK(limit.EvaluateCagePoint(int(f),c0,GfVec3f(1-t,t,0),&p,&du,&dv));
                    CHECK(limit.EvaluateCagePoint(other.face,c1,GfVec3f(1-t,t,0),&q,&qu,&qv));
                    CHECK((p-q).GetLength()<2e-5);
                }
                ++edges;
            }
        }
    }
    CHECK(edges>=88);   // 84 grid edges and the 4 triangle diagonals
    // No seam inside a face: walking a line through every corner sub-patch
    // moves the limit point in small, even steps (a sub-patch mix-up jumps).
    auto smoothWalk=[&](UsdGenLimitSurface const& L,int f,int const c[3],GfVec3f from,GfVec3f to,GfVec3f const& cageNormal){
        GfVec3f prev,pu,pv,cur,cu,cv;float worst=0,total=0;int const steps=400;
        if(!L.EvaluateCagePoint(f,c,from,&prev,&pu,&pv))return false;
        for(int i=1;i<=steps;++i){
            GfVec3f const w=from+(to-from)*(float(i)/steps);
            if(!L.EvaluateCagePoint(f,c,w,&cur,&cu,&cv))return false;
            if(GfDot(GfCross(cu,cv),cageNormal)<=0)return false;   // the face's own winding
            float const step=(cur-prev).GetLength();worst=std::max(worst,step);total+=step;prev=cur;
        }
        return total>0 && worst<4*total/steps;
    };
    for(size_t f=0;f<8;++f){
        GfVec3f const& A=s.restPoints[size_t(s.faceVertexIndices[size_t(start[f])])];
        GfVec3f const& B=s.restPoints[size_t(s.faceVertexIndices[size_t(start[f]+1)])];
        GfVec3f const& C=s.restPoints[size_t(s.faceVertexIndices[size_t(start[f]+2)])];
        GfVec3f const cage=GfCross(B-A,C-A);
        for(int k=0;k<3;++k){   // corner k to the opposite edge's midpoint, through the centre
            GfVec3f from(0),to(.5f,.5f,.5f);from[k]=1;to[k]=0;
            CHECK(smoothWalk(limit,int(f),anyTri,from*.98f+GfVec3f(.01f,.01f,.01f),to*.98f+GfVec3f(.01f,.01f,.01f)/1.f,cage));
        }
        CHECK(smoothWalk(limit,int(f),anyTri,GfVec3f(.8f,.15f,.05f),GfVec3f(.05f,.15f,.8f),cage));
    }
    // A quad keeps its (u,v) patch addressing: the triangle (0,1,2) maps affinely.
    {
        int const f=int(s.faceVertexCounts.size())-1;
        CHECK(s.faceVertexCounts[size_t(f)]==4);
        CHECK(limit.EvaluateCagePoint(f,anyTri,GfVec3f(.2f,.5f,.3f),&p,&du,&dv));
        CHECK(limit.Evaluate(f,.8f,.3f,&q,&qu,&qv));
        CHECK((p-q).GetLength()<1e-6 && (du-qu).GetLength()<1e-5 && (dv-qv).GetLength()<1e-5);
    }
    // An n-gon: one pentagon, its five sub-patches seamless and meeting at the centre.
    {
        UsdGenSurfaceDesc g;g.path=SdfPath("/Penta");g.subdivisionScheme=TfToken("catmullClark");
        for(int k=0;k<5;++k){double a=2*3.14159265358979*k/5;g.restPoints.push_back(GfVec3f(float(std::cos(a)),float(std::sin(a)*1.3),float(.2*k)));}
        g.faceVertexCounts={5};g.faceVertexIndices={0,1,2,3,4};g.points=g.restPoints;
        UsdGenLimitSurface pl;CHECK(pl.Build(g,3,&error));
        int const tri0[3]={0,1,2},tri1[3]={0,2,3},edge23[3]={2,3,0};
        GfVec3f a,b;
        CHECK(pl.EvaluateCagePoint(0,tri0,GfVec3f(0,0,1),&a,&du,&dv));   // corner 2 from two fan triangles
        CHECK(pl.EvaluateCagePoint(0,tri1,GfVec3f(0,1,0),&b,&du,&dv));
        CHECK((a-b).GetLength()<1e-5);
        CHECK(pl.EvaluateCagePoint(0,tri1,GfVec3f(0,.5f,.5f),&a,&du,&dv));   // edge 2-3 midpoint
        CHECK(pl.EvaluateCagePoint(0,edge23,GfVec3f(.5f,.5f,0),&b,&du,&dv));
        CHECK((a-b).GetLength()<1e-5);
        GfVec3f const cage=GfCross(g.restPoints[1]-g.restPoints[0],g.restPoints[2]-g.restPoints[0]);
        // corner 0 to edge 2-3 crosses sub-patches 0, 1/4, 2/3 and the centre
        CHECK(smoothWalk(pl,0,tri1,GfVec3f(.97f,.015f,.015f),GfVec3f(.02f,.49f,.49f),cage));
        CHECK(smoothWalk(pl,0,tri0,GfVec3f(.05f,.9f,.05f),GfVec3f(.05f,.05f,.9f),cage));
    }
    // Even scatter at level 3: the same strands as on the cage (ids, faces,
    // UVs), each moved to the limit point under its cage point with the limit
    // normal; on regular quads that is the analytic B-spline surface.
    UsdGenCurveBuffer cage,smooth,again;
    CHECK(Run(GenDesc(s,0,"even",30.0),&cage,1));
    CHECK(Run(GenDesc(s,3,"even",30.0),&smooth,1));CHECK(Run(GenDesc(s,3,"even",30.0),&again,4));
    CHECK(smooth.totalCurves>1000 && smooth.totalCurves==cage.totalCurves);
    CHECK(smooth.px==again.px && smooth.py==again.py && smooth.pz==again.pz && smooth.curveId==again.curveId);
    std::map<uint64_t,size_t> byId;for(size_t i=0;i<cage.totalCurves;++i)byId[cage.curveId[i]]=i;
    size_t regular=0,triangles=0,moved=0;
    for(size_t i=0;i<smooth.totalCurves;++i){
        auto it=byId.find(smooth.curveId[i]);CHECK(it!=byId.end());
        size_t const j=it->second;
        CHECK(smooth.rootPrim[i]==cage.rootPrim[j] && smooth.rootUV[i]==cage.rootUV[j]);
        int const f=smooth.rootPrim[i],n=s.faceVertexCounts[size_t(f)];
        GfVec2f const x=smooth.rootUV[i];
        // the cage point's barycentrics in the face's fan triangle that holds it
        bool found=false;
        for(int k=1;k+1<n&&!found;++k){
            GfVec2f const A=s.uv[size_t(s.faceVertexIndices[size_t(start[size_t(f)])])];
            GfVec2f const B=s.uv[size_t(s.faceVertexIndices[size_t(start[size_t(f)]+k)])];
            GfVec2f const C=s.uv[size_t(s.faceVertexIndices[size_t(start[size_t(f)]+k+1)])];
            double const det=double(B[0]-A[0])*(C[1]-A[1])-double(B[1]-A[1])*(C[0]-A[0]);
            double const w1=(double(x[0]-A[0])*(C[1]-A[1])-double(x[1]-A[1])*(C[0]-A[0]))/det;
            double const w2=(double(B[0]-A[0])*(x[1]-A[1])-double(B[1]-A[1])*(x[0]-A[0]))/det;
            if(w1<-1e-4||w2<-1e-4||w1+w2>1+1e-4)continue;
            int const corners[3]={0,k,k+1};
            CHECK(limit.EvaluateCagePoint(f,corners,GfVec3f(float(1-w1-w2),float(w1),float(w2)),&p,&du,&dv));
            found=true;
        }
        CHECK(found);
        GfVec3f const root(smooth.px[i],smooth.py[i],smooth.pz[i]);
        CHECK((root-p).GetLength()<2e-4);
        GfVec3f normal=GfCross(du,dv);normal.Normalize();
        CHECK((smooth.rootN[i]-normal).GetLength()<1e-3);
        if((GfVec3f(cage.px[j],cage.py[j],cage.pz[j])-root).GetLength()>1e-3)++moved;
        if(n==3)++triangles;
        if(x[0]>=0&&x[0]<=3&&x[1]>=0&&x[1]<=3){
            ++regular;
            CHECK(std::abs(root[0]-x[0])<2e-4 && std::abs(root[1]-x[1])<2e-4);
            CHECK(std::abs(root[2]-(x[0]*x[0]+x[1]*x[1]+2.0/3))<5e-4);
        }
    }
    CHECK(regular>100 && triangles>50 && moved>smooth.totalCurves/2);
    // Roots on hole faces go; the others stay. Left-handed flips the normals only.
    auto holes=s;holes.holeIndices={0,40};CHECK(Run(GenDesc(holes,3,"even",30.0),&again,1));
    CHECK(again.totalCurves>0 && again.totalCurves<smooth.totalCurves);
    for(size_t i=0;i<again.totalCurves;++i)CHECK(again.rootPrim[i]!=0 && again.rootPrim[i]!=40);
    auto left=s;left.orientation=TfToken("leftHanded");CHECK(Run(GenDesc(left,3,"even",30.0),&again,1));
    CHECK(again.px==smooth.px);for(size_t i=0;i<again.totalCurves;++i)CHECK((again.rootN[i]+smooth.rootN[i]).GetLength()<1e-5);
    // The other generation modes take the same route; the random per-face
    // path still needs quads and says so.
    CHECK(Run(GenDesc(s,3,"vertex",3.0),&again,1) && again.totalCurves>0);
    for(size_t i=0;i<again.totalCurves;++i)
        if(again.rootUV[i][0]>=0&&again.rootUV[i][0]<=3&&again.rootUV[i][1]>=0&&again.rootUV[i][1]<=3)
            CHECK(std::abs(again.pz[i]-(again.rootUV[i][0]*again.rootUV[i][0]+again.rootUV[i][1]*again.rootUV[i][1]+2.0/3))<5e-4);
    CHECK(!Run(GenDesc(s,3,"random",30.0),&again,1));
    auto loop=s;loop.subdivisionScheme=TfToken("none");CHECK(!Run(GenDesc(loop,3,"even",30.0),&again,1));
    return 0;
}
static bool SameRoots(UsdGenCurveBuffer const& a,UsdGenCurveBuffer const& b) {
    return a.totalCurves==b.totalCurves && a.totalCvs==b.totalCvs &&
        a.px==b.px && a.py==b.py && a.pz==b.pz && a.curveId==b.curveId &&
        a.rootPrim==b.rootPrim && a.rootUV==b.rootUV && a.rootT==b.rootT &&
        a.rootN==b.rootN && a.rootB==b.rootB && a.hairT==b.hairT;
}
int main() {
    usdGenRegisterM1Operators();auto s=Surface();std::string error;
    UsdGenLimitSurface limit;CHECK(limit.Build(s,3,&error));
    GfVec3f p,du,dv;CHECK(limit.Evaluate(4,.3f,.4f,&p,&du,&dv));
    CHECK(std::abs(p[0]-.3)<1e-5 && std::abs(p[1]-.4)<1e-5);
    // Independent analytic uniform bicubic B-spline, not cage interpolation.
    CHECK(std::abs(p[2]-(.09+.16+2.0/3))<1e-5);
    CHECK(std::abs(du[2]-.6)<1e-5 && std::abs(dv[2]-.8)<1e-5);
    UsdGenCurveBuffer a,b;CHECK(Run(Desc(s,3),&a,1));CHECK(Run(Desc(s,3),&b,4));
    CHECK(a.totalCurves>100 && a.px==b.px && a.py==b.py && a.pz==b.pz && a.curveId==b.curveId);
    for(size_t i=0;i<a.totalCurves;++i) {
        float u=a.rootUV[i][0],v=a.rootUV[i][1];
        CHECK(a.rootPrim[i]==4 && u>=0 && u<=1 && v>=0 && v<=1);
        CHECK(std::abs(a.px[i]-u)<1e-5 && std::abs(a.py[i]-v)<1e-5);
        CHECK(std::abs(a.pz[i]-(u*u+v*v+2.0/3))<2e-5);
        GfVec3f normal(-2*u,-2*v,1);normal.Normalize();
        CHECK((a.rootN[i]-normal).GetLength()<2e-5);
    }
    auto left=s;left.orientation=TfToken("leftHanded");CHECK(Run(Desc(left,3),&b));
    CHECK(a.px==b.px);for(size_t i=0;i<a.totalCurves;++i)CHECK((a.rootN[i]+b.rootN[i]).GetLength()<1e-6);
    auto holes=s;holes.holeIndices={4};CHECK(Run(Desc(holes,3),&b));CHECK(b.totalCurves==0);
    auto tags=s;tags.cornerIndices={5};tags.cornerSharpnesses={4};
    CHECK(UsdGenSubdivisionDigest(s)!=UsdGenSubdivisionDigest(tags));
    // The subdivision digest covers every topology input: each token and
    // each array below moves it (values are internal keys, never golden).
    {
        uint64_t const base = UsdGenSubdivisionDigest(s);
        CHECK(UsdGenSubdivisionDigest(s) == base);
        auto moved = [&](UsdGenSurfaceDesc m) {
            return UsdGenSubdivisionDigest(m) != base;
        };
        auto t = s; t.subdivisionScheme = TfToken("loop"); CHECK(moved(t));
        t = s; t.orientation = TfToken("leftHanded"); CHECK(moved(t));
        t = s; t.interpolateBoundary = TfToken("none"); CHECK(moved(t));
        t = s; t.faceVaryingLinearInterpolation = TfToken("none"); CHECK(moved(t));
        t = s; t.triangleSubdivisionRule = TfToken("smooth"); CHECK(moved(t));
        t = s; t.creaseMethod = TfToken("chaikin"); CHECK(moved(t));
        t = s; t.faceVertexCounts[0] = 3; CHECK(moved(t));
        t = s; t.faceVertexIndices[0] = 7; CHECK(moved(t));
        t = s; t.holeIndices = {2}; CHECK(moved(t));
        t = s; t.creaseIndices = {0, 1}; CHECK(moved(t));
        t = s; t.creaseLengths = {2}; CHECK(moved(t));
        t = s; t.creaseSharpnesses = {2.5f}; CHECK(moved(t));
        t = s; t.cornerIndices = {3}; CHECK(moved(t));
        t = s; t.cornerSharpnesses = {1.5f}; CHECK(moved(t));
    }
    // The tags digest covers everything except the two face arrays: tags
    // move it, face edits must not (callers mix those arrays themselves).
    {
        uint64_t const base = UsdGenSubdivisionTagsDigest(s);
        CHECK(UsdGenSubdivisionTagsDigest(s) == base);
        auto tmoved = [&](UsdGenSurfaceDesc m) {
            return UsdGenSubdivisionTagsDigest(m) != base;
        };
        auto t = s; t.subdivisionScheme = TfToken("loop"); CHECK(tmoved(t));
        t = s; t.orientation = TfToken("leftHanded"); CHECK(tmoved(t));
        t = s; t.interpolateBoundary = TfToken("none"); CHECK(tmoved(t));
        t = s; t.faceVaryingLinearInterpolation = TfToken("none"); CHECK(tmoved(t));
        t = s; t.triangleSubdivisionRule = TfToken("smooth"); CHECK(tmoved(t));
        t = s; t.creaseMethod = TfToken("chaikin"); CHECK(tmoved(t));
        t = s; t.holeIndices = {2}; CHECK(tmoved(t));
        t = s; t.creaseIndices = {0, 1}; CHECK(tmoved(t));
        t = s; t.creaseLengths = {2}; CHECK(tmoved(t));
        t = s; t.creaseSharpnesses = {2.5f}; CHECK(tmoved(t));
        t = s; t.cornerIndices = {3}; CHECK(tmoved(t));
        t = s; t.cornerSharpnesses = {1.5f}; CHECK(tmoved(t));
        t = s; t.faceVertexCounts[0] = 3; CHECK(!tmoved(t));
        t = s; t.faceVertexIndices[0] = 7; CHECK(!tmoved(t));
    }
    UsdGenScatterOp op;auto d0=Desc(s,3),d1=Desc(tags,3);UsdGenCaptureContext c0,c1;c0.desc=&d0;c1.desc=&d1;
    CHECK(op.CaptureDigest(c0)!=op.CaptureDigest(c1));
    // Pose and parent motion are downstream deformation inputs, not rest
    // scatter inputs. Rest/topology/UV/paint edits must still invalidate.
    d1=d0;d1.surfaces[0].points[0]+=GfVec3f(0,0,3);
    d1.surfaces[0].worldMatrix.SetTranslate(GfVec3d(2,0,0));
    d1.surfaces[0].surfaceGeneration=99;
    d1.surfaces[0].samples.push_back({42,d1.surfaces[0].points});
    CHECK(op.CaptureDigest(c0)==op.CaptureDigest(c1));
    d1.surfaces[0].restPoints[0]+=GfVec3f(0,0,1);
    CHECK(op.CaptureDigest(c0)!=op.CaptureDigest(c1));
    d1=d0;d1.surfaces[0].subsetFaces={3};
    CHECK(op.CaptureDigest(c0)!=op.CaptureDigest(c1));
    d1=d0;d1.surfaces[0].uv.push_back(GfVec2f(.2f,.3f));
    CHECK(op.CaptureDigest(c0)!=op.CaptureDigest(c1));
    d1=d0;d1.surfaces[0].densityMultiplier.assign(9,.5f);
    CHECK(op.CaptureDigest(c0)!=op.CaptureDigest(c1));
    CHECK(Run(Desc(s,0),&b));CHECK(b.totalCurves!=a.totalCurves);
    // Shuffled multi-face subset at level 0: every root lands on a
    // subset face (random corner-table order), deterministically.
    auto sub = s;
    sub.subsetFaces = {7, 2, 5};
    UsdGenCurveBuffer s1, s4;
    CHECK(Run(Desc(sub,0),&s1,1));
    CHECK(Run(Desc(sub,0),&s4,4));
    CHECK(s1.totalCurves > 0 && s1.totalCurves == s4.totalCurves);
    CHECK(s1.px == s4.px && s1.py == s4.py && s1.pz == s4.pz &&
          s1.curveId == s4.curveId && s1.rootPrim == s4.rootPrim);
    for (size_t i = 0; i < s1.totalCurves; ++i)
        CHECK(s1.rootPrim[i] == 7 || s1.rootPrim[i] == 2 || s1.rootPrim[i] == 5);
    // Threaded capture equivalence: 40K-face level-0 grids sit far above
    // the threading threshold, so scheduler(1), scheduler(4) and a direct
    // dispatcher-less Capture must agree bitwise on every plane — both
    // unrestricted (partition corner bases) and shuffled-subset (corner
    // table). Any scheduling-dependent byte fails here.
    {
        auto big = BigGrid(200, 200);
        auto bd = BigDesc(big);
        UsdGenCurveBuffer t1, t4, td;
        CHECK(Run(bd,&t1,1));CHECK(Run(bd,&t4,4));CHECK(RunDirect(bd,&td));
        CHECK(t1.totalCurves > 100000);
        CHECK(SameRoots(t1,t4) && SameRoots(t1,td));
        auto bsub = big;
        for (int f = int(big.faceVertexCounts.size()) - 1; f >= 0; f -= 3)
            bsub.subsetFaces.push_back(f);
        auto sd = BigDesc(bsub);
        UsdGenCurveBuffer u1, u4, ud;
        CHECK(Run(sd,&u1,1));CHECK(Run(sd,&u4,4));CHECK(RunDirect(sd,&ud));
        CHECK(u1.totalCurves > 30000);
        CHECK(SameRoots(u1,u4) && SameRoots(u1,ud));
        std::vector<char> inSub(big.faceVertexCounts.size(), 0);
        for (int f : bsub.subsetFaces) inSub[size_t(f)] = 1;
        for (size_t i = 0; i < u1.totalCurves; ++i)
            CHECK(inSub[size_t(u1.rootPrim[i])]);
    }
    CHECK(!Run(Desc(s,7),&b));
    tags=s;tags.subdivisionScheme=TfToken("none");CHECK(!Run(Desc(tags,3),&b));
    tags=s;tags.creaseIndices={0,1};tags.creaseLengths={2};tags.creaseSharpnesses={3};CHECK(limit.Build(tags,3,&error));
    tags.creaseIndices={999,1};CHECK(!limit.Build(tags,3,&error));
    if(MixedTopology())return 1;
    std::puts("limit scatter: analytic positions/normals, UVs, deterministic IDs, holes, orientation, tags, invalid inputs, "
              "mixed triangle/quad/n-gon limit continuity and generation-mode roots on the limit surface passed");
    return 0;
}
