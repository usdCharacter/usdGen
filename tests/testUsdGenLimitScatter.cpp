#include "usdGen/limitSurface.h"
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/ops/scatter.h"
#include "usdGen/scheduler.h"
#include <cmath>
#include <cstdio>
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
    std::puts("limit scatter: analytic positions/normals, UVs, deterministic IDs, holes, orientation, tags, invalid inputs passed");
    return 0;
}
