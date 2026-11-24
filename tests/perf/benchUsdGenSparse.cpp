// benchUsdGenSparse — gate E-2 (T0, plan 11 §2.2 / 09 §5.1):
//
//   After a full G3 run (100 k curves x 8 CV, 5-op chain), dirty ~1% of the
//   chunks (2 of 196 on the noise node; the engine propagates that to the
//   same 2 chunks of length and width) and re-evaluate: median over 9 runs
//   at 8 threads must be <= 0.10 ms. The clean steady-state run is also
//   reported for context.
//
// Tier T0: engine core only (no Hydra, no stage — gate B-1).
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGen/types.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec3f.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

UsdGenGraphDesc MakeG3(){
   UsdGenGraphDesc d;
   d.description = SdfPath("/groom");
   d.terminal = SdfPath("/groom/width");
   d.time = 0.0;

   UsdGenSurfaceDesc s;
   s.path = SdfPath("/groom/surface");
   s.id = 0;
   const int NX = 1000, NY = 100;
   const int np = (NX + 1) * (NY + 1);
   s.restPoints = VtVec3fArray(np);
   s.uv = VtVec2fArray(np);
   {
       auto *rp = s.restPoints.data();
       auto *uv = s.uv.data();
       for (int j = 0; j <= NY; ++j)
           for (int i = 0; i <= NX; ++i) {
               const int k = j * (NX + 1) + i;
               rp[k] = GfVec3f(i * 0.1f, j * 0.1f, 0.0f);
               uv[k] = GfVec2f(float(i) / NX, float(j) / NY);
           }
   }
   s.faceVertexCounts = VtIntArray(NX * NY, 4);
   s.faceVertexIndices = VtIntArray(NX * NY * 4);
   {
       auto *fvi = s.faceVertexIndices.data();
       for (int j = 0; j < NY; ++j)
           for (int i = 0; i < NX; ++i) {
               const int a = j * (NX + 1) + i;
               const int o = (j * NX + i) * 4;
               fvi[o + 0] = a;
               fvi[o + 1] = a + 1;
               fvi[o + 2] = a + NX + 2;
               fvi[o + 3] = a + NX + 1;
           }
   }
   d.surfaces.push_back(std::move(s));

   auto addNode = [&](std::string const &name, TfToken type,
                      std::string const &input, int seed) {
       UsdGenNodeDesc n;
       n.path = SdfPath("/groom/" + name);
       n.type = type;
       n.enabled = true;
       n.seed = seed;
       if (!input.empty()) n.inputs.push_back(SdfPath("/groom/" + input));
       if (type == TfToken("UsdGenScatter"))
           n.surfaces.push_back(SdfPath("/groom/surface"));
       d.nodes.push_back(std::move(n));
   };
   addNode("scatter", TfToken("UsdGenScatter"), "", 42);
   addNode("grow", TfToken("UsdGenGrow"), "scatter", 43);
   addNode("noise", TfToken("UsdGenNoise"), "grow", 44);
   addNode("length", TfToken("UsdGenLength"), "noise", 45);
   addNode("width", TfToken("UsdGenWidth"), "length", 46);

   auto setp = [&](std::string const &name, TfToken p, VtValue v) {
       std::string const target = "/groom/" + name;
       for (auto &n : d.nodes)
           if (n.path == SdfPath(target))
              n.params.push_back(UsdGenParamValue{p, v, false});
   };
   setp("grow", TfToken("segments"), VtValue(8));
   setp("grow", TfToken("length"), VtValue(1.0));
   setp("noise", TfToken("noise:magnitude"), VtValue(0.05));
   setp("noise", TfToken("noise:frequency"), VtValue(3.0));
   setp("noise", TfToken("noise:octaves"), VtValue(2));
   setp("noise", TfToken("noise:correlation"), VtValue(0.5));
   setp("length", TfToken("length:mode"), VtValue(TfToken("scale")));
   setp("length", TfToken("length:value"), VtValue(1.2));
   setp("width", TfToken("width"), VtValue(0.02));
   return d;
}

double Median(std::vector<double> v){
   std::sort(v.begin(), v.end());
   return v[v.size() / 2];
}

}  // namespace
}  // namespace usdGen

using namespace usdGen;

int main(int argc, char **argv)
{
   bool isTerminal = false;
   for (int i = 1; i < argc; ++i) {
       if (std::string(argv[i]) == "--terminal") {
           isTerminal = true;
           break;
       }
   }
   UsdGenGraphDesc const desc = MakeG3();
   UsdGenCompiler compiler;
   UsdGenGraph graph;
   UsdGenCompileResult const r = compiler.Compile(desc, &graph);
   if (!r.ok) {
       std::printf("compile failed: ");
       for (auto const &e : r.errors) std::printf("%s ", e.c_str());
       std::printf("\n");
       return 1;
   }

   UsdGenScheduler scheduler(8);
   UsdGenEvalContext ctx;
   ctx.desc = &graph.Desc();
   ctx.time = desc.time;

   // Full run (capture + every chunk evaluated).
   UsdGenRunResult const full = scheduler.Run(graph, ctx, 1);
   if (full.diagnostics.HasErrors()) {
       std::printf("full-run diagnostics: ");
       for (auto const &e : full.diagnostics.errors) std::printf("%s ", e.c_str());
       std::printf("\n");
       return 1;
   }
   UsdGenNodeId const noise = graph.NodeIdForPath(SdfPath("/groom/noise"));
   uint32_t const nChunks = static_cast<uint32_t>(graph.Chunks(noise).size());
   UsdGenChunkId const cA = nChunks / 2;
   UsdGenChunkId const dirtyChunks[2] = {cA, cA + 1};

   // Context: clean steady-state commit.
   {
       auto const t0 = std::chrono::steady_clock::now();
       UsdGenRunResult const res = scheduler.Run(graph, ctx, 2);
       auto const t1 = std::chrono::steady_clock::now();
       std::printf("context: clean steady-state run %.3f ms (expect ~0, no chunk dirty)\n",
                   std::chrono::duration<double, std::milli>(t1 - t0).count());
       (void)res;
   }

   // E-2: dirty 2 of nChunks (~1%) on the noise node; the engine propagates
   // to the same chunks of length + width.
   std::vector<double> samples;
   samples.reserve(9);
   for (int i = 0; i < 9; ++i) {
       graph.DirtyChunks(noise, TfSpan<const UsdGenChunkId>(dirtyChunks, 2));
       auto const t0 = std::chrono::steady_clock::now();
       UsdGenRunResult const res = scheduler.Run(graph, ctx, 3 + i);
       auto const t1 = std::chrono::steady_clock::now();
       if (i == 1 && getenv("GENG_STATS")) {          // TPROF-TMP — remove
           for (auto const &st : res.nodeStats)
               ::printf("  node %u: capture %.3f ms eval %.3f ms chunks %llu\n",
                        st.id, st.captureMs, st.evalMs,
                        static_cast<unsigned long long>(st.chunksEvaluated));
       }
       if (res.diagnostics.HasErrors()) {
           std::printf("sparse run %d diagnostics: ", i);
           for (auto const &e : res.diagnostics.errors) std::printf("%s ", e.c_str());
           std::printf("\n");
           return 1;
       }
       samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
   }
   double const median = Median(samples);
   std::printf("E-2: %u of %u chunks dirty (%.2f%%) on noise; median %.3f ms "
               "(best %.3f, worst %.3f) vs budget 0.10 ms @ 8 threads -> %s\n",
               2u, nChunks, 100.0 * 2.0 / double(nChunks), median,
               *std::min_element(samples.begin(), samples.end()),
               *std::max_element(samples.begin(), samples.end()),
               median <= 0.10 ? "PASS" : "FAIL");
   char const *gate = std::getenv("USDGEN_GATE");
   return (median <= 0.10 || !(gate && std::string(gate) == "1")) ? 0 : 1;
}