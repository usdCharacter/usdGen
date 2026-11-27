#include "usdGenCurves/cuda/scatterGrow.h"
#include "usdGenCuda/backendId.h"
#include "usdGen/api/ext/device/executionResources.h"
#include "usdGenMath/hash.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <limits>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "CHECK failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)
struct Relay { std::atomic<int> status{-1}; };
static void Done(cudaStream_t, cudaError_t status, void* p) noexcept { static_cast<Relay*>(p)->status.store(int(status)); }
template <class T> static bool Get(DeviceView<const T> v,std::vector<T>& o,cudaStream_t s) { return cudaMemcpyAsync(o.data(),v.data,o.size()*sizeof(T),cudaMemcpyDeviceToHost,s)==cudaSuccess && cudaStreamSynchronize(s)==cudaSuccess; }
static std::shared_ptr<ScatterGrowRoots> Roots() {
    auto r=std::make_shared<ScatterGrowRoots>();
    r->positions={make_float3(0,0,0)}; r->stableIds={1}; r->rootPrim={0}; r->rootUV={make_float2(0,0)};
    r->rootT={make_float3(1,0,0)}; r->rootB={make_float3(0,1,0)}; r->rootN={make_float3(0,0,1)}; return r;
}

int main(int argc, char** argv) {
    // Cardinality sizing is a host-only preflight and preserves its output
    // on invalid/overflowing requests.
    ScatterGrowRequirements requirements;
    CHECK(GetScatterGrowRequirements(0, 2, &requirements) == ScatterGrowStatus::Ok);
        CHECK(requirements.inputBytes == 0 && requirements.outputBytes == sizeof(uint32_t) &&
          requirements.statusBytes == 2 * sizeof(int) &&
          requirements.peakBytes == sizeof(uint32_t) + requirements.statusBytes);
    CHECK(GetScatterGrowRequirements(std::numeric_limits<size_t>::max(), 64,
                                    &requirements) == ScatterGrowStatus::InvalidTopology);
        CHECK(requirements.peakBytes == sizeof(uint32_t) + requirements.statusBytes);
    int count=0,device=-1; CHECK(cudaGetDeviceCount(&count)==cudaSuccess); if(!count)return 77; CHECK(cudaGetDevice(&device)==cudaSuccess);
    CHECK(ConfigureCudaExecutionResources(device,{size_t{8}<<20,0}));
    auto resources=FindUsdGenExecutionResourcePool({usdGen::UsdGenCudaBackendId(),device}); CHECK(resources);
    auto const baseline=resources->Snapshot(); cudaStream_t stream=nullptr; CHECK(cudaStreamCreate(&stream)==cudaSuccess);
    ScatterGrowControls controls; controls.cvCount=2;
    if (argc == 2 && std::strcmp(argv[1], "--quarantine-host") == 0) {
        // Isolated process: deliberately abandon terminal proof. Both the
        // H2D source and charged device destinations must remain retained.
        auto roots = Roots();
        std::weak_ptr<const ScatterGrowRoots> retained = roots;
        {
            CudaScatterGrow unproved;
            CHECK(unproved.BeginFresh(roots, controls, stream) == ScatterGrowStatus::Ok);
            roots.reset();
            CHECK(!retained.expired());
        }
        CHECK(!retained.expired());
        CHECK(resources->Snapshot().usedBytes > baseline.usedBytes);
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        CHECK(!retained.expired());
        CHECK(cudaStreamDestroy(stream) == cudaSuccess);
        std::puts("testUsdGenCudaScatterGrowHostQuarantine: PASS");
        return 0;
    }
    {
        // Invalid control preflight is allocation-free.
        ScatterGrowControls tooFew=controls; tooFew.cvCount=1; CudaScatterGrow invalidA;
        CHECK(invalidA.BeginFresh(Roots(),tooFew,stream)==ScatterGrowStatus::InvalidArgument);
        ScatterGrowControls tooMany=controls; tooMany.cvCount=65; CudaScatterGrow invalidB;
        CHECK(invalidB.BeginFresh(Roots(),tooMany,stream)==ScatterGrowStatus::InvalidArgument);
        ScatterGrowControls negative=controls; negative.length=-1; CudaScatterGrow invalidC;
        CHECK(invalidC.BeginFresh(Roots(),negative,stream)==ScatterGrowStatus::InvalidArgument);
        CHECK(resources->Snapshot().usedBytes==baseline.usedBytes && resources->Snapshot().byKind==baseline.byKind);

        // An unrelated stream's callback cannot prove the producer complete.
        // Rejection leaves the candidate pending and permits correct finish.
        {
            cudaStream_t other = nullptr;
            CHECK(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking) == cudaSuccess);
            CudaScatterGrow producer;
            Relay done;
            CHECK(producer.BeginFresh(Roots(), controls, stream) == ScatterGrowStatus::Ok);
            CHECK(producer.FinishFreshAsync(other, Done, &done) == ScatterGrowStatus::InvalidArgument);
            CHECK(done.status.load() == -1 && producer.pending());
            CHECK(producer.CommitFreshFinish() == ScatterGrowStatus::NoPendingUpdate);
            CHECK(producer.FinishFreshAsync(stream, Done, &done) == ScatterGrowStatus::Ok);
            CHECK(cudaStreamSynchronize(stream) == cudaSuccess && done.status.load() == int(cudaSuccess));
            CHECK(producer.CommitFreshFinish() == ScatterGrowStatus::Ok);
            CHECK(cudaStreamDestroy(other) == cudaSuccess);
        }

        // Capture is rejected before it can bind the producer or reserve/upload.
        CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal)==cudaSuccess);
        CudaScatterGrow captured;
        CHECK(captured.BeginFresh(Roots(),controls,stream)==ScatterGrowStatus::InvalidArgument);
        cudaGraph_t graph=nullptr; CHECK(cudaStreamEndCapture(stream,&graph)==cudaSuccess); if(graph) CHECK(cudaGraphDestroy(graph)==cudaSuccess);
        CHECK(resources->Snapshot().usedBytes==baseline.usedBytes && resources->Snapshot().byKind==baseline.byKind);

        // Reversed authored range is canonicalized before the CUDA draw, as
        // CPU Grow does, rather than changing its deterministic distribution.
        ScatterGrowControls reversed=controls; reversed.seed=19; reversed.length=2; reversed.randomLo=1.5f; reversed.randomHi=.5f;
        CudaScatterGrow canonical; Relay canonicalRelay;
        CHECK(canonical.BeginFresh(Roots(),reversed,stream)==ScatterGrowStatus::Ok);
        CHECK(canonical.FinishFreshAsync(stream,Done,&canonicalRelay)==ScatterGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess && canonicalRelay.status.load()==int(cudaSuccess));
        CHECK(canonical.CommitFreshFinish()==ScatterGrowStatus::Ok);
        std::vector<float3> canonicalPoints(2); CHECK(Get(canonical.view().points,canonicalPoints,stream));
        float const expected=2.0f*(.5f+UsdGenDraw01(19,1,kSaltGrow));
        CHECK(std::fabs(canonicalPoints[1].z-expected)<2.0e-5f);

        // An empty committed generation is immutable just like a non-empty one.
        CudaScatterGrow empty; auto roots=std::make_shared<ScatterGrowRoots>(); Relay relay;
        CHECK(empty.BeginFresh(roots,controls,stream)==ScatterGrowStatus::Ok);
        CHECK(empty.FinishFreshAsync(stream,Done,&relay)==ScatterGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess && relay.status.load()==int(cudaSuccess));
        CHECK(empty.CommitFreshFinish()==ScatterGrowStatus::Ok && empty.generation()==1);
        CHECK(empty.BeginFresh(roots,controls,stream)==ScatterGrowStatus::InvalidArgument);
    }
    CHECK(resources->Snapshot().usedBytes==baseline.usedBytes && resources->Snapshot().byKind==baseline.byKind);
    CHECK(cudaStreamDestroy(stream)==cudaSuccess); std::puts("testUsdGenCudaScatterGrowPreflight: PASS"); return 0;
}
