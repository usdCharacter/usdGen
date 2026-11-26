// Test-only readback oracle. Production index generation never reads offsets,
// generated indices, primitive IDs, or draw counts back to the host.
#include "usdGenCurves/cuda/curveIndices.h"
#include "usdGenCurves/cuda/curveCompaction.h"
#ifdef USDGEN_HAS_STORM_INDEX_ORACLE
#include "pxr/imaging/hdSt/basisCurvesTopology.h"
#include "pxr/imaging/hd/bufferSource.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/base/tf/errorMark.h"
PXR_NAMESPACE_USING_DIRECTIVE
#endif
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>
using namespace usdGen::gpu;
namespace {
int failures = 0;
void Check(bool value, char const* label) {
    if (!value) { ++failures; std::fprintf(stderr, "FAIL: %s\n", label); }
}
void Cuda(cudaError_t value, char const* label) {
    if (value != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(value));
        std::exit(1);
    }
}
template<class T> DeviceView<const T> Read(DeviceBuffer<T> const& b) { return b.view(); }
template<class T> std::vector<T> Download(DeviceBuffer<T> const& b) {
    std::vector<T> result(b.size());
    if (!result.empty()) Cuda(cudaMemcpy(result.data(), b.data(), result.size()*sizeof(T),
                                        cudaMemcpyDeviceToHost), "oracle readback");
    return result;
}
struct Result { std::vector<int32_t> indices, primitives; uint32_t arity; };

Result Run(std::vector<uint32_t> const& counts, CurveIndexOptions options,
           bool capture = false, int corrupt = 0,
           DeviceView<const uint32_t> generatedOffsets = {}) {
    std::vector<uint32_t> offsets{0};
    for (auto c : counts) offsets.push_back(offsets.back()+c);
    const size_t points = offsets.back();
    cudaStream_t stream = nullptr;
    Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    CurveIndexRequirements req;
    Cuda(GetCurveIndexRequirements(options, counts.size(), points, &req, stream), "requirements");
    DeviceBuffer<uint32_t> input, status;
    DeviceBuffer<uint64_t> sizes, prefix, total;
    DeviceBuffer<unsigned char> scratch;
    DeviceBuffer<int> indices, primitives;
    Cuda(input.reset(offsets.size()), "offset allocation");
    Cuda(status.reset(1), "status allocation"); Cuda(total.reset(1), "total allocation");
    Cuda(sizes.reset(offsets.size()), "count allocation");
    Cuda(prefix.reset(offsets.size()), "prefix allocation");
    Cuda(scratch.reset(req.scanBytes), "scan allocation");
    Cuda(indices.reset(req.maxRecords*req.indexArity+7), "index allocation");
    Cuda(primitives.reset(req.maxRecords+7), "primitive allocation");
    if (corrupt == 1) offsets.front() = 1;
    if (corrupt == 2 && offsets.size()>2) offsets[1] = offsets.back()+1;
    if (corrupt == 3) ++offsets.back();
    if (corrupt == 4 && offsets.size()>2) offsets[1] = 0;
    if (!generatedOffsets.data)
        Cuda(cudaMemcpy(input.data(), offsets.data(), offsets.size()*sizeof(uint32_t),
                        cudaMemcpyHostToDevice), "offset upload");
    auto sourceOffsets = generatedOffsets.data ? generatedOffsets : Read(input);
    Cuda(cudaMemset(indices.data(), 0x5a, indices.size()*sizeof(int)), "index sentinel");
    Cuda(cudaMemset(primitives.data(), 0x5a, primitives.size()*sizeof(int)), "primitive sentinel");
    CurveIndexWorkspace workspace{sizes.view(), prefix.view(), scratch.view()};
    CurveIndexOutput output{{indices.data(), req.maxRecords*req.indexArity},
                            {primitives.data(), req.maxRecords}, total.view(), status.view()};
    if (capture) Cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin graph capture");
    Cuda(BuildCurveIndices(options, counts.size(), points, sourceOffsets, req, workspace, output, stream),
         "enqueue indices");
    if (capture) {
        cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
        Cuda(cudaStreamEndCapture(stream, &graph), "end graph capture");
        Cuda(cudaGraphInstantiate(&executable, graph, 0), "instantiate graph");
        Cuda(cudaGraphLaunch(executable, stream), "launch graph");
        Cuda(cudaStreamSynchronize(stream), "graph completion");
        Cuda(cudaGraphLaunch(executable, stream), "replay prepared graph");
        Cuda(cudaStreamSynchronize(stream), "replayed graph completion");
        Cuda(cudaGraphExecDestroy(executable), "destroy executable");
        Cuda(cudaGraphDestroy(graph), "destroy graph");
    } else Cuda(cudaStreamSynchronize(stream), "index completion");
    auto flat = Download(indices); auto owners = Download(primitives);
    auto n = Download(total)[0]; auto error = Download(status)[0];
    const bool badSegmented = options.wrap == CurveIndexWrap::Segmented &&
        std::any_of(counts.begin(),counts.end(),[](auto c) { return c%2; });
    const bool invalid = corrupt || badSegmented;
    Check(error == (corrupt ? 1u : badSegmented ? 2u : 0u), "exact GPU validation status");
    Check(n <= req.maxRecords && (!invalid || n == 0), "GPU record count respects capacity/error");
    if (n > req.maxRecords) std::exit(1);
    const size_t written = size_t(n)*req.indexArity;
    Check(std::all_of(flat.begin()+written,flat.end(),[](auto x) { return x==0x5a5a5a5a; }),
          "unused index capacity and guard remain untouched");
    Check(std::all_of(owners.begin()+n,owners.end(),[](auto x) { return x==0x5a5a5a5a; }),
          "unused primitive capacity and guard remain untouched");
    flat.resize(written); owners.resize(n);
    for (auto index : flat) Check(index >= 0 && size_t(index)<points, "index within source CV range");
    for (auto curve : owners) Check(curve >= 0 && size_t(curve)<counts.size(), "valid owning curve");
    // Host preflight must reject undersized outputs without enqueueing.
    if (req.maxRecords) {
        --output.indices.size;
        Check(BuildCurveIndices(options,counts.size(),points,sourceOffsets,req,workspace,output,stream)
                  == cudaErrorInvalidValue, "undersized output rejected");
        ++output.indices.size;
    }
    cudaStream_t other = nullptr;
    Cuda(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking),"other stream");
    Check(BuildCurveIndices(options,counts.size(),points,sourceOffsets,req,workspace,output,other)
              == cudaErrorInvalidValue,"prepared requirements reject a different stream");
    Cuda(cudaStreamDestroy(other),"destroy other stream");
    auto broken = req; ++broken.scanBytes;
    Check(BuildCurveIndices(options,counts.size(),points,sourceOffsets,broken,workspace,output,stream)
              == cudaErrorInvalidValue,"incorrect scratch requirements rejected");
    Cuda(cudaStreamDestroy(stream), "destroy stream");
    return {std::move(flat),std::move(owners),req.indexArity};
}

void CompactedTopology() {
    DeviceBuffer<float3> points;
    DeviceBuffer<uint32_t> offsets;
    DeviceBuffer<unsigned char> keep;
    Cuda(points.reset(5),"compaction points");
    Cuda(cudaMemset(points.data(),0,5*sizeof(float3)),"finite source points");
    Cuda(offsets.reset(3),"compaction offsets");
    Cuda(keep.reset(2),"compaction keep flags");
    uint32_t sourceOffsets[]{0,2,5}; unsigned char mask[]{0,1};
    Cuda(cudaMemcpy(offsets.data(),sourceOffsets,sizeof(sourceOffsets),cudaMemcpyHostToDevice),"source offsets");
    Cuda(cudaMemcpy(keep.data(),mask,sizeof(mask),cudaMemcpyHostToDevice),"keep flags");
    DeviceCurveGeometryView geometry;
    geometry.points=Read(points); geometry.curveOffsets=Read(offsets);
    geometry.curveCount=2; geometry.pointCount=5;
    CudaCurveCompaction compact;
    bool okay = compact.Apply(geometry,{},{},{},Read(keep),nullptr)==CurveCompactionStatus::Ok &&
                compact.Finish(nullptr)==CurveCompactionStatus::Ok;
    Check(okay,"actual CUDA compaction succeeds");
    if (!okay) return;
    auto generated=compact.view();
    Check(generated.curveCount==1 && generated.pointCount==3,"compaction scalar shape");
    if (generated.curveCount!=1 || generated.pointCount!=3) return;
    // Feed the generated device offsets directly, without downloading or
    // re-uploading them. Compaction's existing Finish is an explicit boundary;
    // only the subsequent index stage is claimed graph-capture compatible.
    auto result=Run({3},{},true,0,generated.curveOffsets);
    Check(result.indices==std::vector<int32_t>{0,0,0,1, 0,0,1,2, 0,1,2,2, 1,2,2,2, 2,2,2,2},
          "GPU-compacted topology directly feeds native-layout cubic indices");
}

void DrawCountPacking() {
    cudaStream_t stream = nullptr;
    Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "draw count stream");
    DeviceBuffer<uint64_t> records;
    DeviceBuffer<uint32_t> status, output;
    Cuda(records.reset(1), "draw count records");
    Cuda(status.reset(1), "draw count status");
    Cuda(output.reset(1), "draw count output");
    auto run = [&](uint64_t n, uint32_t s, uint32_t arity, size_t cap,
                   uint32_t expected) {
        Cuda(cudaMemcpyAsync(records.data(), &n, sizeof(n), cudaMemcpyHostToDevice, stream),
             "draw count record upload");
        Cuda(cudaMemcpyAsync(status.data(), &s, sizeof(s), cudaMemcpyHostToDevice, stream),
             "draw count status upload");
        Cuda(PackCurveDrawCount(Read(records), Read(status), arity, cap,
                                output.view(), stream), "draw count enqueue");
        Cuda(cudaStreamSynchronize(stream), "draw count synchronize");
        uint32_t actual = 0;
        Cuda(cudaMemcpy(&actual, output.data(), sizeof(actual), cudaMemcpyDeviceToHost),
             "draw count readback");
        Check(actual == expected, "draw count packed value");
    };
    run(0, 0, 1, 5, 0);
    run(5, 0, 4, 5, 20);
    run(6, 0, 2, 5, 0);
    run(uint64_t(UINT32_MAX) / 4 + 1, 0, 4, size_t(UINT64_MAX), 0);
    run(5, 1, 2, 5, 0);
    Check(PackCurveDrawCount(Read(records), Read(status), 3, 5,
                             output.view(), stream) == cudaErrorInvalidValue,
          "draw count rejects unsupported arity");
    DeviceView<uint32_t> emptyOutput{output.data(), 0};
    Check(PackCurveDrawCount(Read(records), Read(status), 2, 5,
                             emptyOutput, stream) == cudaErrorInvalidValue,
          "draw count rejects non-scalar output");

    uint64_t capturedRecords = 3;
    uint32_t capturedStatus = 0;
    Cuda(cudaMemcpy(records.data(), &capturedRecords, sizeof(capturedRecords),
                    cudaMemcpyHostToDevice), "capture records upload");
    Cuda(cudaMemcpy(status.data(), &capturedStatus, sizeof(capturedStatus),
                    cudaMemcpyHostToDevice), "capture status upload");
    Cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "draw count capture");
    Cuda(PackCurveDrawCount(Read(records), Read(status), 2, 5,
                            output.view(), stream), "draw count capture enqueue");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    Cuda(cudaStreamEndCapture(stream, &graph), "draw count capture end");
    Cuda(cudaGraphInstantiate(&executable, graph, 0), "draw count graph instantiate");
    Cuda(cudaGraphLaunch(executable, stream), "draw count graph launch");
    Cuda(cudaStreamSynchronize(stream), "draw count graph synchronize");
    uint32_t actual = 0;
    Cuda(cudaMemcpy(&actual, output.data(), sizeof(actual), cudaMemcpyDeviceToHost),
         "draw count captured readback");
    Check(actual == 6, "draw count graph full result");
    capturedRecords = 9;
    capturedStatus = 1;
    Cuda(cudaMemcpy(records.data(), &capturedRecords, sizeof(capturedRecords),
                    cudaMemcpyHostToDevice), "draw count replay records");
    Cuda(cudaMemcpy(status.data(), &capturedStatus, sizeof(capturedStatus),
                    cudaMemcpyHostToDevice), "draw count replay status");
    Cuda(cudaGraphLaunch(executable, stream), "draw count graph replay");
    Cuda(cudaStreamSynchronize(stream), "draw count replay synchronize");
    Cuda(cudaMemcpy(&actual, output.data(), sizeof(actual), cudaMemcpyDeviceToHost),
         "draw count replay readback");
    Check(actual == 0, "draw count graph upstream error result");
    Cuda(cudaGraphExecDestroy(executable), "draw count graph executable destroy");
    Cuda(cudaGraphDestroy(graph), "draw count graph destroy");
    Cuda(cudaStreamDestroy(stream), "draw count stream destroy");
}

void TileLocalSpans() {
    cudaStream_t stream = nullptr;
    Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
         "tile span stream");
    DeviceBuffer<uint32_t> globalOffsets;
    uint32_t const hostGlobalOffsets[]{5, 9, 14, 18, 23};
    Cuda(globalOffsets.reset(sizeof(hostGlobalOffsets) / sizeof(hostGlobalOffsets[0])),
         "tile span offsets");
    Cuda(cudaMemcpy(globalOffsets.data(), hostGlobalOffsets,
                    sizeof(hostGlobalOffsets), cudaMemcpyHostToDevice),
         "tile span upload");

    auto run = [&](CurveIndexOptions options, DeviceView<const uint32_t> offsets,
                   size_t curves, size_t points, uint32_t base,
                   std::vector<int32_t> const& expectedIndices,
                   std::vector<int32_t> const& expectedPrimitives,
                   bool capture = false) {
        CurveIndexRequirements req;
        Cuda(GetCurveIndexRequirements(options, curves, points, &req, stream),
             "tile span requirements");
        DeviceBuffer<uint64_t> counts, prefix, total;
        DeviceBuffer<unsigned char> scratch;
        DeviceBuffer<uint32_t> status;
        DeviceBuffer<int32_t> indices, primitives;
        Cuda(counts.reset(curves + 1), "tile span counts");
        Cuda(prefix.reset(curves + 1), "tile span prefix");
        Cuda(scratch.reset(req.scanBytes), "tile span scratch");
        Cuda(total.reset(1), "tile span total");
        Cuda(status.reset(1), "tile span status");
        Cuda(indices.reset(req.maxRecords * req.indexArity + 3), "tile span indices");
        Cuda(primitives.reset(req.maxRecords + 3), "tile span primitives");
        Cuda(cudaMemsetAsync(indices.data(), 0x5a, indices.size() * sizeof(int32_t), stream),
             "tile span index sentinel");
        Cuda(cudaMemsetAsync(primitives.data(), 0x5a, primitives.size() * sizeof(int32_t), stream),
             "tile span primitive sentinel");
        CurveIndexWorkspace workspace{counts.view(), prefix.view(), scratch.view()};
        CurveIndexOutput output{indices.view(), primitives.view(), total.view(), status.view()};
        if (capture)
            Cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                 "tile span capture begin");
        Cuda(BuildCurveIndices(options, CurveIndexSpan{offsets, curves, points, base},
                               req, workspace, output, stream),
             "tile span enqueue");
        if (capture) {
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            Cuda(cudaStreamEndCapture(stream, &graph), "tile span capture end");
            Cuda(cudaGraphInstantiate(&executable, graph, 0),
                 "tile span graph instantiate");
            Cuda(cudaGraphLaunch(executable, stream), "tile span graph launch");
            Cuda(cudaStreamSynchronize(stream), "tile span graph completion");
            Cuda(cudaGraphLaunch(executable, stream), "tile span graph replay");
            Cuda(cudaStreamSynchronize(stream), "tile span replay completion");
            Cuda(cudaGraphExecDestroy(executable), "tile span graph destroy");
            Cuda(cudaGraphDestroy(graph), "tile span graph definition destroy");
        }
        Cuda(cudaStreamSynchronize(stream), "tile span completion");
        Check(Download(status)[0] == 0, "tile span GPU status valid");
        Check(Download(total)[0] == expectedPrimitives.size(), "tile span record count");
        auto actualIndices = Download(indices);
        auto actualPrimitives = Download(primitives);
        Check(std::all_of(actualIndices.begin() + expectedIndices.size(), actualIndices.end(),
                          [](int32_t value) { return value == 0x5a5a5a5a; }),
              "tile span unused index capacity remains untouched");
        Check(std::all_of(actualPrimitives.begin() + expectedPrimitives.size(), actualPrimitives.end(),
                          [](int32_t value) { return value == 0x5a5a5a5a; }),
              "tile span unused primitive capacity remains untouched");
        actualIndices.resize(expectedIndices.size());
        actualPrimitives.resize(expectedPrimitives.size());
        Check(actualIndices == expectedIndices, "tile span exact local indices");
        Check(actualPrimitives == expectedPrimitives,
              "tile span exact tile-local primitive parameters");
    };

    CurveIndexOptions const linear{CurveIndexBasis::Linear,
                                   CurveIndexWrap::Nonperiodic,
                                   CurveIndexMode::Curves};
    // The two source views are consecutive, disjoint global point spans.  The
    // shared terminal offset is the next tile's base, not a primitive ID.
    run(linear, {globalOffsets.data(), 3}, 2, 9, 5,
        {0,1, 1,2, 2,3, 4,5, 5,6, 6,7, 7,8}, {0,0,0,1,1,1,1}, true);
    run(linear, {globalOffsets.data() + 2, 3}, 2, 9, 14,
        {0,1, 1,2, 2,3, 4,5, 5,6, 6,7, 7,8}, {0,0,0,1,1,1,1});
    run({CurveIndexBasis::BSpline, CurveIndexWrap::Nonperiodic,
         CurveIndexMode::Curves}, {globalOffsets.data(), 3}, 2, 9, 5,
        {0,1,2,3, 4,5,6,7, 5,6,7,8}, {0,1,1});
    run({CurveIndexBasis::BSpline, CurveIndexWrap::Pinned,
         CurveIndexMode::Points}, {globalOffsets.data(), 3}, 2, 9, 5,
        {0,1,2,3,4,5,6,7,8}, {0,0,0,0,1,1,1,1,1});
    DeviceBuffer<uint32_t> emptyOffset;
    uint32_t const emptyBase = UINT32_MAX;
    Cuda(emptyOffset.reset(1), "empty tile span offset");
    Cuda(cudaMemcpy(emptyOffset.data(), &emptyBase, sizeof(emptyBase),
                    cudaMemcpyHostToDevice), "empty tile span upload");
    run(linear, Read(emptyOffset), 0, 0, emptyBase, {}, {});

    auto failClosed = [&](std::vector<uint32_t> const& hostOffsets,
                          size_t curves, size_t points, uint32_t base,
                          char const* label) {
        CurveIndexRequirements req;
        Cuda(GetCurveIndexRequirements(linear, curves, points, &req, stream),
             "bad tile span requirements");
        DeviceBuffer<uint32_t> offsets, status;
        DeviceBuffer<uint64_t> counts, prefix, total;
        DeviceBuffer<unsigned char> scratch;
        DeviceBuffer<int32_t> indices, primitives;
        Cuda(offsets.reset(hostOffsets.size()), "bad tile span offsets");
        Cuda(cudaMemcpyAsync(offsets.data(), hostOffsets.data(),
                             hostOffsets.size() * sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream), "bad tile span upload");
        Cuda(counts.reset(curves + 1), "bad tile span counts");
        Cuda(prefix.reset(curves + 1), "bad tile span prefix");
        Cuda(scratch.reset(req.scanBytes), "bad tile span scratch");
        Cuda(total.reset(1), "bad tile span total"); Cuda(status.reset(1), "bad tile span status");
        Cuda(indices.reset(req.maxRecords * req.indexArity + 3), "bad tile span indices");
        Cuda(primitives.reset(req.maxRecords + 3), "bad tile span primitives");
        Cuda(cudaMemsetAsync(indices.data(), 0x5a, indices.size() * sizeof(int32_t), stream),
             "bad tile span index sentinel");
        Cuda(cudaMemsetAsync(primitives.data(), 0x5a, primitives.size() * sizeof(int32_t), stream),
             "bad tile span primitive sentinel");
        CurveIndexWorkspace workspace{counts.view(), prefix.view(), scratch.view()};
        CurveIndexOutput output{indices.view(), primitives.view(), total.view(), status.view()};
        Cuda(BuildCurveIndices(linear,
                               CurveIndexSpan{Read(offsets), curves, points, base},
                               req, workspace, output, stream), label);
        Cuda(cudaStreamSynchronize(stream), "bad tile span completion");
        Check(Download(status)[0] == 1 && Download(total)[0] == 0,
              "bad tile span fails closed with zero count");
        auto const flat = Download(indices); auto const owners = Download(primitives);
        Check(std::all_of(flat.begin(), flat.end(), [](int32_t value) {
                  return value == 0x5a5a5a5a;
              }) && std::all_of(owners.begin(), owners.end(), [](int32_t value) {
                  return value == 0x5a5a5a5a;
              }), "bad tile span leaves outputs untouched");
    };
    failClosed({4, 9, 14}, 2, 9, 5, "bad tile first offset");
    failClosed({5, 9, 13}, 2, 9, 5, "bad tile terminal offset");
    failClosed({5, 9, 8, 14}, 3, 9, 5, "bad tile interior offset");
    failClosed({UINT32_MAX, UINT32_MAX}, 1, 1, UINT32_MAX,
               "tile span scalar overflow");
    Cuda(cudaStreamDestroy(stream), "destroy tile span stream");
}

#ifdef USDGEN_HAS_STORM_INDEX_ORACLE
void CompareNative(std::vector<uint32_t> const& counts, CurveIndexOptions options) {
    TfErrorMark mark;
    TfToken basis;
    switch (options.basis) {
    case CurveIndexBasis::Linear: basis=HdTokens->bezier; break;
    case CurveIndexBasis::Bezier: basis=HdTokens->bezier; break;
    case CurveIndexBasis::BSpline: basis=HdTokens->bspline; break;
    case CurveIndexBasis::CatmullRom: basis=HdTokens->catmullRom; break;
    case CurveIndexBasis::CentripetalCatmullRom: basis=HdTokens->centripetalCatmullRom; break;
    }
    TfToken wrap;
    switch (options.wrap) {
    case CurveIndexWrap::Nonperiodic: wrap=HdTokens->nonperiodic; break;
    case CurveIndexWrap::Periodic: wrap=HdTokens->periodic; break;
    case CurveIndexWrap::Pinned: wrap=HdTokens->pinned; break;
    case CurveIndexWrap::Segmented: wrap=HdTokens->segmented; break;
    }
    VtIntArray nativeCounts(counts.begin(), counts.end());
    HdBasisCurvesTopology topology(options.basis==CurveIndexBasis::Linear ?
        HdTokens->linear : HdTokens->cubic, basis, wrap, nativeCounts, {});
    auto storm = HdSt_BasisCurvesTopology::New(topology);
    auto source = options.mode==CurveIndexMode::Points ? storm->GetPointsIndexBuilderComputation() :
        storm->GetIndexBuilderComputation(options.mode==CurveIndexMode::Hull);
    Check(source && (source->IsResolved() || source->Resolve()) && !source->HasResolveError(), "native Storm builder resolves");
    if (!source || !source->IsResolved()) std::exit(1);
    auto actual = Run(counts,options);
    Check(actual.indices.size()==source->GetNumElements()*actual.arity, "native Storm index count parity");
    auto expected = static_cast<int const*>(source->GetData());
    if (actual.indices.size()==source->GetNumElements()*actual.arity)
        Check(std::equal(actual.indices.begin(),actual.indices.end(),expected), "exact native Storm index parity");
    if (options.mode != CurveIndexMode::Points) {
        auto chained = source->GetChainedBuffers();
        Check(chained.size()==1 && chained[0]->GetName()==HdTokens->primitiveParam,
              "native Storm primitiveParam oracle present");
        if (chained.size()!=1) std::exit(1);
        Check(chained[0]->IsResolved() || chained[0]->Resolve(), "native primitiveParam resolves");
        Check(chained[0]->GetNumElements()==actual.primitives.size(), "native primitiveParam count parity");
        if (chained[0]->GetNumElements()==actual.primitives.size())
            Check(std::equal(actual.primitives.begin(),actual.primitives.end(),
                    static_cast<int const*>(chained[0]->GetData())), "exact native primitiveParam parity");
    }
    Check(mark.IsClean(), "native oracle emitted no Tf errors");
}
#endif
} // namespace

int main(int argc,char** argv) {
    bool native = argc>1 && std::string(argv[1])=="--native";
    if (native) {
#ifdef USDGEN_HAS_STORM_INDEX_ORACLE
        std::vector<uint32_t> shortCounts{1,2,3,4,5,7,8,11,257};
        std::vector<uint32_t> denseCounts;
        for (unsigned i=0;i<1031;++i) denseCounts.push_back(2+(i*37)%31);
        for (auto basis : {CurveIndexBasis::Linear,CurveIndexBasis::Bezier,CurveIndexBasis::BSpline,
                           CurveIndexBasis::CatmullRom,CurveIndexBasis::CentripetalCatmullRom})
            for (auto wrap : {CurveIndexWrap::Nonperiodic,CurveIndexWrap::Periodic,CurveIndexWrap::Pinned})
                for (auto mode : {CurveIndexMode::Curves,CurveIndexMode::Hull,CurveIndexMode::Points}) {
                    CompareNative(shortCounts,{basis,wrap,mode});
                    CompareNative(denseCounts,{basis,wrap,mode});
                    CompareNative({},{basis,wrap,mode});
                }
        CompareNative({2,4,8,6},{CurveIndexBasis::Linear,CurveIndexWrap::Segmented,CurveIndexMode::Curves});
        std::printf("Native Storm ragged index parity: %s\n",failures?"FAIL":"PASS");
        return failures?1:0;
#else
        std::fprintf(stderr,"Native oracle requires matching USDGEN_OPENUSD_SOURCE_DIR private headers\n");
        return 77;
#endif
    }
    auto pinned = Run({2},{},true);
    Check(pinned.indices==std::vector<int32_t>{0,0,0,1, 0,0,1,1, 0,1,1,1, 1,1,1,1, 1,1,1,1},
          "two-CV pinned bspline has exact native five-patch layout");
    Check(pinned.primitives==std::vector<int32_t>(5,0), "pinned primitive ownership");
    auto lines = Run({2,3},{CurveIndexBasis::Linear,CurveIndexWrap::Nonperiodic,CurveIndexMode::Curves});
    Check(lines.indices==std::vector<int32_t>{0,1,2,3,3,4} && lines.primitives==std::vector<int32_t>{0,1,1},
          "ragged line strip has no cross-curve segment");
    auto points = Run({2,3},{CurveIndexBasis::BSpline,CurveIndexWrap::Pinned,CurveIndexMode::Points});
    Check(points.indices==std::vector<int32_t>{0,1,2,3,4} && points.primitives==std::vector<int32_t>{0,0,1,1,1},
          "point indices and owning curves");
    Run({},{}); Run({1,1},{});
    CompactedTopology();
    DrawCountPacking();
    TileLocalSpans();
    for (int corrupt=1;corrupt<=4;++corrupt) Run({2,3},{},false,corrupt);
    Run({2,3},{CurveIndexBasis::Linear,CurveIndexWrap::Segmented,CurveIndexMode::Curves});
    CurveIndexRequirements req{123,99,321}, unchanged=req;
    Check(GetCurveIndexRequirements({static_cast<CurveIndexBasis>(99)},2,5,&req)==cudaErrorInvalidValue,
          "invalid basis rejected");
    Check(req.maxRecords==unchanged.maxRecords && req.indexArity==unchanged.indexArity && req.scanBytes==unchanged.scanBytes,
          "failed requirements leave result untouched");
    Check(GetCurveIndexRequirements({},std::numeric_limits<size_t>::max(),5,&req)==cudaErrorInvalidValue,
          "overflowing shape rejected");
    Check(GetCurveIndexRequirements({},0,5,&req)==cudaErrorInvalidValue,"nonempty points with no curves rejected");
    Check(GetCurveIndexRequirements({},5,2,&req)==cudaErrorInvalidValue,"too few points for positive curves rejected");
    Check(GetCurveIndexRequirements({},size_t(INT32_MAX)/2,size_t(INT32_MAX),&req)==cudaErrorInvalidValue,
          "conservative pinned record capacity overflow rejected");
    Check(GetCurveIndexRequirements({CurveIndexBasis::BSpline,CurveIndexWrap::Segmented},2,4,&req)==cudaErrorInvalidValue,
          "unsupported cubic segmented combination rejected");
    std::printf("GPU curve index validation/capture: %s\n",failures?"FAIL":"PASS");
    return failures?1:0;
}
