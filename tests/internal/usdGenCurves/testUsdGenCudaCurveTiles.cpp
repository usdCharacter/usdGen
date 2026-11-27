// Test-only tile-span metadata readback.  Production retains these spans on
// device until the explicit publication boundary.
#include "usdGenCurves/cuda/curveTiles.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
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
template <class T> DeviceView<const T> Read(DeviceBuffer<T> const& buffer) {
    return buffer.view();
}
template <class T> std::vector<T> Download(DeviceBuffer<T> const& buffer) {
    std::vector<T> result(buffer.size());
    if (!result.empty()) Cuda(cudaMemcpy(result.data(), buffer.data(),
        result.size() * sizeof(T), cudaMemcpyDeviceToHost), "tile oracle readback");
    return result;
}

struct Result {
    CurveTileRequirements requirements;
    cudaError_t enqueueStatus = cudaSuccess;
    uint32_t status = 0;
    std::vector<CurveTileSpan> spans;
};

Result Run(CurveTileOptions options, std::vector<uint64_t> const& capture,
           std::vector<uint64_t> const& survivors,
           std::vector<uint32_t> const& offsets, size_t pointCount,
           bool captureGraph = false,
           std::vector<uint64_t> const& replaySurvivors = {},
           std::vector<uint32_t> const& replayOffsets = {},
           CurveTileOrder order = CurveTileOrder::SortedSurvivorSubset,
           bool corruptDomain = false) {
    cudaStream_t stream = nullptr;
    Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "tile stream");
    CurveTileRequirements requirements;
    Cuda(GetCurveTileRequirements(options, capture.size(), survivors.size(), pointCount,
                                  &requirements, stream), "tile requirements");
    DeviceBuffer<uint64_t> captureIds, survivorIds, sortedIds;
    DeviceBuffer<uint32_t> curveOffsets, status, sortedOrdinals;
    std::vector<std::pair<uint64_t,uint32_t>> lookup;
    std::vector<uint64_t> lookupIds;
    std::vector<uint32_t> lookupOrdinals;
    if (order == CurveTileOrder::CaptureOrderSurvivorSubset) {
        for (size_t i=0;i<capture.size();++i) lookup.emplace_back(capture[i],uint32_t(i));
        std::sort(lookup.begin(),lookup.end());
        for (auto const& entry:lookup) { lookupIds.push_back(entry.first); lookupOrdinals.push_back(entry.second); }
        if (corruptDomain && !lookupOrdinals.empty()) lookupOrdinals[0]=uint32_t(capture.size());
        Cuda(sortedIds.reset(lookupIds.size()),"sorted lookup IDs allocation");
        Cuda(sortedOrdinals.reset(lookupOrdinals.size()),"sorted lookup ordinals allocation");
        if (!lookup.empty()) {
            Cuda(cudaMemcpyAsync(sortedIds.data(),lookupIds.data(),lookupIds.size()*sizeof(uint64_t),cudaMemcpyHostToDevice,stream),"sorted lookup IDs upload");
            Cuda(cudaMemcpyAsync(sortedOrdinals.data(),lookupOrdinals.data(),lookupOrdinals.size()*sizeof(uint32_t),cudaMemcpyHostToDevice,stream),"sorted lookup ordinals upload");
        }
    }
    DeviceBuffer<CurveTileSpan> spans;
    Cuda(captureIds.reset(capture.size()), "capture ID allocation");
    Cuda(survivorIds.reset(survivors.size()), "survivor ID allocation");
    Cuda(curveOffsets.reset(offsets.size()), "survivor offset allocation");
    Cuda(status.reset(1), "tile status allocation");
    Cuda(spans.reset(requirements.tileCount + 2), "tile span allocation");
    if (!capture.empty()) Cuda(cudaMemcpyAsync(captureIds.data(), capture.data(),
        capture.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream), "capture ID upload");
    if (!survivors.empty()) Cuda(cudaMemcpyAsync(survivorIds.data(), survivors.data(),
        survivors.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream), "survivor ID upload");
    if (!offsets.empty()) Cuda(cudaMemcpyAsync(curveOffsets.data(), offsets.data(),
        offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), "survivor offset upload");
    Cuda(cudaMemsetAsync(spans.data(), 0x5a, spans.size() * sizeof(CurveTileSpan), stream),
         "tile span sentinel");
    CurveTileInput input{Read(captureIds), Read(survivorIds), Read(curveOffsets),
                         capture.size(), survivors.size(), pointCount, order};
    input.sortedCaptureStableIds=Read(sortedIds);
    input.sortedCaptureOrdinals=Read(sortedOrdinals);
    CurveTileOutput output{spans.view(), status.view()};
    if (captureGraph) Cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                           "tile graph capture begin");
    cudaError_t const enqueue = BuildCurveTiles(input, requirements, output, stream);
    if (enqueue != cudaSuccess) {
        Cuda(cudaStreamSynchronize(stream), "rejected tile completion");
        Result result;
        result.requirements = requirements;
        result.enqueueStatus = enqueue;
        result.spans = Download(spans);
        Cuda(cudaStreamDestroy(stream), "destroy tile stream after rejected enqueue");
        return result;
    }
    if (captureGraph) {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        Cuda(cudaStreamEndCapture(stream, &graph), "tile graph capture end");
        Cuda(cudaGraphInstantiate(&executable, graph, 0), "tile graph instantiate");
        Cuda(cudaGraphLaunch(executable, stream), "tile graph launch");
        Cuda(cudaStreamSynchronize(stream), "tile graph completion");
        if (!replaySurvivors.empty()) Cuda(cudaMemcpyAsync(survivorIds.data(),
            replaySurvivors.data(), replaySurvivors.size() * sizeof(uint64_t),
            cudaMemcpyHostToDevice, stream), "tile graph replay survivor upload");
        if (!replayOffsets.empty()) Cuda(cudaMemcpyAsync(curveOffsets.data(),
            replayOffsets.data(), replayOffsets.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice, stream), "tile graph replay offset upload");
        Cuda(cudaGraphLaunch(executable, stream), "tile graph replay");
        Cuda(cudaStreamSynchronize(stream), "tile graph replay completion");
        Cuda(cudaGraphExecDestroy(executable), "tile graph destroy");
        Cuda(cudaGraphDestroy(graph), "tile graph definition destroy");
    } else {
        Cuda(cudaStreamSynchronize(stream), "tile completion");
    }
    Result result;
    result.requirements = requirements;
    result.status = Download(status)[0];
    result.spans = Download(spans);
    Cuda(cudaStreamDestroy(stream), "destroy tile stream");
    return result;
}

bool Same(CurveTileSpan a, CurveTileSpan b) {
    return a.tile == b.tile && a.firstCurve == b.firstCurve &&
        a.curveCount == b.curveCount && a.firstPoint == b.firstPoint &&
        a.pointCount == b.pointCount;
}
bool Sentinel(CurveTileSpan const& span) {
    CurveTileSpan const expected{0x5a5a5a5aU, 0x5a5a5a5aU, 0x5a5a5a5aU,
                                 0x5a5a5a5aU, 0x5a5a5a5aU};
    return Same(span, expected);
}
bool GuardsUntouched(Result const& result) {
    return result.spans.size() == result.requirements.tileCount + 2 &&
        Sentinel(result.spans[result.requirements.tileCount]) &&
        Sentinel(result.spans[result.requirements.tileCount + 1]);
}

void GoldenRanges() {
    std::vector<uint64_t> capture(300);
    for (uint64_t i = 0; i < capture.size(); ++i) capture[i] = i;
    CurveTileOptions const options{128, 64};
    auto result = Run(options, capture, {0,127,128,129,255,299},
                      {0,2,5,9,14,20,27}, 27);
    Check(result.status == 0 && result.requirements.tileCount == 3,
          "ragged stable-ID tile build succeeds with three capture tiles");
    std::vector<CurveTileSpan> const initial{{0,0,2,0,5}, {1,2,3,5,15}, {2,5,1,20,7}};
    Check(std::equal(initial.begin(), initial.end(), result.spans.begin(), Same),
          "exact initial ragged stable-ID tile ranges");
    result = Run(options, capture, {0,127,128,129,255,299},
                 {0,2,5,9,14,20,27}, 27, true,
                      {0,127,128,255,256,299}, {0,2,5,9,14,20,27});
    Check(result.status == 0 && result.requirements.tileCount == 3,
          "ragged stable-ID graph replay succeeds");
    std::vector<CurveTileSpan> const expected{{0,0,2,0,5}, {1,2,2,5,9}, {2,4,2,14,13}};
    Check(std::equal(expected.begin(), expected.end(), result.spans.begin(), Same),
          "graph replay uses updated ragged stable-ID tile inputs");
    Check(GuardsUntouched(result), "tile graph output guards remain untouched");
    result = Run(options, capture, {0,299}, {0,2,9}, 9);
    std::vector<CurveTileSpan> const sparse{{0,0,1,0,2}, {1,1,0,2,0}, {2,1,1,2,7}};
    Check(result.status == 0 && std::equal(sparse.begin(), sparse.end(),
          result.spans.begin(), Same), "post-cull empty capture tile is retained");
    Check(GuardsUntouched(result), "sparse tile output guards remain untouched");
    result = Run(options, capture, {}, {0}, 0);
    Check(result.status == 0 && std::all_of(result.spans.begin(),
          result.spans.begin() + result.requirements.tileCount,
          [](CurveTileSpan const& span) { return span.curveCount == 0 && span.pointCount == 0; }),
          "all-cull retains every frozen capture tile");
    result = Run(options, {}, {}, {0}, 0);
    Check(result.status == 0 && result.requirements.tileCount == 0,
          "empty capture and cull are valid");
}

void IdentityCaptureOrder() {
    std::vector<uint64_t> capture(300);
    std::vector<uint32_t> offsets(capture.size() + 1);
    for (size_t i = 0; i < capture.size(); ++i) {
        // Deliberately unsorted: Scatter's capture/Morton order is authoritative.
        capture[i] = static_cast<uint64_t>(capture.size() - i);
        offsets[i] = static_cast<uint32_t>(i * 2);
    }
    offsets.back() = static_cast<uint32_t>(capture.size() * 2);

    auto result = Run({128,64}, capture, capture, offsets, capture.size() * 2,
                       false, {}, {}, CurveTileOrder::IdentityCaptureOrder);
    Check(result.enqueueStatus == cudaSuccess && result.status == 0 &&
          result.requirements.tileCount == 3,
          "identity capture order accepts unsorted IDs with matching survivors");
    std::vector<CurveTileSpan> const expected{
        {0, 0, 128, 0, 256}, {1, 128, 128, 256, 256}, {2, 256, 44, 512, 88}};
    Check(std::equal(expected.begin(), expected.end(), result.spans.begin(), Same),
          "identity capture order emits direct capture-chunk boundaries");
    Check(GuardsUntouched(result), "identity capture order leaves output guards untouched");

    auto defaultOrder = Run({128,64}, capture, capture, offsets, capture.size() * 2);
    Check(defaultOrder.status == 1 &&
          std::all_of(defaultOrder.spans.begin(), defaultOrder.spans.end(), Sentinel),
          "unsorted IDs remain rejected by the default sorted-subset mode");

    std::vector<uint64_t> mismatch = capture;
    mismatch[17] = mismatch[17] + 1;
    result = Run({128,64}, capture, mismatch, offsets, capture.size() * 2,
                  false, {}, {}, CurveTileOrder::IdentityCaptureOrder);
    Check(result.enqueueStatus == cudaSuccess && result.status == 1 &&
          std::all_of(result.spans.begin(), result.spans.end(), Sentinel),
          "identity capture order rejects an ID mismatch without publishing spans");

    result = Run({128,64}, capture, std::vector<uint64_t>(capture.begin(), capture.end() - 1),
                  std::vector<uint32_t>(offsets.begin(), offsets.end() - 1),
                  capture.size() * 2 - 2, false, {}, {},
                  CurveTileOrder::IdentityCaptureOrder);
    Check(result.enqueueStatus == cudaErrorInvalidValue &&
          std::all_of(result.spans.begin(), result.spans.end(), Sentinel),
          "identity capture order rejects a survivor-count mismatch at enqueue");

    auto invalidOrder = Run({128,64}, capture, capture, offsets, capture.size() * 2,
                             false, {}, {}, static_cast<CurveTileOrder>(99));
    Check(invalidOrder.enqueueStatus == cudaErrorInvalidValue &&
          std::all_of(invalidOrder.spans.begin(), invalidOrder.spans.end(), Sentinel),
          "invalid tile order is rejected at enqueue");

    // Identity mode intentionally does not sort/hash the capture IDs to prove
    // uniqueness.  The authoritative Scatter capture contract must prevalidate
    // uniqueness before opting into this order-preserving mode.
}

void PartitionEdges() {
    auto exactPartition = [](Result const& result,
                             std::vector<uint32_t> const& curveCounts,
                             char const* label) {
        bool exact = result.status == 0 &&
            result.requirements.tileCount == curveCounts.size() &&
            GuardsUntouched(result);
        uint32_t first = 0;
        for (size_t tile = 0; tile < curveCounts.size(); ++tile) {
            CurveTileSpan const expected{static_cast<uint32_t>(tile), first,
                curveCounts[tile], first, curveCounts[tile]};
            exact = exact && Same(result.spans[tile], expected);
            first += curveCounts[tile];
        }
        Check(exact, label);
    };
    std::vector<uint64_t> capture(33 * 128), survivors(33 * 128);
    std::vector<uint32_t> offsets(capture.size() + 1);
    for (size_t i = 0; i < capture.size(); ++i) {
        capture[i] = survivors[i] = i;
        offsets[i] = static_cast<uint32_t>(i);
    }
    offsets.back() = static_cast<uint32_t>(capture.size());
    auto result = Run({128,32}, capture, survivors, offsets, capture.size());
    std::vector<uint32_t> counts33(32, 128);
    counts33.front() = 256;
    exactPartition(result, counts33,
                   "33 chunks reserve exact 2-plus-31 capture-chunk tile boundaries");

    capture.resize(65 * 128); survivors.resize(capture.size());
    offsets.resize(capture.size() + 1);
    for (size_t i = 0; i < capture.size(); ++i) {
        capture[i] = survivors[i] = i;
        offsets[i] = static_cast<uint32_t>(i);
    }
    offsets.back() = static_cast<uint32_t>(capture.size());
    result = Run({128,32}, capture, survivors, offsets, capture.size());
    std::vector<uint32_t> counts65;
    counts65.insert(counts65.end(), 16, 3 * 128);
    counts65.push_back(2 * 128);
    counts65.insert(counts65.end(), 15, 128);
    exactPartition(result, counts65,
                   "65 chunks reserve exact 16x3-plus-2-plus-15 capture-chunk boundaries");

    std::vector<uint64_t> ids100k(100000);
    for (uint64_t i = 0; i < ids100k.size(); ++i) ids100k[i] = i;
    CurveTileRequirements requirements;
    Cuda(GetCurveTileRequirements({}, ids100k.size(), 0, 0, &requirements),
         "100k tile requirements");
    Check(requirements.chunkCount == 196 && requirements.chunksPerTile == 4 &&
          requirements.tileCount == 49, "100k default partition is 49 tiles");
    Cuda(GetCurveTileRequirements({512,32}, ids100k.size(), 0, 0, &requirements),
         "tile target edit requirements");
    Check(requirements.tileCount == 32, "tileTarget edit changes frozen tile layout");
    Cuda(GetCurveTileRequirements({1,999}, 300, 0, 0, &requirements),
         "clamped tile requirements");
    Check(requirements.chunkSize == 128 && requirements.tileTarget == 256,
          "chunk and tile target clamp to contract bounds");
    Cuda(GetCurveTileRequirements({0,0}, 300, 0, 0, &requirements),
         "defaulted tile requirements");
    Check(requirements.chunkSize == 512 && requirements.tileTarget == 64,
          "zero configuration selects core defaults");
}

void CaptureOrderSurvivors() {
    std::vector<uint64_t> capture(300);
    for(size_t i=0;i<capture.size();++i) capture[i]=1000+(i*137)%300;
    auto const mode=CurveTileOrder::CaptureOrderSurvivorSubset;
    std::vector<uint64_t> survivors{capture[0],capture[127],capture[128],capture[255],capture[299]};
    auto result=Run({128,64},capture,survivors,{0,2,5,9,14,20},20,false,{}, {},mode);
    std::vector<CurveTileSpan> expected{{0,0,2,0,5},{1,2,2,5,9},{2,4,1,14,6}};
    Check(result.status==0 && result.requirements.tileCount==3 &&
          std::equal(expected.begin(),expected.end(),result.spans.begin(),Same) && GuardsUntouched(result),
          "unsorted capture survivors preserve original tile membership");
    result=Run({128,64},capture,survivors,{0,2,5,9,14,20},20,true,
        {capture[128],capture[129],capture[130],capture[131],capture[132]}, {0,2,5,9,14,20},mode);
    expected={{0,0,0,0,0},{1,0,5,0,20},{2,5,0,20,0}};
    Check(result.status==0 && std::equal(expected.begin(),expected.end(),result.spans.begin(),Same) && GuardsUntouched(result),
          "capture-order graph replay preserves empty frozen tiles");
    result=Run({128,64},capture,{}, {0},0,false,{}, {},mode);
    Check(result.status==0 && result.requirements.tileCount==3 && GuardsUntouched(result),"all capture-order survivors culled");
    for(size_t i=0;i<3;++i) Check(Same(result.spans[i],{uint32_t(i),0,0,0,0}),"all-culled tile span is empty");
    for(int variant=0;variant<4;++variant) {
        auto invalid=survivors;
        if(variant==0) invalid[1]=99999;
        if(variant==1) invalid[1]=invalid[0];
        if(variant==2) std::swap(invalid[0],invalid[1]);
        result=Run({128,64},capture,invalid,{0,2,5,9,14,20},20,false,{}, {},mode,variant==3);
        Check(result.status==1 && std::all_of(result.spans.begin(),result.spans.end(),Sentinel),
              "invalid capture-order survivor or lookup leaves spans untouched");
    }
    result=Run({}, {}, {}, {0},0,false,{}, {},mode);
    Check(result.status==0 && result.requirements.tileCount==0 && GuardsUntouched(result),"empty capture-order lookup");
}

void InvalidInput() {
    std::vector<uint64_t> capture{10,20,30};
    auto invalid = [&](std::vector<uint64_t> survivors, std::vector<uint32_t> offsets,
                       size_t points, char const* label) {
        Result result = Run({128,64}, capture, survivors, offsets, points);
        Check(result.status == 1 && std::all_of(result.spans.begin(), result.spans.end(), Sentinel),
              label);
    };
    invalid({10,25}, {0,1,2}, 2, "unknown survivor ID fails closed");
    invalid({10,10}, {0,1,2}, 2, "duplicate survivor IDs fail closed");
    invalid({20,10}, {0,1,2}, 2, "reordered survivor IDs fail closed");
    invalid({10,20}, {1,2,3}, 3, "nonzero first offset fails closed");
    invalid({10,20}, {0,2,2}, 3, "bad terminal offset fails closed");
    invalid({10,20,30}, {0,2,1,3}, 3, "nonmonotonic interior offset fails closed");
    Result duplicateCapture = Run({128,64}, {10,10,30}, {10}, {0,1}, 1);
    Check(duplicateCapture.status == 1 && std::all_of(duplicateCapture.spans.begin(),
          duplicateCapture.spans.end(), Sentinel), "duplicate capture IDs fail closed");
    CurveTileRequirements requirements;
    Check(GetCurveTileRequirements({}, size_t(UINT32_MAX) + 1, 0, 0, &requirements)
              == cudaErrorInvalidValue, "overflowing scalar capture count rejected");
}
} // namespace

int main() {
    GoldenRanges();
    IdentityCaptureOrder();
    CaptureOrderSurvivors();
    PartitionEdges();
    InvalidInput();
    std::printf("GPU stable tile spans: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
