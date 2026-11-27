// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// GPU-native index streams for the native Storm curve primitive layouts.
// No Hydra dependency: renderer adapters supply the basis/wrap as values.
#ifndef USDGEN_HYDRA_CURVES_STORM_INDICES_H
#define USDGEN_HYDRA_CURVES_STORM_INDICES_H

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace usdGen::hydraCurves {

// Borrowed device pointer + element count. Converts from any such view
// (e.g. usdGen::hydraCurves::DeviceView) so no backend header is needed here.
template<class T> struct IndexView {
    T* data = nullptr;
    size_t size = 0;
    IndexView() = default;
    IndexView(T* d, size_t n) : data(d), size(n) {}
    template<class U, class = std::enable_if_t<
        std::is_convertible<decltype(std::declval<U const&>().data), T*>::value>>
    IndexView(U const& v) : data(v.data), size(v.size) {}
};

enum class CurveIndexBasis { Linear, Bezier, BSpline, CatmullRom, CentripetalCatmullRom };
enum class CurveIndexWrap { Nonperiodic, Periodic, Pinned, Segmented };
enum class CurveIndexMode { Curves, Hull, Points };

struct CurveIndexOptions {
    CurveIndexBasis basis = CurveIndexBasis::BSpline;
    CurveIndexWrap wrap = CurveIndexWrap::Pinned;
    CurveIndexMode mode = CurveIndexMode::Curves;
};

struct CurveIndexRequirements {
    size_t maxRecords = 0;
    uint32_t indexArity = 0; // points: 1, lines/hull: 2, cubic patches: 4
    size_t scanBytes = 0;
    int deviceIndex = -1;
    cudaStream_t stream = nullptr;
};

// Setup boundary: derive allocation capacities from scalar shape metadata and
// query CUB scratch size. Does not read device offsets or allocate buffers.
// Indices and native Storm record counts must fit signed int32. Empty grooms
// are allowed; populated curves must have at least one control point.
// Segmented wrap is supported only for linear curves with even CV counts.
cudaError_t GetCurveIndexRequirements(CurveIndexOptions options,
    size_t curveCount, size_t pointCount, CurveIndexRequirements* result,
    cudaStream_t stream = nullptr);

struct CurveIndexWorkspace {
    IndexView<uint64_t> recordCounts;  // curveCount + 1 (last element zero)
    IndexView<uint64_t> recordOffsets; // curveCount + 1, exclusive scan
    IndexView<unsigned char> scan;
};

// A contiguous tile of a globally-addressed curve-offset buffer.  The
// offsets themselves remain global: the first value is pointBase and the
// terminal value is pointBase + pointCount.  Generated indices and primitive
// parameters are deliberately tile-local.
struct CurveIndexSpan {
    IndexView<const uint32_t> curveOffsets; // curveCount + 1 global offsets
    size_t curveCount = 0;
    size_t pointCount = 0;
    uint32_t pointBase = 0;
};

struct CurveIndexOutput {
    IndexView<int32_t> indices;        // packed records, maxRecords * arity
    IndexView<int32_t> primitiveParam; // maxRecords, owning curve per record
    IndexView<uint64_t> recordCount;   // one GPU scalar; no host readback
    IndexView<uint32_t> status;        // one GPU scalar: 0 valid, 1 bad offsets,
                                       // 2 invalid segmented count
};

// Enqueue-only primitive. All pointers are device-accessible, mutually
// nonaliasing borrowed allocations on the stream's current device. The caller
// retains them until stream completion, and owns scheduling between uses.
// No allocation, host wait, geometry/count readback or application lock.
// Requirements must be prepared before graph capture for this shape/options
// and stream. The prepared stream must stay alive through the last launch.
// Shape/enum/capacity errors return immediately without enqueuing. Device
// validation failures set status and recordCount=0, leaving output indices
// and primitiveParam untouched. Only [0, recordCount) records are defined.
// Points use arity 1; primitiveParam is also supplied for tools even though
// Storm's native points-index builder does not request that channel.
// There is no implicit host topology or indexed-USD-curve remapping path.
cudaError_t BuildCurveIndices(CurveIndexOptions options,
    size_t curveCount, size_t pointCount, IndexView<const uint32_t> curveOffsets,
    CurveIndexRequirements const& requirements, CurveIndexWorkspace workspace,
    CurveIndexOutput output, cudaStream_t stream);

// Span overload for a tile in a globally-addressed curve-offset buffer.  A
// scalar pointBase + pointCount overflow and malformed device offsets fail
// closed through status/recordCount, leaving indices and primitiveParam
// untouched.  The legacy overload above forwards pointBase=0.
cudaError_t BuildCurveIndices(CurveIndexOptions options, CurveIndexSpan span,
    CurveIndexRequirements const& requirements, CurveIndexWorkspace workspace,
    CurveIndexOutput output, cudaStream_t stream);

// Enqueue the native indirect-draw count corresponding to the generated
// records.  The result is zero when the upstream GPU validation status is
// nonzero, the record count exceeds maxRecords, or records*indexArity cannot
// be represented by uint32_t.  This performs no host readback or allocation
// and is safe to place in a CUDA graph after BuildCurveIndices.
cudaError_t PackCurveDrawCount(IndexView<const uint64_t> recordCount,
    IndexView<const uint32_t> status, uint32_t indexArity,
    size_t maxRecords, IndexView<uint32_t> drawCount,
    cudaStream_t stream);

} // namespace usdGen::hydraCurves
#endif