#ifdef USDGEN_ENABLE_CUDA
#include "cudaSourceInput.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <unordered_set>

namespace usdGen {
namespace {
void Diag(std::vector<std::string>* d, std::string const& s) { if (d) d->push_back(s); }
template<class T> bool Finite(std::vector<T> const&) { return true; }
template<> bool Finite(std::vector<float> const& a) { for (float v : a) if (!std::isfinite(v)) return false; return true; }
template<> bool Finite(std::vector<float3> const& a) { for (auto const& v : a) if (!std::isfinite(v.x)||!std::isfinite(v.y)||!std::isfinite(v.z)) return false; return true; }
template<> bool Finite(std::vector<float2> const& a) { for (auto const& v : a) if (!std::isfinite(v.x)||!std::isfinite(v.y)) return false; return true; }
bool FiniteRootFrames(std::vector<std::array<double, 16>> const& frames) {
    for (auto const& frame : frames)
        for (double value : frame)
            if (!std::isfinite(value)) return false;
    return true;
}
}

gpu::CurveSourceInput CudaSourcePrepared::Input() const {
    gpu::CurveSourceInput input;
    input.curveVertexCounts = {curveVertexCounts.data(), curveVertexCounts.size()};
    input.points = {points.data(), points.size()}; input.restPoints = {rest.data(), rest.size()};
    input.widths = {widths.data(), widths.size()}; input.hairT = {hairT.data(), hairT.size()};
    input.stableIds = {curveId.data(), curveId.size()}; input.rootPrim = {rootPrim.data(), rootPrim.size()};
    input.rootUV = {rootUV.data(), rootUV.size()};
    return input;
}

CudaSourcePreparationStatus PrepareCudaSourceImpl(
    CudaSourcePreparationInput const& source, CudaSourcePreparationOptions const& options,
    CudaSourcePrepared* out, std::vector<std::string>* diagnostics) {
    if (!out || options.rebind != "never") {
        Diag(diagnostics, "rebind is unsupported by CUDA source preparation");
        return CudaSourcePreparationStatus::UnsupportedFeature;
    }
    if (options.hasRootFrame && source.rootFrames.empty()) {
        Diag(diagnostics, "rootFrame flag requires a per-curve matrix payload");
        return CudaSourcePreparationStatus::InvalidArgument;
    }
    // This stage preserves authored ragged data. CUDA execution resolves the
    // target before upload and resamples device-to-device after source staging.
    if (options.resampleTo < 0 || options.resampleTo == 1) {
        Diag(diagnostics, "resampleTo must be zero or at least two");
        return CudaSourcePreparationStatus::InvalidArgument;
    }
    if (options.staleAction != CudaSourceStaleAction::Warn &&
        options.staleAction != CudaSourceStaleAction::Ignore &&
        options.staleAction != CudaSourceStaleAction::Block)
        return CudaSourcePreparationStatus::InvalidArgument;
    if (options.expectedEpoch.size() && options.expectedEpoch != options.actualEpoch) {
        if (options.staleAction == CudaSourceStaleAction::Block) {
            Diag(diagnostics, "source epoch does not match expected epoch");
            return CudaSourcePreparationStatus::StaleEpoch;
        }
        if (options.staleAction == CudaSourceStaleAction::Warn)
            Diag(diagnostics, "source epoch is stale; evaluating by request");
    }
    if (options.idSource != CudaSourceIdSource::Primvar && options.idSource != CudaSourceIdSource::Index)
        return CudaSourcePreparationStatus::InvalidArgument;
    if (!std::isfinite(options.defaultWidth) || options.defaultWidth < 0)
        return CudaSourcePreparationStatus::InvalidArgument;
    if (source.curveVertexCounts.empty()) {
        if (!source.points.empty() || !source.rest.empty() || !source.widths.empty() ||
            !source.hairT.empty() || !source.curveId.empty() || !source.rootPrim.empty() || !source.rootUV.empty() ||
            !source.rootFrames.empty())
            return CudaSourcePreparationStatus::InvalidArgument;
        out->curveVertexCounts.clear(); out->points.clear(); out->rest.clear();
        out->widths.clear(); out->hairT.clear(); out->curveId.clear();
        out->rootPrim.clear(); out->rootUV.clear(); out->rootFrames.clear(); out->useRest = options.useRest;
        out->warningFlags = gpu::CurveSourceWarningNone;
        return CudaSourcePreparationStatus::Ok;
    }
    if (source.points.empty() || source.points.size() > std::numeric_limits<uint32_t>::max())
        return CudaSourcePreparationStatus::InvalidTopology;
    size_t total = 0;
    for (int32_t n : source.curveVertexCounts) {
        if (n < 2 || total > std::numeric_limits<uint32_t>::max() - size_t(n))
            return CudaSourcePreparationStatus::InvalidTopology;
        total += size_t(n);
    }
    if (source.points.size() != total ||
        (!source.rest.empty() && source.rest.size() != total) ||
        (!source.hairT.empty() && source.hairT.size() != total) ||
        (!source.widths.empty() && source.widths.size() != total && source.widths.size() != 1) ||
        (!source.curveId.empty() && source.curveId.size() != source.curveVertexCounts.size()) ||
        (!source.rootPrim.empty() && source.rootPrim.size() != source.curveVertexCounts.size()) ||
        (!source.rootUV.empty() && source.rootUV.size() != source.curveVertexCounts.size()) ||
        (!source.rootFrames.empty() && source.rootFrames.size() != source.curveVertexCounts.size()))
        return CudaSourcePreparationStatus::InvalidArgument;
    if (!Finite(source.points) || !Finite(source.rest) || !Finite(source.widths) || !Finite(source.hairT) || !Finite(source.rootUV) ||
        !FiniteRootFrames(source.rootFrames))
        return CudaSourcePreparationStatus::NonFiniteInput;
    for (float width : source.widths)
        if (width < 0.0f) return CudaSourcePreparationStatus::InvalidArgument;
    for (float t : source.hairT) if (t < 0 || t > 1) return CudaSourcePreparationStatus::InvalidArgument;
    if (!source.hairT.empty()) {
        size_t base = 0;
        for (int32_t n : source.curveVertexCounts) {
            if (source.hairT[base] != 0.0f || source.hairT[base + n - 1] != 1.0f)
                return CudaSourcePreparationStatus::InvalidArgument;
            for (int32_t j = 1; j < n; ++j)
                if (source.hairT[base + j] < source.hairT[base + j - 1])
                    return CudaSourcePreparationStatus::InvalidArgument;
            base += size_t(n);
        }
    }

    std::vector<uint64_t> ids = source.curveId;
    uint32_t warnings = gpu::CurveSourceWarningNone;
    if (options.idSource == CudaSourceIdSource::Index || ids.empty()) {
        ids.resize(source.curveVertexCounts.size());
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = i;
        warnings |= gpu::CurveSourceWarningSynthesizedStableIds;
        Diag(diagnostics, "stable curve ids synthesized from source indices");
    }
    std::unordered_set<uint64_t> unique;
    for (uint64_t id : ids) if (!unique.insert(id).second) { Diag(diagnostics, "duplicate stable curve id"); return CudaSourcePreparationStatus::DuplicateStableId; }
    std::vector<size_t> order(ids.size()); for (size_t i=0;i<order.size();++i) order[i]=i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a,size_t b){return ids[a] < ids[b];});
    std::vector<size_t> sourceBase(source.curveVertexCounts.size());
    size_t sourceOffset = 0;
    for (size_t i = 0; i < sourceBase.size(); ++i) {
        sourceBase[i] = sourceOffset;
        sourceOffset += size_t(source.curveVertexCounts[i]);
    }
    CudaSourcePrepared candidate;
    candidate.curveVertexCounts.clear(); candidate.points.clear(); candidate.rest.clear(); candidate.widths.clear(); candidate.hairT.clear(); candidate.curveId.clear(); candidate.rootPrim.clear(); candidate.rootUV.clear(); candidate.rootFrames.clear();
    for (size_t old : order) {
        int32_t n = source.curveVertexCounts[old]; candidate.curveVertexCounts.push_back(n); candidate.curveId.push_back(ids[old]);
        if (!source.rootPrim.empty()) candidate.rootPrim.push_back(source.rootPrim[old]);
        if (!source.rootUV.empty()) candidate.rootUV.push_back(source.rootUV[old]);
        if (!source.rootFrames.empty()) candidate.rootFrames.push_back(source.rootFrames[old]);
        const size_t pointBase = sourceBase[old];
        for (int32_t j=0;j<n;++j) { candidate.points.push_back(source.points[pointBase+j]); if (!source.rest.empty()) candidate.rest.push_back(source.rest[pointBase+j]); if (!source.hairT.empty()) candidate.hairT.push_back(source.hairT[pointBase+j]); if (source.widths.size()==total) candidate.widths.push_back(source.widths[pointBase+j]); }
    }
    if (source.widths.empty()) { candidate.widths.assign(candidate.points.size(), options.defaultWidth); warnings |= gpu::CurveSourceWarningSynthesizedWidths; Diag(diagnostics, "widths synthesized from default width"); }
    else if (source.widths.size()==1) candidate.widths.assign(candidate.points.size(), source.widths[0]);
    if (source.rest.empty() && options.useRest) warnings |= gpu::CurveSourceWarningSynthesizedRest;
    if (source.hairT.empty()) warnings |= gpu::CurveSourceWarningSynthesizedHairT;
    candidate.useRest = options.useRest; candidate.warningFlags = warnings;
    *out = std::move(candidate);
    return CudaSourcePreparationStatus::Ok;
}

bool CompactCudaSource(CudaSourcePrepared* prepared,
                       std::vector<uint64_t> const& retainedStableIds) {
    if (!prepared) return false;
    try {
        auto const& source = *prepared;
        size_t const curves = source.curveVertexCounts.size();
        if (source.curveId.size() != curves) return false;
        size_t points = 0;
        for (int32_t count : source.curveVertexCounts) {
            if (count < 2 || size_t(count) > std::numeric_limits<size_t>::max() - points)
                return false;
            points += size_t(count);
        }
        auto optionalSize = [](auto const& values, size_t size) {
            return values.empty() || values.size() == size;
        };
        if (source.points.size() != points ||
            !optionalSize(source.rest, points) ||
            !optionalSize(source.widths, points) ||
            !optionalSize(source.hairT, points) ||
            !optionalSize(source.rootPrim, curves) ||
            !optionalSize(source.rootUV, curves) ||
            !optionalSize(source.rootFrames, curves)) return false;
        std::unordered_set<uint64_t> retain(retainedStableIds.begin(), retainedStableIds.end());
        if (retain.size() != retainedStableIds.size()) return false;
        std::unordered_set<uint64_t> unique;
        CudaSourcePrepared candidate;
        candidate.useRest = source.useRest;
        candidate.warningFlags = source.warningFlags;
        auto append = [](auto const& from, auto& to, size_t first, size_t count) {
            if (!from.empty()) to.insert(to.end(), from.begin() + first,
                                        from.begin() + first + count);
        };
        size_t first = 0;
        for (size_t c = 0; c != curves; ++c) {
            if (!unique.insert(source.curveId[c]).second) return false;
            size_t const count = size_t(source.curveVertexCounts[c]);
            if (retain.erase(source.curveId[c])) {
                candidate.curveVertexCounts.push_back(source.curveVertexCounts[c]);
                candidate.curveId.push_back(source.curveId[c]);
                append(source.points, candidate.points, first, count);
                append(source.rest, candidate.rest, first, count);
                append(source.widths, candidate.widths, first, count);
                append(source.hairT, candidate.hairT, first, count);
                append(source.rootPrim, candidate.rootPrim, c, 1);
                append(source.rootUV, candidate.rootUV, c, 1);
                append(source.rootFrames, candidate.rootFrames, c, 1);
            }
            first += count;
        }
        if (!retain.empty()) return false;
        *prepared = std::move(candidate);
        return true;
    } catch (...) { return false; }
}

CudaSourcePreparationStatus PrepareCudaSource(
    CudaSourcePreparationInput const& source, CudaSourcePreparationOptions const& options,
    CudaSourcePrepared* out, std::vector<std::string>* diagnostics) {
    if (!out) { Diag(diagnostics, "null prepared output"); return CudaSourcePreparationStatus::InvalidArgument; }
    try {
        return PrepareCudaSourceImpl(source, options, out, diagnostics);
    } catch (std::bad_alloc const&) {
        Diag(diagnostics, "insufficient memory preparing CUDA source");
        return CudaSourcePreparationStatus::AllocationFailure;
    }
}
} // namespace usdGen
#endif
