#include "usdGenCurves/cuda/cudaSourceInput.h"

#include <array>
#include <cstdio>
#include <limits>

using namespace usdGen;

int main() {
    int failures = 0;
    auto check = [&](bool ok, char const* msg) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
    };
    CudaSourcePreparationInput source;
    source.curveVertexCounts = {2, 3};
    source.points = {{10,0,0},{11,0,0},{20,0,0},{21,0,0},{22,0,0}};
    source.rest = source.points;
    source.widths = {1,2,3,4,5};
    source.hairT = {0,1,0,.5f,1};
    source.curveId = {9, 3};
    source.rootPrim = {90,30};
    source.rootUV = {{.9f,.1f},{.3f,.7f}};
    std::array<double, 16> frame9{};
    std::array<double, 16> frame3{};
    frame9[0] = frame9[5] = frame9[10] = frame9[15] = 1.0;
    frame3[0] = frame3[5] = frame3[10] = frame3[15] = 1.0;
    frame9[12] = 90.0;
    frame3[12] = 30.0;
    source.rootFrames = {frame9, frame3};
    CudaSourcePrepared prepared;
    std::vector<std::string> diagnostics;
    auto rootFrameOption = CudaSourcePreparationOptions{};
    rootFrameOption.hasRootFrame = true;
    check(PrepareCudaSource(source, rootFrameOption, &prepared, &diagnostics) == CudaSourcePreparationStatus::Ok,
          "prepare and sort source");
    check(prepared.curveId == std::vector<uint64_t>({3,9}) &&
          prepared.curveVertexCounts == std::vector<int32_t>({3,2}), "sort stable IDs and counts");
    check(prepared.points[0].x == 20 && prepared.points[2].x == 22 && prepared.points[3].x == 10,
          "reorder all CV positions");
    check(prepared.widths[0] == 3 && prepared.widths[3] == 1 && prepared.hairT[1] == .5f,
          "reorder CV channels");
    check(prepared.rootPrim == std::vector<int32_t>({30,90}) && prepared.rootUV[0].x == .3f &&
          prepared.rootFrames.size() == 2 && prepared.rootFrames[0][12] == 30.0 &&
          prepared.rootFrames[1][12] == 90.0,
          "reorder root channels");
    auto descriptor = prepared.Input();
    check(descriptor.points.data == prepared.points.data() && descriptor.stableIds.data == prepared.curveId.data(),
          "compute descriptor spans from current ownership");

    auto compacted = prepared;
    check(CompactCudaSource(&compacted, {9}) &&
          compacted.curveId == std::vector<uint64_t>({9}) &&
          compacted.curveVertexCounts == std::vector<int32_t>({2}) &&
          compacted.points.size() == 2 && compacted.points[0].x == 10 &&
          compacted.rest[1].x == 11 && compacted.widths[1] == 2 &&
          compacted.hairT[1] == 1 && compacted.rootPrim[0] == 90 &&
          compacted.rootUV[0].x == .9f && compacted.rootFrames[0][12] == 90,
          "compact every ragged source channel by stable ID");
    check(prepared.points.size() == 5 && prepared.curveId.size() == 2 &&
          prepared.points[0].x == 20,
          "compaction preserves retained prior source owner");
    auto const* owner = compacted.points.data();
    check(!CompactCudaSource(&compacted, {999}) &&
          !CompactCudaSource(&compacted, {9,9}) && compacted.points.data() == owner &&
          compacted.curveId == std::vector<uint64_t>({9}),
          "invalid retained IDs preserve candidate ownership");
    auto reorderedRequest = prepared;
    check(CompactCudaSource(&reorderedRequest, {9,3}) &&
          reorderedRequest.curveId == std::vector<uint64_t>({3,9}),
          "compaction retains canonical source order");
    check(CompactCudaSource(&compacted, {}) && compacted.points.empty() &&
          compacted.rest.empty() && compacted.widths.empty() &&
          compacted.hairT.empty() && compacted.curveId.empty() &&
          compacted.rootPrim.empty() && compacted.rootUV.empty() &&
          compacted.rootFrames.empty(), "all dropped source clears every channel");
    auto malformed = prepared;
    malformed.rest.pop_back();
    check(!CompactCudaSource(&malformed, {9}) && malformed.rest.size() == 4,
          "malformed source compaction fails transactionally");
    auto absent = prepared;
    absent.rest.clear(); absent.hairT.clear(); absent.rootPrim.clear();
    absent.rootUV.clear(); absent.rootFrames.clear();
    check(CompactCudaSource(&absent, {9}) && absent.rest.empty() &&
          absent.hairT.empty() && absent.rootPrim.empty() && absent.rootFrames.empty(),
          "compaction preserves absent optional channels");

    // useRest declares the processing space; it never substitutes the loaded
    // posed points. The authored rest channel remains separate in both modes.
    CudaSourcePreparationInput distinct;
    distinct.curveVertexCounts = {2};
    distinct.points = {{2.2f,0,0},{2.3f,0,0}};
    distinct.rest = {{.2f,0,0},{.3f,0,0}};
    distinct.widths = {.1f,.1f};
    CudaSourcePrepared restSelected, posedSelected;
    CudaSourcePreparationOptions selectRest;
    selectRest.useRest = true;
    check(PrepareCudaSource(distinct, selectRest, &restSelected) == CudaSourcePreparationStatus::Ok,
          "prepare distinct rest and posed source");
    auto restDescriptor = restSelected.Input();
    check(restDescriptor.points.size == 2 && restDescriptor.points.data == restSelected.points.data() &&
          restDescriptor.points.data[0].x == 2.2f &&
          restDescriptor.restPoints.data == restSelected.rest.data() &&
          restDescriptor.restPoints.data[1].x == .3f,
          "useRest true preserves loaded points and separate rest channel");
    selectRest.useRest = false;
    check(PrepareCudaSource(distinct, selectRest, &posedSelected) == CudaSourcePreparationStatus::Ok,
          "prepare posed source selection");
    auto posedDescriptor = posedSelected.Input();
    check(posedDescriptor.points.size == 2 && posedDescriptor.points.data == posedSelected.points.data() &&
          posedDescriptor.points.data[0].x == 2.2f &&
          posedDescriptor.restPoints.data == posedSelected.rest.data() &&
          posedDescriptor.restPoints.data[0].x == .2f,
          "useRest false preserves loaded points without replacing canonical rest");
    auto noRest = distinct;
    noRest.rest.clear();
    CudaSourcePrepared noRestPrepared;
    selectRest.useRest = true;
    check(PrepareCudaSource(noRest, selectRest, &noRestPrepared) == CudaSourcePreparationStatus::Ok,
          "prepare standalone source without rest channel");
    auto noRestDescriptor = noRestPrepared.Input();
    check(noRestDescriptor.points.data == noRestPrepared.points.data() &&
          noRestDescriptor.points.data[0].x == 2.2f &&
          noRestDescriptor.restPoints.size == 0,
          "missing rest keeps canonical-empty fallback without point substitution");
    CudaSourcePrepared lastGood = prepared;
    auto badWidth = source; badWidth.widths[2] = -1.0f;
    check(PrepareCudaSource(badWidth, {}, &prepared, nullptr) == CudaSourcePreparationStatus::InvalidArgument,
          "negative width rejects");
    bool samePoints = prepared.points.size() == lastGood.points.size();
    for (size_t i = 0; samePoints && i < prepared.points.size(); ++i)
        samePoints = prepared.points[i].x == lastGood.points[i].x &&
                     prepared.points[i].y == lastGood.points[i].y &&
                     prepared.points[i].z == lastGood.points[i].z;
    check(prepared.curveId == lastGood.curveId && samePoints,
          "invalid input preserves last good output");
    auto badFrames = source;
    badFrames.rootFrames[0][12] = std::numeric_limits<double>::quiet_NaN();
    check(PrepareCudaSource(badFrames, {}, &prepared, nullptr) == CudaSourcePreparationStatus::NonFiniteInput &&
          prepared.rootFrames == lastGood.rootFrames,
          "non-finite rootFrame rejects and retains last good output");
    auto needsFrames = CudaSourcePreparationOptions{};
    needsFrames.hasRootFrame = true;
    auto missingFrames = source;
    missingFrames.rootFrames.clear();
    check(PrepareCudaSource(missingFrames, needsFrames, &prepared, nullptr) == CudaSourcePreparationStatus::InvalidArgument,
          "rootFrame flag cannot silently accept a missing payload");
    auto shortFrames = source;
    shortFrames.rootFrames.pop_back();
    check(PrepareCudaSource(shortFrames, {}, &prepared, nullptr) == CudaSourcePreparationStatus::InvalidArgument,
          "rootFrame payload must be uniform per curve");
    auto badEnum = CudaSourcePreparationOptions{};
    badEnum.staleAction = static_cast<CudaSourceStaleAction>(99);
    check(PrepareCudaSource(source, badEnum, &prepared, nullptr) == CudaSourcePreparationStatus::InvalidArgument,
          "invalid stale action rejects");
    CudaSourcePreparationInput empty;
    CudaSourcePrepared emptyPrepared;
    check(PrepareCudaSource(empty, {}, &emptyPrepared, nullptr) == CudaSourcePreparationStatus::Ok &&
          emptyPrepared.Input().points.size == 0, "empty C3 source is valid");

    CudaSourcePreparationInput missing = source;
    missing.curveId.clear(); missing.widths.clear();
    CudaSourcePrepared fallback;
    auto status = PrepareCudaSource(missing, {0.25f}, &fallback, &diagnostics);
    check(status == CudaSourcePreparationStatus::Ok && fallback.curveId[0] == 0 &&
          fallback.curveId[1] == 1 && fallback.widths.size() == 5 && fallback.widths[4] == .25f,
          "synthesize IDs and default widths");

    auto stale = CudaSourcePreparationOptions{}; stale.expectedEpoch = "new"; stale.actualEpoch = "old";
    auto constant = source;
    constant.widths = {.75f};
    CudaSourcePrepared constantPrepared;
    check(PrepareCudaSource(constant, {}, &constantPrepared) == CudaSourcePreparationStatus::Ok &&
          constantPrepared.widths == std::vector<float>(5,.75f), "constant widths expand to every CV");
    auto copied = constantPrepared;
    check(copied.Input().points.data == copied.points.data() &&
          copied.Input().points.data != constantPrepared.Input().points.data,
          "copied preparation recomputes borrowed spans from its own storage");
    stale.staleAction = CudaSourceStaleAction::Warn;
    check(PrepareCudaSource(source, stale, &prepared, nullptr) == CudaSourcePreparationStatus::Ok,
          "stale warn evaluates");
    stale.staleAction = CudaSourceStaleAction::Ignore;
    check(PrepareCudaSource(source, stale, &prepared, nullptr) == CudaSourcePreparationStatus::Ok,
          "stale ignore evaluates");
    stale.staleAction = CudaSourceStaleAction::Block;
    check(PrepareCudaSource(source, stale, &prepared, nullptr) == CudaSourcePreparationStatus::StaleEpoch,
          "stale block rejects");

    auto unsupported = CudaSourcePreparationOptions{}; unsupported.resampleTo = 4;
    check(PrepareCudaSource(source, unsupported, &prepared, nullptr) == CudaSourcePreparationStatus::Ok &&
          prepared.curveVertexCounts == std::vector<int32_t>({3,2}) && prepared.points.size() == 5,
          "resample target is accepted while preparation preserves authored ragged data");
    unsupported.resampleTo = 1;
    check(PrepareCudaSource(source, unsupported, &prepared, nullptr) == CudaSourcePreparationStatus::InvalidArgument,
          "one-CV resample target rejects");
    unsupported.resampleTo = -1;
    check(PrepareCudaSource(source, unsupported, &prepared, nullptr) == CudaSourcePreparationStatus::InvalidArgument,
          "negative resample target rejects");
    unsupported.resampleTo = 0; unsupported.rebind = "always";
    check(PrepareCudaSource(source, unsupported, &prepared, nullptr) == CudaSourcePreparationStatus::UnsupportedFeature,
          "rebind is explicit unsupported error");
    source.curveId = {3,3};
    check(PrepareCudaSource(source, {}, &prepared, nullptr) == CudaSourcePreparationStatus::DuplicateStableId,
          "duplicate IDs reject");
    return failures;
}
