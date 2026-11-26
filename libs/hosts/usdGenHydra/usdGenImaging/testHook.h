// usdGen imaging — StormSurgery lock-free test hook (test-only surface).
//
// Exposes the atomic published-tile snapshot for adopted groom roots so
// testUsdGenStormSurgery can assert the P0 race contract from the render
// thread: every observation corresponds to one complete generation
// snapshot, never a torn map. Both entry points perform no owner wait: the
// owner publishes an immutable registry snapshot and each groom map is
// atomic_load'ed, exactly like GetPrim. No new public plugin API.
#ifndef USDGEN_IMAGING_TEST_HOOK_H
#define USDGEN_IMAGING_TEST_HOOK_H

#include "pxr/pxr.h"
#include "pxr/base/tf/declarePtrs.h"
#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/usd/sdf/path.h"

#include <cstddef>
#include <cstdint>
#include <memory>

PXR_NAMESPACE_OPEN_SCOPE

class HdSceneIndexBase;

/// Test-only probe over the live UsdGenGroomSceneIndex registry.
/// `groom` is the adopted UsdGenGroom root path (e.g. `/groomA`).
class UsdGenImagingTestHook
{
public:
    /// Max `generation` stamp across the groom's published tiles (0 when
    /// unknown / unpublished). Atomic snapshot read; safe at any moment,
    /// concurrently with _Republish.
    static int64_t publishedGeneration(SdfPath const &groom);
    static uint64_t issuedIngresses();

    /// Published tile count for the groom across live indices.
    static size_t publishedTileCount(SdfPath const &groom);

    /// Published strands of `groom` (a description or groom root) across
    /// live indices: curve count and summed control-polygon length over
    /// the synthetic tiles under <groom>/__usdGenRender, read through
    /// GetPrim like any Hydra consumer. false when no live index publishes
    /// a tile for it. The usdview brush T3s read it through the brush ABI.
    static bool publishedCurveStats(SdfPath const &groom, uint64_t *curves,
                                    double *totalLength);

    /// Number of registry entries in the published immutable snapshot. This
    /// does not promote weak handles; it is test-only observability.
    static size_t registeredIndexCount();

    /// Test-only bounded-owner-pressure seam.  It reserves/relinquishes one
    /// groom scene owner's command credit without queuing application work.
    static bool holdOneGroomOwnerCredit();
    static void releaseGroomOwnerCredits();
    static void setGroomCommandCapacityForTesting(uint64_t capacity);

    /// Owner-only test seam: completes already-admitted work for this exact
    /// index without polling, sending notices, or capturing input.
    static void drainGroomOwnersWithoutFrontend(HdSceneIndexBase const &index);
    static size_t pendingGroomPublicationCount(HdSceneIndexBase const &index);
    static std::weak_ptr<void const> pendingGroomPublicationSnapshotWeak(
        HdSceneIndexBase const &index);
    static std::weak_ptr<void const> pendingGroomPublicationTileMapWeak(
        HdSceneIndexBase const &index, SdfPath const &groom);
    static int64_t pendingGroomPublishedGeneration(
        HdSceneIndexBase const &index, SdfPath const &groom);
    static uint64_t groomCaptureCount(HdSceneIndexBase const &index);
    static uint64_t groomCookCount(HdSceneIndexBase const &index);
    /// Replay a prior-revision tile callback after a structural cook commits.
    static bool staleGroomProgressRejected(HdSceneIndexBase const &index,
                                           SdfPath const &groom);
    static void groomOwnerCommandBarrier(HdSceneIndexBase const &index);
    static uint64_t groomSequenceLastIssued(HdSceneIndexBase const &index);
    static uint64_t groomSequenceCompletedThrough(HdSceneIndexBase const &index);
    static uint64_t groomSequenceCapacity(HdSceneIndexBase const &index);
    static size_t groomEventHistoryCount(HdSceneIndexBase const &index);
    static size_t groomTombstoneHistoryCount(HdSceneIndexBase const &index);
    static void setGroomSequenceCapacityForTesting(uint64_t capacity);
    static size_t retainedGroomSceneStateCount();
    static size_t liveGroomSceneStateCount();
    static uint64_t groomRetirementRecordCount();
    static void groomSceneServiceCommandBarrier();
    static size_t groomUsedSessionWeakCount(HdSceneIndexBase const &index);
    static size_t groomUsedSessionLiveUniqueCount(HdSceneIndexBase const &index);
    static void armGroomFinalDeleterPauseForTesting();
    static void waitGroomFinalDeleterPauseForTesting();
    static void releaseGroomFinalDeleterPauseForTesting();
    static void armGroomDrainWaitForTesting();
    static void waitGroomDrainWaitForTesting();
    static void releaseGroomDrainWaitForTesting();

    /// External test/shutdown boundary only. Never invoke from graph work or
    /// a callback; waits until earlier hook-owner mutations have completed.
    static void Drain();

    // Registry wiring, called by the groom index ctor/dtor (NOT test API).
    static void _RegisterIndex(HdSceneIndexBase *index);
    static void _UnregisterIndex(HdSceneIndexBase *index);

private:
    static int64_t _PublishedGenerationOn(
        HdSceneIndexBase const &index, SdfPath const &groom);
    static size_t _PublishedTileCountOn(
        HdSceneIndexBase const &index, SdfPath const &groom);
};

PXR_NAMESPACE_CLOSE_SCOPE

#endif  // USDGEN_IMAGING_TEST_HOOK_H
