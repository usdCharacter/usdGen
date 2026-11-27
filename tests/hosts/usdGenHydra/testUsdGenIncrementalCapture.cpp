// Incremental capture regression: unrelated dirties must not rescan every
// adopted groom or rebuild unrelated session state.
#include "utils/pxrInterop.h"
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGenImaging/testHook.h"

#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/retainedSceneIndex.h"
#include "pxr/imaging/hd/systemMessages.h"
#include "pxr/base/vt/array.h"

#include <atomic>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <memory>
#include <string>
#include <thread>
#include <tbb/flow_graph.h>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

class CountingInput final : public HdSceneIndexBase, public HdSceneIndexObserver {
public:
    HdRetainedSceneIndexRefPtr retained = HdRetainedSceneIndex::New();
    std::atomic<unsigned> rootPrimReads{0}, rootTraversal{0};
    std::atomic<unsigned> groomAReads{0}, groomBReads{0};
    std::atomic<unsigned> opAReads{0};
    std::atomic<bool> throwNextA{false};
    std::atomic<bool> holdNextA{false};
    tbb::flow::graph captureEntered, captureRelease;

    CountingInput() { retained->AddObserver(TfCreateWeakPtr(static_cast<HdSceneIndexObserver*>(this))); }
    HdSceneIndexPrim GetPrim(SdfPath const &path) const override {
        auto *self = const_cast<CountingInput *>(this);
        if (path == SdfPath::AbsoluteRootPath()) ++self->rootPrimReads;
        else if (path == SdfPath("/groomA")) {
            ++self->groomAReads;
            if (self->holdNextA.exchange(false, std::memory_order_acq_rel)) {
                self->captureEntered.release_wait();
                self->captureRelease.wait_for_all();
            }
            if (self->throwNextA.exchange(false, std::memory_order_acq_rel))
                throw std::runtime_error("test Groom A capture failure");
        }
        else if (path == SdfPath("/groomB")) ++self->groomBReads;
        else if (path == SdfPath("/groomA/op")) ++self->opAReads;
        return retained->GetPrim(path);
    }
    SdfPathVector GetChildPrimPaths(SdfPath const &path) const override {
        auto *self = const_cast<CountingInput *>(this);
        if (path == SdfPath::AbsoluteRootPath()) ++self->rootTraversal;
        return retained->GetChildPrimPaths(path);
    }
    void PrimsAdded(HdSceneIndexBase const &, AddedPrimEntries const &e) override
    { _SendPrimsAdded(e); }
    void PrimsRemoved(HdSceneIndexBase const &, RemovedPrimEntries const &e) override
    { _SendPrimsRemoved(e); }
    void PrimsDirtied(HdSceneIndexBase const &, DirtiedPrimEntries const &e) override
    { _SendPrimsDirtied(e); }
    void PrimsRenamed(HdSceneIndexBase const &, RenamedPrimEntries const &e) override
    { _SendPrimsRenamed(e); }
    void Reset() {
        rootPrimReads = 0; rootTraversal = 0;
        groomAReads = 0; groomBReads = 0;
        opAReads = 0;
    }
};

HdContainerDataSourceHandle Fields(
    TfToken const &name, HdDataSourceBaseHandle const &value)
{
    return HdRetainedContainerDataSource::New(name, value);
}

HdContainerDataSourceHandle UsdGenFields(
    std::initializer_list<std::pair<TfToken, HdDataSourceBaseHandle>> fields)
{
    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;
    for (auto const &field : fields) {
        names.push_back(field.first);
        values.push_back(field.second);
    }
    return HdRetainedContainerDataSource::New(
        names.size(), names.data(), values.data());
}

int failures = 0;
void Check(bool value, char const *what) {
    if (!value) { ++failures; std::printf("FAIL: %s\n", what); }
    else std::printf("ok:   %s\n", what);
}

class PathNoticeObserver final : public HdSceneIndexObserver {
public:
    explicit PathNoticeObserver(SdfPath path) : _path(std::move(path)) {}
    std::atomic<bool> added{false};
    std::atomic<bool> removed{false};
    void PrimsAdded(HdSceneIndexBase const &, AddedPrimEntries const &entries) override {
        for (auto const &entry : entries)
            if (entry.primPath == _path) added.store(true, std::memory_order_release);
    }
    void PrimsRemoved(HdSceneIndexBase const &, RemovedPrimEntries const &entries) override {
        for (auto const &entry : entries)
            if (entry.primPath == _path) removed.store(true, std::memory_order_release);
    }
    void PrimsDirtied(HdSceneIndexBase const &, DirtiedPrimEntries const &) override {}
    void PrimsRenamed(HdSceneIndexBase const &, RenamedPrimEntries const &) override {}
private:
    SdfPath _path;
};

} // namespace

int main()
{
    // Keep progress visible when CTest kills a stalled run: a pipe makes stdout
    // fully buffered, so a timeout would otherwise discard every result line.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    auto input = TfCreateRefPtr(new CountingInput);
    const SdfPath a("/groomA"), b("/groomB");
    const SdfPath opA("/groomA/op"), surfaceSubset("/externalMesh/subset");
    const SdfPath externalCurves("/externalCurves"), newCurves("/newCurves");
    // usdGen:surface is a description-level binding; operators have no
    // per-node override, so the external mesh dependency is authored here.
    auto groomAData = UsdGenFields({
        {TfToken("operatorOrder"),
         HdRetainedTypedSampledDataSource<SdfPathVector>::New({opA})},
        {TfToken("surface"),
         HdRetainedTypedSampledDataSource<SdfPathVector>::New({surfaceSubset})}});
    auto opData = UsdGenFields({
        {TfToken("curves"),
         HdRetainedTypedSampledDataSource<SdfPathVector>::New({externalCurves})}});
    input->retained->AddPrims({
        {a, TfToken("UsdGenGroom"), groomAData},
        {b, TfToken("UsdGenGroom"), HdRetainedContainerDataSource::New()},
        {SdfPath("/unrelated"), TfToken("Scope"), HdRetainedContainerDataSource::New()},
        {SdfPath("/unrelated/child"), TfToken("Scope"), HdRetainedContainerDataSource::New()},
        {opA, TfToken("UsdGenCurveSource"), opData},
        {SdfPath("/externalMesh"), TfToken("mesh"), HdRetainedContainerDataSource::New()},
        {surfaceSubset, TfToken("geomSubset"),
         Fields(TfToken("geomSubset"), Fields(TfToken("indices"),
                HdRetainedTypedSampledDataSource<VtIntArray>::New(VtIntArray{0})))},
        {externalCurves, TfToken("basisCurves"), HdRetainedContainerDataSource::New()}});

    const int renderInstance = 7310;
    auto index = UsdGenGroomSceneIndex::New(input, renderInstance);
    auto *owner = dynamic_cast<UsdGenGroomSceneIndex *>(index.operator->());
    Check(owner != nullptr, "incremental capture owner cast succeeds");
    if (!owner) return 1;
    owner->Synchronize();
    index->SystemMessage(HdSystemMessageTokens->asyncAllow, nullptr);
    auto &store = usdGenImaging::UsdGenSessionStore::GetInstance();
    const usdGenImaging::UsdGenSessionKey keyA{"", a, renderInstance};
    const usdGenImaging::UsdGenSessionKey keyB{"", b, renderInstance};
    auto sessionA = store.Find(keyA), sessionB = store.Find(keyB);
    Check(sessionA && sessionB, "baseline adopts both groom roots");
    const int64_t generationA = sessionA ? sessionA->Generation() : -1;
    input->Reset();

    input->retained->DirtyPrims({
        {SdfPath("/unrelated"), HdDataSourceLocatorSet(HdDataSourceLocator())}});
    owner->Synchronize();
    Check(input->rootPrimReads == 0 && input->rootTraversal == 0 &&
              input->groomAReads == 0 && input->groomBReads == 0,
          "unrelated dirties do not rescan groom roots");
    Check(sessionA && sessionA->Generation() == generationA,
          "unrelated dirties preserve session generation");

    input->Reset();
    input->retained->DirtyPrims({
        {a, HdDataSourceLocatorSet(HdDataSourceLocator())}});
    owner->Synchronize();
    Check(input->groomAReads > 0 && input->groomBReads == 0 &&
              input->rootTraversal == 0,
          "groom-local dirty captures only that groom");
    Check(input->opAReads > 0, "groom ancestor dirty rereads operator values");

    input->Reset();
    input->retained->DirtyPrims({
        {SdfPath("/externalMesh"), HdDataSourceLocatorSet(HdDataSourceLocator())}});
    owner->Synchronize();
    Check(input->groomAReads > 0 && input->groomBReads == 0 &&
              input->rootTraversal == 0,
          "external mesh dependency captures only affected groom");
    Check(input->opAReads == 0, "external mesh edit reuses unchanged operator values");

    input->Reset();
    input->retained->DirtyPrims({
        {externalCurves, HdDataSourceLocatorSet(HdDataSourceLocator())}});
    owner->Synchronize();
    Check(input->groomAReads > 0 && input->groomBReads == 0 &&
              input->rootTraversal == 0,
          "external curve dependency captures only affected groom");
    Check(input->opAReads == 0, "external curve edit reuses unchanged operator values");

    input->Reset();
    input->retained->DirtyPrims({{opA, HdDataSourceLocatorSet(HdDataSourceLocator())}});
    owner->Synchronize();
    Check(input->opAReads > 0 && input->groomBReads == 0,
          "operator dirty after cached captures still rereads operator values");

    input->retained->AddPrims({{opA, TfToken("UsdGenCurveSource"),
                                UsdGenFields({
                                    {TfToken("surface"),
                                     HdRetainedTypedSampledDataSource<SdfPathVector>::New({surfaceSubset})},
                                    {TfToken("curves"),
                                     HdRetainedTypedSampledDataSource<SdfPathVector>::New({newCurves})}})}});
    owner->Synchronize();
    input->Reset();
    input->retained->DirtyPrims({
        {externalCurves, HdDataSourceLocatorSet(HdDataSourceLocator())}});
    owner->Synchronize();
    Check(input->groomAReads == 0 && input->groomBReads == 0,
          "retargeted old curve dependency is ignored");
    input->retained->AddPrims({{newCurves, TfToken("basisCurves"),
                                HdRetainedContainerDataSource::New()}});
    owner->Synchronize();
    input->Reset();
    input->retained->DirtyPrims({{newCurves, HdDataSourceLocatorSet(HdDataSourceLocator())}});
    owner->Synchronize();
    Check(input->groomAReads > 0 && input->groomBReads == 0,
          "retargeted new curve dependency reaches affected groom");

    // An earlier source query has reserved an ingress sequence but has not
    // submitted its capture yet. The newer dirty cannot trust that old
    // catalog, even though the catalog itself was previously complete.
    // Separate framework reply graphs hold only an external test caller;
    // there is no application mutex or polling/spin gate.
    input->captureEntered.reserve_wait();
    input->captureRelease.reserve_wait();
    input->holdNextA.store(true, std::memory_order_release);
    std::thread delayed([&] {
        input->retained->DirtyPrims({{a, HdDataSourceLocatorSet(HdDataSourceLocator())}});
    });
    input->captureEntered.wait_for_all();
    input->Reset();
    input->retained->DirtyPrims({
        {SdfPath("/unrelated"), HdDataSourceLocatorSet(HdDataSourceLocator())}});
    Check(input->rootTraversal > 0 && input->groomAReads > 0 && input->groomBReads > 0,
          "unresolved older capture forces newer dirty through full discovery");
    input->captureRelease.release_wait();
    delayed.join();
    owner->Synchronize();
    input->Reset();
    input->retained->DirtyPrims({
        {SdfPath("/unrelated"), HdDataSourceLocatorSet(HdDataSourceLocator())}});
    owner->Synchronize();
    Check(input->rootPrimReads == 0 && input->rootTraversal == 0 &&
              input->groomAReads == 0 && input->groomBReads == 0,
          "incremental filtering resumes after out-of-order captures complete");

    // A failed capture must still complete its ingress watermark and leave
    // the prior session identities intact.  Its deferred full discovery is
    // consumed at the next synchronization boundary, after which incremental
    // filtering can resume.
    input->throwNextA.store(true, std::memory_order_release);
    bool threw = false;
    try {
        input->retained->DirtyPrims({
            {a, HdDataSourceLocatorSet(HdDataSourceLocator())}});
    } catch (std::runtime_error const &) {
        threw = true;
    }
    Check(threw, "capture failure propagates from Groom A ingress");
    input->Reset();
    owner->Synchronize();
    Check(store.Find(keyA) == sessionA && store.Find(keyB) == sessionB &&
              input->rootTraversal > 0 && input->groomAReads > 0 &&
              input->groomBReads > 0,
          "synchronization recovers failed capture with full discovery and stable sessions");

    input->Reset();
    input->retained->DirtyPrims({
        {SdfPath("/unrelated"), HdDataSourceLocatorSet(HdDataSourceLocator())}});
    owner->Synchronize();
    Check(input->rootPrimReads == 0 && input->rootTraversal == 0 &&
              input->groomAReads == 0 && input->groomBReads == 0,
          "incremental filtering resumes after synchronization recovery");

    input->retained->RemovePrims({{a}});
    owner->Synchronize();
    Check(!store.Find(keyA) && store.Find(keyB),
          "removing groom A preserves groom B ownership");

    input->retained->RemovePrims({{b}});
    owner->Synchronize();
    index.Reset();
    UsdGenGroomSceneIndex::DrainRetired();
    UsdGenImagingTestHook::Drain();
    sessionA.Reset(); sessionB.Reset();

    // Bound the sequence window even when its oldest capture is stalled.
    // Later source mutations continue changing the input, but once the
    // small window is full they cannot allocate a sequence, run a capture,
    // or grow owner history.  Releasing the prefix gap recovers the final
    // namespace in one authoritative synchronization pass.
    UsdGenImagingTestHook::setGroomSequenceCapacityForTesting(8);
    {
        auto stalledInput = TfCreateRefPtr(new CountingInput);
        SdfPath const stalledGroom("/groomA");
        stalledInput->retained->AddPrims({
            {stalledGroom, TfToken("UsdGenGroom"), HdRetainedContainerDataSource::New()}});
        auto stalledIndex = UsdGenGroomSceneIndex::New(stalledInput, 7311);
        auto *stalledOwner = dynamic_cast<UsdGenGroomSceneIndex *>(stalledIndex.operator->());
        Check(stalledOwner != nullptr, "stalled-prefix fixture creates an owner");
        if (stalledOwner) {
            stalledOwner->Synchronize();
            stalledIndex->SystemMessage(HdSystemMessageTokens->asyncAllow, nullptr);
            SdfPath const finalPath("/stalled/final");
            PathNoticeObserver finalNotice(finalPath);
            stalledIndex->AddObserver(TfCreateWeakPtr(&finalNotice));

            stalledInput->captureEntered.reserve_wait();
            stalledInput->captureRelease.reserve_wait();
            stalledInput->holdNextA.store(true, std::memory_order_release);
            std::thread heldCapture([&] {
                stalledInput->retained->DirtyPrims(
                    {{stalledGroom, HdDataSourceLocatorSet(HdDataSourceLocator())}});
            });
            stalledInput->captureEntered.wait_for_all();
            uint64_t const issuedBefore =
                UsdGenImagingTestHook::groomSequenceLastIssued(*stalledIndex);

            constexpr unsigned kSequenceCapacity = 8;
            constexpr unsigned kLaterMutations = 24;
            for (unsigned i = 0; i != kLaterMutations; ++i) {
                SdfPath const transient("/stalled/transient" + std::to_string(i));
                stalledInput->retained->AddPrims(
                    {{transient, TfToken("Scope"), HdRetainedContainerDataSource::New()}});
                stalledInput->retained->RemovePrims({{transient}});
            }
            // This last mutation must be rejected while the prefix remains
            // held, but it is retained upstream for synchronization recovery.
            stalledInput->retained->AddPrims(
                {{finalPath, TfToken("Scope"), HdRetainedContainerDataSource::New()}});
            UsdGenImagingTestHook::groomOwnerCommandBarrier(*stalledIndex);
            uint64_t const issuedAtCapacity =
                UsdGenImagingTestHook::groomSequenceLastIssued(*stalledIndex);
            uint64_t const completedAtCapacity =
                UsdGenImagingTestHook::groomSequenceCompletedThrough(*stalledIndex);
            stalledInput->Reset();
            stalledInput->retained->DirtyPrims(
                {{SdfPath("/stalled/rejected"), HdDataSourceLocatorSet(HdDataSourceLocator())}});
            UsdGenImagingTestHook::groomOwnerCommandBarrier(*stalledIndex);
            Check(UsdGenImagingTestHook::groomSequenceCapacity(*stalledIndex) ==
                      kSequenceCapacity &&
                      issuedAtCapacity >= issuedBefore &&
                      issuedAtCapacity - completedAtCapacity <= kSequenceCapacity &&
                      UsdGenImagingTestHook::groomEventHistoryCount(*stalledIndex) <=
                          kSequenceCapacity &&
                      UsdGenImagingTestHook::groomTombstoneHistoryCount(*stalledIndex) <=
                          kSequenceCapacity,
                  "stalled prefix bounds issued sequence span and owner histories");
            Check(UsdGenImagingTestHook::groomSequenceLastIssued(*stalledIndex) ==
                      issuedAtCapacity && stalledInput->rootPrimReads == 0 &&
                      stalledInput->rootTraversal == 0 && stalledInput->groomAReads == 0 &&
                      stalledInput->groomBReads == 0,
                  "full sequence window rejects later ingress before sequence allocation or capture");

            stalledInput->captureRelease.release_wait();
            heldCapture.join();
            stalledOwner->Synchronize();
            Check(finalNotice.added.load(std::memory_order_acquire) &&
                      stalledIndex->GetPrim(finalPath).dataSource &&
                      UsdGenImagingTestHook::groomSequenceLastIssued(*stalledIndex) ==
                          UsdGenImagingTestHook::groomSequenceCompletedThrough(*stalledIndex),
                  "released prefix recovers the latest source namespace and closes the sequence gap");

            uint64_t const issuedBeforeReuse =
                UsdGenImagingTestHook::groomSequenceLastIssued(*stalledIndex);
            SdfPath const reusePath("/stalled/reused");
            stalledInput->retained->AddPrims(
                {{reusePath, TfToken("Scope"), HdRetainedContainerDataSource::New()}});
            stalledOwner->Synchronize();
            Check(UsdGenImagingTestHook::groomSequenceLastIssued(*stalledIndex) ==
                      issuedBeforeReuse + 1 &&
                      UsdGenImagingTestHook::groomSequenceCompletedThrough(*stalledIndex) ==
                          issuedBeforeReuse + 1,
                  "sequence capacity is reusable after stalled-prefix recovery");
            stalledIndex->RemoveObserver(TfCreateWeakPtr(&finalNotice));
        }
        stalledIndex.Reset();
        UsdGenGroomSceneIndex::DrainRetired();
        UsdGenImagingTestHook::Drain();
    }
    UsdGenImagingTestHook::setGroomSequenceCapacityForTesting(4096);
    std::printf("testUsdGenIncrementalCapture: %s\n",
                failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
