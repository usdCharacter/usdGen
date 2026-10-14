// Caller-boundary Hydra capture -> per-scene owner -> independent session work.
// One immutable scene snapshot and ordered notices; no application mutex.
#include "usdGenImaging/groomSceneIndexPlugin.h"
#include "usdGenImaging/usdGenDirtyRouter.h"
#include "usdGenImaging/usdGenTilePublisher.h"
#include "usdGenImaging/usdGenGraphDescBuilder.h"
#include "usdGenImaging/usdGenEnable.h"
#include "usdGenImaging/testHook.h"
#include "usdGen/debugCodes.h"
#include "usdGen/executionSequenceWindow.h"
#include "pxr/base/trace/trace.h"
#include "pxr/imaging/hd/sceneIndexPluginRegistry.h"
#include "pxr/imaging/hd/sceneGlobalsSchema.h"
#include "pxr/imaging/hd/legacyDisplayStyleSchema.h"
#include "pxr/imaging/hd/materialBindingSchema.h"
#include "pxr/imaging/hd/materialBindingsSchema.h"
#include "pxr/imaging/hd/overlayContainerDataSource.h"
#include "pxr/imaging/hd/primvarSchema.h"
#include "pxr/imaging/hd/primvarsSchema.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/selectionsSchema.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/imaging/hdsi/extComputationPrimvarPruningSceneIndex.h"
#include "pxr/base/tf/envSetting.h"
#include "pxr/base/tf/refPtr.h"
#include "pxr/imaging/hd/systemMessages.h"
#include <tbb/flow_graph.h>
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <map>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_ENV_SETTING(USDGEN_ENABLE, true, "Enable usdGen scene index plugins.");
TF_REGISTRY_FUNCTION(TfType) {
    HdSceneIndexPluginRegistry::Define<UsdGenGroomSceneIndexPlugin>();
}
TF_REGISTRY_FUNCTION(HdSceneIndexPlugin) {
    HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
        HdSceneIndexPluginRegistryTokens->allRenderers.GetString(),
        TfToken("UsdGenGroomSceneIndexPlugin"), nullptr, 0,
        HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
}
HdSceneIndexBaseRefPtr UsdGenGroomSceneIndexPlugin::_AppendSceneIndex(
    std::string const&, HdSceneIndexBaseRefPtr const& input,
    HdContainerDataSourceHandle const& args) {
    auto renderer = args ? HdStringDataSource::Cast(args->Get(
        HdSceneIndexPluginRegistryTokens->rendererDisplayName)) : nullptr;
    std::string const name = renderer ? renderer->GetTypedValue(0) : std::string();
    // The cap is an overlay for Storm, not physical geometry. Suppress its
    // prim, material and notices for other delegates, not just its shading.
    bool const storm = name == "GL" || name == "Storm";
    return UsdGenGroomSceneIndex::New(input, 0, storm, storm);
}
bool UsdGenGroomSceneIndexPlugin::_IsEnabled(HdContainerDataSourceHandle const&) const {
    return TfGetEnvSetting(USDGEN_ENABLE);
}

namespace {
using Pipeline = usdGen::UsdGenExecutionPipeline;
using Session = ::usdGenImaging::UsdGenImagingSession;
// OpenUSD 26.08 introduces pxr::Handle, so a short Handle alias inside the
// PXR namespace becomes ambiguous under MSVC and Clang. Keep this session
// type explicit at the boundary.
using SessionHandle = ::usdGenImaging::UsdGenSessionHandle;
using Key = ::usdGenImaging::UsdGenSessionKey;
using Desc = usdGen::UsdGenGraphDesc;
using CaptureCache = ::usdGenImaging::UsdGenGraphDescCaptureCache;
using Added = HdSceneIndexObserver::AddedPrimEntries;
using Removed = HdSceneIndexObserver::RemovedPrimEntries;
using Dirtied = HdSceneIndexObserver::DirtiedPrimEntries;
using TileMap = std::map<SdfPath, HdContainerDataSourceHandle>;
std::atomic<uint64_t> s_testCommandCapacity{4096};
std::atomic<uint64_t> s_testSequenceCapacity{4096};
SdfPath RenderPath(SdfPath const& description) {
    return description.AppendChild(TfToken("__usdGenRender"));
}
TfToken TypeName(HdSceneIndexPrim const& prim) {
    if (!prim.primType.IsEmpty()) return prim.primType;
    auto type = HdTokenDataSource::Cast(HdContainerDataSource::Get(prim.dataSource,
        HdDataSourceLocator(TfToken("__usdPrimInfo"), TfToken("typeName"))));
    return type ? type->GetTypedValue(0) : TfToken();
}
bool IsGroom(TfToken const& type) {
    return type == TfToken("UsdGenGroom") || type == TfToken("UsdGenDescription");
}

HdContainerDataSourceHandle RenderDataSource() {
    return HdRetainedContainerDataSource::New();
}

SdfPath MaterialPath(SdfPath const& description) {
    return ::usdGenImaging::UsdGenTilePublisher::MaterialPath(description);
}

SdfPath ScalpShadowPath(SdfPath const& description) {
    return ::usdGenImaging::UsdGenTilePublisher::ScalpShadowPath(description);
}

SdfPath ScalpShadowMaterialPath(SdfPath const& description) {
    return ::usdGenImaging::UsdGenTilePublisher::ScalpShadowMaterialPath(description);
}

// The prim-level containers a synthetic tile must inherit from (or have
// masked by) its owning UsdGenDescription, because the scene indices that
// author them sit UPSTREAM of this one and never see the tiles (06 §4.1
// displayStyle row; hdx/selectionTracker.cpp:38-49 reads selections off the
// TERMINAL index, which is downstream of us, so republishing here is enough).
// materialBindings is in the set because the tile's binding is a FUNCTION of
// the inherited displayStyle (see DescriptionOverlay): moving the usdview
// complexity slider has to rebind the tile, not just re-repr it, or the hair
// material stays bound at the wire level and Storm fails to compile it.
HdDataSourceLocatorSet const& InheritedFromDescriptionLocators() {
    static HdDataSourceLocatorSet const locators{
        HdLegacyDisplayStyleSchema::GetDefaultLocator(),
        HdSelectionsSchema::GetDefaultLocator(),
        HdMaterialBindingsSchema::GetDefaultLocator()};
    return locators;
}

// refineLevel of a prim-level container, or -1 when it states no opinion.
int RefineLevelOf(HdContainerDataSourceHandle const& container) {
    if (!container) return -1;
    if (HdIntDataSourceHandle const level =
            HdLegacyDisplayStyleSchema::GetFromParent(container).GetRefineLevel())
        return level->GetTypedValue(0);
    return -1;
}

// The all-purpose material the tile publisher bound, or an empty path.
SdfPath BoundMaterialPath(HdContainerDataSourceHandle const& tile) {
    HdMaterialBindingsSchema const bindings =
        HdMaterialBindingsSchema::GetFromParent(tile);
    if (!bindings.IsDefined()) return SdfPath();
    if (HdPathDataSourceHandle const path =
            bindings.GetMaterialBinding().GetPath())
        return path->GetTypedValue(0);
    return SdfPath();
}

// The value-preview material (usdGen:preview:*) a groom's tiles bind, or an
// empty path. Every tile of a description binds the same one, so the first
// answers for all; the prim exists exactly while it is bound.
SdfPath BoundPreviewMaterial(SdfPath const& description, TileMap const& tiles) {
    if (tiles.empty()) return SdfPath();
    SdfPath const bound = BoundMaterialPath(tiles.begin()->second);
    return ::usdGenImaging::UsdGenTilePublisher::IsPreviewMaterialPath(description, bound)
        ? bound : SdfPath();
}

// Null when the description states no opinion the tile has to inherit and the
// tile's own binding stands, so the common case allocates nothing and the tile
// data source is returned unwrapped.
HdContainerDataSourceHandle DescriptionOverlay(
    HdSceneIndexBaseRefPtr const& input, SdfPath const& description,
    SdfPath const& tilePath, HdContainerDataSourceHandle const& tile,
    bool stormMaterialPolicy) {
    if (!input) return nullptr;
    HdContainerDataSourceHandle const desc = input->GetPrim(description).dataSource;
    if (!desc) return nullptr;
    TfToken names[3];
    HdDataSourceBaseHandle values[3];
    size_t count = 0;
    for (TfToken const& name : {HdLegacyDisplayStyleSchema::GetSchemaToken(),
                                HdSelectionsSchema::GetSchemaToken()}) {
        if (HdDataSourceBaseHandle value = desc->Get(name)) {
            names[count] = name;
            values[count] = std::move(value);
            ++count;
        }
    }

    // Complexity parity with an unbound native UsdGeomBasisCurves. The
    // displayStyle above passes through UNCLAMPED, so Storm picks exactly the
    // repr the slider asks for (hdSt/basisCurves.cpp:320-343: 0 = WIRE lines,
    // 1 = RIBBON + HAIR normal, 2 = RIBBON + ROUND, 3 = HALFTUBE + ROUND).
    // The binding is what moves instead:
    //
    //   refineLevel 0  no materialBindings at all. The default hair shader
    //                  reads the ribbon orientation vector inData.Neye, which
    //                  the WIRE repr's curve vertex block does not declare
    //                  (basisCurves.glslfx:1212-1218) -> the material fails to
    //                  COMPILE, not just to shade. An AUTHORED binding is
    //                  hidden here too, because usdGen cannot know that
    //                  someone else's shader survives the wire repr either.
    //   refineLevel 1  the synthetic default binding is hidden, so the tile
    //                  falls back to Storm's flat displayColor shading exactly
    //                  like a native curve at Medium. An AUTHORED binding
    //                  wins from here up: the artist asked for it.
    //   refineLevel 2+ bound, as published.
    //
    // Upstream first: the slider's opinion, else the publication's fallback
    // (2, so a host with no opinion still gets shaded ribbons).
    int effective = RefineLevelOf(desc);
    if (effective < 0) effective = RefineLevelOf(tile);
    bool const authored =
        BoundMaterialPath(tile) !=
        ::usdGenImaging::UsdGenTilePublisher::DefaultMaterialPath(tilePath);
    // These repr restrictions belong to Storm. Applying them to MoonRay
    // discards an otherwise valid material at usdview's default Low setting.
    if (stormMaterialPolicy &&
        (effective == 0 || (effective == 1 && !authored))) {
        // A block, not an empty container: HdOverlayContainerDataSource MERGES
        // two containers of the same name, so an empty one would leave the
        // publisher's binding visible underneath. A block resolves to null
        // (hd/overlayContainerDataSource.cpp:94-97).
        names[count] = HdMaterialBindingsSchema::GetSchemaToken();
        values[count] = HdBlockDataSource::New();
        ++count;
    }

    if (count == 0) return nullptr;
    return HdRetainedContainerDataSource::New(count, names, values);
}

// The locators under which two publications of one tile differ. Containers
// are compared child by child and sampled values by content (VtArray equality
// short-cuts on shared storage), so an unchanged widths or colour array is not
// dirtied even when the publication rebuilt it. Under `primvars`, a primvar
// whose value alone changed dirties primvars/<name>/primvarValue (Hydra keeps
// its cached descriptor); any other change to it, such as its interpolation,
// dirties the whole primvar so the descriptor is read again.
void DiffTileDataSources(HdDataSourceBaseHandle const& a, HdDataSourceBaseHandle const& b,
                         HdDataSourceLocator const& at, HdDataSourceLocatorSet* out) {
    if (a == b) return;
    HdContainerDataSourceHandle const ca = HdContainerDataSource::Cast(a);
    HdContainerDataSourceHandle const cb = HdContainerDataSource::Cast(b);
    if (ca && cb) {
        TfTokenVector names = ca->GetNames();
        for (TfToken const& name : cb->GetNames())
            if (std::find(names.begin(), names.end(), name) == names.end())
                names.push_back(name);
        for (TfToken const& name : names)
            DiffTileDataSources(ca->Get(name), cb->Get(name), at.Append(name), out);
        return;
    }
    HdSampledDataSourceHandle const sa = HdSampledDataSource::Cast(a);
    HdSampledDataSourceHandle const sb = HdSampledDataSource::Cast(b);
    if (sa && sb && sa->GetValue(0.0f) == sb->GetValue(0.0f)) return;
    // Each publication blocks the absent motion primvars with a new block.
    if (HdBlockDataSource::Cast(a) && HdBlockDataSource::Cast(b)) return;
    TfToken const& primvars = HdPrimvarsSchema::GetSchemaToken();
    if (at.GetElementCount() >= 3 && at.GetFirstElement() == primvars &&
        at.GetLastElement() != HdPrimvarSchemaTokens->primvarValue)
        out->insert(HdDataSourceLocator(primvars, at.GetElement(1)));
    else
        out->insert(at);
}

// True when every dirtied locator lies in the scene globals container.
bool OnlySceneGlobals(HdDataSourceLocatorSet const& locators) {
    if (locators.IsEmpty()) return false;
    for (HdDataSourceLocator const& locator : locators)
        if (!locator.HasPrefix(HdSceneGlobalsSchema::GetDefaultLocator())) return false;
    return true;
}

// Whether a dirty can change a prim's resolved visibility.
bool TouchesVisibility(HdDataSourceLocatorSet const& locators) {
    return locators.IsEmpty() ||
        locators.Intersects(HdVisibilitySchema::GetDefaultLocator());
}

// Why a groom is dormant, or null when it is awake. A dormant groom is never
// captured or cooked, and its last-good tiles are held hidden; it still
// attaches its session once when first discovered. It is dormant while its
// Description resolves invisible (the input is flattened, so an invisible
// ancestor counts), or while a UsdGenGroom stands in for a Description it no
// longer has: with no active Description child and no operators of its own
// (on a stage only a Description carries usdGen:operatorOrder, served flat
// at the prim root like every mapped property) there is nothing to cook. A
// deactivated groom or Description needs no rule of its own: UsdImaging
// removes it from the input.
char const* DormantReason(HdSceneIndexBase const& input, HdSceneIndexPrim const& root,
                          SdfPath const& rootPath, SdfPath const& description) {
    if (description == rootPath && TypeName(root) == TfToken("UsdGenGroom") &&
        !HdContainerDataSource::Get(root.dataSource,
            HdDataSourceLocator(TfToken("operatorOrder"))))
        return "no active description";
    HdBoolDataSourceHandle const visibility = HdVisibilitySchema::GetFromParent(
        input.GetPrim(description).dataSource).GetVisibility();
    return visibility && !visibility->GetTypedValue(0.0f) ? "invisible" : nullptr;
}

// Overlaid on a hidden groom's tiles and scalp-shadow cap. The tiles keep the
// visibility they were cooked with, and a dormant groom is not re-cooked just
// to bake a new one.
HdContainerDataSourceHandle const& HiddenOverlay() {
    static HdContainerDataSourceHandle const hidden = HdRetainedContainerDataSource::New(
        HdVisibilitySchema::GetSchemaToken(),
        HdVisibilitySchema::Builder()
            .SetVisibility(HdRetainedTypedSampledDataSource<bool>::New(false))
            .Build());
    return hidden;
}

// Whether any expression of the description reads the frame or the time
// (SeExpr $frame / $time, sampler element expressions included).
bool DescReadsTime(Desc const& desc) {
    for (auto const& expression : desc.expressions)
        if (expression.source.find("$frame") != std::string::npos ||
            expression.source.find("$time") != std::string::npos)
            return true;
    return false;
}

// Records actual builder reads, including missing targets and GeomSubset
// parent meshes. No live data-source handle enters the owner catalog.
class RecordingInput final : public HdSceneIndexBase {
public:
    explicit RecordingInput(HdSceneIndexBaseRefPtr input) : input(std::move(input)) {}
    HdSceneIndexPrim GetPrim(SdfPath const& path) const override {
        paths.insert(path);
        return input->GetPrim(path);
    }
    SdfPathVector GetChildPrimPaths(SdfPath const& path) const override {
        paths.insert(path);
        return input->GetChildPrimPaths(path);
    }
    std::shared_ptr<const SdfPathVector> Dependencies(Desc const& desc) const {
        // Some adapter aggregates transport referenced values directly,
        // without a separate GetPrim through this recording facade.
        for (auto const& expression : desc.expressions) {
            paths.insert(expression.path);
            // geoSampler()/ptex() inputs: an edit of any target (or of a gprim
            // found beneath one) re-captures and recooks this groom.
            for (auto const& sampled : expression.inputs) {
                for (auto const& path : sampled.targets) paths.insert(path.GetPrimPath());
                for (auto const& path : sampled.geometries) paths.insert(path);
                for (auto const& path : sampled.maps) paths.insert(path);
            }
        }
        for (auto const& node : desc.nodes) {
            for (auto const& binding : node.expressionBindings) paths.insert(binding.expression.GetPrimPath());
            for (auto const& path : node.references) paths.insert(path.GetPrimPath());
        }
        // The previewed expression, map or operator recolours the strands.
        paths.insert(desc.preview.source.GetPrimPath());
        paths.erase(SdfPath());
        return std::make_shared<const SdfPathVector>(paths.begin(), paths.end());
    }
private:
    HdSceneIndexBaseRefPtr input;
    mutable std::set<SdfPath> paths;
};
}

struct UsdGenGroomSceneIndex::_Ingress {
    struct SourceEntry { SdfPath path; TfToken type; };
    struct Input {
        SdfPath root, description;
        Key key;
        std::shared_ptr<const Desc> desc;
        std::shared_ptr<const SdfPathVector> dependencies;
        std::shared_ptr<const CaptureCache> cache;
        bool authoredRender = false;
        // Dormant (see DormantReason): nothing was captured, so desc, cache
        // and dependencies are empty and the member keeps its previous ones.
        bool dormant = false;
    };
    uint64_t sequence = 0;
    int device = -2;
    double frame = 0;
    bool initial = false, failed = false;
    // A pressure-deferred ingress intentionally re-discovers the complete
    // synthetic catalog; it must not take the dependency-only fast path.
    bool forceFullDiscovery = false;
    bool recoverSourceNamespace = false;
    bool sourceFull = false;
    bool fullPopulation = true;
    // Async pass-through: the frontend already forwarded `dirtied` downstream
    // when it arrived (see _PrimsDirtied), so Apply advances the source stamps
    // without recording locators for the drain to send a second time.
    bool dirtiedDelivered = false;
    SdfPathVector captureRoots;
    Added added;
    Removed removed;
    Dirtied dirtied;
    std::vector<Input> inputs;
    std::vector<SourceEntry> source;
};

namespace {
// Test-only gates exercise the final-reference -> cleanup-enqueue gap. They
// allocate no production wait path: a deleter consults them only when armed.
struct TestFinalDeleterGate {
    tbb::flow::graph entered, release;
    std::atomic<bool> claimed{false};
    TestFinalDeleterGate() { entered.reserve_wait(); release.reserve_wait(); }
    void Pause() {
        entered.release_wait();
        release.wait_for_all();
    }
};
struct TestDrainWaitGate {
    tbb::flow::graph entered;
    std::atomic<bool> signaled{false};
    TestDrainWaitGate() { entered.reserve_wait(); }
    void Signal() { if (!signaled.exchange(true)) entered.release_wait(); }
};
std::shared_ptr<TestFinalDeleterGate> s_testFinalDeleterGate;
std::shared_ptr<TestDrainWaitGate> s_testDrainWaitGate;
}

// Constructed after the store: scenes close before store/retirement teardown.
// Every scene has its OWN pipeline on the shared runtime.
struct UsdGenSceneService {
    struct RetirementRecord {
        std::unique_ptr<tbb::flow::graph> completion{new tbb::flow::graph};
        std::atomic<bool> pending{true};
        static std::atomic<uint64_t> liveCount;
        RetirementRecord() { liveCount.fetch_add(1, std::memory_order_relaxed); completion->reserve_wait(); }
        ~RetirementRecord() { if (completion) Done(); liveCount.fetch_sub(1, std::memory_order_relaxed); }
        void Done() { if (pending.exchange(false)) completion->release_wait(); }
        void Wait() { completion->wait_for_all(); }
        // Called only with a strong live State after all its work is gone.
        // A public static handle can then outlive framework static teardown.
        void Disarm() { Done(); completion.reset(); }
    };
    struct Entry {
        std::weak_ptr<UsdGenGroomSceneIndex::_State> state;
        std::shared_ptr<RetirementRecord> retirement;
    };
    usdGen::UsdGenExecutionRuntime runtime{8};
    tbb::flow::graph retirement;
    tbb::flow::function_node<std::function<void()>> cleanup;
    // State destruction can wait during Pipeline shutdown, so it stays on
    // `cleanup` (unlimited).  This separate serial actor owns only the short
    // registration map mutations; no application mutex protects it.
    tbb::flow::graph registryGraph;
    tbb::flow::function_node<std::function<void()>> registry;
    std::map<RetirementRecord*, Entry> states;
    UsdGenSceneService() : cleanup(retirement, tbb::flow::unlimited,
        [](std::function<void()> action) { action(); return tbb::flow::continue_msg{}; }),
        registry(registryGraph, 1,
        [](std::function<void()> action) { action(); return tbb::flow::continue_msg{}; }) {}
    ~UsdGenSceneService();
    void Retire(std::function<void()> action) {
        if (!cleanup.try_put(std::move(action))) std::terminate();
    }
    void Register(Entry entry) {
        if (!registry.try_put([this, entry=std::move(entry)]() mutable {
            // Done is set only by the serial erase after State deletion, so
            // this retains the final-reference -> cleanup-enqueue gap.
            if (entry.retirement->pending.load(std::memory_order_acquire))
                states[entry.retirement.get()] = std::move(entry);
        })) std::terminate();
    }
    void Erase(std::shared_ptr<RetirementRecord> record) {
        if (!registry.try_put([this, record=std::move(record)] {
            states.erase(record.get());
            // A Drain wait returning now also guarantees a following
            // registry snapshot cannot retain this completed entry.
            record->Done();
        })) std::terminate();
    }
    std::vector<Entry> Snapshot() {
        struct Reply {
            tbb::flow::graph done;
            std::vector<Entry> entries;
            std::exception_ptr error;
            Reply() { done.reserve_wait(); }
        };
        auto reply = std::make_shared<Reply>();
        if (!registry.try_put([this, reply] {
            try {
                reply->entries.reserve(states.size());
                for (auto const& entry : states) reply->entries.push_back(entry.second);
            } catch (...) { reply->error = std::current_exception(); }
            reply->done.release_wait();
        })) std::terminate();
        reply->done.wait_for_all();
        if (reply->error) std::rethrow_exception(reply->error);
        return std::move(reply->entries);
    }
    void Barrier() { (void)Snapshot(); }
    size_t RetainedStateCount() { return Snapshot().size(); }
    size_t LiveStateCount() {
        auto entries = Snapshot();
        size_t count = 0;
        for (auto const& entry : entries) if (entry.state.lock()) ++count;
        return count;
    }
    static uint64_t RetirementRecordCount() {
        return RetirementRecord::liveCount.load(std::memory_order_acquire);
    }
    void DrainRetired();
};
std::atomic<uint64_t> UsdGenSceneService::RetirementRecord::liveCount{0};
namespace {
UsdGenSceneService& SceneService() {
    (void)::usdGenImaging::UsdGenSessionStore::GetInstance();
    // Intentionally leaked, like the process-lifetime registries in
    // executionRetirement.cpp, so that ~UsdGenSceneService does not run during
    // static destruction.
    //
    // That destructor quiesces through the scheduled actors and through each
    // scene's pipeline. Every one of those steps needs some other thread to
    // execute a posted task, and by the time atexit handlers run on Windows
    // the TBB worker threads are gone: the posted task is never executed (a
    // task put to the registry node and given two seconds with no help was
    // still unexecuted) and the wait for it spins at 100% CPU for as long as
    // the process is allowed to live. A flow graph's wait_for_all only runs
    // the tasks belonging to that graph, so a reply-graph wait -- Snapshot's,
    // and UsdGenExecutionPipeline::Await's -- cannot rescue itself by running
    // the work it is waiting for.
    //
    // Leaking removes the need for that quiesce instead of papering over it:
    // the service now outlives every handle rather than racing them, so the
    // deleter's "service already gone" path is simply never taken, and the OS
    // reclaims the mapping at process death. Scenes torn down while the
    // process is live are unaffected -- they retire through the actors as
    // before, with the workers running.
    static UsdGenSceneService* service = new UsdGenSceneService;
    return *service;
}
}

struct UsdGenGroomSceneIndex::_State : std::enable_shared_from_this<_State> {
    // All replies which an accepted attachment may need are admitted as one
    // bundle before AttachAsync starts.  A later public command flood can
    // therefore never strand a session attachment or its callback removal.
    struct AttachmentReplies {
        Pipeline::CommandTicket attachAck;
        Pipeline::CommandTicket detachAck;
        Pipeline::CommandTicket unregisterAck;
        Pipeline::CommandTicket sourceAttach;
        Pipeline::CommandTicket sourceDetach;
        Pipeline::CommandTicket sourceRegister;
        Pipeline::CommandTicket sourceUnregister;
        Pipeline::CommandMailbox republish;
        Pipeline::CommandMailbox progress;
        explicit operator bool() const noexcept {
            return attachAck && detachAck && unregisterAck && sourceAttach &&
                sourceDetach && republish;
        }
    };
    struct Groom {
        uint64_t id = 0, captured = 0;
        // Incremented for every attachment attempt and release.  A key can
        // return to an equal value (A -> B -> A), so key/handle equality is
        // not sufficient to admit delayed store or session callbacks.
        uint64_t attachmentEpoch = 0;
        bool alive = true, authoredRender = false;
        // dormant: the last ingress found the groom hidden or without an
        // active Description, so it is neither captured nor cooked. hidden:
        // its tiles are published invisible. Going dormant hides at once;
        // waking reveals only when the cook that woke it completes, so the
        // tiles never show geometry from before the groom was hidden.
        bool dormant = false, hidden = false;
        SdfPath root, description;
        Key key;
        SessionHandle session;
        int callback = -1, device = -2;
        // A subscription can publish indefinitely.  Its command mailbox owns
        // one durable owner-ingress credit and replaces an undelivered payload
        // with the newest immutable snapshot rather than growing a queue.
        std::shared_ptr<AttachmentReplies> replies;
        double frame = 0;
        std::shared_ptr<const Desc> desc;
        std::shared_ptr<const SdfPathVector> dependencies;
        std::shared_ptr<const CaptureCache> cache;
        std::shared_ptr<const usdGenImaging::UsdGenDirtyRouter> router;
        std::shared_ptr<const TileMap> tiles = std::make_shared<const TileMap>();
        // Tiles shown before the current provisional cook. A failed or
        // superseded attempt restores this exact immutable baseline.
        std::shared_ptr<const TileMap> progressBaseline;
        HdContainerDataSourceHandle progressScalpBaseline;
        uint64_t progressScalpDigest = 0;
        uint64_t progressEpoch = 0, progressSequence = 0;
        bool progressActive = false;
        bool progressWasHidden = false;
        // The scalp-shadow cap, built once per publication beside the tiles.
        // Null when the generation carries none.
        HdContainerDataSourceHandle scalpShadow;
        uint64_t scalpDigest = 0;
        // sessionGeneration is reset when an isolated CUDA session replaces a
        // shared CPU session.  Keeping it separate retains last-good ordinary
        // BasisCurves tiles while unimplemented device publication is rejected.
        int64_t generation = -1;
        int64_t sessionGeneration = -1;
    };
    struct View {
        uint64_t id = 0;
        SdfPath root, description;
        std::shared_ptr<const TileMap> tiles;
        int64_t generation;
        std::shared_ptr<const SdfPathVector> dependencies;
        std::shared_ptr<const CaptureCache> cache;
        double frame = 0;
        // The description's expressions read $frame or $time, so a change of
        // the scene globals' current frame alone changes its result.
        bool readsTime = true;
        // usdGen:look:* carried to the synthetic default material. The tile
        // bakes only rootColor (into displayColor); the tip colour, ramp
        // exponent and jitter reach Storm through the material's parameters,
        // so the View has to carry them to GetPrim().
        usdGen::UsdGenLookDesc look;
        // The scalp-shadow cap and the identity of its contents: the prim is
        // served from here, and nothing else would tell Hydra it changed.
        HdContainerDataSourceHandle scalpShadow;
        uint64_t scalpDigest = 0;
        bool dormant = false, hidden = false;
    };
    struct Snapshot {
        std::vector<View> members;
        uint64_t capturedThrough = 0;
        bool captureTrusted = true;
        struct SourceValue {
            TfToken type;
            uint64_t stamp = 0;
            // Everything dirtied after stamp `since`, up to `stamp`. A
            // frontend showing a value stamped `since` or later forwards just
            // these; an older one, or a value re-announced by discovery
            // (universal here), dirties the whole prim.
            uint64_t since = 0;
            HdDataSourceLocatorSet locators = HdDataSourceLocatorSet::UniversalSet();

            // Records the input dirty `dirtied` at `seq`. `shown` is the
            // stamp of this value in the snapshot the frontend displays.
            void Dirty(uint64_t seq, HdDataSourceLocatorSet const& dirtied, uint64_t shown) {
                if (stamp <= shown) {
                    since = stamp;
                    locators = dirtied;
                } else {
                    locators.insert(dirtied);
                }
                stamp = seq;
            }
            // What a frontend showing `shown` must dirty to reach this value.
            HdDataSourceLocatorSet const& DirtySince(uint64_t shown) const {
                static HdDataSourceLocatorSet const all = HdDataSourceLocatorSet::UniversalSet();
                return shown >= since ? locators : all;
            }
        };
        std::shared_ptr<const std::map<SdfPath, SourceValue>> source;
        SourceValue rootValue;
        Snapshot()
            : source(std::make_shared<const std::map<SdfPath, SourceValue>>()) {}
    };
    // catalog is owner-written/capture-read. visible is only installed by
    // the serialized Hydra frontend immediately before its matching notice.
    std::shared_ptr<const Snapshot> catalog = std::make_shared<const Snapshot>();
    std::shared_ptr<const Snapshot> visible = std::make_shared<const Snapshot>();
    // The owner replaces this single presentation target.  There is no
    // historical notice queue: a stalled async consumer retains at most the
    // newest immutable snapshot and the frontend computes its complete delta
    // from the snapshot it actually displayed.
    std::shared_ptr<const Snapshot> pending;
    HdSceneIndexBasePtr recipient; // initialized before admission; weak
    usdGen::UsdGenExecutionSequenceWindow sequences;
    std::atomic<bool> closing{false};
    std::atomic<bool> quiesced{false};
    // Retired visible snapshots awaiting destruction. The drain's `before`
    // snapshot pins the previous frame's tiles, captures and scalp cap, so
    // destroying it on the Hydra thread costs tens of ms of frees per drain
    // (ss1m: ~17ms); ReleaseRetired runs that destruction on the TBB arena
    // instead, where the cook workers sit idle this late in the frame. The
    // bundle is self-contained (no _State back-references), so concurrent
    // GetPrim readers holding the old snapshot stay correct by shared
    // ownership whichever thread drops the last reference.
    tbb::task_group releaser;
    std::atomic<int> releaserPending{0};
    void ReleaseRetired(std::shared_ptr<const Snapshot>&& retired) {
        if (!retired || retired->members.empty()) return;
        int const pending =
            releaserPending.fetch_add(1, std::memory_order_acq_rel);
        if (pending >= 2) {
            // Releases are already two frames behind: destroy inline (the
            // old cost) rather than letting retired frames pile up.
            releaserPending.fetch_sub(1, std::memory_order_acq_rel);
            return;
        }
        // task_group::run takes a const-callable, so the bundle rides in a
        // copyable handle; resetting it drops the last drain reference.
        auto bundle = std::make_shared<std::shared_ptr<const Snapshot>>(
            std::move(retired));
        releaser.run([this, bundle]() {
            TRACE_SCOPE("usdGen drain: release retired");
            bundle->reset();
            releaserPending.fetch_sub(1, std::memory_order_acq_rel);
        });
    }
    std::atomic<bool> asyncAllowed{false};
    bool const progressiveRenderer;
    // An input notice that cannot enter the bounded owner immediately
    // requests one later authoritative namespace/synthetic rescan.  It
    // deliberately has no sequence:
    // no waiter may depend on work that was never admitted.
    std::atomic<bool> deferredFullCapture{false};
    // asyncPoll recovery back-off (see _SystemMessage): the steady-clock
    // tick before which a poll must not retry after a THROWING recovery
    // capture, the consecutive-failure count that sizes the back-off, and
    // a warn-once latch.
    std::atomic<int64_t> pollRecoveryNotBefore{0};
    std::atomic<uint32_t> pollRecoveryFailures{0};
    std::atomic<bool> pollRecoveryWarned{false};
    // Owner-only state below.
    std::map<SdfPath, std::shared_ptr<Groom>> members;
    std::map<SdfPath, uint64_t> events, tombstones;
    uint64_t nextId = 0, completedPrefix = 0;
    std::map<uint64_t, size_t> holds;
    struct Waiter { uint64_t watermark; std::function<void()> done; };
    std::vector<Waiter> waiters;
    bool closeProcessed = false;
    bool captureTrusted = true;
    // Owner-queued input namespace baseline.  It intentionally advances with
    // catalog packets, not frontend delivery, so a held observer cannot make
    // a later recovery diff against a stale external snapshot.
    std::shared_ptr<const std::map<SdfPath, Snapshot::SourceValue>> source =
        std::make_shared<const std::map<SdfPath, Snapshot::SourceValue>>();
    bool sourceKnown = false;
    uint64_t sourceThrough = 0;
    Snapshot::SourceValue rootValue;
    std::atomic<uint64_t> captureCount{0};
    std::atomic<uint64_t> cookCount{0};
    static std::atomic<uint64_t>& ProcessCooks() {
        static std::atomic<uint64_t> count{0};
        return count;
    }
    static std::atomic<uint64_t>& ProcessPublishes() {
        static std::atomic<uint64_t> count{0};
        return count;
    }
    uint64_t captureTrustSequence = 0;
    std::vector<TfWeakPtr<Session>> usedSessions;
    std::unique_ptr<Pipeline> owner; // removed at process shutdown, even if public index survives
    // Destruction must be able to enqueue Close even when ordinary command
    // admission is saturated.  Keep this credit from construction onward.
    Pipeline::CommandTicket closeTicket;
    std::vector<Pipeline::CommandTicket> testHeldCredits;

    explicit _State(usdGen::UsdGenExecutionRuntime& runtime,
                    bool progressive)
        : sequences(s_testSequenceCapacity.load(std::memory_order_acquire)),
          progressiveRenderer(progressive),
          owner(new Pipeline(runtime, 4096,
                             s_testCommandCapacity.load(std::memory_order_acquire))) {
        closeTicket = owner->ReserveCommandTicket();
        if (!closeTicket)
            throw std::runtime_error("usdGen groom owner requires command capacity for shutdown");
    }
    ~_State() { releaser.wait(); if (owner) owner->Shutdown(); }
    auto SnapshotValue() const { return std::atomic_load(&catalog); }
    auto VisibleSnapshot() const { return std::atomic_load(&visible); }
    std::shared_ptr<const Snapshot> PublishSnapshot() {
        auto next = std::make_shared<Snapshot>();
        next->capturedThrough = completedPrefix;
        next->captureTrusted = captureTrusted;
        next->source = source;
        next->rootValue = rootValue;
        for (auto const& item : members) {
            auto const& g = *item.second;
            next->members.push_back({g.id, g.root, g.description, g.tiles, g.generation,
                                     g.dependencies, g.cache, g.frame,
                                     !g.desc || DescReadsTime(*g.desc),
                                     g.desc ? g.desc->look : usdGen::UsdGenLookDesc(),
                                     g.scalpShadow, g.scalpDigest, g.dormant, g.hidden});
        }
        auto result = std::shared_ptr<const Snapshot>(std::move(next));
        std::atomic_store(&catalog, result);
        return result;
    }
    void QueuePublication(Added = {}, Removed = {}, Dirtied = {}) {
        // Atomic shared_ptr exchange releases every superseded snapshot (and
        // its TileMap owners) immediately when no reader still holds it.
        std::atomic_store(&pending, PublishSnapshot());
    }
    bool Post(Pipeline::CommandTicket&& ticket, std::function<void()> command,
              std::function<void()> onCancel = {}) {
        return owner->PostCommand(std::move(ticket), std::move(command), std::move(onCancel));
    }
    bool Post(std::function<void()> command) {
        return owner->PostCommand(std::move(command));
    }
    void CheckWaiters() {
        auto it = waiters.begin();
        while (it != waiters.end()) {
            bool ready = (!closing.load() || closeProcessed) && completedPrefix >= it->watermark &&
                (holds.empty() || holds.begin()->first > it->watermark);
            if (!ready) { ++it; continue; }
            auto done = std::move(it->done);
            it = waiters.erase(it);
            done();
        }
    }
    void Hold(uint64_t seq) { ++holds[seq]; }
    void Release(uint64_t seq) {
        auto it = holds.find(seq);
        if (it == holds.end() || it->second == 0) std::terminate();
        if (--it->second == 0) holds.erase(it);
        CheckWaiters();
    }
    void CompleteIngress(uint64_t seq) {
        if (!sequences.Complete(seq)) std::terminate();
        completedPrefix = sequences.CompletedThrough();
        // Every capture at/below this prefix has arrived. Its suppression
        // history is no longer needed by any delayed ingress.
        auto prune = [this](auto& history) {
            for (auto it = history.begin(); it != history.end();)
                if (it->second <= completedPrefix) it = history.erase(it);
                else ++it;
        };
        prune(events);
        prune(tombstones);
        // Catalog reuse is safe only after every earlier capture has been
        // applied. Publication callbacks do not advance this source watermark.
        QueuePublication();
        CheckWaiters();
    }
    void Synchronize() {
        if (quiesced.load()) return;
        auto self = shared_from_this();
        const uint64_t watermark = sequences.LastIssued();
        auto ticket = std::make_shared<Pipeline::CommandTicket>(
            owner->ReserveCommandTicket());
        if (!*ticket)
            throw std::runtime_error("usdGen groom owner saturated before synchronize admission");
        owner->Await([self, watermark, ticket](std::function<void()> done) {
            if (!self->Post(std::move(*ticket), [self, watermark, done] {
                self->waiters.push_back({watermark, done});
                self->CheckWaiters();
            }, [done] { done(); }))
                done();
        });
    }
    bool Current(std::shared_ptr<Groom> const& g) const {
        auto it = members.find(g->root);
        return g->alive && it != members.end() && it->second == g;
    }
    void RememberSession(SessionHandle const& session) {
        TfWeakPtr<Session> candidate = TfCreateWeakPtr(session.operator->());
        bool found = false;
        for (auto it = usedSessions.begin(); it != usedSessions.end();) {
            if (it->IsExpired()) { it = usedSessions.erase(it); continue; }
            if (*it == candidate) found = true;
            ++it;
        }
        if (!found) usedSessions.push_back(std::move(candidate));
    }
    bool NewerEvent(SdfPath const& path, uint64_t seq) const {
        for (auto const& e : events)
            if (e.second > seq && (path.HasPrefix(e.first) || e.first.HasPrefix(path))) return true;
        return false;
    }
    bool RemovedSince(SdfPath const& root, uint64_t seq) const {
        for (auto const& e : tombstones)
            if (root.HasPrefix(e.first) && e.second >= seq) return true;
        return false;
    }
    void Notify(Added const& added, Removed const& removed, Dirtied const& dirtied) {
        if (closing.load()) return;
        // Owner callbacks never enter Hydra observers.  The immutable packet
        // is delivered by the serialized frontend on a synchronous ingress or
        // negotiated asyncPoll boundary.
        QueuePublication(added, removed, dirtied);
    }
    void Detach(SessionHandle session, Key key, uint64_t seq,
                Pipeline::CommandTicket&& sourceTicket,
                Pipeline::CommandTicket&& replyTicket) {
        if (!session) return;
        Hold(seq);
        auto self = shared_from_this();
        auto reply = std::make_shared<Pipeline::CommandTicket>(std::move(replyTicket));
        bool accepted = false;
        try {
            accepted = ::usdGenImaging::UsdGenSessionStore::GetInstance().DetachAsync(
                std::move(sourceTicket), key, session,
                [self, session, seq, reply] {
                    (void)self->Post(std::move(*reply), [self, session, seq] { self->Release(seq); },
                                      [self, seq] { self->Release(seq); });
                });
        } catch (...) { std::terminate(); }
        if (!accepted) Release(seq);
    }
    uint64_t AdvanceAttachmentEpoch(Groom& g) {
        // Wrapping would re-admit a callback from a prior lifetime.  This is
        // an unrecoverable owner-state corruption rather than a safe reset.
        if (g.attachmentEpoch == std::numeric_limits<uint64_t>::max())
            std::terminate();
        return ++g.attachmentEpoch;
    }
    void ReleaseSession(std::shared_ptr<Groom> const& g, uint64_t seq) {
        // This also invalidates an AttachAsync that has not acquired a
        // session yet.  It deliberately does not clear last-good display.
        AdvanceAttachmentEpoch(*g);
        if (g->progressActive) {
            g->tiles = g->progressBaseline;
            if (g->progressWasHidden) g->hidden = true;
            g->scalpShadow = g->progressScalpBaseline;
            g->scalpDigest = g->progressScalpDigest;
            Notify({}, {}, {});
        }
        g->progressBaseline.reset();
        g->progressScalpBaseline = nullptr;
        g->progressActive = false;
        g->progressWasHidden = false;
        g->progressEpoch = g->progressSequence = 0;
        if (g->session) {
            SessionHandle const session = g->session;
            Key const key = g->key;
            int const callback = g->callback;
            g->session = {};
            g->callback = -1;
            auto replies = std::move(g->replies);
            if (!replies) std::terminate();
            // The subscription closure retains its own mailbox handle until
            // the session owner erases it.  Drop the groom's handle now, and
            // retain this local one through the unregister acknowledgement so
            // the durable credit cannot disappear while a callback is live.
            auto mailbox = replies->republish;
            if (callback >= 0) {
                Hold(seq);
                auto self = shared_from_this();
                auto reply = std::make_shared<Pipeline::CommandTicket>(
                    std::move(replies->unregisterAck));
                try {
                    if (!session->UnregisterRepublishCallbackAsync(
                        std::move(replies->sourceUnregister), callback,
                        [self, seq, reply, mailbox] {
                            (void)self->Post(std::move(*reply), [self, seq, mailbox] { self->Release(seq); },
                                              [self, seq] { self->Release(seq); });
                        }))
                        Release(seq);
                } catch (...) { std::terminate(); }
            }
            Detach(session, key, seq, std::move(replies->sourceDetach),
                   std::move(replies->detachAck));
        }
    }
    void Remove(std::shared_ptr<Groom> const& g, uint64_t seq) {
        g->alive = false;
        ReleaseSession(g, seq);
    }
    void Close() {
        if (closing.exchange(true)) return;
        auto self = shared_from_this();
        if (!Post(std::move(closeTicket), [self] {
            try {
                const uint64_t seq = self->sequences.LastIssued();
                for (auto const& item : self->members) self->Remove(item.second, seq);
                self->members.clear();
                self->PublishSnapshot();
                self->closeProcessed = true;
                self->CheckWaiters();
            } catch (...) { std::terminate(); }
        })) {
            // The permanent ticket can only fail after owner shutdown.  There
            // is then no live owner state to mutate; retirement will observe
            // the already-closed state without a second callback attempt.
            closeProcessed = true;
        }
    }
    void Publish(std::shared_ptr<Groom> const& g, SessionHandle const& sourceSession,
                 uint64_t attachmentEpoch,
                 Session::CommitPayload const& payload) {
        if (closing.load() || !Current(g) ||
            !sourceSession || g->session != sourceSession ||
            g->attachmentEpoch != attachmentEpoch) return;
        if (payload.progressEpoch && payload.progressEpoch < g->progressEpoch)
            return;
        if (!payload.published || !payload.generation) {
            if (payload.generation &&
                payload.generation->id > g->sessionGeneration) {
                // A newer attempt can begin previewing before an older
                // successful terminal reaches this owner. The core's failed
                // snapshot still carries that last complete generation.
                auto complete = payload;
                complete.published = true;
                Publish(g, sourceSession, attachmentEpoch, complete);
                return;
            }
            if (g->progressActive && payload.progressEpoch == g->progressEpoch) {
                g->tiles = g->progressBaseline;
                if (g->progressWasHidden) g->hidden = true;
                g->scalpShadow = g->progressScalpBaseline;
                g->scalpDigest = g->progressScalpDigest;
                g->progressBaseline.reset();
                g->progressScalpBaseline = nullptr;
                g->progressActive = false;
                g->progressWasHidden = false;
                Notify({}, {}, {});
            }
            g->progressEpoch = (std::max)(g->progressEpoch, payload.progressEpoch);
            return;
        }
        // Fence a terminal attempt before considering the generation id.
        // Progress and terminal notices use separate owner mailboxes, and a
        // late progress delivery must never re-open a completed attempt.
        if (payload.progressEpoch) {
            g->progressEpoch = (std::max)(g->progressEpoch, payload.progressEpoch);
            g->progressActive = false;
        }
        if (
            payload.generation->id <= g->sessionGeneration) return;
        TRACE_SCOPE("usdGen publish tiles to the scene index");
        auto const& generation = *payload.generation;
        if (generation.device) {
            // usdGen has not implemented a stock-Storm GPU-resident
            // BasisCurves handoff.
            // Do not download device geometry or replace the displayed CPU
            // snapshot; the core session already diagnoses this request.
            TF_WARN("usdGen stock-Storm GPU-resident BasisCurves handoff is not implemented");
            return;
        }
        // USDGEN_COMMIT: time the scene-index publish (tile + scalp data
        // source assembly, outside the cook phases) the way the cooker times
        // the cook itself. Zero cost when the code is off (one atomic load).
        bool const timePublish = TfDebug::IsEnabled(USDGEN_COMMIT);
        auto const publishStart = std::chrono::steady_clock::now();
        auto fresh = std::make_shared<TileMap>();
        const auto render = RenderPath(g->description);
        // All-or-nothing namespace check, up front (was folded into the
        // assembly loop): a stale tile discards the whole publication.
        for (auto const& tile : generation.tiles) {
            if (!tile.primPath.HasPrefix(render)) return; // old description namespace
        }
        // Tile data sources are independent: assemble them on the TBB arena
        // (the cook workers are idle this late in the frame), then emplace
        // in tile order. TileMap is std::map, so iteration order — and the
        // notices derived from it — are identical at any worker count.
        std::vector<HdContainerDataSourceHandle> built(generation.tiles.size());
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, generation.tiles.size()),
            [&](tbb::blocked_range<size_t> const& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    built[i] = ::usdGenImaging::UsdGenTilePublisher::
                        BuildTileDataSource(generation.tiles[i], generation.id);
                }
            });
        for (size_t i = 0; i < generation.tiles.size(); ++i) {
            fresh->emplace(generation.tiles[i].primPath, built[i]);
        }
        // The scalp-shadow cap lives beside the tiles, not in the TileMap:
        // it is a mesh, and the notice diff treats the two prim types
        // differently.
        HdContainerDataSourceHandle freshScalp;
        uint64_t freshScalpDigest = 0;
        if (!generation.scalpShadow.IsEmpty() &&
            generation.scalpShadow.primPath.HasPrefix(render)) {
            freshScalp = ::usdGenImaging::UsdGenTilePublisher::
                BuildScalpShadowDataSource(generation.scalpShadow, generation.id);
            freshScalpDigest = generation.scalpShadow.digest;
        }
        Added added;
        Removed removed;
        Dirtied dirtied;
        for (auto const& old : *g->tiles)
            if (!fresh->count(old.first)) removed.emplace_back(old.first);
        for (auto const& tile : *fresh) {
            if (!g->tiles->count(tile.first)) added.emplace_back(tile.first, TfToken("basisCurves"));
            else {
                // Missed intermediate publications make their report an
                // invalid diff against THIS scene's displayed baseline.
                HdDataSourceLocatorSet locators;
                if (g->generation + 1 != generation.id) locators.insert(HdDataSourceLocator());
                else for (auto const& report : payload.report.tiles) {
                    if (report.primPath != tile.first) continue;
                    for (auto const& loc : ::usdGenImaging::UsdGenTilePublisher::NoticesFor(report).all())
                        locators.insert(loc);
                }
                if (!locators.IsEmpty()) dirtied.emplace_back(tile.first, locators);
            }
        }
        auto router = std::make_shared<usdGenImaging::UsdGenDirtyRouter>();
        if (payload.routing) router->Rebuild(*payload.routing);
        g->router = std::move(router);
        g->sessionGeneration = generation.id;
        g->generation = generation.id;
        g->tiles = std::move(fresh);
        g->progressBaseline.reset();
        g->progressScalpBaseline = nullptr;
        g->progressActive = false;
        g->progressWasHidden = false;
        g->scalpShadow = std::move(freshScalp);
        g->scalpDigest = freshScalpDigest;
        ProcessPublishes().fetch_add(1, std::memory_order_acq_rel);
        if (timePublish) {
            double const ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - publishStart).count();
            TF_DEBUG(USDGEN_COMMIT).Msg(
                "usdGen publish   %s gen %llu: %zu tiles + scalp in %.2f ms\n",
                g->description.GetText(), (unsigned long long)generation.id,
                generation.tiles.size(), ms);
        }
        Notify(added, removed, dirtied);
    }
    void PublishProgress(std::shared_ptr<Groom> const& g,
                         SessionHandle const& sourceSession,
                         uint64_t attachmentEpoch,
                         Session::ProgressPayload const& payload) {
        if (closing.load() || !Current(g) || g->dormant || !sourceSession ||
            g->session != sourceSession || g->attachmentEpoch != attachmentEpoch ||
            !payload.epoch || !payload.tiles || payload.tiles->empty() ||
            payload.epoch < g->progressEpoch) return;
        if (payload.epoch != g->progressEpoch) {
            // A newer cook replaces any incomplete attempt. Preserve only
            // the last complete set as its rollback baseline.
            if (g->progressActive) {
                g->tiles = g->progressBaseline;
                if (g->progressWasHidden) g->hidden = true;
                g->scalpShadow = g->progressScalpBaseline;
                g->scalpDigest = g->progressScalpDigest;
            }
            g->progressBaseline = g->tiles;
            g->progressScalpBaseline = g->scalpShadow;
            g->progressScalpDigest = g->scalpDigest;
            g->progressEpoch = payload.epoch;
            g->progressSequence = 0;
            g->progressActive = true;
            g->progressWasHidden = g->hidden;
            if (g->hidden) {
                // A waking groom must reveal only completed tiles. Previous
                // geometry slept through edits and cannot be shown yet.
                g->tiles = std::make_shared<const TileMap>();
                g->scalpShadow = nullptr;
                g->scalpDigest = 0;
                g->hidden = false;
            }
        } else if (!g->progressActive) {
            // Terminal publication or rollback already closed this epoch.
            return;
        }
        if (payload.sequence <= g->progressSequence) return;
        auto fresh = std::make_shared<TileMap>(*g->tiles);
        SdfPath const render = RenderPath(g->description);
        for (auto const& tile : *payload.tiles) {
            if (!tile || tile->epoch != payload.epoch ||
                tile->sequence <= g->progressSequence ||
                !tile->tile.primPath.HasPrefix(render)) continue;
            (*fresh)[tile->tile.primPath] =
                ::usdGenImaging::UsdGenTilePublisher::BuildTileDataSource(
                    tile->tile, g->generation + 1);
        }
        g->progressSequence = payload.sequence;
        g->tiles = std::move(fresh);
        ProcessPublishes().fetch_add(1, std::memory_order_acq_rel);
        Notify({}, {}, {});
    }
    // Shows a woken groom's tiles again. No-op while it is dormant.
    void Reveal(std::shared_ptr<Groom> const& g) {
        if (!g->hidden || g->dormant || closing.load() || !Current(g)) return;
        g->hidden = false;
        Notify({}, {}, {});
    }
    void Cook(std::shared_ptr<Groom> const& g, uint64_t seq) {
        // An Attach reply can land after its groom went dormant again.
        if (g->dormant) return;
        if (!g->session || !g->desc || closing.load() || !Current(g)) { Reveal(g); return; }
        // A hidden groom that is awake is waiting for exactly this cook.
        bool const reveals = g->hidden;
        cookCount.fetch_add(1, std::memory_order_acq_rel);
        ProcessCooks().fetch_add(1, std::memory_order_acq_rel);
        uint64_t const attachmentEpoch = g->attachmentEpoch;
        Session::CommitRequest request;
        request.reason = usdGen::UsdGenCommitReason::NoticeBatchEnd;
        request.desc = g->desc;
        request.callerDevice = g->device;
        // This plugin has no implemented stock-Storm device-resident
        // publication handoff. Set this per request (rather than relying on
        // session state) so CUDA authoring fails closed and retains the
        // previous CPU snapshot.
        request.devicePublication = false;
        request.progressive = progressiveRenderer &&
            asyncAllowed.load(std::memory_order_acquire);
        if (!g->session->HasAppDriver()) request.frame = g->frame;
        auto self = shared_from_this();
        auto ticket = std::make_shared<Pipeline::CommandTicket>(
            owner->ReserveCommandTicket());
        if (!*ticket) { Reveal(g); return; }
        Hold(seq);
        bool accepted = false;
        try {
            SessionHandle const sourceSession = g->session;
            accepted = sourceSession->CommitAsync(std::move(request),
                [self, g, sourceSession, attachmentEpoch, seq, ticket, reveals](
                    Session::CommitPayload const& payload, Pipeline::Outcome) {
                    (void)self->Post(std::move(*ticket), [self, g, sourceSession, attachmentEpoch, seq, payload, reveals] {
                        try {
                            self->Publish(g, sourceSession, attachmentEpoch, payload);
                            // Whether or not it published: an unchanged
                            // groom keeps tiles that are already current.
                            if (reveals) self->Reveal(g);
                        }
                        catch (...) { TF_WARN("usdGen scene publication failed"); }
                        self->Release(seq);
                    }, [self, seq] { self->Release(seq); });
                });
        } catch (...) { TF_WARN("usdGen scene cook request failed"); }
        if (!accepted) {
            Reveal(g);
            Release(seq);
            return;
        }
        // Rejected cook admission leaves the descriptor latch intact for the
        // next ingress.  The accepted producer has its reply ticket already.
        g->session->ConsumeNeedsDesc();
    }
    void Attach(std::shared_ptr<Groom> const& g, uint64_t seq) {
        auto self = shared_from_this();
        Key const requestedKey = g->key;
        uint64_t const attachmentEpoch = AdvanceAttachmentEpoch(*g);
        auto replies = std::make_shared<AttachmentReplies>();
        replies->attachAck = owner->ReserveCommandTicket();
        replies->detachAck = owner->ReserveCommandTicket();
        replies->unregisterAck = owner->ReserveCommandTicket();
        replies->sourceAttach = ::usdGenImaging::UsdGenSessionStore::GetInstance()
            .ReserveLifecycleCommand();
        replies->sourceDetach = ::usdGenImaging::UsdGenSessionStore::GetInstance()
            .ReserveLifecycleCommand();
        replies->republish = owner->ReserveCommandMailbox();
        if (progressiveRenderer) replies->progress = owner->ReserveCommandMailbox();
        // Every path that gives up on the attachment reveals: no cook is
        // coming to do it, and a later ingress retries the attachment.
        if (!*replies) { Reveal(g); return; }
        Hold(seq);
        bool accepted = false;
        try {
            accepted = ::usdGenImaging::UsdGenSessionStore::GetInstance().AttachAsync(
                std::move(replies->sourceAttach), requestedKey,
                [self, g, requestedKey, attachmentEpoch, seq, replies](SessionHandle session) {
                    (void)self->Post(std::move(replies->attachAck), [self, g, requestedKey, attachmentEpoch, seq, session, replies] {
                        try {
                            if (!session) { self->Reveal(g); self->Release(seq); return; }
                            self->RememberSession(session);
                            if (self->closing.load() || !self->Current(g) ||
                                !(g->key == requestedKey) ||
                                g->attachmentEpoch != attachmentEpoch) {
                                self->Detach(session, requestedKey, seq,
                                             std::move(replies->sourceDetach),
                                             std::move(replies->detachAck));
                                self->Release(seq);
                                return;
                            }
                            g->session = session;
                            g->replies = replies;
                            replies->sourceRegister = session->ReserveLifecycleCommand();
                            replies->sourceUnregister = session->ReserveLifecycleCommand();
                            if (!replies->sourceRegister || !replies->sourceUnregister) {
                                g->replies.reset();
                                g->session = {};
                                self->Detach(session, requestedKey, seq,
                                             std::move(replies->sourceDetach),
                                             std::move(replies->detachAck));
                                self->Reveal(g);
                                self->Release(seq);
                                return;
                            }
                            std::weak_ptr<_State> weak(self);
                            std::weak_ptr<Groom> groom(g);
                            TfWeakPtr<Session> weakSession(session);
                            Pipeline::CommandMailbox mailbox = replies->republish;
                            Pipeline::CommandMailbox progressMailbox = replies->progress;
                            Session::ProgressCallback progressCallback;
                            if (self->progressiveRenderer && progressMailbox) progressCallback =
                                [weak, groom, weakSession, attachmentEpoch,
                                 progressMailbox](Session::ProgressPayload const& payload) {
                                    auto state = weak.lock();
                                    auto member = groom.lock();
                                    SessionHandle sourceSession =
                                        TfCreateRefPtrFromProtectedWeakPtr(weakSession);
                                    if (!state || !member || !sourceSession ||
                                        state->closing.load()) return;
                                    (void)state->owner->PostLatestCommand(progressMailbox,
                                        [state, member, sourceSession,
                                         attachmentEpoch, payload] {
                                            state->PublishProgress(member, sourceSession,
                                                                   attachmentEpoch, payload);
                                        });
                                };
                            g->callback = session->RegisterRepublishCallback(
                                std::move(replies->sourceRegister),
                                [weak, groom, weakSession, attachmentEpoch, mailbox](
                                    Session::CommitPayload const& payload) {
                                    auto state = weak.lock();
                                    auto member = groom.lock();
                                    SessionHandle sourceSession =
                                        TfCreateRefPtrFromProtectedWeakPtr(weakSession);
                                    if (!state || !member || !sourceSession ||
                                        state->closing.load()) return;
                                    (void)state->owner->PostLatestCommand(mailbox,
                                        [state, member, sourceSession,
                                         attachmentEpoch, payload] {
                                        state->Publish(member, sourceSession,
                                                       attachmentEpoch, payload);
                                    });
                                }, {}, std::move(progressCallback));
                            self->Cook(g, seq);
                            self->Release(seq);
                        } catch (...) { std::terminate(); }
                    }, [self, seq] { self->Release(seq); });
                });
        } catch (...) { TF_WARN("usdGen scene attachment request failed"); }
        if (!accepted) { Reveal(g); Release(seq); }
    }
    void Apply(_Ingress const& packet) {
        const uint64_t seq = packet.sequence;
        if (packet.failed && seq >= captureTrustSequence) {
            captureTrusted = false;
            captureTrustSequence = seq;
        }
        if (closing.load() || packet.failed) { CompleteIngress(seq); return; }
        if (packet.fullPopulation && seq >= captureTrustSequence) {
            captureTrusted = true;
            captureTrustSequence = seq;
        }
        Added forwardAdded;
        Removed forwardRemoved;
        Dirtied forwardDirtied;
        for (auto const& e : packet.removed) {
            if (NewerEvent(e.primPath, seq)) continue;
            events[e.primPath] = std::max(events[e.primPath], seq);
            tombstones[e.primPath] = std::max(tombstones[e.primPath], seq);
            for (auto it = members.begin(); it != members.end();) {
                if (it->first.HasPrefix(e.primPath)) {
                    Remove(it->second, seq);
                    it = members.erase(it);
                } else ++it;
            }
            forwardRemoved.push_back(e);
        }
        for (auto const& e : packet.added) {
            if (NewerEvent(e.primPath, seq)) continue;
            events[e.primPath] = std::max(events[e.primPath], seq);
            forwardAdded.push_back(e);
        }
        for (auto const& e : packet.dirtied) {
            if (NewerEvent(e.primPath, seq)) continue;
            events[e.primPath] = std::max(events[e.primPath], seq);
            forwardDirtied.push_back(e);
        }
        if (packet.sourceFull) {
            if (seq < sourceThrough) {
                deferredFullCapture.store(true, std::memory_order_release);
            } else {
                std::map<SdfPath, Snapshot::SourceValue> current;
                for (auto const& entry : packet.source) {
                    auto old = source->find(entry.path);
                    const bool unchanged = old != source->end() && old->second.type == entry.type;
                    // A recovery is the authoritative replacement for notices
                    // that may have been dropped while ingress was full.  Give
                    // every surviving value a new stamp; ordinary full
                    // discovery preserves stable values.
                    current.emplace(entry.path,
                        unchanged && !packet.recoverSourceNamespace ?
                            old->second : Snapshot::SourceValue{entry.type, seq});
                }
                if (packet.recoverSourceNamespace && sourceKnown) {
                    for (auto const& old : *source) {
                        if (current.count(old.first) && current[old.first].type == old.second.type) continue;
                        bool covered = false;
                        for (SdfPath p = old.first.GetParentPath(); !p.IsEmpty(); p = p.GetParentPath()) {
                            auto was = source->find(p);
                            auto now = current.find(p);
                            if (was != source->end() && (now == current.end() || now->second.type != was->second.type))
                                { covered = true; break; }
                        }
                        if (!covered) forwardRemoved.emplace_back(old.first);
                    }
                    for (auto const& now : current) {
                        auto old = source->find(now.first);
                        bool changedAncestor = false;
                        for (SdfPath p = now.first.GetParentPath(); !p.IsEmpty(); p = p.GetParentPath()) {
                            auto was = source->find(p);
                            auto is = current.find(p);
                            if (was != source->end() && (is == current.end() || is->second.type != was->second.type))
                                { changedAncestor = true; break; }
                        }
                        if (old == source->end() || old->second.type != now.second.type || changedAncestor)
                            forwardAdded.emplace_back(now.first, now.second.type);
                    }
                }
                source = std::make_shared<const std::map<SdfPath, Snapshot::SourceValue>>(std::move(current));
                sourceKnown = true; sourceThrough = seq;
                // Root is an implicit namespace value and can never appear
                // as Added/Removed.  A full authoritative capture therefore
                // carries its own root dirty stamp, including initial state.
                rootValue = Snapshot::SourceValue{TfToken(), seq};
            }
        } else if (!forwardAdded.empty() || !forwardRemoved.empty()) {
            if (sourceKnown) {
                auto next = std::make_shared<std::map<SdfPath, Snapshot::SourceValue>>(*source);
                for (auto const& removed : forwardRemoved)
                    for (auto it = next->begin(); it != next->end();)
                        if (it->first.HasPrefix(removed.primPath)) it = next->erase(it); else ++it;
                for (auto const& added : forwardAdded)
                    (*next)[added.primPath] = {added.primType, seq};
                source = std::static_pointer_cast<const std::map<SdfPath, Snapshot::SourceValue>>(next);
            }
            sourceThrough = std::max(sourceThrough, seq);
        }
        // A full traversal establishes namespace, but the same ingress can
        // also carry an ordinary same-type Added notice.  Preserve the
        // recovered stable stamp unless that explicit value event says the
        // path changed.
        if (packet.sourceFull && sourceKnown && !forwardAdded.empty()) {
            auto next = std::make_shared<std::map<SdfPath, Snapshot::SourceValue>>(*source);
            bool changed = false;
            for (auto const& added : forwardAdded) {
                auto it = next->find(added.primPath);
                if (it != next->end()) {
                    it->second = Snapshot::SourceValue{it->second.type, seq};
                    changed = true;
                }
            }
            if (changed)
                source = std::static_pointer_cast<const std::map<SdfPath, Snapshot::SourceValue>>(next);
        }
        if (sourceKnown && !forwardDirtied.empty()) {
            auto next = std::make_shared<std::map<SdfPath, Snapshot::SourceValue>>(*source);
            bool changed = false;
            // The frontend diffs against the snapshot it displays, so keep
            // the input's own locators for it rather than a universal dirty
            // (an animated mesh would otherwise re-sync its topology and
            // every primvar each frame). A stale read of `shown` only makes
            // the locators accumulate longer.
            auto const shown = VisibleSnapshot();
            // A delivered dirty (async pass-through) already reached every
            // downstream observer.  The stamp still advances -- a
            // Description's tiles inherit its displayStyle and selections by
            // that stamp, and those synthetic dirties stay on the drain -- but
            // no locators are left for the drain to forward again.  Dirties
            // recorded earlier and not yet shown are kept (Dirty accumulates).
            static HdDataSourceLocatorSet const delivered;
            for (auto const& dirty : forwardDirtied) {
                HdDataSourceLocatorSet const& locators =
                    packet.dirtiedDelivered ? delivered : dirty.dirtyLocators;
                if (dirty.primPath == SdfPath::AbsoluteRootPath()) {
                    rootValue.Dirty(seq, locators, shown->rootValue.stamp);
                    continue;
                }
                auto it = next->find(dirty.primPath);
                if (it == next->end()) continue;
                auto const was = shown->source->find(dirty.primPath);
                it->second.Dirty(seq, locators,
                                 was == shown->source->end() ? 0 : was->second.stamp);
                changed = true;
            }
            if (changed)
                source = std::static_pointer_cast<const std::map<SdfPath, Snapshot::SourceValue>>(next);
        }
        std::vector<std::shared_ptr<Groom>> startAttach, startCook;
        // A Hydra Added notice can resync an existing groom to a non-groom
        // type without an explicit Removed. Retire roots absent from this
        // authoritative capture, but never overwrite a newer ingress.
        for (auto it = members.begin(); it != members.end();) {
            bool found = std::any_of(packet.inputs.begin(), packet.inputs.end(),
                [&](auto const& input) { return input.root == it->first; });
            bool captured = packet.fullPopulation ||
                std::find(packet.captureRoots.begin(), packet.captureRoots.end(), it->first) !=
                    packet.captureRoots.end();
            if (captured && !found && it->second->captured <= seq && !NewerEvent(it->first, seq)) {
                if (!it->second->authoredRender)
                    forwardRemoved.emplace_back(RenderPath(it->second->description));
                Remove(it->second, seq);
                it = members.erase(it);
            } else ++it;
        }
        for (auto const& input : packet.inputs) {
            if (RemovedSince(input.root, seq) || NewerEvent(input.root, seq)) continue;
            auto it = members.find(input.root);
            if (it != members.end() && it->second->captured > seq) continue;
            if (it != members.end() && it->second->description != input.description) {
                if (!it->second->authoredRender)
                    forwardRemoved.emplace_back(RenderPath(it->second->description));
                Remove(it->second, seq);
                members.erase(it);
                it = members.end();
            }
            std::shared_ptr<Groom> groom;
            if (it == members.end()) {
                groom = std::make_shared<Groom>();
                groom->id = ++nextId;
                groom->root = input.root;
                groom->description = input.description;
                groom->key = input.key;
                groom->authoredRender = input.authoredRender;
                members.emplace(input.root, groom);
                if (!input.authoredRender)
                    forwardAdded.emplace_back(RenderPath(input.description), TfToken("scope"));
                // Adoption attaches the session even when dormant; the
                // attachment's cook then declines (Cook).
                startAttach.push_back(groom);
            } else {
                groom = it->second;
                if (input.dormant) {
                    // Nothing was captured, so the key is incomplete: keep
                    // the session, key and last-good tiles for the wake-up.
                } else if (!(groom->key == input.key)) {
                    // CUDA graphs use renderer-local keys even when their
                    // stock-Storm handoff is not implemented. Retain displayed
                    // CPU tiles while the replacement is rejected; old
                    // callbacks carry their old handle and are rejected.
                    ReleaseSession(groom, seq);
                    groom->key = input.key;
                    groom->sessionGeneration = -1;
                    startAttach.push_back(groom);
                } else if (groom->session) {
                    startCook.push_back(groom);
                } else {
                    startAttach.push_back(groom);
                }
            }
            groom->captured = seq;
            groom->dormant = input.dormant;
            if (input.dormant) {
                groom->hidden = true;
            } else {
                groom->desc = input.desc;
                groom->dependencies = input.dependencies;
                groom->cache = input.cache;
            }
            groom->device = packet.device;
            groom->frame = packet.frame;
        }
        // A recovered source type-change can remove an ancestor of a groom
        // while the groom and its retained synthetic render subtree survive.
        // Hydra treats that ancestor removal as removing all descendants, so
        // explicitly re-add the current synthetic scope/tiles in the same
        // packet.  Do not resurrect authored collisions or members retired
        // above by the authoritative full capture.
        auto addedAlready = [&forwardAdded](SdfPath const& path) {
            return std::any_of(forwardAdded.begin(), forwardAdded.end(),
                [&path](auto const& entry) { return entry.primPath == path; });
        };
        for (auto const& item : members) {
            auto const& groom = *item.second;
            SdfPath const render = RenderPath(groom.description);
            bool covered = std::any_of(forwardRemoved.begin(), forwardRemoved.end(),
                [&render](auto const& entry) { return render.HasPrefix(entry.primPath); });
            if (!covered) continue;
            if (!groom.authoredRender && !addedAlready(render))
                forwardAdded.emplace_back(render, TfToken("scope"));
            for (auto const& tile : *groom.tiles)
                if (!source->count(tile.first) && !addedAlready(tile.first))
                    forwardAdded.emplace_back(tile.first, TfToken("basisCurves"));
        }
        // Source recovery may overlap ordinary accepted entries. Keep the
        // first (source) addition for a path, then order parents before
        // children.  Removal duplication is similarly collapsed.
        Added uniqueAdded;
        std::set<SdfPath> seenAdded, seenRemoved;
        for (auto const& entry : forwardAdded)
            if (seenAdded.insert(entry.primPath).second) uniqueAdded.push_back(entry);
        std::stable_sort(uniqueAdded.begin(), uniqueAdded.end(),
            [](auto const& a, auto const& b) {
                return a.primPath.GetPathElementCount() < b.primPath.GetPathElementCount();
            });
        forwardAdded = std::move(uniqueAdded);
        Removed uniqueRemoved;
        for (auto const& entry : forwardRemoved)
            if (seenRemoved.insert(entry.primPath).second) uniqueRemoved.push_back(entry);
        forwardRemoved = std::move(uniqueRemoved);
        Notify(forwardAdded, forwardRemoved, forwardDirtied);
        for (auto const& groom : startAttach) Attach(groom, seq);
        for (auto const& groom : startCook) Cook(groom, seq);
        CompleteIngress(seq);
    }
};

UsdGenSceneService::~UsdGenSceneService() {
    // Process shutdown has stopped public scene construction. Snapshot the
    // serial registry before closing states and disarming their records.
    auto entries = Snapshot();
    std::vector<std::shared_ptr<UsdGenGroomSceneIndex::_State>> live;
    std::vector<std::shared_ptr<RetirementRecord>> liveRecords;
    for (auto const& entry : entries) {
        if (auto state = entry.state.lock()) {
            live.push_back(std::move(state));
            liveRecords.push_back(entry.retirement);
        }
        else entry.retirement->Wait(); // includes final-ref -> enqueue gap
    }
    for (auto const& state : live) state->Close();
    for (auto const& state : live) state->Synchronize();
    ::usdGenImaging::UsdGenSessionStore::GetInstance().Drain();
    // Process shutdown only: finish source callback frames before destroying
    // the retirement service captured by a scene state's final deleter.
    std::vector<SessionHandle> sessions;
    for (auto const& state : live) state->owner->InvokeOwner([&] {
        for (auto const& weak : state->usedSessions)
            if (auto session = TfCreateRefPtrFromProtectedWeakPtr(weak))
                sessions.push_back(std::move(session));
    });
    for (auto const& session : sessions) session->Shutdown();
    for (auto const& session : sessions) if (session->Engine()) session->Engine()->Drain();
    for (size_t i = 0; i < live.size(); ++i) {
        auto const& state = live[i];
        state->owner->Shutdown();
        state->owner.reset();
        liveRecords[i]->Disarm();
        state->quiesced.store(true, std::memory_order_release);
    }
    live.clear();
    retirement.wait_for_all();
    // Cleanup deletion events enqueue their short registry erase only after
    // State destruction.  Public construction is stopped, so this exclusive
    // shutdown wait prevents member teardown from racing a late erase.
    registryGraph.wait_for_all();
    sessions.clear();
    Session::DrainRetired();
}
void UsdGenSceneService::DrainRetired() {
    if (Pipeline::IsExecuting()) throw std::logic_error("scene callback cannot drain retirement");
    // The serial registry snapshot is causally after registrations accepted
    // before this external boundary. Later producers remain independent.
    auto entries = Snapshot();
    std::vector<std::shared_ptr<UsdGenGroomSceneIndex::_State>> retired;
    std::vector<std::shared_ptr<RetirementRecord>> records;
    std::set<RetirementRecord*> expired;
    for (auto const& entry : entries) {
        if (auto state = entry.state.lock()) {
            if (state->closing.load()) {
                retired.push_back(std::move(state));
                records.push_back(entry.retirement);
            }
        } else {
            // The State may have lost its last reference before its custom
            // deleter could enqueue cleanup. Its record bridges that gap.
            records.push_back(entry.retirement);
            expired.insert(entry.retirement.get());
        }
    }
    for (auto const& state : retired) state->Synchronize();
    // Release the references captured above before waiting: their final
    // release can be the producer that enqueues retirement cleanup.
    retired.clear();
    // Do not call retirement.wait_for_all here. The pinned TBB graph resets
    // its root count after a wait, so a concurrent final-reference producer
    // can race a global drain. Each record instead waits through exactly its
    // State's final-reference -> enqueue -> deletion completion boundary.
    for (auto const& record : records) {
        if (expired.count(record.get()) && record->pending.load(std::memory_order_acquire))
            if (auto gate = std::atomic_load(&s_testDrainWaitGate)) gate->Signal();
        record->Wait();
    }
}

UsdGenGroomSceneIndex::UsdGenGroomSceneIndex(HdSceneIndexBaseRefPtr const& input, int id,
                                         bool publishScalpShadow, bool stormMaterialPolicy)
    : HdSingleInputFilteringSceneIndexBase(input),
      _pruned(HdSiExtComputationPrimvarPruningSceneIndex::New(input)),
      _renderInstanceId(static_cast<uint32_t>(id)),
      _publishScalpShadow(publishScalpShadow),
      _stormMaterialPolicy(stormMaterialPolicy) {
    static std::atomic<uint64_t> next{uint64_t(1) << 32};
    if (id == 0) _renderInstanceId = next.fetch_add(1);
    auto& service = SceneService();
    auto record = std::make_shared<UsdGenSceneService::RetirementRecord>();
    _state = std::shared_ptr<_State>(new _State(service.runtime, stormMaterialPolicy),
        [&service, record](_State* state) {
        if (state->quiesced.load(std::memory_order_acquire)) {
            // A static public handle may outlive the process service. All
            // pipelines/subscriptions have already been removed externally.
            delete state;
        } else {
            if (auto gate = std::atomic_load(&s_testFinalDeleterGate);
                gate && !gate->claimed.exchange(true)) gate->Pause();
            service.Retire([&service, state, record] {
                delete state;
                service.Erase(record);
            });
        }
    });
    // The accepted serial registration precedes any later external snapshot
    // causally issued by this caller.  It retains the record through the
    // final-reference -> cleanup-enqueue gap.
    service.Register(UsdGenSceneService::Entry{_state, record});
}
HdSceneIndexBaseRefPtr UsdGenGroomSceneIndex::New(HdSceneIndexBaseRefPtr const& input, int id) {
    return New(input, id, true);
}
HdSceneIndexBaseRefPtr UsdGenGroomSceneIndex::New(HdSceneIndexBaseRefPtr const& input,
                                               int id, bool publishScalpShadow) {
    return New(input, id, publishScalpShadow, true);
}
HdSceneIndexBaseRefPtr UsdGenGroomSceneIndex::New(HdSceneIndexBaseRefPtr const& input,
                                               int id, bool publishScalpShadow,
                                               bool stormMaterialPolicy) {
    auto index = TfCreateRefPtr(new UsdGenGroomSceneIndex(
        input, id, publishScalpShadow, stormMaterialPolicy));
    index->_state->recipient = TfCreateWeakPtr(index.operator->());
    UsdGenImagingTestHook::_RegisterIndex(index.operator->());
    _Ingress initial;
    initial.initial = true;
    index->_CaptureAndSubmit(std::move(initial));
    return index;
}
UsdGenGroomSceneIndex::~UsdGenGroomSceneIndex() {
    _state->Close();
    if (!_state->quiesced.load()) UsdGenImagingTestHook::_UnregisterIndex(this);
}
void UsdGenGroomSceneIndex::_DrainPublications(bool waitForIngress, bool explicitWait) {
    // HdSceneIndex frontend entry points are serialized by Hydra.  Query
    // methods remain concurrent immutable snapshot reads; do not add an
    // application mutex here.
    if (_dispatching) {
        if (explicitWait)
            throw std::logic_error("Synchronize cannot re-enter from a scene-index notice");
        if (waitForIngress) _deferredSynchronousFlush = true;
        return;
    }
    auto live = TfCreateRefPtrFromProtectedWeakPtr(_state->recipient);
    auto *const index = dynamic_cast<UsdGenGroomSceneIndex *>(
        live ? live.operator->() : nullptr);
    if (!index) return;

    _dispatching = true;
    try {
        do {
            if (waitForIngress) {
                TRACE_SCOPE("usdGen wait for capture, cook and publish");
                _state->Synchronize();
            }
            _deferredSynchronousFlush = false;
            // A poll consumes precisely one newest target.  Snapshot diffing
            // here (rather than retaining owner-side notice history) keeps
            // presentation bounded even when a renderer stops polling.
            auto target = std::atomic_exchange(&_state->pending,
                std::shared_ptr<const _State::Snapshot>());
            if (target) {
                TRACE_SCOPE("usdGen diff snapshot and notify Hydra");
                // The notice lists outlive the diff scope: the notify scope
                // below sends them.
                Added added;
                Removed removed;
                Dirtied dirtied;
                // Staged-out retired snapshot (see the move below): its
                // destruction runs on the background releaser.
                std::shared_ptr<const _State::Snapshot> retired;
                {
                TRACE_SCOPE("usdGen drain: diff snapshots");
                auto before = _state->VisibleSnapshot();
                struct Synthetic {
                    TfToken type;
                    uint64_t rootId = 0;
                    int64_t generation = -1;
                    std::shared_ptr<const TileMap> tiles;
                    // Stamp of the owning UsdGenDescription in the source
                    // map.  GetPrim overlays that prim's upstream
                    // displayStyle and selections onto every tile, and the
                    // indices that author them
                    // (HdsiLegacyDisplayStyleOverrideSceneIndex for the
                    // usdview complexity slider,
                    // UsdImagingSelectionSceneIndex for the highlight) sit
                    // upstream of this one and dirty only stage prims. A
                    // change of this stamp is therefore what invalidates the
                    // inherited containers on the tiles.
                    uint64_t descriptionStamp = 0;
                    // Digest of the usdGen:look:* fields the synthetic default
                    // material carries. GetPrim builds that material from the
                    // path plus this look, so a look edit that changes nothing
                    // else would otherwise never be announced: the material is
                    // an Sprim Hydra only learns about from this diff, and the
                    // branch below deliberately leaves path-built prims alone
                    // across a new generation.
                    uint64_t lookDigest = 0;
                    // Digest of the scalp-shadow cap's geometry and depths.
                    // The cap is a mesh built from the View, not from its
                    // path, so like the material it needs its own identity
                    // here or a rebaked cap would never reach Hydra.
                    uint64_t scalpDigest = 0;
                    // ... and the cap itself, so the diff can name exactly
                    // what moved instead of dirtying the prim universally: a
                    // deforming groom rebakes it every frame, and a universal
                    // dirty there costs a full re-sync per frame.
                    HdContainerDataSourceHandle scalp;
                    // The groom is hidden: GetPrim overlays visibility false
                    // on its tiles and cap, so a flip is a visibility dirty.
                    bool hidden = false;
                };
                std::map<SdfPath, TfToken> beforeNames, targetNames;
                std::map<SdfPath, Synthetic> beforeSynthetic, targetSynthetic;
                auto collect = [this](std::shared_ptr<const _State::Snapshot> const& snapshot,
                                  std::map<SdfPath, TfToken>& names,
                                  std::map<SdfPath, Synthetic>& synthetic) {
                    for (auto const& source : *snapshot->source)
                        names.emplace(source.first, source.second.type);
                    for (auto const& g : snapshot->members) {
                        // The source map is authoritative for collisions.
                        SdfPath const render = RenderPath(g.description);
                        auto const source = snapshot->source->find(g.description);
                        uint64_t const descStamp =
                            source == snapshot->source->end() ? 0 : source->second.stamp;
                        if (!names.count(render)) {
                            names.emplace(render, TfToken("scope"));
                            synthetic.emplace(render, Synthetic{TfToken("scope"),
                                g.id, g.generation, g.tiles, descStamp});
                        }
                        // The synthetic default material every tile binds
                        // when the description authors none (06 §4.4). It is
                        // an Sprim, so Hydra only ever learns about it from
                        // this diff.
                        SdfPath const material = MaterialPath(g.description);
                        if (!names.count(material)) {
                            names.emplace(material, TfToken("material"));
                            synthetic.emplace(material, Synthetic{TfToken("material"),
                                g.id, g.generation, g.tiles, descStamp,
                                ::usdGenImaging::UsdGenTilePublisher::
                                    DefaultMaterialLookDigest(g.look)});
                        }
                        // The value-preview material, while the tiles bind it.
                        SdfPath const preview = BoundPreviewMaterial(g.description, *g.tiles);
                        if (!preview.IsEmpty() && !names.count(preview)) {
                            names.emplace(preview, TfToken("material"));
                            synthetic.emplace(preview, Synthetic{TfToken("material"),
                                g.id, g.generation, g.tiles, descStamp});
                        }
                        // The scalp-shadow cap and the material it binds, both
                        // present only while the groom has one.
                        if (_publishScalpShadow && g.scalpShadow) {
                            SdfPath const cap = ScalpShadowPath(g.description);
                            if (!names.count(cap)) {
                                names.emplace(cap, TfToken("mesh"));
                                synthetic.emplace(cap, Synthetic{TfToken("mesh"),
                                    g.id, g.generation, g.tiles, descStamp, 0,
                                    g.scalpDigest, g.scalpShadow, g.hidden});
                            }
                            SdfPath const capMaterial =
                                ScalpShadowMaterialPath(g.description);
                            if (!names.count(capMaterial)) {
                                names.emplace(capMaterial, TfToken("material"));
                                synthetic.emplace(capMaterial, Synthetic{
                                    TfToken("material"), g.id, g.generation,
                                    g.tiles, descStamp,
                                    ::usdGenImaging::UsdGenTilePublisher::
                                        DefaultMaterialLookDigest(g.look)});
                            }
                        }
                        for (auto const& tile : *g.tiles) if (!names.count(tile.first)) {
                            names.emplace(tile.first, TfToken("basisCurves"));
                            synthetic.emplace(tile.first, Synthetic{TfToken("basisCurves"),
                                g.id, g.generation, g.tiles, descStamp, 0, 0, nullptr,
                                g.hidden});
                        }
                    }
                };
                { TRACE_SCOPE("usdGen drain: collect");
                collect(before, beforeNames, beforeSynthetic);
                collect(target, targetNames, targetSynthetic);
                }
                std::set<SdfPath> removedPaths;
                for (auto const& old : beforeNames) {
                    auto now = targetNames.find(old.first);
                    const bool oldSource = before->source->count(old.first);
                    const bool nowSource = target->source->count(old.first);
                    if (now == targetNames.end() || now->second != old.second || oldSource != nowSource) {
                        if (old.first != SdfPath::AbsoluteRootPath()) {
                            removed.emplace_back(old.first);
                            removedPaths.insert(old.first);
                        }
                    }
                }
                auto removedAncestor = [&removedPaths](SdfPath const& path) {
                    for (SdfPath parent = path.GetParentPath(); !parent.IsEmpty();
                         parent = parent.GetParentPath())
                        if (removedPaths.count(parent)) return true;
                    return false;
                };
                // Tile/scalp content diffs are pure (they read only immutable
                // data-source handles) and dominate the drain on deform frames
                // (tens of ms of VtArray compares while the cook workers sit
                // idle), so run them on the TBB arena ahead of the serial
                // notice merge. The gates below mirror the main loop's diff
                // call conditions exactly; the loop looks each result up by
                // prim path and falls back to the serial diff if a gate ever
                // skews (same notices, just slower).
                struct _PendingDiff {
                    SdfPath path;
                    HdDataSourceBaseHandle before, after;
                };
                std::vector<_PendingDiff> pendingDiffs;
                std::vector<HdDataSourceLocatorSet> pendingResults;
                { TRACE_SCOPE("usdGen drain: content diffs");
                for (auto const& now : targetNames) {
                    if (now.first == SdfPath::AbsoluteRootPath()) continue;
                    auto old = beforeNames.find(now.first);
                    if (old == beforeNames.end() || old->second != now.second ||
                        removedPaths.count(now.first) || removedAncestor(now.first))
                        continue;
                    auto targetSource = target->source->find(now.first);
                    auto beforeSource = before->source->find(now.first);
                    if (targetSource != target->source->end() &&
                        beforeSource != before->source->end())
                        continue;
                    auto const& a = beforeSynthetic[now.first];
                    auto const& b = targetSynthetic[now.first];
                    bool const sameRoot = a.rootId == b.rootId;
                    HdContainerDataSourceHandle beforeTile, targetTile;
                    if (sameRoot && b.type == TfToken("basisCurves") &&
                        a.tiles && b.tiles) {
                        auto ia = a.tiles->find(now.first);
                        auto ib = b.tiles->find(now.first);
                        if (ia != a.tiles->end() && ib != b.tiles->end()) {
                            beforeTile = ia->second;
                            targetTile = ib->second;
                        }
                    }
                    if (beforeTile && targetTile) {
                        if (beforeTile != targetTile)
                            pendingDiffs.push_back(
                                _PendingDiff{now.first, beforeTile, targetTile});
                    } else if (b.type != TfToken("basisCurves")) {
                        if (!sameRoot || a.lookDigest != b.lookDigest) {
                            // Universal dirty below; no content diff.
                        } else if (b.type == TfToken("mesh") &&
                                   a.scalpDigest != b.scalpDigest) {
                            pendingDiffs.push_back(
                                _PendingDiff{now.first, a.scalp, b.scalp});
                        }
                    }
                }
                pendingResults.resize(pendingDiffs.size());
                { TRACE_SCOPE("usdGen drain: content compare");
                if (pendingDiffs.size() > 1) {
                    tbb::parallel_for(
                        tbb::blocked_range<size_t>(0, pendingDiffs.size()),
                        [&](tbb::blocked_range<size_t> const& range) {
                            for (size_t i = range.begin(); i != range.end(); ++i) {
                                DiffTileDataSources(pendingDiffs[i].before,
                                                    pendingDiffs[i].after,
                                                    HdDataSourceLocator(),
                                                    &pendingResults[i]);
                            }
                        });
                } else if (pendingDiffs.size() == 1) {
                    DiffTileDataSources(pendingDiffs[0].before,
                                        pendingDiffs[0].after,
                                        HdDataSourceLocator(),
                                        &pendingResults[0]);
                }
                } /* drain: content compare */
                } /* drain: content diffs */
                { TRACE_SCOPE("usdGen drain: merge");
                std::map<SdfPath, HdDataSourceLocatorSet> precomputedDiffs;
                for (size_t i = 0; i < pendingDiffs.size(); ++i)
                    precomputedDiffs.emplace(pendingDiffs[i].path,
                                             std::move(pendingResults[i]));
                for (auto const& now : targetNames) {
                    if (now.first == SdfPath::AbsoluteRootPath()) continue;
                    auto old = beforeNames.find(now.first);
                    if (old == beforeNames.end() || old->second != now.second ||
                        removedPaths.count(now.first) || removedAncestor(now.first)) {
                        added.emplace_back(now.first, now.second);
                        continue;
                    }
                    auto targetSource = target->source->find(now.first);
                    auto beforeSource = before->source->find(now.first);
                    if (targetSource != target->source->end() && beforeSource != before->source->end()) {
                        if (targetSource->second.stamp != beforeSource->second.stamp) {
                            HdDataSourceLocatorSet const& locators =
                                targetSource->second.DirtySince(beforeSource->second.stamp);
                            if (!locators.IsEmpty()) dirtied.emplace_back(now.first, locators);
                        }
                    } else {
                        auto const& a = beforeSynthetic[now.first];
                        auto const& b = targetSynthetic[now.first];
                        bool const sameRoot = a.rootId == b.rootId;
                        HdContainerDataSourceHandle beforeTile, targetTile;
                        if (sameRoot && b.type == TfToken("basisCurves") && a.tiles && b.tiles) {
                            auto ia = a.tiles->find(now.first);
                            auto ib = b.tiles->find(now.first);
                            if (ia != a.tiles->end() && ib != b.tiles->end()) {
                                beforeTile = ia->second;
                                targetTile = ib->second;
                            }
                        }
                        if (beforeTile && targetTile) {
                            // A republished tile dirties only what differs, so
                            // a renderer re-syncs (and, for pinned cubic
                            // curves, re-interpolates) just the primvars that
                            // moved, not widths/hairT/colours that did not.
                            if (beforeTile != targetTile) {
                                auto const found =
                                    precomputedDiffs.find(now.first);
                                if (found != precomputedDiffs.end()) {
                                    if (!found->second.IsEmpty())
                                        dirtied.emplace_back(now.first,
                                                             found->second);
                                } else {
                                    // Defensive: the pre-pass gates mirror
                                    // these call conditions, so this only runs
                                    // if they ever skew — same notices.
                                    HdDataSourceLocatorSet changed;
                                    DiffTileDataSources(beforeTile, targetTile,
                                                        HdDataSourceLocator(),
                                                        &changed);
                                    if (!changed.IsEmpty())
                                        dirtied.emplace_back(now.first, changed);
                                }
                            }
                            if (a.descriptionStamp != b.descriptionStamp)
                                dirtied.emplace_back(now.first,
                                                     InheritedFromDescriptionLocators());
                        } else if (b.type != TfToken("basisCurves")) {
                            // The render scope and the value-preview material
                            // are built from their path alone (GetPrim), so a
                            // new generation leaves them as they were; only a
                            // replaced groom announces them again. The default
                            // hair material also carries the description's
                            // look, so a look edit has to announce it too.
                            if (!sameRoot || a.lookDigest != b.lookDigest) {
                                dirtied.emplace_back(now.first, HdDataSourceLocatorSet::UniversalSet());
                            } else if (b.type == TfToken("mesh") &&
                                       a.scalpDigest != b.scalpDigest) {
                                // The scalp-shadow cap is rebaked whenever the
                                // groom deforms, so it is diffed like a tile
                                // rather than dirtied universally: a universal
                                // dirty here would cost a full mesh re-sync
                                // every frame of playback. The diff ran on the
                                // pre-pass workers; the serial call below is
                                // the same skew-only fallback as the tiles'.
                                auto const found =
                                    precomputedDiffs.find(now.first);
                                if (found != precomputedDiffs.end()) {
                                    if (!found->second.IsEmpty())
                                        dirtied.emplace_back(now.first,
                                                             found->second);
                                } else {
                                    HdDataSourceLocatorSet changed;
                                    DiffTileDataSources(a.scalp, b.scalp,
                                                        HdDataSourceLocator(),
                                                        &changed);
                                    if (!changed.IsEmpty())
                                        dirtied.emplace_back(now.first, changed);
                                }
                            }
                        } else if (!sameRoot || a.generation != b.generation || a.tiles != b.tiles) {
                            dirtied.emplace_back(now.first, HdDataSourceLocatorSet::UniversalSet());
                            // HdSceneIndexAdapterSceneDelegate keeps a prim's
                            // cached primvar descriptors across a universal
                            // dirty and drops them only for an explicit
                            // primvars locator (sceneIndexAdapterSceneDelegate
                            // .cpp PrimsDirtied). Without this a primvar that
                            // appears, or changes interpolation (displayColor
                            // per curve <-> per CV under usdGen:preview or the
                            // look's bakeMode), is read with its old
                            // descriptor.
                            if (b.type == TfToken("basisCurves"))
                                dirtied.emplace_back(now.first, HdDataSourceLocatorSet{
                                    HdPrimvarsSchema::GetDefaultLocator()});
                        }
                        else if (a.descriptionStamp != b.descriptionStamp &&
                                 b.type == TfToken("basisCurves"))
                            // The owning Description changed upstream: the
                            // displayStyle, selections and materialBindings
                            // GetPrim overlays on the tile may have moved with
                            // it. The stamp is a scalar, so dirty all three
                            // inherited containers rather than guess; each is
                            // a cheap state flag (DirtyDisplayStyle /
                            // DirtyMaterialId / the selection tracker) and
                            // never re-uploads geometry.
                            dirtied.emplace_back(now.first,
                                                 InheritedFromDescriptionLocators());
                        // Hiding or revealing a groom moves only the
                        // visibility GetPrim overlays on its geometry. A
                        // replaced root was dirtied universally above.
                        if (sameRoot && a.hidden != b.hidden &&
                            (b.type == TfToken("basisCurves") || b.type == TfToken("mesh")))
                            dirtied.emplace_back(now.first, HdDataSourceLocatorSet{
                                HdVisibilitySchema::GetDefaultLocator()});
                    }
                }
                // Root is never structural, but a root value notice remains
                // observable, with the input's locators when the displayed
                // snapshot allows it (the scene globals' current frame).
                if (before->rootValue.stamp != target->rootValue.stamp) {
                    HdDataSourceLocatorSet const& locators =
                        target->rootValue.DirtySince(before->rootValue.stamp);
                    if (!locators.IsEmpty())
                        dirtied.emplace_back(SdfPath::AbsoluteRootPath(), locators);
                }
                // A removed ancestor already removes its descendants.  Keep
                // the complete set above solely to identify the descendants
                // that must be re-added from the target snapshot.
                Removed collapsedRemoved;
                for (auto const& entry : removed) {
                    bool covered = false;
                    for (SdfPath parent = entry.primPath.GetParentPath(); !parent.IsEmpty();
                         parent = parent.GetParentPath())
                        if (removedPaths.count(parent)) { covered = true; break; }
                    if (!covered) collapsedRemoved.push_back(entry);
                }
                removed = std::move(collapsedRemoved);
                std::stable_sort(added.begin(), added.end(), [](auto const& a, auto const& b) {
                    return a.primPath.GetPathElementCount() < b.primPath.GetPathElementCount();
                });
                std::atomic_store(&_state->visible, target);
                } /* drain: merge */
                // The retired visible snapshot pins the previous frame's
                // tiles, captures and scalp cap; destroying it here costs
                // tens of ms of main-thread frees, so stage it out for the
                // background releaser. Everything else in this block either
                // references the new snapshot (pinned by `visible` above, so
                // only reference counts drop here) or is plain keys, and
                // still dies here cheaply.
                retired = std::move(before);
                }
                _state->ReleaseRetired(std::move(retired));
                {
                TRACE_SCOPE("usdGen drain: notify Hydra");
                try { if (!removed.empty()) index->_SendPrimsRemoved(removed); }
                catch (...) { TF_WARN("usdGen removal observer threw"); }
                try { if (!added.empty()) index->_SendPrimsAdded(added); }
                catch (...) { TF_WARN("usdGen addition observer threw"); }
                try { if (!dirtied.empty()) index->_SendPrimsDirtied(dirtied); }
                catch (...) { TF_WARN("usdGen dirty observer threw"); }
                }
            }
            // An observer can synchronously cause another input ingress.  It
            // cannot wait recursively; its causal flush belongs to this outer
            // frontend turn and remains FIFO after the current packet.
            waitForIngress = _deferredSynchronousFlush;
        } while (waitForIngress);
    } catch (...) {
        _dispatching = false;
        throw;
    }
    _dispatching = false;
}
void UsdGenGroomSceneIndex::Synchronize() {
    // An observer invoked by the drain may release the caller's final public
    // reference.  Keep this frontend object alive until this method returns.
    auto live = TfCreateRefPtrFromProtectedWeakPtr(_state->recipient);
    if (!live) return;
    _DrainPublications(true, true);
    // Retry one authoritative source-namespace and synthetic discovery after
    // pressure deferred an ingress.
    // This external frontend boundary, unlike asyncPoll, may capture/start
    // owner work.
    if (_state->deferredFullCapture.load(std::memory_order_acquire)) {
        _Ingress recovery;
        recovery.forceFullDiscovery = true;
        recovery.recoverSourceNamespace = true;
        _CaptureAndSubmit(std::move(recovery));
        _DrainPublications(true, true);
    }
}
void UsdGenGroomSceneIndex::DrainRetired() { SceneService().DrainRetired(); }

void UsdGenGroomSceneIndex::_CaptureAndSubmit(_Ingress packet) {
    TRACE_FUNCTION();
    // A synchronous default-mode drain below can re-enter arbitrary client
    // code. Pin the public index for the entire caller-boundary operation.
    auto live = TfCreateRefPtrFromProtectedWeakPtr(_state->recipient);
    if (!live) return;
    auto state = _state;
    if (state->closing.load()) return;
    // Admission precedes sequence allocation and caller-boundary capture.
    // At capacity, record that an authoritative synthetic rescan is owed but
    // never leave a watermark for work that was not admitted.  Recovering
    // complete upstream notice forwarding requires a namespace resync, not
    // eager forwarding against the old synthetic snapshot.
    auto ingressTicket = std::make_shared<Pipeline::CommandTicket>(
        state->owner->ReserveCommandTicket());
    if (!*ingressTicket) {
        state->deferredFullCapture.store(true, std::memory_order_release);
        return;
    }
    packet.sequence = state->sequences.TryIssue();
    if (!packet.sequence) {
        // The reserved command credit is released by its RAII ticket.  Do
        // not consume recovery state or touch caller input without a bounded
        // sequence slot to eventually complete.
        state->deferredFullCapture.store(true, std::memory_order_release);
        return;
    }
    state->captureCount.fetch_add(1, std::memory_order_acq_rel);
    // Consume the pressure latch only after this command owns admission. A
    // later rejection can set it again while this capture runs.
    if (state->deferredFullCapture.exchange(false, std::memory_order_acq_rel)) {
        packet.forceFullDiscovery = true;
        packet.recoverSourceNamespace = true;
    }
    packet.sourceFull = packet.forceFullDiscovery || packet.initial;
    packet.device = usdGen::UsdGenSession::CaptureCallerDevice();
    try {
        // Use only an owner-published complete catalog. An older in-flight
        // capture may establish new external dependencies, so fall back to
        // full discovery until its ingress has been applied.
        auto catalog = state->SnapshotValue();
        std::vector<SdfPath> stack;
        if (!packet.initial && !packet.forceFullDiscovery &&
            packet.added.empty() && packet.removed.empty() &&
            catalog->captureTrusted && catalog->capturedThrough == packet.sequence - 1) {
            packet.fullPopulation = false;
            for (auto const& member : catalog->members) {
                if (member.dormant) {
                    // Edits of a dormant groom's operators, surfaces and
                    // inputs, and frame changes, are no-ops. Only a change
                    // of its Description's resolved visibility can wake it.
                    if (std::any_of(packet.dirtied.begin(), packet.dirtied.end(),
                            [&member](auto const& dirty) {
                                return member.description.HasPrefix(dirty.primPath) &&
                                    TouchesVisibility(dirty.dirtyLocators);
                            }))
                        stack.push_back(member.root);
                    continue;
                }
                bool affected = !member.dependencies;
                for (auto const& dirty : packet.dirtied) {
                    auto const& path = dirty.primPath;
                    // The scene globals on the root carry the current frame.
                    // During playback they change after the stage time has
                    // already dirtied every animated input, so a frame change
                    // on its own only matters to a description that reads
                    // $frame or $time.
                    if (path.IsAbsoluteRootPath() &&
                        OnlySceneGlobals(dirty.dirtyLocators)) {
                        if (member.readsTime) affected = true;
                        if (affected) break;
                        continue;
                    }
                    if (path.HasPrefix(member.root) || member.root.HasPrefix(path)) affected = true;
                    if (member.dependencies) for (auto const& dependency : *member.dependencies)
                        // A dirty on an ancestor can change a resolved input.
                        // A sibling child of a queried metadata container
                        // cannot change that container's own sampled values.
                        if (dependency.HasPrefix(path)) affected = true;
                    if (affected) break;
                }
                if (affected) stack.push_back(member.root);
            }
            packet.captureRoots = stack;
            TF_DEBUG(USDGEN_INGRESS).Msg(
                "usdGen ingress   %zu dirty entries (first %s): %zu of %zu grooms affected\n",
                packet.dirtied.size(),
                packet.dirtied.empty() ? "-" : packet.dirtied.front().primPath.GetText(),
                stack.size(), catalog->members.size());
        } else {
            stack.push_back(SdfPath::AbsoluteRootPath());
            TF_DEBUG(USDGEN_INGRESS).Msg(
                "usdGen ingress   %zu dirty/%zu added/%zu removed entries: full discovery (%s)\n",
                packet.dirtied.size(), packet.added.size(), packet.removed.size(),
                packet.initial ? "initial" :
                packet.forceFullDiscovery ? "forced" :
                !packet.added.empty() || !packet.removed.empty() ? "namespace change" :
                !catalog->captureTrusted ? "catalog untrusted" :
                "an earlier ingress is not applied yet");
        }

        auto input = _GetInputSceneIndex();
        // GetPrim projects empty primType groom records through __usdPrimInfo.
        // Project corresponding Added records at this caller boundary so
        // frontend notices and queries use the same type.
        if (input) for (auto& added : packet.added) if (added.primType.IsEmpty()) {
            TfToken const projected = TypeName(input->GetPrim(added.primPath));
            if (IsGroom(projected)) added.primType = projected;
        }
        if (input && packet.sourceFull) {
            std::vector<SdfPath> sourceStack{SdfPath::AbsoluteRootPath()};
            while (!sourceStack.empty()) {
                SdfPath path = sourceStack.back(); sourceStack.pop_back();
                HdSceneIndexPrim prim = input->GetPrim(path);
                // The absolute root is a namespace anchor, not an emitted
                // scene prim; every descendant is recorded, including groom
                // internals that the cooking traversal intentionally skips.
                if (path != SdfPath::AbsoluteRootPath()) {
                    TfToken projected = prim.primType;
                    if (projected.IsEmpty()) {
                        TfToken const fallback = TypeName(prim);
                        if (IsGroom(fallback)) projected = fallback;
                    }
                    packet.source.push_back({path, projected});
                }
                auto children = input->GetChildPrimPaths(path);
                sourceStack.insert(sourceStack.end(), children.begin(), children.end());
            }
        }
        if (input && !stack.empty()) {
            auto frame = HdSceneGlobalsSchema::GetFromParent(
                input->GetPrim(SdfPath::AbsoluteRootPath()).dataSource).GetCurrentFrame();
            if (frame) {
                double value = frame->GetTypedValue(0);
                if (std::isfinite(value)) packet.frame = value;
            }
            while (!stack.empty()) {
                const auto path = stack.back(); stack.pop_back();
                const auto prim = input->GetPrim(path);
                if (path != SdfPath::AbsoluteRootPath() && IsGroom(TypeName(prim))) {
                    auto known = std::find_if(catalog->members.begin(), catalog->members.end(),
                        [&](auto const& member) { return member.root == path; });
                    _Ingress::Input captured;
                    captured.root = captured.description = path;
                    captured.key.groomRoot = path;
                    captured.key.renderInstanceId = _renderInstanceId;
                    auto id = HdStringDataSource::Cast(HdContainerDataSource::Get(
                        prim.dataSource, HdDataSourceLocator(TfToken("sessionId"))));
                    if (id) captured.key.sessionId = id->GetTypedValue(0);
                    if (!packet.fullPopulation && known != catalog->members.end()) {
                        // Dirty-only notices cannot change child prim types.
                        // Structural resyncs always take the discovery path.
                        captured.description = known->description;
                    } else if (TypeName(prim) == TfToken("UsdGenGroom")) {
                        for (auto const& child : input->GetChildPrimPaths(path))
                            if (TypeName(input->GetPrim(child)) == TfToken("UsdGenDescription")) {
                                captured.description = child; break;
                            }
                    }
                    auto authored = input->GetPrim(RenderPath(captured.description));
                    captured.authoredRender = authored.dataSource || !authored.primType.IsEmpty();
                    if (char const* const reason =
                            DormantReason(*input, prim, path, captured.description)) {
                        TF_DEBUG(USDGEN_INGRESS).Msg(
                            "usdGen ingress   %s is dormant (%s): capture and cook skipped\n",
                            path.GetText(), reason);
                        captured.dormant = true;
                        packet.inputs.push_back(std::move(captured));
                        continue;
                    }
                    RecordingInput recorder(_pruned);
                    ::usdGenImaging::UsdGenGraphDescBuildOptions options;
                    // Hydra samples are relative to the current stage frame:
                    // keep offset zero, and gate operator reuse with the
                    // absolute packet frame separately. Passing packet.frame
                    // as the offset here would sample at twice the current
                    // frame. Geometry has no procedural time dependence, so
                    // it reuses across frames while its prims stay clean. A
                    // groom waking from dormancy slept through dirties this
                    // packet does not carry, so it captures in full.
                    if (!packet.fullPopulation && known != catalog->members.end() &&
                        !known->dormant && known->cache) {
                        options.previousCache = known->cache;
                        for (auto const& dirty : packet.dirtied)
                            options.dirtyPrimPaths.push_back(dirty.primPath);
                        options.reuseGeometry = true;
                        if (known->frame == packet.frame)
                            options.reuseNodes = true;
                    }
                    auto const captureStart = std::chrono::steady_clock::now();
                    auto result = ::usdGenImaging::CaptureGraphDescFromHydra(
                        recorder, captured.description, options);
                    TF_DEBUG(USDGEN_INGRESS).Msg(
                        "usdGen ingress   capture %s at frame %g: %.2f ms (%s, operator reuse %s, geometry reuse %s)\n",
                        captured.description.GetText(), packet.frame,
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - captureStart).count(),
                        packet.fullPopulation ? "full population" : "dirty notice",
                        options.reuseNodes ? "on" :
                        known == catalog->members.end() ? "off: new groom" :
                        packet.fullPopulation ? "off: full population" :
                        known->dormant ? "off: waking from dormancy" :
                        "off: the frame changed",
                        options.reuseGeometry ? "on" :
                        known == catalog->members.end() ? "off: new groom" :
                        packet.fullPopulation ? "off: full population" :
                        known->dormant ? "off: waking from dormancy" :
                        "off: no previous cache");
                    captured.desc = std::make_shared<const Desc>(std::move(result.desc));
                    // A CUDA graph always has renderer-local session identity,
                    // even though this plugin does not yet hand its device
                    // buffers to stock Storm.
                    // CPU graphs retain existing explicit-session sharing.
                    captured.key.rendererLocal = captured.desc->executionBackend ==
                        usdGen::UsdGenExecutionBackend::Cuda;
                    captured.cache = std::move(result.cache);
                    captured.dependencies = recorder.Dependencies(*captured.desc);
                    if (!result.reusedInputs.empty()) {
                        // Reused inputs bypass Hydra reads, so the recorder
                        // never saw them; without this their next dirty
                        // would not recapture the groom.
                        auto merged = std::make_shared<SdfPathVector>(
                            *captured.dependencies);
                        merged->insert(merged->end(),
                                       result.reusedInputs.begin(),
                                       result.reusedInputs.end());
                        captured.dependencies = std::move(merged);
                    }
                    packet.inputs.push_back(std::move(captured));
                    continue; // never adopt nested roots under a groom
                }
                auto children = input->GetChildPrimPaths(path);
                stack.insert(stack.end(), children.begin(), children.end());
            }
        }
    } catch (...) {
        packet.failed = true;
        state->deferredFullCapture.store(true, std::memory_order_release);
        packet.inputs.clear();
        uint64_t const sequence = packet.sequence;
        (void)state->Post(std::move(*ingressTicket),
            [state, packet=std::move(packet)] { state->Apply(packet); },
            [state, sequence] { state->CompleteIngress(sequence); });
        throw;
    }
    uint64_t const sequence = packet.sequence;
    (void)state->Post(std::move(*ingressTicket), [state, packet=std::move(packet)] {
        try { state->Apply(packet); }
        catch (...) {
            // Required ownership/hold updates are not transactionally
            // recoverable after an allocation/framework failure yet.
            // Never disguise partial mutation as a completed ingress.
            std::terminate();
        }
    }, [state, sequence] { state->CompleteIngress(sequence); });
    // Without asyncAllow, the upstream observer call is the only legal
    // frontend delivery point.  Wait through causal attach/cook/publication
    // work, then send from this caller rather than the owner worker.
    if (!state->asyncAllowed.load(std::memory_order_acquire))
        _DrainPublications(true);
}
void UsdGenGroomSceneIndex::_PrimsAdded(HdSceneIndexBase const&, Added const& entries) {
    _Ingress packet; packet.added = entries; _CaptureAndSubmit(std::move(packet));
}
void UsdGenGroomSceneIndex::_PrimsRemoved(HdSceneIndexBase const&, Removed const& entries) {
    _Ingress packet; packet.removed = entries; _CaptureAndSubmit(std::move(packet));
}
void UsdGenGroomSceneIndex::_PrimsDirtied(HdSceneIndexBase const&, Dirtied const& entries) {
    _Ingress packet; packet.dirtied = entries;
    // Async pass-through.  With asynchronous scene processing, notices reach
    // Hydra only when the drain runs, and the drain runs on asyncPoll -- a
    // host timer (usdview: every 100 ms).  Holding upstream dirties until
    // then delays EVERY viewport update behind that timer (edits, animation,
    // selection, rig deformation), groom or not.  A dirty never changes
    // namespace, and GetPrim serves every upstream prim straight from the
    // input, so forwarding it now is exactly what the drain would send later.
    // Namespace notices and everything synthetic (tiles, render scopes,
    // materials, the scalp cap) stay on the drain, whose snapshot ordering
    // they depend on; the packet still enters the owner so cooks and
    // dependency tracking see the dirty.
    if (_state->asyncAllowed.load(std::memory_order_acquire) &&
        !_state->closing.load(std::memory_order_acquire) && !entries.empty()) {
        _SendPrimsDirtied(entries);
        packet.dirtiedDelivered = true;
    }
    _CaptureAndSubmit(std::move(packet));
}
void UsdGenGroomSceneIndex::_PrimsRenamed(HdSceneIndexBase const& sender,
    HdSceneIndexObserver::RenamedPrimEntries const& entries) {
    _Ingress packet;
    HdSceneIndexObserver::ConvertPrimsRenamedToRemovedAndAdded(sender, entries,
        &packet.removed, &packet.added);
    _CaptureAndSubmit(std::move(packet));
}

void UsdGenGroomSceneIndex::_SystemMessage(
    TfToken const& messageType, HdDataSourceBaseHandle const&) {
    if (messageType == HdSystemMessageTokens->asyncAllow) {
        _state->asyncAllowed.store(true, std::memory_order_release);
        return;
    }
    if (messageType == HdSystemMessageTokens->asyncPoll &&
        _state->asyncAllowed.load(std::memory_order_acquire)) {
        // Poll is a render-thread flush: it never awaits owner work, and it
        // starts no capture or cook of its own.  The engine's temporary
        // observer sees these notices.
        auto live = TfCreateRefPtrFromProtectedWeakPtr(_state->recipient);
        if (!live) return;
        _DrainPublications(false);
        // The one exception: a notice dropped under admission backpressure
        // (deferredFullCapture) is owed an authoritative rescan, and in async
        // mode nothing else would run it until the NEXT external notice --
        // so the last write of a brush drag could stay uncooked until the
        // user touched the stage again.  The poll is the Hydra-thread
        // boundary that keeps arriving without one, so it retries the
        // recovery here.  _CaptureAndSubmit is non-blocking in async mode
        // (it posts; a later poll delivers), and while admission is still
        // saturated it only re-arms the latch, so a stalled owner costs one
        // failed ticket reservation per poll.
        //
        // Threading: this capture reads the INPUT scene index (the UsdImaging
        // stage scene index), which is only safe on the thread that edits
        // the stage and delivers its notices.  asyncPoll is sent from
        // UsdImagingGLEngine::PollForAsynchronousUpdates, which usdview (and
        // every client we support) calls on that same application/main
        // thread; a client polling from another thread must not enable
        // asyncAllowed.
        //
        // Scope: the recovery restamps EVERY source and rescans the whole
        // catalog, re-cooking every groom.  Recovering only the dropped
        // paths is not possible: a latch set by Apply (a full-source packet
        // older than sourceThrough) or by a throwing capture has no path set,
        // and a dropped Added/Removed needs the namespace resync regardless.
        //
        // Back-off: a recovery that THROWS (not one merely refused by
        // admission) waits 50 ms doubling to 2 s before the next poll
        // retries, and warns once until a recovery succeeds.
        int64_t const now = std::chrono::steady_clock::now().time_since_epoch().count();
        if (_state->deferredFullCapture.load(std::memory_order_acquire) &&
            !_state->closing.load(std::memory_order_acquire) &&
            now >= _state->pollRecoveryNotBefore.load(std::memory_order_acquire)) {
            TF_DEBUG(USDGEN_INGRESS).Msg(
                "usdGen ingress   asyncPoll retries a pressure-deferred rescan\n");
            _Ingress recovery;
            recovery.forceFullDiscovery = true;
            recovery.recoverSourceNamespace = true;
            // A failed capture re-arms the latch itself; never let it
            // escape into the engine's poll.
            try {
                _CaptureAndSubmit(std::move(recovery));
                _state->pollRecoveryFailures.store(0, std::memory_order_release);
                _state->pollRecoveryWarned.store(false, std::memory_order_release);
            } catch (...) {
                uint32_t const failures =
                    _state->pollRecoveryFailures.fetch_add(1, std::memory_order_acq_rel) + 1;
                auto const backoff = (std::min)(std::chrono::milliseconds(2000),
                    std::chrono::milliseconds(50) * (int64_t(1) << (std::min<uint32_t>)(failures - 1, 6)));
                _state->pollRecoveryNotBefore.store(
                    now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(backoff).count(),
                    std::memory_order_release);
                if (!_state->pollRecoveryWarned.exchange(true, std::memory_order_acq_rel))
                    TF_WARN("usdGen deferred rescan failed; asyncPoll backs off and retries");
            }
        }
    }
}

HdSceneIndexPrim UsdGenGroomSceneIndex::GetPrim(SdfPath const& path) const {
    auto snapshot = _state->VisibleSnapshot();
    auto input = _GetInputSceneIndex();
    auto authored = input ? input->GetPrim(path) : HdSceneIndexPrim();
    // Authored namespace collisions win, including authored tile paths.
    if (authored.dataSource || !authored.primType.IsEmpty()) {
        if (authored.primType.IsEmpty() && IsGroom(TypeName(authored)))
            authored.primType = TypeName(authored);
        return authored;
    }
    for (auto const& g : snapshot->members) {
        if (path == g.root) return {TfToken("UsdGenGroom"), {}};
        if (path == RenderPath(g.description))
            return {TfToken("scope"), RenderDataSource()};
        if (path == MaterialPath(g.description))
            return {TfToken("material"),
                    ::usdGenImaging::UsdGenTilePublisher::
                        BuildDefaultMaterialDataSource(g.look)};
        bool flat = false;
        if (::usdGenImaging::UsdGenTilePublisher::IsPreviewMaterialPath(
                g.description, path, &flat) &&
            BoundPreviewMaterial(g.description, *g.tiles) == path)
            return {TfToken("material"),
                    ::usdGenImaging::UsdGenTilePublisher::
                        BuildPreviewMaterialDataSource(flat)};
        // The scalp-shadow cap. It carries no primOrigin and takes no
        // DescriptionOverlay: picking passes straight through it to whatever
        // the user authored underneath, and it never highlights.
        if (_publishScalpShadow && g.scalpShadow && path == ScalpShadowPath(g.description))
            return {TfToken("mesh"), g.hidden
                ? HdOverlayContainerDataSource::New(HiddenOverlay(), g.scalpShadow)
                : g.scalpShadow};
        if (_publishScalpShadow && g.scalpShadow && path == ScalpShadowMaterialPath(g.description))
            return {TfToken("material"),
                    ::usdGenImaging::UsdGenTilePublisher::
                        BuildScalpShadowMaterialDataSource(g.look)};
        auto tile = g.tiles->find(path);
        if (tile == g.tiles->end()) continue;
        // The synthetic tiles are not in the input scene, so the filters that
        // sit UPSTREAM of this index never reach them:
        //   * HdsiLegacyDisplayStyleOverrideSceneIndex (the usdview
        //     complexity slider, pushed as SetRefineLevelFallback) and
        //   * UsdImagingSelectionSceneIndex (the viewport selection
        //     highlight, stamped as a `selections` vector)
        // both only touch prims of the stage.  Both DO reach the owning
        // UsdGenDescription, which is a real stage prim, so republish its
        // opinion onto every tile of that groom.  Upstream goes FIRST: the
        // slider wins, and the publication's own refineLevel stays as the
        // underlay fallback for a host that offers no opinion at all.  The
        // On Storm only, the same overlay masks material bindings at the
        // two lowest levels to match its native curve repr restrictions.
        // A hidden groom's tiles read invisible over all of it.
        HdContainerDataSourceHandle upstream = DescriptionOverlay(
            input, g.description, path, tile->second, _stormMaterialPolicy);
        if (g.hidden)
            upstream = upstream
                ? HdOverlayContainerDataSource::New(HiddenOverlay(), upstream)
                : HiddenOverlay();
        if (upstream)
            return {TfToken("basisCurves"),
                    HdOverlayContainerDataSource::New(upstream, tile->second)};
        return {TfToken("basisCurves"), tile->second};
    }
    return {};
}
SdfPathVector UsdGenGroomSceneIndex::GetChildPrimPaths(SdfPath const& path) const {
    auto snapshot = _state->VisibleSnapshot();
    auto input = _GetInputSceneIndex();
    auto result = input ? input->GetChildPrimPaths(path) : SdfPathVector();
    auto append = [&](SdfPath const& child) {
        if (std::find(result.begin(), result.end(), child) == result.end()) result.push_back(child);
    };
    for (auto const& g : snapshot->members) {
        if (path == g.description) append(RenderPath(g.description));
        if (path == RenderPath(g.description)) {
            // Tiles first: callers index the render scope's children
            // positionally to reach "a tile".
            for (auto const& tile : *g.tiles) append(tile.first);
            append(MaterialPath(g.description));
            SdfPath const preview = BoundPreviewMaterial(g.description, *g.tiles);
            if (!preview.IsEmpty()) append(preview);
            if (_publishScalpShadow && g.scalpShadow) {
                append(ScalpShadowPath(g.description));
                append(ScalpShadowMaterialPath(g.description));
            }
        }
    }
    return result;
}
int64_t UsdGenGroomSceneIndex::_TestPublishedGeneration(SdfPath const& root) const {
    auto snapshot = _state->VisibleSnapshot();
    for (auto const& g : snapshot->members) if (g.root == root) return g.generation;
    return -1;
}
size_t UsdGenGroomSceneIndex::_TestPublishedTileCount(SdfPath const& root) const {
    auto snapshot = _state->VisibleSnapshot();
    for (auto const& g : snapshot->members) if (g.root == root) return g.tiles->size();
    return 0;
}
uint64_t UsdGenGroomSceneIndex::_TestIssuedIngress() const noexcept {
    return _state->sequences.LastIssued();
}
bool UsdGenGroomSceneIndex::_TestHoldOwnerCredit() {
    auto ticket = _state->owner->ReserveCommandTicket();
    if (!ticket) return false;
    _state->testHeldCredits.push_back(std::move(ticket));
    return true;
}
void UsdGenGroomSceneIndex::_TestReleaseOwnerCredit() {
    _state->testHeldCredits.clear();
}
void UsdGenGroomSceneIndex::_TestDrainOwnerWithoutFrontend() const {
    _state->Synchronize();
}
size_t UsdGenGroomSceneIndex::_TestPendingPublicationCount() const {
    return std::atomic_load(&_state->pending) ? 1 : 0;
}
std::weak_ptr<void const> UsdGenGroomSceneIndex::_TestPendingSnapshotWeak() const {
    std::shared_ptr<void const> owner = std::atomic_load(&_state->pending);
    return owner;
}
std::weak_ptr<void const> UsdGenGroomSceneIndex::_TestPendingTileMapWeak(
    SdfPath const& root) const {
    auto pending = std::atomic_load(&_state->pending);
    if (!pending) return {};
    for (auto const& g : pending->members) if (g.root == root) {
        std::shared_ptr<void const> owner = g.tiles;
        return owner;
    }
    return {};
}
int64_t UsdGenGroomSceneIndex::_TestPendingPublishedGeneration(SdfPath const& root) const {
    auto pending = std::atomic_load(&_state->pending);
    if (!pending) return -1;
    for (auto const& g : pending->members) if (g.root == root) return g.generation;
    return -1;
}
uint64_t UsdGenGroomSceneIndex::_TestCaptureCount() const noexcept {
    return _state->captureCount.load(std::memory_order_acquire);
}
uint64_t UsdGenGroomSceneIndex::ProcessCookCount() noexcept {
    return _State::ProcessCooks().load(std::memory_order_acquire);
}
uint64_t UsdGenGroomSceneIndex::ProcessPublishCount() noexcept {
    return _State::ProcessPublishes().load(std::memory_order_acquire);
}
uint64_t UsdGenGroomSceneIndex::_TestCookCount() const noexcept {
    return _state->cookCount.load(std::memory_order_acquire);
}
void UsdGenGroomSceneIndex::_TestOwnerCommandBarrier() const {
    _state->owner->InvokeOwner([] {});
}
uint64_t UsdGenGroomSceneIndex::_TestSequenceLastIssued() const noexcept {
    return _state->sequences.LastIssued();
}
uint64_t UsdGenGroomSceneIndex::_TestSequenceCompletedThrough() const noexcept {
    return _state->sequences.CompletedThrough();
}
uint64_t UsdGenGroomSceneIndex::_TestSequenceCapacity() const noexcept {
    return _state->sequences.Capacity();
}
size_t UsdGenGroomSceneIndex::_TestEventHistoryCount() const {
    return _state->events.size();
}
size_t UsdGenGroomSceneIndex::_TestTombstoneHistoryCount() const {
    return _state->tombstones.size();
}
size_t UsdGenGroomSceneIndex::_TestUsedSessionWeakCount() const {
    auto value = std::make_shared<size_t>(0);
    _state->owner->InvokeOwner([state=_state, value] { *value = state->usedSessions.size(); });
    return *value;
}
size_t UsdGenGroomSceneIndex::_TestUsedSessionLiveUniqueCount() const {
    auto value = std::make_shared<size_t>(0);
    _state->owner->InvokeOwner([state=_state, value] {
        std::vector<TfWeakPtr<Session>> seen;
        for (auto const& weak : state->usedSessions) {
            if (weak.IsExpired()) continue;
            if (std::find(seen.begin(), seen.end(), weak) == seen.end()) {
                seen.push_back(weak); ++*value;
            }
        }
    });
    return *value;
}
void UsdGenGroomSceneIndex::_TestSetCommandCapacity(uint64_t capacity) {
    if (capacity < 2)
        throw std::invalid_argument("groom test command capacity must retain close and ingress credits");
    s_testCommandCapacity.store(capacity, std::memory_order_release);
}
void UsdGenGroomSceneIndex::_TestSetSequenceCapacity(uint64_t capacity) {
    if (!capacity)
        throw std::invalid_argument("groom test sequence capacity must be nonzero");
    s_testSequenceCapacity.store(capacity, std::memory_order_release);
}
size_t UsdGenGroomSceneIndex::_TestRetainedSceneStateCount() { return SceneService().RetainedStateCount(); }
size_t UsdGenGroomSceneIndex::_TestLiveSceneStateCount() { return SceneService().LiveStateCount(); }
uint64_t UsdGenGroomSceneIndex::_TestRetirementRecordCount() {
    return UsdGenSceneService::RetirementRecordCount();
}
void UsdGenGroomSceneIndex::_TestSceneServiceBarrier() { SceneService().Barrier(); }
void UsdGenGroomSceneIndex::_TestArmFinalDeleterPause() {
    std::atomic_store(&s_testFinalDeleterGate, std::make_shared<TestFinalDeleterGate>());
}
void UsdGenGroomSceneIndex::_TestWaitFinalDeleterPause() {
    if (auto gate = std::atomic_load(&s_testFinalDeleterGate)) gate->entered.wait_for_all();
}
void UsdGenGroomSceneIndex::_TestReleaseFinalDeleterPause() {
    if (auto gate = std::atomic_load(&s_testFinalDeleterGate)) gate->release.release_wait();
    std::atomic_store(&s_testFinalDeleterGate, std::shared_ptr<TestFinalDeleterGate>());
}
void UsdGenGroomSceneIndex::_TestArmDrainWait() {
    std::atomic_store(&s_testDrainWaitGate, std::make_shared<TestDrainWaitGate>());
}
void UsdGenGroomSceneIndex::_TestWaitDrainWait() {
    if (auto gate = std::atomic_load(&s_testDrainWaitGate)) gate->entered.wait_for_all();
}
void UsdGenGroomSceneIndex::_TestReleaseDrainWait() {
    std::atomic_store(&s_testDrainWaitGate, std::shared_ptr<TestDrainWaitGate>());
}

PXR_NAMESPACE_CLOSE_SCOPE
