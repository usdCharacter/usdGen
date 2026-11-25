// Backend-neutral, immutable execution-plan metadata.  This is descriptive
// compiler output: dispatchers must still obey the concrete backend executor.
#ifndef USDGEN_EXECUTION_PLAN_H
#define USDGEN_EXECUTION_PLAN_H

#include "usdGen/graphDesc.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace usdGen {

enum class UsdGenExecutionCapabilityStatus : uint8_t {
    Supported,
    UnsupportedOperator,
    UnsupportedConfiguration,
    UnsupportedComposition,
    InvalidAuthoring
};

enum UsdGenExecutionCapabilityFlag : uint32_t {
    UsdGenCapabilityNone = 0,
    UsdGenCapabilityExpressions = 1u << 0,
    UsdGenCapabilityChangesTopology = 1u << 1,
    UsdGenCapabilityTopologyBarrier = 1u << 2,
    UsdGenCapabilityExclusiveWorkspace = 1u << 3,
    UsdGenCapabilitySurfaceInput = 1u << 4,
    UsdGenCapabilityTransactionalPublication = 1u << 5,
    // The backend can lower this operator so every written resource is
    // task-private while untouched inputs remain shared immutable views.
    // This is plane/resource-granular copy-on-write, not a whole-geometry
    // clone. A concrete plan may still choose an exclusive linear lowering.
    UsdGenCapabilityCopyOnWriteWrites = 1u << 6
};

// One versioned row in a backend's operator capability matrix.  A row says
// that the backend recognizes the operator family; per-node validation may
// still reject a configuration with a precise diagnostic.
struct UsdGenExecutionCapability {
    TfToken type;
    uint32_t capabilityVersion = 0;
    int minimumAlgorithmVersion = 0;
    int maximumAlgorithmVersion = 0;
    uint32_t flags = UsdGenCapabilityNone;
};

class UsdGenExecutionCapabilityMatrix {
public:
    UsdGenExecutionCapabilityMatrix(std::string backend, uint32_t version,
        bool available, std::vector<UsdGenExecutionCapability> capabilities)
        : _backend(std::move(backend)), _version(version), _available(available),
          _capabilities(std::move(capabilities)) {}

    std::string const& Backend() const noexcept { return _backend; }
    uint32_t Version() const noexcept { return _version; }
    bool Available() const noexcept { return _available; }
    std::vector<UsdGenExecutionCapability> const& Capabilities() const noexcept {
        return _capabilities;
    }
    UsdGenExecutionCapability const* Find(TfToken const& type) const noexcept {
        for (auto const& capability : _capabilities)
            if (capability.type == type) return &capability;
        return nullptr;
    }

private:
    std::string _backend;
    uint32_t _version = 0;
    bool _available = false;
    std::vector<UsdGenExecutionCapability> _capabilities;
};

enum class UsdGenExecutionTaskKind : uint8_t {
    Source,
    Operator,
    Publication
};

enum class UsdGenExecutionDataKind : uint8_t {
    // A scheduler-only identity for a backend workspace lane.  It is not a
    // geometry value and therefore must never appear in ResourceUse.
    ExecutionWorkspace,
    CurveGeometry,
    CurveTopology,
    StableIds,
    RootBindings,
    Widths,
    NamedChannels,
    SurfaceGeometry,
    ParameterValues,
    // An immutable decoded ImageMap payload. It has no physical workspace
    // hazard; CUDA Width candidates make private device uploads from it.
    ImageMaps,
    TerminalGeneration
};

enum class UsdGenExecutionResourceAccess : uint8_t { Read, Write, ReadWrite };

// A declared physical-storage hazard independent of logical value flow.  A
// Read records a real reader of a mutable identity; Write/ReadWrite records a
// mutator.  The compiler emits standard writer->reader, reader->writer, and
// writer->writer edges in a semantic topological order, with
// authored-order-key / task-id only as the ready-set tie-break. Values that
// are immutable or task-private must not be declared here at all: their
// logical producer/consumer edges are sufficient and keep sibling CoW work
// concurrent. This contract deliberately has no CUDA, Metal, or Vulkan handle
// types.
enum class UsdGenExecutionResourceHazardAccess : uint8_t {
    Read, Write, ReadWrite
};

struct UsdGenExecutionResourceHazard {
    UsdGenExecutionDataKind resource =
        UsdGenExecutionDataKind::ExecutionWorkspace;
    // Backend-neutral physical allocation/lane identity scoped to a compiled
    // plan.  Equal DataKind values only conflict when this identity also
    // matches; CUDA's single workspace lane currently uses the default zero.
    uint32_t identity = 0;
    UsdGenExecutionResourceHazardAccess access =
        UsdGenExecutionResourceHazardAccess::Read;
};

// Why an edge exists.  More than one reason can apply to a predecessor, so
// provenance is a bit set rather than a one-of enum.
enum UsdGenExecutionDependencyProvenance : uint8_t {
    UsdGenExecutionDependencyNone = 0,
    UsdGenExecutionDependencySemanticData = 1u << 0,
    UsdGenExecutionDependencyResourceHazard = 1u << 1,
    UsdGenExecutionDependencyLifetimePublicationJoin = 1u << 2
};

struct UsdGenExecutionDependency {
    uint32_t predecessor = UINT32_MAX;
    uint8_t provenance = UsdGenExecutionDependencyNone;
};

struct UsdGenExecutionResourceUse {
    UsdGenExecutionDataKind resource = UsdGenExecutionDataKind::CurveGeometry;
    UsdGenExecutionResourceAccess access = UsdGenExecutionResourceAccess::Read;
    // UINT32_MAX denotes an external/authored input.  Otherwise this names
    // the task whose logical resource version is observed.  The dependency
    // compiler derives semantic edges from this logical value relationship.
    uint32_t producerTask = UINT32_MAX;
    // Logical input/output value versions. UINT32_MAX means this access has
    // no value on that side (Write has no input; Read has no output).
    uint32_t inputValue = UINT32_MAX;
    uint32_t outputValue = UINT32_MAX;
};

// Describes the storage guarantee behind a logical resource version.  The
// current CUDA linear executor deliberately reports workspace aliasing; a
// future fan-out lowering must publish immutable versions instead of merely
// adding dependency edges around the shared mutable lane.
enum class UsdGenExecutionValueStorage : uint8_t {
    ExternalImmutable,
    ExclusiveWorkspaceVersion,
    // A job-owned allocation retained until every consumer and terminal
    // publication has completed. Multiple readers may share this immutable
    // value; distinct logical output values never alias one another.
    JobOwnedImmutable,
    PublishedImmutable
};

struct UsdGenExecutionValueMetadata {
    uint32_t id = 0;
    UsdGenExecutionDataKind resource = UsdGenExecutionDataKind::CurveGeometry;
    uint32_t producerTask = UINT32_MAX;
    uint32_t version = 0;
    UsdGenExecutionValueStorage storage =
        UsdGenExecutionValueStorage::ExclusiveWorkspaceVersion;
};

enum class UsdGenExecutionCancellationPoint : uint8_t {
    // The dispatcher may reject before invoking the task. Once invoked, the
    // backend completes submitted work and discards publication if stale.
    BeforeTaskOnly
};

// Shared checked arithmetic for backend byte estimators. Overflow is a hard
// planning failure; saturation must never be mistaken for an admissible bound.
struct UsdGenExecutionCheckedBytes {
    static bool Add(uint64_t* total, uint64_t value) noexcept {
        if (!total || value > std::numeric_limits<uint64_t>::max() - *total)
            return false;
        *total += value;
        return true;
    }
    static bool Multiply(uint64_t count, uint64_t bytes,
                         uint64_t* result) noexcept {
        if (!result || (count &&
            bytes > std::numeric_limits<uint64_t>::max() / count))
            return false;
        *result = count * bytes;
        return true;
    }
};

struct UsdGenExecutionTaskEstimate {
    // Explicit usdGen-owned payload accounting only. CUDA events, driver and
    // library-internal allocations are intentionally outside this contract.
    // Persistent task-owned workspace or cache. This intentionally excludes
    // the task's logical output value.
    uint64_t steadyBytes = 0;
    // Bytes held by the task's produced logical value after task completion.
    uint64_t retainedOutputBytes = 0;
    // Bytes of an upstream producer that must remain live while this task's
    // output is retained (for example source data during resampling).
    uint64_t producerRetentionBytes = 0;
    uint64_t scratchPeakBytes = 0;
    // A true value means the four byte fields form a conservative payload
    // upper bound for this task. False leaves known components populated but
    // forbids admission from treating their sum as complete.
    bool memoryAvailable = false;
    bool memoryConservativeUpperBound = false;
    // No backend timing model has been calibrated yet.
    bool timeAvailable = false;
    uint64_t estimatedMicroseconds = 0;
};

struct UsdGenExecutionGraphMemoryEstimate {
    // A deterministic upper bound for declared usdGen payload only, not a
    // measured allocation snapshot and not an admission reservation.
    // Reservation-ticket consumption is a later enforcement step; consumers
    // must not treat this descriptive metadata as an allocation permit.
    uint64_t concurrentPeakBytes = 0;
    // Immutable source payload counted once even when many CoW branches read
    // it. It is included in concurrentPeakBytes when memoryAvailable is true.
    uint64_t immutableSharedInputBytes = 0;
    bool memoryAvailable = false;
    bool conservativeUpperBound = false;
    // True means the compiled plan contains enough backend-neutral shape
    // information for its backend to obtain a conservative bound after a
    // device is selected but before execution or task submission. The
    // backend-specific workspace query and its result are intentionally not
    // embedded in immutable plan metadata. This is mutually exclusive with
    // memoryAvailable for the current executors.
    bool runtimeRefinementAvailable = false;
};

struct UsdGenExecutionTaskMetadata {
    uint32_t id = 0;
    // Dense semantic node ordinal for authored Source/Operator tasks. A
    // publication or compiler-lowered internal task has no authored node and
    // uses UINT32_MAX.
    uint32_t semanticNode = UINT32_MAX;
    uint64_t authoredOrderKey = 0;
    UsdGenExecutionTaskKind kind = UsdGenExecutionTaskKind::Operator;
    SdfPath path;
    TfToken type;
    // Input to dependency lowering.  Direct semantic ordering is inferred
    // from ResourceUse::inputValue; only a publication lifetime join may be
    // declared directly here.  Keeping this separate prevents an executor
    // from mistaking an ad-hoc hidden serialization edge for a data edge.
    std::vector<UsdGenExecutionDependency> declaredDependencies;
    // Output of UsdGenExecutionDependencyCompiler.  `dependencies` remains
    // the compact executor-facing form; dependencyProvenance makes every
    // edge auditable by compiler tests and later backend adapters.
    std::vector<uint32_t> dependencies;
    std::vector<UsdGenExecutionDependency> dependencyProvenance;
    std::vector<UsdGenExecutionResourceUse> resources;
    std::vector<UsdGenExecutionResourceHazard> resourceHazards;
    UsdGenExecutionTaskEstimate estimate;
    UsdGenExecutionCancellationPoint cancellation =
        UsdGenExecutionCancellationPoint::BeforeTaskOnly;
    bool topologyBarrier = false;
    // Legacy compact executor flag. Dependency lowering verifies that it
    // exactly mirrors a mutating ExecutionWorkspace hazard.
    bool exclusiveWorkspace = true;
};

// Backend-neutral deterministic dependency lowering.  It derives semantic
// edges from logical value producer/consumer relationships, adds only declared
// physical-resource hazards, and preserves explicit publication joins.  It
// does not invent merge semantics: callers must model one logical input value
// per unary operator before invoking it.
class UsdGenExecutionDependencyCompiler {
public:
    static bool Lower(std::vector<UsdGenExecutionTaskMetadata>* tasks,
                      std::vector<UsdGenExecutionValueMetadata> const& values,
                      std::string* reason = nullptr) {
        if (!tasks) return _Fail(reason, "dependency compiler received no tasks");
        auto& taskList = *tasks;
        if (taskList.empty()) return _Fail(reason, "dependency compiler received no tasks");

        std::vector<UsdGenExecutionTaskMetadata const*> byId(taskList.size());
        for (auto const& task : taskList) {
            if (task.id >= taskList.size() || byId[task.id])
                return _Fail(reason, "execution task ids must be dense and unique");
            byId[task.id] = &task;
        }
        for (auto const* task : byId)
            if (!task) return _Fail(reason, "execution task ids must be dense and unique");

        std::map<uint32_t, UsdGenExecutionValueMetadata const*> valuesById;
        for (auto const& value : values) {
            if (!valuesById.emplace(value.id, &value).second)
                return _Fail(reason, "execution value ids must be unique");
            if (value.resource == UsdGenExecutionDataKind::ExecutionWorkspace)
                return _Fail(reason,
                    "workspace identity is only valid in a resource hazard");
            if (value.producerTask != UINT32_MAX &&
                (value.producerTask >= taskList.size() || !byId[value.producerTask]))
                return _Fail(reason, "execution value names an invalid producer task");
        }

        std::vector<std::map<uint32_t, uint8_t>> edges(taskList.size());
        auto add = [&](uint32_t successor, uint32_t predecessor,
                       uint8_t provenance) -> bool {
            if (successor >= taskList.size() || predecessor >= taskList.size() ||
                successor == predecessor || provenance == UsdGenExecutionDependencyNone)
                return false;
            edges[successor][predecessor] |= provenance;
            return true;
        };
        for (auto const& task : taskList) {
            for (auto const& declared : task.declaredDependencies) {
                if (task.kind != UsdGenExecutionTaskKind::Publication ||
                    declared.provenance !=
                    UsdGenExecutionDependencyLifetimePublicationJoin ||
                    !add(task.id, declared.predecessor, declared.provenance))
                    return _Fail(reason, "only valid publication lifetime joins may be declared directly");
            }
            for (auto const& use : task.resources) {
                if (use.resource == UsdGenExecutionDataKind::ExecutionWorkspace)
                    return _Fail(reason,
                        "workspace identity is only valid in a resource hazard");
                const bool reads = use.access != UsdGenExecutionResourceAccess::Write;
                const bool writes = use.access != UsdGenExecutionResourceAccess::Read;
                if (reads) {
                    if (use.inputValue == UINT32_MAX)
                        return _Fail(reason, "resource read has no logical input value");
                    auto found = valuesById.find(use.inputValue);
                    if (found == valuesById.end() || found->second->resource != use.resource)
                        return _Fail(reason, "resource read names an invalid logical value");
                    auto const producer = found->second->producerTask;
                    if (use.producerTask != producer)
                        return _Fail(reason, "resource read producer does not match its logical value");
                    if (producer != UINT32_MAX && !add(task.id, producer,
                            UsdGenExecutionDependencySemanticData))
                        return _Fail(reason, "resource data dependency is malformed");
                } else if (use.inputValue != UINT32_MAX ||
                           use.producerTask != UINT32_MAX) {
                    return _Fail(reason, "write-only resource must not name an input value");
                }
                if (writes) {
                    if (use.outputValue == UINT32_MAX)
                        return _Fail(reason, "resource write has no logical output value");
                    auto found = valuesById.find(use.outputValue);
                    if (found == valuesById.end() || found->second->resource != use.resource ||
                        found->second->producerTask != task.id)
                        return _Fail(reason, "resource write names an invalid logical value");
                } else if (use.outputValue != UINT32_MAX) {
                    return _Fail(reason, "read-only resource must not name an output value");
                }
            }
            bool mutatesWorkspace = false;
            for (auto const& hazard : task.resourceHazards)
                if (hazard.resource == UsdGenExecutionDataKind::ExecutionWorkspace &&
                    hazard.access != UsdGenExecutionResourceHazardAccess::Read) {
                    mutatesWorkspace = true;
                    break;
                }
            if (task.exclusiveWorkspace != mutatesWorkspace)
                return _Fail(reason,
                    "exclusiveWorkspace disagrees with workspace hazard metadata");
        }

        // A publication is the ownership boundary for the entire submitted
        // task set.  Make the join explicit rather than relying on transitive
        // ancestry: a side branch may otherwise finish after terminal value
        // construction and lose its task-private resource too early.
        for (auto const& task : taskList) if (
            task.kind == UsdGenExecutionTaskKind::Publication) {
            std::vector<bool> joined(taskList.size());
            for (auto const& declared : task.declaredDependencies)
                if (declared.predecessor < joined.size())
                    joined[declared.predecessor] = true;
            for (uint32_t id = 0; id != joined.size(); ++id)
                if (id != task.id && !joined[id])
                    return _Fail(reason,
                        "publication must join every launched task");
        }

        auto stableTopologicalOrder = [&](std::vector<uint32_t>* result) {
            std::vector<uint32_t> indegree(taskList.size());
            std::vector<std::vector<uint32_t>> children(taskList.size());
            for (uint32_t successor = 0; successor != edges.size(); ++successor)
                for (auto const& edge : edges[successor]) {
                    ++indegree[successor]; children[edge.first].push_back(successor);
                }
            auto before = [&](uint32_t a, uint32_t b) {
                auto const& left = *byId[a]; auto const& right = *byId[b];
                return left.authoredOrderKey != right.authoredOrderKey
                    ? left.authoredOrderKey < right.authoredOrderKey : a < b;
            };
            std::vector<uint32_t> ready;
            for (uint32_t id = 0; id != indegree.size(); ++id)
                if (!indegree[id]) ready.push_back(id);
            result->clear(); result->reserve(taskList.size());
            while (!ready.empty()) {
                auto next = std::min_element(ready.begin(), ready.end(), before);
                auto id = *next; ready.erase(next); result->push_back(id);
                for (auto child : children[id])
                    if (!--indegree[child]) ready.push_back(child);
            }
            return result->size() == taskList.size();
        };

        // Hazards are oriented by a stable *semantic* topological order;
        // authoredOrderKey/task-id is only the ready-set tie-break.  This
        // preserves valid data flow if descriptors are stored out of authored
        // order, while keeping independent conflict resolution deterministic.
        std::vector<uint32_t> order;
        if (!stableTopologicalOrder(&order))
            return _Fail(reason,
                "semantic/data/publication dependencies contain a cycle");
        // Immutable logical values intentionally have no hazard declaration,
        // which is the CoW promise that permits their sibling concurrency.
        struct HazardState {
            uint32_t lastWriter = UINT32_MAX;
            std::vector<uint32_t> readers;
        };
        using HazardKey = std::pair<UsdGenExecutionDataKind, uint32_t>;
        std::map<HazardKey, HazardState> hazards;
        for (auto id : order) {
            auto const& task = *byId[id];
            std::map<HazardKey,
                UsdGenExecutionResourceHazardAccess> accesses;
            for (auto const& hazard : task.resourceHazards) {
                if (hazard.access != UsdGenExecutionResourceHazardAccess::Read &&
                    hazard.access != UsdGenExecutionResourceHazardAccess::Write &&
                    hazard.access != UsdGenExecutionResourceHazardAccess::ReadWrite)
                    return _Fail(reason, "resource hazard has an invalid access mode");
                auto inserted = accesses.emplace(
                    HazardKey{hazard.resource, hazard.identity}, hazard.access);
                if (!inserted.second) {
                    auto const oldReads = inserted.first->second !=
                        UsdGenExecutionResourceHazardAccess::Write;
                    auto const oldWrites = inserted.first->second !=
                        UsdGenExecutionResourceHazardAccess::Read;
                    auto const reads = hazard.access !=
                        UsdGenExecutionResourceHazardAccess::Write;
                    auto const writes = hazard.access !=
                        UsdGenExecutionResourceHazardAccess::Read;
                    inserted.first->second = oldReads || reads
                        ? (oldWrites || writes
                            ? UsdGenExecutionResourceHazardAccess::ReadWrite
                            : UsdGenExecutionResourceHazardAccess::Read)
                        : UsdGenExecutionResourceHazardAccess::Write;
                }
            }
            for (auto const& pair : accesses) {
                auto& state = hazards[pair.first];
                bool const reads = pair.second !=
                    UsdGenExecutionResourceHazardAccess::Write;
                bool const writes = pair.second !=
                    UsdGenExecutionResourceHazardAccess::Read;
                if (reads && state.lastWriter != UINT32_MAX &&
                    !add(id, state.lastWriter,
                        UsdGenExecutionDependencyResourceHazard))
                    return _Fail(reason, "writer/read resource hazard is malformed");
                if (writes) {
                    if (state.lastWriter != UINT32_MAX &&
                        !add(id, state.lastWriter,
                            UsdGenExecutionDependencyResourceHazard))
                        return _Fail(reason, "writer/write resource hazard is malformed");
                    for (auto reader : state.readers)
                        if (!add(id, reader,
                                UsdGenExecutionDependencyResourceHazard))
                            return _Fail(reason, "reader/write resource hazard is malformed");
                    state.readers.clear();
                    state.lastWriter = id;
                } else if (reads) {
                    state.readers.push_back(id);
                }
            }
        }

        // Retain a post-lowering cycle check as an invariant: future hazard
        // policies may add edges that are not necessarily forward-only.
        std::vector<uint32_t> finalOrder;
        if (!stableTopologicalOrder(&finalOrder))
            return _Fail(reason,
                "lowered semantic/resource/publication dependencies contain a cycle");

        for (auto& task : taskList) {
            task.dependencies.clear(); task.dependencyProvenance.clear();
            for (auto const& edge : edges[task.id]) {
                task.dependencies.push_back(edge.first);
                task.dependencyProvenance.push_back({edge.first, edge.second});
            }
        }
        return true;
    }

    static bool Validate(std::vector<UsdGenExecutionTaskMetadata> const& tasks,
                         std::vector<UsdGenExecutionValueMetadata> const& values,
                         std::string* reason = nullptr) {
        auto expected = tasks;
        if (!Lower(&expected, values, reason)) return false;
        if (expected.size() != tasks.size())
            return _Fail(reason, "dependency compiler changed task count");
        for (size_t i = 0; i != tasks.size(); ++i) {
            if (expected[i].dependencies != tasks[i].dependencies ||
                !_Equal(expected[i].dependencyProvenance,
                        tasks[i].dependencyProvenance))
                return _Fail(reason, "execution task dependencies are not compiler-lowered");
        }
        return true;
    }

private:
    static bool _Fail(std::string* reason, char const* message) {
        if (reason) *reason = message;
        return false;
    }
    static bool _Equal(std::vector<UsdGenExecutionDependency> const& left,
                       std::vector<UsdGenExecutionDependency> const& right) {
        if (left.size() != right.size()) return false;
        for (size_t i = 0; i != left.size(); ++i)
            if (left[i].predecessor != right[i].predecessor ||
                left[i].provenance != right[i].provenance) return false;
        return true;
    }
};

struct UsdGenCompiledOperatorCapability {
    SdfPath path;
    TfToken type;
    uint64_t authoredOrderKey = 0;
    uint32_t capabilityVersion = 0;
    uint32_t flags = UsdGenCapabilityNone;
    UsdGenExecutionCapabilityStatus status =
        UsdGenExecutionCapabilityStatus::Supported;
};

enum class UsdGenExecutionPlanShape : uint8_t {
    // Explicitly records the current CUDA limitation.  It must not be
    // interpreted as proof that arbitrary semantic fan-in is executable.
    LinearAuthoredChain,
    // One source and unary operators.  Immutable value versions allow an
    // executor to run semantically independent branches concurrently; the
    // concrete stream/event synchronization remains backend-specific.
    SourceRootedUnaryDag,
    // One source and an explicitly admitted value fan-in.  This is distinct
    // from the unary DAG contract: every binary edge must be represented in
    // dependencies, resources, synchronization, and retained ownership.
    SourceRootedValueDag
};

class UsdGenExecutionPlanMetadata {
public:
    UsdGenExecutionPlanMetadata(std::string backend, uint32_t capabilityVersion,
        UsdGenExecutionPlanShape shape,
        std::vector<UsdGenCompiledOperatorCapability> operators,
        std::vector<UsdGenExecutionTaskMetadata> tasks,
        std::vector<UsdGenExecutionValueMetadata> values, uint32_t terminalTask,
        UsdGenExecutionGraphMemoryEstimate memoryEstimate = {})
        : _backend(std::move(backend)), _capabilityVersion(capabilityVersion),
          _shape(shape), _operators(std::move(operators)),
          _tasks(std::move(tasks)), _values(std::move(values)),
          _terminalTask(terminalTask), _memoryEstimate(memoryEstimate) {}

    std::string const& Backend() const noexcept { return _backend; }
    uint32_t CapabilityVersion() const noexcept { return _capabilityVersion; }
    UsdGenExecutionPlanShape Shape() const noexcept { return _shape; }
    std::vector<UsdGenCompiledOperatorCapability> const& Operators() const noexcept {
        return _operators;
    }
    std::vector<UsdGenExecutionTaskMetadata> const& Tasks() const noexcept {
        return _tasks;
    }
    std::vector<UsdGenExecutionValueMetadata> const& Values() const noexcept {
        return _values;
    }
    uint32_t TerminalTask() const noexcept { return _terminalTask; }
    UsdGenExecutionGraphMemoryEstimate const& MemoryEstimate() const noexcept {
        return _memoryEstimate;
    }
    UsdGenExecutionTaskMetadata const* FindTask(uint32_t id) const noexcept {
        for (auto const& task : _tasks) if (task.id == id) return &task;
        return nullptr;
    }
    UsdGenExecutionTaskMetadata const* FindSemanticTask(uint32_t node) const noexcept {
        for (auto const& task : _tasks)
            if (task.kind != UsdGenExecutionTaskKind::Publication &&
                task.semanticNode == node) return &task;
        return nullptr;
    }
    UsdGenExecutionValueMetadata const* FindValue(uint32_t id) const noexcept {
        for (auto const& value : _values) if (value.id == id) return &value;
        return nullptr;
    }

private:
    std::string _backend;
    uint32_t _capabilityVersion = 0;
    UsdGenExecutionPlanShape _shape = UsdGenExecutionPlanShape::LinearAuthoredChain;
    std::vector<UsdGenCompiledOperatorCapability> _operators;
    std::vector<UsdGenExecutionTaskMetadata> _tasks;
    std::vector<UsdGenExecutionValueMetadata> _values;
    uint32_t _terminalTask = 0;
    UsdGenExecutionGraphMemoryEstimate _memoryEstimate;
};

} // namespace usdGen
#endif
