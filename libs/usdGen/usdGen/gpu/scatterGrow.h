#ifndef USDGEN_GPU_SCATTER_GROW_H
#define USDGEN_GPU_SCATTER_GROW_H

#include "curveGeometry.h"
#include "imageSampler.h"

#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"
#include "pxr/pxr.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace usdGen::gpu {

// Pod conversions between the capture's VtArray element types and the
// float2/float3/int/uint64_t currency every roots consumer already speaks.
// Same bytes both ways (the static_asserts pin the layouts, and VtArray
// storage is malloc-aligned); overload resolution picks by argument type.
static_assert(sizeof(float2) == sizeof(PXR_NS::GfVec2f),
              "float2/GfVec2f layout");
static_assert(sizeof(float3) == sizeof(PXR_NS::GfVec3f),
              "float3/GfVec3f layout");
static_assert(sizeof(int) == sizeof(int32_t), "int/int32_t layout");
inline float2 RootsToPod(PXR_NS::GfVec2f const& e)
{
    return float2{e[0], e[1]};
}
inline float3 RootsToPod(PXR_NS::GfVec3f const& e)
{
    return float3{e[0], e[1], e[2]};
}
inline uint64_t RootsToPod(uint64_t e) { return e; }
inline int32_t RootsToPod(int e) { return int32_t(e); }
inline PXR_NS::GfVec2f RootsFromPod(float2 const& p)
{
    return PXR_NS::GfVec2f(p.x, p.y);
}
inline PXR_NS::GfVec3f RootsFromPod(float3 const& p)
{
    return PXR_NS::GfVec3f(p.x, p.y, p.z);
}
inline uint64_t RootsFromPod(uint64_t p) { return p; }
inline int RootsFromPod(int32_t p) { return int(p); }

// A VtArray-backed roots plane with the std::vector read API every consumer
// already uses (size/empty/data/const-subscript) plus the small mutation
// subset tests use (reserve/push_back/pop_back/init-list assign/clear and
// a mutable data() for word-level fault planting).
// The producer adopts capture planes with O(1) moves instead of copying
// them element-wise; the stored bytes are identical either way, so every
// consumer observes the same values. Copies share storage copy-on-write
// (VtArray's own guarantee): mutating a copy detaches first, exactly as
// a deep copy would read. Const subscript returns by value (converted
// out of the stored element, so no aliasing); mutable subscript returns
// a proxy converting to/from Pod on read/write.
template <class Store, class Pod, class StoreElem>
class ScatterGrowPlane {
public:
    typedef Pod PodType;
    typedef Store StoreType;
    typedef StoreElem StoreElemType;

    class Ref {
    public:
        Ref() = delete;
        explicit Ref(StoreElem* e) : elem(e) {}
        operator Pod() const { return RootsToPod(*elem); }
        Ref& operator=(Pod const& v)
        {
            *elem = RootsFromPod(v);
            return *this;
        }
        Ref& operator=(Ref const& r)
        {
            *elem = *r.elem;
            return *this;
        }
    private:
        StoreElem* elem;
    };

    ScatterGrowPlane() = default;
    ScatterGrowPlane(ScatterGrowPlane const&) = default;
    ScatterGrowPlane(ScatterGrowPlane&&) = default;
    ScatterGrowPlane& operator=(ScatterGrowPlane const&) = default;
    ScatterGrowPlane& operator=(ScatterGrowPlane&&) = default;
    ScatterGrowPlane& operator=(std::initializer_list<Pod> xs)
    {
        store.clear();
        store.reserve(xs.size());
        for (Pod const& v : xs)
            store.push_back(RootsFromPod(v));
        return *this;
    }

    size_t size() const noexcept { return store.size(); }
    bool empty() const noexcept { return store.empty(); }
    Pod const* data() const noexcept
    {
        return reinterpret_cast<Pod const*>(store.cdata());
    }
    // Mutable base (detaches a COW-shared plane first, like every other
    // VtArray mutation, so a live copy keeps its bytes).
    Pod* data() noexcept
    {
        return reinterpret_cast<Pod*>(store.data());
    }
    Pod operator[](size_t i) const { return RootsToPod(store[i]); }
    Ref operator[](size_t i) { return Ref(&store[i]); }
    void clear() { store.clear(); }
    void reserve(size_t n) { store.reserve(n); }
    void push_back(Pod const& v) { store.push_back(RootsFromPod(v)); }
    void pop_back() { store.pop_back(); }
    // Uninitialized fill: clear + resize(n, fill), whose filler runs over
    // reused-or-fresh storage, skipping the value-init a plain resize
    // would zero first. The filler must construct every element exactly
    // once (placement new); disjoint ranges fill disjoint lanes, so the
    // threaded transpose below writes identical bytes under any chunking.
    template <class Fill>
    void resize_fill(size_t n, Fill&& fill)
    {
        store.clear();
        store.resize(n, std::forward<Fill>(fill));
    }
    // O(1) ownership take; the source is left moved-from (valid, empty).
    void adopt(Store&& s) { store = std::move(s); }
    // O(1) ownership give (adopt's inverse); this plane is left empty.
    void donate(Store& dst) { dst = std::move(store); }

private:
    Store store;
};

// Immutable, captured Scatter roots.  The producer retains the shared owner
// until the producer stream's terminal callback has been proved and committed.
// The six layout-identical planes are adopted (moved) from the capture's
// VtArrays; positions is transposed from the SoA point planes through an
// uninitialized fill (a vector cannot fill thread-parallel without a
// zero-init pass).  Same bytes as the old all-vector spelling, without
// the copies.
struct ScatterGrowRoots {
    ScatterGrowPlane<PXR_NS::VtVec3fArray, float3, PXR_NS::GfVec3f> positions;
    ScatterGrowPlane<PXR_NS::VtArray<uint64_t>, uint64_t, uint64_t> stableIds;
    ScatterGrowPlane<PXR_NS::VtIntArray, int32_t, int> rootPrim;
    ScatterGrowPlane<PXR_NS::VtVec2fArray, float2, PXR_NS::GfVec2f> rootUV;
    ScatterGrowPlane<PXR_NS::VtVec3fArray, float3, PXR_NS::GfVec3f> rootT, rootB,
        rootN;
};

enum class ScatterGrowDirection : uint8_t { RootNormal, RootTangent, Literal };

// This deliberately accepts literals only.  Maps, masks, animated controls,
// and arbitrary direction primvars have no representation here and therefore
// cannot be silently approximated by this strict Scatter -> Grow slice.
struct ScatterGrowControls {
    uint32_t cvCount = 2;
    int seed = 0;
    // CPU Grow evaluates the authored length/random expression in double and
    // narrows the per-curve target to float only after the deterministic draw.
    // Keep these controls double through the CUDA launch for the same rule.
    double length = 1.0;
    double randomLo = 1.0, randomHi = 1.0;
    float lift = 0.0f;
    float fallbackWidth = 0.01f;
    ScatterGrowDirection direction = ScatterGrowDirection::RootNormal;
    float3 literalDirection = {0.0f, 1.0f, 0.0f};
    // Root-normal rotation after lift; random=1 spans a full circle.
    // Appended to preserve aggregate initialization order.
    float azimuth = 0.0f;
    float azimuthRandom = 0.0f;
};

enum class ScatterGrowStatus {
    Ok, InvalidArgument, NonFiniteInput, InvalidTopology, DuplicateStableId,
    NoPendingUpdate, CudaError
};

struct ScatterGrowRequirements {
    size_t pointCount = 0;
    size_t inputBytes = 0;
    size_t outputBytes = 0;
    // One device status and one pinned host status are part of the producer's
    // transaction and remain retained until its terminal proof.
    size_t statusBytes = 0;
    // Optional length-map texels plus one sampled scalar per captured root.
    size_t peakBytes = 0;
};

// Exact explicit payload sizes, available before selecting a device.  The
// statusBytes field includes one device word and one pinned host relay word;
// native events and other driver allocations are outside this ledger.
// inputBytes is the transient input staging (root positions); ids, prims,
// uvs, and frames upload directly into their published output buffers and
// are counted in outputBytes.
ScatterGrowStatus GetScatterGrowRequirements(size_t curveCount, uint32_t cvCount,
                                            ScatterGrowRequirements* result);

// A fresh, immutable Grow topology owner.  BeginFresh never mutates a
// published generation.  A caller must commit only from the relay which saw
// FinishFreshAsync report cudaSuccess; if that proof is lost it must call
// MarkUnprovenWork, causing destruction to quarantine device allocations.
class CudaScatterGrow {
public:
    CudaScatterGrow() = default;
    ~CudaScatterGrow();
    CudaScatterGrow(CudaScatterGrow const&) = delete;
    CudaScatterGrow& operator=(CudaScatterGrow const&) = delete;

    ScatterGrowStatus BeginFresh(std::shared_ptr<const ScatterGrowRoots> roots,
        ScatterGrowControls controls, cudaStream_t stream,
        UsdGenExecutionMemoryReservation* reservation = nullptr);
    ScatterGrowStatus FinishFreshAsync(cudaStream_t stream,
        void (*callback)(cudaStream_t, cudaError_t, void*) noexcept, void* userdata);
    ScatterGrowStatus CommitFreshFinish();
    void MarkUnprovenWork() noexcept { unprovenWork_ = true; }
    bool HasUnprovenWork() const noexcept { return unprovenWork_; }
    bool pending() const noexcept { return pendingWork_; }
    uint64_t generation() const noexcept { return generation_; }
    int deviceIndex() const noexcept { return deviceIndex_; }
    size_t curveCount() const noexcept { return curves_; }
    size_t pointCount() const noexcept { return points_; }
    DeviceCurveGeometryView view() const;
    DeviceView<const float> hairT() const;
    DeviceView<const int32_t> rootPrim() const;
    DeviceView<const float2> rootUV() const;
    DeviceView<const float3> rootT() const;
    DeviceView<const float3> rootB() const;
    DeviceView<const float3> rootN() const;
    ScatterGrowStatus recordUse(cudaStream_t stream);
    ScatterGrowStatus waitOn(cudaStream_t stream) const;
    // Host-only transition after Session ownership has published this fresh
    // generation.  Staging roots have already been released by commit.
    void ReclassifyPublishedGeneration() noexcept;
    size_t ExclusiveRetainedBytes() const noexcept;
    // CPU-only preflight of grow roots: the same checks BeginFresh runs
    // before touching the device (controls, topology, finiteness,
    // duplicate stable ids, per-curve overflow), without needing a
    // stream or a device.  Writes the point total through `total`
    // (which must be non-null) whenever it reports Ok.
    static ScatterGrowStatus ValidateRoots(
        std::shared_ptr<const ScatterGrowRoots> const& roots,
        ScatterGrowControls const& controls, size_t* total);

private:
    // No restPoints buffer: rest == points elementwise for grown roots
    // (the kernel writes every point; the old D2D rest copy is gone), and
    // a published generation is immutable (BeginFresh never mutates one;
    // downstream styled points override the view with separately-owned
    // buffers), so view() publishes the points buffer under both view
    // slots. Identical bytes, one less allocation and 96MB less traffic
    // per 1M-root/8cv grow.
    struct Storage {
        DeviceBuffer<float3> points, rootT, rootB, rootN;
        DeviceBuffer<float> widths, hairT;
        DeviceBuffer<uint32_t> offsets;
        DeviceBuffer<uint64_t> stableIds;
        DeviceBuffer<int32_t> rootPrim;
        DeviceBuffer<float2> rootUV;
        void quarantine() noexcept;
        size_t bytes() const noexcept;
        ScatterGrowStatus recordUse(cudaStream_t);
        ScatterGrowStatus waitOn(cudaStream_t) const;
        ScatterGrowStatus synchronizeUse() const;
        void Reclassify(UsdGenExecutionResourceKind) noexcept;
    } active_, pending_;
    // Device copy of the immutable captured roots.  It is retained until the
    // terminal callback is committed, rather than borrowing caller memory.
    Storage pendingInput_;

    ScatterGrowStatus validate(std::shared_ptr<const ScatterGrowRoots> const&,
                               ScatterGrowControls const&, size_t* total) const;
    ScatterGrowStatus validateStream(cudaStream_t) const;
    void discardPending() noexcept;
    std::shared_ptr<const ScatterGrowRoots> rootsOwner_;
    // Preallocated before submission so unproved H2D retirement can retain
    // its immutable host source without allocating in a noexcept destructor.
    std::unique_ptr<std::shared_ptr<const ScatterGrowRoots>> rootsQuarantineOwner_;
    DeviceBuffer<int> error_;
    int* hostError_ = nullptr;
    UsdGenExecutionResourcePermit hostErrorPermit_;
    cudaEvent_t ready_ = nullptr;
    // A completion callback proves this producer's uploads/kernel only when
    // it is ordered on the exact stream that submitted them.
    cudaStream_t producerStream_ = nullptr;
    size_t curves_ = 0, points_ = 0, pendingCurves_ = 0, pendingPoints_ = 0;
    uint64_t generation_ = 0;
    int deviceIndex_ = -1;
    bool pendingWork_ = false, finishScheduled_ = false, unprovenWork_ = false;
};

} // namespace usdGen::gpu
#endif
