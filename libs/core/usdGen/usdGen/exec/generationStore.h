// usdGen engine — generation store (03 §6.1/§6.2).
//
// A generation is immutable once published. The publish/read pair is a
// UsdGenSharedCell (atomic holder pointer, epoch-protected weak lock): no
// lock, and the displaced generation dies when its last reference drops.
// What changed between two generations is the publisher's diff
// (registries/publisher.h).
#ifndef USDGEN_GENERATION_STORE_H
#define USDGEN_GENERATION_STORE_H

#include "usdGen/api/primRef.h"
#include "usdGen/exec/device/deviceGeneration.h"
#include "usdGen/exec/sharedCell.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/publisher.h"

#include "pxr/base/gf/matrix4d.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// Scalar scene-presentation state paired with a device generation.  This is
/// intentionally separate from UsdGenDeviceGeneration: the device payload is
/// geometry/tool data, while these values are the immutable descriptor
/// snapshot that a renderer needs to publish that payload.  No geometry or
/// per-element host mirror is retained here.
struct UsdGenDevicePresentationMetadata
{
    UsdGenNodeKey description;   // the host composes the render namespace under it
    GfMatrix4d xformMatrix{1.0};
    TfToken purpose;
    TfToken visibility;
    UsdGenPrimRef material;
    TfToken materialPurpose = UsdGenTokens().allPurpose;
    int refineLevel = 2;
    UsdGenPrimRef origin;
    UsdGenPrimRef dependency;
};

struct UsdGenGeneration
{
    int64_t id = -1;
    double frame = 0.0;
    // What the terminal type's publisher built (UsdGenPublisherDef); null on
    // the device lane and when no publisher is registered.
    UsdGenRef<const UsdGenPublication> publication;
    // Mutually exclusive with host geometry. Device-aware consumers acquire
    // leases from this immutable payload; never cast its pointers to VtArray.
    std::shared_ptr<const UsdGenDeviceGeneration> device;
    // Present only for a device publication.  It is built from the same accepted
    // descriptor snapshot as `device`, never from a later scene read.
    std::optional<UsdGenDevicePresentationMetadata> devicePresentation;
    // The cooked spec's opaque host data (UsdGenGraphSpec::host): the host
    // names the keys above through it.
    std::shared_ptr<const void> host;
};
using UsdGenGenerationConstPtr = std::shared_ptr<const UsdGenGeneration>;

class UsdGenGenerationStore
{
public:
    using PreparedPublication = UsdGenSharedCell<UsdGenGeneration const>::Prepared;
    /// Private worker candidate stores start from the last published immutable
    /// generation. Discarded candidates consume no public generation ids;
    /// only the accepted publication advances the public sequence.
    explicit UsdGenGenerationStore(UsdGenGenerationConstPtr baseline = {});
    /// Owner-lane only: no concurrent reader of either store.
    UsdGenGenerationStore(UsdGenGenerationStore &&other);
    UsdGenGenerationStore &operator=(UsdGenGenerationStore &&other);
    /// Compatibility convenience for an unfenced publication.  Fenced
    /// callers must use PreparePublication/PublishPrepared so allocation and
    /// displaced-owner destruction occur outside the enclosing domain lock.
    void Publish(UsdGenGeneration gen);

    /// Allocate the immutable COW publication before entering a publication
    /// fence.  The sole command-owner lane guarantees that NextId cannot
    /// change between preparation and the matching PublishPrepared call.
    PreparedPublication PreparePublication(UsdGenGeneration gen) const;

    /// Allocation-free commit of a prepared immutable publication.  Returns
    /// the displaced snapshot so its potentially arbitrary owner destructor
    /// can run after the caller releases its publication fence.
    UsdGenGenerationConstPtr PublishPrepared(
        PreparedPublication prepared) noexcept;

    /// Lock-free snapshot read from any thread.
    UsdGenGenerationConstPtr Get() const noexcept;

    int64_t NextId() const noexcept;
private:
    UsdGenSharedCell<UsdGenGeneration const> _current;
    int64_t _nextId = 0;
};

}  // namespace usdGen

#endif  // USDGEN_GENERATION_STORE_H
