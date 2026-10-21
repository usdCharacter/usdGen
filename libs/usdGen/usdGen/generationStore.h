// usdGen engine — generation store and publication diff (03 §6.1/§6.2).
//
// A generation is immutable once published. The publish/read pair is
// std::atomic_store/std::atomic_load over std::shared_ptr — NOT TfRefPtr
// (no atomic overloads in 26.08; verified grep of pxr/base/tf/refPtr.h).
// "Did this tile change" is VtArray::IsIdentical() (data-pointer comparison,
// pxr/base/vt/array.h:945-950), not memcmp.
#ifndef USDGEN_GENERATION_STORE_H
#define USDGEN_GENERATION_STORE_H

#include "usdGen/curveBuffer.h"
#include "usdGen/deviceGeneration.h"
#include "usdGen/rbfAnimation.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// Scalar scene-presentation state paired with a CUDA generation.  This is
/// intentionally separate from UsdGenDeviceGeneration: the device payload is
/// geometry/tool data, while these values are the immutable descriptor
/// snapshot that a renderer needs to publish that payload.  No geometry or
/// per-curve host mirror is retained here.
struct UsdGenDevicePresentationMetadata
{
    SdfPath description;
    SdfPath renderNamespace;
    GfMatrix4d xformMatrix{1.0};
    TfToken purpose;
    TfToken visibility;
    SdfPath materialPath;
    TfToken materialPurpose = TfToken("allPurpose");
    int refineLevel = 2;
    SdfPath primOrigin;
    SdfPath dependencySurface;
};

struct UsdGenGeneration
{
    int64_t id = -1;
    double frame = 0.0;
    std::vector<UsdGenTilePublication> tiles;        // C2 payload, sorted by tile id
    std::vector<UsdGenTilePublication> guides;       // guides/<setName> prims (§4.2)
    std::vector<UsdGenInstancerPublication> instancers; // M6
    // The scalp-shadow cap: hair shadowing the emitting surface. Empty when
    // the description has no occluder, no hair over it, or the bake is off.
    UsdGenScalpShadowPublication scalpShadow;
    UsdGenPrimSetSignature signature;
    // Identity of the usdGen:preview colours the tiles carry (0 == none), so a
    // preview edit rebuilds tiles whose geometry did not move.
    uint64_t colorDigest = 0;
    // Mutually exclusive with host geometry. Device-aware consumers acquire
    // leases from this immutable payload; never cast its pointers to VtArray.
    std::shared_ptr<const UsdGenDeviceGeneration> device;
    // Present only for a CUDA publication.  It is built from the same accepted
    // descriptor snapshot as `device`, never from a later scene read.
    std::optional<UsdGenDevicePresentationMetadata> devicePresentation;
    std::shared_ptr<const UsdGenRbfAnimationBinding> animationBinding;
};
using UsdGenGenerationConstPtr = std::shared_ptr<const UsdGenGeneration>;

class UsdGenGenerationStore
{
public:
    using PreparedPublication = UsdGenGenerationConstPtr;
    /// Private worker candidate stores start from the last published immutable
    /// generation. Discarded candidates consume no public generation ids;
    /// only the accepted publication advances the public sequence.
    explicit UsdGenGenerationStore(UsdGenGenerationConstPtr baseline = {});
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

    /// Atomic snapshot read without an application-owned guard. The standard
    /// library does not promise lock-free shared_ptr atomic operations.
    UsdGenGenerationConstPtr Get() const noexcept;

    /// Cheap structural diff of two generations (03 §6.1): payload changes via
    /// IsIdentical, structural changes via the signature. Consumed by the
    /// imaging notice emitter (06 §5.1 rules).
    UsdGenDirtyReport Diff(UsdGenGeneration const &prev,
                           UsdGenGeneration const &next) const;

    int64_t NextId() const noexcept;
private:
    UsdGenGenerationConstPtr _current;
    int64_t _nextId = 0;
};

}  // namespace usdGen

#endif  // USDGEN_GENERATION_STORE_H
