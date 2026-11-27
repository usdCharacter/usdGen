// usdGen engine — generation store (03 §6.1/§6.2).
//
// A generation is immutable once published. The publish/read pair is a
// UsdGenSharedCell (atomic holder pointer, epoch-protected weak lock): no
// lock, and the displaced generation dies when its last reference drops.
// What changed between two generations is the publisher's diff
// (registries/publisher.h).
#ifndef USDGEN_GENERATION_STORE_H
#define USDGEN_GENERATION_STORE_H

#include "usdGen/api/export.h"
#include "usdGen/api/generation.h"
#include "usdGen/api/primRef.h"
#include "usdGen/api/ext/device/deviceGeneration.h"
#include "usdGen/api/sharedCell.h"
#include "usdGen/api/engineTokens.h"
#include "usdGen/api/ext/publisher.h"

#include "usdGenMath/math.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
namespace usdGen {

class USDGEN_INTERNAL_API UsdGenGenerationStore
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
