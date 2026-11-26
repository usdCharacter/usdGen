// usdGen engine — generation store implementation.
//
// Publish/Get go through a UsdGenSharedCell (I7, 03 §6.1): an atomic holder
// pointer with epoch-protected weak locks, validated by gate SI-4 (8 readers
// x 100 publishes, 0 torn reads) and testUsdGenSharedCell.
#include "usdGen/exec/generationStore.h"

namespace usdGen {

UsdGenGenerationStore::UsdGenGenerationStore(UsdGenGenerationConstPtr baseline)
{
    if (baseline) _nextId = baseline->id + 1;
    _current.Store(std::move(baseline));
}

UsdGenGenerationStore::UsdGenGenerationStore(UsdGenGenerationStore &&other)
    : _nextId(other._nextId)
{
    _current.Store(other._current.Exchange(UsdGenGenerationConstPtr{}));
}

UsdGenGenerationStore &UsdGenGenerationStore::operator=(UsdGenGenerationStore &&other)
{
    if (this != &other) {
        // Keep the displaced value alive until both cells are consistent.
        UsdGenGenerationConstPtr displaced =
            _current.Exchange(other._current.Exchange(UsdGenGenerationConstPtr{}));
        _nextId = other._nextId;
    }
    return *this;
}

void UsdGenGenerationStore::Publish(UsdGenGeneration gen)
{
    PreparedPublication prepared = PreparePublication(std::move(gen));
    // Keep the displaced immutable owner alive until this call has completed.
    auto retired = PublishPrepared(std::move(prepared));
    (void)retired;
}

UsdGenGenerationStore::PreparedPublication
UsdGenGenerationStore::PreparePublication(UsdGenGeneration gen) const
{
    // Zero-based ids: the first published generation has id 0, so across any
    // published history `id == Stats().commits - 1` (03 §9: exactly one publish
    // per committed request; superseded or aborted commits allocate no id).
    gen.id = _nextId;
    return UsdGenSharedCell<UsdGenGeneration const>::Prepare(
        std::make_shared<const UsdGenGeneration>(std::move(gen)));
}

UsdGenGenerationConstPtr UsdGenGenerationStore::PublishPrepared(
    PreparedPublication prepared) noexcept
{
    if (!prepared || prepared->id != _nextId) return prepared.Value();
    ++_nextId;
    return _current.Exchange(std::move(prepared));
}

UsdGenGenerationConstPtr UsdGenGenerationStore::Get() const noexcept
{
    return _current.Load();
}

int64_t UsdGenGenerationStore::NextId() const noexcept { return _nextId; }
}  // namespace usdGen
