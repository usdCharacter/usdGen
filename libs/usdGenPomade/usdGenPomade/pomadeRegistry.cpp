// usdGenPomade — model registry implementation (plan/18 §2.1).
#include "usdGenPomade/pomadeRegistry.h"

#include "usdGen/epoch.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace usdGenPomade {

using usdGen::UsdGenEpochGuard;
using usdGen::UsdGenRef;

// One immutable registry state. Models are sorted by id.
struct PomadeRegistryTable : usdGen::UsdGenRefCounted {
    std::vector<std::pair<int, PomadeModel *>> models;
    std::vector<PomadePublishTarget *> targets;
    int activeId = 0;

    PomadeModel *Find(int id) const
    {
        auto it = std::lower_bound(
            models.begin(), models.end(), id,
            [](std::pair<int, PomadeModel *> const &kv, int key) {
                return kv.first < key;
            });
        return it != models.end() && it->first == id ? it->second : nullptr;
    }
};

namespace {

using _TableRef = UsdGenRef<PomadeRegistryTable const>;

_TableRef
_Load(std::atomic<PomadeRegistryTable const *> const &cell)
{
    UsdGenEpochGuard guard;
    return _TableRef(cell.load(std::memory_order_acquire));
}

// Copy-on-write update: `edit` changes a copy of the current table and
// returns false to leave the registry untouched. Lock-free: a failed CAS
// means another writer published first, so the edit re-runs on its table.
// The expected table is held by reference, so its address cannot be
// reused under the CAS (no ABA).
template <class Edit>
void
_Update(std::atomic<PomadeRegistryTable const *> &cell, Edit &&edit)
{
    for (;;) {
        _TableRef const current = _Load(cell);
        UsdGenRef<PomadeRegistryTable> next =
            current ? usdGen::UsdGenMakeRef<PomadeRegistryTable>(*current)
                    : usdGen::UsdGenMakeRef<PomadeRegistryTable>();
        if (!edit(*next)) {
            return;
        }
        PomadeRegistryTable const *expected = current.Get();
        if (cell.compare_exchange_strong(expected, next.Get(),
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
            next.Detach();  // the cell's reference
            if (current) {
                usdGen::UsdGenEpochRetire(
                    const_cast<PomadeRegistryTable *>(current.Get()),
                    &usdGen::UsdGenEpochReleaseNode);
            }
            return;
        }
    }
}

}  // namespace

PomadePublishTarget::~PomadePublishTarget() = default;

PomadeRegistry PomadeRegistry::_instance;

PomadeRegistry &
PomadeRegistry::Get()
{
    return _instance;
}

int
PomadeRegistry::Register(PomadeModel *model)
{
    if (!model) {
        return 0;
    }
    int id = 0;
    int fresh = 0;
    _Update(_table, [&](PomadeRegistryTable &t) {
        for (auto const &kv : t.models) {
            if (kv.second == model) {
                id = kv.first;
                return false;
            }
        }
        if (fresh == 0) {
            fresh = _nextId.fetch_add(1, std::memory_order_relaxed);
        }
        id = fresh;
        t.models.emplace_back(id, model);  // ids only grow: stays sorted
        return true;
    });
    return id;
}

void
PomadeRegistry::Unregister(int modelId)
{
    _Update(_table, [&](PomadeRegistryTable &t) {
        auto it = std::find_if(
            t.models.begin(), t.models.end(),
            [&](std::pair<int, PomadeModel *> const &kv) {
                return kv.first == modelId;
            });
        bool changed = false;
        if (t.activeId == modelId) {
            t.activeId = 0;
            changed = true;
        }
        if (it != t.models.end()) {
            t.models.erase(it);
            changed = true;
        }
        return changed;
    });
}

int
PomadeRegistry::ModelCount() const
{
    _TableRef const t = _Load(_table);
    return t ? int(t->models.size()) : 0;
}

bool
PomadeRegistry::SetActive(int modelId)
{
    bool known = true;
    _Update(_table, [&](PomadeRegistryTable &t) {
        if (modelId != 0 && !t.Find(modelId)) {
            known = false;
            return false;
        }
        known = true;
        if (t.activeId == modelId) {
            return false;
        }
        t.activeId = modelId;
        return true;
    });
    return known;
}

int
PomadeRegistry::GetActiveId() const
{
    _TableRef const t = _Load(_table);
    return t ? t->activeId : 0;
}

PomadeModel *
PomadeRegistry::Active() const
{
    _TableRef const t = _Load(_table);
    return t ? t->Find(t->activeId) : nullptr;
}

void
PomadeRegistry::AttachIndex(PomadePublishTarget *target)
{
    if (!target) {
        return;
    }
    _Update(_table, [&](PomadeRegistryTable &t) {
        if (std::find(t.targets.begin(), t.targets.end(), target) !=
            t.targets.end()) {
            return false;
        }
        t.targets.push_back(target);
        return true;
    });
}

void
PomadeRegistry::DetachIndex(PomadePublishTarget *target)
{
    _Update(_table, [&](PomadeRegistryTable &t) {
        auto const end =
            std::remove(t.targets.begin(), t.targets.end(), target);
        if (end == t.targets.end()) {
            return false;
        }
        t.targets.erase(end, t.targets.end());
        return true;
    });
}

int
PomadeRegistry::IndexCount() const
{
    _TableRef const t = _Load(_table);
    return t ? int(t->targets.size()) : 0;
}

int
PomadeRegistry::Publish(uint32_t dirtyMask)
{
    // The table reference is held across the calls (never an epoch guard):
    // PublishModel reaches back into the registry and into Hydra observers.
    _TableRef const t = _Load(_table);
    if (!t) {
        return 0;
    }
    PomadeModel *const model = t->Find(t->activeId);
    int published = 0;
    for (PomadePublishTarget *target : t->targets) {
        if (target->PublishModel(model, dirtyMask)) {
            ++published;
        }
    }
    return published;
}

bool
PomadeRegistry::QueryPublishedLevel(int level, int *outFaceCount,
                                   int *outPointCount,
                                   int *outTubeCount) const
{
    _TableRef const t = _Load(_table);
    if (!t) {
        return false;
    }
    for (PomadePublishTarget const *target : t->targets) {
        if (target->QueryPublishedLevel(level, outFaceCount, outPointCount,
                                        outTubeCount)) {
            return true;
        }
    }
    return false;
}

} // namespace usdGenPomade
