// usdGenPomade — model registry implementation (plan/18 §2.1).
#include "usdGenPomade/pomadeRegistry.h"

#include "usdGen/exec/epoch.h"
#include "usdGen/exec/sharedCell.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace usdGenPomade {

using usdGen::UsdGenRef;

// One immutable registry state. Models are sorted by id: `nextId` is
// assigned inside the copy-on-write edit, so a registration takes the id
// after every model already in the table it publishes over.
struct PomadeRegistryTable : usdGen::UsdGenRefCounted {
    std::vector<std::pair<int, PomadeModel *>> models;
    std::vector<PomadePublishTarget *> targets;
    int activeId = 0;
    int nextId = 1;

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

using _Cell = usdGen::UsdGenAtomicRef<PomadeRegistryTable>;
using _TableRef = UsdGenRef<PomadeRegistryTable>;

// The registry's one cell, leaked: nothing is destroyed at exit, so a scene
// index that detaches during static destruction still finds it.
std::atomic<_Cell *> s_table{nullptr};

_Cell &
_Table()
{
    return usdGen::UsdGenLazyLeaked(s_table);
}

_TableRef
_Load()
{
    return _Table().Load();
}

// `edit` changes a copy of the current table and returns false to leave
// the registry untouched; it re-runs on the newer table when a CAS loses.
template <class Edit>
void
_Update(Edit &&edit)
{
    (void)usdGen::UsdGenCowUpdateIf(_Table(), std::forward<Edit>(edit));
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
    _Update([&](PomadeRegistryTable &t) {
        for (auto const &kv : t.models) {
            if (kv.second == model) {
                id = kv.first;
                return false;
            }
        }
        // Larger than every id in this table: appending keeps it sorted.
        id = t.nextId++;
        t.models.emplace_back(id, model);
        return true;
    });
    return id;
}

void
PomadeRegistry::Unregister(int modelId)
{
    _Update([&](PomadeRegistryTable &t) {
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
    _TableRef const t = _Load();
    return t ? int(t->models.size()) : 0;
}

bool
PomadeRegistry::SetActive(int modelId)
{
    bool known = true;
    _Update([&](PomadeRegistryTable &t) {
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
    _TableRef const t = _Load();
    return t ? t->activeId : 0;
}

PomadeModel *
PomadeRegistry::Active() const
{
    _TableRef const t = _Load();
    return t ? t->Find(t->activeId) : nullptr;
}

void
PomadeRegistry::AttachIndex(PomadePublishTarget *target)
{
    if (!target) {
        return;
    }
    _Update([&](PomadeRegistryTable &t) {
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
    _Update([&](PomadeRegistryTable &t) {
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
    _TableRef const t = _Load();
    return t ? int(t->targets.size()) : 0;
}

int
PomadeRegistry::Publish(uint32_t dirtyMask)
{
    // The table reference is held across the calls (never an epoch guard):
    // PublishModel reaches back into the registry and into Hydra observers.
    _TableRef const t = _Load();
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
    _TableRef const t = _Load();
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
