// Authored attribute maps into groom cooking/instancing.
//
// A capture-time cook over a UsdGenAttributeMap (maps/attributeMap.h): it
// samples one channel at every strand root, exactly as the capture loop
// samples, and turns the values into the two decisions the instancing flow
// needs — keep/drop per strand (density culling) and one prototype index
// per kept strand (card/archive variant selection, 07 §10.5: variety comes
// from multiple prototypes).
//
// This is deliberately NOT Pomade's flow and shares no code with it:
// Pomade rasterises region ids on CUDA into regionMap.v<m>.ptx and assigns
// guides through GuideInterpolate (usdGenPomade/pomadeBake.h); this cook runs
// on the CPU, pre-bake, straight from the authored map, with no USD, CUDA
// or Hydra dependency, like its attributeMap/attributeBake siblings. It
// never reads or writes Pomade state.
//
// Rules:
//   - Sampling is bilinear at (face, u, v), the same call the groom
//     preview makes; a root the map cannot sample (face out of range,
//     non-finite uv) reports `defaultValue` and is explicitly dropped,
//     like curveRootCapture's explicit drop entries — no unresolved root
//     is assigned prototype zero.
//   - keep = sampled && value >= threshold. The prototype is
//     min(numPrototypes - 1, floor(clamp01(value) * numPrototypes)), so a
//     single prototype assigns 0 and value 1.0 lands on the last one.
//   - The cook is a pure function of its input: one pass, no threads, so
//     repeated cooks are bitwise identical (the L-3 determinism shape).
//     Digest folds the map digest, the parameters and every assignment,
//     for the capture epoch (07 §5.4).
#ifndef USDGEN_MAPS_ATTRIBUTE_INSTANCE_H
#define USDGEN_MAPS_ATTRIBUTE_INSTANCE_H

#include "usdGen/export.h"
#include "usdGen/maps/attributeMap.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace usdGen {

// One strand root: a parent-mesh face plus its face-local coordinate.
struct UsdGenAttributeInstanceRoot {
    int face = -1;
    float u = 0.0f;
    float v = 0.0f;
};

struct UsdGenAttributeInstanceInput {
    std::shared_ptr<const UsdGenAttributeMap> map;
    int channel = 0;
    std::vector<UsdGenAttributeInstanceRoot> roots;
    // Keep threshold and unreadable-root value (cf. usdGen:map:default,
    // 07 §5.1: the value a failed sample returns).
    float threshold = 0.5f;
    float defaultValue = 0.0f;
    // Card/archive variants; must be >= 1.
    int numPrototypes = 1;
    // Most threads the cook may use; 0 is the process scheduler's width,
    // 1 cooks serially. Results are bit-identical either way.
    int maxThreads = 0;
};

struct UsdGenAttributeInstanceResult {
    // Per strand, in root order: the sampled value (or defaultValue for
    // an unreadable root), the keep flag, and the prototype index
    // (-1 for a dropped strand).
    std::vector<float> values;
    std::vector<uint8_t> keep;
    std::vector<int> prototype;
    // Per prototype: the kept strand indices, in strand order — the same
    // shape as UsdGenInstancerPublication::instanceIndices.
    std::vector<std::vector<int>> instanceIndices;
    size_t kept = 0;
    uint64_t digest = 0;
};

// Cooks `input` into `*result`. Returns false, leaving `*result` untouched
// and setting *error (when non-null), for a null map, a channel outside
// the map, an empty root list, numPrototypes < 1, a non-finite threshold
// or default, or a null result.
USDGEN_CORE_API bool UsdGenAttributeCookInstances(
    UsdGenAttributeInstanceInput const &input,
    UsdGenAttributeInstanceResult *result, std::string *error);

}  // namespace usdGen

#endif  // USDGEN_MAPS_ATTRIBUTE_INSTANCE_H
