// testUsdGenPomadeCommit — T1: the P1 asynchronous commit pipeline.
//
// Plan/17 P1 exit: PomadeCommitter (worker + idle swap), hydrate, "Save
// groom", and the GuideInterpolate relationship fill-in. Proven here:
//   * a commit lands the groom subtree (tube, Guides, RegionMap, RegionExpr)
//     plus the GuideInterpolate op in one idle swap;
//   * version coalescing (N enqueues under pause produce one build);
//   * cancellation (a built layer superseded before the swap never lands);
//   * no swap runs during a gesture; no stale swap runs while a newer build
//     is in flight;
//   * the partial-transfer fallback converges to the same stage content, one
//     subtree per idle slot;
//   * hydrate round-trips the model bit-exactly (tube + guides);
//   * the fill-in creates the op only when absent and only touches empty
//     relationships/connections;
//   * "Save groom" writes .usdc and re-parents the live sublayer beneath it;
//   * TN-4: the reference-scale swap stays near the 5 ms slot budget
//     (wall-clock sample allows one scheduler slice, <= 8 ms);
//   * the committer C ABI drives the same pipeline over identifiers;
//   * P4 hooks: subdivide params + lock flags ride the commit and survive
//     hydrate (childIndex/deltas/level channels land with P4).
//   * P6 reload: Detach idles the worker (model survives, swap reports
//     Detached, shelf layer counts as dropped); Reattach resets the
//     committed lineage and the next swap re-creates the groom prim,
//     carrying the post-detach edits; the C ABI pair mirrors this.
//
// Needs the schema plugin (typed groom prims), so this registers with the
// _usdgen_m1_env environment like every adapter-dependent T1 test.

#include "utils/pxrInterop.h"
#include "usdGenPomade/pomadeApi.h"
#include "usdGenPomade/pomadeApiStage.h"
#include "usdGenPomade/pomadeCommit.h"
#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadeScalp.h"
#include "usdGenPomade/pomadeTube.h"

#include "usdGenImaging/usdGenImagingSession.h"

#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/subset.h"
#include "pxr/usd/usdGeom/tokens.h"
#include "pxr/usd/usdGeom/xform.h"
#include "pxr/usd/sdf/assetPath.h"
#include "pxr/usd/sdf/attributeSpec.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/sdf/listOp.h"
#include "pxr/usd/sdf/primSpec.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/relationship.h"
#include "pxr/usd/usd/stage.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
    std::fflush(stdout);
}

bool DoubleEq(double a, double b, double eps = 1e-9)
{
    return std::abs(a - b) <= eps;
}

// Pump idle swaps until `want` commits or the timeout expires.
bool WaitCommitted(usdGenPomade::PomadeCommitter &committer,
                   SdfLayerHandle const &live, uint64_t want,
                   int timeoutMs = 5000)
{
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    for (;;) {
        committer.SwapIfIdle(live, /*gestureActive*/ false);
        if (committer.CommittedVersion() >= want) {
            return true;
        }
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

UsdStageRefPtr MakeStage(SdfLayerHandle const &live,
                         bool withDescription = true)
{
    UsdStageRefPtr stage = UsdStage::CreateInMemory("pomadeCommit");
    if (withDescription) {
        stage->DefinePrim(SdfPath("/Groom/Hair"), TfToken("UsdGenDescription"));
        stage->DefinePrim(SdfPath("/Groom/Hair/Ops"), TfToken("Scope"));
    }
    stage->GetSessionLayer()->InsertSubLayerPath(live->GetIdentifier(), 0);
    return stage;
}

bool GetPoints(UsdStagePtr const &stage, SdfPath const &path, VtVec3fArray *out)
{
    UsdPrim const prim = stage->GetPrimAtPath(path);
    return bool(prim) && prim.GetAttribute(TfToken("points")).Get(out);
}

SdfPathVector Targets(UsdRelationship const &rel)
{
    SdfPathVector targets;
    rel.GetTargets(&targets);
    return targets;
}

SdfPathVector Connections(UsdAttribute const &attr)
{
    SdfPathVector conns;
    attr.GetConnections(&conns);
    return conns;
}

// V0b: bit-exact comparison of two tubes' authored shapes and deltas.
bool SameSections(std::vector<usdGenPomade::PomadeTubeSection> const &a,
                  std::vector<usdGenPomade::PomadeTubeSection> const &b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].t != b[i].t || a[i].scale != b[i].scale ||
            a[i].twist != b[i].twist || a[i].u != b[i].u ||
            a[i].v != b[i].v) {
            return false;
        }
    }
    return true;
}

bool SameDesc(usdGenPomade::PomadeTubeDesc const &a,
              usdGenPomade::PomadeTubeDesc const &b)
{
    if (a.inheritedBoundaryBindings.size() !=
        b.inheritedBoundaryBindings.size()) {
        return false;
    }
    for (size_t i = 0; i < a.inheritedBoundaryBindings.size(); ++i) {
        usdGenPomade::PomadeParentBoundaryBinding const &x =
            a.inheritedBoundaryBindings[i];
        usdGenPomade::PomadeParentBoundaryBinding const &y =
            b.inheritedBoundaryBindings[i];
        if (x.section != y.section || x.parentSlot != y.parentSlot ||
            x.childSlot != y.childSlot) {
            return false;
        }
    }
    return a.centerX == b.centerX && a.centerY == b.centerY &&
           a.centerZ == b.centerZ && a.ringVerts == b.ringVerts &&
           a.tubeId == b.tubeId && a.regionId == b.regionId &&
           a.frameReference == b.frameReference &&
           SameSections(a.sections, b.sections);
}

bool SameDeltas(usdGenPomade::PomadeShapeDeltas const &a,
                usdGenPomade::PomadeShapeDeltas const &b)
{
    return a.centerDu == b.centerDu && a.centerDv == b.centerDv &&
           a.centerDw == b.centerDw && SameSections(a.sections, b.sections);
}

bool SameTessellation(usdGenPomade::PomadeTubeDesc const &a,
                      usdGenPomade::PomadeTubeDesc const &b)
{
    std::vector<usdGenPomade::PomadeFrame> af, bf;
    std::vector<float> ap, an, at, bp, bn, bt;
    std::string err;
    return usdGenPomade::PomadeTubeFramesCpu(a, &af, &err) &&
           usdGenPomade::PomadeTubeFramesCpu(b, &bf, &err) &&
           usdGenPomade::PomadeTessellateCpu(a, af, 4, &ap, &an, &at, &err) &&
           usdGenPomade::PomadeTessellateCpu(b, bf, 4, &bp, &bn, &bt, &err) &&
           ap == bp && an == bn && at == bt;
}

// Commit/hydrate must preserve enough attachment identity that a later graph
// reposition can carry the whole root wall. Compare K5 samples instead of
// only center CVs: a stale base ring can leave centers moving while the
// visible footprint remains behind.
struct TubeWorldGeometry {
    usdGenPomade::PomadeTubeDesc desc;
    std::vector<float> positions;
};

bool CaptureTubeWorld(usdGenPomade::PomadeModel const &model, int tubeId,
                      TubeWorldGeometry *out)
{
    if (!out || !model.GetTubeDesc(tubeId, &out->desc)) return false;
    std::vector<usdGenPomade::PomadeFrame> frames;
    std::vector<float> normals, ringT;
    std::string error;
    return usdGenPomade::PomadeTubeFramesCpu(out->desc, &frames, &error) &&
           usdGenPomade::PomadeTessellateCpu(out->desc, frames, 4,
                                           &out->positions, &normals, &ringT,
                                           &error);
}

bool MovedWorld(TubeWorldGeometry const &before, TubeWorldGeometry const &after)
{
    if (before.positions.size() != after.positions.size()) return false;
    for (size_t i = 0; i < before.positions.size(); ++i) {
        if (std::abs(before.positions[i] - after.positions[i]) > 2e-4f) {
            return true;
        }
    }
    return false;
}

bool MovedBaseRing(TubeWorldGeometry const &before, TubeWorldGeometry const &after)
{
    if (before.desc.ringVerts < 3 ||
        before.desc.ringVerts != after.desc.ringVerts) return false;
    size_t const values = size_t(before.desc.ringVerts) * 3;
    if (before.positions.size() < values || after.positions.size() < values) {
        return false;
    }
    for (size_t i = 0; i < values; ++i) {
        if (std::abs(before.positions[i] - after.positions[i]) > 2e-4f) {
            return true;
        }
    }
    return false;
}

bool SameSectionsAboveBase(usdGenPomade::PomadeTubeDesc const &before,
                           usdGenPomade::PomadeTubeDesc const &after)
{
    if (before.sections.size() != after.sections.size()) return false;
    for (size_t section = 1; section < before.sections.size(); ++section) {
        usdGenPomade::PomadeTubeSection const &a = before.sections[section];
        usdGenPomade::PomadeTubeSection const &b = after.sections[section];
        if (a.t != b.t || a.scale != b.scale || a.twist != b.twist ||
            a.u != b.u || a.v != b.v) return false;
    }
    return true;
}

bool SameSection(usdGenPomade::PomadeTubeSection const &a,
                 usdGenPomade::PomadeTubeSection const &b)
{
    return a.t == b.t && a.scale == b.scale && a.twist == b.twist &&
           a.u == b.u && a.v == b.v;
}

bool RootBaseWorld(usdGenPomade::PomadeTubeDesc const &desc,
                   std::vector<std::array<float, 3>> *out);

bool RootBaseMatchesRegion(usdGenPomade::PomadeModel const &model,
                           usdGenPomade::PomadeTubeDesc const &desc)
{
    using namespace usdGenPomade;
    PomadeGraphRegion const *region = nullptr;
    for (PomadeGraphRegion const &candidate : model.GetGraph().Regions()) {
        if (!candidate.isOutside && candidate.id == desc.regionId) {
            region = &candidate;
            break;
        }
    }
    if (!region || desc.ringVerts != int(region->loop.size()) ||
        desc.sections.empty() || desc.sections.front().u.size() !=
            size_t(desc.ringVerts) || desc.sections.front().v.size() !=
            size_t(desc.ringVerts)) return false;
    std::vector<PomadeFrame> frames;
    std::string error;
    if (!PomadeTubeFramesCpu(desc, &frames, &error) || frames.empty()) return false;
    PomadeTubeSection const &section = desc.sections.front();
    float const c = std::cos(section.twist), s = std::sin(section.twist);
    std::vector<std::array<float, 3>> actual;
    std::vector<std::array<float, 3>> expected;
    actual.reserve(size_t(desc.ringVerts));
    expected.reserve(size_t(desc.ringVerts));
    size_t seam = 0;
    for (size_t i = 1; i < region->loop.size(); ++i)
        if (region->loop[i] < region->loop[seam]) seam = i;
    for (int slot = 0; slot < desc.ringVerts; ++slot) {
        float const u = section.scale * section.u[size_t(slot)];
        float const v = section.scale * section.v[size_t(slot)];
        actual.push_back({{desc.centerX[0] + frames[0].nx * (u * c - v * s) +
                                      frames[0].bx * (u * s + v * c),
                           desc.centerY[0] + frames[0].ny * (u * c - v * s) +
                                      frames[0].by * (u * s + v * c),
                           desc.centerZ[0] + frames[0].nz * (u * c - v * s) +
                                      frames[0].bz * (u * s + v * c)}});
        int const nodeId = region->loop[(seam + size_t(slot)) % region->loop.size()];
        PomadeGraphNode const *node = model.GetGraph().FindNode(nodeId);
        if (!node) return false;
        expected.push_back({{node->p[0], node->p[1], node->p[2]}});
    }
    // Region fitting consistently uses the minimum stable id as its seam,
    // while a valid reversed support orientation reverses the remaining slots.
    bool forward = true, reverse = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        size_t const reversed = i == 0 ? 0 : actual.size() - i;
        for (int axis = 0; axis < 3; ++axis) {
            forward = forward && std::abs(actual[i][axis] - expected[i][axis]) <= 4e-4f;
            reverse = reverse && std::abs(actual[i][axis] - expected[reversed][axis]) <= 4e-4f;
        }
    }
    return forward || reverse;
}

bool RootBaseWorld(usdGenPomade::PomadeTubeDesc const &desc,
                   std::vector<std::array<float, 3>> *out)
{
    using namespace usdGenPomade;
    if (!out || desc.ringVerts < 3 || desc.sections.empty() ||
        desc.sections.front().u.size() != size_t(desc.ringVerts) ||
        desc.sections.front().v.size() != size_t(desc.ringVerts)) return false;
    std::vector<PomadeFrame> frames;
    std::string error;
    if (!PomadeTubeFramesCpu(desc, &frames, &error) || frames.empty()) return false;
    PomadeTubeSection const &section = desc.sections.front();
    float const c = std::cos(section.twist), s = std::sin(section.twist);
    out->clear();
    out->reserve(size_t(desc.ringVerts));
    for (int slot = 0; slot < desc.ringVerts; ++slot) {
        float const u = section.scale * section.u[size_t(slot)];
        float const v = section.scale * section.v[size_t(slot)];
        out->push_back({{desc.centerX[0] + frames[0].nx * (u * c - v * s) +
                                      frames[0].bx * (u * s + v * c),
                         desc.centerY[0] + frames[0].ny * (u * c - v * s) +
                                      frames[0].by * (u * s + v * c),
                         desc.centerZ[0] + frames[0].nz * (u * c - v * s) +
                                      frames[0].bz * (u * s + v * c)}});
    }
    return true;
}

bool SectionWorld(usdGenPomade::PomadeTubeDesc const &desc, int sectionIndex,
                  std::vector<std::array<float, 3>> *out)
{
    using namespace usdGenPomade;
    if (!out || sectionIndex < 0 || size_t(sectionIndex) >= desc.sections.size() ||
        desc.ringVerts < 3) return false;
    PomadeTubeSection const &section = desc.sections[size_t(sectionIndex)];
    if (section.u.size() != size_t(desc.ringVerts) ||
        section.v.size() != size_t(desc.ringVerts)) return false;
    std::vector<PomadeFrame> frames;
    std::string error;
    if (!PomadeTubeFramesCpu(desc, &frames, &error) ||
        size_t(sectionIndex) >= frames.size()) return false;
    PomadeFrame const &frame = frames[size_t(sectionIndex)];
    float const c = std::cos(section.twist), s = std::sin(section.twist);
    out->clear();
    out->reserve(size_t(desc.ringVerts));
    for (int slot = 0; slot < desc.ringVerts; ++slot) {
        float const u = section.scale * section.u[size_t(slot)];
        float const v = section.scale * section.v[size_t(slot)];
        out->push_back({{desc.centerX[size_t(sectionIndex)] +
                             frame.nx * (u * c - v * s) +
                             frame.bx * (u * s + v * c),
                         desc.centerY[size_t(sectionIndex)] +
                             frame.ny * (u * c - v * s) +
                             frame.by * (u * s + v * c),
                         desc.centerZ[size_t(sectionIndex)] +
                             frame.nz * (u * c - v * s) +
                             frame.bz * (u * s + v * c)}});
    }
    return true;
}

// A planar support move must carry every upper wall sample by one common
// rigid translation. Section zero is intentionally excluded: attachment
// conformance is allowed to replace only that footprint.
bool UpperSectionsShareTranslation(usdGenPomade::PomadeTubeDesc const &before,
                                   usdGenPomade::PomadeTubeDesc const &after)
{
    if (before.sections.size() != after.sections.size() ||
        before.sections.size() < 2) return false;
    bool haveDelta = false;
    float delta[3] = {};
    for (size_t section = 1; section < before.sections.size(); ++section) {
        std::vector<std::array<float, 3>> a, b;
        if (!SectionWorld(before, int(section), &a) ||
            !SectionWorld(after, int(section), &b) || a.size() != b.size()) {
            return false;
        }
        for (size_t slot = 0; slot < a.size(); ++slot) {
            float const d[3] = {b[slot][0] - a[slot][0],
                                b[slot][1] - a[slot][1],
                                b[slot][2] - a[slot][2]};
            if (!haveDelta) {
                delta[0] = d[0]; delta[1] = d[1]; delta[2] = d[2];
                haveDelta = true;
            } else if (std::abs(d[0] - delta[0]) > 3e-4f ||
                       std::abs(d[1] - delta[1]) > 3e-4f ||
                       std::abs(d[2] - delta[2]) > 3e-4f) {
                return false;
            }
        }
    }
    return haveDelta;
}

bool SameCenterCageShape(usdGenPomade::PomadeTubeDesc const &before,
                          usdGenPomade::PomadeTubeDesc const &after)
{
    if (before.centerX.size() != after.centerX.size() ||
        before.centerY.size() != after.centerY.size() ||
        before.centerZ.size() != after.centerZ.size() ||
        before.centerX.size() < 2) return false;
    for (size_t i = 1; i < before.centerX.size(); ++i) {
        float const beforeDelta[3] = {before.centerX[i] - before.centerX[0],
                                      before.centerY[i] - before.centerY[0],
                                      before.centerZ[i] - before.centerZ[0]};
        float const afterDelta[3] = {after.centerX[i] - after.centerX[0],
                                     after.centerY[i] - after.centerY[0],
                                     after.centerZ[i] - after.centerZ[0]};
        for (int axis = 0; axis < 3; ++axis) {
            if (std::abs(beforeDelta[axis] - afterDelta[axis]) > 3e-5f) {
                return false;
            }
        }
    }
    return before.frameReference == after.frameReference;
}

struct RootEdgeBinding {
    int a = -1;
    int b = -1;
    float t = 0.0f;
};

bool BindRootSlotsToRegionEdges(usdGenPomade::PomadeModel const &model,
                                usdGenPomade::PomadeTubeDesc const &desc,
                                std::vector<RootEdgeBinding> *out)
{
    using namespace usdGenPomade;
    if (!out) return false;
    PomadeGraphRegion const *region = nullptr;
    for (PomadeGraphRegion const &candidate : model.GetGraph().Regions()) {
        if (!candidate.isOutside && candidate.id == desc.regionId) {
            region = &candidate;
            break;
        }
    }
    std::vector<std::array<float, 3>> points;
    if (!region || !RootBaseWorld(desc, &points)) return false;
    out->clear();
    for (std::array<float, 3> const &point : points) {
        RootEdgeBinding binding;
        float best = std::numeric_limits<float>::infinity();
        for (size_t edge = 0; edge < region->loop.size(); ++edge) {
            int const a = region->loop[edge];
            int const b = region->loop[(edge + 1) % region->loop.size()];
            PomadeGraphNode const *pa = model.GetGraph().FindNode(a);
            PomadeGraphNode const *pb = model.GetGraph().FindNode(b);
            if (!pa || !pb) return false;
            float const dx = pb->p[0] - pa->p[0];
            float const dy = pb->p[1] - pa->p[1];
            float const dz = pb->p[2] - pa->p[2];
            float const length2 = dx * dx + dy * dy + dz * dz;
            if (!(length2 > 1e-12f)) return false;
            float t = ((point[0] - pa->p[0]) * dx +
                       (point[1] - pa->p[1]) * dy +
                       (point[2] - pa->p[2]) * dz) / length2;
            t = std::max(0.0f, std::min(1.0f, t));
            float const ex = point[0] - (pa->p[0] + dx * t);
            float const ey = point[1] - (pa->p[1] + dy * t);
            float const ez = point[2] - (pa->p[2] + dz * t);
            float const distance2 = ex * ex + ey * ey + ez * ez;
            if (distance2 < best) {
                best = distance2;
                binding = {a, b, t};
            }
        }
        if (best > 2e-7f) return false;
        out->push_back(binding);
    }
    return true;
}

bool RootSlotsFollowMaterialEdges(usdGenPomade::PomadeModel const &model,
                                  usdGenPomade::PomadeTubeDesc const &desc,
                                  std::vector<RootEdgeBinding> const &bindings)
{
    std::vector<std::array<float, 3>> actual;
    if (!RootBaseWorld(desc, &actual) || actual.size() != bindings.size()) return false;
    std::vector<bool> used(actual.size(), false);
    for (RootEdgeBinding const &binding : bindings) {
        usdGenPomade::PomadeGraphNode const *a = model.GetGraph().FindNode(binding.a);
        usdGenPomade::PomadeGraphNode const *b = model.GetGraph().FindNode(binding.b);
        if (!a || !b) return false;
        float const expected[3] = {a->p[0] + (b->p[0] - a->p[0]) * binding.t,
                                   a->p[1] + (b->p[1] - a->p[1]) * binding.t,
                                   a->p[2] + (b->p[2] - a->p[2]) * binding.t};
        bool found = false;
        for (size_t slot = 0; slot < actual.size(); ++slot) {
            if (used[slot]) continue;
            float const dx = actual[slot][0] - expected[0];
            float const dy = actual[slot][1] - expected[1];
            float const dz = actual[slot][2] - expected[2];
            if (dx * dx + dy * dy + dz * dz <= 2e-7f) {
                used[slot] = true;
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

// Keep the explicit-count failure actionable.  The native recovery path uses
// the frozen, pre-transport root ring, so these values distinguish a bad test
// fixture from a lost edge/t mapping without making successful test output
// noisy.
void PrintRootSlotDiagnostics(usdGenPomade::PomadeModel const &model,
                              usdGenPomade::PomadeTubeDesc const &desc,
                              std::vector<RootEdgeBinding> const &bindings,
                              char const *label)
{
    std::vector<std::array<float, 3>> actual;
    if (!RootBaseWorld(desc, &actual)) {
        std::printf("V6 explicit slots %s: cannot reconstruct root ring; model=%s\n",
                    label, model.GetDiagnostic());
        return;
    }
    std::printf("V6 explicit slots %s: tube=%d rv=%d bindings=%zu model=%s\n",
                label, desc.tubeId, desc.ringVerts, bindings.size(),
                model.GetDiagnostic());
    for (size_t i = 0; i < bindings.size(); ++i) {
        RootEdgeBinding const &binding = bindings[i];
        usdGenPomade::PomadeGraphNode const *a = model.GetGraph().FindNode(binding.a);
        usdGenPomade::PomadeGraphNode const *b = model.GetGraph().FindNode(binding.b);
        if (!a || !b) {
            std::printf("  bind[%zu]=missing %d->%d t=%.8g\n", i, binding.a,
                        binding.b, binding.t);
            continue;
        }
        float const expected[3] = {a->p[0] + (b->p[0] - a->p[0]) * binding.t,
                                   a->p[1] + (b->p[1] - a->p[1]) * binding.t,
                                   a->p[2] + (b->p[2] - a->p[2]) * binding.t};
        float best2 = std::numeric_limits<float>::infinity();
        size_t best = 0;
        for (size_t slot = 0; slot < actual.size(); ++slot) {
            float const dx = actual[slot][0] - expected[0];
            float const dy = actual[slot][1] - expected[1];
            float const dz = actual[slot][2] - expected[2];
            float const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best2) { best2 = d2; best = slot; }
        }
        std::printf("  bind[%zu]=%d->%d t=%.8g nearestSlot=%zu d2=%.8g\n",
                    i, binding.a, binding.b, binding.t, best, best2);
    }
}

bool ChildRootRefreshesButUpperSculptRemains(usdGenPomade::PomadeModel::TubeRecord const &record)
{
    if (record.actual.sections.empty() || record.derived.sections.empty() ||
        record.actual.sections.size() != record.derived.sections.size() ||
        record.actual.centerX.empty() || record.derived.centerX.empty()) return false;
    bool upperDiffers = false;
    for (size_t i = 1; i < record.actual.sections.size(); ++i) {
        upperDiffers = upperDiffers || !SameSection(record.actual.sections[i],
                                                    record.derived.sections[i]);
    }
    // Attachment reconciliation deliberately retains the transported child
    // center/frame reference so it cannot shear the authored upper cage.
    // The fresh K14 partition is therefore equivalent at the inherited root
    // ring in world space, not necessarily as raw center/chart bytes.
    std::vector<std::array<float, 3>> actualRoot, derivedRoot;
    bool const rootMatches = RootBaseWorld(record.actual, &actualRoot) &&
        RootBaseWorld(record.derived, &derivedRoot) &&
        actualRoot.size() == derivedRoot.size();
    if (!rootMatches) return false;
    for (size_t i = 0; i < actualRoot.size(); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            if (std::abs(actualRoot[i][axis] - derivedRoot[i][axis]) >
                4e-4f) return false;
        }
    }
    return upperDiffers;
}

bool SameTranslation(TubeWorldGeometry const &before,
                     TubeWorldGeometry const &after, float dx, float dy, float dz)
{
    if (before.positions.size() != after.positions.size()) return false;
    float const delta[3] = {dx, dy, dz};
    for (size_t i = 0; i < before.positions.size(); ++i) {
        if (std::abs((after.positions[i] - before.positions[i]) -
                     delta[i % 3]) > 4e-4f) return false;
    }
    return true;
}

// -- V6: a two-region scalp (plan/18 §7 G14) ------------------------------
//
// One L1 tube per closed region is the whole point of the region->tube map,
// and the committer has to serialise every one of them. These three helpers
// build the smallest scalp that carries two adjoining regions: an n x n grid
// in the XZ plane, and two rects sharing the edge at x = n/2.
struct Grid {
    std::vector<float> points;
    std::vector<int> counts;
    std::vector<int> indices;
};

Grid MakeGrid(int n)
{
    Grid grid;
    for (int ix = 0; ix <= n; ++ix) {
        for (int iz = 0; iz <= n; ++iz) {
            grid.points.push_back(float(ix));
            grid.points.push_back(0.0f);
            grid.points.push_back(float(iz));
        }
    }
    auto pid = [&](int ix, int iz) { return ix * (n + 1) + iz; };
    for (int ix = 0; ix < n; ++ix) {
        for (int iz = 0; iz < n; ++iz) {
            grid.counts.push_back(4);
            grid.indices.push_back(pid(ix, iz));
            grid.indices.push_back(pid(ix, iz + 1));
            grid.indices.push_back(pid(ix + 1, iz + 1));
            grid.indices.push_back(pid(ix + 1, iz));
        }
    }
    return grid;
}

usdGenPomade::PomadeHit Locate(usdGenPomade::PomadeScalpMesh const &mesh, int n,
                             float x, float z)
{
    using namespace usdGenPomade;
    int const ix = std::min(std::max(int(std::floor(x)), 0), n - 1);
    int const iz = std::min(std::max(int(std::floor(z)), 0), n - 1);
    PomadeHit hit;
    hit.hit = true;
    hit.faceId = ix * n + iz;
    hit.u = z - float(iz);
    hit.v = x - float(ix);
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    if (!PomadeFacePosition(mesh, hit.faceId, hit.u, hit.v, &px, &py, &pz)) {
        hit.hit = false;
        return hit;
    }
    hit.px = px;
    hit.py = py;
    hit.pz = pz;
    hit.nx = 0.0f;
    hit.ny = 1.0f;
    hit.nz = 0.0f;
    return hit;
}

// A: x in [0, n/2]; B: x in [n/2, n]; both z in [1, n-1]. They share the
// edge a1-a2, which is what makes them two regions of ONE planar graph
// rather than two graphs.
void BuildTwoRegions(usdGenPomade::PomadeModel *model,
                     usdGenPomade::PomadeScalpMesh const &mesh, int n)
{
    float const mid = float(n) / 2.0f;
    float const farX = float(n);  // `far` is a windows.h macro
    float const zLo = 1.0f;
    float const zHi = float(n) - 1.0f;
    int const a0 = model->GraphAddNode(Locate(mesh, n, 0.0f, zLo));
    int const a1 = model->GraphAddNode(Locate(mesh, n, mid, zLo));
    int const a2 = model->GraphAddNode(Locate(mesh, n, mid, zHi));
    int const a3 = model->GraphAddNode(Locate(mesh, n, 0.0f, zHi));
    model->GraphConnect(a0, a1);
    model->GraphConnect(a1, a2);
    model->GraphConnect(a2, a3);
    model->GraphConnect(a3, a0);
    int const b1 = model->GraphAddNode(Locate(mesh, n, farX, zLo));
    int const b2 = model->GraphAddNode(Locate(mesh, n, farX, zHi));
    model->GraphConnect(a1, b1);
    model->GraphConnect(b1, b2);
    model->GraphConnect(b2, a2);
}

// A small disconnected triangle below the two adjoining V6 regions. It gives
// the transport check both an unrelated L1 owner and the default three-column
// region-CV fixture in the same graph/model.
int AddIsolatedRegion(usdGenPomade::PomadeModel *model,
                      usdGenPomade::PomadeScalpMesh const &mesh, int n,
                      std::vector<int> *outNodes = nullptr)
{
    int const a = model->GraphAddNode(Locate(mesh, n, 0.20f, 0.15f));
    int const b = model->GraphAddNode(Locate(mesh, n, 1.20f, 0.15f));
    int const c = model->GraphAddNode(Locate(mesh, n, 0.70f, 0.75f));
    if (a < 0 || b < 0 || c < 0 || model->GraphConnect(a, b) < 0 ||
        model->GraphConnect(b, c) < 0 || model->GraphConnect(c, a) < 0 ||
        !model->Rasterise()) return -1;
    if (outNodes) *outNodes = {a, b, c};
    for (usdGenPomade::PomadeGraphRegion const &region : model->GetGraph().Regions()) {
        if (!region.isOutside &&
            std::find(region.loop.begin(), region.loop.end(), a) != region.loop.end()) {
            return region.id;
        }
    }
    return -1;
}

} // namespace

int
main()
{
    using usdGenPomade::PomadeCommitter;
    using usdGenPomade::PomadeCommitPaths;
    using usdGenPomade::PomadeModel;

    PomadeCommitPaths paths;
    paths.groomPath = SdfPath("/PomadeGroom");
    paths.descriptionPath = SdfPath("/Groom/Hair");

    // -- basic commit ------------------------------------------------------
    PomadeModel model;
    Check(model.BuildTestTube(), "model builds the test tube");
    PomadeModel::FillParams fill;
    fill.density = 16.0f;
    fill.cvCount = 8;
    fill.seed = 7;
    fill.edgeBias = 0.25f;
    Check(model.SetFillParams(fill), "model takes P1 fill params");
    uint64_t const v1 = model.GetVersion();

    SdfLayerRefPtr live = SdfLayer::CreateAnonymous("usdGenPomade-live");
    UsdStageRefPtr stage = MakeStage(live);
    PomadeCommitter committer(&model, paths);
    committer.Enqueue(stage);
    Check(WaitCommitted(committer, live, v1), "first commit swaps at idle");
    Check(committer.CommittedVersion() == v1, "committed version is v1");
    Check(committer.BuildCount() == 1, "one enqueue produces one build");

    UsdPrim const groom = stage->GetPrimAtPath(SdfPath("/PomadeGroom"));
    Check(bool(groom) && groom.GetTypeName() == TfToken("UsdGenPomadeGroom"),
          "live carries the UsdGenPomadeGroom prim");
    TfToken groomVersion;
    Check(groom.GetAttribute(TfToken("usdGen:pomade:version")).Get(&groomVersion) &&
              groomVersion == TfToken("1"),
          "groom version token is \"1\"");
    Check(Targets(groom.GetRelationship(TfToken("usdGen:pomade:description"))) ==
              SdfPathVector{SdfPath("/Groom/Hair")},
          "groom links the description");
    VtVec3fArray centers;
    Check(stage->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/tube0"))
                  .GetAttribute(TfToken("usdGen:pomade:centerPoints"))
                  .Get(&centers) &&
              centers.size() == 5,
          "tube0 carries 5 center points");
    VtVec3fArray guidePoints;
    Check(GetPoints(stage, SdfPath("/PomadeGroom/Guides"), &guidePoints) &&
              guidePoints.size() == 16 * 8,
          "Guides carries 16 guides x 8 CVs");
    UsdPrim const guides = stage->GetPrimAtPath(SdfPath("/PomadeGroom/Guides"));
    Check(guides.HasAPI(TfToken("UsdGenCurveAPI")),
          "Guides applies UsdGenCurveAPI");
    TfToken role;
    Check(guides.GetAttribute(TfToken("primvars:usdGen:role")).Get(&role) &&
              role == TfToken("guide"),
          "guide role primvar is \"guide\"");
    VtUInt64Array curveIds;
    VtIntArray tubeIds;
    Check(guides.GetAttribute(TfToken("primvars:usdGen:curveId"))
                  .Get(&curveIds) &&
              curveIds.size() == 16 && curveIds[0] == 1000 &&
              curveIds[15] == 1015,
          "curve ids are 1000 + guide");
    Check(guides.GetAttribute(TfToken("primvars:tubeId")).Get(&tubeIds) &&
              tubeIds.size() == 16 && tubeIds[0] == 0,
          "tubeId primvar names tube0");
    TfToken interp;
    Check(guides.GetAttribute(TfToken("primvars:tubeId"))
                  .GetMetadata(TfToken("interpolation"), &interp) &&
              interp == TfToken("uniform"),
          "tubeId interpolates uniform");
    std::string exprSource;
    Check(stage->GetPrimAtPath(SdfPath("/PomadeGroom/RegionExpr"))
                  .GetAttribute(TfToken("usdGen:expr:source"))
                  .Get(&exprSource) &&
              exprSource == "ptex(\"regionMap\")",
          "RegionExpr reads ptex(\"regionMap\")");
    TfToken mapFilter;
    Check(stage->GetPrimAtPath(SdfPath("/PomadeGroom/RegionMap"))
                  .GetAttribute(TfToken("usdGen:map:filter"))
                  .Get(&mapFilter) &&
              mapFilter == TfToken("nearest"),
          "RegionMap filters nearest");
    UsdPrim const pomadeOp =
        stage->GetPrimAtPath(SdfPath("/Groom/Hair/Ops/pomadeInterp"));
    Check(bool(pomadeOp) &&
              pomadeOp.GetTypeName() == TfToken("UsdGenGuideInterpolate"),
          "fill-in creates the GuideInterpolate op");
    Check(Targets(pomadeOp.GetRelationship(TfToken("usdGen:guides"))) ==
              SdfPathVector{SdfPath("/PomadeGroom/Guides")},
          "fill-in connects usdGen:guides to Guides");
    Check(Connections(pomadeOp.GetAttribute(TfToken("usdGen:region"))) ==
              SdfPathVector{SdfPath("/PomadeGroom/RegionExpr")},
          "fill-in connects usdGen:region to RegionExpr");
    std::printf("info: first swap took %.3f ms\n", committer.LastSwapMs());

    // -- coalescing: ten enqueues under pause produce one build ------------
    size_t const buildsBefore = committer.BuildCount();
    committer.PauseWorker(true);
    for (int i = 0; i < 10; ++i) {
        model.MoveCenterRing(i % 5, 0.1f, 0.0f);
        committer.Enqueue(stage);
    }
    uint64_t const vCoalesced = model.GetVersion();
    Check(committer.BuildCount() == buildsBefore,
          "paused worker builds nothing");
    committer.PauseWorker(false);
    Check(WaitCommitted(committer, live, vCoalesced),
          "coalesced commit swaps at idle");
    Check(committer.BuildCount() == buildsBefore + 1,
          "ten fast strokes produce one layer build");
    Check(committer.CommittedVersion() == vCoalesced,
          "the one build carries the latest version");

    // -- cancellation: a built layer superseded before the swap never lands
    committer.PauseWorker(true);
    model.MoveCenterRing(0, 1.0f, 0.0f);
    committer.Enqueue(stage);
    committer.PauseWorker(false);
    // Wait until vA is built but do NOT swap it yet.
    {
        auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(5000);
        while (committer.BuildCount() < buildsBefore + 2 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    Check(committer.BuildCount() == buildsBefore + 2, "vA layer built");
    size_t const droppedBefore = committer.DroppedCount();
    committer.PauseWorker(true);
    model.MoveCenterRing(1, 1.0f, 0.0f);
    committer.Enqueue(stage);
    model.MoveCenterRing(2, 1.0f, 0.0f);
    committer.Enqueue(stage);
    uint64_t const vCancel = model.GetVersion();
    size_t const buildsAtCancel = committer.BuildCount();
    committer.PauseWorker(false);
    // Let the worker finish the vC build with NO swap in between, so the
    // unswapped vA layer is necessarily replaced on the shelf (dropped).
    {
        auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(5000);
        while (committer.BuildCount() < buildsAtCancel + 1 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    Check(committer.BuildCount() == buildsAtCancel + 1,
          "superseding layer built with no swap between");
    Check(WaitCommitted(committer, live, vCancel),
          "superseding commit swaps at idle");
    Check(committer.DroppedCount() > droppedBefore,
          "the superseded vA layer was dropped, never swapped");
    Check(committer.CommittedVersion() == vCancel,
          "live carries the superseding version");
    {
        VtVec3fArray livePoints;
        usdGenPomade::PomadeGuideSet regen =
            usdGenPomade::PomadeGenerateGuides(model.Snapshot());
        Check(GetPoints(stage, SdfPath("/PomadeGroom/Guides"), &livePoints) &&
                  livePoints.size() == regen.points.size() / 3 &&
                  std::memcmp(livePoints.data(), regen.points.data(),
                              regen.points.size() * sizeof(float)) == 0,
              "landed guides equal the latest model state, not vA");
    }

    // -- no swap runs during a gesture -------------------------------------
    model.MoveCenterRing(3, 0.5f, 0.0f);
    committer.Enqueue(stage);
    uint64_t const vGesture = model.GetVersion();
    // The gesture gate is checked first, so this refuses whether or not the
    // worker has finished building.
    Check(committer.SwapIfIdle(live, /*gestureActive*/ true) ==
              PomadeCommitter::SkippedGesture,
          "swap during a gesture is refused");
    Check(committer.CommittedVersion() < vGesture,
          "the held swap leaves live on the old version");
    Check(WaitCommitted(committer, live, vGesture),
          "the held swap lands once the gesture ends");
    Check(committer.SwapIfIdle(live, /*gestureActive*/ false) ==
              PomadeCommitter::NothingPending,
          "a settled committer reports nothing pending");

    // -- partial-transfer fallback converges to identical content -----------
    committer.SetSwapBudgetMs(0.0);  // every full swap is now over budget
    model.MoveCenterRing(4, 0.25f, 0.0f);
    committer.Enqueue(stage);
    uint64_t const vFull = model.GetVersion();
    Check(WaitCommitted(committer, live, vFull),
          "over-budget commit still swaps (full, once)");
    Check(committer.PartialMode(),
          "an over-budget swap arms the partial fallback");
    // Twin committer, forced straight into partial mode by the same budget.
    SdfLayerRefPtr livePartial = SdfLayer::CreateAnonymous("pomade-live-p");
    UsdStageRefPtr stagePartial = MakeStage(livePartial);
    PomadeCommitter partial(&model, paths);
    partial.SetSwapBudgetMs(0.0);
    partial.Enqueue(stagePartial);
    uint64_t const vPartial = model.GetVersion();
    Check(WaitCommitted(partial, livePartial, vPartial),
          "partial-mode commit converges");
    Check(partial.PartialMode(), "twin committer ran the partial fallback");
    {
        VtVec3fArray a, b;
        VtVec3fArray ca, cb;
        bool sameGuides =
            GetPoints(stage, SdfPath("/PomadeGroom/Guides"), &a) &&
            GetPoints(stagePartial, SdfPath("/PomadeGroom/Guides"), &b) &&
            a.size() == b.size() &&
            std::memcmp(a.data(), b.data(), a.size() * sizeof(GfVec3f)) == 0;
        bool sameTube =
            stage->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/tube0"))
                .GetAttribute(TfToken("usdGen:pomade:centerPoints"))
                .Get(&ca) &&
            stagePartial->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/tube0"))
                .GetAttribute(TfToken("usdGen:pomade:centerPoints"))
                .Get(&cb) &&
            ca == cb;
        bool sameOp =
            Targets(stagePartial
                        ->GetPrimAtPath(SdfPath("/Groom/Hair/Ops/pomadeInterp"))
                        .GetRelationship(TfToken("usdGen:guides"))) ==
            SdfPathVector{SdfPath("/PomadeGroom/Guides")};
        Check(sameGuides, "partial swap lands identical guides");
        Check(sameTube, "partial swap lands an identical tube");
        Check(sameOp, "partial swap lands the op fill-in");
    }
    // Single-slot timing at this scale is far under budget; record it.
    std::printf("info: partial-mode slot took %.3f ms\n",
                partial.LastSwapMs());

    // -- hydrate round-trips the model bit-exactly -------------------------
    PomadeModel hydrated;
    usdGenPomade::PomadeHydrateResult hr =
        usdGenPomade::PomadeHydrateModel(stage, SdfPath("/PomadeGroom"), &hydrated);
    Check(hr.ok, "hydrate accepts the committed stage: " + hr.diagnostic);
    Check(hr.guidesBitEqual, "regenerated guides are bit-equal to stored");
    {
        TfToken rootSampler;
        bool const marked =
            stage->GetPrimAtPath(SdfPath("/PomadeGroom"))
                .GetAttribute(TfToken("usdGen:pomade:rootSampler"))
                .Get(&rootSampler);
        Check(marked && rootSampler == "region-v3",
              "new commits mark the exact region root sampler");
    }
    {
        PomadeModel::TubeSnapshot a = model.Snapshot();
        PomadeModel::TubeSnapshot b = hydrated.Snapshot();
        bool sameShape = a.shape.rings == b.shape.rings &&
            a.shape.ringVerts == b.shape.ringVerts &&
            DoubleEq(a.shape.radius, b.shape.radius, 1e-6) &&
            DoubleEq(a.shape.length, b.shape.length, 1e-6);
        bool sameCenters = a.centerX == b.centerX && a.centerY == b.centerY &&
            a.centerZ == b.centerZ;
        bool sameFill = a.fill.density == b.fill.density &&
            a.fill.cvCount == b.fill.cvCount && a.fill.seed == b.fill.seed &&
            a.fill.edgeBias == b.fill.edgeBias;
        bool sameSections = a.sections.size() == b.sections.size();
        for (size_t i = 0; sameSections && i < a.sections.size(); ++i) {
            sameSections = a.sections[i].t == b.sections[i].t &&
                           a.sections[i].u == b.sections[i].u &&
                           a.sections[i].v == b.sections[i].v;
        }
        Check(sameShape, "hydrated shape matches the model");
        Check(sameCenters, "hydrated centers match bit-exactly");
        Check(sameFill, "hydrated fill params match");
        Check(sameSections, "hydrated sections match bit-exactly");
    }
    // Re-commit the hydrated model: identical guide bytes.
    {
        SdfLayerRefPtr live2 = SdfLayer::CreateAnonymous("pomade-live-2");
        UsdStageRefPtr stage2 = MakeStage(live2);
        PomadeCommitter c2(&hydrated, paths);
        c2.Enqueue(stage2);
        Check(WaitCommitted(c2, live2, hydrated.GetVersion()),
              "hydrated model re-commits");
        VtVec3fArray a, b;
        Check(GetPoints(stage, SdfPath("/PomadeGroom/Guides"), &a) &&
                  GetPoints(stage2, SdfPath("/PomadeGroom/Guides"), &b) &&
                  a.size() == b.size() &&
                  std::memcmp(a.data(), b.data(),
                              a.size() * sizeof(GfVec3f)) == 0,
                  "re-commit emits bit-identical guides");
    }
    // A region-v2 layer authenticated with its historical mean-radius K9
    // bytes must be upgraded to the material sampler before it is saved
    // again. The old bytes stay strict: changing one guide still fails.
    {
        PomadeModel legacyAuthor;
        Check(legacyAuthor.BuildTestTube(),
              "v2 migration fixture builds a tube");
        PomadeModel::FillParams legacyFill = legacyAuthor.GetFillParams();
        legacyFill.density = 16.0f;
        legacyFill.cvCount = 8;
        legacyFill.seed = 7;
        legacyFill.edgeBias = 0.25f;
        legacyFill.sampler = usdGenPomade::PomadeGuideSampler::Legacy;
        Check(legacyAuthor.SetFillParams(legacyFill),
              "v2 migration fixture selects legacy fill");
        SdfLayerRefPtr v2Live = SdfLayer::CreateAnonymous("pomade-v2-live");
        UsdStageRefPtr v2Stage = MakeStage(v2Live);
        PomadeCommitter v2Committer(&legacyAuthor, paths);
        v2Committer.Enqueue(v2Stage);
        bool const v2Committed =
            WaitCommitted(v2Committer, v2Live, legacyAuthor.GetVersion());
        SdfLayerRefPtr v2Over = SdfLayer::CreateAnonymous("pomade-v2-over");
        v2Stage->GetSessionLayer()->InsertSubLayerPath(v2Over->GetIdentifier(),
                                                        0);
        v2Stage->SetEditTarget(v2Over);
        UsdPrim v2Groom = v2Stage->GetPrimAtPath(SdfPath("/PomadeGroom"));
        bool const markedV2 = bool(v2Groom) &&
            v2Groom.GetAttribute(TfToken("usdGen:pomade:rootSampler"))
                .Set(TfToken("region-v2"));
        PomadeModel migrated;
        usdGenPomade::PomadeHydrateResult const v2Hydrate =
            usdGenPomade::PomadeHydrateModel(v2Stage, SdfPath("/PomadeGroom"),
                                            &migrated);
        Check(v2Committed && markedV2 && v2Hydrate.ok &&
                  v2Hydrate.guidesBitEqual,
              "region-v2 guides hydrate through the exact legacy sampler");
        Check(migrated.GetFillParams().sampler ==
                  usdGenPomade::PomadeGuideSampler::RegionV3,
              "authenticated region-v2 hydrate upgrades live fill sampling");

        SdfLayerRefPtr migratedLive =
            SdfLayer::CreateAnonymous("pomade-v3-migrated-live");
        UsdStageRefPtr migratedStage = MakeStage(migratedLive);
        PomadeCommitter migratedCommitter(&migrated, paths);
        migratedCommitter.Enqueue(migratedStage);
        bool const migratedCommitted = WaitCommitted(
            migratedCommitter, migratedLive, migrated.GetVersion());
        TfToken migratedSampler;
        bool const markedV3 = migratedCommitted &&
            migratedStage->GetPrimAtPath(SdfPath("/PomadeGroom"))
                .GetAttribute(TfToken("usdGen:pomade:rootSampler"))
                .Get(&migratedSampler) && migratedSampler == "region-v3";
        PomadeModel reopened;
        usdGenPomade::PomadeHydrateResult const reopenedResult =
            usdGenPomade::PomadeHydrateModel(migratedStage,
                                            SdfPath("/PomadeGroom"),
                                            &reopened);
        Check(markedV3 && reopenedResult.ok && reopenedResult.guidesBitEqual,
              "migrated region-v3 commit reopens with material guides");

        UsdPrim v2Guides =
            v2Stage->GetPrimAtPath(SdfPath("/PomadeGroom/Guides"));
        VtVec3fArray v2Points;
        bool const gotV2Points = bool(v2Guides) &&
            v2Guides.GetAttribute(TfToken("points")).Get(&v2Points) &&
            !v2Points.empty();
        if (gotV2Points) {
            v2Points[0] += GfVec3f(0.01f, 0.0f, 0.0f);
            v2Guides.GetAttribute(TfToken("points")).Set(v2Points);
        }
        PomadeModel corruptV2;
        usdGenPomade::PomadeHydrateResult const corruptV2Result =
            usdGenPomade::PomadeHydrateModel(v2Stage, SdfPath("/PomadeGroom"),
                                            &corruptV2);
        Check(gotV2Points && !corruptV2Result.ok &&
                  !corruptV2Result.guidesBitEqual,
              "region-v2 migration still rejects corrupted legacy guides");
    }
    // The custom frame reference is optional for existing grooms. A missing
    // payload means identity, preserving their pre-Q K4 geometry exactly.
    {
        SdfLayerRefPtr legacyQ = SdfLayer::CreateAnonymous("pomade-legacy-q");
        legacyQ->TransferContent(live);
        SdfPrimSpecHandle legacyTube =
            legacyQ->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/tube0"));
        SdfPropertySpecHandle const qSpec = legacyQ->GetPropertyAtPath(
            SdfPath("/PomadeGroom/Tubes/tube0").AppendProperty(
                TfToken("usdGen:pomade:frameReference")));
        bool const removedQ = bool(legacyTube) && bool(qSpec);
        if (removedQ) {
            legacyTube->RemoveProperty(qSpec);
        }
        UsdStageRefPtr legacyQStage = MakeStage(legacyQ);
        PomadeModel legacyQModel;
        usdGenPomade::PomadeHydrateResult const legacyQResult =
            usdGenPomade::PomadeHydrateModel(legacyQStage,
                                           SdfPath("/PomadeGroom"),
                                           &legacyQModel);
        PomadeModel::TubeSnapshot legacyQSnapshot = legacyQModel.Snapshot();
        Check(removedQ && legacyQResult.ok && legacyQResult.guidesBitEqual &&
                  legacyQSnapshot.frameReference ==
                      std::array<float, 9>{{1.0f, 0.0f, 0.0f,
                                            0.0f, 1.0f, 0.0f,
                                            0.0f, 0.0f, 1.0f}},
              "missing frameReference hydrates legacy geometry as identity");
    }
    // Reject a malformed authored matrix before it can re-derive children or
    // apply deltas in a reflected/non-orthonormal frame.
    {
        SdfLayerRefPtr invalidQLayer =
            SdfLayer::CreateAnonymous("pomade-invalid-q");
        invalidQLayer->TransferContent(live);
        UsdStageRefPtr invalidQStage = MakeStage(invalidQLayer);
        SdfLayerRefPtr invalidQOver =
            SdfLayer::CreateAnonymous("pomade-invalid-q-over");
        invalidQStage->GetSessionLayer()->InsertSubLayerPath(
            invalidQOver->GetIdentifier(), 0);
        invalidQStage->SetEditTarget(invalidQOver);
        VtFloatArray badQ(9);
        std::fill(badQ.begin(), badQ.end(), 0.0f);
        UsdPrim badTube =
            invalidQStage->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/tube0"));
        bool const wroteBadQ = bool(badTube) &&
            badTube.GetAttribute(TfToken("usdGen:pomade:frameReference"))
                .Set(badQ);
        VtFloatArray composedBad;
        bool const hasComposedBadQ = wroteBadQ &&
            badTube.GetAttribute(TfToken("usdGen:pomade:frameReference"))
                .Get(&composedBad) && composedBad == badQ;
        PomadeModel invalidQModel;
        usdGenPomade::PomadeHydrateResult const invalidQResult =
            usdGenPomade::PomadeHydrateModel(invalidQStage,
                                           SdfPath("/PomadeGroom"),
                                           &invalidQModel);
        Check(hasComposedBadQ && !invalidQResult.ok &&
                  invalidQResult.diagnostic.find("proper rotation") !=
                      std::string::npos,
              "hydrate rejects a non-orthonormal frameReference");
    }
    // Foreign guides fail closed, never silently.
    {
        UsdStageRefPtr fs = UsdStage::CreateInMemory("pomadeForeign");
        SdfLayerRefPtr over =
            SdfLayer::CreateAnonymous("pomade-foreign-over");
        fs->GetSessionLayer()->InsertSubLayerPath(live->GetIdentifier(), 0);
        fs->GetSessionLayer()->InsertSubLayerPath(over->GetIdentifier(), 0);
        fs->SetEditTarget(over);
        UsdPrim fg = fs->GetPrimAtPath(SdfPath("/PomadeGroom/Guides"));
        VtVec3fArray pts;
        fg.GetAttribute(TfToken("points")).Get(&pts);
        pts[0] = GfVec3f(123.0f, 456.0f, 789.0f);
        fg.GetAttribute(TfToken("points")).Set(pts);
        PomadeModel fm;
        usdGenPomade::PomadeHydrateResult fr =
            usdGenPomade::PomadeHydrateModel(fs, SdfPath("/PomadeGroom"), &fm);
        Check(!fr.ok && !fr.guidesBitEqual,
              "a marked current sampler rejects hand-edited guides");
    }
    // Old layers do not carry rootSampler.  They may use the legacy fill
    // only when every guide byte validates; removing the marker must not let
    // an edited legacy guide set through the compatibility branch.
    {
        SdfLayerRefPtr legacyLayer = SdfLayer::CreateAnonymous("pomade-legacy");
        legacyLayer->TransferContent(live);
        SdfPrimSpecHandle legacyGroom =
            legacyLayer->GetPrimAtPath(SdfPath("/PomadeGroom"));
        SdfPropertySpecHandle const samplerSpec =
            legacyLayer->GetPropertyAtPath(
                SdfPath("/PomadeGroom").AppendProperty(
                    TfToken("usdGen:pomade:rootSampler")));
        bool const removedSampler = bool(legacyGroom) && bool(samplerSpec);
        if (removedSampler) {
            legacyGroom->RemoveProperty(samplerSpec);
        }
        UsdStageRefPtr legacyStage = MakeStage(legacyLayer);
        SdfLayerRefPtr legacyOver =
            SdfLayer::CreateAnonymous("pomade-legacy-edited-over");
        legacyStage->GetSessionLayer()->InsertSubLayerPath(
            legacyOver->GetIdentifier(), 0);
        legacyStage->SetEditTarget(legacyOver);
        UsdPrim legacyGuides =
            legacyStage->GetPrimAtPath(SdfPath("/PomadeGroom/Guides"));
        VtVec3fArray points;
        bool const gotPoints =
            bool(legacyGuides) &&
            legacyGuides.GetAttribute(TfToken("points")).Get(&points) &&
            !points.empty();
        bool pointsEdited = false;
        if (gotPoints) {
            points[0] = GfVec3f(123.0f, 456.0f, 789.0f);
            legacyGuides.GetAttribute(TfToken("points")).Set(points);
            VtVec3fArray composedPoints;
            pointsEdited =
                legacyGuides.GetAttribute(TfToken("points"))
                    .Get(&composedPoints) &&
                !composedPoints.empty() && composedPoints[0] == points[0];
        }
        PomadeModel legacyModel;
        usdGenPomade::PomadeHydrateResult const legacyResult =
            usdGenPomade::PomadeHydrateModel(legacyStage,
                                           SdfPath("/PomadeGroom"),
                                           &legacyModel);
        Check(removedSampler && gotPoints && pointsEdited && !legacyResult.ok &&
                  !legacyResult.guidesBitEqual,
              "a markerless legacy groom still rejects modified guides");
    }

    // -- GuideInterpolate fill-in only touches what is empty ---------------
    {
        // (b) authored guides survive; the empty region is filled.
        SdfLayerRefPtr liveB = SdfLayer::CreateAnonymous("pomade-live-b");
        UsdStageRefPtr sb = MakeStage(liveB);
        UsdPrim mine =
            sb->DefinePrim(SdfPath("/Groom/Hair/Ops/myInterp"),
                           TfToken("UsdGenGuideInterpolate"));
        mine.CreateRelationship(TfToken("usdGen:guides"))
            .SetTargets({SdfPath("/Other")});
        usdGenPomade::PomadeFillPlan plan =
            usdGenPomade::PomadePlanGuideInterpolateFill(sb, paths);
        Check(!plan.createInterpOp && !plan.setInterpGuides &&
                  plan.setInterpRegion &&
                  plan.opPath == SdfPath("/Groom/Hair/Ops/myInterp"),
              "existing op plans a region-only fill on its own path");
        PomadeCommitter cb(&model, paths);
        cb.Enqueue(sb);
        Check(WaitCommitted(cb, liveB, model.GetVersion()),
              "region-only fill swaps");
        Check(Targets(sb->GetPrimAtPath(SdfPath("/Groom/Hair/Ops/myInterp"))
                           .GetRelationship(TfToken("usdGen:guides"))) ==
                  SdfPathVector{SdfPath("/Other")},
              "authored guides are never rewritten");
        Check(Connections(sb->GetPrimAtPath(SdfPath("/Groom/Hair/Ops/myInterp"))
                              .GetAttribute(TfToken("usdGen:region"))) ==
                  SdfPathVector{SdfPath("/PomadeGroom/RegionExpr")},
              "the empty region is connected");
        Check(!sb->GetPrimAtPath(SdfPath("/Groom/Hair/Ops/pomadeInterp")),
              "no second op is authored beside the artist's");
    }
    {
        // (c) a fully authored op is left alone.
        SdfLayerRefPtr liveC = SdfLayer::CreateAnonymous("pomade-live-c");
        UsdStageRefPtr sc = MakeStage(liveC);
        UsdPrim full =
            sc->DefinePrim(SdfPath("/Groom/Hair/Ops/full"),
                           TfToken("UsdGenGuideInterpolate"));
        full.CreateRelationship(TfToken("usdGen:guides"))
            .SetTargets({SdfPath("/Other")});
        full.CreateAttribute(TfToken("usdGen:region"),
                             SdfValueTypeNames->Float)
            .Set(2.0f);
        usdGenPomade::PomadeFillPlan plan =
            usdGenPomade::PomadePlanGuideInterpolateFill(sc, paths);
        Check(!plan.createInterpOp && !plan.setInterpGuides &&
                  !plan.setInterpRegion,
              "a fully authored op plans no fill-in");
        // (d) a missing description plans nothing.
        SdfLayerRefPtr liveD = SdfLayer::CreateAnonymous("pomade-live-d");
        UsdStageRefPtr sd = MakeStage(liveD, /*withDescription*/ false);
        plan = usdGenPomade::PomadePlanGuideInterpolateFill(sd, paths);
        Check(!plan.createInterpOp && !plan.setInterpGuides &&
                  !plan.setInterpRegion,
              "a missing description plans no fill-in");
    }

    // -- Save groom ---------------------------------------------------------
    {
        std::filesystem::path savePath =
            std::filesystem::temp_directory_path() / "pomade-groom-p1.usdc";
        std::error_code ec;
        std::filesystem::remove(savePath, ec);
        std::string err;
        Check(usdGenPomade::PomadeSaveGroom(stage, live, savePath.string(), &err),
              "Save groom writes .usdc: " + err);
        Check(std::filesystem::exists(savePath, ec),
              "the saved .usdc exists on disk");
        std::vector<std::string> sublayers =
            stage->GetSessionLayer()->GetSubLayerPaths();
        Check(sublayers.size() >= 2 && sublayers[0] == live->GetIdentifier() &&
                  sublayers[1] ==
                      SdfLayer::FindOrOpen(savePath.string())->GetIdentifier(),
              "live stays strongest with the file beneath it");
        UsdStageRefPtr saved = UsdStage::Open(savePath.string());
        Check(bool(saved) &&
                  saved->GetPrimAtPath(SdfPath("/PomadeGroom/Guides")),
              "the saved file carries the groom");
        Check(!usdGenPomade::PomadeSaveGroom(stage, live, "groom.usda", &err),
              ".usda save is refused (S42)");
        std::filesystem::remove(savePath, ec);
    }

    // -- TN-4: reference-scale swap in <= 5 ms -------------------------------
    {
        // Plan/17 section 7 reference: 2400 tubes, 12000 guides x 16 CVs.
        usdGenPomade::PomadeSnapshot big;
        big.version = 4242;
        big.createInterpOp = true;
        big.setInterpGuides = true;
        big.setInterpRegion = true;
        big.interpOpPath = paths.InterpOpPath();
        for (int t = 0; t < 2400; ++t) {
            usdGenPomade::PomadeSnapshotTube entry;
            entry.tubeId = t;
            entry.regionId = t % 80;
            entry.level = 3;
            entry.tube.hasTube = true;
            entry.tube.shape.rings = 20;
            entry.tube.shape.ringVerts = 16;
            entry.tube.shape.radius = 0.5f;
            entry.tube.shape.length = 4.0f;
            entry.tube.centerX.assign(20, float(t) * 0.01f);
            entry.tube.centerY.resize(20);
            entry.tube.centerZ.assign(20, 0.0f);
            for (int r = 0; r < 20; ++r) {
                entry.tube.centerY[size_t(r)] = 4.0f * float(r) / 19.0f;
            }
            entry.tube.fill.density = 5.0f;
            entry.tube.fill.cvCount = 16;
            entry.tube.fill.seed = t;
            big.tubes.push_back(std::move(entry));
        }
        SdfLayerRefPtr built;
        std::string err;
        auto buildT0 = std::chrono::steady_clock::now();
        Check(usdGenPomade::PomadeBuildCommitLayer(big, paths, &built, &err),
              "reference-scale layer builds: " + err);
        auto buildT1 = std::chrono::steady_clock::now();
        std::printf(
            "info: reference build took %.1f ms (worker thread, off the UI)\n",
            std::chrono::duration<double, std::milli>(buildT1 - buildT0)
                .count());
        // A full TransferContent at this scale is over budget by design (a
        // tens-of-thousands-spec copy); that is what the partial fallback
        // exists for. Record it, then prove every partial slot is inside it.
        {
            SdfLayerRefPtr slot = SdfLayer::CreateAnonymous("tn4-full");
            auto t0 = std::chrono::steady_clock::now();
            {
                SdfChangeBlock block;
                slot->TransferContent(built);
            }
            auto t1 = std::chrono::steady_clock::now();
            std::printf("info: TN-4 full TransferContent took %.1f ms "
                        "(over budget; partial fallback engaged)\n",
                        std::chrono::duration<double, std::milli>(t1 - t0)
                            .count());
        }
        SdfLayerRefPtr liveBig = SdfLayer::CreateAnonymous("tn4-live");
        UsdStageRefPtr stageBig = MakeStage(liveBig);
        PomadeModel bigModel;  // unused: the ready layer is injected
        PomadeCommitter bigCommitter(&bigModel, paths);
        bigCommitter.SetReadyForTest(built, big.version);
        bigCommitter.ForcePartialModeForTest(true);
        double worstSlot = 0.0;
        size_t slots = 0;
        PomadeCommitter::SwapResult last = PomadeCommitter::PartialProgress;
        for (;;) {
            last = bigCommitter.SwapIfIdle(liveBig,
                                           /*gestureActive*/ false);
            worstSlot = std::max(worstSlot, bigCommitter.LastSwapMs());
            ++slots;
            if (last == PomadeCommitter::Swapped) {
                break;
            }
            if (last != PomadeCommitter::PartialProgress || slots > 100000) {
                break;
            }
        }
        std::printf("info: TN-4 partial swap took %zu slots, worst %.3f ms\n",
                    slots, worstSlot);
        Check(last == PomadeCommitter::Swapped,
              "TN-4: reference partial swap converges");
        Check(bigCommitter.CommittedVersion() == big.version,
              "TN-4: reference version commits");
        // The product slot budget stays 5 ms (_swapBudgetMs). This sample is
        // steady_clock around the copy, so one preemption on a loaded
        // `ctest -j` runner counts. The push gate on 6b392ca measured 5.098 ms
        // with every other slot inside the budget; allow one extra slice and
        // still fail a real stall.
        Check(worstSlot <= 8.0,
              "TN-4: every idle slot stays within 8 ms (got " +
                  std::to_string(worstSlot) + ")");
        {
            UsdPrim tubesPrim =
                stageBig->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes"));
            size_t tubeCount = 0;
            for (UsdPrim const &child : tubesPrim.GetChildren()) {
                (void)child;
                ++tubeCount;
            }
            VtVec3fArray bigPoints;
            Check(tubeCount == 2400, "TN-4: all 2400 tubes land");
            Check(GetPoints(stageBig, SdfPath("/PomadeGroom/Guides"),
                            &bigPoints) &&
                      bigPoints.size() == 12000 * 16,
                  "TN-4: all 12000 guides land");
        }
    }

    // -- partial swap carries the scalp over (UsdGenRestAPI for Output) -----
    //
    // Regression: once a full swap blew the budget, partial mode latched and
    // its slot list never included the scalp over, so UsdGenRestAPI never
    // reached the live layer and Output's surface-cage CurveSource capture
    // failed ("surface-cage requires authored/default-time surface rest").
    {
        int const n = 4;
        Grid const grid = MakeGrid(n);
        PomadeModel om;
        Check(om.BindScalp(grid.points, grid.counts, grid.indices),
              "partial scalp: the Output model binds its scalp");
        std::shared_ptr<usdGenPomade::PomadeScalpMesh const> const mesh =
            om.GetScalp();
        if (mesh) {
            BuildTwoRegions(&om, *mesh, n);
        }
        Check(om.Rasterise() && om.BuildTubeFromRegion(0, 5, 0, 2.0f),
              "partial scalp: a region tube builds: " +
                  std::string(om.GetDiagnostic()));
        PomadeModel::OutputSettings output = om.GetOutputSettings();
        output.enabled = true;
        Check(om.SetOutputSettings(output), "partial scalp: Output enables");

        SdfLayerRefPtr liveO = SdfLayer::CreateAnonymous("pomade-live-partial-scalp");
        UsdStageRefPtr so = MakeStage(liveO);
        SdfPath const scalpPath("/Scalp");
        {
            UsdGeomMesh scalp = UsdGeomMesh::Define(so, scalpPath);
            VtVec3fArray pts(grid.points.size() / 3);
            for (size_t i = 0; i < pts.size(); ++i) {
                pts[i] = GfVec3f(grid.points[i * 3 + 0],
                                 grid.points[i * 3 + 1],
                                 grid.points[i * 3 + 2]);
            }
            scalp.GetPointsAttr().Set(pts);
            scalp.GetFaceVertexCountsAttr().Set(
                VtIntArray(grid.counts.begin(), grid.counts.end()));
            scalp.GetFaceVertexIndicesAttr().Set(
                VtIntArray(grid.indices.begin(), grid.indices.end()));
        }
        PomadeCommitPaths outputPaths = paths;
        outputPaths.scalpPath = scalpPath;
        PomadeCommitter co(&om, outputPaths);
        co.ForcePartialModeForTest(true);

        // Drain every idle slot of one version; counts the partial slots so
        // the case proves it really went through _SwapPartialSlot.
        auto drainPartial = [&](uint64_t want, size_t *partialSlots) {
            *partialSlots = 0;
            auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(10000);
            while (std::chrono::steady_clock::now() < deadline) {
                if (co.SwapIfIdle(liveO, /*gestureActive*/ false) ==
                    PomadeCommitter::PartialProgress) {
                    ++*partialSlots;
                }
                if (co.CommittedVersion() >= want) {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return false;
        };
        auto liveScalpHasRest = [&]() {
            SdfPrimSpecHandle const over = liveO->GetPrimAtPath(scalpPath);
            if (!over) {
                return false;
            }
            VtValue const value = over->GetInfo(TfToken("apiSchemas"));
            if (!value.IsHolding<SdfTokenListOp>()) {
                return false;
            }
            TfTokenVector applied;
            value.UncheckedGet<SdfTokenListOp>().ApplyOperations(&applied);
            return std::find(applied.begin(), applied.end(),
                             TfToken("UsdGenRestAPI")) != applied.end();
        };

        co.Enqueue(so);
        size_t partialSlots = 0;
        Check(drainPartial(om.GetVersion(), &partialSlots),
              "partial scalp: the Output groom commits in partial mode: " +
                  co.TakeDiagnostic());
        Check(partialSlots > 0,
              "partial scalp: the commit went through partial slots (" +
                  std::to_string(partialSlots) + ")");
        Check(bool(liveO->GetPrimAtPath(outputPaths.OutputPath())),
              "partial scalp: the Output description lands");
        Check(liveScalpHasRest(),
              "partial scalp: the live scalp over carries UsdGenRestAPI");

        // Turning Output off drops RestAPI from the built over; the partial
        // scalp slot must replace the live over, not leave it stale.
        output.enabled = false;
        Check(om.SetOutputSettings(output), "partial scalp: Output disables");
        co.Enqueue(so);
        Check(drainPartial(om.GetVersion(), &partialSlots),
              "partial scalp: the Output-off groom commits in partial mode");
        Check(!liveScalpHasRest(),
              "partial scalp: a stale UsdGenRestAPI does not survive on live");

        // A groom authored UNDER the scalp prim (/Scalp/PomadeGroom, e.g. a
        // resumed hand-placed groom): the scalp slot is the last one, and a
        // whole-spec copy of the scalp re-copied the entire groom subtree
        // in that single slot. A sentinel child planted on live before every
        // slot must survive the last one (a whole-spec copy replaces the
        // scalp's children with the built layer's), and the scalp over must
        // still carry UsdGenRestAPI.
        output.enabled = true;
        Check(om.SetOutputSettings(output),
              "partial scalp: Output re-enables for the nested groom");
        SdfLayerRefPtr liveN =
            SdfLayer::CreateAnonymous("pomade-live-partial-nested");
        UsdStageRefPtr sn = MakeStage(liveN);
        {
            UsdGeomMesh scalp = UsdGeomMesh::Define(sn, scalpPath);
            VtVec3fArray pts(grid.points.size() / 3);
            for (size_t i = 0; i < pts.size(); ++i) {
                pts[i] = GfVec3f(grid.points[i * 3 + 0],
                                 grid.points[i * 3 + 1],
                                 grid.points[i * 3 + 2]);
            }
            scalp.GetPointsAttr().Set(pts);
            scalp.GetFaceVertexCountsAttr().Set(
                VtIntArray(grid.counts.begin(), grid.counts.end()));
            scalp.GetFaceVertexIndicesAttr().Set(
                VtIntArray(grid.indices.begin(), grid.indices.end()));
        }
        PomadeCommitPaths nestedPaths = outputPaths;
        nestedPaths.groomPath = scalpPath.AppendChild(TfToken("PomadeGroom"));
        PomadeCommitter cn(&om, nestedPaths);
        cn.ForcePartialModeForTest(true);
        SdfPath const sentinel =
            nestedPaths.groomPath.AppendChild(TfToken("ZzSlotSentinel"));
        cn.Enqueue(sn);
        bool nestedSwapped = false;
        bool sentinelSurvivedLast = false;
        size_t nestedSlots = 0;
        {
            auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(10000);
            while (std::chrono::steady_clock::now() < deadline) {
                SdfCreatePrimInLayer(liveN, sentinel);
                PomadeCommitter::SwapResult const r =
                    cn.SwapIfIdle(liveN, /*gestureActive*/ false);
                if (r == PomadeCommitter::PartialProgress) {
                    ++nestedSlots;
                } else if (r == PomadeCommitter::Swapped) {
                    nestedSwapped = true;
                    sentinelSurvivedLast = bool(liveN->GetPrimAtPath(sentinel));
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        Check(nestedSwapped && nestedSlots > 0,
              "partial scalp: a groom under the scalp commits in partial "
              "slots (" + std::to_string(nestedSlots) + "): " +
                  cn.TakeDiagnostic());
        Check(sentinelSurvivedLast,
              "partial scalp: the last (scalp) slot does not re-copy the "
              "groom subtree beneath the scalp");
        if (SdfPrimSpecHandle const sentinelSpec =
                liveN->GetPrimAtPath(sentinel)) {
            liveN->GetPrimAtPath(nestedPaths.groomPath)
                ->RemoveNameChild(sentinelSpec);
        }
        {
            SdfPrimSpecHandle const over = liveN->GetPrimAtPath(scalpPath);
            TfTokenVector applied;
            if (over) {
                VtValue const value = over->GetInfo(TfToken("apiSchemas"));
                if (value.IsHolding<SdfTokenListOp>()) {
                    value.UncheckedGet<SdfTokenListOp>().ApplyOperations(
                        &applied);
                }
            }
            Check(std::find(applied.begin(), applied.end(),
                            TfToken("UsdGenRestAPI")) != applied.end(),
                  "partial scalp: the nested groom's scalp over still "
                  "carries UsdGenRestAPI");
        }
        Check(bool(liveN->GetPrimAtPath(nestedPaths.GuidesPath())) &&
                  bool(liveN->GetPrimAtPath(nestedPaths.OutputPath())),
              "partial scalp: the nested groom's Guides and Output land");
    }

    // -- plan/02 §2.20: a face GeomSubset scalp commits and hydrates -------
    //
    // The scalp link and Output's usdGen:surface name the subset; the rest
    // binding and the live pomadeRegion primvar land on the parent Mesh,
    // parent-sized (a subset never renumbers faces); hydrate re-binds
    // through the subset and round-trips bit-exactly, on the full and the
    // partial swap alike. Every invalid subset is an error naming the prim.
    {
        int const n = 4;
        Grid const grid = MakeGrid(n);
        SdfPath const meshPath("/Scalp");
        SdfPath const subsetPath("/Scalp/patch");
        // Every face but 5 (x, z in [1, 2]), unsorted with a repeat as a
        // union of subsets produces; the ring region below surrounds 5.
        VtIntArray const authored = {15, 0, 1, 2, 3, 4, 6, 7,
                                     8, 9, 10, 11, 12, 13, 14, 0};
        int const hole = 5;
        std::vector<int> wantActive;
        for (int f = 0; f < n * n; ++f) {
            if (f != hole) {
                wantActive.push_back(f);
            }
        }
        auto defineScene = [&](UsdStagePtr const &stage) {
            UsdGeomMesh scalp = UsdGeomMesh::Define(stage, meshPath);
            VtVec3fArray pts(grid.points.size() / 3);
            for (size_t i = 0; i < pts.size(); ++i) {
                pts[i] = GfVec3f(grid.points[i * 3 + 0],
                                 grid.points[i * 3 + 1],
                                 grid.points[i * 3 + 2]);
            }
            scalp.GetPointsAttr().Set(pts);
            scalp.GetFaceVertexCountsAttr().Set(
                VtIntArray(grid.counts.begin(), grid.counts.end()));
            scalp.GetFaceVertexIndicesAttr().Set(
                VtIntArray(grid.indices.begin(), grid.indices.end()));
            UsdGeomSubset patch = UsdGeomSubset::Define(stage, subsetPath);
            patch.GetElementTypeAttr().Set(UsdGeomTokens->face);
            patch.GetIndicesAttr().Set(authored);
            // What the resolver must refuse, each by name.
            UsdGeomSubset pointSet =
                UsdGeomSubset::Define(stage, SdfPath("/Scalp/pointSet"));
            pointSet.GetElementTypeAttr().Set(TfToken("point"));
            pointSet.GetIndicesAttr().Set(VtIntArray{0, 1});
            UsdGeomSubset wide =
                UsdGeomSubset::Define(stage, SdfPath("/Scalp/wide"));
            wide.GetElementTypeAttr().Set(UsdGeomTokens->face);
            wide.GetIndicesAttr().Set(VtIntArray{0, 16});
            UsdGeomSubset none =
                UsdGeomSubset::Define(stage, SdfPath("/Scalp/none"));
            none.GetElementTypeAttr().Set(UsdGeomTokens->face);
            none.GetIndicesAttr().Set(VtIntArray());
            UsdGeomXform::Define(stage, SdfPath("/Holder"));
            UsdGeomSubset orphan =
                UsdGeomSubset::Define(stage, SdfPath("/Holder/orphan"));
            orphan.GetElementTypeAttr().Set(UsdGeomTokens->face);
            orphan.GetIndicesAttr().Set(VtIntArray{0});
        };
        auto overHasRest = [](SdfLayerHandle const &layer, SdfPath const &at) {
            SdfPrimSpecHandle const over = layer->GetPrimAtPath(at);
            TfTokenVector applied;
            if (over) {
                VtValue const value = over->GetInfo(TfToken("apiSchemas"));
                if (value.IsHolding<SdfTokenListOp>()) {
                    value.UncheckedGet<SdfTokenListOp>().ApplyOperations(
                        &applied);
                }
            }
            return std::find(applied.begin(), applied.end(),
                             TfToken("UsdGenRestAPI")) != applied.end();
        };

        SdfLayerRefPtr liveS = SdfLayer::CreateAnonymous("pomade-live-subset");
        UsdStageRefPtr ss = MakeStage(liveS);
        defineScene(ss);
        usdGenPomade::PomadeScalpTarget target;
        std::string err;
        bool const wholeOk =
            usdGenPomade::PomadeResolveScalpTarget(ss, meshPath, &target,
                                                 &err) &&
            !target.isSubset && target.activeFaces.empty() &&
            target.meshPath == meshPath;
        Check(wholeOk, "subset: a Mesh link resolves to the whole mesh: " + err);
        bool const subsetOk =
            usdGenPomade::PomadeResolveScalpTarget(ss, subsetPath, &target,
                                                 &err) &&
            target.isSubset && target.targetPath == subsetPath &&
            target.meshPath == meshPath && target.activeFaces == wantActive &&
            target.faceVertexCounts.size() == size_t(n * n) &&
            target.points == grid.points;
        Check(subsetOk,
              "subset: a face GeomSubset resolves to its parent's geometry "
              "plus sorted, unique parent face ids: " + err);
        struct BadTarget {
            char const *path;
            char const *says;
        };
        for (BadTarget const &bad :
             {BadTarget{"/Scalp/pointSet", "elementType \"point\""},
              BadTarget{"/Scalp/wide", "names face 16"},
              BadTarget{"/Scalp/none", "names no faces"},
              BadTarget{"/Holder/orphan", "is not a child of a Mesh"},
              BadTarget{"/Holder", "not a Mesh or a face GeomSubset"},
              BadTarget{"/Missing", "no prim"}}) {
            std::string why;
            bool const refused = !usdGenPomade::PomadeResolveScalpTarget(
                                     ss, SdfPath(bad.path), &target, &why) &&
                                 why.find(bad.says) != std::string::npos &&
                                 why.find(bad.path) != std::string::npos;
            Check(refused, "subset: " + std::string(bad.path) +
                               " is refused by name (" + why + ")");
        }

        bool const resolved =
            usdGenPomade::PomadeResolveScalpTarget(ss, subsetPath, &target, &err);
        Check(resolved, "subset: the patch resolves again: " + err);
        PomadeModel sm;
        Check(sm.BindScalp(target.points, target.faceVertexCounts,
                           target.faceVertexIndices, target.activeFaces),
              "subset: the model binds the parent mesh through the subset");
        std::shared_ptr<usdGenPomade::PomadeScalpMesh const> const mesh =
            sm.GetScalp();
        if (mesh) {
            // A ring around the hole: its corners and every traced edge sit
            // on subset faces, and face 5 lies strictly inside it.
            int const r0 = sm.GraphAddNode(Locate(*mesh, n, 0.5f, 0.5f));
            int const r1 = sm.GraphAddNode(Locate(*mesh, n, 3.5f, 0.5f));
            int const r2 = sm.GraphAddNode(Locate(*mesh, n, 3.5f, 3.5f));
            int const r3 = sm.GraphAddNode(Locate(*mesh, n, 0.5f, 3.5f));
            sm.GraphConnect(r0, r1);
            sm.GraphConnect(r1, r2);
            sm.GraphConnect(r2, r3);
            sm.GraphConnect(r3, r0);
        }
        Check(sm.Rasterise() && sm.BuildTubeFromRegion(0, 5, 0, 2.0f),
              "subset: the ring region builds its tube: " +
                  std::string(sm.GetDiagnostic()));
        PomadeModel::OutputSettings output = sm.GetOutputSettings();
        output.enabled = true;
        Check(sm.SetOutputSettings(output), "subset: Output enables");
        PomadeCommitPaths subsetPaths = paths;
        subsetPaths.scalpPath = subsetPath;
        subsetPaths.scalpMeshPath = meshPath;
        Check(subsetPaths.ScalpMeshPath() == meshPath,
              "subset: the commit paths name the parent as the scalp mesh");
        PomadeCommitter cs(&sm, subsetPaths);
        cs.Enqueue(ss);
        Check(WaitCommitted(cs, liveS, sm.GetVersion()),
              "subset: the subset groom commits: " + cs.TakeDiagnostic());
        UsdPrim const groomS = ss->GetPrimAtPath(SdfPath("/PomadeGroom"));
        Check(bool(groomS) &&
                  Targets(groomS.GetRelationship(
                      TfToken("usdGen:pomade:scalp"))) ==
                      SdfPathVector{subsetPath},
              "subset: the groom's scalp link names the subset");
        UsdPrim const outputS = ss->GetPrimAtPath(subsetPaths.OutputPath());
        Check(bool(outputS) &&
                  Targets(outputS.GetRelationship(
                      TfToken("usdGen:surface"))) ==
                      SdfPathVector{subsetPath},
              "subset: Output grows on the subset (usdGen:surface names it)");
        Check(overHasRest(liveS, meshPath) &&
                  !liveS->GetPrimAtPath(subsetPath),
              "subset: UsdGenRestAPI lands on the parent mesh, and the subset "
              "itself carries no usdGen opinion");
        VtIntArray regions;
        bool primvarOk = ss->GetPrimAtPath(meshPath)
                             .GetAttribute(
                                 TfToken("primvars:usdGen:pomadeRegion"))
                             .Get(&regions) &&
                         regions.size() == size_t(n * n);
        for (int f = 0; primvarOk && f < n * n; ++f) {
            primvarOk = f == hole ? regions[size_t(f)] == -1
                                  : regions[size_t(f)] == 0;
        }
        Check(primvarOk && !ss->GetPrimAtPath(subsetPath).GetAttribute(
                               TfToken("primvars:usdGen:pomadeRegion")),
              "subset: the live primvar is on the parent mesh, parent-sized, "
              "-1 on the face the subset leaves out");

        PomadeModel hs;
        usdGenPomade::PomadeHydrateResult const hsr =
            usdGenPomade::PomadeHydrateModel(ss, SdfPath("/PomadeGroom"), &hs);
        Check(hsr.ok && hsr.guidesBitEqual && hsr.graphRoundTrip,
              "subset: the groom hydrates through the subset link: " +
                  hsr.diagnostic);
        std::shared_ptr<usdGenPomade::PomadeScalpMesh const> const back =
            hs.GetScalp();
        Check(back && back->activeFaces == wantActive &&
                  back->faceVertexCounts.size() == size_t(n * n) &&
                  hs.SnapshotGraph().faceRegions ==
                      sm.SnapshotGraph().faceRegions,
              "subset: hydrate re-binds the parent mesh + subset and the "
              "same parent-indexed region map");

        // A link that names a bad subset fails hydrate by name; it never
        // falls back to the whole mesh.
        ss->SetEditTarget(UsdEditTarget(ss->GetSessionLayer()));
        groomS.GetRelationship(TfToken("usdGen:pomade:scalp"))
            .SetTargets({SdfPath("/Scalp/pointSet")});
        ss->SetEditTarget(UsdEditTarget(ss->GetRootLayer()));
        PomadeModel refused;
        usdGenPomade::PomadeHydrateResult const hbad =
            usdGenPomade::PomadeHydrateModel(ss, SdfPath("/PomadeGroom"),
                                           &refused);
        Check(!hbad.ok &&
                  hbad.diagnostic.find("/Scalp/pointSet") !=
                      std::string::npos &&
                  hbad.diagnostic.find("elementType") != std::string::npos,
              "subset: hydrate refuses a non-face subset link by name (" +
                  hbad.diagnostic + ")");

        // The partial swap syncs the over onto the parent mesh too.
        SdfLayerRefPtr liveP = SdfLayer::CreateAnonymous("pomade-live-subset-p");
        UsdStageRefPtr sp = MakeStage(liveP);
        defineScene(sp);
        PomadeCommitter cp(&sm, subsetPaths);
        cp.ForcePartialModeForTest(true);
        cp.Enqueue(sp);
        size_t partialSlots = 0;
        bool partialDone = false;
        {
            auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(10000);
            while (std::chrono::steady_clock::now() < deadline) {
                if (cp.SwapIfIdle(liveP, /*gestureActive*/ false) ==
                    PomadeCommitter::PartialProgress) {
                    ++partialSlots;
                }
                if (cp.CommittedVersion() >= sm.GetVersion()) {
                    partialDone = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        Check(partialDone && partialSlots > 0 && overHasRest(liveP, meshPath) &&
                  !liveP->GetPrimAtPath(subsetPath) &&
                  bool(sp->GetPrimAtPath(meshPath).GetAttribute(
                      TfToken("primvars:usdGen:pomadeRegion"))),
              "subset: a partial swap puts RestAPI and the primvar on the "
              "parent mesh (" + std::to_string(partialSlots) + " slots): " +
                  cp.TakeDiagnostic());
    }

    // -- committer C ABI ----------------------------------------------------
    {
        Check(Pomade_CommitterCreate(nullptr, "/G", nullptr, nullptr) ==
                  POMADE_ERROR,
              "C ABI create rejects nulls");
        PomadeModelContext *ctx = nullptr;
        Check(Pomade_Create(&ctx) == POMADE_OK, "C ABI model creates");
        Check(Pomade_BuildTestTube(ctx, 0, 0, 0.0f, 0.0f) == POMADE_OK,
              "C ABI test tube builds");
        PomadeCommitterContext *cc = nullptr;
        Check(Pomade_CommitterCreate(ctx, "/PomadeGroom", "/Groom/Hair", &cc) ==
                  POMADE_OK && cc != nullptr,
              "C ABI committer creates");
        Check(Pomade_CommitterCreate(ctx, "relative", nullptr, &cc) ==
                  POMADE_ERROR,
              "C ABI create rejects a relative groom path");
        SdfLayerRefPtr liveC = SdfLayer::CreateAnonymous("pomade-live-cabi");
        Check(Pomade_CommitterEnqueue(cc, 0, 0, 0, nullptr) == POMADE_OK,
              "C ABI enqueue succeeds");
        Check(Pomade_CommitterPendingVersion(cc) == Pomade_GetVersion(ctx),
              "C ABI pending tracks the model version");
        int swapResult = PomadeCommitter_NothingPending;
        auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(5000);
        while (std::chrono::steady_clock::now() < deadline) {
            swapResult = Pomade_CommitterSwap(cc, liveC->GetIdentifier().c_str(),
                                             /*gestureActive*/ 0);
            if (Pomade_CommitterCommittedVersion(cc) == Pomade_GetVersion(ctx)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Check(swapResult == PomadeCommitter_Swapped,
              "C ABI swap lands the commit");
        Check(Pomade_CommitterSwap(cc, "no-such-layer", 0) ==
                  PomadeCommitter_Error,
              "C ABI swap rejects an unknown live layer");
        Check(Pomade_CommitterSetSwapBudgetMs(cc, 5.0) == POMADE_OK,
              "C ABI budget sets");
        // A face GeomSubset scalp names its parent Mesh (plan/02 §2.20).
        Check(Pomade_CommitterSetScalpTarget(cc, "/Scalp/patch", "/Other") ==
                  POMADE_ERROR,
              "C ABI scalp target rejects a mesh that is not the parent");
        Check(Pomade_CommitterSetScalpTarget(cc, "patch", nullptr) ==
                  POMADE_ERROR,
              "C ABI scalp target rejects a relative path");
        Check(Pomade_CommitterSetScalpTarget(cc, "/Scalp/patch", "/Scalp") ==
                  POMADE_OK,
              "C ABI scalp target takes a subset and its parent mesh");
        Check(Pomade_CommitterSetScalpTarget(cc, nullptr, nullptr) ==
                  POMADE_OK,
              "C ABI scalp target clears the link");
        Check(Pomade_CommitterDestroy(cc) == POMADE_OK, "C ABI destroys");
        Check(Pomade_Destroy(ctx) == POMADE_OK, "C ABI model destroys");
    }

    // -- P4 hooks: subdivide params + lock flags ride the commit --------------
    // The K14 fields already serialise (builder) and read back (hydrate);
    // this pins that contract so the P4 hierarchy test can subdivide with
    // it. Still TODO(P4): childIndex/deltas/level channels per tube, the
    // multi-tube prim hierarchy, K14 re-derivation bit-equality.
    {
        PomadeModel p4model;
        Check(p4model.BuildTestTube(), "P4: the hook model builds");
        PomadeModel::SubdivideParams sub;
        sub.count = 6;
        sub.seed = 42;
        sub.splitMode = "edge";
        Check(p4model.SetSubdivideParams(sub), "P4: an edge split validates");
        PomadeModel::SubdivideParams bad = sub;
        bad.count = 9;
        Check(!p4model.SetSubdivideParams(bad), "P4: count 9 is refused");
        Check(p4model.GetSubdivideParams().count == 6,
              "P4: a refused set keeps the old params");
        bad = sub;
        bad.splitMode = "bogus";
        Check(!p4model.SetSubdivideParams(bad),
              "P4: an unknown splitMode is refused");
        p4model.SetLockFlags(true, true, false);
        bool locked = false, lockParents = false, lockChildren = true;
        p4model.GetLockFlags(&locked, &lockParents, &lockChildren);
        Check(locked && lockParents && !lockChildren,
              "P4: lock flags round-trip on the model");
        SdfLayerRefPtr liveP4 = SdfLayer::CreateAnonymous("pomade-live-p4");
        UsdStageRefPtr stageP4 = MakeStage(liveP4);
        PomadeCommitter p4committer(&p4model, paths);
        p4committer.Enqueue(stageP4);
        Check(WaitCommitted(p4committer, liveP4, p4model.GetVersion()),
              "P4: the hook commit swaps");
        UsdPrim const p4tube =
            stageP4->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/tube0"));
        int subCount = 0, subSeed = 0;
        TfToken splitMode;
        Check(p4tube.GetAttribute(TfToken("usdGen:pomade:subdivide:count"))
                          .Get(&subCount) &&
                      subCount == 6 &&
                  p4tube.GetAttribute(TfToken("usdGen:pomade:subdivide:seed"))
                          .Get(&subSeed) &&
                      subSeed == 42 &&
                  p4tube
                      .GetAttribute(
                          TfToken("usdGen:pomade:subdivide:splitMode"))
                      .Get(&splitMode) &&
                  splitMode == TfToken("edge"),
              "P4: subdivide params land on the tube");
        bool stLocked = false, stLP = false, stLC = true;
        Check(p4tube.GetAttribute(TfToken("usdGen:pomade:locked"))
                          .Get(&stLocked) &&
                      stLocked &&
                  p4tube.GetAttribute(TfToken("usdGen:pomade:lockParents"))
                          .Get(&stLP) &&
                      stLP &&
                  p4tube.GetAttribute(TfToken("usdGen:pomade:lockChildren"))
                          .Get(&stLC) &&
                      !stLC,
              "P4: lock flags land on the tube");
        int childIndex = 999;
        Check(p4tube.GetAttribute(TfToken("usdGen:pomade:childIndex"))
                          .Get(&childIndex) &&
                      childIndex == -1,
              "P4: the L1 tube carries childIndex -1");
        PomadeModel p4hydrated;
        usdGenPomade::PomadeHydrateResult p4hr =
            usdGenPomade::PomadeHydrateModel(stageP4, SdfPath("/PomadeGroom"),
                                          &p4hydrated);
        Check(p4hr.ok,
              "P4: hydrate accepts the hook stage: " + p4hr.diagnostic);
        Check(p4hydrated.GetSubdivideParams().count == 6 &&
                  p4hydrated.GetSubdivideParams().seed == 42 &&
                  p4hydrated.GetSubdivideParams().splitMode == "edge",
              "P4: subdivide params survive hydrate");
        bool hyLocked = true, hyLP = false, hyLC = true;
        p4hydrated.GetLockFlags(&hyLocked, &hyLP, &hyLC);
        Check(hyLP && !hyLC, "P4: lock propagation flags survive hydrate");
    }

    // -- P6 reload/reattach: Detach idles, Reattach re-creates ---------------
    {
        PomadeModel rmodel;
        Check(rmodel.BuildTestTube(), "P6: the reload model builds");
        PomadeCommitter rcommitter(&rmodel, paths);
        SdfLayerRefPtr liveOld = SdfLayer::CreateAnonymous("pomade-live-old");
        UsdStageRefPtr stageOld = MakeStage(liveOld);
        rcommitter.Enqueue(stageOld);
        uint64_t const vOld = rmodel.GetVersion();
        Check(WaitCommitted(rcommitter, liveOld, vOld),
              "P6: the pre-reload commit swaps");
        Check(!rcommitter.IsDetached(), "P6: attached by default");
        VtVec3fArray centersOld;
        Check(stageOld->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/tube0"))
                      .GetAttribute(TfToken("usdGen:pomade:centerPoints"))
                      .Get(&centersOld) &&
                  centersOld.size() == 5,
              "P6: pre-reload centers read back");

        // The stage closes under the tool: detach; the model survives.
        rcommitter.Detach();
        Check(rcommitter.IsDetached(), "P6: Detach detaches");
        Check(rcommitter.SwapIfIdle(liveOld, /*gestureActive*/ false) ==
                  PomadeCommitter::Detached,
              "P6: swap while detached reports Detached");
        // Edits continue on the surviving model; the detached worker builds
        // nothing (the pending plan targets the dead stage).
        size_t const buildsAtDetach = rcommitter.BuildCount();
        rmodel.MoveCenterRing(0, 0.5f, 0.0f);
        uint64_t const vEdit = rmodel.GetVersion();
        rcommitter.Enqueue(stageOld);
        Check(rcommitter.PendingVersion() == vEdit,
              "P6: a detached enqueue is recorded but unbuilt");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        Check(rcommitter.BuildCount() == buildsAtDetach,
              "P6: the detached worker builds nothing");

        // A fresh stage: Reattach drops the dead-stage enqueue, and the next
        // enqueue + swap re-creates the groom prim from the surviving model.
        SdfLayerRefPtr liveNew = SdfLayer::CreateAnonymous("pomade-live-new");
        UsdStageRefPtr stageNew = MakeStage(liveNew);
        Check(!stageNew->GetPrimAtPath(SdfPath("/PomadeGroom")),
              "P6: the fresh stage starts groom-free");
        rcommitter.Reattach();
        Check(!rcommitter.IsDetached(), "P6: Reattach reattaches");
        Check(rcommitter.PendingVersion() == 0,
              "P6: Reattach drops the dead-stage enqueue");
        Check(rcommitter.CommittedVersion() == 0,
              "P6: Reattach resets the committed lineage");
        Check(rcommitter.SwapIfIdle(liveNew, /*gestureActive*/ false) ==
                  PomadeCommitter::NothingPending,
              "P6: nothing swaps until the post-reattach enqueue");
        rcommitter.Enqueue(stageNew);
        Check(WaitCommitted(rcommitter, liveNew, vEdit),
              "P6: the post-reattach commit swaps");
        UsdPrim const regroom =
            stageNew->GetPrimAtPath(SdfPath("/PomadeGroom"));
        Check(bool(regroom) &&
                  regroom.GetTypeName() == TfToken("UsdGenPomadeGroom"),
              "P6: the groom prim is re-created in the new live layer");
        VtVec3fArray centersNew;
        Check(stageNew->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/tube0"))
                      .GetAttribute(TfToken("usdGen:pomade:centerPoints"))
                      .Get(&centersNew) &&
                  centersNew.size() == 5 &&
                  std::abs(centersNew[0][0] - centersOld[0][0] - 0.5f) <
                      1e-5f,
              "P6: the post-detach edit survives the reload");

        // A shelf layer built but never swapped counts as dropped at Detach.
        PomadeCommitter dcommitter(&rmodel, paths);
        dcommitter.Enqueue(stageNew);
        size_t const b0 = dcommitter.BuildCount();
        auto dl = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(5000);
        while (dcommitter.BuildCount() == b0 &&
               std::chrono::steady_clock::now() < dl) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Check(dcommitter.BuildCount() == b0 + 1, "P6: the shelf layer builds");
        size_t const dr0 = dcommitter.DroppedCount();
        dcommitter.Detach();
        Check(dcommitter.DroppedCount() == dr0 + 1,
              "P6: Detach drops the unswapped shelf layer");

        // The C ABI pair: nulls rejected, Detached code flows over the swap.
        Check(Pomade_CommitterDetach(nullptr) == POMADE_ERROR,
              "P6: C Detach rejects null");
        Check(Pomade_CommitterReattach(nullptr) == POMADE_ERROR,
              "P6: C Reattach rejects null");
        PomadeModelContext *cctx = nullptr;
        Check(Pomade_Create(&cctx) == POMADE_OK, "P6: C reload model creates");
        Check(Pomade_BuildTestTube(cctx, 0, 0, 0.0f, 0.0f) == POMADE_OK,
              "P6: C reload tube builds");
        PomadeCommitterContext *ccc = nullptr;
        Check(Pomade_CommitterCreate(cctx, "/PomadeGroom", nullptr, &ccc) ==
                      POMADE_OK &&
                  ccc != nullptr,
              "P6: C reload committer creates");
        SdfLayerRefPtr liveC6 = SdfLayer::CreateAnonymous("pomade-live-c6");
        Check(Pomade_CommitterEnqueue(ccc, 0, 0, 0, nullptr) == POMADE_OK,
              "P6: C reload enqueue succeeds");
        auto dlC = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(5000);
        while (Pomade_CommitterCommittedVersion(ccc) !=
                   Pomade_GetVersion(cctx) &&
               std::chrono::steady_clock::now() < dlC) {
            Pomade_CommitterSwap(ccc, liveC6->GetIdentifier().c_str(), 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Check(Pomade_CommitterCommittedVersion(ccc) == Pomade_GetVersion(cctx),
              "P6: C pre-reload commit swaps");
        Check(Pomade_CommitterDetach(ccc) == POMADE_OK,
              "P6: C Detach succeeds");
        Check(Pomade_CommitterSwap(ccc, liveC6->GetIdentifier().c_str(), 0) ==
                  PomadeCommitter_Detached,
              "P6: C swap while detached reports Detached");
        Check(Pomade_CommitterReattach(ccc) == POMADE_OK,
              "P6: C Reattach succeeds");
        Check(Pomade_CommitterSwap(ccc, liveC6->GetIdentifier().c_str(), 0) ==
                  PomadeCommitter_NothingPending,
              "P6: C swap after reattach waits for a fresh enqueue");
        Check(Pomade_CommitterDestroy(ccc) == POMADE_OK, "P6: C destroys");
        Check(Pomade_Destroy(cctx) == POMADE_OK, "P6: C model destroys");
    }

    // -- V0b: the whole hierarchy round-trips the stage (plan/18 §7 G1/G3) --
    {
        PomadeModel hm;
        Check(hm.BuildTestTube(), "V0b: hierarchy model builds its L1 tube");
        PomadeModel::TubeSnapshot rotatedRoot = hm.Snapshot();
        // 90 degrees about +Z. This is deliberately non-identity before
        // subdivision, so every derived child and its stored deltas are
        // evaluated in the transported frame rather than in the old axes.
        rotatedRoot.frameReference = {{0.0f, -1.0f, 0.0f,
                                        1.0f,  0.0f, 0.0f,
                                        0.0f,  0.0f, 1.0f}};
        Check(hm.Restore(rotatedRoot),
              "V0b: rotated frame reference restores on the L1 tube");
        usdGenPomade::PomadeSnapshot qHashSnapshot =
            usdGenPomade::PomadeSnapshotFromModel(hm);
        usdGenPomade::PomadeSnapshotTube qHashChanged =
            qHashSnapshot.tubes.front();
        qHashChanged.tube.frameReference[0] = 1.0f;
        Check(usdGenPomade::PomadeSnapshotTubeHash(qHashSnapshot.tubes.front()) !=
                  usdGenPomade::PomadeSnapshotTubeHash(qHashChanged),
              "V0b: frameReference changes invalidate a tube guide hash");
        PomadeModel::FillParams hf;
        hf.density = 8.0f;
        hf.cvCount = 6;
        hf.seed = 3;
        Check(hm.SetFillParams(hf), "V0b: L1 fill params set");
        std::vector<int> kids;
        Check(hm.SubdivideTube(0, 4, "kmeans", 11, &kids) && kids.size() == 4,
              "V0b: L1 subdivides into four children");
        std::vector<int> grand;
        Check(hm.SubdivideTube(kids[1], 2, "kmeans", 5, &grand) &&
                  grand.size() == 2,
              "V0b: one L2 child subdivides into two");
        Check(hm.MoveTubeCenterCV(kids[2], 2, 0.05f, 0.0f, -0.03f),
              "V0b: a child center CV moves");
        Check(hm.MoveTubeSectionRing(kids[2], 1, 0.01f, 0.0f),
              "V0b: a child section ring moves (per-tube op)");
        Check(hm.ScaleTubeSectionRing(kids[2], 1, 1.25f) &&
                  hm.TwistTubeSectionRing(kids[2], 1, 0.19f),
              "V0b: a child section scale and twist are authored");
        hm.SetTubeLockChildren(kids[3], true);
        hm.SetTubeLockParents(kids[3], true);
        PomadeModel::FillParams cf;
        cf.density = 5.0f;
        cf.cvCount = 4;
        cf.seed = 9;
        Check(hm.SetTubeFillParams(grand[0], cf),
              "V0b: a grandchild carries its own fill params");
        int groupId = 0;
        Check(hm.GroupTubes({kids[0], kids[3]}, /*transient*/ false,
                            &groupId) &&
                  groupId == -1,
              "V0b: two siblings group under an on-the-fly parent");
        Check(hm.MakeTubePersistent(groupId),
              "V0b: the on-the-fly parent is made persistent");
        int groupId2 = 0;
        Check(hm.GroupTubes({grand[0], grand[1]}, /*transient*/ false,
                            &groupId2) &&
                  groupId2 == -2,
              "V0b: a second on-the-fly parent mints -2");
        Check(hm.MakeTubePersistent(groupId2),
              "V0b: the second on-the-fly parent is kept too");
        int transientId = 0;
        Check(hm.GroupTubes({kids[2]}, /*transient*/ true, &transientId),
              "V0b: a transient parent exists in the model");

        SdfLayerRefPtr liveH = SdfLayer::CreateAnonymous("pomade-live-h");
        UsdStageRefPtr sh = MakeStage(liveH);
        PomadeCommitter ch(&hm, paths);
        ch.Enqueue(sh);
        Check(WaitCommitted(ch, liveH, hm.GetVersion()),
              "V0b: the hierarchy commits");

        VtFloatArray storedFrameReference;
        Check(sh->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/tube0"))
                      .GetAttribute(TfToken("usdGen:pomade:frameReference"))
                      .Get(&storedFrameReference) &&
                  storedFrameReference.size() == 9 &&
                  storedFrameReference[0] == 0.0f &&
                  storedFrameReference[1] == -1.0f &&
                  storedFrameReference[3] == 1.0f &&
                  storedFrameReference[8] == 1.0f,
              "V0b: row-major frameReference persists on the root tube");

        // Nesting: children are namespace children of their parent.
        Check(bool(sh->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/tube0"))),
              "V0b: the L1 tube lands under Tubes");
        Check(bool(sh->GetPrimAtPath(
                  SdfPath("/PomadeGroom/Tubes/tube0/tube2"))),
              "V0b: an L2 child nests under its parent");
        UsdPrim const l3 = sh->GetPrimAtPath(
            SdfPath("/PomadeGroom/Tubes/tube0/tube2/tube33"));
        Check(bool(l3), "V0b: an L3 child nests two deep");
        if (l3) {
            int level = 0, childIndex = -99;
            l3.GetAttribute(TfToken("usdGen:pomade:level")).Get(&level);
            l3.GetAttribute(TfToken("usdGen:pomade:childIndex"))
                .Get(&childIndex);
            Check(level == 3 && childIndex == 0,
                  "V0b: level and childIndex are the real values");
        }
        UsdPrim const edited =
            sh->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/tube0/tube3"));
        Check(bool(edited), "V0b: the sculpted child is on the stage");
        if (edited) {
            VtVec3fArray deltas;
            edited.GetAttribute(TfToken("usdGen:pomade:centerDeltas"))
                .Get(&deltas);
            bool nonZero = false;
            for (auto const &d : deltas) {
                nonZero = nonZero || d[0] != 0.0f || d[1] != 0.0f ||
                          d[2] != 0.0f;
            }
            Check(!deltas.empty() && nonZero,
                  "V0b: the sculpted child commits non-empty centerDeltas");
            VtVec2fArray sectionDeltas;
            edited.GetAttribute(TfToken("usdGen:pomade:sectionDeltas"))
                .Get(&sectionDeltas);
            Check(!sectionDeltas.empty(),
                  "V0b: section deltas commit alongside them");
            VtVec2fArray sectionDeltaTransforms;
            edited.GetAttribute(TfToken("usdGen:pomade:sectionDeltaTransforms"))
                .Get(&sectionDeltaTransforms);
            bool nonZeroTransform = false;
            for (GfVec2f const &d : sectionDeltaTransforms) {
                nonZeroTransform = nonZeroTransform || d[0] != 0.0f ||
                                   d[1] != 0.0f;
            }
            Check(!sectionDeltaTransforms.empty() && nonZeroTransform,
                  "V0b: section scale/twist residuals commit alongside UV deltas");
            VtIntArray bindings;
            PomadeModel::TubeRecord editedRecord;
            bool bindingsExact =
                hm.GetTubeRecord(kids[2], &editedRecord) &&
                edited.GetAttribute(
                    TfToken("usdGen:pomade:inheritedBoundaryBindings"))
                    .Get(&bindings) &&
                bindings.size() ==
                    editedRecord.actual.inheritedBoundaryBindings.size() * 3;
            for (size_t i = 0; bindingsExact &&
                 i < editedRecord.actual.inheritedBoundaryBindings.size();
                 ++i) {
                usdGenPomade::PomadeParentBoundaryBinding const &binding =
                    editedRecord.actual.inheritedBoundaryBindings[i];
                bindingsExact = bindings[i * 3 + 0] == binding.section &&
                                bindings[i * 3 + 1] == binding.parentSlot &&
                                bindings[i * 3 + 2] == binding.childSlot;
            }
            Check(bindingsExact,
                  "V0b: inherited parent-boundary bindings commit exactly");
        }
        UsdPrim const group =
            sh->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/group1"));
        Check(bool(group), "V0b: the persistent on-the-fly parent commits");
        if (group) {
            bool persistent = false;
            group.GetAttribute(TfToken("usdGen:pomade:persistent"))
                .Get(&persistent);
            SdfPathVector members =
                Targets(group.GetRelationship(TfToken("usdGen:pomade:members")));
            Check(persistent && members.size() == 2,
                  "V0b: HierarchyAPI carries persistent + two members");
            Check(group.HasAPI(TfToken("UsdGenTubeHierarchyAPI")) ||
                      !members.empty(),
                  "V0b: the hierarchy API schema is applied");
        }
        Check(bool(sh->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/group2"))),
              "V0b: a second on-the-fly parent sits beside the first, not "
              "under it");
        Check(!sh->GetPrimAtPath(SdfPath("/PomadeGroom/Tubes/group3")),
              "V0b: the transient parent is not committed");

        // Guides: per leaf tube, never for a subdivided parent.
        UsdPrim const guidesPrim =
            sh->GetPrimAtPath(SdfPath("/PomadeGroom/Guides"));
        VtIntArray guideTubeIds, guideLevels;
        guidesPrim.GetAttribute(TfToken("primvars:tubeId")).Get(&guideTubeIds);
        guidesPrim.GetAttribute(TfToken("primvars:hierarchyLevel"))
            .Get(&guideLevels);
        bool anyParentGuides = false;
        bool anyLeafGuides = false;
        for (size_t i = 0; i < guideTubeIds.size(); ++i) {
            if (guideTubeIds[i] == 0 || guideTubeIds[i] == kids[1] ||
                guideTubeIds[i] < 0) {
                // A subdivided parent's fill is suspended, and an
                // on-the-fly parent aggregates tubes that fill themselves.
                anyParentGuides = true;
            }
            if (guideTubeIds[i] == grand[0] && guideLevels[i] == 3) {
                anyLeafGuides = true;
            }
        }
        Check(!anyParentGuides,
              "V0b: a subdivided parent's fill is suspended");
        Check(anyLeafGuides,
              "V0b: leaf tubes carry their own tubeId/hierarchyLevel");

        // Hydrate into a fresh model and compare every tube bit for bit.
        PomadeModel hy;
        usdGenPomade::PomadeHydrateResult const hyr =
            usdGenPomade::PomadeHydrateModel(sh, SdfPath("/PomadeGroom"), &hy);
        Check(hyr.ok, "V0b: hydrate accepts the committed hierarchy: " +
                          hyr.diagnostic);
        Check(hyr.guidesBitEqual, "V0b: hydrated guides are bit-equal");
        std::vector<int> const beforeIds = hm.TubeIds();
        std::vector<int> const afterIds = hy.TubeIds();
        // The transient parent is not committed, so it is the one tube the
        // round trip drops by design.
        std::vector<int> wantIds;
        for (int id : beforeIds) {
            if (id != transientId) {
                wantIds.push_back(id);
            }
        }
        Check(wantIds == afterIds,
              "V0b: every committed tube id comes back (transient dropped)");
        bool allEqual = true;
        for (int id : afterIds) {
            PomadeModel::TubeRecord a, b;
            if (!hm.GetTubeRecord(id, &a) || !hy.GetTubeRecord(id, &b)) {
                allEqual = false;
                break;
            }
            if (!SameDesc(a.actual, b.actual)) {
                std::printf("      desc differs at tube %d: centers %d "
                            "sections %d ringVerts %d/%d region %d/%d\n",
                            id,
                            int(a.actual.centerX == b.actual.centerX &&
                                a.actual.centerY == b.actual.centerY &&
                                a.actual.centerZ == b.actual.centerZ),
                            int(SameSections(a.actual.sections,
                                             b.actual.sections)),
                            a.actual.ringVerts, b.actual.ringVerts,
                            a.actual.regionId, b.actual.regionId);
            }
            if (!SameDeltas(a.deltas, b.deltas)) {
                std::printf("      deltas differ at tube %d (%zu/%zu cv, "
                            "%zu/%zu sections)\n",
                            id, a.deltas.centerDu.size(),
                            b.deltas.centerDu.size(), a.deltas.sections.size(),
                            b.deltas.sections.size());
            }
            allEqual = allEqual && SameDesc(a.actual, b.actual) &&
                       SameTessellation(a.actual, b.actual) &&
                       SameDeltas(a.deltas, b.deltas) &&
                       a.actual.level == b.actual.level &&
                       a.actual.childIndex == b.actual.childIndex &&
                       a.actual.parentTubeId == b.actual.parentTubeId &&
                       a.subdivide.count == b.subdivide.count &&
                       a.subdivide.seed == b.subdivide.seed &&
                       a.subdivide.splitMode == b.subdivide.splitMode &&
                       a.fill.density == b.fill.density &&
                       a.fill.cvCount == b.fill.cvCount &&
                       a.fill.seed == b.fill.seed &&
                       a.lockParents == b.lockParents &&
                       a.lockChildren == b.lockChildren &&
                       a.persistent == b.persistent &&
                       a.imported == b.imported;
            if (!allEqual) {
                std::printf("      first mismatch at tube %d\n", id);
                break;
            }
        }
        Check(allEqual,
              "V0b: rotated frame, shape, deltas, links, fill, locks and persistent "
              "round-trip bit-exactly");

        // Older layers have no boundary-link attribute. Hydrate reconstructs
        // those material identities from the freshly derived parent chart so
        // their next K7 uses the same inherited outer corners.
        {
            SdfLayerRefPtr legacyBindings =
                SdfLayer::CreateAnonymous("pomade-legacy-boundaries");
            legacyBindings->TransferContent(liveH);
            SdfPrimSpecHandle legacyChild = legacyBindings->GetPrimAtPath(
                SdfPath("/PomadeGroom/Tubes/tube0/tube3"));
            SdfPropertySpecHandle boundarySpec =
                legacyBindings->GetPropertyAtPath(
                    SdfPath("/PomadeGroom/Tubes/tube0/tube3")
                        .AppendProperty(TfToken(
                            "usdGen:pomade:inheritedBoundaryBindings")));
            bool const removedBindings = bool(legacyChild) &&
                bool(boundarySpec);
            if (removedBindings) {
                legacyChild->RemoveProperty(boundarySpec);
            }
            PomadeModel legacyBindingModel;
            usdGenPomade::PomadeHydrateResult const legacyBindingResult =
                usdGenPomade::PomadeHydrateModel(MakeStage(legacyBindings),
                                               SdfPath("/PomadeGroom"),
                                               &legacyBindingModel);
            PomadeModel::TubeRecord beforeLegacy, afterLegacy;
            Check(removedBindings && legacyBindingResult.ok &&
                      hm.GetTubeRecord(kids[2], &beforeLegacy) &&
                      legacyBindingModel.GetTubeRecord(kids[2], &afterLegacy) &&
                      SameDesc(beforeLegacy.actual, afterLegacy.actual),
                  "V0b: missing boundary bindings reconstruct from the derived parent chart");
        }

        // Re-commit the hydrated model: the same guide bytes.
        {
            SdfLayerRefPtr live2 = SdfLayer::CreateAnonymous("pomade-live-h2");
            UsdStageRefPtr s2 = MakeStage(live2);
            PomadeCommitter c2(&hy, paths);
            c2.Enqueue(s2);
            Check(WaitCommitted(c2, live2, hy.GetVersion()),
                  "V0b: the hydrated hierarchy re-commits");
            VtVec3fArray a, b;
            Check(GetPoints(sh, SdfPath("/PomadeGroom/Guides"), &a) &&
                      GetPoints(s2, SdfPath("/PomadeGroom/Guides"), &b) &&
                      a.size() == b.size() && !a.empty() &&
                      std::memcmp(a.data(), b.data(),
                                  a.size() * sizeof(GfVec3f)) == 0,
                  "V0b: rotated hierarchy re-commits byte-identical world guides");
        }

        // The optional raw section representation is needed for continued
        // hierarchy editing, not merely equivalent first-frame rendering.
        // Move the shared parent after hydrate and compare the full affected
        // child record, including scalar and per-CV residuals.
        // Keep the committed source model intact: the foreign-guide fixture
        // below composes the already committed stage and must use positions
        // from that same snapshot.  Continue hierarchy editing in two fresh
        // clones instead.
        PomadeModel sourceEdit, hydratedEdit;
        usdGenPomade::PomadeHydrateResult const sourceEditHydrate =
            usdGenPomade::PomadeHydrateModel(sh, SdfPath("/PomadeGroom"),
                                           &sourceEdit);
        usdGenPomade::PomadeHydrateResult const hydratedEditHydrate =
            usdGenPomade::PomadeHydrateModel(sh, SdfPath("/PomadeGroom"),
                                           &hydratedEdit);
        PomadeModel::TubeRecord sourceAfterK6, hydratedAfterK6;
        bool const continuedK6 = sourceEditHydrate.ok &&
            hydratedEditHydrate.ok &&
            sourceEdit.MoveTubeCenterCV(0, 1, 0.02f, -0.01f, 0.01f) &&
            hydratedEdit.MoveTubeCenterCV(0, 1, 0.02f, -0.01f, 0.01f) &&
            sourceEdit.GetTubeRecord(kids[2], &sourceAfterK6) &&
            hydratedEdit.GetTubeRecord(kids[2], &hydratedAfterK6);
        Check(continuedK6 && SameDesc(sourceAfterK6.actual,
                                      hydratedAfterK6.actual) &&
                  SameTessellation(sourceAfterK6.actual,
                                   hydratedAfterK6.actual) &&
                  SameDeltas(sourceAfterK6.deltas,
                             hydratedAfterK6.deltas),
              "V0b: hydrated scale/twist child remains exact through a later parent K6");

        // K14 generation marker (review 2026-09-24): K14's child slot order
        // changed with the ring alignment, and stored residuals are per slot
        // of the derivation that measured them. A marked groom installs them
        // verbatim (bit-exact hydrate); a groom without the marker was
        // written by the old K14, so its residuals are re-measured from the
        // stored actual against today's derivation instead of being added
        // to the wrong slots by the next K6. Simulate the old slot order by
        // rotating tube3's stored section residual one slot per section.
        {
            TfToken subdivider;
            UsdPrim const groomPrim = sh->GetPrimAtPath(SdfPath("/PomadeGroom"));
            Check(groomPrim &&
                      groomPrim.GetAttribute(TfToken("usdGen:pomade:subdivider"))
                          .Get(&subdivider) &&
                      subdivider == TfToken("aligned-v1"),
                  "K14 marker: the commit records usdGen:pomade:subdivider = "
                  "aligned-v1");
            SdfPath const childPath("/PomadeGroom/Tubes/tube0/tube3");
            PomadeModel::TubeRecord source;
            bool const haveSource = hm.GetTubeRecord(kids[2], &source) &&
                                    source.actual.ringVerts >= 3;
            int const ringVerts = haveSource ? source.actual.ringVerts : 0;
            VtVec2fArray rotatedDeltas;
            // marker: "keep", "drop" or any other token to author
            auto variant = [&](char const *name, char const *marker) {
                SdfLayerRefPtr layer = SdfLayer::CreateAnonymous(name);
                layer->TransferContent(liveH);
                SdfAttributeSpecHandle const deltasSpec =
                    layer->GetAttributeAtPath(childPath.AppendProperty(
                        TfToken("usdGen:pomade:sectionDeltas")));
                if (deltasSpec && ringVerts > 0) {
                    VtValue const value = deltasSpec->GetDefaultValue();
                    if (value.IsHolding<VtVec2fArray>()) {
                        VtVec2fArray d = value.UncheckedGet<VtVec2fArray>();
                        VtVec2fArray r = d;
                        size_t const n = size_t(ringVerts);
                        for (size_t base = 0; base + n <= d.size(); base += n) {
                            for (size_t i = 0; i < n; ++i) {
                                r[base + i] = d[base + (i + 1) % n];
                            }
                        }
                        deltasSpec->SetDefaultValue(VtValue(r));
                        rotatedDeltas = r;
                    }
                }
                SdfPath const markerPath =
                    SdfPath("/PomadeGroom")
                        .AppendProperty(TfToken("usdGen:pomade:subdivider"));
                SdfAttributeSpecHandle const markerSpec =
                    layer->GetAttributeAtPath(markerPath);
                if (std::string(marker) == "drop") {
                    if (markerSpec) {
                        layer->GetPrimAtPath(SdfPath("/PomadeGroom"))
                            ->RemoveProperty(markerSpec);
                    }
                } else if (std::string(marker) != "keep" && markerSpec) {
                    markerSpec->SetDefaultValue(VtValue(TfToken(marker)));
                }
                return layer;
            };
            auto sectionsNear = [](usdGenPomade::PomadeShapeDeltas const &a,
                                   usdGenPomade::PomadeShapeDeltas const &b) {
                if (a.sections.size() != b.sections.size()) {
                    return false;
                }
                for (size_t s = 0; s < a.sections.size(); ++s) {
                    auto const &x = a.sections[s];
                    auto const &y = b.sections[s];
                    if (x.u.size() != y.u.size() || x.v.size() != y.v.size() ||
                        std::fabs(x.scale - y.scale) > 1e-4f ||
                        std::fabs(x.twist - y.twist) > 1e-4f) {
                        return false;
                    }
                    for (size_t i = 0; i < x.u.size(); ++i) {
                        if (std::fabs(x.u[i] - y.u[i]) > 1e-4f ||
                            std::fabs(x.v[i] - y.v[i]) > 1e-4f) {
                            return false;
                        }
                    }
                }
                return true;
            };

            // Marked: the rotated residual is installed exactly as stored.
            PomadeModel marked;
            usdGenPomade::PomadeHydrateResult const markedResult =
                usdGenPomade::PomadeHydrateModel(
                    MakeStage(variant("pomade-k14-marked", "keep")),
                    SdfPath("/PomadeGroom"), &marked);
            PomadeModel::TubeRecord markedRecord;
            bool verbatim = haveSource && markedResult.ok &&
                            !rotatedDeltas.empty() &&
                            marked.GetTubeRecord(kids[2], &markedRecord);
            if (verbatim) {
                size_t k = 0;
                for (auto const &sec : markedRecord.deltas.sections) {
                    for (size_t i = 0; verbatim && i < sec.u.size(); ++i, ++k) {
                        verbatim = k < rotatedDeltas.size() &&
                                   sec.u[i] == rotatedDeltas[k][0] &&
                                   sec.v[i] == rotatedDeltas[k][1];
                    }
                }
                verbatim = verbatim && k == rotatedDeltas.size();
            }
            Check(verbatim,
                  "K14 marker: a marked groom installs its stored residuals "
                  "verbatim");

            // Unmarked (pre-alignment): re-measured against today's K14.
            PomadeModel legacy;
            usdGenPomade::PomadeHydrateResult const legacyResult =
                usdGenPomade::PomadeHydrateModel(
                    MakeStage(variant("pomade-k14-legacy", "drop")),
                    SdfPath("/PomadeGroom"), &legacy);
            PomadeModel::TubeRecord legacyRecord;
            Check(haveSource && legacyResult.ok &&
                      legacy.GetTubeRecord(kids[2], &legacyRecord) &&
                      SameDesc(legacyRecord.actual, source.actual) &&
                      sectionsNear(legacyRecord.deltas, source.deltas),
                  "K14 marker: an unmarked groom re-measures its residuals "
                  "from the stored actual (the stale slot order is gone): " +
                      legacyResult.diagnostic);
            PomadeModel::TubeRecord legacyAfterK6, sourceAfterK6b;
            PomadeModel sourceClone;
            bool const k6 =
                usdGenPomade::PomadeHydrateModel(sh, SdfPath("/PomadeGroom"),
                                               &sourceClone).ok &&
                legacy.MoveTubeCenterCV(0, 1, 0.02f, -0.01f, 0.01f) &&
                sourceClone.MoveTubeCenterCV(0, 1, 0.02f, -0.01f, 0.01f) &&
                legacy.GetTubeRecord(kids[2], &legacyAfterK6) &&
                sourceClone.GetTubeRecord(kids[2], &sourceAfterK6b);
            Check(k6 && sectionsNear(legacyAfterK6.deltas,
                                     sourceAfterK6b.deltas),
                  "K14 marker: and a later parent K6 treats the migrated "
                  "child like the source");

            PomadeModel unknown;
            usdGenPomade::PomadeHydrateResult const unknownResult =
                usdGenPomade::PomadeHydrateModel(
                    MakeStage(variant("pomade-k14-unknown", "aligned-v9")),
                    SdfPath("/PomadeGroom"), &unknown);
            Check(!unknownResult.ok &&
                      unknownResult.diagnostic.find("subdivider") !=
                          std::string::npos,
                  "K14 marker: an unknown subdivider is refused, not "
                  "guessed (" + unknownResult.diagnostic + ")");
        }

        // A hand-authored foreign guide becomes a locked L3 tube.
        {
            UsdStageRefPtr fs = UsdStage::CreateInMemory("pomade-foreign-v0b");
            SdfLayerRefPtr over =
                SdfLayer::CreateAnonymous("pomade-foreign-over-v0b");
            fs->GetSessionLayer()->InsertSubLayerPath(liveH->GetIdentifier(),
                                                      0);
            fs->GetSessionLayer()->InsertSubLayerPath(over->GetIdentifier(), 0);
            fs->SetEditTarget(over);
            UsdPrim fg = fs->GetPrimAtPath(SdfPath("/PomadeGroom/Guides"));
            VtVec3fArray pts;
            VtIntArray counts;
            VtUInt64Array ids;
            VtIntArray tubeIds, levels, regions;
            fg.GetAttribute(TfToken("points")).Get(&pts);
            fg.GetAttribute(TfToken("curveVertexCounts")).Get(&counts);
            fg.GetAttribute(TfToken("primvars:usdGen:curveId")).Get(&ids);
            fg.GetAttribute(TfToken("primvars:tubeId")).Get(&tubeIds);
            fg.GetAttribute(TfToken("primvars:hierarchyLevel")).Get(&levels);
            fg.GetAttribute(TfToken("primvars:regionId")).Get(&regions);
            // Root it inside the first L2 child's cell: that child has no
            // children of its own, so the import lands at L3.
            float rx = 0.0f, ry = 0.0f, rz = 0.0f;
            Check(hm.GetTubeCenterCV(kids[0], 0, &rx, &ry, &rz),
                  "V0b: the import parent's root is readable");
            for (int c = 0; c < 4; ++c) {
                pts.push_back(
                    GfVec3f(rx, ry + 0.4f * float(c), rz + 0.02f * float(c)));
            }
            counts.push_back(4);
            ids.push_back(999999);
            tubeIds.push_back(-777);  // claimed by no tube: foreign
            levels.push_back(0);
            regions.push_back(-1);
            fg.GetAttribute(TfToken("points")).Set(pts);
            fg.GetAttribute(TfToken("curveVertexCounts")).Set(counts);
            fg.GetAttribute(TfToken("primvars:usdGen:curveId")).Set(ids);
            fg.GetAttribute(TfToken("primvars:tubeId")).Set(tubeIds);
            fg.GetAttribute(TfToken("primvars:hierarchyLevel")).Set(levels);
            fg.GetAttribute(TfToken("primvars:regionId")).Set(regions);

            PomadeModel fm;
            usdGenPomade::PomadeHydrateResult const fr =
                usdGenPomade::PomadeHydrateModel(fs, SdfPath("/PomadeGroom"),
                                               &fm);
            Check(fr.ok, "V0b: a foreign guide does not fail hydrate: " +
                             fr.diagnostic);
            Check(fr.importedTubeCount == 1,
                  "V0b: the foreign guide is imported as one tube");
            int importedId = -1;
            for (int id : fm.TubeIds()) {
                if (fm.IsTubeImported(id)) {
                    importedId = id;
                }
            }
            Check(importedId != -1, "V0b: the import is in the model");
            if (importedId != -1) {
                usdGenPomade::PomadeTubeDesc desc;
                fm.GetTubeDesc(importedId, &desc);
                Check(desc.level == 3,
                      "V0b: the foreign guide imports as an L3 tube");
                Check(desc.parentTubeId == kids[0],
                      "V0b: it hangs off the tube whose region roots it");
                Check(int(desc.centerX.size()) == 4,
                      "V0b: its centers are the foreign curve's CVs");
            }
        }
    }

    // -- V0b: a worker throw keeps the previous layer and the thread -------
    {
        PomadeModel tm;
        Check(tm.BuildTestTube(), "V0b: throw-test model builds");
        SdfLayerRefPtr liveT = SdfLayer::CreateAnonymous("pomade-live-throw");
        UsdStageRefPtr st = MakeStage(liveT);
        PomadeCommitter ct(&tm, paths);
        ct.Enqueue(st);
        Check(WaitCommitted(ct, liveT, tm.GetVersion()),
              "V0b: the first commit lands");
        VtVec3fArray before;
        Check(GetPoints(st, SdfPath("/PomadeGroom/Guides"), &before) &&
                  !before.empty(),
              "V0b: the committed guides are readable");
        uint64_t const committedBefore = ct.CommittedVersion();

        ct.ThrowOnNextBuildsForTest(1);
        Check(tm.MoveCenterRing(1, 0.25f, 0.0f), "V0b: the model moves");
        ct.Enqueue(st);
        auto const deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(5000);
        while (ct.WorkerThrowCount() == 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Check(ct.WorkerThrowCount() == 1,
              "V0b: the commit worker caught the throw");
        Check(ct.SwapIfIdle(liveT, false) == PomadeCommitter::NothingPending,
              "V0b: nothing swaps after the failed build");
        VtVec3fArray after;
        Check(GetPoints(st, SdfPath("/PomadeGroom/Guides"), &after) &&
                  after.size() == before.size() &&
                  std::memcmp(after.data(), before.data(),
                              after.size() * sizeof(GfVec3f)) == 0,
              "V0b: the previous live layer is untouched");
        Check(ct.CommittedVersion() == committedBefore,
              "V0b: the committed version does not advance");
        Check(!ct.TakeDiagnostic().empty(),
              "V0b: the error reaches the status path");
        Check(tm.GetVersion() > committedBefore, "V0b: the model is intact");

        // The thread survives: a newer version still builds and swaps.
        Check(tm.MoveCenterRing(1, -0.25f, 0.0f), "V0b: the model moves back");
        ct.Enqueue(st);
        Check(WaitCommitted(ct, liveT, tm.GetVersion()),
              "V0b: the worker survives the throw and commits again");
    }

    // -- V0b: Save groom writes a RELATIVE map path ------------------------
    {
        namespace fs = std::filesystem;
        PomadeModel sm;
        Check(sm.BuildTestTube(), "V0b: save-test model builds");
        SdfLayerRefPtr liveS = SdfLayer::CreateAnonymous("pomade-live-save");
        UsdStageRefPtr ss = MakeStage(liveS);
        PomadeCommitter cs(&sm, paths);
        cs.Enqueue(ss);
        Check(WaitCommitted(cs, liveS, sm.GetVersion()),
              "V0b: the save-test commit lands");
        std::error_code ec;
        fs::path const dir =
            fs::temp_directory_path(ec) / "usdGenPomadeSaveV0b";
        fs::remove_all(dir, ec);
        fs::create_directories(dir, ec);
        fs::path const mapFile = dir / "regionMap.v7.ptx";
        {
            std::FILE *f = std::fopen(mapFile.string().c_str(), "wb");
            if (f) {
                std::fputs("not a real ptex, only a file to copy", f);
                std::fclose(f);
            }
        }
        fs::path const saved = dir / "groom.usdc";
        std::string err;
        Check(usdGenPomade::PomadeSaveGroomAndMaps(ss, liveS, saved.string(),
                                                 mapFile.string(), paths,
                                                 &err),
              "V0b: Save groom with maps succeeds: " + err);
        Check(fs::exists(dir / "regionMap.ptx", ec),
              "V0b: the map is copied beside the saved layer");
        SdfLayerRefPtr savedLayer = SdfLayer::FindOrOpen(saved.string());
        Check(bool(savedLayer), "V0b: the saved layer reopens");
        if (savedLayer) {
            SdfPrimSpecHandle const mapPrim =
                savedLayer->GetPrimAtPath(paths.RegionMapPath());
            std::string authored;
            if (mapPrim) {
                for (SdfAttributeSpecHandle const &attr :
                     mapPrim->GetAttributes()) {
                    if (attr->GetName() == "usdGen:map:file") {
                        authored =
                            attr->GetDefaultValue().Get<SdfAssetPath>()
                                .GetAssetPath();
                    }
                }
            }
            Check(authored == "./regionMap.ptx",
                  "V0b: usdGen:map:file is relative to the saved layer, got \"" +
                      authored + "\"");
        }
        fs::remove_all(dir, ec);
    }

    // -- V6 / G7: the committer cancels the description's cook ------------
    //
    // plan/17 §3.2 rule 2. The swap dirties <groom>/Guides, the dirty router
    // dirties the linked description and its session cooks; a press means
    // that cook describes a groom the model has left. The committer bumps
    // the cancellation token on the sessions rooted at its description, and
    // nowhere else.
    {
        using usdGenImaging::UsdGenSessionKey;
        using usdGenImaging::UsdGenSessionStore;
        PomadeModel cookModel;
        Check(cookModel.BuildTestTube(), "G7: the cook-token model builds");
        PomadeCommitter cookCommitter(&cookModel, paths);
        Check(cookCommitter.CancelDescriptionCooks() == 0,
              "G7: with no session attached there is nothing to cancel");

        UsdGenSessionKey ours;
        ours.sessionId = "pomade-commit-g7";
        ours.groomRoot = paths.descriptionPath;  // /Groom/Hair
        ours.renderInstanceId = 7001;
        UsdGenSessionKey other = ours;
        other.sessionId = "pomade-commit-g7-other";
        other.groomRoot = SdfPath("/Groom/OtherHair");
        other.renderInstanceId = 7002;
        auto &store = UsdGenSessionStore::GetInstance();
        auto oursSession = store.Attach(ours);
        auto otherSession = store.Attach(other);
        Check(bool(oursSession) && bool(otherSession),
              "G7: two sessions attach, one under our description");
        if (oursSession && otherSession) {
            uint64_t const ourBefore = oursSession->CookToken();
            uint64_t const otherBefore = otherSession->CookToken();
            Check(cookCommitter.CancelDescriptionCooks() == 1,
                  "G7: exactly our description's session is cancelled");
            Check(oursSession->CookToken() == ourBefore + 1,
                  "G7: its token moved");
            Check(otherSession->CookToken() == otherBefore,
                  "G7: another groom's session is untouched");
            // The C ABI drives the same thing (the ctypes path the tool
            // takes at gesture press).
            PomadeModelContext *cookCtx = nullptr;
            PomadeCommitterContext *cookCc = nullptr;
            Check(Pomade_Create(&cookCtx) == POMADE_OK &&
                      Pomade_BuildTestTube(cookCtx, 0, 0, 0.0f, 0.0f) == POMADE_OK,
                  "G7: the C ABI model builds");
            Check(Pomade_CommitterCreate(cookCtx, "/PomadeGroom", "/Groom/Hair",
                                        &cookCc) == POMADE_OK,
                  "G7: the C ABI committer creates");
            Check(Pomade_CommitterCancelCooks(cookCc) == 1,
                  "G7: Pomade_CommitterCancelCooks cancels the same one");
            Check(oursSession->CookToken() == ourBefore + 2,
                  "G7: through the ABI the token moves again");
            Check(Pomade_CommitterCancelCooks(nullptr) == -1,
                  "G7: a null committer is an error, not a crash");
            Pomade_CommitterDestroy(cookCc);
            Pomade_Destroy(cookCtx);
        }
        store.Detach(ours);
        store.Detach(other);
    }

    // -- V6 / G6: the per-version dirty set for the guide refill ----------
    //
    // Before V6 every commit re-ran K8/K9/K10 over every tube, so moving one
    // CV of one tube in the reference groom refilled all 12 000 guides. The
    // cache keys each tube's slice on the bytes the fill reads, so a version
    // that touched one tube refills one tube. Two things have to hold: the
    // cached result is BYTE-IDENTICAL to the uncached one (otherwise hydrate's
    // bit-equality assertion would start failing at random), and the saving is
    // real at reference scale.
    {
        usdGenPomade::PomadeSnapshot big;
        big.version = 909;
        for (int t = 0; t < 2400; ++t) {
            usdGenPomade::PomadeSnapshotTube entry;
            entry.tubeId = t;
            entry.regionId = t % 80;
            entry.level = 3;
            entry.tube.hasTube = true;
            entry.tube.shape.rings = 20;
            entry.tube.shape.ringVerts = 16;
            entry.tube.shape.radius = 0.5f;
            entry.tube.shape.length = 4.0f;
            entry.tube.centerX.assign(20, float(t) * 0.01f);
            entry.tube.centerY.resize(20);
            entry.tube.centerZ.assign(20, 0.0f);
            for (int r = 0; r < 20; ++r) {
                entry.tube.centerY[size_t(r)] = 4.0f * float(r) / 19.0f;
            }
            entry.tube.fill.density = 5.0f;
            entry.tube.fill.cvCount = 16;
            entry.tube.fill.seed = t;
            big.tubes.push_back(std::move(entry));
        }
        usdGenPomade::PomadeHashSnapshot(&big);

        auto timeGuides = [](usdGenPomade::PomadeSnapshot const &snap,
                             usdGenPomade::PomadeGuideCache *cache,
                             usdGenPomade::PomadeSnapshotGuides *out) {
            auto const t0 = std::chrono::steady_clock::now();
            *out = usdGenPomade::PomadeGuidesFromSnapshot(snap, cache);
            auto const t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        usdGenPomade::PomadeSnapshotGuides cold;
        double const coldMs = timeGuides(big, nullptr, &cold);
        Check(cold.counts.size() == 12000,
              "G6: the reference groom fills 12000 guides (got " +
                  std::to_string(cold.counts.size()) + ")");
        std::printf("info: G6 guide refill, whole groom (before): %.1f ms\n",
                    coldMs);

        usdGenPomade::PomadeGuideCache cache;
        usdGenPomade::PomadeSnapshotGuides warmUp;
        double const fillMs = timeGuides(big, &cache, &warmUp);
        Check(cache.LastRefilled() == 2400 && cache.LastReused() == 0,
              "G6: the first version through the cache refills every tube");
        Check(warmUp.points == cold.points && warmUp.counts == cold.counts &&
                  warmUp.frames == cold.frames && warmUp.ids == cold.ids &&
                  warmUp.tubeIds == cold.tubeIds &&
                  warmUp.levels == cold.levels &&
                  warmUp.regionIds == cold.regionIds,
              "G6: filling through the cache changes not one byte");
        std::printf("info: G6 cache-cold refill: %.1f ms (%zu entries)\n",
                    fillMs, cache.Size());

        // One tube moves, exactly as one drag on one CV does.
        usdGenPomade::PomadeSnapshot edited = big;
        edited.version = 910;
        edited.tubes[1234].tube.centerX[7] += 0.25f;
        usdGenPomade::PomadeHashSnapshot(&edited);
        Check(edited.tubes[1234].contentHash != big.tubes[1234].contentHash,
              "G6: the moved tube's content hash changes");
        Check(edited.tubes[1233].contentHash == big.tubes[1233].contentHash,
              "G6: its neighbour's does not");

        usdGenPomade::PomadeSnapshotGuides editedCold;
        double const editedColdMs = timeGuides(edited, nullptr, &editedCold);
        usdGenPomade::PomadeSnapshotGuides editedWarm;
        double const editedWarmMs = timeGuides(edited, &cache, &editedWarm);
        Check(cache.LastRefilled() == 1 && cache.LastReused() == 2399,
              "G6: a one-tube edit refills exactly one tube (refilled " +
                  std::to_string(cache.LastRefilled()) + ", reused " +
                  std::to_string(cache.LastReused()) + ")");
        Check(editedWarm.points == editedCold.points &&
                  editedWarm.counts == editedCold.counts &&
                  editedWarm.frames == editedCold.frames &&
                  editedWarm.ids == editedCold.ids &&
                  editedWarm.tubeIds == editedCold.tubeIds &&
                  editedWarm.levels == editedCold.levels &&
                  editedWarm.regionIds == editedCold.regionIds,
              "G6: the partially cached refill is bit-exact against a full one");
        Check(editedWarm.points != cold.points,
              "G6: and it really carries the edit");
        std::printf("info: G6 one-tube version: %.1f ms before, %.1f ms after "
                    "(%.1fx)\n",
                    editedColdMs, editedWarmMs,
                    editedWarmMs > 0.0 ? editedColdMs / editedWarmMs : 0.0);
        Check(editedWarmMs < editedColdMs * 0.5,
              "G6: the dirty set at least halves the refill of a one-tube "
              "version");

        // What the worker actually pays: the whole layer build, which is
        // the guide refill plus the Sdf spec authoring the dirty set does
        // not touch. This is the TN-4-adjacent number (TN-4 itself times
        // the main-thread swap, which G6 does not change).
        {
            usdGenPomade::PomadeGuideCache buildCache;
            usdGenPomade::PomadeSnapshot warmTarget = big;
            SdfLayerRefPtr layer;
            std::string buildErr;
            Check(usdGenPomade::PomadeBuildCommitLayer(warmTarget, paths,
                                                     &layer, &buildErr,
                                                     &buildCache),
                  "G6: the reference layer builds through a cold cache: " +
                      buildErr);
            usdGenPomade::PomadeSnapshot oneTube = warmTarget;
            oneTube.version = 911;
            oneTube.tubes[77].tube.centerZ[4] += 0.125f;
            usdGenPomade::PomadeHashSnapshot(&oneTube);
            SdfLayerRefPtr coldLayer, warmLayer;
            auto const c0 = std::chrono::steady_clock::now();
            Check(usdGenPomade::PomadeBuildCommitLayer(oneTube, paths,
                                                     &coldLayer, &buildErr),
                  "G6: the one-tube version builds with no cache");
            auto const c1 = std::chrono::steady_clock::now();
            Check(usdGenPomade::PomadeBuildCommitLayer(oneTube, paths,
                                                     &warmLayer, &buildErr,
                                                     &buildCache),
                  "G6: and again through the cache");
            auto const c2 = std::chrono::steady_clock::now();
            double const coldBuild =
                std::chrono::duration<double, std::milli>(c1 - c0).count();
            double const warmBuild =
                std::chrono::duration<double, std::milli>(c2 - c1).count();
            std::printf("info: G6 reference layer build, one-tube version: "
                        "%.1f ms before, %.1f ms after\n",
                        coldBuild, warmBuild);
            Check(warmBuild < coldBuild,
                  "G6: the dirty set makes the whole worker build cheaper");
        }

        // Eviction: a tube the new version dropped must not survive, or the
        // cache would grow for the life of the session.
        usdGenPomade::PomadeSnapshot shrunk = edited;
        shrunk.tubes.resize(100);
        usdGenPomade::PomadeSnapshotGuides shrunkGuides;
        (void)timeGuides(shrunk, &cache, &shrunkGuides);
        Check(cache.Size() == 100,
              "G6: tubes the version dropped are evicted (size " +
                  std::to_string(cache.Size()) + ")");

        // A scalp change is a shared input: every slice has to go at once.
        usdGenPomade::PomadeSnapshot rescalped = shrunk;
        rescalped.hasScalp = true;
        rescalped.scalp.points.assign(12, 1.0f);
        usdGenPomade::PomadeSnapshotGuides afterScalp;
        (void)timeGuides(rescalped, &cache, &afterScalp);
        Check(cache.LastRefilled() == 100 && cache.LastReused() == 0,
              "G6: a changed scalp invalidates every cached slice");
    }

    // -- V6: two L1 roots commit and hydrate bit-exactly (plan/18 §7 G14) --
    //
    // plan/17 §5.1 gives every closed region its own tube stub. Until V6 the
    // model held ONE L1 tube, so a second region was silently unrepresentable
    // and the committer serialised a single root. This case proves the whole
    // chain: two regions -> two L1 tubes with distinct channel-0
    // (interpolation) ids -> both roots on the stage under Tubes -> guides
    // carrying both region ids -> hydrate rebuilding both bit-exactly.
    {
        int const n = 4;
        Grid const grid = MakeGrid(n);
        PomadeModel rm;
        Check(rm.BindScalp(grid.points, grid.counts, grid.indices),
              "V6: the two-region model binds its scalp");
        std::shared_ptr<usdGenPomade::PomadeScalpMesh const> const mesh =
            rm.GetScalp();
        Check(bool(mesh), "V6: the scalp mesh is readable");
        if (mesh) {
            BuildTwoRegions(&rm, *mesh, n);
        }
        Check(rm.Rasterise(), "V6: K3 rasterises the two-region graph");
        Check(rm.GetRegionLoops().regionIds.size() == 2,
              "V6: the graph extracts exactly two regions (got " +
                  std::to_string(rm.GetRegionLoops().regionIds.size()) + ")");
        // One stub per region, exactly as PomadeSession.ensureRegionTubes
        // does it after a graph gesture.
        Check(rm.BuildTubeFromRegion(0, 5, 0, 2.0f),
              "V6: region 0 gets its L1 stub: " + std::string(rm.GetDiagnostic()));
        Check(rm.BuildTubeFromRegion(1, 5, 0, 2.0f),
              "V6: region 1 gets its own L1 stub: " + std::string(rm.GetDiagnostic()));
        std::vector<int> const roots = rm.L1TubeIds();
        Check(roots.size() == 2 && roots[0] == 0,
              "V6: the model holds two L1 roots, the first being tube 0");
        int const secondRoot = roots.size() == 2 ? roots[1] : -1;
        Check(secondRoot > 0 && secondRoot % 16 == 0,
              "V6: the second root is minted in the L1 id family (got " +
                  std::to_string(secondRoot) + ")");
        Check(rm.TubeForRegion(0) != rm.TubeForRegion(1) &&
                  rm.TubeForRegion(0) >= 0 && rm.TubeForRegion(1) >= 0,
              "V6: the region->tube map is one tube per region");
        // Each root carries its own region, which IS its channel-0 id: the
        // bake writes the interpolation id of the region the root sits in.
        usdGenPomade::PomadeTubeDesc d0, d1;
        Check(rm.GetTubeDesc(0, &d0) && rm.GetTubeDesc(secondRoot, &d1) &&
                  d0.regionId >= 0 && d1.regionId >= 0 &&
                  d0.regionId != d1.regionId,
              "V6: the two roots carry distinct channel-0 region ids");
        // The second root is editable in its own right (G2 + G14 together).
        // Re-read both descs afterwards: they are the bytes the commit ->
        // hydrate round trip below has to reproduce.
        Check(rm.MoveTubeCenterCV(secondRoot, 2, 0.07f, 0.0f, -0.02f),
              "V6: the second root's center CV moves");
        Check(rm.GetTubeDesc(0, &d0) && rm.GetTubeDesc(secondRoot, &d1),
              "V6: both roots' post-edit shapes are readable");

        SdfLayerRefPtr liveR = SdfLayer::CreateAnonymous("pomade-live-2region");
        UsdStageRefPtr sr = MakeStage(liveR);
        // The scalp link is what carries the graph across a hydrate: the
        // committer targets it from the groom and hydrate rebinds the mesh
        // behind it before it replays the nodes' (faceId, uv).
        SdfPath const scalpPath("/Scalp");
        {
            UsdGeomMesh scalp = UsdGeomMesh::Define(sr, scalpPath);
            VtVec3fArray pts(grid.points.size() / 3);
            for (size_t i = 0; i < pts.size(); ++i) {
                pts[i] = GfVec3f(grid.points[i * 3 + 0],
                                 grid.points[i * 3 + 1],
                                 grid.points[i * 3 + 2]);
            }
            scalp.GetPointsAttr().Set(pts);
            scalp.GetFaceVertexCountsAttr().Set(
                VtIntArray(grid.counts.begin(), grid.counts.end()));
            scalp.GetFaceVertexIndicesAttr().Set(
                VtIntArray(grid.indices.begin(), grid.indices.end()));
        }
        PomadeCommitPaths regionPaths = paths;
        regionPaths.scalpPath = scalpPath;
        PomadeCommitter cr(&rm, regionPaths);
        cr.Enqueue(sr);
        Check(WaitCommitted(cr, liveR, rm.GetVersion()),
              "V6: the two-region groom commits");

        UsdPrim const root0 = sr->GetPrimAtPath(
            SdfPath("/PomadeGroom/Tubes/tube0"));
        UsdPrim const root1 = sr->GetPrimAtPath(SdfPath(
            "/PomadeGroom/Tubes/" +
            usdGenPomade::PomadeCommitPaths::TubeName(secondRoot)));
        Check(bool(root0) && bool(root1),
              "V6: both L1 roots land directly under Tubes");
        if (root0 && root1) {
            int r0 = -1, r1 = -1, l0 = 0, l1 = 0;
            root0.GetAttribute(TfToken("usdGen:pomade:regionId")).Get(&r0);
            root1.GetAttribute(TfToken("usdGen:pomade:regionId")).Get(&r1);
            root0.GetAttribute(TfToken("usdGen:pomade:level")).Get(&l0);
            root1.GetAttribute(TfToken("usdGen:pomade:level")).Get(&l1);
            Check(l0 == 1 && l1 == 1, "V6: both commit as level 1");
            Check(r0 >= 0 && r1 >= 0 && r0 != r1,
                  "V6: their committed channel-0 ids differ (" +
                      std::to_string(r0) + " vs " + std::to_string(r1) + ")");
        }
        {
            UsdPrim const guides =
                sr->GetPrimAtPath(SdfPath("/PomadeGroom/Guides"));
            VtIntArray committedRegions, committedTubes;
            Check(bool(guides) &&
                      guides.GetAttribute(TfToken("primvars:regionId"))
                          .Get(&committedRegions) &&
                      guides.GetAttribute(TfToken("primvars:tubeId"))
                          .Get(&committedTubes),
                  "V6: the committed guides carry regionId and tubeId");
            bool sawRoot0 = false, sawRoot1 = false;
            for (int id : committedTubes) {
                sawRoot0 = sawRoot0 || id == 0;
                sawRoot1 = sawRoot1 || id == secondRoot;
            }
            Check(sawRoot0 && sawRoot1,
                  "V6: both roots contribute guides to the one Guides prim");
        }

        PomadeModel hydrated;
        usdGenPomade::PomadeHydrateResult const hr =
            usdGenPomade::PomadeHydrateModel(sr, SdfPath("/PomadeGroom"),
                                           &hydrated);
        Check(hr.ok, "V6: the two-region groom hydrates: " + hr.diagnostic);
        Check(hr.guidesBitEqual,
              "V6: the regenerated guides are bit-equal to the committed set");
        Check(hr.graphRoundTrip, "V6: the graph round-trips");
        Check(hr.tubeCount == 2 && hr.importedTubeCount == 0,
              "V6: hydrate rebuilds exactly the two roots (got " +
                  std::to_string(hr.tubeCount) + ")");
        std::vector<int> const hydratedRoots = hydrated.L1TubeIds();
        Check(hydratedRoots == roots,
              "V6: the hydrated model re-mints the same L1 ids");
        usdGenPomade::PomadeTubeDesc h0, h1;
        bool const gotBoth = hydrated.GetTubeDesc(0, &h0) &&
                             hydrated.GetTubeDesc(secondRoot, &h1);
        Check(gotBoth && SameDesc(h0, d0) && SameDesc(h1, d1),
              "V6: both roots' authored shapes round-trip bit-exactly");
        Check(hydrated.TubeForRegion(d0.regionId) == 0 &&
                  hydrated.TubeForRegion(d1.regionId) == secondRoot,
              "V6: the region->tube map comes back with them");

        // The first shared vertex belongs to both original regions. After a
        // saved/hydrated groom, moving it must replace each affected L1 base
        // with the new region-CV footprint; a rigidly moved old outline is
        // specifically insufficient. A disconnected triangle supplies both
        // an unrelated root and the auto three-column control case.
        std::shared_ptr<usdGenPomade::PomadeScalpMesh const> const hydratedScalp =
            hydrated.GetScalp();
        std::vector<int> controlNodes;
        int const isolatedRegion = hydratedScalp
            ? AddIsolatedRegion(&hydrated, *hydratedScalp, n, &controlNodes) : -1;
        int const isolatedRoot = isolatedRegion >= 0
            ? hydrated.TubeForRegion(isolatedRegion) : -1;
        bool const builtIsolated = isolatedRoot < 0 && isolatedRegion >= 0 &&
            hydrated.BuildTubeFromRegion(isolatedRegion, 5, 0, 2.0f);
        int const controlRoot = builtIsolated
            ? hydrated.TubeForRegion(isolatedRegion) : isolatedRoot;
        std::vector<int> children;
        bool const builtChild = hydrated.SubdivideTube(0, 2, "kmeans", 41,
                                                        &children) &&
            children.size() == 2 &&
            // A persisted artist offset on the old base must not become the
            // next graph attachment footprint.
            hydrated.MoveTubeSectionCV(0, 0, 0, 0.075f, -0.030f) &&
            hydrated.MoveTubeSectionCV(children[0], 2, 1, 0.045f, -0.025f);
        TubeWorldGeometry root0Before, root1Before, childBefore, controlBefore;
        PomadeModel::TubeRecord childRecordBefore, controlRecordBefore;
        bool const capturedBefore = builtChild && controlRoot >= 0 &&
            CaptureTubeWorld(hydrated, 0, &root0Before) &&
            CaptureTubeWorld(hydrated, secondRoot, &root1Before) &&
            CaptureTubeWorld(hydrated, children[0], &childBefore) &&
            CaptureTubeWorld(hydrated, controlRoot, &controlBefore) &&
            hydrated.GetTubeRecord(children[0], &childRecordBefore) &&
            hydrated.GetTubeRecord(controlRoot, &controlRecordBefore);
        bool const baseWasOffset = capturedBefore &&
            !RootBaseMatchesRegion(hydrated, root0Before.desc) &&
            RootBaseMatchesRegion(hydrated, root1Before.desc) &&
            RootBaseMatchesRegion(hydrated, controlBefore.desc) &&
            controlBefore.desc.ringVerts == 3;
        usdGenPomade::PomadeGraphNode const *sharedBeforeMove =
            hydrated.GetGraph().FindNode(1);
        TubeWorldGeometry noOpRoot;
        bool const noOpPreservesLegacyBase = baseWasOffset && sharedBeforeMove &&
            hydrated.GraphMoveNode(1, Locate(*hydratedScalp, n,
                                              sharedBeforeMove->p[0],
                                              sharedBeforeMove->p[2])) &&
            CaptureTubeWorld(hydrated, 0, &noOpRoot) &&
            SameDesc(root0Before.desc, noOpRoot.desc) &&
            noOpRoot.positions == root0Before.positions &&
            !RootBaseMatchesRegion(hydrated, noOpRoot.desc);
        hydrated.ClearUndo();
        bool const movedShared = noOpPreservesLegacyBase && hydratedScalp &&
            hydrated.GraphMoveNode(1, Locate(*hydratedScalp, n, 2.25f, 1.0f));
        TubeWorldGeometry root0After, root1After, childAfter, controlAfter;
        PomadeModel::TubeRecord childRecordAfter, controlRecordAfter;
        bool const capturedAfter = movedShared &&
            CaptureTubeWorld(hydrated, 0, &root0After) &&
            CaptureTubeWorld(hydrated, secondRoot, &root1After) &&
            CaptureTubeWorld(hydrated, children[0], &childAfter) &&
            CaptureTubeWorld(hydrated, controlRoot, &controlAfter) &&
            hydrated.GetTubeRecord(children[0], &childRecordAfter) &&
            hydrated.GetTubeRecord(controlRoot, &controlRecordAfter);
        Check(capturedAfter && noOpPreservesLegacyBase && MovedWorld(root0Before, root0After) &&
                  MovedWorld(root1Before, root1After) &&
                  MovedWorld(childBefore, childAfter) &&
                  MovedBaseRing(root0Before, root0After) &&
                  MovedBaseRing(root1Before, root1After) &&
                  RootBaseMatchesRegion(hydrated, root0After.desc) &&
                  RootBaseMatchesRegion(hydrated, root1After.desc) &&
                  SameSectionsAboveBase(root0Before.desc, root0After.desc) &&
                  SameSectionsAboveBase(root1Before.desc, root1After.desc) &&
                  ChildRootRefreshesButUpperSculptRemains(childRecordAfter) &&
                  SameDesc(controlRecordBefore.actual, controlRecordAfter.actual) &&
                  controlBefore.positions == controlAfter.positions,
              "V6: no-op leaves legacy base stale; asymmetric shared-CV move refits it");

        // One graph move owns one undo step. Redo must reproduce the exact
        // transported K5 walls and stored child residual, while a cancelled
        // second move restores that same committed attachment pose.
        TubeWorldGeometry undoRoot0, undoRoot1, undoChild, undoControl;
        PomadeModel::TubeRecord undoChildRecord;
        uint32_t undoDirty = 0, redoDirty = 0, cancelDirty = 0;
        bool const undone = capturedAfter && hydrated.Undo(&undoDirty) &&
            CaptureTubeWorld(hydrated, 0, &undoRoot0) &&
            CaptureTubeWorld(hydrated, secondRoot, &undoRoot1) &&
            CaptureTubeWorld(hydrated, children[0], &undoChild) &&
            CaptureTubeWorld(hydrated, controlRoot, &undoControl) &&
            hydrated.GetTubeRecord(children[0], &undoChildRecord);
        Check(undone && SameDesc(root0Before.desc, undoRoot0.desc) &&
                  SameDesc(root1Before.desc, undoRoot1.desc) &&
                  SameDesc(childBefore.desc, undoChild.desc) &&
                  SameDesc(controlBefore.desc, undoControl.desc) &&
                  undoRoot0.positions == root0Before.positions &&
                  undoRoot1.positions == root1Before.positions &&
                  undoChild.positions == childBefore.positions &&
                  undoControl.positions == controlBefore.positions &&
                  SameDeltas(undoChildRecord.deltas, childRecordBefore.deltas) &&
                  undoDirty != usdGenPomade::PomadeDirty_Clean,
              "V6: undo restores both hydrated attachments, child sculpt and control root exactly");
        TubeWorldGeometry redoRoot0, redoRoot1, redoChild, redoControl;
        PomadeModel::TubeRecord redoChildRecord;
        bool const redone = undone && hydrated.Redo(&redoDirty) &&
            CaptureTubeWorld(hydrated, 0, &redoRoot0) &&
            CaptureTubeWorld(hydrated, secondRoot, &redoRoot1) &&
            CaptureTubeWorld(hydrated, children[0], &redoChild) &&
            CaptureTubeWorld(hydrated, controlRoot, &redoControl) &&
            hydrated.GetTubeRecord(children[0], &redoChildRecord);
        Check(redone && SameDesc(root0After.desc, redoRoot0.desc) &&
                  SameDesc(root1After.desc, redoRoot1.desc) &&
                  SameDesc(childAfter.desc, redoChild.desc) &&
                  SameDesc(controlAfter.desc, redoControl.desc) &&
                  redoRoot0.positions == root0After.positions &&
                  redoRoot1.positions == root1After.positions &&
                  redoChild.positions == childAfter.positions &&
                  redoControl.positions == controlAfter.positions &&
                  SameDeltas(redoChildRecord.deltas, childRecordAfter.deltas) &&
                  redoDirty != usdGenPomade::PomadeDirty_Clean,
              "V6: redo restores the exact transported hydrated subtree");
        TubeWorldGeometry cancelRoot0, cancelRoot1, cancelChild, cancelControl;
        bool const cancelled = redone && hydrated.BeginGesture("legacy place") &&
            hydrated.GraphMoveNode(1, Locate(*hydratedScalp, n, 2.35f, 1.0f)) &&
            hydrated.CancelGesture(&cancelDirty) &&
            CaptureTubeWorld(hydrated, 0, &cancelRoot0) &&
            CaptureTubeWorld(hydrated, secondRoot, &cancelRoot1) &&
            CaptureTubeWorld(hydrated, children[0], &cancelChild) &&
            CaptureTubeWorld(hydrated, controlRoot, &cancelControl);
        Check(cancelled && cancelRoot0.positions == root0After.positions &&
                  cancelRoot1.positions == root1After.positions &&
                  cancelChild.positions == childAfter.positions &&
                  cancelControl.positions == controlAfter.positions &&
                  cancelDirty != usdGenPomade::PomadeDirty_Clean,
              "V6: cancelled attached-node move restores the transported subtree exactly");

        // Place changes graph topology first. Its newly split point has no
        // compatible pre-split attachment baseline, so following it during
        // the same gesture remains graph-only rather than trying to transport
        // the established tube owners through a mismatched support chart.
        int placeEdge = -1;
        for (usdGenPomade::PomadeGraphEdge const &edge : hydrated.GetGraph().Edges()) {
            if (edge.alive) { placeEdge = edge.id; break; }
        }
        TubeWorldGeometry placeRoot0, placeRoot1, placeChild, placeControl;
        uint32_t placeUndoDirty = 0;
        int const placedNode = redone && placeEdge >= 0 &&
            hydrated.BeginGesture("place split")
            ? hydrated.GraphSplitEdge(placeEdge, Locate(*hydratedScalp, n, 1.0f, 1.0f))
            : -1;
        bool const placed = placedNode >= 0 &&
            hydrated.GraphMoveNode(placedNode, Locate(*hydratedScalp, n, 1.20f, 1.0f)) &&
            hydrated.EndGesture() &&
            CaptureTubeWorld(hydrated, 0, &placeRoot0) &&
            CaptureTubeWorld(hydrated, secondRoot, &placeRoot1) &&
            CaptureTubeWorld(hydrated, children[0], &placeChild) &&
            CaptureTubeWorld(hydrated, controlRoot, &placeControl);
        Check(placed && placeRoot0.positions == root0After.positions &&
                  placeRoot1.positions == root1After.positions &&
                  placeChild.positions == childAfter.positions &&
                  placeControl.positions == controlAfter.positions &&
                  hydrated.Undo(&placeUndoDirty) &&
                  placeUndoDirty != usdGenPomade::PomadeDirty_Clean,
              "V6: legacy split/place point remains graph-only and commits one cancellable gesture");

        // Moving the two shared endpoints reshapes the common edge, rather
        // than merely translating it. Both affected roots must refit their
        // section-zero corners, while the disconnected triangle stays exact.
        std::vector<int> const sharedEdgeNodes = {1, 2};
        std::vector<usdGenPomade::PomadeHit> const reshapedEdge = {
            Locate(*hydratedScalp, n, 2.32f, 1.0f),
            Locate(*hydratedScalp, n, 2.46f, 3.0f)};
        bool const movedEdge = placed &&
            hydrated.GraphMoveNodes(sharedEdgeNodes, reshapedEdge);
        TubeWorldGeometry edgeRoot0, edgeRoot1, edgeChild, edgeControl;
        PomadeModel::TubeRecord edgeChildRecord, edgeControlRecord;
        bool const capturedEdge = movedEdge &&
            CaptureTubeWorld(hydrated, 0, &edgeRoot0) &&
            CaptureTubeWorld(hydrated, secondRoot, &edgeRoot1) &&
            CaptureTubeWorld(hydrated, children[0], &edgeChild) &&
            CaptureTubeWorld(hydrated, controlRoot, &edgeControl) &&
            hydrated.GetTubeRecord(children[0], &edgeChildRecord) &&
            hydrated.GetTubeRecord(controlRoot, &edgeControlRecord);
        Check(capturedEdge && RootBaseMatchesRegion(hydrated, edgeRoot0.desc) &&
                  RootBaseMatchesRegion(hydrated, edgeRoot1.desc) &&
                  MovedBaseRing(root0After, edgeRoot0) &&
                  MovedBaseRing(root1After, edgeRoot1) &&
                  SameSectionsAboveBase(root0After.desc, edgeRoot0.desc) &&
                  SameSectionsAboveBase(root1After.desc, edgeRoot1.desc) &&
                  ChildRootRefreshesButUpperSculptRemains(edgeChildRecord) &&
                  SameDesc(controlRecordBefore.actual, edgeControlRecord.actual) &&
                  edgeControl.positions == controlBefore.positions,
              "V6: shared-edge reshape refits both region-root bases without moving the other root");

        // A true whole-region translation is the fast path: there is no
        // footprint residual to repair, so the auto triangle keeps an exact
        // rigid translation through all of its K5 wall samples.
        std::vector<usdGenPomade::PomadeHit> translatedNodes;
        bool controlTargets = controlNodes.size() == 3;
        for (int nodeId : controlNodes) {
            usdGenPomade::PomadeGraphNode const *node = hydrated.GetGraph().FindNode(nodeId);
            if (!node) { controlTargets = false; break; }
            translatedNodes.push_back(
                Locate(*hydratedScalp, n, node->p[0] + 0.12f, node->p[2] + 0.08f));
        }
        bool const translated = capturedEdge && controlTargets &&
            hydrated.GraphMoveNodes(controlNodes, translatedNodes);
        TubeWorldGeometry translatedControl;
        bool const capturedTranslation = translated &&
            CaptureTubeWorld(hydrated, controlRoot, &translatedControl);
        Check(capturedTranslation && RootBaseMatchesRegion(hydrated, translatedControl.desc) &&
                  SameTranslation(edgeControl, translatedControl, 0.12f, 0.0f, 0.08f),
              "V6: pure region translation preserves exact rigid triangle geometry");

        // Persist the reshaped roots and translated control, then hydrate a
        // fresh model. Region ownership and the corrected base footprints
        // must survive a real commit instead of only the live model state.
        PomadeCommitter persistedCommit(&hydrated, regionPaths);
        persistedCommit.Enqueue(sr);
        bool const persisted = capturedTranslation &&
            WaitCommitted(persistedCommit, liveR, hydrated.GetVersion());
        PomadeModel persistedModel;
        usdGenPomade::PomadeHydrateResult const persistedHydrate = persisted
            ? usdGenPomade::PomadeHydrateModel(sr, SdfPath("/PomadeGroom"),
                                             &persistedModel)
            : usdGenPomade::PomadeHydrateResult{};
        TubeWorldGeometry persistedRoot0, persistedRoot1, persistedChild,
                          persistedControl;
        PomadeModel::TubeRecord persistedChildRecord;
        bool const capturedPersisted = persistedHydrate.ok &&
            CaptureTubeWorld(persistedModel, 0, &persistedRoot0) &&
            CaptureTubeWorld(persistedModel, secondRoot, &persistedRoot1) &&
            CaptureTubeWorld(persistedModel, children[0], &persistedChild) &&
            CaptureTubeWorld(persistedModel, controlRoot, &persistedControl) &&
            persistedModel.GetTubeRecord(children[0], &persistedChildRecord);
        Check(capturedPersisted &&
                  SameDesc(edgeRoot0.desc, persistedRoot0.desc) &&
                  SameDesc(edgeRoot1.desc, persistedRoot1.desc) &&
                  SameDesc(edgeChild.desc, persistedChild.desc) &&
                  SameDesc(translatedControl.desc, persistedControl.desc) &&
                  SameDeltas(edgeChildRecord.deltas, persistedChildRecord.deltas) &&
                  RootBaseMatchesRegion(persistedModel, persistedRoot0.desc) &&
                  RootBaseMatchesRegion(persistedModel, persistedRoot1.desc) &&
                  RootBaseMatchesRegion(persistedModel, persistedControl.desc),
              "V6: save/hydrate preserves reshaped bases, child sculpt and root ownership");

        // Replacing a child's attachment root during the region refit must
        // also rebuild its section-zero holding triples.  A later direct
        // child-root edit must therefore drive the matching installed parent
        // boundary slot, rather than the stale pre-refit parent slot.
        usdGenPomade::PomadeParentBoundaryBinding rootHolding;
        bool haveRootHolding = false;
        if (capturedPersisted) {
            for (usdGenPomade::PomadeParentBoundaryBinding const &binding :
                 persistedChildRecord.actual.inheritedBoundaryBindings) {
                if (binding.section == 0 && binding.parentSlot >= 0 &&
                    binding.childSlot >= 0 &&
                    binding.parentSlot < persistedRoot0.desc.ringVerts &&
                    binding.childSlot < persistedChild.desc.ringVerts) {
                    rootHolding = binding;
                    haveRootHolding = true;
                    break;
                }
            }
        }
        std::vector<std::array<float, 3>> holdingParentBefore, holdingChildBefore,
                                         holdingParentAfter, holdingChildAfter;
        usdGenPomade::PomadeTubeDesc holdingParentDesc, holdingChildDesc,
                                   holdingOtherRoot;
        bool const childRootEdit = haveRootHolding &&
            RootBaseWorld(persistedRoot0.desc, &holdingParentBefore) &&
            RootBaseWorld(persistedChild.desc, &holdingChildBefore) &&
            persistedModel.MoveTubeSectionCV(children[0], 0,
                                              rootHolding.childSlot,
                                              0.032f, -0.019f) &&
            persistedModel.GetTubeDesc(0, &holdingParentDesc) &&
            persistedModel.GetTubeDesc(children[0], &holdingChildDesc) &&
            persistedModel.GetTubeDesc(secondRoot, &holdingOtherRoot) &&
            RootBaseWorld(holdingParentDesc, &holdingParentAfter) &&
            RootBaseWorld(holdingChildDesc, &holdingChildAfter) &&
            holdingParentBefore.size() == holdingParentAfter.size() &&
            holdingChildBefore.size() == holdingChildAfter.size();
        bool holdingAligned = childRootEdit;
        if (holdingAligned) {
            std::array<float, 3> const &parentPoint =
                holdingParentAfter[size_t(rootHolding.parentSlot)];
            std::array<float, 3> const &childPoint =
                holdingChildAfter[size_t(rootHolding.childSlot)];
            float const dx = parentPoint[0] - childPoint[0];
            float const dy = parentPoint[1] - childPoint[1];
            float const dz = parentPoint[2] - childPoint[2];
            holdingAligned = dx * dx + dy * dy + dz * dz <= 4e-8f;
            float const px = parentPoint[0] -
                holdingParentBefore[size_t(rootHolding.parentSlot)][0];
            float const py = parentPoint[1] -
                holdingParentBefore[size_t(rootHolding.parentSlot)][1];
            float const pz = parentPoint[2] -
                holdingParentBefore[size_t(rootHolding.parentSlot)][2];
            holdingAligned = holdingAligned && px * px + py * py + pz * pz > 1e-8f;
        }
        Check(holdingAligned && SameDesc(persistedRoot1.desc, holdingOtherRoot),
              "V6: post-refit child root edit drives its rebuilt parent holding edge");

        // A stale authored whole-tube translation can put its complete cage
        // above and beside the graph support. Reposition must repair the
        // normal offset as one subtree transport and solve only section zero
        // to the edited region; it must not drag CV0 to the graph centroid
        // and bend the upper wall. The earlier disconnected control remains
        // the independent-root check for this same conformance path.
        PomadeModel staleWhole;
        std::shared_ptr<usdGenPomade::PomadeScalpMesh const> const staleScalp =
            staleWhole.BindScalp(grid.points, grid.counts, grid.indices)
                ? staleWhole.GetScalp() : nullptr;
        std::vector<int> staleNodes;
        int const staleRegion = staleScalp
            ? AddIsolatedRegion(&staleWhole, *staleScalp, n, &staleNodes) : -1;
        bool const builtStale = staleRegion >= 0 &&
            staleWhole.BuildTubeFromRegion(staleRegion, 5, 0, 2.0f);
        int const staleRoot = builtStale ? staleWhole.TubeForRegion(staleRegion) : -1;
        bool const translatedStale = staleRoot >= 0 &&
            staleWhole.TranslateTube(staleRoot, 0.14f, 0.08f, -0.06f);
        TubeWorldGeometry staleBefore, staleAfter;
        bool const capturedStale = translatedStale &&
            CaptureTubeWorld(staleWhole, staleRoot, &staleBefore) &&
            !RootBaseMatchesRegion(staleWhole, staleBefore.desc);
        bool const refitStale = capturedStale && staleNodes.size() == 3 &&
            staleWhole.GraphMoveNode(staleNodes[2],
                                     Locate(*staleScalp, n, 0.92f, 1.55f)) &&
            CaptureTubeWorld(staleWhole, staleRoot, &staleAfter);
        Check(refitStale && RootBaseMatchesRegion(staleWhole, staleAfter.desc) &&
                  MovedBaseRing(staleBefore, staleAfter) &&
                  SameCenterCageShape(staleBefore.desc, staleAfter.desc) &&
                  UpperSectionsShareTranslation(staleBefore.desc, staleAfter.desc),
              "V6: stale whole-tube offset refits only the base and keeps its upper cage rigid");

        // Explicit counts retain every corner and keep their existing extra
        // controls on the same material edge/parameter.  These two reshapes
        // deliberately change the triangle's longest edge, so a fresh
        // longest-edge allocator would fail even if it still enclosed the
        // new region.
        PomadeModel explicitSlots;
        std::shared_ptr<usdGenPomade::PomadeScalpMesh const> const slotScalp =
            explicitSlots.BindScalp(grid.points, grid.counts, grid.indices)
                ? explicitSlots.GetScalp() : nullptr;
        std::vector<int> slotNodes;
        int const slotRegion = slotScalp
            ? AddIsolatedRegion(&explicitSlots, *slotScalp, n, &slotNodes) : -1;
        bool const builtSlots = slotRegion >= 0 &&
            explicitSlots.BuildTubeFromRegion(slotRegion, 5, 6, 2.0f);
        int const slotRoot = builtSlots ? explicitSlots.TubeForRegion(slotRegion) : -1;
        usdGenPomade::PomadeTubeDesc slotBefore, slotAfterFirst, slotAfterSecond;
        std::vector<RootEdgeBinding> materialSlots;
        bool const boundSlots = slotRoot >= 0 &&
            explicitSlots.GetTubeDesc(slotRoot, &slotBefore) &&
            slotBefore.ringVerts == 6 &&
            BindRootSlotsToRegionEdges(explicitSlots, slotBefore, &materialSlots) &&
            materialSlots.size() == 6;
        bool const firstSlotReshape = boundSlots && slotNodes.size() == 3 &&
            explicitSlots.GraphMoveNode(slotNodes[2],
                                        Locate(*slotScalp, n, 0.70f, 1.75f)) &&
            explicitSlots.GetTubeDesc(slotRoot, &slotAfterFirst);
        // Evaluate against the same graph state that authored this root.
        // The next reshape changes the material edges, so postponing this
        // check would incorrectly compare the first descriptor to a future
        // polygon.
        bool const firstSlotsFollow = firstSlotReshape &&
            RootSlotsFollowMaterialEdges(explicitSlots, slotAfterFirst,
                                         materialSlots);
        if (boundSlots && firstSlotReshape && !firstSlotsFollow) {
            PrintRootSlotDiagnostics(explicitSlots, slotAfterFirst,
                                     materialSlots, "after first reshape");
        }
        bool const secondSlotReshape = firstSlotReshape &&
            explicitSlots.GraphMoveNode(slotNodes[1],
                                        Locate(*slotScalp, n, 1.70f, 0.15f)) &&
            explicitSlots.GetTubeDesc(slotRoot, &slotAfterSecond);
        bool const secondSlotsFollow = secondSlotReshape &&
            RootSlotsFollowMaterialEdges(explicitSlots, slotAfterSecond,
                                         materialSlots);
        if (!(builtSlots && boundSlots && firstSlotReshape && secondSlotReshape &&
              firstSlotsFollow && secondSlotsFollow)) {
            std::printf("V6 explicit slots state: built=%d bound=%d first=%d "
                        "second=%d firstFollow=%d secondFollow=%d diag=%s\n",
                        int(builtSlots), int(boundSlots), int(firstSlotReshape),
                        int(secondSlotReshape), int(firstSlotsFollow),
                        int(secondSlotsFollow), explicitSlots.GetDiagnostic());
            if (boundSlots && secondSlotReshape) {
                PrintRootSlotDiagnostics(explicitSlots, slotAfterSecond,
                                         materialSlots, "after second reshape");
            }
        }
        Check(builtSlots && boundSlots && firstSlotReshape && secondSlotReshape &&
                  firstSlotsFollow && secondSlotsFollow,
              "V6: explicit extra base slots retain old material edges through two reshapes");
    }

    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
