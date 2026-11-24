// usdGenPomade — PomadeModel implementation (plan/17 §2.1): the tube, the
// scalp graph, the hierarchy store, the guide fill, picking and undo.
#include "usdGenPomade/pomadeModel.h"

#include "usdGenPomade/pomadeCheck.h"
#include "usdGenPomade/pomadeModelState.h"
#include "usdGen/concaveMaterialRemap.h"

#ifdef USDGEN_POMADE_HAS_CUDA
#include "usdGenPomade/pomadeKernels.h"
#include "usdGen/gpu/deviceBuffer.h"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <set>

namespace usdGenPomade {

struct PomadeModel::_Device {
#ifdef USDGEN_POMADE_HAS_CUDA
    usdGen::gpu::DeviceBuffer<float> positions;  // 3 * vertexCount
    usdGen::gpu::DeviceBuffer<float> normals;    // 3 * vertexCount
    usdGen::gpu::DeviceBuffer<float> centerX;
    usdGen::gpu::DeviceBuffer<float> centerY;
    usdGen::gpu::DeviceBuffer<float> centerZ;
    // P3 K4/K5 lane (sized on the sections path only). Frames ride a
    // float buffer (9 per frame): DeviceBuffer instantiates a fixed type
    // list in usdGenGpu, which cannot see PomadeFrame, so the launch
    // reinterprets the data pointer (see the static_assert below).
    usdGen::gpu::DeviceBuffer<float> frames;
    usdGen::gpu::DeviceBuffer<float> sectionT;
    usdGen::gpu::DeviceBuffer<float> sectionU;  // nSec * ringVerts
    usdGen::gpu::DeviceBuffer<float> sectionV;
    usdGen::gpu::DeviceBuffer<float> sectionScale;
    usdGen::gpu::DeviceBuffer<float> sectionTwist;
    usdGen::gpu::DeviceBuffer<float> rootSurfaceOffsets;
    usdGen::gpu::DeviceBuffer<float> ringT;
    // K11 production pick caches (TN-2): guide CVs uploaded once per
    // guide generation, reduction scratch sized per pick (same-size
    // reset is a no-op, so steady-state picks allocate nothing). Best/
    // scratch ride float buffers (4 per PomadeDevicePickBest: the fixed
    // DeviceBuffer type list cannot see that struct either).
    usdGen::gpu::DeviceBuffer<float> guideCVs;  // 3 * guides * cv
    uint64_t guideCVsVersion = 0;               // 0 = nothing uploaded
    usdGen::gpu::DeviceBuffer<float> pickBest;
    usdGen::gpu::DeviceBuffer<float> pickScratch;
    // K11b marquee lane (V1): one byte per candidate plus the lasso
    // polygon. Grow-only like the pick scratch, so a drag that marquees
    // on every sample allocates once.
    usdGen::gpu::DeviceBuffer<unsigned char> selectMask;
    usdGen::gpu::DeviceBuffer<float> selectPolygon;
    cudaStream_t stream = nullptr;
    bool streamOwned = false;
#endif
};

struct PomadeModel::_Published {
    usdGen::UsdGenAtomicRef<PomadeModelState const> cell;
};

PomadeModel::PomadeModel()
    : _device(new _Device())
    , _published(new _Published())
{
#ifdef USDGEN_POMADE_HAS_CUDA
    if (cudaStreamCreate(&_device->stream) == cudaSuccess) {
        _device->streamOwned = true;
    }
#endif
}

PomadeModel::~PomadeModel()
{
    _published.reset();  // retires the published state through the epochs
#ifdef USDGEN_POMADE_HAS_CUDA
    if (_device && _device->streamOwned && _device->stream) {
        cudaStreamDestroy(_device->stream);
        _device->stream = nullptr;
    }
#endif
}

namespace {

// Production lane choice (plan/17 section 4.1, plan/18 section 7 G11): the
// device lane when the CUDA mirror is healthy, the CPU twin otherwise and
// whenever the lane refuses (a tube over the device caps, an allocation
// failure). Only lanes proven BIT-exact against their twin are dispatched
// this way, so the answer never depends on which one ran; K8/K9/K10 agree
// only to a tolerance and stay on the twin (see pomadeKernels.h). The lanes
// take the legacy default stream, which a cudaStreamCreate stream
// synchronises with, so they stay ordered behind the model's K4/K5 work.
bool _LaneParentAverage(bool device,
                        std::vector<PomadeTubeDesc> const &children,
                        PomadeTubeDesc *out, std::string *err)
{
#ifdef USDGEN_POMADE_HAS_CUDA
    if (device) {
        std::vector<PomadeTubeDesc> one;
        std::string lerr;
        if (PomadeParentAverageDevice(&children, 1, &one, nullptr, &lerr) &&
            one.size() == 1) {
            *out = one[0];
            return true;
        }
    }
#else
    (void)device;
#endif
    return PomadeParentAverageCpu(children, out, err);
}

bool _LaneSmoothnessScore(bool device, float const *cx, float const *cy,
                          float const *cz, int nCv, float *out,
                          std::string *err)
{
#ifdef USDGEN_POMADE_HAS_CUDA
    if (device) {
        std::string lerr;
        if (PomadeSmoothnessScoreDevice(cx, cy, cz, nCv, out, nullptr,
                                       &lerr)) {
            return true;
        }
    }
#else
    (void)device;
#endif
    return PomadeSmoothnessScoreCpu(cx, cy, cz, nCv, out, err);
}

bool _LaneTubeIntersect(bool device, PomadeTubeDesc const *tubes,
                        int tubeCount, int *outFlags, std::string *err)
{
#ifdef USDGEN_POMADE_HAS_CUDA
    if (device) {
        std::string lerr;
        if (PomadeTubeIntersectDevice(tubes, tubeCount, outFlags, nullptr,
                                     &lerr)) {
            return true;
        }
    }
#else
    (void)device;
#endif
    return PomadeTubeIntersectCpu(tubes, tubeCount, outFlags, err);
}

struct _RegionSupport {
    float origin[3] = {0.0f, 0.0f, 0.0f};
    float normal[3] = {0.0f, 1.0f, 0.0f};
};

struct _RegionFootprint {
    float origin[3] = {0.0f, 0.0f, 0.0f};
    float normal[3] = {0.0f, 1.0f, 0.0f};
    float uAxis[3] = {1.0f, 0.0f, 0.0f};
    float vAxis[3] = {0.0f, 0.0f, 1.0f};
    std::vector<std::array<float, 3>> corners;
};

bool _RegionCanonicalFootprint(PomadeScalpMesh const &scalp,
                               PomadeScalpGraph const &graph, int regionId,
                               _RegionFootprint *out, std::string *err)
{
    auto fail = [&](char const *why) {
        if (err) {
            *err = why;
        }
        return false;
    };
    if (!out || regionId < 0 || regionId >= graph.RegionCount()) {
        return fail("invalid region footprint");
    }
    PomadeGraphRegion const &region = graph.Regions()[size_t(regionId)];
    size_t const boundaryCount = region.boundary.size() / 3;
    if (boundaryCount < 3 || region.loop.size() < 3) {
        return fail("region has no closed boundary");
    }
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (size_t i = 0; i < boundaryCount; ++i) {
        cx += region.boundary[i * 3 + 0];
        cy += region.boundary[i * 3 + 1];
        cz += region.boundary[i * 3 + 2];
    }
    float const centroid[3] = {float(cx / double(boundaryCount)),
                               float(cy / double(boundaryCount)),
                               float(cz / double(boundaryCount))};
    PomadeHit const support = PomadeClosestPointCpu(scalp, centroid);
    if (!support.hit) {
        return fail("region centroid is off scalp");
    }
    out->origin[0] = support.px;
    out->origin[1] = support.py;
    out->origin[2] = support.pz;
    float n[3] = {0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < boundaryCount; ++i) {
        float const *a = &region.boundary[i * 3];
        float const *b = &region.boundary[((i + 1) % boundaryCount) * 3];
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    float const nlen = PomadeLen3(n);
    if (!(nlen > 1e-12f)) {
        return fail("region support plane is degenerate");
    }
    n[0] /= nlen;
    n[1] /= nlen;
    n[2] /= nlen;
    if (n[0] * support.nx + n[1] * support.ny + n[2] * support.nz < 0.0f) {
        n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2];
    }
    std::vector<PomadeGraphNode const *> nodes;
    nodes.reserve(region.loop.size());
    for (int id : region.loop) {
        PomadeGraphNode const *node = graph.FindNode(id);
        if (!node || !node->alive) {
            return fail("region loop has a missing node");
        }
        nodes.push_back(node);
    }
    size_t anchor = 0;
    for (size_t i = 1; i < nodes.size(); ++i) {
        if (nodes[i]->id < nodes[anchor]->id) {
            anchor = i;
        }
    }
    std::rotate(nodes.begin(), nodes.begin() + anchor, nodes.end());
    bool haveU = false;
    for (PomadeGraphNode const *node : nodes) {
        float d[3] = {node->p[0] - out->origin[0],
                      node->p[1] - out->origin[1],
                      node->p[2] - out->origin[2]};
        float const along = PomadeDot3(d, n);
        d[0] -= along * n[0]; d[1] -= along * n[1]; d[2] -= along * n[2];
        float const dlen = PomadeLen3(d);
        if (dlen > 1e-6f) {
            out->uAxis[0] = d[0] / dlen;
            out->uAxis[1] = d[1] / dlen;
            out->uAxis[2] = d[2] / dlen;
            haveU = true;
            break;
        }
    }
    if (!haveU) {
        return fail("region authored corners collapse in support plane");
    }
    PomadeCross3(n, out->uAxis, out->vAxis);
    out->normal[0] = n[0]; out->normal[1] = n[1]; out->normal[2] = n[2];
    out->corners.clear();
    out->corners.reserve(nodes.size());
    double twiceArea = 0.0;
    std::vector<std::array<float, 2>> chart;
    chart.reserve(nodes.size());
    for (PomadeGraphNode const *node : nodes) {
        float const d[3] = {node->p[0] - out->origin[0],
                            node->p[1] - out->origin[1],
                            node->p[2] - out->origin[2]};
        float const u = PomadeDot3(d, out->uAxis);
        float const v = PomadeDot3(d, out->vAxis);
        chart.push_back({{u, v}});
        out->corners.push_back({{out->origin[0] + out->uAxis[0] * u + out->vAxis[0] * v,
                                 out->origin[1] + out->uAxis[1] * u + out->vAxis[1] * v,
                                 out->origin[2] + out->uAxis[2] * u + out->vAxis[2] * v}});
    }
    for (size_t i = 0; i < chart.size(); ++i) {
        std::array<float, 2> const &a = chart[i];
        std::array<float, 2> const &b = chart[(i + 1) % chart.size()];
        twiceArea += double(a[0]) * b[1] - double(a[1]) * b[0];
    }
    if (std::fabs(twiceArea) <= 1e-10) {
        return fail("region footprint has zero area");
    }
    if (twiceArea < 0.0) {
        std::reverse(out->corners.begin() + 1, out->corners.end());
    }
    return true;
}

bool _SameRegionFootprint(_RegionFootprint const &a,
                          _RegionFootprint const &b)
{
    if (a.corners.size() != b.corners.size()) return false;
    auto samePoint = [](float const *p, float const *q) {
        float const dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
        return dx * dx + dy * dy + dz * dz <= 1e-12f;
    };
    if (!samePoint(a.origin, b.origin)) return false;
    for (size_t i = 0; i < a.corners.size(); ++i) {
        if (!samePoint(a.corners[i].data(), b.corners[i].data())) return false;
    }
    return true;
}

std::array<float, 3> _FootprintPoint(_RegionFootprint const &footprint,
                                     float wrappedS)
{
    int const count = int(footprint.corners.size());
    while (wrappedS < 0.0f) wrappedS += float(count);
    while (wrappedS >= float(count)) wrappedS -= float(count);
    int const edge = int(std::floor(wrappedS));
    float const t = wrappedS - float(edge);
    std::array<float, 3> const &a = footprint.corners[size_t(edge)];
    std::array<float, 3> const &b =
        footprint.corners[size_t((edge + 1) % count)];
    return {{a[0] + (b[0] - a[0]) * t,
             a[1] + (b[1] - a[1]) * t,
             a[2] + (b[2] - a[2]) * t}};
}

std::vector<float> _TemplateFootprintSlots(_RegionFootprint const &footprint,
                                           int ringVerts)
{
    std::vector<float> slots;
    int const count = int(footprint.corners.size());
    if (ringVerts < count) {
        return slots;
    }
    for (int i = 0; i < count; ++i) slots.push_back(float(i));
    while (int(slots.size()) < ringVerts) {
        size_t best = 0;
        float bestLength2 = -1.0f;
        for (size_t i = 0; i < slots.size(); ++i) {
            std::array<float, 3> const a = _FootprintPoint(footprint, slots[i]);
            float next = slots[(i + 1) % slots.size()];
            if (i + 1 == slots.size()) next += float(count);
            std::array<float, 3> const b = _FootprintPoint(footprint, next);
            float const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
            float const length2 = dx * dx + dy * dy + dz * dz;
            if (length2 > bestLength2) { bestLength2 = length2; best = i; }
        }
        float next = slots[(best + 1) % slots.size()];
        if (best + 1 == slots.size()) next += float(count);
        slots.insert(slots.begin() + static_cast<std::vector<float>::difference_type>(best + 1),
                     0.5f * (slots[best] + next));
    }
    for (float &s : slots) while (s >= float(count)) s -= float(count);
    return slots;
}

bool _SectionZeroWorld(PomadeTubeDesc const &desc,
                       std::vector<std::array<float, 3>> *out)
{
    if (!out || desc.centerX.empty() || desc.sections.empty()) return false;
    PomadeTubeSection const &section = desc.sections.front();
    if (section.u.size() != section.v.size() || section.u.empty() ||
        !(section.scale > 1e-6f) || !std::isfinite(section.twist)) return false;
    std::vector<PomadeFrame> frames;
    std::string err;
    if (!PomadeTubeFramesCpu(desc, &frames, &err) || frames.empty()) return false;
    float const ct = std::cos(section.twist), st = std::sin(section.twist);
    PomadeFrame const &frame = frames.front();
    out->resize(section.u.size());
    for (size_t i = 0; i < section.u.size(); ++i) {
        float const u = section.u[i] * section.scale;
        float const v = section.v[i] * section.scale;
        float const ru = u * ct - v * st;
        float const rv = u * st + v * ct;
        (*out)[i] = {{desc.centerX[0] + frame.nx * ru + frame.bx * rv,
                      desc.centerY[0] + frame.ny * ru + frame.by * rv,
                      desc.centerZ[0] + frame.nz * ru + frame.bz * rv}};
    }
    return true;
}

// Attachment conformance owns the inherited boundary but must not regenerate
// a child's sculpted upper cage from a new material partition.  Keep the
// child's transported center/frame reference and express the freshly derived
// root boundary in that chart. K4 therefore sees the same upper control
// curve and every authored non-root ring remains in its transported world
// pose. (The Hermite span adjacent to the changed root remains intentionally
// influenced by the new attachment.)
bool _FitSectionZeroWorld(PomadeTubeDesc *desc,
                          std::vector<std::array<float, 3>> const &points,
                          std::string *err)
{
    auto fail = [&](char const *why) {
        if (err) {
            *err = why;
        }
        return false;
    };
    if (!desc || desc->centerX.empty() || desc->sections.empty()) {
        return fail("attachment child has no root section");
    }
    PomadeTubeSection &section = desc->sections.front();
    if (points.size() != section.u.size() || section.u.size() != section.v.size() ||
        !(std::fabs(section.scale) > 1e-6f) ||
        !std::isfinite(section.scale) || !std::isfinite(section.twist)) {
        return fail("attachment child root layout is invalid");
    }
    std::vector<PomadeFrame> frames;
    std::string frameErr;
    if (!PomadeTubeFramesCpu(*desc, &frames, &frameErr) || frames.empty()) {
        return fail(frameErr.empty() ? "attachment child has no root frame"
                                     : frameErr.c_str());
    }
    PomadeFrame const &frame = frames.front();
    float const ct = std::cos(section.twist);
    float const st = std::sin(section.twist);
    std::vector<float> u(points.size()), v(points.size());
    for (size_t slot = 0; slot < points.size(); ++slot) {
        float const dx = points[slot][0] - desc->centerX[0];
        float const dy = points[slot][1] - desc->centerY[0];
        float const dz = points[slot][2] - desc->centerZ[0];
        float const ru = dx * frame.nx + dy * frame.ny + dz * frame.nz;
        float const rv = dx * frame.bx + dy * frame.by + dz * frame.bz;
        float const normal = dx * frame.tx + dy * frame.ty + dz * frame.tz;
        // Both child and parent inherit the same transported support frame;
        // anything materially off that plane is not a valid root attachment.
        if (!std::isfinite(ru) || !std::isfinite(rv) ||
            std::fabs(normal) > 1e-4f) {
            return fail("attachment root does not lie in child chart plane");
        }
        u[slot] = (ru * ct + rv * st) / section.scale;
        v[slot] = (-ru * st + rv * ct) / section.scale;
    }
    section.u = std::move(u);
    section.v = std::move(v);
    return true;
}

// K14 binds only physically retained parent boundary corners.  During an
// attachment refresh, a child section-zero layout may have been resampled;
// reconstruct its root triples from the *installed* world points and the
// current parent root.  Exact coincidence is required -- an internal cut or
// interpolated slot is deliberately not a holding edge.
bool _InstalledRootBoundaryBindings(
    PomadeTubeDesc const &parent, PomadeTubeDesc const &child,
    std::vector<PomadeParentBoundaryBinding> *out)
{
    if (!out) return false;
    std::vector<std::array<float, 3>> parentPoints, childPoints;
    if (!_SectionZeroWorld(parent, &parentPoints) ||
        !_SectionZeroWorld(child, &childPoints)) {
        return false;
    }
    float extent = 0.0f;
    for (std::array<float, 3> const &p : parentPoints) {
        float const dx = p[0] - parent.centerX[0];
        float const dy = p[1] - parent.centerY[0];
        float const dz = p[2] - parent.centerZ[0];
        extent = std::max(extent, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    float const tolerance = std::max(1e-5f, extent * 1e-5f);
    std::vector<PomadeParentBoundaryBinding> bindings;
    std::vector<bool> used(parentPoints.size(), false);
    for (size_t childSlot = 0; childSlot < childPoints.size(); ++childSlot) {
        int owner = -1;
        for (size_t parentSlot = 0; parentSlot < parentPoints.size(); ++parentSlot) {
            float const dx = childPoints[childSlot][0] - parentPoints[parentSlot][0];
            float const dy = childPoints[childSlot][1] - parentPoints[parentSlot][1];
            float const dz = childPoints[childSlot][2] - parentPoints[parentSlot][2];
            if (dx * dx + dy * dy + dz * dz > tolerance * tolerance) continue;
            // Repeated polygon points have no unambiguous material owner.
            if (owner >= 0 || used[parentSlot]) {
                owner = -2;
                break;
            }
            owner = int(parentSlot);
        }
        if (owner >= 0) {
            used[size_t(owner)] = true;
            bindings.push_back({0, owner, int(childSlot)});
        }
    }
    *out = std::move(bindings);
    return true;
}

std::vector<float> _RecoverFootprintSlots(
    _RegionFootprint const &oldFootprint, PomadeTubeDesc const &oldActual)
{
    int const count = int(oldFootprint.corners.size());
    int const ringVerts = oldActual.ringVerts;
    if (ringVerts == count) {
        return _TemplateFootprintSlots(oldFootprint, ringVerts);
    }
    std::vector<std::array<float, 3>> points;
    if (!_SectionZeroWorld(oldActual, &points) || int(points.size()) != ringVerts) {
        return _TemplateFootprintSlots(oldFootprint, ringVerts);
    }
    float extent = 0.0f;
    for (size_t i = 0; i < oldFootprint.corners.size(); ++i) {
        std::array<float, 3> const &a = oldFootprint.corners[i];
        std::array<float, 3> const &b =
            oldFootprint.corners[(i + 1) % oldFootprint.corners.size()];
        float const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
        extent = std::max(extent, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    float const tolerance = std::max(1e-4f, extent * 1e-4f);
    std::vector<float> slots;
    slots.reserve(points.size());
    bool valid = true;
    float previous = -1.0f;
    std::vector<bool> corners(size_t(count), false);
    for (size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex) {
        std::array<float, 3> const &p = points[pointIndex];
        float bestDistance2 = std::numeric_limits<float>::infinity();
        float bestS = 0.0f;
        for (int edge = 0; edge < count; ++edge) {
            std::array<float, 3> const &a = oldFootprint.corners[size_t(edge)];
            std::array<float, 3> const &b =
                oldFootprint.corners[size_t((edge + 1) % count)];
            float const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
            float const len2 = dx * dx + dy * dy + dz * dz;
            if (!(len2 > 1e-12f)) { valid = false; continue; }
            float t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy +
                       (p[2] - a[2]) * dz) / len2;
            t = std::max(0.0f, std::min(1.0f, t));
            float const qx = a[0] + dx * t, qy = a[1] + dy * t,
                        qz = a[2] + dz * t;
            float const ex = p[0] - qx, ey = p[1] - qy, ez = p[2] - qz;
            float const distance2 = ex * ex + ey * ey + ez * ez;
            if (distance2 < bestDistance2) {
                bestDistance2 = distance2;
                bestS = float(edge) + t;
            }
        }
        if (bestDistance2 > tolerance * tolerance) { valid = false; break; }
        for (int corner = 0; corner < count; ++corner) {
            std::array<float, 3> const &q = oldFootprint.corners[size_t(corner)];
            float const dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
            if (dx * dx + dy * dy + dz * dz <= tolerance * tolerance) {
                bestS = float(corner);
                corners[size_t(corner)] = true;
                break;
            }
        }
        // A complete recovered map must retain the canonical min-id seam
        // at slot zero and run once, strictly forward, around the old loop.
        // Do not unwrap later samples: doing so can incorrectly accept a
        // scrambled polygon after multiple wraps.
        if ((pointIndex == 0 && std::fabs(bestS) > 1e-4f) ||
            (previous >= 0.0f && bestS <= previous + 1e-4f)) {
            valid = false;
            break;
        }
        previous = bestS;
        slots.push_back(bestS);
    }
    for (bool found : corners) valid = valid && found;
    return valid ? slots : _TemplateFootprintSlots(oldFootprint, ringVerts);
}

std::vector<int> _StableRegionLoopKey(PomadeGraphRegion const &region)
{
    std::vector<int> key = region.loop;
    std::sort(key.begin(), key.end());
    key.erase(std::unique(key.begin(), key.end()), key.end());
    return key;
}

int _RegionForStableLoop(PomadeScalpGraph const &graph,
                         std::vector<int> const &key)
{
    for (PomadeGraphRegion const &region : graph.Regions()) {
        if (!region.isOutside && _StableRegionLoopKey(region) == key) {
            return region.id;
        }
    }
    return -1;
}

// A geometry-only graph move may transport attached tube subtrees.  Keep the
// topology test independent of extracted dense region ids, which are allowed
// to renumber while the same directed loop remains alive.
std::vector<std::string> _GraphTopologySignature(PomadeScalpGraph const &graph)
{
    std::vector<std::string> out;
    for (PomadeGraphNode const &node : graph.Nodes()) {
        if (node.alive) {
            out.push_back("N" + std::to_string(node.id));
        }
    }
    for (PomadeGraphEdge const &edge : graph.Edges()) {
        if (edge.alive) {
            out.push_back("E" + std::to_string(edge.id) + ":" +
                          std::to_string(edge.a) + ":" +
                          std::to_string(edge.b));
        }
    }
    for (PomadeGraphRegion const &region : graph.Regions()) {
        std::vector<int> loop = region.loop;
        if (loop.empty()) {
            continue;
        }
        size_t best = 0;
        for (size_t begin = 1; begin < loop.size(); ++begin) {
            for (size_t i = 0; i < loop.size(); ++i) {
                int const a = loop[(begin + i) % loop.size()];
                int const b = loop[(best + i) % loop.size()];
                if (a == b) {
                    continue;
                }
                if (a < b) {
                    best = begin;
                }
                break;
            }
        }
        std::string key = "R" + std::to_string(region.isOutside) + ":";
        for (size_t i = 0; i < loop.size(); ++i) {
            key += std::to_string(loop[(best + i) % loop.size()]) + ",";
        }
        out.push_back(std::move(key));
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool _RegionSupportFromGraph(PomadeScalpMesh const &scalp,
                             PomadeGraphRegion const &region,
                             _RegionSupport *out)
{
    if (!out || region.boundary.size() < 9) {
        return false;
    }
    size_t const count = region.boundary.size() / 3;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (size_t i = 0; i < count; ++i) {
        cx += region.boundary[i * 3 + 0];
        cy += region.boundary[i * 3 + 1];
        cz += region.boundary[i * 3 + 2];
    }
    float const centroid[3] = {float(cx / double(count)),
                               float(cy / double(count)),
                               float(cz / double(count))};
    PomadeHit const surface = PomadeClosestPointCpu(scalp, centroid);
    if (!surface.hit) {
        return false;
    }
    out->origin[0] = surface.px;
    out->origin[1] = surface.py;
    out->origin[2] = surface.pz;
    float pn[3] = {0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < count; ++i) {
        float const *a = &region.boundary[i * 3];
        float const *b = &region.boundary[((i + 1) % count) * 3];
        pn[0] += (a[1] - b[1]) * (a[2] + b[2]);
        pn[1] += (a[2] - b[2]) * (a[0] + b[0]);
        pn[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    float const len = PomadeLen3(pn);
    if (len > 1e-12f) {
        pn[0] /= len;
        pn[1] /= len;
        pn[2] /= len;
        if (PomadeDot3(pn, &surface.nx) < 0.0f) {
            pn[0] = -pn[0];
            pn[1] = -pn[1];
            pn[2] = -pn[2];
        }
    } else {
        pn[0] = surface.nx;
        pn[1] = surface.ny;
        pn[2] = surface.nz;
    }
    float const nlen = PomadeLen3(pn);
    if (!(nlen > 1e-12f)) {
        return false;
    }
    out->normal[0] = pn[0] / nlen;
    out->normal[1] = pn[1] / nlen;
    out->normal[2] = pn[2] / nlen;
    return true;
}

bool _RigidBetweenRegionSupports(_RegionSupport const &from,
                                 _RegionSupport const &to,
                                 PomadeRigidTransform *out)
{
    if (!out) {
        return false;
    }
    float const dot = std::max(-1.0f,
                               std::min(1.0f, PomadeDot3(from.normal,
                                                         to.normal)));
    float axis[3];
    PomadeCross3(from.normal, to.normal, axis);
    float const sinTheta = PomadeLen3(axis);
    float r[9] = {1.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f,
                  0.0f, 0.0f, 1.0f};
    if (sinTheta > 1e-7f) {
        axis[0] /= sinTheta;
        axis[1] /= sinTheta;
        axis[2] /= sinTheta;
        float const x = axis[0], y = axis[1], z = axis[2];
        float const oneMinus = 1.0f - dot;
        r[0] = dot + x * x * oneMinus;
        r[1] = x * y * oneMinus - z * sinTheta;
        r[2] = x * z * oneMinus + y * sinTheta;
        r[3] = y * x * oneMinus + z * sinTheta;
        r[4] = dot + y * y * oneMinus;
        r[5] = y * z * oneMinus - x * sinTheta;
        r[6] = z * x * oneMinus - y * sinTheta;
        r[7] = z * y * oneMinus + x * sinTheta;
        r[8] = dot + z * z * oneMinus;
    } else if (dot < 0.0f) {
        // A half-turn has no unique minimal axis. Choose a deterministic
        // in-plane axis; continuous scalp moves take the branch above.
        PomadePerp3(from.normal, axis);
        float const x = axis[0], y = axis[1], z = axis[2];
        r[0] = 2.0f * x * x - 1.0f;
        r[1] = 2.0f * x * y;
        r[2] = 2.0f * x * z;
        r[3] = 2.0f * y * x;
        r[4] = 2.0f * y * y - 1.0f;
        r[5] = 2.0f * y * z;
        r[6] = 2.0f * z * x;
        r[7] = 2.0f * z * y;
        r[8] = 2.0f * z * z - 1.0f;
    }
    std::copy(r, r + 9, out->rotation);
    float rotatedOrigin[3] = {
        r[0] * from.origin[0] + r[1] * from.origin[1] + r[2] * from.origin[2],
        r[3] * from.origin[0] + r[4] * from.origin[1] + r[5] * from.origin[2],
        r[6] * from.origin[0] + r[7] * from.origin[1] + r[8] * from.origin[2]};
    out->translation[0] = to.origin[0] - rotatedOrigin[0];
    out->translation[1] = to.origin[1] - rotatedOrigin[1];
    out->translation[2] = to.origin[2] - rotatedOrigin[2];
    return true;
}

}  // namespace

bool
PomadeModel::BuildTestTube(PomadeTubeShape shape)
{
    if (shape.rings < 2 || shape.ringVerts < 3 ||
        shape.radius <= 0.0f || shape.length <= 0.0f) {
        _diagnostic = "PomadeModel::BuildTestTube: shape out of range "
                      "(rings >= 2, ringVerts >= 3, radius/length > 0)";
        return false;
    }
    _ClearUndoLocked();  // rebuild invalidates pre-mutation snapshots
    _shape = shape;
    _host.centerX.assign(size_t(shape.rings), 0.0f);
    _host.centerY.resize(size_t(shape.rings));
    _host.centerZ.assign(size_t(shape.rings), 0.0f);
    for (int r = 0; r < shape.rings; ++r) {
        _host.centerY[size_t(r)] =
            shape.length * float(r) / float(shape.rings - 1);
    }
    // P3 rings start as default circles on the legacy display path; the
    // first section op switches to K5. Guides start empty (RefillGuides).
    {
        TubeSnapshot seed;
        seed.shape = shape;
        seed.centerX = _host.centerX;
        seed.centerY = _host.centerY;
        seed.centerZ = _host.centerZ;
        _sections = PomadeDefaultSections(seed);
    }
    _useSections = false;
    _tubeRegionId = -1;
    _rootFramePinned = false;
    _rootFrame = PomadeFrame();
    _frameReference = {{1.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, 1.0f}};
    _roots.clear();
    _tubeRoots.clear();
    _guides = PomadeGuideSet();
    ++_guideVersion;
    if (!Sync()) {
        return false;
    }
    // Topology always changes on a rebuild (counts/faces are re-emitted).
    _dirty |= PomadeDirty_Topology | PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::MoveCenterRing(int ring, float dx, float dz)
{
    if (ring < 0 || ring >= _shape.rings ||
        _host.centerX.size() != size_t(_shape.rings)) {
        _diagnostic = "PomadeModel::MoveCenterRing: ring index out of range";
        return false;
    }
    _PushUndoLocked();
    _host.centerX[size_t(ring)] += dx;
    _host.centerZ[size_t(ring)] += dz;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::Sync()
{
    if (!_TessellateHost()) {
        return false;
    }
    if (!_SyncDevice()) {
        return false;
    }
    return true;
}

bool
PomadeModel::HasCudaMirror() const
{
#ifdef USDGEN_POMADE_HAS_CUDA
    return _device && _device->streamOwned;
#else
    return false;
#endif
}

bool
PomadeModel::DeviceFallback() const
{
    return _deviceFallback;
}

char const *
PomadeModel::DeviceFallbackReason() const
{
    return _deviceFallbackReason.c_str();
}

#ifdef USDGEN_POMADE_HAS_CUDA
void
PomadeModel::_DropDeviceLocked(char const *reason)
{
    if (_deviceFallback) {
        return;
    }
    _deviceFallback = true;
    _deviceFallbackReason = reason ? reason : "device mirror dropped";
    if (!_device) {
        return;
    }
    // Release every buffer first so the OOM that brought us here is
    // actually relieved, then destroy the stream.
    _device->positions.release();
    _device->normals.release();
    _device->centerX.release();
    _device->centerY.release();
    _device->centerZ.release();
    _device->frames.release();
    _device->sectionT.release();
    _device->sectionU.release();
    _device->sectionV.release();
    _device->sectionScale.release();
    _device->sectionTwist.release();
    _device->rootSurfaceOffsets.release();
    _device->ringT.release();
    _device->guideCVs.release();
    _device->guideCVsVersion = 0;
    _device->pickBest.release();
    _device->pickScratch.release();
    if (_device->streamOwned && _device->stream) {
        cudaStreamDestroy(_device->stream);
    }
    _device->stream = nullptr;
    _device->streamOwned = false;
}

// Fault injection for the P6 T0 (and only the T0): a non-empty
// USDGEN_POMADE_FORCE_DEVICE_FALLBACK makes the next device sync drop
// the mirror deterministically, proving the fallback path runs.
static bool
DeviceFallbackForced()
{
    char const *forced = std::getenv("USDGEN_POMADE_FORCE_DEVICE_FALLBACK");
    return forced && *forced;
}
#endif

bool
PomadeModel::CopyDeviceToHost(float *posOut, float *nrmOut,
                             size_t floatCount) const
{
#ifdef USDGEN_POMADE_HAS_CUDA
    if (!posOut || !nrmOut || !_device || !_device->streamOwned ||
        !_device->stream) {
        return false;
    }
    if (_device->positions.size() != floatCount ||
        _device->normals.size() != floatCount) {
        return false;
    }
    // The tessellate kernel was enqueued on _device->stream by Sync; wait for
    // it before the D2H so the staged snapshot is exactly the last Sync.
    if (cudaStreamSynchronize(_device->stream) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(posOut, _device->positions.data(),
                   floatCount * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(nrmOut, _device->normals.data(),
                   floatCount * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    return true;
#else
    (void)posOut;
    (void)nrmOut;
    (void)floatCount;
    return false;
#endif
}

uint32_t
PomadeModel::TakeDirty()
{
    uint32_t dirty = _dirty;
    _dirty = PomadeDirty_Clean;
    return dirty;
}

bool
PomadeModel::SetFillParams(FillParams params)
{
    if (!(params.density >= 0.0f) || params.cvCount < 2 ||
        params.cvCount > 64) {
        _diagnostic = "PomadeModel::SetFillParams: density must be finite "
                      "and >= 0, cvCount in [2, 64]";
        return false;
    }
    if (params.edgeBias < -1.0f || params.edgeBias > 1.0f) {
        _diagnostic = "PomadeModel::SetFillParams: edgeBias in [-1, 1]";
        return false;
    }
    if (params.lengthProfile.size() % 2 != 0) {
        _diagnostic = "PomadeModel::SetFillParams: lengthProfile holds "
                      "(position, value) pairs";
        return false;
    }
    _fill = std::move(params);
    ++_version;
    return true;
}

PomadeModel::FillParams
PomadeModel::GetFillParams() const
{
    return _fill;
}

bool
PomadeModel::SetOutputSettings(OutputSettings settings)
{
    if (!std::isfinite(settings.densityMultiplier) ||
        !(settings.densityMultiplier > 0.0f) ||
        !std::isfinite(settings.width) || !(settings.width >= 0.0f) ||
        settings.ptexResolution < -1 || settings.ptexResolution > 12) {
        _diagnostic = "PomadeModel::SetOutputSettings: density multiplier must "
                      "be finite and > 0; width must be finite and >= 0; "
                      "ptex resolution must be -1 or in [0, 12]";
        return false;
    }
    if (_output.enabled == settings.enabled &&
        _output.densityMultiplier == settings.densityMultiplier &&
        _output.width == settings.width &&
        _output.ptexResolution == settings.ptexResolution) {
        return true;
    }
    HierarchyRollback const before = _SnapshotHierarchyLocked();
    _output = settings;
    _PushUndoSnapshotLocked(before);
    ++_version;
    return true;
}

PomadeModel::OutputSettings
PomadeModel::GetOutputSettings() const
{
    return _output;
}

bool
PomadeModel::SetOutputPtexResolution(int resOverride)
{
    if (resOverride < -1 || resOverride > 12) {
        _diagnostic = "PomadeModel::SetOutputPtexResolution: resolution must "
                      "be -1 or in [0, 12]";
        return false;
    }
    if (_output.ptexResolution == resOverride) {
        return true;
    }
    HierarchyRollback const before = _SnapshotHierarchyLocked();
    _output.ptexResolution = resOverride;
    _PushUndoSnapshotLocked(before);
    ++_version;
    return true;
}

bool
PomadeModel::SetSubdivideParams(SubdivideParams params)
{
    if (params.count < 2 || params.count > 8 ||
        (params.splitMode != "kmeans" && params.splitMode != "edge")) {
        _diagnostic = "PomadeModel::SetSubdivideParams: count in [2, 8], "
                      "splitMode kmeans | edge";
        return false;
    }
    _subdivide = std::move(params);
    ++_version;
    return true;
}

PomadeModel::SubdivideParams
PomadeModel::GetSubdivideParams() const
{
    return _subdivide;
}

void
PomadeModel::SetLockFlags(bool locked, bool lockParents, bool lockChildren)
{
    _locked = locked;
    _lockParents = lockParents;
    _lockChildren = lockChildren;
    ++_version;
}

void
PomadeModel::GetLockFlags(bool *locked, bool *lockParents,
                         bool *lockChildren) const
{
    if (locked) {
        *locked = _locked;
    }
    if (lockParents) {
        *lockParents = _lockParents;
    }
    if (lockChildren) {
        *lockChildren = _lockChildren;
    }
}

PomadeModelState const *
PomadeModel::CaptureState() const
{
    usdGen::UsdGenRef<PomadeModelState> state =
        usdGen::UsdGenMakeRef<PomadeModelState>();
    state->version = GetVersion();
    state->tube = Snapshot();
    state->generatedCurvesSuppressed = GeneratedCurvesSuppressed();
    state->output = GetOutputSettings();
    state->tubeRegionId = GetTubeRegionId();
    state->l1TubeIds = L1TubeIds();
    state->tubeIds = TubeIds();
    auto capture = [&](int id) {
        if (id == 0 || state->records.count(id)) {
            return;
        }
        TubeRecord record;
        if (GetTubeRecord(id, &record)) {
            state->records.emplace(id, std::move(record));
        }
    };
    for (int id : state->l1TubeIds) {
        capture(id);
    }
    for (int id : state->tubeIds) {
        capture(id);
    }
    state->graph = SnapshotGraph();
    state->scalp = GetScalp();
    return state.Detach();
}

void
PomadeModel::PublishState()
{
    _published->cell.Store(
        usdGen::UsdGenRef<PomadeModelState const>::Adopt(CaptureState()));
}

PomadeModelState const *
PomadeModel::AcquireState() const
{
    return _published->cell.Load().Detach();
}

PomadeModel::TubeSnapshot
PomadeModel::Snapshot() const
{
    TubeSnapshot snapshot;
    snapshot.shape = _shape;
    snapshot.centerX = _host.centerX;
    snapshot.centerY = _host.centerY;
    snapshot.centerZ = _host.centerZ;
    snapshot.sections = _sections;
    snapshot.rootFramePinned = _rootFramePinned;
    snapshot.rootFrame = _rootFrame;
    snapshot.frameReference = _frameReference;
    snapshot.fill = _fill;
    snapshot.subdivide = _subdivide;
    snapshot.locked = _locked;
    snapshot.lockParents = _lockParents;
    snapshot.lockChildren = _lockChildren;
    snapshot.version = _version.load();
    snapshot.hasTube =
        !_host.positions.empty() &&
        _host.centerX.size() == size_t(_shape.rings);
    return snapshot;
}

bool
PomadeModel::Restore(TubeSnapshot const &snapshot)
{
    if (!snapshot.hasTube || snapshot.shape.rings < 2 ||
        snapshot.shape.ringVerts < 3 || snapshot.shape.radius <= 0.0f ||
        snapshot.shape.length <= 0.0f ||
        snapshot.centerX.size() != size_t(snapshot.shape.rings) ||
        snapshot.centerY.size() != size_t(snapshot.shape.rings) ||
        snapshot.centerZ.size() != size_t(snapshot.shape.rings)) {
        _diagnostic = "PomadeModel::Restore: snapshot holds no valid tube";
        return false;
    }
    _shape = snapshot.shape;
    _host.centerX = snapshot.centerX;
    _host.centerY = snapshot.centerY;
    _host.centerZ = snapshot.centerZ;
    _rootFramePinned = snapshot.rootFramePinned;
    _rootFrame = snapshot.rootFrame;
    _frameReference = snapshot.frameReference;
    // Committed sections are authoritative (hydrate installs them); an
    // empty ring list means a pre-P3 snapshot and rebuilds the default
    // circles on the legacy display path.
    if (snapshot.sections.empty()) {
        TubeSnapshot seed = snapshot;
        _sections = PomadeDefaultSections(seed);
        _useSections = false;
    } else {
        if (snapshot.sections.size() < 2) {
            _diagnostic = "PomadeModel::Restore: need >= 2 sections";
            return false;
        }
        for (auto const &s : snapshot.sections) {
            if (int(s.u.size()) != snapshot.shape.ringVerts ||
                int(s.v.size()) != snapshot.shape.ringVerts) {
                _diagnostic = "PomadeModel::Restore: section ring mismatch";
                return false;
            }
        }
        _sections = snapshot.sections;
        _useSections = true;
    }
    _roots.clear();
    _tubeRoots.clear();
    _guides = PomadeGuideSet();
    ++_guideVersion;
    _fill = snapshot.fill;
    _subdivide = snapshot.subdivide;
    _locked = snapshot.locked;
    _lockParents = snapshot.lockParents;
    _lockChildren = snapshot.lockChildren;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Topology | PomadeDirty_Points;
    ++_version;
    return true;
}

int
PomadeGuideCountForDensity(float density)
{
    if (!(density > 0.0f)) {
        return 0;
    }
    // The no-scalp count rule (root area pinned to 1); the scalp-bound
    // form (K8 per-face, density * area) lives in PomadeRootSampleMeshCpu.
    return int(density + 0.5f);
}

std::vector<PomadeTubeSection>
PomadeDefaultSections(PomadeModel::TubeSnapshot const &snapshot)
{
    std::vector<PomadeTubeSection> sections;
    int const rings = int(snapshot.centerX.size()) > 0
                          ? int(snapshot.centerX.size())
                          : snapshot.shape.rings;
    int const ringVerts = snapshot.shape.ringVerts;
    if (rings < 2 || ringVerts < 3) {
        return sections;
    }
    float const twoPi = 6.28318530717958647692f;
    sections.resize(size_t(rings));
    for (int r = 0; r < rings; ++r) {
        PomadeTubeSection section;
        section.t = float(r) / float(rings - 1);
        section.u.resize(size_t(ringVerts));
        section.v.resize(size_t(ringVerts));
        for (int s = 0; s < ringVerts; ++s) {
            float const a = twoPi * float(s) / float(ringVerts);
            section.u[size_t(s)] = snapshot.shape.radius * std::cos(a);
            section.v[size_t(s)] = snapshot.shape.radius * std::sin(a);
        }
        sections[size_t(r)] = std::move(section);
    }
    return sections;
}

PomadeTubeDesc
PomadeTubeDescFromSnapshot(PomadeModel::TubeSnapshot const &snapshot)
{
    PomadeTubeDesc desc;
    desc.centerX = snapshot.centerX;
    desc.centerY = snapshot.centerY;
    desc.centerZ = snapshot.centerZ;
    desc.ringVerts = snapshot.shape.ringVerts;
    desc.rootFramePinned = snapshot.rootFramePinned;
    desc.rootFrame = snapshot.rootFrame;
    desc.frameReference = snapshot.frameReference;
    bool sectionsOk = snapshot.sections.size() >= 2;
    for (auto const &s : snapshot.sections) {
        if (int(s.u.size()) != desc.ringVerts ||
            int(s.v.size()) != desc.ringVerts) {
            sectionsOk = false;
            break;
        }
    }
    desc.sections =
        sectionsOk ? snapshot.sections : PomadeDefaultSections(snapshot);
    return desc;
}

namespace {

// Shared K9/K10 tail: fill `roots` through the tube and resample to the
// fill CV count with root frames. Returns false on any kernel error,
// carrying the kernel's message in `err` when it is not null.
bool _FillAndResample(PomadeTubeDesc const &tube,
                      std::vector<PomadeFrame> const &frames,
                      std::vector<PomadeGuideRoot> const &roots,
                      PomadeModel::FillParams const &fill, PomadeGuideSet *guides,
                      std::string *err = nullptr)
{
    PomadeFillDesc fd;
    fd.density = fill.density;
    fd.cvCount = fill.cvCount;
    fd.seed = fill.seed;
    fd.edgeBias = fill.edgeBias;
    fd.lengthProfile = fill.lengthProfile;
    fd.sampler = fill.sampler;
    std::vector<float> filled;
    std::vector<float> lengths;
    std::string local;
    if (!PomadeGuideFillCpu(tube, frames, roots, fd, &filled, &lengths,
                           &local)) {
        if (err) {
            *err = local;
        }
        return false;
    }
    int const guideCount = int(roots.size());
    std::vector<int> inCounts(size_t(guideCount), fill.cvCount);
    float rootDirs[9] = {frames[0].nx, frames[0].ny, frames[0].nz,
                         frames[0].bx, frames[0].by, frames[0].bz,
                         frames[0].tx, frames[0].ty, frames[0].tz};
    std::vector<float> resampled;
    std::vector<int> outCounts;
    std::vector<double> outFrames;
    if (!PomadeGuideResampleCpu(filled.data(), inCounts.data(), guideCount,
                               fill.cvCount, rootDirs, &resampled, &outCounts,
                               &outFrames, &local)) {
        if (err) {
            *err = local;
        }
        return false;
    }
    guides->points = std::move(resampled);
    guides->counts = std::move(outCounts);
    guides->frames = std::move(outFrames);
    guides->ids.resize(size_t(guideCount));
    for (int g = 0; g < guideCount; ++g) {
        guides->ids[size_t(g)] = uint64_t(1000 + g);
    }
    guides->guideCount = guideCount;
    guides->cvCount = fill.cvCount;
    return true;
}

void _AnchorGuideRoots(std::vector<PomadeGuideRoot> const &roots,
                       PomadeGuideSet *guides)
{
    if (!guides || guides->cvCount <= 0 ||
        guides->points.size() < roots.size() * size_t(guides->cvCount) * 3) {
        return;
    }
    for (size_t g = 0; g < roots.size(); ++g) {
        size_t const first = g * size_t(guides->cvCount) * 3;
        float const dx = roots[g].px - guides->points[first];
        float const dy = roots[g].py - guides->points[first + 1];
        float const dz = roots[g].pz - guides->points[first + 2];
        for (int cv = 0; cv < guides->cvCount; ++cv) {
            size_t const at = first + size_t(cv) * 3;
            guides->points[at] += dx;
            guides->points[at + 1] += dy;
            guides->points[at + 2] += dz;
        }
        size_t const frame = g * 16;
        if (guides->frames.size() >= frame + 15) {
            guides->frames[frame + 12] = roots[g].px;
            guides->frames[frame + 13] = roots[g].py;
            guides->frames[frame + 14] = roots[g].pz;
        }
    }
}

} // namespace

PomadeGuideSet
PomadeGenerateGuides(PomadeModel::TubeSnapshot const &snapshot)
{
    return PomadeGenerateGuidesForTube(snapshot, 0);
}

PomadeGuideSet
PomadeGenerateGuidesOnScalp(PomadeModel::TubeSnapshot const &snapshot,
                           PomadeScalpMesh const &scalp, int const *regionFaces,
                           int regionFaceCount)
{
    return PomadeGenerateGuidesOnScalpForTube(snapshot, scalp, regionFaces,
                                             regionFaceCount, 0);
}

PomadeModel::TubeSnapshot
PomadeSnapshotFromTubeRecord(PomadeModel::TubeRecord const &record)
{
    PomadeModel::TubeSnapshot snapshot;
    if (!record.hasTube || record.actual.centerX.size() < 2) {
        return snapshot;
    }
    PomadeTubeDesc const &desc = record.actual;
    snapshot.hasTube = true;
    snapshot.shape.rings = int(desc.centerX.size());
    snapshot.shape.ringVerts = desc.ringVerts;
    snapshot.centerX = desc.centerX;
    snapshot.centerY = desc.centerY;
    snapshot.centerZ = desc.centerZ;
    snapshot.sections = desc.sections;
    snapshot.rootFramePinned = desc.rootFramePinned;
    snapshot.rootFrame = desc.rootFrame;
    snapshot.frameReference = desc.frameReference;
    // The legacy cylinder fields only feed PomadeDefaultSections, which an
    // authored tube never reaches; keep them consistent anyway.
    snapshot.shape.radius = desc.sections.empty()
                                ? 0.5f
                                : PomadeSectionMeanRadius(desc.sections.front());
    snapshot.shape.length =
        PomadeCenterArcLength(desc.centerX.data(), desc.centerY.data(),
                             desc.centerZ.data(), int(desc.centerX.size()));
    snapshot.fill = record.fill;
    snapshot.subdivide = record.subdivide;
    snapshot.locked = record.locked;
    snapshot.lockParents = record.lockParents;
    snapshot.lockChildren = record.lockChildren;
    return snapshot;
}

PomadeGuideSet
PomadeGenerateGuidesForTube(PomadeModel::TubeSnapshot const &snapshot,
                           int tubeId, bool usePinnedRootFrame)
{
    PomadeGuideSet guides;
    if (!snapshot.hasTube) {
        return guides;
    }
    int const count = PomadeGuideCountForDensity(snapshot.fill.density);
    int const cvCount = snapshot.fill.cvCount;
    if (count <= 0 || cvCount < 2) {
        return guides;
    }
    PomadeTubeDesc const tube = PomadeTubeDescFromSnapshot(snapshot);
    if (tube.centerX.size() < 2 || tube.sections.size() < 2) {
        return guides;
    }
    std::vector<PomadeFrame> frames;
    std::string err;
    PomadeTubeDesc frameTube = tube;
    frameTube.rootFramePinned = usePinnedRootFrame && tube.rootFramePinned;
    if (!PomadeTubeFramesCpu(frameTube, &frames, &err)) {
        return guides;
    }
    float const rootRadius =
        PomadeSectionMeanRadius(tube.sections.front());
    if (!(rootRadius > 0.0f)) {
        return guides;
    }
    float const rootCenter[3] = {tube.centerX[0], tube.centerY[0],
                                 tube.centerZ[0]};
    std::vector<PomadeGuideRoot> roots;
    // The root stream is keyed by (tubeId, seed), so siblings that share
    // fill params still draw distinct root patterns (plan/17 §4.1).
    if (!PomadeRootSampleDiscCpu(tubeId, snapshot.fill.seed, rootRadius,
                                rootCenter, frames[0], count, &roots, 0,
                                &err)) {
        return guides;
    }
    if (!_FillAndResample(tube, frames, roots, snapshot.fill, &guides)) {
        return PomadeGuideSet();
    }
    guides.tubeIds.assign(size_t(guides.guideCount), tubeId);
    return guides;
}

PomadeGuideSet
PomadeGenerateGuidesOnScalpForTube(PomadeModel::TubeSnapshot const &snapshot,
                                  PomadeScalpMesh const &scalp,
                                  int const *regionFaces, int regionFaceCount,
                                  int tubeId, bool usePinnedRootFrame)
{
    PomadeGuideSet guides;
    if (!snapshot.hasTube || !scalp.finalized) {
        return guides;
    }
    if (!regionFaces || regionFaceCount <= 0) {
        return PomadeGenerateGuidesForTube(snapshot, tubeId,
                                          usePinnedRootFrame);
    }
    PomadeTubeDesc const tube = PomadeTubeDescFromSnapshot(snapshot);
    if (tube.centerX.size() < 2 || tube.sections.size() < 2) {
        return guides;
    }
    std::vector<PomadeFrame> frames;
    std::string err;
    PomadeTubeDesc frameTube = tube;
    frameTube.rootFramePinned = usePinnedRootFrame && tube.rootFramePinned;
    if (!PomadeTubeFramesCpu(frameTube, &frames, &err)) {
        return guides;
    }
    float const rootRadius =
        PomadeSectionMeanRadius(tube.sections.front());
    if (!(rootRadius > 0.0f)) {
        return guides;
    }
    float const rootCenter[3] = {tube.centerX[0], tube.centerY[0],
                                 tube.centerZ[0]};
    std::vector<PomadeGuideRoot> roots;
    if (!PomadeRootSampleMeshCpu(
            scalp.points.data(), scalp.faceVertexCounts.data(),
            scalp.faceVertexIndices.data(), scalp.faceOffsets.data(),
            int(scalp.faceVertexCounts.size()), regionFaces, regionFaceCount,
            snapshot.fill.density, tubeId, snapshot.fill.seed, rootCenter,
            frames[0], rootRadius, &roots, 0, &err)) {
        return guides;
    }
    if (roots.empty()) {
        return guides;
    }
    if (!_FillAndResample(tube, frames, roots, snapshot.fill, &guides)) {
        return PomadeGuideSet();
    }
    guides.tubeIds.assign(size_t(guides.guideCount), tubeId);
    return guides;
}

PomadeGuideSet
PomadeGenerateGuidesOnRegionForTube(PomadeModel::TubeSnapshot const &snapshot,
                                   PomadeScalpMesh const &scalp,
                                   PomadeRegionLoops const &loops,
                                   int sourceRegionId, int tubeId,
                                   std::vector<PomadeRootOwnershipCell> const *ownership,
                                   bool usePinnedRootFrame)
{
    PomadeGuideSet guides;
    PomadeTubeDesc const tube = PomadeTubeDescFromSnapshot(snapshot);
    if (!snapshot.hasTube || !scalp.finalized || tube.centerX.size() < 2 ||
        tube.sections.size() < 2) {
        return guides;
    }
    std::vector<PomadeFrame> frames;
    std::string err;
    PomadeTubeDesc frameTube = tube;
    frameTube.rootFramePinned = usePinnedRootFrame && tube.rootFramePinned;
    if (!PomadeTubeFramesCpu(frameTube, &frames, &err)) {
        return guides;
    }
    float const radius = PomadeSectionMeanRadius(tube.sections.front());
    float const root[3] = {tube.centerX[0], tube.centerY[0], tube.centerZ[0]};
    std::vector<PomadeGuideRoot> roots;
    if (!(radius > 0.0f) || !PomadeRootSampleRegionMeshCpu(
                               scalp, loops, sourceRegionId,
                               snapshot.fill.density, tubeId, snapshot.fill.seed,
                               root, frames[0], radius, &roots, 0, &err) ||
        roots.empty()) {
        return PomadeGuideSet();
    }
    if (ownership) {
        PomadeFilterGuideRootsByOwnershipCells(*ownership, &roots);
    }
    if (roots.empty() ||
        !_FillAndResample(tube, frames, roots, snapshot.fill, &guides)) {
        return PomadeGuideSet();
    }
    _AnchorGuideRoots(roots, &guides);
    guides.tubeIds.assign(size_t(guides.guideCount), tubeId);
    return guides;
}

// -- P2: scalp binding, graph, region maps -------------------------------

bool
PomadeModel::BindScalp(std::vector<float> const &points,
                      std::vector<int> const &faceVertexCounts,
                      std::vector<int> const &faceVertexIndices,
                      std::vector<int> const &activeFaces)
{
    std::shared_ptr<PomadeScalpMesh> scalp(new PomadeScalpMesh());
    scalp->points = points;
    scalp->faceVertexCounts = faceVertexCounts;
    scalp->faceVertexIndices = faceVertexIndices;
    scalp->activeFaces = activeFaces;
    std::string err;
    if (!PomadeScalpFinalize(scalp.get(), &err)) {
        _diagnostic = err;
        return false;
    }
    PomadeScalpBvh bvh;
    if (!PomadeScalpBvhBuild(*scalp, &bvh, &err)) {
        _diagnostic = err;
        return false;
    }
    _ClearUndoLocked();  // rebind invalidates region-linked snapshots
    _scalp = std::move(scalp);
    _bvh = std::move(bvh);
    _graph.Clear();
    _maps = PomadeRegionMaps();
    _loops = PomadeRegionLoops();
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
    ++_version;
    ++_mapVersion;
    return true;
}

bool
PomadeModel::HasScalp() const
{
    return bool(_scalp) && _scalp->finalized;
}

std::shared_ptr<PomadeScalpMesh const>
PomadeModel::GetScalp() const
{
    return _scalp;
}

PomadeHit
PomadeModel::Raycast(float const origin[3], float const dir[3]) const
{
    if (!_scalp || !_scalp->finalized || !_bvh.valid) {
        return PomadeHit();
    }
    return PomadeRaycastCpu(*_scalp, _bvh, origin, dir);
}

PomadeHit
PomadeModel::ClosestPoint(float const p[3]) const
{
    if (!_scalp || !_scalp->finalized || !p) {
        return PomadeHit();
    }
    return PomadeClosestPointCpu(*_scalp, p);
}

namespace {

// Mirror-twin a placed node across x = 0 (model flag). Best-effort: a miss
// (asymmetric scalp) twins nothing, and the placed node survives.
void _MaybeMirror(PomadeScalpMesh const &scalp, PomadeScalpGraph *graph,
                  PomadeHit const &hit)
{
    float const mirrored[3] = {-hit.px, hit.py, hit.pz};
    PomadeHit const twin = PomadeClosestPointCpu(scalp, mirrored);
    if (twin.hit) {
        graph->AddNode(twin);
    }
}

}  // namespace

int
PomadeModel::GraphAddNode(PomadeHit const &hit)
{
    auto const graphBefore = _GraphUndoStateLocked();
    if (!_scalp) {
        _diagnostic = "PomadeModel::GraphAddNode: no scalp bound";
        return -1;
    }
    int const id = _graph.AddNode(hit);
    if (id < 0) {
        _diagnostic = _graph.GetDiagnostic();
        return -1;
    }
    if (_mirrorX) {
        _MaybeMirror(*_scalp, &_graph, hit);
    }
    _graph.ExtractRegions(*_scalp);
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
    _PushGraphUndoLocked(graphBefore, "add node");
    ++_version;
    ++_mapVersion;
    return id;
}

int
PomadeModel::GraphCreateRegion(std::vector<int> const &nodeIds,
                              std::vector<PomadeHit> const &hits)
{
    if (!_scalp || !_scalp->finalized) {
        _diagnostic = "PomadeModel::GraphCreateRegion: no scalp bound";
        return -1;
    }
    if (nodeIds.size() != hits.size() || nodeIds.size() < 3) {
        _diagnostic = "PomadeModel::GraphCreateRegion: need three or more nodes";
        return -1;
    }
    // Validate the whole request before adding even a single node.  A
    // non-negative id is a stable graph id, never the compact display index.
    for (size_t i = 0; i < nodeIds.size(); ++i) {
        if (nodeIds[i] >= 0) {
            if (!_graph.FindNode(nodeIds[i])) {
                _diagnostic = "PomadeModel::GraphCreateRegion: unknown node id";
                return -1;
            }
        } else if (nodeIds[i] != -1 || !hits[i].hit) {
            _diagnostic = "PomadeModel::GraphCreateRegion: invalid new node";
            return -1;
        }
    }
    auto const graphBefore = _GraphUndoStateLocked();
    // Restore() rebuilds ids from live vectors, which is correct for undo but
    // not for this all-or-nothing batch: a rejected chain must leave existing
    // sparse edge ids and the graph's id counters exactly untouched.
    PomadeScalpGraph const graphBeforeExact = _graph;
    auto rollback = [&](char const *reason) {
        _graph = graphBeforeExact;
        _diagnostic = reason;
        return -1;
    };

    std::vector<int> chain = nodeIds;
    for (size_t i = 0; i < chain.size(); ++i) {
        if (chain[i] != -1) {
            continue;
        }
        chain[i] = _graph.AddNode(hits[i]);  // AddNode itself does no K3.
        if (chain[i] < 0) {
            return rollback("PomadeModel::GraphCreateRegion: add node failed");
        }
    }
    std::set<std::pair<int, int>> connected;
    for (PomadeGraphEdge const &edge : _graph.Edges()) {
        if (edge.alive) {
            connected.insert(std::minmax(edge.a, edge.b));
        }
    }
    for (size_t i = 0; i < chain.size(); ++i) {
        int const a = chain[i];
        int const b = chain[(i + 1) % chain.size()];
        if (a == b) {
            return rollback("PomadeModel::GraphCreateRegion: repeated adjacent node");
        }
        std::pair<int, int> const key = std::minmax(a, b);
        if (connected.insert(key).second &&
            _graph.Connect(*_scalp, a, b, /*extract*/ false) < 0) {
            return rollback("PomadeModel::GraphCreateRegion: edge trace failed");
        }
    }
    // The chain is deliberately the only batch in this mutation: extraction
    // observes the complete shared-edge topology exactly once.
    if (!_graph.ExtractRegions(*_scalp)) {
        return rollback("PomadeModel::GraphCreateRegion: region extraction failed");
    }
    auto sameCycle = [&chain](std::vector<int> const &loop) {
        if (loop.size() != chain.size()) {
            return false;
        }
        for (size_t begin = 0; begin < loop.size(); ++begin) {
            bool forward = true, backward = true;
            for (size_t i = 0; i < loop.size() && (forward || backward); ++i) {
                forward = forward && loop[(begin + i) % loop.size()] == chain[i];
                backward = backward &&
                    loop[(begin + loop.size() - i) % loop.size()] == chain[i];
            }
            if (forward || backward) {
                return true;
            }
        }
        return false;
    };
    int regionId = -1;
    for (PomadeGraphRegion const &region : _graph.Regions()) {
        if (sameCycle(region.loop)) {
            regionId = region.id;
            break;
        }
    }
    if (regionId < 0) {
        return rollback("PomadeModel::GraphCreateRegion: chain encloses no region");
    }
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
    _PushGraphUndoLocked(graphBefore, "create region");
    ++_version;
    ++_mapVersion;
    return regionId;
}

bool
PomadeModel::GraphMoveNode(int nodeId, PomadeHit const &hit)
{
    // Place has historically allowed a single-node drag to change graph
    // topology (for example a deliberate collapse before weld).  For the
    // ordinary existing-region drag, however, the attached tube hierarchy
    // must follow the same support transport as Reposition's batch path.
    // Capture the complete state so transport failure cannot leave a moved
    // graph paired with stale tube geometry.
    HierarchyRollback const before = _SnapshotHierarchyLocked();
    uint32_t const dirtyBefore = _dirty;
    uint64_t const versionBefore = _version.load();
    uint64_t const mapVersionBefore = _mapVersion;
    if (!_scalp) {
        _diagnostic = "PomadeModel::GraphMoveNode: no scalp bound";
        return false;
    }
    std::vector<std::string> const topologyBefore =
        _GraphTopologySignature(_graph);
    // A Place gesture can create/split graph topology before its later drag
    // samples move an existing or newly created point.  Those samples are
    // locally geometry-only, but their press-time graph is not a compatible
    // attachment baseline.  Keep the entire topology-edit gesture on the
    // permissive legacy route rather than rejecting its later move.
    bool gestureTopologyCompatible = true;
    if (_gestureDepth > 0) {
        PomadeScalpGraph gestureGraph;
        std::string restoreError;
        if (!_gestureBase.graph ||
            !gestureGraph.Restore(*_scalp, _gestureBase.graph->nodes,
                                  _gestureBase.graph->edges,
                                  _gestureBase.graph->linked,
                                  &restoreError) ||
            _GraphTopologySignature(gestureGraph) != topologyBefore) {
            gestureTopologyCompatible = false;
        }
    }
    if (!_graph.MoveNode(*_scalp, nodeId, hit)) {
        _diagnostic = _graph.GetDiagnostic();
        return false;
    }
    if (gestureTopologyCompatible &&
        _GraphTopologySignature(_graph) == topologyBefore) {
        // Gesture samples are absolute from the press-time graph and tube
        // state, not compounded from the previous sample.
        HierarchyRollback const &transportSource =
            _gestureDepth > 0 ? _gestureBase : before;
        if (!_TransportRegionAttachmentsLocked(transportSource, {nodeId})) {
            std::string const why = _diagnostic;
            _RestoreHierarchyLocked(before);
            _dirty = dirtyBefore;
            _version.store(versionBefore);
            _mapVersion = mapVersionBefore;
            _diagnostic = why;
            return false;
        }
        _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
        if (_nextUndoLabel.empty()) {
            _nextUndoLabel = "move node";
        }
        _PushUndoSnapshotLocked(before);
        ++_version;
        ++_mapVersion;
        return true;
    }
    // Retain Place's permissive legacy topology-edit behavior.  It must not
    // be forced through attachment transport, whose stable-loop contract
    // intentionally rejects a collapse/rewire.
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
    _PushGraphUndoLocked(before.graph, "move node");
    ++_version;
    ++_mapVersion;
    return true;
}

bool
PomadeModel::GraphMoveNodes(std::vector<int> const &nodeIds,
                           std::vector<PomadeHit> const &hits)
{
    HierarchyRollback const before = _SnapshotHierarchyLocked();
    uint32_t const dirtyBefore = _dirty;
    uint64_t const versionBefore = _version.load();
    uint64_t const mapVersionBefore = _mapVersion;
    if (!_scalp || nodeIds.empty() || nodeIds.size() != hits.size()) {
        _diagnostic = "PomadeModel::GraphMoveNodes: invalid targets";
        return false;
    }
    // Keep stable graph identity and every directed region boundary intact
    // while a drag changes only geometry. A broad sample can otherwise
    // collapse a loop or rewire extraction; reject it and keep the last
    // valid sample rather than dropping a region-rooted groom mid-drag.
    auto topology = [](PomadeScalpGraph const &graph) {
        std::vector<std::string> out;
        for (PomadeGraphNode const &node : graph.Nodes()) {
            if (node.alive) {
                out.push_back("N" + std::to_string(node.id));
            }
        }
        for (PomadeGraphEdge const &edge : graph.Edges()) {
            if (edge.alive) {
                out.push_back("E" + std::to_string(edge.id) + ":" +
                              std::to_string(edge.a) + ":" +
                              std::to_string(edge.b));
            }
        }
        for (PomadeGraphRegion const &region : graph.Regions()) {
            std::vector<int> loop = region.loop;
            if (!loop.empty()) {
                size_t best = 0;
                for (size_t begin = 1; begin < loop.size(); ++begin) {
                    for (size_t i = 0; i < loop.size(); ++i) {
                        int const a = loop[(begin + i) % loop.size()];
                        int const b = loop[(best + i) % loop.size()];
                        if (a == b) {
                            continue;
                        }
                        if (a < b) {
                            best = begin;
                        }
                        break;
                    }
                }
                std::string key = "R" + std::to_string(region.isOutside) +
                                  ":";
                for (size_t i = 0; i < loop.size(); ++i) {
                    key += std::to_string(loop[(best + i) % loop.size()]) +
                           ",";
                }
                out.push_back(std::move(key));
            }
        }
        std::sort(out.begin(), out.end());
        return out;
    };
    PomadeScalpGraph const graphBeforeExact = _graph;
    std::vector<std::string> const topologyBefore = topology(_graph);
    if (!_graph.MoveNodes(*_scalp, nodeIds, hits)) {
        _diagnostic = _graph.GetDiagnostic();
        return false;
    }
    if (topology(_graph) != topologyBefore) {
        _graph = graphBeforeExact;
        _diagnostic = "PomadeModel::GraphMoveNodes: target changes graph topology";
        return false;
    }
    // During a gesture the target hits are expressed from the press-time
    // graph.  Use that same hierarchy source for the attached tube pose so
    // consecutive samples are absolute rather than compounded. A one-shot
    // move uses the immediately preceding state captured above.
    HierarchyRollback const &transportSource =
        _gestureDepth > 0 ? _gestureBase : before;
    if (!_TransportRegionAttachmentsLocked(transportSource, nodeIds)) {
        std::string const why = _diagnostic;
        _RestoreHierarchyLocked(before);
        _dirty = dirtyBefore;
        _version.store(versionBefore);
        _mapVersion = mapVersionBefore;
        _diagnostic = why;
        return false;
    }
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
    if (_nextUndoLabel.empty()) {
        _nextUndoLabel = nodeIds.size() == 1 ? "move node" : "move nodes";
    }
    _PushUndoSnapshotLocked(before);
    ++_version;
    ++_mapVersion;
    return true;
}

int
PomadeModel::GraphConnect(int a, int b)
{
    auto const graphBefore = _GraphUndoStateLocked();
    if (!_scalp) {
        _diagnostic = "PomadeModel::GraphConnect: no scalp bound";
        return -1;
    }
    int const id = _graph.Connect(*_scalp, a, b);
    if (id < 0) {
        _diagnostic = _graph.GetDiagnostic();
        return -1;
    }
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
    _PushGraphUndoLocked(graphBefore, "connect");
    ++_version;
    ++_mapVersion;
    return id;
}

int
PomadeModel::GraphSplitEdge(int edgeId, PomadeHit const &hit)
{
    auto const graphBefore = _GraphUndoStateLocked();
    if (!_scalp) {
        _diagnostic = "PomadeModel::GraphSplitEdge: no scalp bound";
        return -1;
    }
    int const id = _graph.SplitEdge(*_scalp, edgeId, hit);
    if (id < 0) {
        _diagnostic = _graph.GetDiagnostic();
        return -1;
    }
    if (_mirrorX) {
        _MaybeMirror(*_scalp, &_graph, hit);
    }
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
    _PushGraphUndoLocked(graphBefore, "split edge");
    ++_version;
    ++_mapVersion;
    return id;
}

bool
PomadeModel::GraphWeld(int keep, int drop)
{
    auto const graphBefore = _GraphUndoStateLocked();
    if (!_scalp) {
        _diagnostic = "PomadeModel::GraphWeld: no scalp bound";
        return false;
    }
    if (!_graph.Weld(*_scalp, keep, drop)) {
        _diagnostic = _graph.GetDiagnostic();
        return false;
    }
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
    _PushGraphUndoLocked(graphBefore, "weld");
    ++_version;
    ++_mapVersion;
    return true;
}

int
PomadeModel::GraphWeldAll(float radius)
{
    auto const graphBefore = _GraphUndoStateLocked();
    if (!_scalp) {
        _diagnostic = "PomadeModel::GraphWeldAll: no scalp bound";
        return -1;
    }
    int const welds = _graph.WeldAllWithinRadius(*_scalp, radius);
    if (welds > 0) {
        _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
        _PushGraphUndoLocked(graphBefore, "weld all");
        ++_version;
        ++_mapVersion;
    }
    return welds;
}

std::vector<int>
PomadeModel::GraphUnweld(int nodeId)
{
    auto const graphBefore = _GraphUndoStateLocked();
    std::vector<int> created;
    if (!_scalp) {
        _diagnostic = "PomadeModel::GraphUnweld: no scalp bound";
        return created;
    }
    created = _graph.Unweld(*_scalp, nodeId);
    if (!created.empty()) {
        _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
        _PushGraphUndoLocked(graphBefore, "unweld");
        ++_version;
        ++_mapVersion;
    }
    return created;
}

bool
PomadeModel::GraphDeleteEdge(int edgeId)
{
    auto const graphBefore = _GraphUndoStateLocked();
    if (!_scalp) {
        _diagnostic = "PomadeModel::GraphDeleteEdge: no scalp bound";
        return false;
    }
    if (!_graph.DeleteEdge(*_scalp, edgeId)) {
        _diagnostic = _graph.GetDiagnostic();
        return false;
    }
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
    _PushGraphUndoLocked(graphBefore, "delete edge");
    ++_version;
    ++_mapVersion;
    return true;
}

bool
PomadeModel::GraphDeleteNode(int nodeId)
{
    auto const graphBefore = _GraphUndoStateLocked();
    if (!_scalp) {
        _diagnostic = "PomadeModel::GraphDeleteNode: no scalp bound";
        return false;
    }
    if (!_graph.DeleteNode(*_scalp, nodeId)) {
        _diagnostic = _graph.GetDiagnostic();
        return false;
    }
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
    _PushGraphUndoLocked(graphBefore, "delete node");
    ++_version;
    ++_mapVersion;
    return true;
}

int
PomadeModel::GraphSnapNode(float const p[3], float radius) const
{
    return _graph.SnapNode(p, radius);
}

int
PomadeModel::GraphSnapEdge(float const p[3], float radius) const
{
    return _graph.SnapEdge(p, radius);
}

bool
PomadeModel::GraphGetNode(int nodeId, PomadeGraphNode *out) const
{
    PomadeGraphNode const *node = _graph.FindNode(nodeId);
    if (!node || !out) {
        return false;
    }
    *out = *node;
    return true;
}

bool
PomadeModel::GraphGetEdge(int edgeId, int outNodeIds[2]) const
{
    PomadeGraphEdge const *edge = _graph.FindEdge(edgeId);
    if (!edge || !outNodeIds) {
        return false;
    }
    outNodeIds[0] = edge->a;
    outNodeIds[1] = edge->b;
    return true;
}

float
PomadeGraphDisplayLift(PomadeScalpMesh const *scalp)
{
    if (!scalp || !scalp->finalized) {
        return 0.0f;
    }
    return 4.0e-3f * scalp->boundsDiagonal;
}

void
PomadeGraphDisplayPosition(PomadeGraphNode const &node,
                          PomadeScalpMesh const *scalp, float outP[3])
{
    if (!outP) {
        return;
    }
    outP[0] = node.p[0];
    outP[1] = node.p[1];
    outP[2] = node.p[2];
    float const lift = PomadeGraphDisplayLift(scalp);
    float const n2 = node.n[0] * node.n[0] + node.n[1] * node.n[1] +
                     node.n[2] * node.n[2];
    if (!(lift > 0.0f) || !(n2 > 1.0e-20f)) {
        return;
    }
    float const scale = lift / std::sqrt(n2);
    outP[0] += node.n[0] * scale;
    outP[1] += node.n[1] * scale;
    outP[2] += node.n[2] * scale;
}

void
PomadeGraphDisplaySurfacePosition(float const canonicalP[3],
                                 PomadeScalpMesh const *scalp,
                                 float outP[3])
{
    if (!canonicalP || !outP) {
        return;
    }
    outP[0] = canonicalP[0];
    outP[1] = canonicalP[1];
    outP[2] = canonicalP[2];
    float const lift = PomadeGraphDisplayLift(scalp);
    if (!(lift > 0.0f) || !scalp) {
        return;
    }
    PomadeHit const hit = PomadeClosestPointCpu(*scalp, canonicalP);
    float const n2 = hit.nx * hit.nx + hit.ny * hit.ny + hit.nz * hit.nz;
    if (!hit.hit || !(n2 > 1.0e-20f)) {
        return;
    }
    float const scale = lift / std::sqrt(n2);
    outP[0] += hit.nx * scale;
    outP[1] += hit.ny * scale;
    outP[2] += hit.nz * scale;
}

bool
PomadeModel::GraphGetNodeDisplayPosition(int nodeId, float outP[3]) const
{
    PomadeGraphNode const *node = _graph.FindNode(nodeId);
    if (!node || !outP) {
        return false;
    }
    PomadeGraphDisplayPosition(*node, _scalp.get(), outP);
    return true;
}

bool
PomadeModel::GraphLinkRegions(int r0, int r1)
{
    auto const graphBefore = _GraphUndoStateLocked();
    if (!_graph.LinkRegions(r0, r1)) {
        _diagnostic = _graph.GetDiagnostic();
        return false;
    }
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
    _PushGraphUndoLocked(graphBefore, "link regions");
    ++_version;
    ++_mapVersion;
    return true;
}

bool
PomadeModel::GraphUnlinkRegions(int r0, int r1)
{
    auto const graphBefore = _GraphUndoStateLocked();
    if (!_graph.UnlinkRegions(r0, r1)) {
        _diagnostic = "PomadeModel::GraphUnlinkRegions: no such link";
        return false;
    }
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
    _PushGraphUndoLocked(graphBefore, "unlink regions");
    ++_version;
    ++_mapVersion;
    return true;
}

PomadeStrokeResult
PomadeModel::GraphStroke(std::vector<PomadeHit> const &samples, float snapRadius,
                        float simplifyEps)
{
    auto const graphBefore = _GraphUndoStateLocked();
    PomadeStrokeResult empty;
    if (!_scalp) {
        _diagnostic = "PomadeModel::GraphStroke: no scalp bound";
        return empty;
    }
    PomadeStrokeResult result =
        PomadeStrokeToChain(*_scalp, &_graph, samples, snapRadius, simplifyEps);
    if (!result.nodeIds.empty()) {
        _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
        _PushGraphUndoLocked(graphBefore, "graph stroke");
        ++_version;
        ++_mapVersion;
    }
    return result;
}

std::vector<std::pair<int, int>>
PomadeModel::GraphMirrorX()
{
    auto const graphBefore = _GraphUndoStateLocked();
    std::vector<std::pair<int, int>> mapping;
    if (!_scalp) {
        _diagnostic = "PomadeModel::GraphMirrorX: no scalp bound";
        return mapping;
    }
    mapping = _graph.MirrorX(*_scalp);
    if (!mapping.empty()) {
        _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
        _PushGraphUndoLocked(graphBefore, "mirror graph");
        ++_version;
        ++_mapVersion;
    }
    return mapping;
}

void
PomadeModel::SetSnapRadius(float radius)
{
    _snapRadius = radius;
}

float
PomadeModel::GetSnapRadius() const
{
    return _snapRadius;
}

void
PomadeModel::SetMirrorX(bool on)
{
    _mirrorX = on;
}

bool
PomadeModel::GetMirrorX() const
{
    return _mirrorX;
}

uint64_t
PomadeModel::GetMapVersion() const
{
    return _mapVersion;
}

bool
PomadeModel::Rasterise()
{
    if (!_scalp) {
        _diagnostic = "PomadeModel::Rasterise: no scalp bound";
        return false;
    }
    std::string err;
    if (!PomadeRasteriseRegionsCpu(*_scalp, _graph, &_maps, &err)) {
        _diagnostic = err;
        return false;
    }
    if (!PomadeFlattenLoops(_graph, &_loops, &err)) {
        _diagnostic = err;
        return false;
    }
    _dirty |= PomadeDirty_Regions;
    // K3 is the gesture-end step (§4.2), and it is the first moment the new
    // region ids have claimed faces, so this is where the L1 tubes are
    // re-attached to the regions a graph edit left behind (plan/17 §5.1,
    // plan/18 §7 G14). A refusal leaves the maps and the tubes untouched
    // and reports through GetDiagnostic; the rasterise itself stands.
    if (!_SyncRegionTubesLocked(nullptr, nullptr)) {
        return false;
    }
    return true;
}

int
PomadeModel::RegionAtSurface(int faceId, float u, float v) const
{
    if (!_scalp || !_scalp->finalized ||
        !PomadeScalpFaceActive(*_scalp, faceId)) {
        return -1;  // a face outside a bound subset is off the scalp
    }
    float p[3] = {0.0f, 0.0f, 0.0f};
    if (!PomadeFacePosition(*_scalp, faceId, u, v, &p[0], &p[1], &p[2])) {
        return -1;
    }
    // Graph edits deliberately defer K3, so do not consult the cached loop
    // or face map here.  Flattening is small and makes a just-drawn closed
    // loop pickable before the gesture's rasterise/publish pass.
    PomadeRegionLoops loops;
    std::string err;
    if (!PomadeFlattenLoops(_graph, &loops, &err)) {
        return -1;
    }
    int best = -1;
    size_t const faceCount = _scalp->faceVertexCounts.size();
    for (size_t r = 0; r < loops.loopCount.size(); ++r) {
        if (!PomadePointInRegionCpu(loops, int(r), p)) {
            continue;
        }
        int const regionId = loops.regionIds[r];
        int seed = -1;
        if (regionId >= 0 && size_t(regionId) < _graph.Regions().size()) {
            seed = _graph.Regions()[size_t(regionId)].seedFace;
        }
        bool connected = (faceId == seed);
        if (!connected && seed >= 0 && size_t(seed) < faceCount) {
            // This is the same chart-local connectivity gate K3 applies to
            // its face claims.  The queried endpoint is admitted from its
            // exact polygon test above; intermediate faces use their
            // centroids, exactly as PomadeRasteriseRegionsCpu does.
            std::vector<char> inside(faceCount, 0);
            for (size_t f = 0; f < faceCount; ++f) {
                float const *c = &_scalp->faceCentroids[f * 3];
                inside[f] = PomadeScalpFaceActive(*_scalp, int(f)) &&
                                    PomadePointInRegionCpu(loops, int(r), c)
                                ? 1
                                : 0;
            }
            inside[size_t(faceId)] = 1;
            std::queue<int> work;
            std::vector<char> seen(faceCount, 0);
            if (inside[size_t(seed)]) {
                work.push(seed);
                seen[size_t(seed)] = 1;
            }
            while (!work.empty() && !connected) {
                int const current = work.front();
                work.pop();
                if (current == faceId) {
                    connected = true;
                    break;
                }
                for (int neighbour : _scalp->faceNeighbours[size_t(current)]) {
                    if (!seen[size_t(neighbour)] && inside[size_t(neighbour)]) {
                        seen[size_t(neighbour)] = 1;
                        work.push(neighbour);
                    }
                }
            }
        }
        if (connected && (best < 0 || regionId < best)) {
            best = regionId;
        }
    }
    return best;
}

void
PomadeModel::NoteBakedMapFile(uint64_t mapVersion, std::string const &path)
{
    _bakedMapVersion = mapVersion;
    _bakedMapFile = path;
}

PomadeModel::GraphSnapshot
PomadeModel::SnapshotGraph() const
{
    GraphSnapshot snapshot;
    snapshot.snapRadius = _snapRadius;
    snapshot.mapVersion = _mapVersion;
    snapshot.bakedMapFile = _bakedMapFile;
    snapshot.bakedMapVersion = _bakedMapVersion;
    snapshot.hasScalp = bool(_scalp) && _scalp->finalized;
    if (!snapshot.hasScalp) {
        return snapshot;
    }
    snapshot.scalpFaceCount = _scalp->faceVertexCounts.size();
    for (auto const &nd : _graph.Nodes()) {
        if (nd.alive) {
            snapshot.nodes.push_back(nd);
        }
    }
    for (auto const &e : _graph.Edges()) {
        if (e.alive) {
            snapshot.edges.emplace_back(e.a, e.b);
        }
    }
    snapshot.linked = _graph.LinkedPairs();
    for (auto const &r : _graph.Regions()) {
        snapshot.regionLoops.push_back(r.loop);
        snapshot.regionBoundaries.push_back(r.boundary);
        snapshot.regionColors.push_back(r.color[0]);
        snapshot.regionColors.push_back(r.color[1]);
        snapshot.regionColors.push_back(r.color[2]);
    }
    // The live primvar is derived, not stored: rasterise here so the
    // snapshot always carries the graph's current map, even when K3 has not
    // run since the last edit (the worker must never read the live model).
    PomadeRegionMaps maps;
    std::string err;
    if (PomadeRasteriseRegionsCpu(*_scalp, _graph, &maps, &err)) {
        snapshot.faceRegions = std::move(maps.faceRegion);
        snapshot.faceRegionIds = std::move(maps.faceRegionId);
        snapshot.uncoveredCount = maps.uncoveredCount;
        snapshot.intersectedCount = maps.intersectedCount;
    }
    return snapshot;
}

bool
PomadeModel::RestoreGraph(GraphSnapshot const &snapshot)
{
    if (!_scalp) {
        _diagnostic = "PomadeModel::RestoreGraph: no scalp bound";
        return false;
    }
    std::string err;
    if (!_graph.Restore(*_scalp, snapshot.nodes, snapshot.edges,
                        snapshot.linked, &err)) {
        _diagnostic = err;
        return false;
    }
    _snapRadius = snapshot.snapRadius;
    if (!PomadeRasteriseRegionsCpu(*_scalp, _graph, &_maps, &err) ||
        !PomadeFlattenLoops(_graph, &_loops, &err)) {
        _diagnostic = err;
        return false;
    }
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
    ++_version;
    ++_mapVersion;
    return true;
}

bool
PomadeModel::_TessellateHost()
{
    if (_useSections) {
        return _TessellateSectionsHost();
    }
    // Legacy cylinder path (the P0/P1 position contract). The K4 frame
    // cache still refreshes so gizmo and fill code reads one source.
    {
        std::string ferr;
        std::vector<PomadeFrame> frames;
        if (PomadeTubeFramesCpu(_BuildTubeDescLocked(), &frames, &ferr)) {
            _frames = std::move(frames);
        }
    }
    int const vertexCount = PomadeTubeVertexCount(_shape);
    int const quadCount = PomadeTubeQuadCount(_shape);
    if (vertexCount <= 0 || quadCount <= 0) {
        _diagnostic = "PomadeModel: tessellate with no tube built";
        return false;
    }
    _host.positions.resize(size_t(vertexCount) * 3);
    _host.normals.resize(size_t(vertexCount) * 3);
    float const *cx = _host.centerX.data();
    float const *cy = _host.centerY.data();
    float const *cz = _host.centerZ.data();
    float minX = 0, minY = 0, minZ = 0, maxX = 0, maxY = 0, maxZ = 0;
    for (int v = 0; v < vertexCount; ++v) {
        PomadeTessellatedVertex const tv =
            PomadeTessellateVertex(_shape, v, cx, cy, cz);
        _host.positions[size_t(v) * 3 + 0] = tv.px;
        _host.positions[size_t(v) * 3 + 1] = tv.py;
        _host.positions[size_t(v) * 3 + 2] = tv.pz;
        _host.normals[size_t(v) * 3 + 0] = tv.nx;
        _host.normals[size_t(v) * 3 + 1] = tv.ny;
        _host.normals[size_t(v) * 3 + 2] = tv.nz;
        if (v == 0) {
            minX = maxX = tv.px;
            minY = maxY = tv.py;
            minZ = maxZ = tv.pz;
        } else {
            minX = std::min(minX, tv.px);
            minY = std::min(minY, tv.py);
            minZ = std::min(minZ, tv.pz);
            maxX = std::max(maxX, tv.px);
            maxY = std::max(maxY, tv.py);
            maxZ = std::max(maxZ, tv.pz);
        }
    }
    _host.extentMin[0] = minX;
    _host.extentMin[1] = minY;
    _host.extentMin[2] = minZ;
    _host.extentMax[0] = maxX;
    _host.extentMax[1] = maxY;
    _host.extentMax[2] = maxZ;
    // Quad strip between adjacent rings, CCW from outside.
    _host.faceVertexCounts.assign(size_t(quadCount), 4);
    _host.faceVertexIndices.resize(size_t(quadCount) * 4);
    int const rv = _shape.ringVerts;
    for (int r = 0; r < _shape.rings - 1; ++r) {
        for (int s = 0; s < rv; ++s) {
            int const q = r * rv + s;
            int const s1 = (s + 1) % rv;
            _host.faceVertexIndices[size_t(q) * 4 + 0] = r * rv + s;
            _host.faceVertexIndices[size_t(q) * 4 + 1] = r * rv + s1;
            _host.faceVertexIndices[size_t(q) * 4 + 2] = (r + 1) * rv + s1;
            _host.faceVertexIndices[size_t(q) * 4 + 3] = (r + 1) * rv + s;
        }
    }
    return true;
}

bool
PomadeModel::_TessellateSectionsHost()
{
    PomadeTubeDesc const tube = _BuildTubeDescLocked();
    if (tube.centerX.size() < 2 || tube.sections.size() < 2) {
        _diagnostic = "PomadeModel: tessellate with no tube built";
        return false;
    }
    std::string err;
    std::vector<PomadeFrame> frames;
    if (!PomadeTubeFramesCpu(tube, &frames, &err)) {
        _diagnostic = err;
        return false;
    }
    _frames = frames;
    std::vector<float> ringT;
    if (!PomadeTessellateCpu(tube, frames, _segmentsPerSpan, &_host.positions,
                            &_host.normals, &ringT, &err)) {
        _diagnostic = err;
        return false;
    }
    int const rv = tube.ringVerts;
    int const nRings = int(_host.positions.size() / size_t(rv) / 3);
    float minX = 0, minY = 0, minZ = 0, maxX = 0, maxY = 0, maxZ = 0;
    size_t const vertexCount = _host.positions.size() / 3;
    for (size_t v = 0; v < vertexCount; ++v) {
        float const px = _host.positions[v * 3 + 0];
        float const py = _host.positions[v * 3 + 1];
        float const pz = _host.positions[v * 3 + 2];
        if (v == 0) {
            minX = maxX = px;
            minY = maxY = py;
            minZ = maxZ = pz;
        } else {
            minX = std::min(minX, px);
            minY = std::min(minY, py);
            minZ = std::min(minZ, pz);
            maxX = std::max(maxX, px);
            maxY = std::max(maxY, py);
            maxZ = std::max(maxZ, pz);
        }
    }
    _host.extentMin[0] = minX;
    _host.extentMin[1] = minY;
    _host.extentMin[2] = minZ;
    _host.extentMax[0] = maxX;
    _host.extentMax[1] = maxY;
    _host.extentMax[2] = maxZ;
    int const quadCount = (nRings - 1) * rv;
    _host.faceVertexCounts.assign(size_t(quadCount), 4);
    _host.faceVertexIndices.resize(size_t(quadCount) * 4);
    for (int r = 0; r < nRings - 1; ++r) {
        for (int s = 0; s < rv; ++s) {
            int const q = r * rv + s;
            int const s1 = (s + 1) % rv;
            _host.faceVertexIndices[size_t(q) * 4 + 0] = r * rv + s;
            _host.faceVertexIndices[size_t(q) * 4 + 1] = r * rv + s1;
            _host.faceVertexIndices[size_t(q) * 4 + 2] = (r + 1) * rv + s1;
            _host.faceVertexIndices[size_t(q) * 4 + 3] = (r + 1) * rv + s;
        }
    }
    return true;
}

bool
PomadeModel::_SyncDevice()
{
#ifdef USDGEN_POMADE_HAS_CUDA
    if (_useSections) {
        return _SyncSectionsDevice();
    }
    if (!_device || !_device->streamOwned || !_device->stream) {
        // No device on this host (GPU-less CI) or dropped by the P6
        // fallback: the host mirror is authoritative and the CUDA mirror
        // simply does not exist. This is the §3.4 CPU fallback, not an
        // error.
        return true;
    }
    if (DeviceFallbackForced()) {
        _DropDeviceLocked("forced by USDGEN_POMADE_FORCE_DEVICE_FALLBACK");
        return true;
    }
    int const vertexCount = PomadeTubeVertexCount(_shape);
    int const rings = _shape.rings;
    if (_device->positions.reset(size_t(vertexCount) * 3) != cudaSuccess ||
        _device->normals.reset(size_t(vertexCount) * 3) != cudaSuccess ||
        _device->centerX.reset(size_t(rings)) != cudaSuccess ||
        _device->centerY.reset(size_t(rings)) != cudaSuccess ||
        _device->centerZ.reset(size_t(rings)) != cudaSuccess) {
        // P6 OOM fallback: the op still succeeds on the host mirror.
        _DropDeviceLocked("PomadeModel: device buffer allocation failed");
        return true;
    }
    if (cudaMemcpyAsync(_device->centerX.data(), _host.centerX.data(),
                        sizeof(float) * size_t(rings),
                        cudaMemcpyHostToDevice,
                        _device->stream) != cudaSuccess ||
        cudaMemcpyAsync(_device->centerY.data(), _host.centerY.data(),
                        sizeof(float) * size_t(rings),
                        cudaMemcpyHostToDevice,
                        _device->stream) != cudaSuccess ||
        cudaMemcpyAsync(_device->centerZ.data(), _host.centerZ.data(),
                        sizeof(float) * size_t(rings),
                        cudaMemcpyHostToDevice,
                        _device->stream) != cudaSuccess) {
        _DropDeviceLocked("PomadeModel: center column upload failed");
        return true;
    }
    char errBuf[256] = {0};
    if (!PomadeLaunchTessellate(_shape,
                               _device->centerX.data(),
                               _device->centerY.data(),
                               _device->centerZ.data(),
                               _device->positions.data(),
                               _device->normals.data(),
                               _device->stream, errBuf, sizeof(errBuf))) {
        std::string const why =
            std::string("PomadeModel: tessellate kernel failed: ") + errBuf;
        _DropDeviceLocked(why.c_str());
        return true;
    }
    if (_device->positions.recordUse(_device->stream) != cudaSuccess ||
        _device->normals.recordUse(_device->stream) != cudaSuccess) {
        _DropDeviceLocked("PomadeModel: device use tracking failed");
        return true;
    }
    (void)std::memset(errBuf, 0, sizeof(errBuf));
#else
    (void)0;
#endif
    return true;
}

#ifdef USDGEN_POMADE_HAS_CUDA
bool
PomadeModel::_SyncSectionsDevice()
{
    if (!_device || !_device->streamOwned || !_device->stream) {
        return true;  // GPU-less CI (or the P6 fallback dropped the
                      // mirror): the host mirror is authoritative.
    }
    if (DeviceFallbackForced()) {
        _DropDeviceLocked("forced by USDGEN_POMADE_FORCE_DEVICE_FALLBACK");
        return true;
    }
    PomadeTubeDesc const tube = _BuildTubeDescLocked();
    int const nCv = int(tube.centerX.size());
    int const nSec = int(tube.sections.size());
    int const rv = tube.ringVerts;
    int const nRings = PomadeTessellatedRingCount(tube, _segmentsPerSpan);
    if (nCv < 2 || nSec < 2 || nRings <= 0) {
        _diagnostic = "PomadeModel: sections device sync with no tube";
        return false;
    }
    static_assert(sizeof(PomadeFrame) == 9 * sizeof(float) &&
                      alignof(PomadeFrame) == alignof(float),
                  "PomadeFrame must stay 9 packed floats");
    if (_device->positions.reset(size_t(nRings) * size_t(rv) * 3) !=
            cudaSuccess ||
        _device->normals.reset(size_t(nRings) * size_t(rv) * 3) !=
            cudaSuccess ||
        _device->centerX.reset(size_t(nCv)) != cudaSuccess ||
        _device->centerY.reset(size_t(nCv)) != cudaSuccess ||
        _device->centerZ.reset(size_t(nCv)) != cudaSuccess ||
        _device->frames.reset(size_t(nCv) * 9) != cudaSuccess ||
        _device->sectionT.reset(size_t(nSec)) != cudaSuccess ||
        _device->sectionU.reset(size_t(nSec) * size_t(rv)) != cudaSuccess ||
        _device->sectionV.reset(size_t(nSec) * size_t(rv)) != cudaSuccess ||
        _device->sectionScale.reset(size_t(nSec)) != cudaSuccess ||
        _device->sectionTwist.reset(size_t(nSec)) != cudaSuccess ||
        _device->rootSurfaceOffsets.reset(size_t(rv) * 3) != cudaSuccess ||
        _device->ringT.reset(size_t(nRings)) != cudaSuccess) {
        _DropDeviceLocked("PomadeModel: device buffer allocation failed");
        return true;
    }
    // Sized by assign: `X(size_t(n))` would parse as a function
    // declaration (the most vexing parse).
    std::vector<float> secT, secU, secV, secS, secTw;
    secT.assign(size_t(nSec), 0.0f);
    secU.assign(size_t(nSec) * size_t(rv), 0.0f);
    secV.assign(size_t(nSec) * size_t(rv), 0.0f);
    secS.assign(size_t(nSec), 0.0f);
    secTw.assign(size_t(nSec), 0.0f);
    for (int i = 0; i < nSec; ++i) {
        PomadeTubeSection const &s = tube.sections[size_t(i)];
        secT[size_t(i)] = s.t;
        secS[size_t(i)] = s.scale;
        secTw[size_t(i)] = s.twist;
        for (int k = 0; k < rv; ++k) {
            secU[size_t(i) * size_t(rv) + size_t(k)] = s.u[size_t(k)];
            secV[size_t(i) * size_t(rv) + size_t(k)] = s.v[size_t(k)];
        }
    }
    cudaStream_t const stream = _device->stream;
    auto upload = [&](void *dst, void const *src, size_t bytes) {
        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                               stream) == cudaSuccess;
    };
    if (!upload(_device->centerX.data(), tube.centerX.data(),
                sizeof(float) * size_t(nCv)) ||
        !upload(_device->centerY.data(), tube.centerY.data(),
                sizeof(float) * size_t(nCv)) ||
        !upload(_device->centerZ.data(), tube.centerZ.data(),
                sizeof(float) * size_t(nCv)) ||
        !upload(_device->sectionT.data(), secT.data(),
                sizeof(float) * size_t(nSec)) ||
        !upload(_device->sectionU.data(), secU.data(),
                sizeof(float) * secU.size()) ||
        !upload(_device->sectionV.data(), secV.data(),
                sizeof(float) * secV.size()) ||
        !upload(_device->sectionScale.data(), secS.data(),
                sizeof(float) * size_t(nSec)) ||
        !upload(_device->sectionTwist.data(), secTw.data(),
                sizeof(float) * size_t(nSec))) {
        _DropDeviceLocked("PomadeModel: tube upload failed");
        return true;
    }
    char errBuf[256] = {0};
    PomadeFrame *frames = reinterpret_cast<PomadeFrame *>(_device->frames.data());
    bool const attached = tube.rootSurfaceOffsets.size() == size_t(rv) * 3;
    if (attached && !upload(_device->rootSurfaceOffsets.data(),
                            tube.rootSurfaceOffsets.data(),
                            sizeof(float) * tube.rootSurfaceOffsets.size())) {
        _DropDeviceLocked("PomadeModel: root attachment upload failed");
        return true;
    }
    if (!PomadeLaunchCenterFrames(_device->centerX.data(),
                                 _device->centerY.data(),
                                 _device->centerZ.data(), nCv, frames, stream,
                                 errBuf, sizeof(errBuf))) {
        std::string const why =
            std::string("PomadeModel: center frames kernel failed: ") + errBuf;
        _DropDeviceLocked(why.c_str());
        return true;
    }
    // K4 is still the CUDA source for legacy identity descriptors. A pinned
    // root or material frame reference changes the CPU chart, so upload the
    // complete matching frame stream before K5 consumes it.
    static std::array<float, 9> const identity = {{
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f}};
    if (tube.rootFramePinned || tube.frameReference != identity) {
        std::vector<PomadeFrame> hostFrames;
        std::string frameErr;
        if (!PomadeTubeFramesCpu(tube, &hostFrames, &frameErr) ||
            int(hostFrames.size()) != nCv ||
            cudaMemcpyAsync(frames, hostFrames.data(),
                            sizeof(PomadeFrame) * size_t(nCv),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            _DropDeviceLocked("PomadeModel: host frame upload failed");
            return true;
        }
    }
    if (!PomadeLaunchTubeTessellate(
            _device->centerX.data(), _device->centerY.data(),
            _device->centerZ.data(), nCv, frames,
            _device->sectionT.data(), _device->sectionU.data(),
            _device->sectionV.data(), _device->sectionScale.data(),
            _device->sectionTwist.data(), nSec, rv, _segmentsPerSpan,
            _device->positions.data(), _device->normals.data(),
            _device->ringT.data(), stream, errBuf, sizeof(errBuf),
            attached ? _device->rootSurfaceOffsets.data() : nullptr)) {
        std::string const why =
            std::string("PomadeModel: tube tessellate kernel failed: ") +
            errBuf;
        _DropDeviceLocked(why.c_str());
        return true;
    }
    if (_device->positions.recordUse(stream) != cudaSuccess ||
        _device->normals.recordUse(stream) != cudaSuccess) {
        _DropDeviceLocked("PomadeModel: device use tracking failed");
        return true;
    }
    return true;
}
#endif

// -- P3: Tube mode --------------------------------------------------------

bool
PomadeModel::_RegionTubeDescLocked(int regionId, int centerCount,
                                  int ringVerts, float length,
                                  PomadeTubeDesc *out, float *outRadius)
{
    if (!out || !outRadius) {
        return false;
    }
    if (!_scalp || !_scalp->finalized) {
        _diagnostic = "PomadeModel::BuildTubeFromRegion: no scalp bound";
        return false;
    }
    if (regionId < 0 || regionId >= _graph.RegionCount() || centerCount < 2 ||
        (ringVerts != 0 && (ringVerts < 3 || ringVerts > 32)) ||
        !(length > 0.0f)) {
        _diagnostic = "PomadeModel::BuildTubeFromRegion: bad region, counts "
                      "or length";
        return false;
    }
    // The closed graph polygon, rather than a coarse face centroid, owns
    // tube placement. A small region entirely inside one mesh face therefore
    // remains a real root support even when that face's centroid lies outside
    // every region.  Snap its polygon centroid back to the scalp so curved
    // charts retain the same on-surface root contract.
    PomadeGraphRegion const &region = _graph.Regions()[size_t(regionId)];
    if (region.boundary.size() < 9) {
        _diagnostic = "PomadeModel::BuildTubeFromRegion: region has no closed "
                      "boundary";
        return false;
    }
    double cx = 0, cy = 0, cz = 0;
    size_t const boundaryCount = region.boundary.size() / 3;
    for (size_t i = 0; i < boundaryCount; ++i) {
        cx += region.boundary[i * 3 + 0];
        cy += region.boundary[i * 3 + 1];
        cz += region.boundary[i * 3 + 2];
    }
    cx /= double(boundaryCount);
    cy /= double(boundaryCount);
    cz /= double(boundaryCount);
    float centroid[3] = {float(cx), float(cy), float(cz)};
    PomadeHit const surface = PomadeClosestPointCpu(*_scalp, centroid);
    if (!surface.hit) {
        _diagnostic = "PomadeModel::BuildTubeFromRegion: region centroid is "
                      "off the scalp";
        return false;
    }
    cx = surface.px;
    cy = surface.py;
    cz = surface.pz;
    // The canonical graph nodes are the authored CVs. Trace samples in
    // `boundary` serve region classification and support placement, but must
    // never replace or angle-sort the polygon the artist drew. Rotate only
    // to its lowest stable node id, preserving its cyclic winding.
    std::vector<PomadeGraphNode const *> corners;
    corners.reserve(region.loop.size());
    for (int nodeId : region.loop) {
        PomadeGraphNode const *node = _graph.FindNode(nodeId);
        if (!node || !node->alive) {
            _diagnostic = "PomadeModel::BuildTubeFromRegion: region loop has "
                          "a missing graph node";
            return false;
        }
        corners.push_back(node);
    }
    if (corners.size() < 3 || corners.size() > 32) {
        _diagnostic = corners.size() > 32
                          ? "PomadeModel::BuildTubeFromRegion: region has "
                            "more than 32 authored corners"
                          : "PomadeModel::BuildTubeFromRegion: region needs "
                            "at least 3 authored corners";
        return false;
    }
    size_t anchor = 0;
    for (size_t i = 1; i < corners.size(); ++i) {
        if (corners[i]->id < corners[anchor]->id) {
            anchor = i;
        }
    }
    std::rotate(corners.begin(), corners.begin() + anchor, corners.end());

    // Keep the support-plane construction shared with Reposition: its
    // Newell fit uses the traced boundary, while the ring below keeps only
    // canonical authored corners projected into that plane.
    float pn[3] = {0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < boundaryCount; ++i) {
        float const *p0 = &region.boundary[i * 3];
        float const *p1 = &region.boundary[((i + 1) % boundaryCount) * 3];
        pn[0] += (p0[1] - p1[1]) * (p0[2] + p1[2]);
        pn[1] += (p0[2] - p1[2]) * (p0[0] + p1[0]);
        pn[2] += (p0[0] - p1[0]) * (p0[1] + p1[1]);
    }
    float const pnl = std::sqrt(pn[0] * pn[0] + pn[1] * pn[1] +
                                pn[2] * pn[2]);
    if (pnl > 1e-12f) {
        pn[0] /= pnl;
        pn[1] /= pnl;
        pn[2] /= pnl;
        if (pn[0] * surface.nx + pn[1] * surface.ny +
                pn[2] * surface.nz <
            0.0f) {
            pn[0] = -pn[0];
            pn[1] = -pn[1];
            pn[2] = -pn[2];
        }
    } else {
        _diagnostic = "PomadeModel::BuildTubeFromRegion: fitted support plane "
                      "is degenerate";
        return false;
    }

    // The first stable corner supplies the in-plane axis. This makes the
    // authored seam deterministic without inventing a polar ordering.
    float axisU[3] = {0.0f, 0.0f, 0.0f};
    for (PomadeGraphNode const *corner : corners) {
        float d[3] = {corner->p[0] - float(cx), corner->p[1] - float(cy),
                      corner->p[2] - float(cz)};
        float const along = PomadeDot3(d, pn);
        d[0] -= along * pn[0];
        d[1] -= along * pn[1];
        d[2] -= along * pn[2];
        float const len = PomadeLen3(d);
        if (len > 1e-6f) {
            axisU[0] = d[0] / len;
            axisU[1] = d[1] / len;
            axisU[2] = d[2] / len;
            break;
        }
    }
    if (!(PomadeLen3(axisU) > 1e-6f)) {
        _diagnostic = "PomadeModel::BuildTubeFromRegion: authored corners "
                      "collapse in their fitted plane";
        return false;
    }
    float axisV[3];
    PomadeCross3(pn, axisU, axisV);

    struct _RingPoint { float u, v; };
    std::vector<_RingPoint> ring;
    ring.reserve(32);
    for (PomadeGraphNode const *corner : corners) {
        float const d[3] = {corner->p[0] - float(cx),
                            corner->p[1] - float(cy),
                            corner->p[2] - float(cz)};
        ring.push_back(_RingPoint{PomadeDot3(d, axisU),
                                  PomadeDot3(d, axisV)});
    }
    double twiceArea = 0.0;
    for (size_t i = 0; i < ring.size(); ++i) {
        _RingPoint const &a = ring[i];
        _RingPoint const &b = ring[(i + 1) % ring.size()];
        float const du = b.u - a.u, dv = b.v - a.v;
        if (du * du + dv * dv <= 1e-12f) {
            _diagnostic = "PomadeModel::BuildTubeFromRegion: adjacent "
                          "authored corners collapse in the fitted plane";
            return false;
        }
        twiceArea += double(a.u) * b.v - double(a.v) * b.u;
    }
    if (std::fabs(twiceArea) <= 1e-10) {
        _diagnostic = "PomadeModel::BuildTubeFromRegion: authored corner "
                      "polygon has zero fitted-plane area";
        return false;
    }
    if (twiceArea < 0.0) {
        // Keep the min-id seam while orienting the K5 ring consistently with
        // the support normal. Graph extraction normally already has this
        // winding; this protects a legacy/reversed loop without angle sort.
        std::reverse(ring.begin() + 1, ring.end());
    }
    int const resolvedRingVerts =
        std::max(ringVerts == 0 ? int(corners.size()) : ringVerts,
                 int(corners.size()));
    if (resolvedRingVerts > 32) {
        _diagnostic = "PomadeModel::BuildTubeFromRegion: region needs more "
                      "than 32 section CVs";
        return false;
    }
    // Extra slots only split existing edges. The longest projected edge wins
    // (lowest cyclic edge index breaks ties), so every authored corner stays
    // verbatim and the interpolation is deterministic.
    while (int(ring.size()) < resolvedRingVerts) {
        size_t best = 0;
        float bestLength2 = -1.0f;
        for (size_t i = 0; i < ring.size(); ++i) {
            _RingPoint const &a = ring[i];
            _RingPoint const &b = ring[(i + 1) % ring.size()];
            float const du = b.u - a.u, dv = b.v - a.v;
            float const length2 = du * du + dv * dv;
            if (length2 > bestLength2) {
                bestLength2 = length2;
                best = i;
            }
        }
        _RingPoint const &a = ring[best];
        _RingPoint const &b = ring[(best + 1) % ring.size()];
        ring.insert(ring.begin() +
                        static_cast<std::vector<_RingPoint>::difference_type>(
                            best + 1),
                    _RingPoint{0.5f * (a.u + b.u),
                               0.5f * (a.v + b.v)});
    }

    out->centerX.assign(size_t(centerCount), 0.0f);
    out->centerY.assign(size_t(centerCount), 0.0f);
    out->centerZ.assign(size_t(centerCount), 0.0f);
    for (int i = 0; i < centerCount; ++i) {
        float const s = float(i) / float(centerCount - 1);
        out->centerX[size_t(i)] = float(cx) + pn[0] * length * s;
        out->centerY[size_t(i)] = float(cy) + pn[1] * length * s;
        out->centerZ[size_t(i)] = float(cz) + pn[2] * length * s;
    }
    // Build Q from a reference tangent with three distinct magnitudes. A
    // local +Y spine would tie PomadePerp3's X/Z least-axis choice; after a
    // float Q^T round-trip that can rotate K4 by 90 degrees above the pinned
    // root. This stable material basis maps the exact K4 tuple (t,n,b) onto
    // (plane normal, polygon U, polygon V), aligning every spine section.
    float localT[3] = {1.0f, 2.0f, 3.0f};
    float const localLength = PomadeLen3(localT);
    localT[0] /= localLength;
    localT[1] /= localLength;
    localT[2] /= localLength;
    float localN[3];
    PomadePerp3(localT, localN);
    float localB[3];
    PomadeCross3(localT, localN, localB);
    for (int row = 0; row < 3; ++row) {
        float const worldT = pn[row];
        float const worldN = axisU[row];
        float const worldB = axisV[row];
        out->frameReference[size_t(row * 3 + 0)] =
            worldT * localT[0] + worldN * localN[0] + worldB * localB[0];
        out->frameReference[size_t(row * 3 + 1)] =
            worldT * localT[1] + worldN * localN[1] + worldB * localB[1];
        out->frameReference[size_t(row * 3 + 2)] =
            worldT * localT[2] + worldN * localN[2] + worldB * localB[2];
    }
    out->rootFramePinned = true;
    out->rootFrame.tx = pn[0];
    out->rootFrame.ty = pn[1];
    out->rootFrame.tz = pn[2];
    out->rootFrame.nx = axisU[0];
    out->rootFrame.ny = axisU[1];
    out->rootFrame.nz = axisU[2];
    out->rootFrame.bx = axisV[0];
    out->rootFrame.by = axisV[1];
    out->rootFrame.bz = axisV[2];
    out->ringVerts = resolvedRingVerts;
    out->regionId = regionId;
    out->level = 1;
    out->parentTubeId = -1;
    out->childIndex = -1;
    out->sections.resize(size_t(centerCount));
    for (int r = 0; r < centerCount; ++r) {
        PomadeTubeSection &section = out->sections[size_t(r)];
        section.t = float(r) / float(centerCount - 1);
        section.scale = 1.0f;
        section.twist = 0.0f;
        section.u.resize(ring.size());
        section.v.resize(ring.size());
        for (size_t s = 0; s < ring.size(); ++s) {
            section.u[s] = ring[s].u;
            section.v[s] = ring[s].v;
        }
    }
    double mean = 0.0;
    for (_RingPoint const &point : ring) {
        mean += std::sqrt(double(point.u) * point.u +
                          double(point.v) * point.v);
    }
    mean /= double(ring.size());
    _diagnostic.clear();
    *outRadius = float(mean);
    return true;
}

bool
PomadeModel::PinRegionRootFrames()
{
    if (!_scalp || !_scalp->finalized) {
        _diagnostic = "PomadeModel::PinRegionRootFrames: no scalp bound";
        return false;
    }
    auto pin = [&](PomadeTubeDesc *desc) {
        if (!desc || desc->regionId < 0 ||
            desc->regionId >= int(_graph.Regions().size()) ||
            desc->centerX.size() < 2) {
            return false;
        }
        PomadeGraphRegion const &region =
            _graph.Regions()[size_t(desc->regionId)];
        size_t const count = region.boundary.size() / 3;
        if (count < 3) {
            return false;
        }
        float pn[3] = {0.0f, 0.0f, 0.0f};
        for (size_t i = 0; i < count; ++i) {
            float const *a = &region.boundary[i * 3];
            float const *b = &region.boundary[((i + 1) % count) * 3];
            pn[0] += (a[1] - b[1]) * (a[2] + b[2]);
            pn[1] += (a[2] - b[2]) * (a[0] + b[0]);
            pn[2] += (a[0] - b[0]) * (a[1] + b[1]);
        }
        float const nl = std::sqrt(pn[0] * pn[0] + pn[1] * pn[1] +
                                   pn[2] * pn[2]);
        if (!(nl > 1e-12f)) {
            return false;
        }
        pn[0] /= nl;
        pn[1] /= nl;
        pn[2] /= nl;
        float root[3] = {desc->centerX[0], desc->centerY[0],
                         desc->centerZ[0]};
        PomadeHit const support = PomadeClosestPointCpu(*_scalp, root);
        if (support.hit && pn[0] * support.nx + pn[1] * support.ny +
                               pn[2] * support.nz <
                               0.0f) {
            pn[0] = -pn[0];
            pn[1] = -pn[1];
            pn[2] = -pn[2];
        }
        std::vector<PomadeFrame> raw;
        std::string err;
        if (!PomadeTubeFramesCpu(*desc, &raw, &err) ||
            raw.empty()) {
            return false;
        }
        PomadeFrame frame = raw.front();
        frame.tx = pn[0];
        frame.ty = pn[1];
        frame.tz = pn[2];
        float const dot = frame.nx * pn[0] + frame.ny * pn[1] +
                          frame.nz * pn[2];
        frame.nx -= dot * pn[0];
        frame.ny -= dot * pn[1];
        frame.nz -= dot * pn[2];
        float const il = std::sqrt(frame.nx * frame.nx +
                                   frame.ny * frame.ny +
                                   frame.nz * frame.nz);
        if (il > 1e-12f) {
            frame.nx /= il;
            frame.ny /= il;
            frame.nz /= il;
        } else {
            PomadePerp3(pn, &frame.nx);
        }
        PomadeCross3(pn, &frame.nx, &frame.bx);
        desc->rootFramePinned = true;
        desc->rootFrame = frame;
        return true;
    };

    bool any = false;
    if (_host.centerX.size() >= 2 && _tubeRegionId >= 0) {
        PomadeTubeDesc tube0 = _BuildTubeDescLocked();
        if (!pin(&tube0)) {
            _diagnostic = "PomadeModel::PinRegionRootFrames: bad tube-0 region";
            return false;
        }
        _rootFramePinned = tube0.rootFramePinned;
        _rootFrame = tube0.rootFrame;
        std::string err;
        if (!PomadeTubeFramesCpu(tube0, &_frames, &err)) {
            _diagnostic = err;
            return false;
        }
        any = true;
    }
    for (auto &kv : _tubes) {
        PomadeTubeDesc &desc = kv.second.actual;
        if (desc.parentTubeId >= 0 || desc.regionId < 0) {
            continue;
        }
        if (!pin(&desc)) {
            _diagnostic = "PomadeModel::PinRegionRootFrames: bad L1 region";
            return false;
        }
        // L1's derived copy is the live root baseline; descendants already
        // carry their own parent-relative geometry and are not re-derived.
        kv.second.derived.rootFramePinned = desc.rootFramePinned;
        kv.second.derived.rootFrame = desc.rootFrame;
        kv.second.derived.frameReference = desc.frameReference;
        any = true;
    }
    _diagnostic.clear();
    // A legacy groom may legitimately contain no region-rooted L1 tube.
    // Treat that as a successful no-op so hydration remains compatible.
    (void)any;
    return true;
}

bool
PomadeModel::_InstallTube0DescLocked(PomadeTubeDesc const &desc)
{
    _shape.rings = int(desc.centerX.size());
    _shape.ringVerts = desc.ringVerts;
    _host.centerX = desc.centerX;
    _host.centerY = desc.centerY;
    _host.centerZ = desc.centerZ;
    _sections = desc.sections;
    _useSections = true;  // authored tubes always take the K5 path
    _tubeRegionId = desc.regionId;
    _rootFramePinned = desc.rootFramePinned;
    _rootFrame = desc.rootFrame;
    _frameReference = desc.frameReference;
    _roots.clear();
    _tubeRoots.clear();
    _guides = PomadeGuideSet();
    ++_guideVersion;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Topology | PomadeDirty_Points | PomadeDirty_Guides;
    return true;
}

int
PomadeModel::_NextL1TubeIdLocked() const
{
    for (int id = 16;; id += 16) {
        if (_tubes.find(id) == _tubes.end()) {
            return id;
        }
    }
}

bool
PomadeModel::BuildTubeFromRegion(int regionId, int centerCount, int ringVerts,
                                float length)
{
    if (ringVerts != 0 && (ringVerts < 3 || ringVerts > 32)) {
        _diagnostic = "PomadeModel::BuildTubeFromRegion: ring CV count is "
                      "0 (match authored region corners) or 3..32";
        return false;
    }
    PomadeTubeDesc desc;
    float fitted = 0.0f;
    if (!_RegionTubeDescLocked(regionId, centerCount, ringVerts, length,
                               &desc, &fitted)) {
        return false;
    }
    // Which tube owns this region? An existing owner is refreshed in
    // place, the first build takes tube 0, and every later region appends
    // a new L1 root.
    int target = -1;
    auto const owner = _regionTube.find(regionId);
    if (owner != _regionTube.end()) {
        target = owner->second;
    } else if (_host.centerX.size() < 2) {
        target = 0;  // the model has no first L1 tube yet
    } else if (_tubeRegionId < 0 || _tubeRegionId == regionId) {
        // The first L1 tube is unrooted (a test tube, or a groom hydrated
        // before the map existed), or it already owns this region.
        target = 0;
    } else {
        target = _NextL1TubeIdLocked();
    }
    if (target != 0 && _SubtreeCarriesDeltasLocked(target)) {
        _diagnostic = "PomadeModel::BuildTubeFromRegion: the region's tube "
                      "carries child deltas; merge the children first";
        return false;
    }
    desc.tubeId = target;
    HierarchyRollback const snap = _SnapshotHierarchyLocked();
    bool ok = true;
    if (target == 0) {
        _shape.length = length;
        _shape.radius = fitted;
        ok = _InstallTube0DescLocked(desc);
    } else {
        auto it = _tubes.find(target);
        if (it == _tubes.end()) {
            HierarchyTube entry;
            entry.actual = desc;
            entry.derived = desc;
            PomadeClearDeltas(desc, &entry.deltas);
            entry.fill = _fill;
            _tubes[target] = entry;
        } else {
            it->second.actual = desc;
            it->second.derived = desc;
            PomadeClearDeltas(desc, &it->second.deltas);
        }
        _dirty |= PomadeDirty_Topology | PomadeDirty_Points;
        ok = _PropagateDownLocked(target);
    }
    if (!ok) {
        _RestoreHierarchyLocked(snap);
        return false;
    }
    _PushUndoSnapshotLocked(snap);
    _regionTube[regionId] = target;
    _tubeRegionKey[target] = _RegionKeyLocked(regionId);
    ++_version;
    ++_mapVersion;  // a new L1 root changes the level-0 channel layout
    return true;
}

int
PomadeModel::TubeForRegion(int regionId) const
{
    auto const it = _regionTube.find(regionId);
    if (it != _regionTube.end()) {
        return it->second;
    }
    if (_tubeRegionId == regionId && _host.centerX.size() >= 2) {
        return 0;
    }
    return -1;
}

int
PomadeModel::RegionForTube(int tubeId) const
{
    PomadeTubeDesc desc;
    if (!_TubeDescLocked(tubeId, &desc)) {
        return -1;
    }
    return desc.regionId;
}

std::vector<int>
PomadeModel::L1TubeIds() const
{
    return _L1TubeIdsLocked();
}

std::vector<int>
PomadeModel::_L1TubeIdsLocked() const
{
    std::vector<int> out;
    if (_host.centerX.size() >= 2) {
        out.push_back(0);
    }
    for (auto const &kv : _tubes) {
        if (kv.first > 0 && kv.second.actual.parentTubeId < 0 &&
            kv.second.members.empty() && !kv.second.transientParent) {
            out.push_back(kv.first);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<int>
PomadeModel::_RegionKeyLocked(int regionId) const
{
    std::vector<int> key;
    if (regionId < 0 || regionId >= int(_graph.Regions().size())) {
        return key;
    }
    key = _graph.Regions()[size_t(regionId)].loop;
    std::sort(key.begin(), key.end());
    key.erase(std::unique(key.begin(), key.end()), key.end());
    return key;
}

bool
PomadeModel::_SubtreeCarriesDeltasLocked(int tubeId) const
{
    for (auto const &kv : _tubes) {
        if (kv.second.actual.parentTubeId != tubeId) {
            continue;
        }
        if (kv.second.imported) {
            return true;
        }
        for (float d : kv.second.deltas.centerDu) {
            if (std::fabs(d) > 1e-7f) {
                return true;
            }
        }
        for (float d : kv.second.deltas.centerDv) {
            if (std::fabs(d) > 1e-7f) {
                return true;
            }
        }
        for (float d : kv.second.deltas.centerDw) {
            if (std::fabs(d) > 1e-7f) {
                return true;
            }
        }
        for (auto const &sec : kv.second.deltas.sections) {
            for (float d : sec.u) {
                if (std::fabs(d) > 1e-7f) {
                    return true;
                }
            }
            for (float d : sec.v) {
                if (std::fabs(d) > 1e-7f) {
                    return true;
                }
            }
        }
        if (_SubtreeCarriesDeltasLocked(kv.first)) {
            return true;
        }
    }
    return false;
}

void
PomadeModel::_DropSubtreeLocked(int tubeId)
{
    if (tubeId < 0) {
        // Group parents carry negative ids and own their members through
        // `members`, not parentTubeId; their own parentTubeId is -1, the
        // same as every L1 root's. Expanding -1 as a parent id collected
        // the group itself plus every root and recursed forever (Delete
        // on the first group of a session). A group has no subtree.
        _tubes.erase(tubeId);
        _tubeRegionKey.erase(tubeId);
        _PruneActiveCutLocked();
        return;
    }
    std::vector<int> kids;
    for (auto const &kv : _tubes) {
        if (kv.second.actual.parentTubeId == tubeId) {
            kids.push_back(kv.first);
        }
    }
    for (int k : kids) {
        _DropSubtreeLocked(k);
    }
    _tubes.erase(tubeId);
    _tubeRegionKey.erase(tubeId);
    _PruneActiveCutLocked();
}

bool
PomadeModel::_RerootTubeLocked(int tubeId, int regionId)
{
    PomadeTubeDesc old;
    if (!_TubeDescLocked(tubeId, &old)) {
        return false;
    }
    float length = 0.0f;
    {
        size_t const n = old.centerX.size();
        float const dx = old.centerX[n - 1] - old.centerX[0];
        float const dy = old.centerY[n - 1] - old.centerY[0];
        float const dz = old.centerZ[n - 1] - old.centerZ[0];
        length = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    if (!(length > 1e-6f)) {
        length = 1.0f;
    }
    PomadeTubeDesc desc;
    float fitted = 0.0f;
    if (!_RegionTubeDescLocked(regionId, int(old.centerX.size()),
                               old.ringVerts, length, &desc, &fitted)) {
        return false;
    }
    desc.tubeId = tubeId;
    if (tubeId == 0) {
        _shape.length = length;
        _shape.radius = fitted;
        return _InstallTube0DescLocked(desc);
    }
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end()) {
        return false;
    }
    it->second.actual = desc;
    it->second.derived = desc;
    PomadeClearDeltas(desc, &it->second.deltas);
    _dirty |= PomadeDirty_Topology | PomadeDirty_Points;
    return _PropagateDownLocked(tubeId);
}

bool
PomadeModel::_SyncRegionTubesLocked(std::vector<int> *outRebuilt,
                                   std::vector<int> *outRemoved)
{
    if (outRebuilt) {
        outRebuilt->clear();
    }
    if (outRemoved) {
        outRemoved->clear();
    }
    // Every L1 tube that is rooted in a region, with the node loop it was
    // rooted in. A tube with no key (an unrooted disc tube, or one built
    // before a scalp was bound) is left alone.
    std::vector<int> const l1 = _L1TubeIdsLocked();
    std::map<int, std::vector<int>> keys;
    for (int id : l1) {
        auto const it = _tubeRegionKey.find(id);
        if (it != _tubeRegionKey.end() && !it->second.empty()) {
            keys[id] = it->second;
        }
    }
    if (keys.empty()) {
        _regionTube.clear();
        for (int id : l1) {
            PomadeTubeDesc desc;
            if (_TubeDescLocked(id, &desc) && desc.regionId >= 0 &&
                desc.regionId < _graph.RegionCount()) {
                _regionTube[desc.regionId] = id;
            }
        }
        return true;
    }
    int const regionCount = _graph.RegionCount();
    std::vector<std::vector<int>> current(size_t(std::max(regionCount, 0)));
    for (int r = 0; r < regionCount; ++r) {
        current[size_t(r)] = _RegionKeyLocked(r);
    }
    // Pass 1: exact loop matches keep their tube (a pure renumbering).
    std::map<int, int> claim;  // region -> tube
    std::map<int, int> placed;  // tube -> region
    for (int r = 0; r < regionCount; ++r) {
        for (auto const &kv : keys) {
            if (placed.count(kv.first) || claim.count(r)) {
                continue;
            }
            if (kv.second == current[size_t(r)]) {
                claim[r] = kv.first;
                placed[kv.first] = r;
            }
        }
    }
    // Pass 2: the rest go to the unclaimed region they share most nodes
    // with. That covers both directions of §5.1: a split region hands its
    // tube to the larger half, and a merge hands the merged region to the
    // tube that contributed more of its loop.
    std::vector<int> rebuilt, removed;
    for (auto const &kv : keys) {
        if (placed.count(kv.first)) {
            continue;
        }
        int best = -1;
        size_t bestShare = 0;
        for (int r = 0; r < regionCount; ++r) {
            if (claim.count(r)) {
                continue;
            }
            size_t share = 0;
            for (int node : current[size_t(r)]) {
                if (std::binary_search(kv.second.begin(), kv.second.end(),
                                       node)) {
                    ++share;
                }
            }
            if (share > bestShare) {
                bestShare = share;
                best = r;
            }
        }
        if (best >= 0 && bestShare >= 2) {
            claim[best] = kv.first;
            placed[kv.first] = best;
            rebuilt.push_back(kv.first);
        } else {
            removed.push_back(kv.first);
        }
    }
    // A re-keyed tube keeps its shape (the loop matched exactly) but its
    // stored region id would go stale: the committer reads the stored id,
    // so it must follow the claim. Rebuilt tubes already carry the new id
    // from their re-root; re-keying them again is a no-op.
    auto rekey = [&](int tubeId, int region) {
        if (tubeId == 0) {
            _tubeRegionId = region;
            return;
        }
        auto tubeIt = _tubes.find(tubeId);
        if (tubeIt != _tubes.end()) {
            tubeIt->second.actual.regionId = region;
            tubeIt->second.derived.regionId = region;
        }
    };
    if (rebuilt.empty() && removed.empty()) {
        _regionTube.clear();
        for (auto const &kv : claim) {
            _regionTube[kv.first] = kv.second;
            _tubeRegionKey[kv.second] = current[size_t(kv.first)];
            rekey(kv.second, kv.first);
        }
        return true;
    }
    // §5.1's refusal: re-deriving or dropping a tube whose children carry
    // authored deltas would throw that work away.
    for (int id : rebuilt) {
        if (_SubtreeCarriesDeltasLocked(id)) {
            _diagnostic = "PomadeModel::SyncRegionTubes: the graph edit "
                          "re-roots a tube whose children carry deltas; "
                          "merge the children first";
            return false;
        }
    }
    for (int id : removed) {
        if (id == 0 || _SubtreeCarriesDeltasLocked(id)) {
            _diagnostic = "PomadeModel::SyncRegionTubes: the graph edit "
                          "removes a tube that carries child deltas (or the "
                          "first L1 tube); merge the children first";
            return false;
        }
    }
    HierarchyRollback const snap = _SnapshotHierarchyLocked();
    for (int id : rebuilt) {
        if (!_RerootTubeLocked(id, placed[id])) {
            _RestoreHierarchyLocked(snap);
            return false;
        }
    }
    for (int id : removed) {
        _DropSubtreeLocked(id);
    }
    _regionTube.clear();
    for (auto const &kv : claim) {
        _regionTube[kv.first] = kv.second;
        _tubeRegionKey[kv.second] = current[size_t(kv.first)];
        rekey(kv.second, kv.first);
    }
    _dirty |= PomadeDirty_Topology | PomadeDirty_Points;
    if (outRebuilt) {
        *outRebuilt = rebuilt;
    }
    if (outRemoved) {
        *outRemoved = removed;
    }
    return true;
}

bool
PomadeModel::SyncRegionTubes(std::vector<int> *outRebuilt,
                            std::vector<int> *outRemoved)
{
    return _SyncRegionTubesLocked(outRebuilt, outRemoved);
}

bool
PomadeModel::InstallL1Tube(int tubeId, TubeRecord const &record)
{
    if (tubeId <= 0 || (tubeId % 16) != 0) {
        _diagnostic = "PomadeModel::InstallL1Tube: an L1 root id is a "
                      "positive multiple of 16";
        return false;
    }
    if (_tubes.count(tubeId)) {
        _diagnostic = "PomadeModel::InstallL1Tube: the id is taken";
        return false;
    }
    if (record.actual.centerX.size() < 2 ||
        record.actual.sections.size() < 2) {
        _diagnostic = "PomadeModel::InstallL1Tube: degenerate shape";
        return false;
    }
    HierarchyTube entry;
    entry.actual = record.actual;
    entry.actual.tubeId = tubeId;
    entry.actual.level = 1;
    entry.actual.parentTubeId = -1;
    entry.actual.childIndex = -1;
    entry.derived = entry.actual;
    PomadeClearDeltas(entry.actual, &entry.deltas);
    entry.subdivide.count = record.subdivide.count;
    entry.subdivide.seed = record.subdivide.seed;
    entry.subdivide.splitMode = record.subdivide.splitMode == "edge"
                                    ? PomadeSplit_Edge
                                    : PomadeSplit_KMeans;
    entry.fill = record.fill;
    entry.lockParents = record.lockParents;
    entry.lockChildren = record.lockChildren;
    _tubes[tubeId] = entry;
    if (entry.actual.regionId >= 0) {
        _regionTube[entry.actual.regionId] = tubeId;
        _tubeRegionKey[tubeId] = _RegionKeyLocked(entry.actual.regionId);
    }
    ++_version;
    ++_mapVersion;
    _dirty |= PomadeDirty_Topology | PomadeDirty_Points;
    return true;
}

bool
PomadeModel::MoveCenterCV(int cv, float dx, float dy, float dz)
{
    int const n = int(_host.centerX.size());
    if (cv < 0 || cv >= n || n < 2) {
        _diagnostic = "PomadeModel::MoveCenterCV: CV index out of range";
        return false;
    }
    _PushUndoLocked();
    float const centerT = float(cv) / float(n - 1);
    for (int i = 0; i < n; ++i) {
        float const t = float(i) / float(n - 1);
        float const w = PomadeSoftWeight(t, centerT, _softRadius);
        if (w <= 0.0f && i != cv) {
            continue;
        }
        float const k = (_softRadius > 0.0f) ? w : (i == cv ? 1.0f : 0.0f);
        _host.centerX[size_t(i)] += dx * k;
        _host.centerY[size_t(i)] += dy * k;
        _host.centerZ[size_t(i)] += dz * k;
    }
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::InsertCenterCV(int atIndex)
{
    int const n = int(_host.centerX.size());
    if (atIndex < 0 || atIndex > n || n < 2) {
        _diagnostic = "PomadeModel::InsertCenterCV: index out of range";
        return false;
    }
    // Insert at the Catmull-Rom midpoint of the bracketing span (linear
    // midpoint at the ends).
    int i0 = atIndex - 1, i1 = atIndex;
    if (i0 < 0) {
        i0 = 0;
        i1 = 1;
    } else if (i1 >= n) {
        i0 = n - 2;
        i1 = n - 1;
    }
    float const mx = 0.5f * (_host.centerX[size_t(i0)] +
                             _host.centerX[size_t(i1)]);
    float const my = 0.5f * (_host.centerY[size_t(i0)] +
                             _host.centerY[size_t(i1)]);
    float const mz = 0.5f * (_host.centerZ[size_t(i0)] +
                             _host.centerZ[size_t(i1)]);
    _PushUndoLocked();
    _host.centerX.insert(_host.centerX.begin() + atIndex, mx);
    _host.centerY.insert(_host.centerY.begin() + atIndex, my);
    _host.centerZ.insert(_host.centerZ.begin() + atIndex, mz);
    _shape.rings = int(_host.centerX.size());
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Topology | PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::DeleteCenterCV(int index)
{
    int const n = int(_host.centerX.size());
    if (index < 0 || index >= n || n <= 2) {
        _diagnostic = "PomadeModel::DeleteCenterCV: need >= 2 CVs left";
        return false;
    }
    _PushUndoLocked();
    _host.centerX.erase(_host.centerX.begin() + index);
    _host.centerY.erase(_host.centerY.begin() + index);
    _host.centerZ.erase(_host.centerZ.begin() + index);
    _shape.rings = int(_host.centerX.size());
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Topology | PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::SetTubeLength(float length)
{
    int const n = int(_host.centerX.size());
    if (!(length > 0.0f) || n < 2) {
        _diagnostic = "PomadeModel::SetTubeLength: need a tube + length > 0";
        return false;
    }
    double total = 0;
    for (int i = 1; i < n; ++i) {
        double dx = double(_host.centerX[size_t(i)] -
                           _host.centerX[size_t(i - 1)]);
        double dy = double(_host.centerY[size_t(i)] -
                           _host.centerY[size_t(i - 1)]);
        double dz = double(_host.centerZ[size_t(i)] -
                           _host.centerZ[size_t(i - 1)]);
        total += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    if (!(total > 1e-12)) {
        _diagnostic = "PomadeModel::SetTubeLength: degenerate center";
        return false;
    }
    double const k = double(length) / total;
    _PushUndoLocked();
    for (int i = 1; i < n; ++i) {
        _host.centerX[size_t(i)] =
            _host.centerX[0] +
            float((double(_host.centerX[size_t(i)]) -
                   double(_host.centerX[0])) *
                  k);
        _host.centerY[size_t(i)] =
            _host.centerY[0] +
            float((double(_host.centerY[size_t(i)]) -
                   double(_host.centerY[0])) *
                  k);
        _host.centerZ[size_t(i)] =
            _host.centerZ[0] +
            float((double(_host.centerZ[size_t(i)]) -
                   double(_host.centerZ[0])) *
                  k);
    }
    _shape.length = length;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::MatchSurface()
{
    if (!_scalp || !_scalp->finalized || _host.centerX.empty()) {
        _diagnostic = "PomadeModel::MatchSurface: need a bound scalp + tube";
        return false;
    }
    float const p[3] = {_host.centerX[0], _host.centerY[0], _host.centerZ[0]};
    PomadeHit const hit = PomadeClosestPointCpu(*_scalp, p);
    if (!hit.hit) {
        _diagnostic = "PomadeModel::MatchSurface: no surface point";
        return false;
    }
    _PushUndoLocked();
    _host.centerX[0] = hit.px;
    _host.centerY[0] = hit.py;
    _host.centerZ[0] = hit.pz;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Points;
    ++_version;
    return true;
}

int
PomadeModel::GetCenterCVCount() const
{
    return int(_host.centerX.size());
}

bool
PomadeModel::GetCenterCV(int cv, float *x, float *y, float *z) const
{
    if (cv < 0 || cv >= int(_host.centerX.size()) || !x || !y || !z) {
        return false;
    }
    *x = _host.centerX[size_t(cv)];
    *y = _host.centerY[size_t(cv)];
    *z = _host.centerZ[size_t(cv)];
    return true;
}

int
PomadeModel::GetSectionCount() const
{
    return int(_sections.size());
}

bool
PomadeModel::GetSection(int ring, PomadeTubeSection *section) const
{
    if (ring < 0 || ring >= int(_sections.size()) || !section) {
        return false;
    }
    *section = _sections[size_t(ring)];
    return true;
}

bool
PomadeModel::MoveSectionRing(int ring, float du, float dv)
{
    if (ring < 0 || ring >= int(_sections.size())) {
        _diagnostic = "PomadeModel::MoveSectionRing: ring out of range";
        return false;
    }
    _PushUndoLocked();
    PomadeTubeSection &s = _sections[size_t(ring)];
    for (size_t i = 0; i < s.u.size(); ++i) {
        s.u[i] += du;
        s.v[i] += dv;
    }
    _useSections = true;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::ScaleSectionRing(int ring, float scale)
{
    if (ring < 0 || ring >= int(_sections.size()) || !(scale > 0.0f)) {
        _diagnostic = "PomadeModel::ScaleSectionRing: bad ring or scale";
        return false;
    }
    _PushUndoLocked();
    _sections[size_t(ring)].scale *= scale;
    _useSections = true;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::TwistSectionRing(int ring, float radians)
{
    if (ring < 0 || ring >= int(_sections.size())) {
        _diagnostic = "PomadeModel::TwistSectionRing: ring out of range";
        return false;
    }
    _PushUndoLocked();
    _sections[size_t(ring)].twist += radians;
    _useSections = true;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::MoveSectionCV(int ring, int slot, float du, float dv)
{
    if (ring < 0 || ring >= int(_sections.size()) || slot < 0 ||
        slot >= int(_sections[size_t(ring)].u.size())) {
        _diagnostic = "PomadeModel::MoveSectionCV: ring/slot out of range";
        return false;
    }
    _PushUndoLocked();
    PomadeTubeSection &s = _sections[size_t(ring)];
    s.u[size_t(slot)] += du;
    s.v[size_t(slot)] += dv;
    _useSections = true;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::AddSectionRing(float t)
{
    int const nSec = int(_sections.size());
    if (nSec < 2) {
        _diagnostic = "PomadeModel::AddSectionRing: no tube";
        return false;
    }
    float const t0 = _sections.front().t;
    float const t1 = _sections.back().t;
    if (!(t > t0) || !(t < t1)) {
        _diagnostic = "PomadeModel::AddSectionRing: t must sit strictly "
                      "inside the section range";
        return false;
    }
    int k = 0;
    while (k + 1 < nSec - 1 && _sections[size_t(k + 1)].t < t) {
        ++k;
    }
    PomadeTubeSection const &s0 = _sections[size_t(k)];
    PomadeTubeSection const &s1 = _sections[size_t(k + 1)];
    float f = (s1.t > s0.t) ? (t - s0.t) / (s1.t - s0.t) : 0.0f;
    f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
    PomadeTubeSection added;
    added.t = t;
    added.u.resize(s0.u.size());
    added.v.resize(s0.v.size());
    for (size_t i = 0; i < s0.u.size(); ++i) {
        added.u[i] = s0.u[i] + (s1.u[i] - s0.u[i]) * f;
        added.v[i] = s0.v[i] + (s1.v[i] - s0.v[i]) * f;
    }
    added.scale = s0.scale + (s1.scale - s0.scale) * f;
    added.twist = s0.twist + (s1.twist - s0.twist) * f;
    _PushUndoLocked();
    _sections.insert(_sections.begin() + k + 1, std::move(added));
    _useSections = true;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Topology | PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::RemoveSectionRing(int ring)
{
    if (ring < 0 || ring >= int(_sections.size()) ||
        _sections.size() <= 2) {
        _diagnostic = "PomadeModel::RemoveSectionRing: need >= 2 rings left";
        return false;
    }
    _PushUndoLocked();
    _sections.erase(_sections.begin() + ring);
    _useSections = true;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Topology | PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::CopySectionRing(int src, int dst)
{
    if (src < 0 || src >= int(_sections.size()) || dst < 0 ||
        dst >= int(_sections.size())) {
        _diagnostic = "PomadeModel::CopySectionRing: ring out of range";
        return false;
    }
    _PushUndoLocked();
    float const t = _sections[size_t(dst)].t;
    _sections[size_t(dst)] = _sections[size_t(src)];
    _sections[size_t(dst)].t = t;
    _useSections = true;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::SetSoftSelection(float center, float radius)
{
    if (!(radius >= 0.0f)) {
        _diagnostic = "PomadeModel::SetSoftSelection: radius must be >= 0";
        return false;
    }
    _softCenter = center;
    _softRadius = radius;
    ++_version;
    return true;
}

void
PomadeModel::GetSoftSelection(float *center, float *radius) const
{
    if (center) {
        *center = _softCenter;
    }
    if (radius) {
        *radius = _softRadius;
    }
}

bool
PomadeModel::RelaxCenter(float strength, int iterations)
{
    int const n = int(_host.centerX.size());
    if (!(strength >= 0.0f) || !(strength <= 1.0f) || iterations < 1 ||
        n < 3) {
        _diagnostic = "PomadeModel::RelaxCenter: strength in [0, 1], "
                      "iterations >= 1, need >= 3 CVs";
        return false;
    }
    _PushUndoLocked();
    // K13-driven Laplacian smooth of the interior CVs: each CV's share
    // of `strength` ramps with its kink score over [threshold/2,
    // threshold], recomputed per iteration as the curve settles. Quiet
    // CVs (weight 0) are bit-identical no-ops.
    float const lo = 0.5f * kPomadeSmoothnessSpike;
    float const span = kPomadeSmoothnessSpike - lo;
    for (int it = 0; it < iterations; ++it) {
        std::vector<float> scores(size_t(n), 0.0f);
        std::string derr;
        if (!_LaneSmoothnessScore(HasCudaMirror(), _host.centerX.data(),
                                  _host.centerY.data(), _host.centerZ.data(),
                                  n, scores.data(), &derr)) {
            _diagnostic = derr;
            return false;
        }
        std::vector<float> nx = _host.centerX, ny = _host.centerY,
                            nz = _host.centerZ;
        for (int i = 1; i + 1 < n; ++i) {
            float w =
                (scores[size_t(i)] - lo) / span;
            w = w < 0.0f ? 0.0f : (w > 1.0f ? 1.0f : w);
            float const ax = 0.5f * (_host.centerX[size_t(i - 1)] +
                                     _host.centerX[size_t(i + 1)]);
            float const ay = 0.5f * (_host.centerY[size_t(i - 1)] +
                                     _host.centerY[size_t(i + 1)]);
            float const az = 0.5f * (_host.centerZ[size_t(i - 1)] +
                                     _host.centerZ[size_t(i + 1)]);
            nx[size_t(i)] += (ax - nx[size_t(i)]) * strength * w;
            ny[size_t(i)] += (ay - ny[size_t(i)]) * strength * w;
            nz[size_t(i)] += (az - nz[size_t(i)]) * strength * w;
        }
        _host.centerX = std::move(nx);
        _host.centerY = std::move(ny);
        _host.centerZ = std::move(nz);
    }
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::SnapRootToScalp()
{
    if (!_scalp || !_scalp->finalized || _host.centerX.empty()) {
        _diagnostic = "PomadeModel::SnapRootToScalp: need a bound scalp + "
                      "tube";
        return false;
    }
    float const p[3] = {_host.centerX[0], _host.centerY[0], _host.centerZ[0]};
    PomadeHit const hit = PomadeClosestPointCpu(*_scalp, p);
    if (!hit.hit) {
        _diagnostic = "PomadeModel::SnapRootToScalp: no surface point";
        return false;
    }
    _PushUndoLocked();
    float const dx = hit.px - _host.centerX[0];
    float const dy = hit.py - _host.centerY[0];
    float const dz = hit.pz - _host.centerZ[0];
    for (size_t i = 0; i < _host.centerX.size(); ++i) {
        _host.centerX[i] += dx;
        _host.centerY[i] += dy;
        _host.centerZ[i] += dz;
    }
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Points;
    ++_version;
    return true;
}

bool
PomadeModel::SetDisplaySegments(int segments)
{
    if (segments < 1 || segments > 16) {
        _diagnostic = "PomadeModel::SetDisplaySegments: segments in [1, 16]";
        return false;
    }
    if (segments == _segmentsPerSpan) {
        return true;
    }
    _segmentsPerSpan = segments;
    if (!_host.positions.empty() && !Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Topology | PomadeDirty_Points;
    ++_version;
    return true;
}

int
PomadeModel::GetDisplaySegments() const
{
    return _segmentsPerSpan;
}

void
PomadeModel::SetTubeRegionId(int regionId)
{
    _tubeRegionId = regionId;
    ++_version;
}

int
PomadeModel::GetTubeRegionId() const
{
    return _tubeRegionId;
}

std::vector<int>
PomadeModel::TubeRegionFaces() const
{
    std::vector<int> faces;
    if (!_scalp || _tubeRegionId < 0 ||
        _tubeRegionId >= _graph.RegionCount()) {
        return faces;
    }
    int const interp = _graph.InterpId(_tubeRegionId);
    for (size_t f = 0; f < _maps.faceRegion.size(); ++f) {
        if (_maps.faceRegion[f] == interp) {
            faces.push_back(int(f));
        }
    }
    return faces;
}

PomadeTubeDesc
PomadeModel::BuildTubeDesc() const
{
    return _BuildTubeDescLocked();
}

PomadeTubeDesc
PomadeModel::_BuildTubeDescLocked() const
{
    PomadeTubeDesc desc;
    desc.centerX = _host.centerX;
    desc.centerY = _host.centerY;
    desc.centerZ = _host.centerZ;
    desc.sections = _sections;
    desc.ringVerts = _shape.ringVerts;
    desc.regionId = _tubeRegionId;
    desc.level = 1;
    desc.rootFramePinned = _rootFramePinned;
    desc.rootFrame = _rootFrame;
    desc.frameReference = _frameReference;
    _AttachTubeRootLocked(&desc);
    return desc;
}

void PomadeModel::_AttachTubeRootLocked(PomadeTubeDesc *desc) const
{
    desc->rootSurfaceOffsets.clear();
    if (!_scalp || !_bvh.valid || !desc->rootFramePinned ||
        desc->regionId < 0 || desc->centerX.empty() || desc->sections.empty())
        return;
    // An intentional whole-tube lift remains an edit. Surface attachment
    // applies while the owning L1 support is on the scalp; child root
    // centers may themselves lie above a curved scalp's planar chart.
    float support[3] = {desc->centerX[0], desc->centerY[0], desc->centerZ[0]};
    int parent = desc->parentTubeId;
    while (parent >= 0) {
        if (parent == 0) {
            support[0] = _host.centerX[0]; support[1] = _host.centerY[0];
            support[2] = _host.centerZ[0];
            break;
        }
        auto const it = _tubes.find(parent);
        if (it == _tubes.end() || it->second.actual.centerX.empty()) return;
        auto const &p = it->second.actual;
        support[0] = p.centerX[0]; support[1] = p.centerY[0];
        support[2] = p.centerZ[0];
        parent = p.parentTubeId;
    }
    auto const &section = desc->sections.front();
    float const reach = std::max(2.0f * PomadeSectionMeanRadius(section), 0.01f);
    PomadeHit const supportHit = PomadeClosestPointCpu(*_scalp, support);
    if (!supportHit.hit || supportHit.t > reach * 1e-3f) return;
    auto const &frame = desc->rootFrame;
    float const direction[3] = {-frame.tx, -frame.ty, -frame.tz};
    float const ct = std::cos(section.twist), st = std::sin(section.twist);
    if (section.u.size() != size_t(desc->ringVerts) ||
        section.v.size() != section.u.size()) return;
    desc->rootSurfaceOffsets.resize(size_t(desc->ringVerts) * 3, 0.0f);
    for (int s = 0; s < desc->ringVerts; ++s) {
        float const u = section.u[size_t(s)] * section.scale;
        float const v = section.v[size_t(s)] * section.scale;
        float const ru = u * ct - v * st, rv = u * st + v * ct;
        float const p[3] = {
            desc->centerX[0] + frame.nx * ru + frame.bx * rv,
            desc->centerY[0] + frame.ny * ru + frame.by * rv,
            desc->centerZ[0] + frame.nz * ru + frame.bz * rv};
        float const origin[3] = {p[0] + frame.tx * reach,
                                 p[1] + frame.ty * reach,
                                 p[2] + frame.tz * reach};
        PomadeHit hit = PomadeRaycastCpu(*_scalp, _bvh, origin, direction);
        // A concavity or open mesh edge can miss the normal ray. Keep that
        // slot attached to the nearest support instead of leaving a gap.
        if (!hit.hit || hit.t >= 2.0f * reach)
            hit = PomadeClosestPointCpu(*_scalp, p);
        if (hit.hit) {
            desc->rootSurfaceOffsets[size_t(s) * 3] = hit.px - p[0];
            desc->rootSurfaceOffsets[size_t(s) * 3 + 1] = hit.py - p[1];
            desc->rootSurfaceOffsets[size_t(s) * 3 + 2] = hit.pz - p[2];
        }
    }
}

std::vector<PomadeFrame>
PomadeModel::GetFrames() const
{
    return _frames;
}

// -- P3: Fill mode --------------------------------------------------------

bool
PomadeModel::SetPreviewFraction(float fraction)
{
    if (!(fraction >= 0.0f) || !(fraction <= 1.0f)) {
        _diagnostic = "PomadeModel::SetPreviewFraction: fraction in [0, 1]";
        return false;
    }
    _previewFraction = fraction;
    ++_version;
    return true;
}

float
PomadeModel::GetPreviewFraction() const
{
    return _previewFraction;
}

void
PomadeModel::SetFreezeRoots(bool freeze)
{
    _freezeRoots = freeze;
    ++_version;
}

bool
PomadeModel::GetFreezeRoots() const
{
    return _freezeRoots;
}

bool
PomadeModel::RefillGuides(float fraction)
{
    if (_generatedCurvesSuppressed) {
        return true;
    }
    return _RefillGuidesLocked(fraction);
}

bool
PomadeModel::GenerateGuides(float fraction)
{
    if (!_generatedCurvesSuppressed) {
        return _RefillGuidesLocked(fraction);
    }
    HierarchyRollback const snap = _SnapshotHierarchyLocked();
    _generatedCurvesSuppressed = false;
    if (!_RefillGuidesLocked(fraction)) {
        _RestoreHierarchyLocked(snap);
        return false;
    }
    _PushUndoSnapshotLocked(snap);
    return true;
}

bool
PomadeModel::ClearGeneratedCurves()
{
    if (_generatedCurvesSuppressed && _guides.guideCount == 0) {
        return true;
    }
    HierarchyRollback const snap = _SnapshotHierarchyLocked();
    _generatedCurvesSuppressed = true;
    _roots.clear();
    _tubeRoots.clear();
    _guides = PomadeGuideSet();
    _refillDrops.clear();
    ++_guideVersion;
    _dirty |= PomadeDirty_Guides;
    _PushUndoSnapshotLocked(snap);
    ++_version;
    return true;
}

bool
PomadeModel::SetGeneratedCurvesVisible(bool visible)
{
    if (_generatedCurvesVisible == visible) {
        return true;
    }
    _generatedCurvesVisible = visible;
    _dirty |= PomadeDirty_Display;
    ++_version;
    return true;
}

bool
PomadeModel::GetGeneratedCurvesVisible() const
{
    return _generatedCurvesVisible;
}

bool
PomadeModel::GeneratedCurvesSuppressed() const
{
    return _generatedCurvesSuppressed;
}

void
PomadeModel::SetGeneratedCurvesSuppressed(bool suppressed)
{
    _generatedCurvesSuppressed = suppressed;
    if (suppressed) {
        _roots.clear();
        _tubeRoots.clear();
        _guides = PomadeGuideSet();
        _refillDrops.clear();
        ++_guideVersion;
        _dirty |= PomadeDirty_Guides;
    }
}

bool
PomadeModel::_RefillGuidesLocked(float fraction)
{
    if (fraction < 0.0f) {
        fraction = _previewFraction;
    }
    fraction = fraction < 0.0f ? 0.0f : (fraction > 1.0f ? 1.0f : fraction);
    // The producing tubes, ascending, tube 0 first: tube 0 fills unless
    // a subdivision suspended it, and a store tube fills unless it is
    // suspended, imported, or a group parent (the committer's skip rule,
    // PomadeGuidesFromSnapshot). A one-tube groom produces exactly tube 0.
    std::vector<int> producing;
    if (_host.centerX.size() >= 2 && !_FillSuspendedLocked(0)) {
        producing.push_back(0);
    }
    for (auto const &kv : _tubes) {
        if (kv.second.imported || !kv.second.members.empty() ||
            _FillSuspendedLocked(kv.first)) {
            continue;
        }
        producing.push_back(kv.first);
    }
    if (producing.empty()) {
        _diagnostic = "PomadeModel::RefillGuides: no tube";
        return false;
    }
    bool const scalpBound = bool(_scalp) && _scalp->finalized;
    // Region faces per tube: each L1 root collects the faces at its
    // interpolation id; children receive their share from the partition
    // below, never directly (the snapshot's collection gate).
    std::map<int, std::vector<int>> facesByTube;
    if (scalpBound) {
        if (_host.centerX.size() >= 2 && _tubeRegionId >= 0 &&
            _tubeRegionId < _graph.RegionCount()) {
            int const interp = _graph.InterpId(_tubeRegionId);
            std::vector<int> &faces = facesByTube[0];
            for (size_t f = 0; f < _maps.faceRegion.size(); ++f) {
                if (_maps.faceRegion[f] == interp) {
                    faces.push_back(int(f));
                }
            }
        }
        for (auto const &kv : _tubes) {
            int const regionId = kv.second.actual.regionId;
            if (kv.second.actual.level != 1 || regionId < 0 ||
                regionId >= _graph.RegionCount()) {
                continue;
            }
            int const interp = _graph.InterpId(regionId);
            std::vector<int> &faces = facesByTube[kv.first];
            for (size_t f = 0; f < _maps.faceRegion.size(); ++f) {
                if (_maps.faceRegion[f] == interp) {
                    faces.push_back(int(f));
                }
            }
        }
        // The face partition, step for step the snapshot's
        // _PartitionRegionFaces: every parent hands its faces down to the
        // child cell that claims each face centroid, parents before
        // children (breadth-first from the roots), children in childIndex
        // order. A parent whose frames fail keeps its faces, exactly as
        // the snapshot does (they go unused: a partitioned parent is
        // suspended, and its children fall back to the disc stream).
        std::vector<int> order;
        order.push_back(0);
        for (auto const &kv : _tubes) {
            if (kv.second.actual.parentTubeId < 0) {
                order.push_back(kv.first);
            }
        }
        for (size_t head = 0; head < order.size(); ++head) {
            int const parentId = order[head];
            // Groups carry negative ids (minted from _nextGroupId) and own
            // their members through membership, not parentTubeId; their
            // own parentTubeId is -1, so expanding a negative id would
            // re-enqueue every root plus the group itself and the walk
            // never terminates (hydrating a scene with an on-the-fly
            // group spun here forever).
            if (parentId < 0) {
                continue;
            }
            for (auto const &kv : _tubes) {
                if (kv.second.actual.parentTubeId == parentId) {
                    order.push_back(kv.first);
                }
            }
            auto fit = facesByTube.find(parentId);
            if (fit == facesByTube.end() || fit->second.empty()) {
                continue;
            }
            std::vector<std::pair<int, int>> cells;  // (childIndex, tubeId)
            for (auto const &kv : _tubes) {
                if (kv.second.actual.parentTubeId == parentId &&
                    !kv.second.imported &&
                    !kv.second.actual.centerX.empty()) {
                    cells.push_back(std::make_pair(
                        kv.second.actual.childIndex, kv.first));
                }
            }
            if (cells.empty()) {
                continue;
            }
            std::sort(cells.begin(), cells.end());
            std::vector<float> parentCenter;
            if (parentId == 0) {
                parentCenter = _host.centerX;
            } else {
                auto pit = _tubes.find(parentId);
                if (pit == _tubes.end()) {
                    continue;
                }
                parentCenter = pit->second.actual.centerX;
            }
            if (parentCenter.size() < 2) {
                continue;
            }
            std::vector<float> parentY, parentZ;
            if (parentId == 0) {
                parentY = _host.centerY;
                parentZ = _host.centerZ;
            } else {
                parentY = _tubes[parentId].actual.centerY;
                parentZ = _tubes[parentId].actual.centerZ;
            }
            std::vector<PomadeFrame> parentFrames;
            std::string perr;
            PomadeTubeDesc parentDesc;
            parentDesc.centerX = parentCenter;
            parentDesc.centerY = parentY;
            parentDesc.centerZ = parentZ;
            if (parentId == 0) {
                parentDesc = _BuildTubeDescLocked();
            } else {
                parentDesc = _tubes[parentId].actual;
            }
            if (!PomadeTubeFramesCpu(parentDesc, &parentFrames, &perr) ||
                parentFrames.empty()) {
                continue;
            }
            float const parentRoot[3] = {parentCenter[0], parentY[0],
                                         parentZ[0]};
            std::vector<float> childCenters;
            childCenters.reserve(cells.size() * 3);
            for (auto const &cell : cells) {
                PomadeTubeDesc const &desc = _tubes[cell.second].actual;
                childCenters.push_back(desc.centerX[0]);
                childCenters.push_back(desc.centerY[0]);
                childCenters.push_back(desc.centerZ[0]);
            }
            std::vector<int> faces = std::move(fit->second);
            fit->second.clear();
            for (int f : faces) {
                if (size_t(f) * 3 + 2 >= _scalp->faceCentroids.size()) {
                    continue;
                }
                int const cell = PomadeOwningChildCell(
                    parentRoot, parentFrames[0], childCenters.data(),
                    int(cells.size()),
                    &_scalp->faceCentroids[size_t(f) * 3]);
                if (cell >= 0) {
                    facesByTube[cells[size_t(cell)].second].push_back(f);
                }
            }
        }
    }
    // One tube's diagnostics keep their historical spelling; a store
    // tube's name its id. Only the first survives to the caller, on a
    // total failure: a degenerate tube is skipped, never fatal to its
    // siblings. Every skipped tube is also kept in `drops`, published as
    // RefillDrops() on success: a partial refill that silently lost a
    // tube's guides looked exactly like a healthy one to the artist.
    std::string firstErr;
    std::vector<std::pair<int, std::string>> drops;
    auto note = [&](int tubeId, std::string const &problem) {
        drops.emplace_back(tubeId, problem);
        if (!firstErr.empty()) {
            return;
        }
        firstErr = (tubeId == 0)
                       ? problem
                       : "tube " + std::to_string(tubeId) + ": " + problem;
    };
    std::map<int, std::vector<PomadeGuideRoot>> newStores;
    std::vector<PomadeGuideRoot> mergedRoots;
    PomadeGuideSet merged;
    uint64_t nextId = 1000;
    bool anyOk = false;
    bool cvTaken = false;
    std::vector<PomadeFrame> tube0Frames;
    bool tube0FramesOk = false;
    std::string err;
    auto ownershipCells = [&](int tubeId) {
        std::vector<PomadeRootOwnershipCell> chain;
        int childId = tubeId;
        for (;;) {
            auto childIt = _tubes.find(childId);
            if (childIt == _tubes.end() ||
                childIt->second.actual.parentTubeId < 0) {
                break;
            }
            int const parentId = childIt->second.actual.parentTubeId;
            std::vector<std::pair<int, int>> children;
            for (auto const &kv : _tubes) {
                if (kv.second.actual.parentTubeId == parentId &&
                    !kv.second.imported &&
                    !kv.second.actual.centerX.empty()) {
                    children.emplace_back(kv.second.actual.childIndex, kv.first);
                }
            }
            std::sort(children.begin(), children.end());
            if (children.empty()) {
                break;
            }
            PomadeTubeDesc parent;
            if (parentId == 0) {
                parent = _BuildTubeDescLocked();
            } else {
                auto parentIt = _tubes.find(parentId);
                if (parentIt == _tubes.end()) {
                    break;
                }
                parent = parentIt->second.actual;
            }
            std::vector<PomadeFrame> parentFrames;
            std::string frameErr;
            if (parent.centerX.size() < 2 ||
                !PomadeTubeFramesCpu(parent, &parentFrames, &frameErr) ||
                parentFrames.empty()) {
                break;
            }
            PomadeRootOwnershipCell cell;
            cell.rootCenter[0] = parent.centerX[0];
            cell.rootCenter[1] = parent.centerY[0];
            cell.rootCenter[2] = parent.centerZ[0];
            cell.frame = parentFrames[0];
            for (size_t i = 0; i < children.size(); ++i) {
                PomadeTubeDesc const &sibling = _tubes[children[i].second].actual;
                cell.childCenters.push_back(sibling.centerX[0]);
                cell.childCenters.push_back(sibling.centerY[0]);
                cell.childCenters.push_back(sibling.centerZ[0]);
                if (children[i].second == childId) {
                    cell.childIndex = int(i);
                }
            }
            if (cell.childIndex < 0) {
                break;
            }
            chain.push_back(std::move(cell));
            childId = parentId;
        }
        return chain;
    };
    for (int tubeId : producing) {
        PomadeTubeDesc tube;
        FillParams fill;
        if (tubeId == 0) {
            tube = _BuildTubeDescLocked();
            fill = _fill;
        } else {
            auto it = _tubes.find(tubeId);
            tube = it->second.actual;
            fill = it->second.fill;
        }
        if (tube.centerX.size() < 2 || tube.sections.size() < 2) {
            note(tubeId, "PomadeModel::RefillGuides: degenerate tube");
            continue;
        }
        std::vector<PomadeFrame> frames;
        if (!PomadeTubeFramesCpu(tube, &frames, &err)) {
            note(tubeId, err);
            continue;
        }
        if (tubeId == 0) {
            tube0Frames = frames;
            tube0FramesOk = true;
        }
        float const rootRadius =
            PomadeSectionMeanRadius(tube.sections.front());
        if (!(rootRadius > 0.0f)) {
            note(tubeId,
                 "PomadeModel::RefillGuides: degenerate root section");
            continue;
        }
        float const rootCenter[3] = {tube.centerX[0], tube.centerY[0],
                                     tube.centerZ[0]};
        // Mesh roots when the tube is region-rooted on a bound scalp
        // with claimed faces; otherwise the root-disc stream.
        std::vector<int> regionFaces;
        auto fit = facesByTube.find(tubeId);
        if (fit != facesByTube.end()) {
            regionFaces = fit->second;
        }
        int const regionId = (tubeId == 0) ? _tubeRegionId : tube.regionId;
        bool const meshFill = scalpBound && regionId >= 0 &&
                              regionId < _graph.RegionCount();
        // Every descendant inherits its L1 graph region. Sampling that exact
        // support first keeps a subface groom on the scalp even when coarse
        // face partitioning has no centroid to hand down to the child.
        bool const useExactRegion = meshFill && _loops.valid;
        bool const useMesh = meshFill && !regionFaces.empty() &&
                             !useExactRegion;
        std::vector<PomadeGuideRoot> roots;
        // Keep the unpartitioned exact stream for freeze. A child filters a
        // copy for K9; storing that filtered subset would feed it back as a
        // prefix of the full stream on the next refill and grow its count.
        std::vector<PomadeGuideRoot> exactFreezeStore;
        {
            auto sit = _tubeRoots.find(tubeId);
            if (sit != _tubeRoots.end()) {
                roots = sit->second;
            }
        }
        if (useExactRegion) {
            float const density = fill.density * fraction;
            int const frozen = _freezeRoots ? int(roots.size()) : 0;
            if (!PomadeRootSampleRegionMeshCpu(*_scalp, _loops, regionId,
                                               density, tubeId, fill.seed,
                                               rootCenter, frames[0],
                                               rootRadius, &roots, frozen,
                                               &err)) {
                note(tubeId, err);
                continue;
            }
            exactFreezeStore = roots;
            PomadeFilterGuideRootsByOwnershipCells(ownershipCells(tubeId),
                                                   &roots);
        } else if (!useMesh) {
            int const full =
                PomadeGuideCountForDensity(fill.density);
            int const count = int(float(full) * fraction + 0.5f);
            // Frozen roots keep their stored positions; only the tail is
            // re-sampled, continuing the same dart stream.
            int const frozen =
                _freezeRoots ? std::min(int(roots.size()), count) : 0;
            if (!PomadeRootSampleDiscCpu(tubeId, fill.seed, rootRadius,
                                        rootCenter, frames[0], count, &roots,
                                        frozen, &err)) {
                note(tubeId, err);
                continue;
            }
        } else {
            // Area-weighted mesh stream at the preview-scaled density.
            // Under freeze the kept prefix seeds the stream, so a frozen
            // full set differs from a from-scratch one (per-face spacing
            // depends on density); frozen roots are sacred by design. The
            // committer and hydrate always run frozen=0, so they agree
            // bit-exactly.
            float const density = fill.density * fraction;
            int frozen = _freezeRoots ? int(roots.size()) : 0;
            if (!PomadeRootSampleMeshCpu(
                    _scalp->points.data(), _scalp->faceVertexCounts.data(),
                    _scalp->faceVertexIndices.data(),
                    _scalp->faceOffsets.data(),
                    int(_scalp->faceVertexCounts.size()), regionFaces.data(),
                    int(regionFaces.size()), density, tubeId, fill.seed,
                    rootCenter, frames[0], rootRadius, &roots, frozen,
                    &err)) {
                note(tubeId, err);
                continue;
            }
        }
        PomadeGuideSet perTube;
        if (!roots.empty() &&
            !_FillAndResample(tube, frames, roots, fill, &perTube, &err)) {
            note(tubeId, err);
            continue;
        }
        if (useExactRegion) {
            _AnchorGuideRoots(roots, &perTube);
        }
        anyOk = true;
        mergedRoots.insert(mergedRoots.end(), roots.begin(), roots.end());
        newStores[tubeId] = useExactRegion ? std::move(exactFreezeStore)
                                            : std::move(roots);
        merged.points.insert(merged.points.end(), perTube.points.begin(),
                             perTube.points.end());
        merged.counts.insert(merged.counts.end(), perTube.counts.begin(),
                             perTube.counts.end());
        merged.frames.insert(merged.frames.end(), perTube.frames.begin(),
                             perTube.frames.end());
        for (int g = 0; g < perTube.guideCount; ++g) {
            merged.ids.push_back(nextId++);
            merged.tubeIds.push_back(tubeId);
        }
        if (!cvTaken && perTube.guideCount > 0) {
            merged.cvCount = perTube.cvCount;
            cvTaken = true;
        }
    }
    _refillDrops = std::move(drops);
    if (!anyOk) {
        _diagnostic = firstErr.empty()
                          ? "PomadeModel::RefillGuides: no tube"
                          : firstErr;
        return false;
    }
    // Dropped tubes leave no roots behind; living tubes keep their
    // freeze stores (a suspended parent keeps its own for the merge).
    for (auto it = _tubeRoots.begin(); it != _tubeRoots.end();) {
        if (it->first != 0 && _tubes.find(it->first) == _tubes.end()) {
            it = _tubeRoots.erase(it);
        } else {
            ++it;
        }
    }
    for (auto const &kv : newStores) {
        _tubeRoots[kv.first] = kv.second;
    }
    _roots = std::move(mergedRoots);
    merged.guideCount = int(merged.ids.size());
    _guides = std::move(merged);
    if (tube0FramesOk) {
        _frames = tube0Frames;
    }
    ++_guideVersion;
    _dirty |= PomadeDirty_Guides;
    ++_version;
    return true;
}

PomadeModel::GuidePreview
PomadeModel::GetGuidePreview() const
{
    // The current set IS the preview: RefillGuides(preview) during a drag
    // leaves the preview-density set here, RefillGuides(1.0) on release
    // leaves the full set (the release frame always shows full fidelity,
    // §7). No second fraction applies here.
    GuidePreview preview;
    if (!_generatedCurvesVisible) {
        return preview;
    }
    preview.points = _guides.points;
    preview.counts = _guides.counts;
    preview.tubeIds = _guides.tubeIds;
    preview.guideCount = _guides.guideCount;
    preview.cvCount = _guides.cvCount;
    return preview;
}

PomadeGuideSet const &
PomadeModel::GetGuides() const
{
    return _guides;
}

std::vector<std::pair<int, std::string>>
PomadeModel::RefillDrops() const
{
    return _refillDrops;
}

std::vector<PomadeGuideRoot> const &
PomadeModel::GetRoots() const
{
    return _roots;
}

// -- P3: pick -------------------------------------------------------------

namespace {

// The PomadePickCpu consider() rule as a fold predicate: `cand` (a
// later kind's winner) replaces `best` on a strictly nearer pixel, or
// an equal pixel at strictly lesser depth. Full ties keep `best`
// (the earlier kind), exactly like the sequential scan.
// The CPU pick path (USDGEN_ENABLE_CUDA off, which is the CI recipe)
// calls this; keep it outside the CUDA guard.
bool
PickHitBetter(PomadePickHit const &cand, PomadePickHit const &best)
{
    return cand.hit &&
           (!best.hit || cand.distPx < best.distPx ||
            (cand.distPx == best.distPx && cand.depth < best.depth));
}

}  // namespace

#ifdef USDGEN_POMADE_HAS_CUDA
namespace {

// Heavyweight (tube verts + guide CVs) candidates above this reduce on
// the device; at or below it the launch overhead exceeds the scan, so
// the CPU twin runs (same rule and tie order either way; the lane's
// TN-6 parity is bit-exact on identical inputs).
constexpr int kPickDeviceThreshold = 4096;

// Test hook (P6 precedent): a non-empty USDGEN_POMADE_FORCE_CPU_PICK
// disables the device path so the T0 can prove CPU/device agreement.
bool
CpuPickForced()
{
    char const *forced = std::getenv("USDGEN_POMADE_FORCE_CPU_PICK");
    return forced && *forced;
}

}  // namespace

bool
PomadeModel::_ReducePickLocked(float const *devicePositions, int candidateCount,
                              float const viewProj[16], int w, int h, float x,
                              float y, float radiusPx,
                              PomadeDevicePickBest *out) const
{
    static_assert(sizeof(PomadeDevicePickBest) == 4 * sizeof(float) &&
                      alignof(PomadeDevicePickBest) <= alignof(float),
                  "the pick record must pack into 4 floats");
    if (!devicePositions || !viewProj || !out || candidateCount <= 0 ||
        !_device || !_device->stream) {
        return false;
    }
    int const grid = (candidateCount + 255) / 256;
    int const slices = (grid + 255) / 256;
    // Grow-only: tube/guide reduces alternate sizes every pick, so a
    // plain reset would free+malloc twice per pick. The high-water mark
    // persists (bounded by the largest pick ever served); the kernels
    // only ever write their own prefix.
    if (_device->pickScratch.size() < size_t(grid) * 4 &&
        _device->pickScratch.reset(size_t(grid) * 4) != cudaSuccess) {
        return false;
    }
    if (_device->pickBest.size() < size_t(slices) * 4 &&
        _device->pickBest.reset(size_t(slices) * 4) != cudaSuccess) {
        return false;
    }
    char errBuf[256] = {0};
    if (!PomadeLaunchPickReduce(
            devicePositions, candidateCount, viewProj, w, h, x, y, radiusPx,
            reinterpret_cast<PomadeDevicePickBest *>(
                _device->pickBest.data()),
            reinterpret_cast<PomadeDevicePickBest *>(
                _device->pickScratch.data()),
            _device->stream, errBuf, sizeof(errBuf))) {
        return false;
    }
    if (cudaStreamSynchronize(_device->stream) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(out, _device->pickBest.data(),
                   sizeof(PomadeDevicePickBest),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    return true;
}

bool
PomadeModel::_UploadGuideCVsLocked(PomadePickSets const &sets) const
{
    if (!sets.guideCVs || sets.guideCount <= 0 || sets.guideCvCount <= 0 ||
        !_device) {
        return false;
    }
    size_t const flat = size_t(sets.guideCount) * size_t(sets.guideCvCount);
    if (flat == 0 || flat > size_t(INT_MAX)) {
        return false;
    }
    if (_device->guideCVsVersion == _guideVersion &&
        _device->guideCVs.size() == flat * 3) {
        return true;  // already uploaded this generation
    }
    if (_device->guideCVs.reset(flat * 3) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(_device->guideCVs.data(), sets.guideCVs,
                   flat * 3 * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    _device->guideCVsVersion = _guideVersion;
    return true;
}

bool
PomadeModel::_PickDeviceLocked(PomadePickSets const &sets, uint32_t kindMask,
                              float const viewProj[16], int w, int h, float x,
                              float y, float radiusPx,
                              PomadePickHit *out) const
{
    if (!out || !_device) {
        return false;
    }
    PomadePickHit best;  // kind order below mirrors PomadePickCpu exactly:
                        // tube verts, small kinds, guides.
    if ((kindMask & PomadePick_TubeVert) && sets.tubeVertCount > 0) {
        if (_device->positions.size() !=
            size_t(sets.tubeVertCount) * 3) {
            return false;  // mirror desync: let the CPU twin run instead
        }
        PomadeDevicePickBest wb;
        if (!_ReducePickLocked(_device->positions.data(), sets.tubeVertCount,
                               viewProj, w, h, x, y, radiusPx, &wb)) {
            return false;
        }
        if (wb.hit) {
            best.hit = true;
            best.kind = PomadePick_TubeVert;
            best.index = wb.index;
            best.subIndex = -1;
            best.distPx = wb.distPx;
            best.depth = wb.depth;
        }
    }
    {
        // Centers, sections and graph nodes are small-kind CPU work (no
        // device mirror exists for their variable layouts); the twin
        // call preserves their internal order.
        PomadePickSets small = sets;
        small.tubeVerts = nullptr;
        small.tubeVertCount = 0;
        small.guideCVs = nullptr;
        small.guideCount = 0;
        small.guideCvCount = 0;
        PomadePickHit folded =
            PomadePickCpu(small, kindMask, viewProj, w, h, x, y, radiusPx);
        if (PickHitBetter(folded, best)) {
            best = folded;
        }
    }
    if ((kindMask & PomadePick_Guide) && sets.guideCount > 0 &&
        sets.guideCvCount > 0) {
        size_t const flat =
            size_t(sets.guideCount) * size_t(sets.guideCvCount);
        if (flat == 0 || flat > size_t(INT_MAX)) {
            return false;
        }
        if (!_UploadGuideCVsLocked(sets)) {
            return false;
        }
        PomadeDevicePickBest wb;
        if (!_ReducePickLocked(_device->guideCVs.data(), int(flat), viewProj,
                               w, h, x, y, radiusPx, &wb)) {
            return false;
        }
        if (wb.hit) {
            PomadePickHit folded;
            folded.hit = true;
            folded.kind = PomadePick_Guide;
            folded.index = wb.index / sets.guideCvCount;
            folded.subIndex = wb.index % sets.guideCvCount;
            folded.distPx = wb.distPx;
            folded.depth = wb.depth;
            if (PickHitBetter(folded, best)) {
                best = folded;
            }
        }
    }
    *out = best;
    return true;
}
#endif  // USDGEN_POMADE_HAS_CUDA

PomadePickSets
PomadeModel::_BuildPickSetsLocked(_PickScratch *scratch,
                                 uint32_t displayGraphKinds) const
{
    PomadePickSets sets;
    if (!scratch) {
        return sets;
    }
    // Tube-owned controls must agree with the surface the viewport exposes.
    // The active cut is authoritative when enabled; otherwise the legacy
    // focus level is the active target. Hidden levels, centers-only levels
    // and suppressed center overlays are not click targets.
    auto tubePickable = [&](PomadeTubeDesc const &desc, bool mesh) {
        auto const drawIt = _levelDisplay.find(desc.level);
        LevelDisplay const draw = drawIt == _levelDisplay.end()
                                      ? LevelDisplay()
                                      : drawIt->second;
        if (!draw.visible || !_IsTubeVisibleInActiveCutLocked(desc.tubeId) ||
            (!_activeCutEnabled && _focusLevel > 0 &&
             desc.level != _focusLevel)) {
            return false;
        }
        return mesh ? !draw.centersOnly : draw.centers;
    };
    if (!_host.positions.empty() && _tubes.empty()) {
        // Single tube: point at the host mirror (no copy, and the
        // device pick mirror stays applicable) when the active cut exposes
        // it. A collapsed parent has no surface candidates at all.
        PomadeTubeDesc root;
        if (_TubeDescLocked(0, &root) && tubePickable(root, /*mesh*/ true)) {
            sets.tubeVerts = _host.positions.data();
            sets.tubeVertCount = int(_host.positions.size() / 3);
            _PickScratch::SurfaceStrip strip;
            strip.tubeId = 0;
            strip.level = root.level;
            strip.firstVertex = 0;
            strip.ringVerts = root.ringVerts;
            strip.ringCount = strip.ringVerts > 0
                                  ? sets.tubeVertCount / strip.ringVerts
                                  : 0;
            if (strip.ringCount >= 2 && strip.ringVerts >= 3) {
                scratch->surfaceStrips.push_back(strip);
            }
        }
    } else if (!_tubes.empty()) {
        // With children, concatenate only the active-cut surface frontier.
        // The map and strips use this same filtered stream, so raw K11,
        // point selection and face picking cannot reach a hidden branch.
        PomadeTubeDesc root;
        if (_TubeDescLocked(0, &root) && tubePickable(root, /*mesh*/ true)) {
            scratch->tubeVertPositions = _host.positions;
            scratch->tubeVertTubeIds.assign(_host.positions.size() / 3, 0);
            _PickScratch::SurfaceStrip strip;
            strip.tubeId = 0;
            strip.level = root.level;
            strip.firstVertex = 0;
            strip.ringVerts = root.ringVerts;
            strip.ringCount = strip.ringVerts > 0
                                  ? int(_host.positions.size() / 3) /
                                        strip.ringVerts
                                  : 0;
            if (strip.ringCount >= 2 && strip.ringVerts >= 3) {
                scratch->surfaceStrips.push_back(strip);
            }
        }
        int const segments = std::max(_segmentsPerSpan, 1);
        for (int tubeId : _LiveTubeIdsLocked()) {
            if (tubeId == 0) {
                continue;
            }
            PomadeTubeDesc desc;
            if (!_TubeDescLocked(tubeId, &desc) ||
                desc.centerX.size() < 2 || desc.sections.size() < 2 ||
                !tubePickable(desc, /*mesh*/ true)) {
                continue;
            }
            std::vector<PomadeFrame> frames;
            std::string err;
            if (!PomadeTubeFramesCpu(desc, &frames, &err)) {
                continue;
            }
            std::vector<float> positions;
            std::vector<float> normals;
            std::vector<float> ringT;
            if (!PomadeTessellateCpu(desc, frames, segments, &positions,
                                    &normals, &ringT, &err) ||
                positions.empty()) {
                continue;
            }
            _PickScratch::SurfaceStrip strip;
            strip.tubeId = tubeId;
            strip.level = desc.level;
            strip.firstVertex =
                int(scratch->tubeVertPositions.size() / 3);
            strip.ringVerts = desc.ringVerts;
            strip.ringCount = strip.ringVerts > 0
                                  ? int(positions.size() / 3) /
                                        strip.ringVerts
                                  : 0;
            scratch->tubeVertPositions.insert(
                scratch->tubeVertPositions.end(), positions.begin(),
                positions.end());
            scratch->tubeVertTubeIds.insert(scratch->tubeVertTubeIds.end(),
                                            positions.size() / 3, tubeId);
            if (strip.ringCount >= 2 && strip.ringVerts >= 3) {
                scratch->surfaceStrips.push_back(strip);
            }
        }
        if (!scratch->tubeVertPositions.empty()) {
            sets.tubeVerts = scratch->tubeVertPositions.data();
            sets.tubeVertCount =
                int(scratch->tubeVertPositions.size() / 3);
        }
    }
    PomadeTubeDesc rootDesc;
    bool const rootPickable = _TubeDescLocked(0, &rootDesc) &&
                              tubePickable(rootDesc, /*mesh*/ false);
    if (!_host.centerX.empty() && rootPickable) {
        scratch->center.reserve(_host.centerX.size() * 3);
        scratch->centerTubeIds.reserve(_host.centerX.size());
        scratch->centerCvIds.reserve(_host.centerX.size());
        std::vector<PomadeFrame> frames;
        std::string err;
        bool const haveFrames = PomadeTubeFramesCpu(rootDesc, &frames, &err);
        for (size_t i = 0; i < _host.centerX.size(); ++i) {
            float x = _host.centerX[i], y = _host.centerY[i],
                  z = _host.centerZ[i];
            if (haveFrames)
                PomadeCenterHandlePointCpu(rootDesc, frames, int(i), &x, &y, &z, &err);
            scratch->center.push_back(x);
            scratch->center.push_back(y);
            scratch->center.push_back(z);
            scratch->centerTubeIds.push_back(0);
            scratch->centerCvIds.push_back(int(i));
        }
    }
    // Children read through their descs, in live-id order, so the layout
    // is a pure function of the model.
    for (int tubeId : _LiveTubeIdsLocked()) {
        if (tubeId == 0) {
            continue;
        }
        PomadeTubeDesc desc;
        if (!_TubeDescLocked(tubeId, &desc) ||
            !tubePickable(desc, /*mesh*/ false)) {
            continue;
        }
        std::vector<PomadeFrame> frames;
        std::string err;
        bool const haveFrames = PomadeTubeFramesCpu(desc, &frames, &err);
        for (size_t i = 0; i < desc.centerX.size(); ++i) {
            float x = desc.centerX[i], y = desc.centerY[i],
                  z = desc.centerZ[i];
            if (haveFrames)
                PomadeCenterHandlePointCpu(desc, frames, int(i), &x, &y, &z, &err);
            scratch->center.push_back(x);
            scratch->center.push_back(y);
            scratch->center.push_back(z);
            scratch->centerTubeIds.push_back(tubeId);
            scratch->centerCvIds.push_back(int(i));
        }
    }
    if (!scratch->center.empty()) {
        sets.centerCVs = scratch->center.data();
        sets.centerCVCount = int(scratch->center.size() / 3);
    }
    // Section and ring handles share the draw/tessellation geometry for
    // every live tube.  The scratch maps retain their real owners because
    // children may have a different ring-vertex count from tube 0.
    for (int tubeId : _LiveTubeIdsLocked()) {
        PomadeTubeDesc desc;
        if (!_TubeDescLocked(tubeId, &desc) || desc.centerX.size() < 2 ||
            desc.sections.empty() || !tubePickable(desc, /*mesh*/ true)) {
            continue;
        }
        std::vector<PomadeFrame> frames;
        std::string err;
        if (!PomadeTubeFramesCpu(desc, &frames, &err)) {
            continue;
        }
        for (size_t ring = 0; ring < desc.sections.size(); ++ring) {
            PomadeTubeSection const &s = desc.sections[ring];
            if (s.u.empty() || s.u.size() != s.v.size()) {
                continue;
            }
            float cp[3];
            PomadeFrame fr;
            if (!PomadeSampleCenterCpu(desc, frames, s.t, &cp[0], &cp[1],
                                      &cp[2], &fr, &err)) {
                continue;
            }
            float const ct = std::cos(s.twist), st = std::sin(s.twist);
            double rx = 0.0, ry = 0.0, rz = 0.0;
            for (size_t slot = 0; slot < s.u.size(); ++slot) {
                float const u = s.u[slot] * s.scale;
                float const v = s.v[slot] * s.scale;
                float const ru = u * ct - v * st;
                float const rv = u * st + v * ct;
                bool const attached = ring == 0 &&
                    desc.rootSurfaceOffsets.size() == s.u.size() * 3;
                float const px = cp[0] + fr.nx * ru + fr.bx * rv +
                    (attached ? desc.rootSurfaceOffsets[slot * 3] : 0.0f);
                float const py = cp[1] + fr.ny * ru + fr.by * rv +
                    (attached ? desc.rootSurfaceOffsets[slot * 3 + 1] : 0.0f);
                float const pz = cp[2] + fr.nz * ru + fr.bz * rv +
                    (attached ? desc.rootSurfaceOffsets[slot * 3 + 2] : 0.0f);
                scratch->section.push_back(px);
                scratch->section.push_back(py);
                scratch->section.push_back(pz);
                scratch->sectionTubeIds.push_back(tubeId);
                scratch->sectionRingIds.push_back(int(ring));
                scratch->sectionSlotIds.push_back(int(slot));
                rx += px;
                ry += py;
                rz += pz;
            }
            // A ring candidate is its centroid, so marquee selection follows
            // the visible handle rather than a CV grazing a box corner.
            double const n = double(s.u.size());
            scratch->ringCenters.push_back(float(rx / n));
            scratch->ringCenters.push_back(float(ry / n));
            scratch->ringCenters.push_back(float(rz / n));
            scratch->ringTubeIds.push_back(tubeId);
            scratch->ringIds.push_back(int(ring));
        }
    }
    if (!scratch->section.empty()) {
        sets.sectionCVs = scratch->section.data();
        sets.sectionCVCount = int(scratch->section.size() / 3);
        // Keep the public pick payload unchanged: this only transports a
        // reversible flat candidate ordinal. _ItemFromCandidateLocked uses
        // the explicit maps above, so child ring sizes need not match it.
        // The pick payload spells a flat candidate as (ordinal, 0).  Keep
        // the scratch-side decoder in the same convention; a child can own
        // a different number of section vertices than its parent.
        scratch->ringVerts = 1;
        sets.sectionRingVerts = scratch->ringVerts;
        sets.ringCenters = scratch->ringCenters.data();
        sets.ringCount = int(scratch->ringCenters.size() / 3);
    }
    for (auto const &nd : _graph.Nodes()) {
        if (nd.alive) {
            float p[3] = {nd.p[0], nd.p[1], nd.p[2]};
            if (displayGraphKinds & PomadePick_GraphNode) {
                PomadeGraphDisplayPosition(nd, _scalp.get(), p);
            }
            scratch->nodes.push_back(p[0]);
            scratch->nodes.push_back(p[1]);
            scratch->nodes.push_back(p[2]);
            scratch->nodeIds.push_back(nd.id);
        }
    }
    if (!scratch->nodes.empty()) {
        sets.graphNodes = scratch->nodes.data();
        sets.graphNodeCount = int(scratch->nodes.size() / 3);
    }
    for (auto const &e : _graph.Edges()) {
        if (!e.alive || e.polyline.size() < 6) {
            continue;
        }
        int const n = int(e.polyline.size() / 3);
        for (int i = 0; i < n; ++i) {
            float p[3] = {e.polyline[size_t(i) * 3 + 0],
                          e.polyline[size_t(i) * 3 + 1],
                          e.polyline[size_t(i) * 3 + 2]};
            if (displayGraphKinds & PomadePick_GraphEdge) {
                PomadeGraphDisplaySurfacePosition(p, _scalp.get(), p);
            }
            scratch->edgeCVs.push_back(p[0]);
            scratch->edgeCVs.push_back(p[1]);
            scratch->edgeCVs.push_back(p[2]);
            scratch->edgeIds.push_back(e.id);
        }
    }
    if (!scratch->edgeIds.empty()) {
        sets.graphEdgeCVs = scratch->edgeCVs.data();
        sets.graphEdgeIds = scratch->edgeIds.data();
        sets.graphEdgeCVCount = int(scratch->edgeIds.size());
    }
    for (auto const &r : _graph.Regions()) {
        if (r.isOutside || r.boundary.size() < 9) {
            continue;
        }
        double cx = 0.0, cy = 0.0, cz = 0.0;
        size_t const n = r.boundary.size() / 3;
        for (size_t i = 0; i < n; ++i) {
            cx += r.boundary[i * 3 + 0];
            cy += r.boundary[i * 3 + 1];
            cz += r.boundary[i * 3 + 2];
        }
        scratch->regionCenters.push_back(float(cx / double(n)));
        scratch->regionCenters.push_back(float(cy / double(n)));
        scratch->regionCenters.push_back(float(cz / double(n)));
        scratch->regionIds.push_back(r.id);
    }
    if (!scratch->regionIds.empty()) {
        sets.regionCenters = scratch->regionCenters.data();
        sets.regionIds = scratch->regionIds.data();
        sets.regionCount = int(scratch->regionIds.size());
    }
    if (!_guides.points.empty() && _guides.cvCount > 0) {
        sets.guideCVs = _guides.points.data();
        sets.guideCount = _guides.guideCount;
        sets.guideCvCount = _guides.cvCount;
    }
    return sets;
}

PomadePickHit
PomadeModel::Pick(float const viewProj[16], int w, int h, float x, float y,
                 float radiusPx, uint32_t kindMask) const
{
    // Raw K11 contract: candidate indices and distances always refer to the
    // nearest projected candidate position.  Pomade_Pick is used to verify
    // CPU/GPU projection parity, so it cannot synthesize a face hit with a
    // representative vertex index and a zero distance.
    _PickScratch scratch;
    PomadePickSets const sets = _BuildPickSetsLocked(&scratch);
    PomadePickHit const extra = PomadePickExtraKindsCpu(
        sets, kindMask, viewProj, w, h, x, y, radiusPx);
#ifdef USDGEN_POMADE_HAS_CUDA
    if (_device && _device->streamOwned && _device->stream &&
        !CpuPickForced()) {
        size_t heavy = 0;
        if (kindMask & PomadePick_TubeVert) {
            heavy += size_t(std::max(sets.tubeVertCount, 0));
        }
        if (kindMask & PomadePick_Guide) {
            heavy += size_t(std::max(sets.guideCount, 0)) *
                     size_t(std::max(sets.guideCvCount, 0));
        }
        if (heavy > size_t(kPickDeviceThreshold)) {
            PomadePickHit hit;
            if (_PickDeviceLocked(sets, kindMask, viewProj, w, h, x, y,
                                  radiusPx, &hit)) {
                return PickHitBetter(extra, hit) ? extra : hit;
            }
        }
    }
#else
    (void)0;
#endif
    PomadePickHit const base =
        PomadePickCpu(sets, kindMask, viewProj, w, h, x, y, radiusPx);
    return PickHitBetter(extra, base) ? extra : base;
}

PomadePickHit
PomadeModel::PickItem(float const viewProj[16], int w, int h, float x,
                     float y, float radiusPx, uint32_t kindMask) const
{
    _PickScratch scratch;
    // Pomade_Pick remains a raw K11 projection query.  Selection acts on
    // the visible graph dots, whose normal lift keeps them above the scalp.
    PomadePickSets const sets = _BuildPickSetsLocked(
        &scratch, kindMask & (PomadePick_GraphNode | PomadePick_GraphEdge));

    // A tube mesh is the broad, whole-object target.  Its vertices were
    // historically folded before the edit handles, which meant a surface
    // vertex at the same pixel could make a center or section CV impossible
    // to pick.  Resolve every requested component first; the mesh is only a
    // fallback when no requested handle is under the cursor.
    uint32_t const componentMask = kindMask & ~uint32_t(PomadePick_TubeVert);
    auto pointPick = [&](uint32_t mask) {
#ifdef USDGEN_POMADE_HAS_CUDA
        if (_device && _device->streamOwned && _device->stream &&
            !CpuPickForced()) {
        size_t heavy = 0;
        if (mask & PomadePick_TubeVert) {
            heavy += size_t(std::max(sets.tubeVertCount, 0));
        }
        if (mask & PomadePick_Guide) {
            heavy += size_t(std::max(sets.guideCount, 0)) *
                     size_t(std::max(sets.guideCvCount, 0));
        }
        if (heavy > size_t(kPickDeviceThreshold)) {
            PomadePickHit hit;
            if (_PickDeviceLocked(sets, mask, viewProj, w, h, x, y,
                                  radiusPx, &hit)) {
                    return hit;
            }
            // Any device failure falls through to the CPU twin (P6
            // style, but non-sticky: the mirror itself is untouched).
        }
        }
#else
        (void)0;
#endif
        return PomadePickCpu(sets, mask, viewProj, w, h, x, y, radiusPx);
    };

    PomadePickHit const extra = PomadePickExtraKindsCpu(
        sets, componentMask, viewProj, w, h, x, y, radiusPx);
    PomadePickHit const component = pointPick(componentMask);
    bool const extraWins =
        extra.hit && (!component.hit || extra.distPx < component.distPx ||
                      (extra.distPx == component.distPx &&
                       extra.depth < component.depth));
    if (extraWins || component.hit) {
        return extraWins ? extra : component;
    }

    // The normal point picker intentionally has a snap radius.  The exact
    // projected quads of the *published* tube strips establish visible
    // ownership first; a vertex snap survives only when no face owns the
    // cursor. This is not a ray against an unbounded analytic tube: a hit
    // has to land inside a visible rendered quad. X-ray levels remain
    // pickable as drawn, but an active focus limits whole-tube picks to
    // that level, matching the editing policy.
    if (!(kindMask & PomadePick_TubeVert) || !viewProj || w <= 0 || h <= 0 ||
        !(radiusPx >= 0.0f)) {
        return PomadePickHit();
    }
    auto surfacePickable = [&](int tubeId, int level) {
        auto const drawIt = _levelDisplay.find(level);
        LevelDisplay const draw = drawIt == _levelDisplay.end()
                                      ? LevelDisplay()
                                      : drawIt->second;
        return draw.visible && !draw.centersOnly &&
               _IsTubeVisibleInActiveCutLocked(tubeId) &&
               (_activeCutEnabled || _focusLevel <= 0 ||
                level == _focusLevel);
    };
    bool surfaceFiltered = _activeCutEnabled || _focusLevel > 0;
    for (_PickScratch::SurfaceStrip const &strip : scratch.surfaceStrips) {
        if (!surfacePickable(strip.tubeId, strip.level)) {
            surfaceFiltered = true;
            break;
        }
    }

    // With the default display every vertex is eligible and the existing
    // device reduction keeps its performance contract.  A hidden/focused
    // display needs the same filtering as the face path, so scan only the
    // visible strips on the host and preserve the raw vertex ordinal.
    PomadePickHit vertex;
    if (!surfaceFiltered) {
        vertex = pointPick(uint32_t(PomadePick_TubeVert));
    } else {
        float const r2 = radiusPx * radiusPx;
        for (_PickScratch::SurfaceStrip const &strip : scratch.surfaceStrips) {
            if (!surfacePickable(strip.tubeId, strip.level)) {
                continue;
            }
            int const end = strip.firstVertex +
                            strip.ringCount * strip.ringVerts;
            for (int i = strip.firstVertex; i < end; ++i) {
                float px, py, depth;
                float const *p = sets.tubeVerts + size_t(i) * 3;
                if (!PomadeProjectPoint(p, viewProj, w, h, &px, &py, &depth)) {
                    continue;
                }
                float const dx = px - x, dy = py - y;
                float const d2 = dx * dx + dy * dy;
                if (d2 > r2) {
                    continue;
                }
                float const dist = std::sqrt(d2);
                if (!vertex.hit || dist < vertex.distPx ||
                    (dist == vertex.distPx && depth < vertex.depth)) {
                    vertex.hit = true;
                    vertex.kind = PomadePick_TubeVert;
                    vertex.index = i;
                    vertex.subIndex = -1;
                    vertex.distPx = dist;
                    vertex.depth = depth;
                }
            }
        }
    }

    PomadePickHit surface;
    auto considerTriangle = [&](int representative,
                                float const *a, float const *b,
                                float const *c) {
        float ax, ay, az, bx, by, bz, cx, cy, cz;
        if (!PomadeProjectPoint(a, viewProj, w, h, &ax, &ay, &az) ||
            !PomadeProjectPoint(b, viewProj, w, h, &bx, &by, &bz) ||
            !PomadeProjectPoint(c, viewProj, w, h, &cx, &cy, &cz)) {
            return;
        }
        float const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if (std::abs(area) < 1e-8f) {
            return;
        }
        float const wa = ((bx - x) * (cy - y) - (by - y) * (cx - x)) /
                         area;
        float const wb = ((cx - x) * (ay - y) - (cy - y) * (ax - x)) /
                         area;
        float const wc = 1.0f - wa - wb;
        constexpr float kEdgeTolerance = 1e-5f;
        if (wa < -kEdgeTolerance || wb < -kEdgeTolerance ||
            wc < -kEdgeTolerance) {
            return;
        }
        float const depth = wa * az + wb * bz + wc * cz;
        if (!surface.hit || depth < surface.depth) {
            surface.hit = true;
            surface.kind = PomadePick_TubeVert;
            surface.index = representative;
            surface.subIndex = -1;
            surface.distPx = 0.0f;
            surface.depth = depth;
        }
    };
    for (_PickScratch::SurfaceStrip const &strip : scratch.surfaceStrips) {
        if (!surfacePickable(strip.tubeId, strip.level)) {
            continue;
        }
        for (int ring = 0; ring + 1 < strip.ringCount; ++ring) {
            for (int slot = 0; slot < strip.ringVerts; ++slot) {
                int const next = (slot + 1) % strip.ringVerts;
                int const i0 = strip.firstVertex + ring * strip.ringVerts + slot;
                int const i1 = strip.firstVertex + ring * strip.ringVerts + next;
                int const i2 = strip.firstVertex + (ring + 1) * strip.ringVerts + next;
                int const i3 = strip.firstVertex + (ring + 1) * strip.ringVerts + slot;
                float const *p0 = sets.tubeVerts + size_t(i0) * 3;
                float const *p1 = sets.tubeVerts + size_t(i1) * 3;
                float const *p2 = sets.tubeVerts + size_t(i2) * 3;
                float const *p3 = sets.tubeVerts + size_t(i3) * 3;
                considerTriangle(strip.firstVertex, p0, p1, p2);
                considerTriangle(strip.firstVertex, p0, p2, p3);
            }
        }
    }
    // An exact visible face owns the click.  A vertex from a rear tube is a
    // useful snap target only when the cursor did not land on any visible
    // surface, otherwise it defeats normal front-depth occlusion.
    return surface.hit ? surface : vertex;
}

// -- V1: selection (plan/18 §2.3) --------------------------------------------

std::vector<int>
PomadeModel::_LiveTubeIdsLocked() const
{
    std::vector<int> ids;
    if (_host.centerX.size() >= 2) {
        ids.push_back(0);
    }
    for (auto const &kv : _tubes) {
        ids.push_back(kv.first);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

bool
PomadeModel::_IsDescendantOfLocked(int tubeId, int ancestorTubeId) const
{
    if (tubeId == ancestorTubeId) {
        return false;
    }
    PomadeTubeDesc desc;
    if (!_TubeDescLocked(tubeId, &desc)) {
        return false;
    }
    // A malformed imported hierarchy must not let a cycle make every tube
    // disappear. Live ids bound the parent walk and unknown parents stop it.
    size_t const limit = _tubes.size() + 1;
    for (size_t steps = 0; steps < limit && desc.parentTubeId >= 0; ++steps) {
        if (desc.parentTubeId == ancestorTubeId) {
            return true;
        }
        if (!_TubeDescLocked(desc.parentTubeId, &desc)) {
            return false;
        }
    }
    return false;
}

void
PomadeModel::_PruneActiveCutLocked()
{
    for (auto it = _expandedTubeIds.begin(); it != _expandedTubeIds.end();) {
        PomadeTubeDesc desc;
        bool hasChild = false;
        if (_TubeDescLocked(*it, &desc)) {
            for (auto const &entry : _tubes) {
                if (entry.second.actual.parentTubeId == *it) {
                    hasChild = true;
                    break;
                }
            }
        }
        if (!hasChild) {
            it = _expandedTubeIds.erase(it);
        } else {
            ++it;
        }
    }
}

bool
PomadeModel::_IsTubeVisibleInActiveCutLocked(int tubeId) const
{
    PomadeTubeDesc desc;
    if (!_TubeDescLocked(tubeId, &desc)) {
        return false;
    }
    if (!_activeCutEnabled) {
        return true;
    }
    // The cut has exactly one owner along each root-to-leaf path: an
    // unexpanded tube with every ancestor expanded. An expanded non-leaf is
    // replaced by its direct children. A stale bit on a leaf must never
    // blank the frontier while an undo/redo restore is being reconciled.
    if (_expandedTubeIds.count(tubeId) != 0) {
        for (auto const &entry : _tubes) {
            if (entry.second.actual.parentTubeId == tubeId) {
                return false;
            }
        }
    }
    size_t const limit = _tubes.size() + 1;
    for (size_t steps = 0; steps < limit && desc.parentTubeId >= 0; ++steps) {
        if (_expandedTubeIds.count(desc.parentTubeId) == 0) {
            return false;
        }
        if (!_TubeDescLocked(desc.parentTubeId, &desc)) {
            return false;
        }
    }
    return desc.parentTubeId < 0;
}

PomadeSelectionItem
PomadeModel::_ItemFromCandidateLocked(_PickScratch const &scratch,
                                     uint32_t kind, int index,
                                     int subIndex) const
{
    PomadeSelectionItem item;
    item.kind = kind;
    switch (kind) {
    case PomadePick_TubeVert:
        // A surface hit selects the tube it belongs to. The single-tube
        // path leaves the map empty and every surface candidate is tube
        // 0's; with children the map names the tessellated tube.
        if (scratch.tubeVertTubeIds.empty()) {
            item.id = 0;
        } else if (index < 0 ||
                   index >= int(scratch.tubeVertTubeIds.size())) {
            item.kind = 0;
        } else {
            item.id = scratch.tubeVertTubeIds[size_t(index)];
        }
        break;
    case PomadePick_CenterCV:
        if (index < 0 || index >= int(scratch.centerTubeIds.size()) ||
            index >= int(scratch.centerCvIds.size())) {
            item.kind = 0;
            break;
        }
        item.id = scratch.centerTubeIds[size_t(index)];
        item.subId = scratch.centerCvIds[size_t(index)];
        break;
    case PomadePick_SectionCV:
        // The shared PomadePickSets ABI carries a compact (index, subIndex)
        // spelling. Reconstruct its flat candidate ordinal, then recover the
        // actual tube/ring/slot from the per-candidate scratch maps.
        {
            if (index < 0 || subIndex < 0 || scratch.ringVerts <= 0) {
                item.kind = 0;
                break;
            }
            size_t const flat = size_t(index) * size_t(scratch.ringVerts) +
                                size_t(subIndex);
            if (flat >= scratch.sectionTubeIds.size() ||
                flat >= scratch.sectionRingIds.size() ||
                flat >= scratch.sectionSlotIds.size()) {
                item.kind = 0;
                break;
            }
            item.id = scratch.sectionTubeIds[flat];
            item.subId = scratch.sectionRingIds[flat];
            item.subSubId = scratch.sectionSlotIds[flat];
        }
        break;
    case PomadePick_SectionRing:
        if (index < 0 || index >= int(scratch.ringTubeIds.size()) ||
            index >= int(scratch.ringIds.size())) {
            item.kind = 0;
            break;
        }
        item.id = scratch.ringTubeIds[size_t(index)];
        item.subId = scratch.ringIds[size_t(index)];
        break;
    case PomadePick_GraphNode:
        item.id = index >= 0 && index < int(scratch.nodeIds.size())
                      ? scratch.nodeIds[size_t(index)]
                      : -1;
        break;
    case PomadePick_GraphEdge:
    case PomadePick_Region:
        item.id = index;  // already the stable id (the sets carry it)
        break;
    case PomadePick_Guide:
        item.id = index;
        break;
    default:
        item.kind = 0;
        break;
    }
    return item;
}

PomadeSelectionItem
PomadeModel::SelectionItemFromHit(PomadePickHit const &hit) const
{
    if (!hit.hit) {
        return PomadeSelectionItem();
    }
    _PickScratch scratch;
    _BuildPickSetsLocked(&scratch);
    return _ItemFromCandidateLocked(scratch, hit.kind, hit.index,
                                    hit.subIndex);
}

void
PomadeModel::SelectionClear(uint32_t kindMask)
{
    uint64_t const before = _selection.Generation();
    _selection.Clear(kindMask);
    if (_selection.Generation() != before) {
        _dirty |= PomadeDirty_Selection;
        ++_version;
    }
}

void
PomadeModel::SelectionApply(PomadeSelectMode mode,
                           std::vector<PomadeSelectionItem> const &items)
{
    uint64_t const before = _selection.Generation();
    _selection.Apply(mode, items);
    if (_selection.Generation() != before) {
        _dirty |= PomadeDirty_Selection;
        ++_version;
    }
}

void
PomadeModel::SelectionSetHover(PomadeSelectionItem const &item)
{
    uint64_t const before = _selection.Generation();
    _selection.SetHover(item);
    if (_selection.Generation() != before) {
        _dirty |= PomadeDirty_Selection;
        ++_version;
    }
}

PomadeSelectionItem
PomadeModel::SelectionHover() const
{
    _selection.PruneTubes(_LiveTubeIdsLocked());
    return _selection.Hover();
}

std::vector<PomadeSelectionItem>
PomadeModel::SelectionItems(uint32_t kindMask) const
{
    _selection.PruneTubes(_LiveTubeIdsLocked());
    return _selection.Items(kindMask);
}

size_t
PomadeModel::SelectionCount(uint32_t kindMask) const
{
    _selection.PruneTubes(_LiveTubeIdsLocked());
    return _selection.Count(kindMask);
}

uint64_t
PomadeModel::SelectionGeneration() const
{
    _selection.PruneTubes(_LiveTubeIdsLocked());
    return _selection.Generation();
}

bool
PomadeModel::_ItemPositionLocked(PomadeSelectionItem const &item,
                                _PickScratch const &scratch,
                                float out[3]) const
{
    auto tubeCentroid = [&](int tubeId, float o[3]) {
        PomadeTubeDesc desc;
        if (!_TubeDescLocked(tubeId, &desc) || desc.centerX.empty()) {
            return false;
        }
        double cx = 0.0, cy = 0.0, cz = 0.0;
        for (size_t i = 0; i < desc.centerX.size(); ++i) {
            cx += desc.centerX[i];
            cy += desc.centerY[i];
            cz += desc.centerZ[i];
        }
        double const n = double(desc.centerX.size());
        o[0] = float(cx / n);
        o[1] = float(cy / n);
        o[2] = float(cz / n);
        return true;
    };
    // Section handles cannot use the root-only pick scratch for child
    // tubes: a child owns its own centre curve, sections and (for region
    // tubes) pinned root frame.  Resolve its displayed section directly
    // from the same descriptor/frame path as tessellation.
    auto sectionPosition = [&](int tubeId, int ring, int slot, bool center,
                               float o[3]) {
        PomadeTubeDesc desc;
        if (!_TubeDescLocked(tubeId, &desc) || ring < 0 ||
            ring >= int(desc.sections.size())) {
            return false;
        }
        PomadeTubeSection const &section = desc.sections[size_t(ring)];
        if (section.u.empty() || section.u.size() != section.v.size() ||
            (!center && (slot < 0 || slot >= int(section.u.size())))) {
            return false;
        }
        std::vector<PomadeFrame> frames;
        std::string err;
        if (!PomadeTubeFramesCpu(desc, &frames, &err)) {
            return false;
        }
        float cp[3];
        PomadeFrame frame;
        if (!PomadeSampleCenterCpu(desc, frames, section.t, &cp[0], &cp[1],
                                  &cp[2], &frame, &err)) {
            return false;
        }
        float const ct = std::cos(section.twist);
        float const st = std::sin(section.twist);
        auto point = [&](size_t i, float p[3]) {
            float const u = section.u[i] * section.scale;
            float const v = section.v[i] * section.scale;
            float const ru = u * ct - v * st;
            float const rv = u * st + v * ct;
            p[0] = cp[0] + frame.nx * ru + frame.bx * rv;
            p[1] = cp[1] + frame.ny * ru + frame.by * rv;
            p[2] = cp[2] + frame.nz * ru + frame.bz * rv;
            if (ring == 0 && desc.rootSurfaceOffsets.size() == section.u.size() * 3)
                for (int axis = 0; axis < 3; ++axis)
                    p[axis] += desc.rootSurfaceOffsets[i * 3 + size_t(axis)];
        };
        if (!center) {
            point(size_t(slot), o);
            return true;
        }
        double x = 0.0, y = 0.0, z = 0.0;
        for (size_t i = 0; i < section.u.size(); ++i) {
            float p[3];
            point(i, p);
            x += p[0];
            y += p[1];
            z += p[2];
        }
        double const n = double(section.u.size());
        o[0] = float(x / n);
        o[1] = float(y / n);
        o[2] = float(z / n);
        return true;
    };
    switch (item.kind) {
    case PomadePick_TubeVert:
        return tubeCentroid(item.id, out);
    case PomadePick_CenterCV: {
        PomadeTubeDesc desc;
        if (!_TubeDescLocked(item.id, &desc) || item.subId < 0 ||
            item.subId >= int(desc.centerX.size())) {
            return false;
        }
        std::string err;
        if (!PomadeCenterHandlePointCpu(desc, item.subId, &out[0], &out[1],
                                       &out[2], &err)) {
            // Match the visible/pick fallback for malformed imported data.
            out[0] = desc.centerX[size_t(item.subId)];
            out[1] = desc.centerY[size_t(item.subId)];
            out[2] = desc.centerZ[size_t(item.subId)];
        }
        return true;
    }
    case PomadePick_SectionCV: {
        return sectionPosition(item.id, item.subId, item.subSubId,
                               /*center*/ false, out);
    }
    case PomadePick_SectionRing: {
        return sectionPosition(item.id, item.subId, -1,
                               /*center*/ true, out);
    }
    case PomadePick_GraphNode: {
        PomadeGraphNode const *node = _graph.FindNode(item.id);
        if (!node || !node->alive) {
            return false;
        }
        out[0] = node->p[0];
        out[1] = node->p[1];
        out[2] = node->p[2];
        return true;
    }
    case PomadePick_GraphEdge: {
        PomadeGraphEdge const *edge = _graph.FindEdge(item.id);
        if (!edge || !edge->alive || edge->polyline.size() < 6) {
            return false;
        }
        size_t const mid = (edge->polyline.size() / 3) / 2;
        out[0] = edge->polyline[mid * 3 + 0];
        out[1] = edge->polyline[mid * 3 + 1];
        out[2] = edge->polyline[mid * 3 + 2];
        return true;
    }
    case PomadePick_Region: {
        for (size_t i = 0; i < scratch.regionIds.size(); ++i) {
            if (scratch.regionIds[i] == item.id) {
                out[0] = scratch.regionCenters[i * 3 + 0];
                out[1] = scratch.regionCenters[i * 3 + 1];
                out[2] = scratch.regionCenters[i * 3 + 2];
                return true;
            }
        }
        return false;
    }
    case PomadePick_Guide: {
        if (_guides.cvCount <= 0 || item.id < 0 ||
            item.id >= _guides.guideCount) {
            return false;
        }
        size_t const root = size_t(item.id) * size_t(_guides.cvCount) * 3;
        if (root + 2 >= _guides.points.size()) {
            return false;
        }
        out[0] = _guides.points[root + 0];
        out[1] = _guides.points[root + 1];
        out[2] = _guides.points[root + 2];
        return true;
    }
    case PomadePick_Level: {
        // A level's position is the centroid of the tubes at it, which is
        // where a "frame this level" gizmo belongs.
        double cx = 0.0, cy = 0.0, cz = 0.0;
        int n = 0;
        for (int id : _LiveTubeIdsLocked()) {
            PomadeTubeDesc desc;
            if (!_TubeDescLocked(id, &desc) || desc.level != item.id) {
                continue;
            }
            float c[3];
            if (tubeCentroid(id, c)) {
                cx += c[0];
                cy += c[1];
                cz += c[2];
                ++n;
            }
        }
        if (!n) {
            return false;
        }
        out[0] = float(cx / n);
        out[1] = float(cy / n);
        out[2] = float(cz / n);
        return true;
    }
    default:
        return false;
    }
}

bool
PomadeModel::SelectionBounds(float outMin[3], float outMax[3]) const
{
    if (!outMin || !outMax) {
        return false;
    }
    _selection.PruneTubes(_LiveTubeIdsLocked());
    _PickScratch scratch;
    _BuildPickSetsLocked(&scratch);
    bool any = false;
    float mn[3] = {0.0f, 0.0f, 0.0f};
    float mx[3] = {0.0f, 0.0f, 0.0f};
    for (PomadeSelectionItem const &item : _selection.Items(0)) {
        float p[3];
        if (!_ItemPositionLocked(item, scratch, p)) {
            continue;
        }
        for (int a = 0; a < 3; ++a) {
            mn[a] = any ? std::min(mn[a], p[a]) : p[a];
            mx[a] = any ? std::max(mx[a], p[a]) : p[a];
        }
        any = true;
    }
    if (!any) {
        return false;
    }
    for (int a = 0; a < 3; ++a) {
        outMin[a] = mn[a];
        outMax[a] = mx[a];
    }
    return true;
}

bool
PomadeModel::_ApplyRegionSelectLocked(
    std::vector<PomadePickCandidate> const &hits, _PickScratch const &scratch,
    PomadeSelectMode mode, uint32_t kindMask)
{
    std::vector<PomadeSelectionItem> items;
    items.reserve(hits.size());
    auto wholeTubePickable = [&](int tubeId) {
        PomadeTubeDesc desc;
        if (!_TubeDescLocked(tubeId, &desc)) {
            return false;
        }
        auto const drawIt = _levelDisplay.find(desc.level);
        LevelDisplay const draw = drawIt == _levelDisplay.end()
                                      ? LevelDisplay()
                                      : drawIt->second;
        return draw.visible && !draw.centersOnly &&
               _IsTubeVisibleInActiveCutLocked(tubeId) &&
               (_activeCutEnabled || _focusLevel <= 0 ||
                desc.level == _focusLevel);
    };
    for (PomadePickCandidate const &hit : hits) {
        PomadeSelectionItem const item =
            _ItemFromCandidateLocked(scratch, hit.kind, hit.index,
                                     hit.subIndex);
        if (item.kind &&
            (item.kind != PomadePick_TubeVert || wholeTubePickable(item.id))) {
            items.push_back(item);
        }
    }
    uint64_t const before = _selection.Generation();
    // A band over displayed controls is an edit selection, not a second way
    // to select every tessellated surface vertex underneath.  Keep the body
    // only when the band found no tube component at all. Section CVs are
    // more specific than their ring centroid; retaining both would apply a
    // translate twice to that one control.
    std::vector<int> componentTubes;
    bool hasSectionCV = false;
    for (PomadeSelectionItem const &item : items) {
        if (item.kind == PomadePick_CenterCV ||
            item.kind == PomadePick_SectionCV ||
            item.kind == PomadePick_SectionRing) {
            componentTubes.push_back(item.id);
            hasSectionCV = hasSectionCV || item.kind == PomadePick_SectionCV;
        }
    }
    if (!componentTubes.empty()) {
        std::sort(componentTubes.begin(), componentTubes.end());
        componentTubes.erase(
            std::unique(componentTubes.begin(), componentTubes.end()),
            componentTubes.end());
        items.erase(std::remove_if(items.begin(), items.end(),
                                   [hasSectionCV](PomadeSelectionItem const &item) {
            return item.kind == PomadePick_TubeVert ||
                   (hasSectionCV && item.kind == PomadePick_SectionRing);
        }), items.end());
        if (mode != PomadeSelect_Set) {
            _selection.RemoveWholeTubeItems(componentTubes);
        }
    }
    if (mode == PomadeSelect_Set) {
        // An empty band clears the kinds it was asked for: dragging over
        // nothing is how an artist deselects.
        _selection.Clear(kindMask);
    }
    _selection.Apply(mode == PomadeSelect_Set ? PomadeSelect_Add : mode, items);
    if (_selection.Generation() == before) {
        return false;
    }
    _dirty |= PomadeDirty_Selection;
    ++_version;
    return true;
}

#ifdef USDGEN_POMADE_HAS_CUDA
bool
PomadeModel::_SelectMaskDeviceLocked(float const *devicePositions,
                                    int candidateCount,
                                    float const viewProj[16], int w, int h,
                                    float x0, float y0, float x1, float y1,
                                    float const *xy, int pointCount,
                                    std::vector<unsigned char> *out) const
{
    if (!devicePositions || candidateCount <= 0 || !out || !_device ||
        !_device->stream) {
        return false;
    }
    if (_device->selectMask.size() < size_t(candidateCount) &&
        _device->selectMask.reset(size_t(candidateCount)) != cudaSuccess) {
        return false;
    }
    float const *devicePolygon = nullptr;
    if (xy && pointCount >= 3) {
        size_t const floats = size_t(pointCount) * 2;
        if (_device->selectPolygon.size() < floats &&
            _device->selectPolygon.reset(floats) != cudaSuccess) {
            return false;
        }
        if (cudaMemcpy(_device->selectPolygon.data(), xy,
                       floats * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }
        devicePolygon = _device->selectPolygon.data();
    }
    char errBuf[256] = {0};
    if (!PomadeLaunchSelectMask(devicePositions, candidateCount, viewProj, w,
                               h, x0, y0, x1, y1, devicePolygon,
                               devicePolygon ? pointCount : 0,
                               _device->selectMask.data(), _device->stream,
                               errBuf, sizeof(errBuf))) {
        return false;
    }
    if (cudaStreamSynchronize(_device->stream) != cudaSuccess) {
        return false;
    }
    out->resize(size_t(candidateCount));
    return cudaMemcpy(out->data(), _device->selectMask.data(),
                      size_t(candidateCount) * sizeof(unsigned char),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

void
PomadeModel::_MaskHeavyKindsLocked(PomadePickSets const &sets,
                                  uint32_t kindMask,
                                  float const viewProj[16], int w, int h,
                                  float x0, float y0, float x1, float y1,
                                  float const *xy, int pointCount,
                                  std::vector<unsigned char> *tubeMask,
                                  std::vector<unsigned char> *guideMask,
                                  unsigned char const **tubePtr,
                                  unsigned char const **guidePtr) const
{
    *tubePtr = nullptr;
    *guidePtr = nullptr;
    if (!_device || !_device->streamOwned || !_device->stream ||
        CpuPickForced()) {
        return;
    }
    if ((kindMask & PomadePick_TubeVert) &&
        sets.tubeVertCount > kPickDeviceThreshold &&
        _device->positions.size() == size_t(sets.tubeVertCount) * 3) {
        if (_SelectMaskDeviceLocked(_device->positions.data(),
                                    sets.tubeVertCount, viewProj, w, h, x0,
                                    y0, x1, y1, xy, pointCount, tubeMask)) {
            *tubePtr = tubeMask->data();
        }
    }
    if ((kindMask & PomadePick_Guide) && sets.guideCount > 0 &&
        sets.guideCvCount > 0) {
        size_t const flat =
            size_t(sets.guideCount) * size_t(sets.guideCvCount);
        if (flat > size_t(kPickDeviceThreshold) && flat <= size_t(INT_MAX) &&
            _UploadGuideCVsLocked(sets) &&
            _SelectMaskDeviceLocked(_device->guideCVs.data(), int(flat),
                                    viewProj, w, h, x0, y0, x1, y1, xy,
                                    pointCount, guideMask)) {
            *guidePtr = guideMask->data();
        }
    }
}
#endif

bool
PomadeModel::SelectRect(float const viewProj[16], int w, int h, float x0,
                       float y0, float x1, float y1, uint32_t kindMask,
                       PomadeSelectMode mode)
{
    if (!viewProj || w <= 0 || h <= 0) {
        _diagnostic = "PomadeModel::SelectRect: bad viewport";
        return false;
    }
    _selection.PruneTubes(_LiveTubeIdsLocked());
    _PickScratch scratch;
    PomadePickSets const sets = _BuildPickSetsLocked(
        &scratch, kindMask & (PomadePick_GraphNode | PomadePick_GraphEdge));
    std::vector<unsigned char> tubeMask;
    std::vector<unsigned char> guideMask;
    unsigned char const *tubePtr = nullptr;
    unsigned char const *guidePtr = nullptr;
#ifdef USDGEN_POMADE_HAS_CUDA
    _MaskHeavyKindsLocked(sets, kindMask, viewProj, w, h, x0, y0, x1, y1,
                          nullptr, 0, &tubeMask, &guideMask, &tubePtr,
                          &guidePtr);
#endif
    std::vector<PomadePickCandidate> hits;
    if (!PomadeSelectRectCpu(sets, kindMask, viewProj, w, h, x0, y0, x1, y1,
                            tubePtr, guidePtr, &hits)) {
        _diagnostic = "PomadeModel::SelectRect: bad arguments";
        return false;
    }
    _ApplyRegionSelectLocked(hits, scratch, mode, kindMask);
    return true;
}

bool
PomadeModel::SelectPolygon(float const viewProj[16], int w, int h,
                          float const *xy, int pointCount, uint32_t kindMask,
                          PomadeSelectMode mode)
{
    if (!viewProj || w <= 0 || h <= 0 || !xy || pointCount < 3) {
        _diagnostic = "PomadeModel::SelectPolygon: a lasso needs >= 3 points";
        return false;
    }
    _selection.PruneTubes(_LiveTubeIdsLocked());
    _PickScratch scratch;
    PomadePickSets const sets = _BuildPickSetsLocked(
        &scratch, kindMask & (PomadePick_GraphNode | PomadePick_GraphEdge));
    std::vector<unsigned char> tubeMask;
    std::vector<unsigned char> guideMask;
    unsigned char const *tubePtr = nullptr;
    unsigned char const *guidePtr = nullptr;
#ifdef USDGEN_POMADE_HAS_CUDA
    _MaskHeavyKindsLocked(sets, kindMask, viewProj, w, h, 0.0f, 0.0f, 0.0f,
                          0.0f, xy, pointCount, &tubeMask, &guideMask,
                          &tubePtr, &guidePtr);
#endif
    std::vector<PomadePickCandidate> hits;
    if (!PomadeSelectPolygonCpu(sets, kindMask, viewProj, w, h, xy, pointCount,
                               tubePtr, guidePtr, &hits)) {
        _diagnostic = "PomadeModel::SelectPolygon: bad arguments";
        return false;
    }
    _ApplyRegionSelectLocked(hits, scratch, mode, kindMask);
    return true;
}

// -- V1: gizmo and brush overlays (plan/18 §2.4) -----------------------------

bool
PomadeModel::SetGizmo(PomadeGizmoRecord const &record)
{
    if (record.kind < PomadeGizmo_None ||
        record.kind > PomadeGizmo_Scale) {
        _diagnostic = "PomadeModel::SetGizmo: unknown gizmo kind";
        return false;
    }
    if (_gizmo == record) {
        return true;  // the move loop re-sets an unchanged gizmo per sample
    }
    _gizmo = record;
    _dirty |= PomadeDirty_Gizmo;
    ++_version;
    return true;
}

PomadeGizmoRecord
PomadeModel::GetGizmo() const
{
    return _gizmo;
}

bool
PomadeModel::SetBrushRing(PomadeBrushRingRecord const &record)
{
    if (_brush == record) {
        return true;
    }
    _brush = record;
    _dirty |= PomadeDirty_Brush;
    ++_version;
    return true;
}

PomadeBrushRingRecord
PomadeModel::GetBrushRing() const
{
    return _brush;
}

// -- P4: hierarchy store -----------------------------------------------------

namespace {

// Ring-layout match: same ringVerts, section count and t values. K7
// aggregates carry the child layout; writeback refits them to the target
// tube's own layout -- EXCEPT on a match, where the aggregate passes
// through untouched so hint-exact merges stay bit-exact.
bool _SameRingLayout(PomadeTubeDesc const &a, PomadeTubeDesc const &b)
{
    if (a.ringVerts != b.ringVerts ||
        a.sections.size() != b.sections.size()) {
        return false;
    }
    for (size_t i = 0; i < a.sections.size(); ++i) {
        if (a.sections[i].t != b.sections[i].t) {
            return false;
        }
    }
    return true;
}

bool _FitAggregateForWriteback(PomadeTubeDesc const &agg,
                               PomadeTubeDesc const &current,
                               PomadeTubeDesc *out, std::string *derr)
{
    if (_SameRingLayout(agg, current)) {
        *out = agg;
        return true;
    }
    return PomadeResampleDescRingsCpu(agg, current.ringVerts, out, derr);
}

// K14 output gate.  Every ring K9 will triangulate for this tube -- the
// authored section t's and the common fill stations of `cvCount` CVs --
// must be a simple polygon, or the refill drops the whole tube.  The chart
// is built exactly as PomadeBuildGuideMaterialBindingsCpu builds it (ring
// minus center, projected on the frame's normal/binormal), and a complete
// point taper passes as it does there.  A subdivide, merge or K6
// re-derivation that fails here is rejected and rolled back like any
// other K7/K14 reject; never loosen the triangulator instead: a twisted
// ring has no honest material chart.
bool _ValidateChildMaterialRings(PomadeTubeDesc const &tube, int cvCount,
                                 char const *stage, std::string *err)
{
    auto fail = [&](std::string const &text) {
        if (err) {
            *err = text;
        }
        return false;
    };
    std::string const who =
        std::string(stage) + ": tube " + std::to_string(tube.tubeId);
    int const rv = tube.ringVerts;
    if (rv < 3 || tube.sections.size() < 2) {
        return fail(who + " has a malformed section layout");
    }
    std::vector<PomadeFrame> frames;
    std::string derr;
    if (!PomadeTubeFramesCpu(tube, &frames, &derr)) {
        return fail(who + ": " + derr);
    }
    std::vector<float> stations;
    for (PomadeTubeSection const &section : tube.sections) {
        stations.push_back(section.t);
    }
    float const t0 = tube.sections.front().t;
    float const t1 = tube.sections.back().t;
    float const span = t1 > t0 ? t1 - t0 : 1.0f;
    for (int c = 1; c + 1 < cvCount; ++c) {
        stations.push_back(t0 + span * (float(c) / float(cvCount - 1)));
    }
    std::vector<float> ring;
    std::vector<PXR_NS::GfVec2f> chart(static_cast<size_t>(rv));
    std::vector<std::array<int, 3>> triangles;
    for (float const t : stations) {
        PomadeFrame frame;
        float center[3] = {0.0f, 0.0f, 0.0f};
        if (!PomadeSampleTubeRingCpu(tube, frames, t, &ring, &derr) ||
            !PomadeSampleCenterCpu(tube, frames, t, &center[0], &center[1],
                                  &center[2], &frame, &derr) ||
            ring.size() != size_t(rv) * 3) {
            return fail(who + ": " + derr);
        }
        float maxRadius2 = 0.0f;
        float maxExtent2 = 0.0f;
        for (int slot = 0; slot < rv; ++slot) {
            size_t const at = size_t(slot) * 3;
            float const dx = ring[at + 0] - center[0];
            float const dy = ring[at + 1] - center[1];
            float const dz = ring[at + 2] - center[2];
            float const u = dx * frame.nx + dy * frame.ny + dz * frame.nz;
            float const v = dx * frame.bx + dy * frame.by + dz * frame.bz;
            chart[size_t(slot)] = PXR_NS::GfVec2f(u, v);
            maxRadius2 = std::max(maxRadius2, u * u + v * v);
            float const du = u - chart[0][0];
            float const dv = v - chart[0][1];
            maxExtent2 = std::max(maxExtent2, du * du + dv * dv);
        }
        float const epsilon = std::numeric_limits<float>::epsilon();
        bool const collapsed = maxExtent2 == 0.0f ||
            (maxRadius2 > 0.0f &&
             maxExtent2 <= 64.0f * epsilon * epsilon * maxRadius2);
        if (collapsed) {
            continue;
        }
        std::string triangulateErr;
        if (!usdGen::UsdGenTriangulateConcaveMaterialSlots(
                chart, &triangles, &triangulateErr)) {
            return fail(who + " ring at t=" + std::to_string(t) +
                        " is not a simple polygon (" + triangulateErr +
                        "); try another seed or smooth the parent");
        }
    }
    return true;
}

// The K6 re-derivation gate: the same check for a descendant rebuilt from
// fresh K14 output plus its stored residuals. Per-slot residuals applied to
// a re-split whose slot layout changed can twist one section just as a
// misaligned split does, and the refill would then drop the tube. The
// material chart is the K5 section interpolation expressed in the tube's
// own frame, so only the sections decide it: an edit that left them
// bit-identical cannot have broken them, and skipping it keeps a drag's
// per-move cost to the tubes whose rings actually changed.
bool _ValidateRederivedChild(PomadeTubeDesc const &before,
                             PomadeTubeDesc const &after, int cvCount,
                             std::string *err)
{
    bool same = before.ringVerts == after.ringVerts &&
                before.sections.size() == after.sections.size();
    for (size_t i = 0; same && i < after.sections.size(); ++i) {
        PomadeTubeSection const &a = before.sections[i];
        PomadeTubeSection const &b = after.sections[i];
        same = a.t == b.t && a.scale == b.scale && a.twist == b.twist &&
               a.u == b.u && a.v == b.v;
    }
    return same || _ValidateChildMaterialRings(
                       after, cvCount, "PomadeHierarchicalSculptApplyCpu", err);
}

}  // namespace

bool PomadeModel::_TubeDescLocked(int tubeId, PomadeTubeDesc *out) const
{
    if (!out) {
        return false;
    }
    if (tubeId == 0) {
        if (_host.centerX.size() < 2) {
            return false;
        }
        *out = _BuildTubeDescLocked();
        out->tubeId = 0;
        out->level = 1;
        out->regionId = _tubeRegionId;
        out->parentTubeId = -1;
        out->childIndex = -1;
        return true;
    }
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end()) {
        return false;
    }
    *out = it->second.actual;
    _AttachTubeRootLocked(out);
    return true;
}

bool PomadeModel::_SetCenterCVLocked(int cv, float x, float y, float z)
{
    int const n = int(_host.centerX.size());
    if (cv < 0 || cv >= n || n < 2) {
        _diagnostic = "PomadeModel: CV index out of range";
        return false;
    }
    _host.centerX[size_t(cv)] = x;
    _host.centerY[size_t(cv)] = y;
    _host.centerZ[size_t(cv)] = z;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Points;
    return true;
}

bool PomadeModel::_WriteDescToTube0Locked(PomadeTubeDesc const &desc)
{
    int const nCv = int(desc.centerX.size());
    if (nCv < 2 || desc.sections.size() < 2 || desc.ringVerts < 3 ||
        desc.ringVerts > 32) {
        _diagnostic = "PomadeModel: merged desc fails tube-0 limits";
        return false;
    }
    _shape.rings = nCv;
    _shape.ringVerts = desc.ringVerts;
    _host.centerX = desc.centerX;
    _host.centerY = desc.centerY;
    _host.centerZ = desc.centerZ;
    _sections = desc.sections;
    _useSections = true;
    _rootFramePinned = desc.rootFramePinned;
    _rootFrame = desc.rootFrame;
    _frameReference = desc.frameReference;
    _roots.clear();
    _tubeRoots.clear();
    _guides = PomadeGuideSet();
    ++_guideVersion;
    if (!Sync()) {
        return false;
    }
    _dirty |= PomadeDirty_Topology;
    return true;
}

bool
PomadeModel::_ConformRegionRootSectionLocked(
    int tubeId, PomadeScalpGraph const &oldGraph, int oldRegionId,
    int newRegionId, PomadeTubeDesc const &oldActual, bool *outChanged)
{
    if (outChanged) {
        *outChanged = false;
    }
    if (!_scalp) {
        _diagnostic = "PomadeModel::GraphMoveNodes: missing scalp for "
                      "attachment conformance";
        return false;
    }
    _RegionFootprint oldFootprint, newFootprint;
    std::string err;
    if (!_RegionCanonicalFootprint(*_scalp, oldGraph, oldRegionId,
                                   &oldFootprint, &err) ||
        !_RegionCanonicalFootprint(*_scalp, _graph, newRegionId,
                                   &newFootprint, &err)) {
        _diagnostic = err.empty()
                          ? "PomadeModel::GraphMoveNodes: invalid attachment "
                            "footprint"
                          : err;
        return false;
    }
    // The baseline sample of a gesture must be a true no-op, even for a
    // legacy root whose stored footprint is already stale. Repair is an
    // intentional response to an edited region, never a side effect of
    // pressing/releasing without moving a graph CV.
    if (_SameRegionFootprint(oldFootprint, newFootprint)) {
        return true;
    }
    PomadeTubeDesc current;
    if (!_TubeDescLocked(tubeId, &current) || current.centerX.empty() ||
        current.sections.empty() || current.ringVerts < 3 ||
        current.ringVerts != oldActual.ringVerts ||
        int(current.sections.front().u.size()) != current.ringVerts ||
        int(current.sections.front().v.size()) != current.ringVerts) {
        _diagnostic = "PomadeModel::GraphMoveNodes: invalid attached root";
        return false;
    }
    std::vector<float> const slots =
        _RecoverFootprintSlots(oldFootprint, oldActual);
    if (int(slots.size()) != current.ringVerts) {
        _diagnostic = "PomadeModel::GraphMoveNodes: invalid attachment "
                      "slot mapping";
        return false;
    }

    // Keep the rigidly transported center cage intact.  CV0 may be offset
    // within the support chart (for example after an artist whole-tube move),
    // and moving it alone would rotate K4 above the root and shear every
    // upper sculpted section.  Fit the graph footprint through the existing
    // center/frame chart instead.
    PomadeTubeDesc candidate = current;
    PomadeTubeSection &section = candidate.sections.front();
    if (!(section.scale > 1e-6f) || !std::isfinite(section.scale) ||
        !std::isfinite(section.twist)) {
        _diagnostic = "PomadeModel::GraphMoveNodes: invalid attached root "
                      "section transform";
        return false;
    }
    std::vector<PomadeFrame> frames;
    if (!PomadeTubeFramesCpu(candidate, &frames, &err) || frames.empty()) {
        _diagnostic = err;
        return false;
    }
    PomadeFrame const &frame = frames.front();
    float const ct = std::cos(section.twist);
    float const st = std::sin(section.twist);
    std::vector<std::array<float, 3>> targets;
    targets.reserve(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        std::array<float, 3> const target =
            _FootprintPoint(newFootprint, slots[i]);
        targets.push_back(target);
        float const dx = target[0] - candidate.centerX[0];
        float const dy = target[1] - candidate.centerY[0];
        float const dz = target[2] - candidate.centerZ[0];
        float const ru = dx * frame.nx + dy * frame.ny + dz * frame.nz;
        float const rv = dx * frame.bx + dy * frame.by + dz * frame.bz;
        // Invert K5's scale then twist application.  Do not change these
        // scalar controls: they affect the upper interpolation too.
        section.u[i] = (ct * ru + st * rv) / section.scale;
        section.v[i] = (-st * ru + ct * rv) / section.scale;
    }

    float extent = 0.0f;
    for (std::array<float, 3> const &target : targets) {
        float const dx = target[0] - newFootprint.origin[0];
        float const dy = target[1] - newFootprint.origin[1];
        float const dz = target[2] - newFootprint.origin[2];
        extent = std::max(extent, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    float const tolerance = std::max(1e-5f, extent * 1e-5f);
    std::vector<std::array<float, 3>> candidateBoundary;
    if (!_SectionZeroWorld(candidate, &candidateBoundary) ||
        candidateBoundary.size() != targets.size()) {
        _diagnostic = "PomadeModel::GraphMoveNodes: invalid fitted "
                      "attachment section";
        return false;
    }
    bool same = true;
    std::vector<std::array<float, 3>> currentBoundary;
    if (!_SectionZeroWorld(current, &currentBoundary) ||
        currentBoundary.size() != targets.size()) {
        same = false;
    }
    for (size_t i = 0; i < targets.size(); ++i) {
        float const ex = candidateBoundary[i][0] - targets[i][0];
        float const ey = candidateBoundary[i][1] - targets[i][1];
        float const ez = candidateBoundary[i][2] - targets[i][2];
        if (ex * ex + ey * ey + ez * ez > tolerance * tolerance) {
            _diagnostic = "PomadeModel::GraphMoveNodes: attachment root "
                          "chart cannot represent region footprint";
            return false;
        }
        if (same) {
            float const dx = currentBoundary[i][0] - targets[i][0];
            float const dy = currentBoundary[i][1] - targets[i][1];
            float const dz = currentBoundary[i][2] - targets[i][2];
            if (dx * dx + dy * dy + dz * dz > tolerance * tolerance) {
                same = false;
            }
        }
    }
    if (same) {
        return true;  // Exact rigid transport: retain every stored residual.
    }

    if (tubeId == 0) {
        _host.centerX[0] = candidate.centerX[0];
        _host.centerY[0] = candidate.centerY[0];
        _host.centerZ[0] = candidate.centerZ[0];
        _sections.front() = std::move(candidate.sections.front());
    } else {
        auto it = _tubes.find(tubeId);
        if (it == _tubes.end()) {
            return false;
        }
        it->second.actual = candidate;
        // L1 roots retain their rigidly transported center cage.  Keep the
        // reference aligned with the section-zero-only attachment edit;
        // children get fresh attachment references below.
        it->second.derived = std::move(candidate);
    }
    if (outChanged) {
        *outChanged = true;
    }
    return true;
}

bool
PomadeModel::_PropagateAttachmentDownLocked(int tubeId)
{
    std::vector<int> kids;
    for (auto const &kv : _tubes) {
        if (kv.second.actual.parentTubeId == tubeId) {
            kids.push_back(kv.first);
        }
    }
    if (kids.empty()) {
        return true;
    }
    PomadeTubeDesc parent;
    if (!_TubeDescLocked(tubeId, &parent)) {
        return false;
    }
    std::vector<PomadeFrame> parentFrames;
    std::string err;
    if (!PomadeTubeFramesCpu(parent, &parentFrames, &err)) {
        _diagnostic = err;
        return false;
    }
    PomadeSubdivideDesc params;
    if (tubeId == 0) {
        params.count = _subdivide.count;
        params.seed = _subdivide.seed;
        params.splitMode = _subdivide.splitMode == "edge" ? PomadeSplit_Edge
                                                            : PomadeSplit_KMeans;
        params.edgeA = _subdivide.edgeA;
        params.edgeB = _subdivide.edgeB;
        params.edgeC = _subdivide.edgeC;
    } else {
        auto parentIt = _tubes.find(tubeId);
        if (parentIt == _tubes.end()) {
            return false;
        }
        params = parentIt->second.subdivide;
    }
    std::vector<PomadeTubeDesc> derivedAll;
    if (!PomadeSubdivideTubeCpu(parent, parentFrames, params, &derivedAll,
                               &err)) {
        _diagnostic = err;
        return false;
    }
    for (int kid : kids) {
        auto it = _tubes.find(kid);
        if (it == _tubes.end()) {
            return false;
        }
        HierarchyTube &entry = it->second;
        if (entry.imported) {
            continue;
        }
        int const childIndex = entry.actual.childIndex;
        if (childIndex < 0 || size_t(childIndex) >= derivedAll.size()) {
            _diagnostic = "PomadeDeriveChildCpu: child index out of range";
            return false;
        }
        PomadeTubeDesc fresh = derivedAll[size_t(childIndex)];
        if (!_SameRingLayout(fresh, entry.actual)) {
            PomadeTubeDesc matched;
            if (!PomadeResampleDescRingsCpu(fresh, entry.actual.ringVerts,
                                            &matched, &err)) {
                _diagnostic = err;
                return false;
            }
            fresh = std::move(matched);
        }
        // All descendants were already transported by the region's proper
        // support transform.  A changed root boundary gives K14 a new
        // partition reference, but it must not make K6 regenerate the
        // child's sculpted upper cage from that new partition.  Preserve the
        // transported actual descriptor and fit only the fresh inherited
        // root polygon into its existing material chart; then remeasure its
        // residual against `fresh` below.
        PomadeTubeDesc actual = entry.actual;
        std::vector<std::array<float, 3>> freshRoot;
        if (!_SectionZeroWorld(fresh, &freshRoot) ||
            !_FitSectionZeroWorld(&actual, freshRoot, &err)) {
            _diagnostic = err.empty()
                ? "PomadeModel::GraphMoveNodes: cannot fit child attachment"
                : err;
            return false;
        }
        // The refit root is a fresh K14 split, and K14 starts its ring at
        // whichever parent corner first falls in the clip: against the
        // transported section 1 it can arrive rotated, and K5 would twist
        // the root span into a figure-eight at the first fill stations.
        // Find the renumbering that aligns the root to its neighbour (as
        // PomadeSubdivideTubeCpu aligns consecutive stations), but apply its
        // inverse to every UPPER section instead of renumbering the root:
        // the root keeps the fresh split's slot order, K14 anchors section
        // 0 the same way, so the actual and `fresh` share one slot order at
        // every station and the residual below stays sculpt-sized.
        // (Renumbering only the root stored a rotation-sized section-0
        // residual: an unsculpted child turned "sculpted", and any later
        // K6 re-split that started its root on another corner re-applied
        // it to the wrong slots and was refused as a non-simple ring.)
        // Every upper section moves by the same permutation, so the
        // actual's own section-to-section correspondence -- its surface --
        // is unchanged; only slot numbers move.
        std::vector<int> rootFrom;
        if (actual.sections.size() > 1) {
            PomadeTubeSection probe = actual.sections[0];
            PomadeAlignSectionRingCpu(&probe, actual.sections[1], &rootFrom);
        }
        size_t const slotCount = rootFrom.size();
        bool const relabelUpper =
            slotCount > 0 &&
            std::all_of(actual.sections.begin() + 1, actual.sections.end(),
                        [slotCount](PomadeTubeSection const &sec) {
                            return sec.u.size() == slotCount &&
                                   sec.v.size() == slotCount;
                        });
        if (relabelUpper) {
            // probe[i] = root[rootFrom[i]] pairs with upper[i]; so upper
            // slot i moves to rootFrom[i].
            for (size_t s = 1; s < actual.sections.size(); ++s) {
                PomadeTubeSection &sec = actual.sections[s];
                std::vector<float> u(slotCount), v(slotCount);
                for (size_t i = 0; i < slotCount; ++i) {
                    u[size_t(rootFrom[i])] = sec.u[i];
                    v[size_t(rootFrom[i])] = sec.v[i];
                }
                sec.u = std::move(u);
                sec.v = std::move(v);
            }
            for (PomadeParentBoundaryBinding &binding :
                 actual.inheritedBoundaryBindings) {
                if (binding.section > 0 && binding.childSlot >= 0 &&
                    size_t(binding.childSlot) < slotCount) {
                    binding.childSlot = rootFrom[size_t(binding.childSlot)];
                }
            }
        }
        // PomadeHierarchicalSculptApplyCpu retains old K14 triples when the
        // layouts match.  Those root slots now name freshly derived geometry,
        // so discard only section-zero triples and reconstruct exact retained
        // parent corners from the installed child/parent root sections.
        std::vector<PomadeParentBoundaryBinding> rootBindings;
        if (!_InstalledRootBoundaryBindings(parent, actual, &rootBindings)) {
            _diagnostic = "PomadeModel::GraphMoveNodes: invalid child "
                          "attachment boundary";
            return false;
        }
        actual.inheritedBoundaryBindings.erase(
            std::remove_if(actual.inheritedBoundaryBindings.begin(),
                           actual.inheritedBoundaryBindings.end(),
                           [](PomadeParentBoundaryBinding const &binding) {
                               return binding.section == 0;
                           }),
            actual.inheritedBoundaryBindings.end());
        actual.inheritedBoundaryBindings.insert(
            actual.inheritedBoundaryBindings.end(), rootBindings.begin(),
            rootBindings.end());
        fresh.inheritedBoundaryBindings.erase(
            std::remove_if(fresh.inheritedBoundaryBindings.begin(),
                           fresh.inheritedBoundaryBindings.end(),
                           [](PomadeParentBoundaryBinding const &binding) {
                               return binding.section == 0;
                           }),
            fresh.inheritedBoundaryBindings.end());
        // rootBindings name the installed root slots, which keep the fresh
        // reference's slot order.
        fresh.inheritedBoundaryBindings.insert(
            fresh.inheritedBoundaryBindings.end(), rootBindings.begin(),
            rootBindings.end());
        std::vector<PomadeFrame> childFrames;
        PomadeShapeDeltas stored;
        if (!PomadeTubeFramesCpu(fresh, &childFrames, &err) ||
            !PomadeComputeDeltasCpu(actual, fresh, childFrames, &stored,
                                   &err)) {
            _diagnostic = err;
            return false;
        }
        // The refit root of a tube that fills must stay triangulable too;
        // the graph-move callers restore the hierarchy on false.
        if (!_FillSuspendedLocked(kid) &&
            !_ValidateRederivedChild(entry.actual, actual,
                                     entry.fill.cvCount, &err)) {
            _diagnostic = err;
            return false;
        }
        entry.actual = std::move(actual);
        entry.derived = std::move(fresh);
        entry.deltas = std::move(stored);
        if (relabelUpper) {
            // Grandchildren bind this child's upper slots by number: follow
            // the relabel (their section-0 triples are rebuilt below).
            for (auto &gkv : _tubes) {
                if (gkv.second.actual.parentTubeId != kid) {
                    continue;
                }
                for (PomadeTubeDesc *desc :
                     {&gkv.second.actual, &gkv.second.derived}) {
                    for (PomadeParentBoundaryBinding &binding :
                         desc->inheritedBoundaryBindings) {
                        if (binding.section > 0 && binding.parentSlot >= 0 &&
                            size_t(binding.parentSlot) < slotCount) {
                            binding.parentSlot =
                                rootFrom[size_t(binding.parentSlot)];
                        }
                    }
                }
            }
        }
        if (!_PropagateAttachmentDownLocked(kid)) {
            return false;
        }
    }
    return true;
}

bool PomadeModel::_PropagateDownLocked(int tubeId)
{
    // Children in ascending id order (map order is deterministic).
    std::vector<int> kids;
    for (auto const &kv : _tubes) {
        if (kv.second.actual.parentTubeId == tubeId) {
            kids.push_back(kv.first);
        }
    }
    if (kids.empty()) {
        return true;
    }
    PomadeTubeDesc parentDesc;
    if (!_TubeDescLocked(tubeId, &parentDesc)) {
        return false;
    }
    std::vector<PomadeFrame> frames;
    std::string derr;
    if (!PomadeTubeFramesCpu(parentDesc, &frames, &derr)) {
        _diagnostic = derr;
        return false;
    }
    // The params this parent split with (tube 0: the model params).
    PomadeSubdivideDesc params;
    if (tubeId == 0) {
        params.count = _subdivide.count;
        params.seed = _subdivide.seed;
        params.splitMode =
            _subdivide.splitMode == "edge" ? PomadeSplit_Edge : PomadeSplit_KMeans;
        params.edgeA = _subdivide.edgeA;
        params.edgeB = _subdivide.edgeB;
        params.edgeC = _subdivide.edgeC;
    } else {
        auto pit = _tubes.find(tubeId);
        if (pit == _tubes.end()) {
            return false;
        }
        params = pit->second.subdivide;
    }
    // TN-1: derive ALL children with one subdivision (was: a full
    // subdivide per child, twice — inside the sculpt and again for the
    // derived refresh). Same deterministic function, same inputs, so
    // the per-child descs — and everything downstream — are
    // bit-identical; only the redundant recomputation is gone.
    std::vector<PomadeTubeDesc> derivedAll;
    if (!PomadeSubdivideTubeCpu(parentDesc, frames, params, &derivedAll,
                               &derr)) {
        _diagnostic = derr;
        return false;
    }
    for (int kid : kids) {
        auto it = _tubes.find(kid);
        if (it == _tubes.end()) {
            return false;
        }
        HierarchyTube &entry = it->second;
        if (entry.imported) {
            continue;  // bridge imports keep their explicit shape: K6
                       // never re-derives them (and their subtree, rooted
                       // at an unchanged actual, cannot move either)
        }
        int const childIndex = entry.actual.childIndex;
        if (childIndex < 0 ||
            size_t(childIndex) >= derivedAll.size()) {
            // Same spelling the per-child derive used to report.
            _diagnostic = "PomadeDeriveChildCpu: child index out of range";
            return false;
        }
        // K14 may choose a different ring count after a parent move.  A
        // child re-based by K7 owns its current layout, so use that layout
        // for both the K6 application and the stored reference instead of
        // allowing a no-op K6 to silently re-sample its authored shape.
        PomadeTubeDesc fresh = derivedAll[size_t(childIndex)];
        if (!_SameRingLayout(fresh, entry.actual)) {
            PomadeTubeDesc matched;
            if (!PomadeResampleDescRingsCpu(fresh, entry.actual.ringVerts,
                                            &matched, &derr)) {
                _diagnostic = derr;
                return false;
            }
            fresh = std::move(matched);
        }
        bool const locked = _globalLockChildren || entry.lockChildren;
        PomadeTubeDesc actual;
        PomadeShapeDeltas stored;
        if (!PomadeHierarchicalSculptApplyCpu(
                fresh, entry.actual, entry.derived,
                entry.deltas, locked, /*preserveLength=*/true, &actual,
                &stored, &derr)) {
            _diagnostic = derr;
            return false;
        }
        // Every caller rolls the whole hierarchy back on false, exactly as
        // for a K14 reject above. Only a tube that fills is checked: a
        // subdivided parent's own stations feed no guides (its children
        // re-split its authored sections, not its interpolated rings).
        if (!_FillSuspendedLocked(kid) &&
            !_ValidateRederivedChild(entry.actual, actual,
                                     entry.fill.cvCount, &derr)) {
            _diagnostic = derr;
            return false;
        }
        entry.actual = actual;
        entry.derived = std::move(fresh);
        entry.deltas = stored;
        if (!_PropagateDownLocked(kid)) {
            return false;
        }
    }
    return true;
}

bool PomadeModel::_PropagateUpLocked(
    int tubeId, HierarchyRollback const *beforeEdit)
{
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end()) {
        return true;  // tube 0 has no parent
    }
    int const parentId = it->second.actual.parentTubeId;
    if (parentId < 0) {
        return true;
    }
    if (_globalLockParents || it->second.lockParents) {
        return true;  // the edited tube gates the whole upward chain
    }
    // Gather the parent's children actuals in ascending id order.  A direct
    // child edit supplies the pre-edit snapshot in the same stable order so
    // K7 can inspect only frozen inherited boundary slots.  Comparing a
    // child to a fresh K14 derivation is invalid here: its local frames may
    // already differ from the parent before the edit.
    std::vector<PomadeTubeDesc> sibs;
    std::vector<PomadeTubeDesc> priorSibs;
    bool havePriorSibs = beforeEdit != nullptr;
    for (auto const &kv : _tubes) {
        if (kv.second.actual.parentTubeId == parentId) {
            sibs.push_back(kv.second.actual);
            if (havePriorSibs) {
                auto const old = beforeEdit->tubes.find(kv.first);
                if (old == beforeEdit->tubes.end() ||
                    old->second.actual.parentTubeId != parentId) {
                    // Do not make an incomplete snapshot look like a stable
                    // before-state.  The merge retains its legacy path.
                    havePriorSibs = false;
                    priorSibs.clear();
                } else {
                    priorSibs.push_back(old->second.actual);
                }
            }
        }
    }
    if (sibs.empty()) {
        return true;
    }
    PomadeTubeDesc parentDesc;
    if (!_TubeDescLocked(parentId, &parentDesc)) {
        return false;
    }
    PomadeSubdivideDesc params;
    if (parentId == 0) {
        params.count = _subdivide.count;
        params.seed = _subdivide.seed;
        params.splitMode =
            _subdivide.splitMode == "edge" ? PomadeSplit_Edge : PomadeSplit_KMeans;
        params.edgeA = _subdivide.edgeA;
        params.edgeB = _subdivide.edgeB;
        params.edgeC = _subdivide.edgeC;
    } else {
        auto pit = _tubes.find(parentId);
        if (pit == _tubes.end()) {
            return false;
        }
        if (pit->second.imported) {
            // Bridge imports keep their explicit shape: K7 reads them
            // (as siblings, above) but never overwrites them. The chain
            // still continues upward so ancestors refresh against the
            // import's unchanged actual.
            return _PropagateUpLocked(parentId, beforeEdit);
        }
        params = pit->second.subdivide;
    }
    std::string derr;
    PomadeTubeDesc agg;
    if (!PomadeMergeTubesCpu(sibs, params, &parentDesc, &agg, &derr,
                            havePriorSibs ? &priorSibs : nullptr)) {
        _diagnostic = derr;
        return false;
    }
    // K7 writes a new actual parent from all of its children.  The children
    // themselves did not move, so do not run K6 here: that would apply their
    // old residuals to the new parent a second time.  Instead make each
    // direct child's reference the fresh split of the written parent and
    // measure its unchanged actual against that reference.  A later K6 then
    // starts from this matched pair instead of stale pre-K7 geometry.
    auto rebaseDirectChildren = [&](int rewrittenParentId) {
        PomadeTubeDesc writtenParent;
        if (!_TubeDescLocked(rewrittenParentId, &writtenParent)) {
            return false;
        }
        std::vector<PomadeFrame> parentFrames;
        if (!PomadeTubeFramesCpu(writtenParent, &parentFrames, &derr)) {
            _diagnostic = derr;
            return false;
        }
        std::vector<PomadeTubeDesc> freshAll;
        if (!PomadeSubdivideTubeCpu(writtenParent, parentFrames, params,
                                   &freshAll, &derr)) {
            _diagnostic = derr;
            return false;
        }
        for (auto &kv : _tubes) {
            HierarchyTube &child = kv.second;
            if (child.actual.parentTubeId != rewrittenParentId ||
                child.imported) {
                continue;
            }
            int const childIndex = child.actual.childIndex;
            if (childIndex < 0 || size_t(childIndex) >= freshAll.size()) {
                _diagnostic = "PomadeDeriveChildCpu: child index out of range";
                return false;
            }
            // K14 can repartition ring-vertex counts.  Preserve the authored
            // actual layout exactly; only resample the newly derived reference
            // before computing its residual.
            PomadeTubeDesc fresh = freshAll[size_t(childIndex)];
            if (!_SameRingLayout(fresh, child.actual)) {
                PomadeTubeDesc matched;
                if (!PomadeResampleDescRingsCpu(fresh,
                                                child.actual.ringVerts,
                                                &matched, &derr)) {
                    _diagnostic = derr;
                    return false;
                }
                fresh = std::move(matched);
            }
            // Fresh K14 bindings describe its own generated material slots.
            // The actual child keeps its frozen inherited-corner identity:
            // matching a reference layout does not prove the two slot orders
            // represent the same source edge.
            std::vector<PomadeFrame> childFrames;
            if (!PomadeTubeFramesCpu(fresh, &childFrames, &derr)) {
                _diagnostic = derr;
                return false;
            }
            PomadeShapeDeltas rebased;
            if (!PomadeComputeDeltasCpu(child.actual, fresh, childFrames,
                                       &rebased, &derr)) {
                _diagnostic = derr;
                return false;
            }
            child.derived = std::move(fresh);
            child.deltas = std::move(rebased);
        }
        return true;
    };
    if (parentId == 0) {
        // Keep tube 0 on its own ring layout (a layout change here would
        // desync every later K6 re-derivation); pass hint-exact merges
        // through untouched.
        PomadeTubeDesc fit;
        if (!_FitAggregateForWriteback(agg, parentDesc, &fit, &derr)) {
            _diagnostic = derr;
            return false;
        }
        if (!_WriteDescToTube0Locked(fit)) {
            return false;
        }
        return rebaseDirectChildren(parentId);
    }
    auto pit = _tubes.find(parentId);
    if (pit == _tubes.end()) {
        return false;
    }
    // Refit the aggregate to the parent's own ring layout, then refresh
    // its deltas against its derived baseline (hint-exact merges pass
    // through untouched).
    PomadeTubeDesc refit;
    if (!_FitAggregateForWriteback(agg, pit->second.actual, &refit, &derr)) {
        _diagnostic = derr;
        return false;
    }
    refit.tubeId = pit->second.actual.tubeId;
    refit.regionId = pit->second.actual.regionId;
    refit.level = pit->second.actual.level;
    refit.parentTubeId = pit->second.actual.parentTubeId;
    refit.childIndex = pit->second.actual.childIndex;
    std::vector<PomadeFrame> dframes;
    if (!PomadeTubeFramesCpu(pit->second.derived, &dframes, &derr)) {
        _diagnostic = derr;
        return false;
    }
    PomadeShapeDeltas deltas;
    if (!PomadeComputeDeltasCpu(refit, pit->second.derived, dframes, &deltas,
                               &derr)) {
        _diagnostic = derr;
        return false;
    }
    pit->second.actual = refit;
    pit->second.deltas = deltas;
    if (!rebaseDirectChildren(parentId)) {
        return false;
    }
    return _PropagateUpLocked(parentId, beforeEdit);
}

bool PomadeModel::SubdivideTube(int tubeId, int count, const char *splitMode,
                               int seed, std::vector<int> *outChildren)
{
    return SubdivideTube(tubeId, count, splitMode, seed, 1.0f, 0.0f, 0.0f,
                         outChildren);
}

bool PomadeModel::SubdivideTube(int tubeId, int count, const char *splitMode,
                               int seed, float edgeA, float edgeB,
                               float edgeC, std::vector<int> *outChildren)
{
    if (!outChildren) {
        _diagnostic = "PomadeModel::SubdivideTube: null output";
        return false;
    }
    PomadeSubdivideDesc params;
    params.count = count;
    params.seed = seed;
    params.edgeA = edgeA;
    params.edgeB = edgeB;
    params.edgeC = edgeC;
    std::string mode = splitMode ? splitMode : "kmeans";
    params.splitMode =
        mode == "edge" ? PomadeSplit_Edge : PomadeSplit_KMeans;
    if (mode != "kmeans" && mode != "edge") {
        _diagnostic = "PomadeModel::SubdivideTube: splitMode kmeans | edge";
        return false;
    }
    std::string verr;
    if (!PomadeValidateSubdivide(params, &verr)) {
        _diagnostic = verr;
        return false;
    }
    PomadeTubeDesc desc;
    if (!_TubeDescLocked(tubeId, &desc)) {
        _diagnostic = "PomadeModel::SubdivideTube: unknown tube";
        return false;
    }
    for (auto const &kv : _tubes) {
        if (kv.second.actual.parentTubeId == tubeId) {
            _diagnostic =
                "PomadeModel::SubdivideTube: tube already has children "
                "(merge or re-subdivide instead)";
            return false;
        }
    }
    std::vector<PomadeFrame> frames;
    std::string derr;
    if (!PomadeTubeFramesCpu(desc, &frames, &derr)) {
        _diagnostic = derr;
        return false;
    }
    std::vector<PomadeTubeDesc> kids;
    if (!PomadeSubdivideTubeCpu(desc, frames, params, &kids, &derr)) {
        _diagnostic = derr;
        return false;
    }
    // Children inherit the parent's fill params (§2.3 step 4); the parent's
    // own fill is suspended from here until the children are merged back.
    FillParams inherited = _fill;
    {
        auto pit = _tubes.find(tubeId);
        if (pit != _tubes.end()) {
            inherited = pit->second.fill;
        }
    }
    // Reject (model untouched, nothing to roll back yet) a split whose
    // child rings K9 could not triangulate: accepting it would make every
    // later refill drop those children's guides.
    for (PomadeTubeDesc const &k : kids) {
        if (!_ValidateChildMaterialRings(k, inherited.cvCount,
                                         "PomadeSubdivideTubeCpu", &derr)) {
            _diagnostic = derr;
            return false;
        }
    }
    _PushUndoLocked();
    outChildren->clear();
    for (auto &k : kids) {
        HierarchyTube entry;
        entry.actual = k;
        entry.derived = k;
        PomadeClearDeltas(k, &entry.deltas);
        entry.fill = inherited;
        int const id = k.tubeId;
        _tubes[id] = entry;
        outChildren->push_back(id);
    }
    if (tubeId == 0) {
        _subdivide.count = count;
        _subdivide.seed = seed;
        _subdivide.splitMode = mode;
        _subdivide.edgeA = edgeA;
        _subdivide.edgeB = edgeB;
        _subdivide.edgeC = edgeC;
    } else {
        auto pit = _tubes.find(tubeId);
        if (pit != _tubes.end()) {
            pit->second.subdivide = params;
        }
    }
    // plan/18 §2.3: a subdivide of a selected parent moves the selection
    // to its children. The parent is still there (it is now a control
    // cage), but the artist's next edit belongs to what they just made.
    _selection.RemapTube(tubeId, *outChildren);
    ++_version;
    ++_mapVersion;
    _dirty |= PomadeDirty_Topology | PomadeDirty_Selection;
    return true;
}

bool PomadeModel::MergeChildren(int tubeId)
{
    std::vector<int> kids;
    for (auto const &kv : _tubes) {
        if (kv.second.actual.parentTubeId == tubeId) {
            kids.push_back(kv.first);
        }
    }
    if (kids.empty()) {
        return true;
    }
    std::vector<PomadeTubeDesc> actuals;
    for (int id : kids) {
        actuals.push_back(_tubes.find(id)->second.actual);
    }
    PomadeTubeDesc parentDesc;
    if (!_TubeDescLocked(tubeId, &parentDesc)) {
        _diagnostic = "PomadeModel::MergeChildren: unknown tube";
        return false;
    }
    PomadeSubdivideDesc params;
    if (tubeId == 0) {
        params.count = _subdivide.count;
        params.seed = _subdivide.seed;
        params.splitMode =
            _subdivide.splitMode == "edge" ? PomadeSplit_Edge : PomadeSplit_KMeans;
        params.edgeA = _subdivide.edgeA;
        params.edgeB = _subdivide.edgeB;
        params.edgeC = _subdivide.edgeC;
    } else {
        auto pit = _tubes.find(tubeId);
        if (pit == _tubes.end()) {
            return false;
        }
        params = pit->second.subdivide;
    }
    std::string derr;
    PomadeTubeDesc merged;
    if (!PomadeMergeTubesCpu(actuals, params, &parentDesc, &merged, &derr)) {
        _diagnostic = derr;
        return false;
    }
    // The merged parent is NOT gated like a split: Re-subdivide is a merge
    // followed by a split, and refusing an unfillable aggregate would block
    // the very re-split that repairs it. If the merged tube's rings do not
    // triangulate, the next refill skips it and RefillDrops() reports it
    // (the dock shows a warning row) instead of losing it silently.
    _PushUndoLocked();
    // Remove the whole subtree below tubeId.
    std::vector<int> doomed = kids;
    for (size_t i = 0; i < doomed.size(); ++i) {
        for (auto const &kv : _tubes) {
            if (kv.second.actual.parentTubeId == doomed[i]) {
                doomed.push_back(kv.first);
            }
        }
    }
    for (int id : doomed) {
        _tubes.erase(id);
    }
    _PruneActiveCutLocked();
    if (tubeId == 0) {
        PomadeTubeDesc fit;
        if (!_FitAggregateForWriteback(merged, parentDesc, &fit, &derr)) {
            _diagnostic = derr;
            return false;
        }
        if (!_WriteDescToTube0Locked(fit)) {
            return false;
        }
    } else {
        auto pit = _tubes.find(tubeId);
        if (pit == _tubes.end()) {
            return false;
        }
        PomadeTubeDesc refit;
        if (!_FitAggregateForWriteback(merged, pit->second.actual, &refit,
                                       &derr)) {
            _diagnostic = derr;
            return false;
        }
        refit.tubeId = pit->second.actual.tubeId;
        refit.regionId = pit->second.actual.regionId;
        refit.level = pit->second.actual.level;
        refit.parentTubeId = pit->second.actual.parentTubeId;
        refit.childIndex = pit->second.actual.childIndex;
        std::vector<PomadeFrame> dframes;
        if (!PomadeTubeFramesCpu(pit->second.derived, &dframes, &derr)) {
            _diagnostic = derr;
            return false;
        }
        PomadeShapeDeltas deltas;
        if (!PomadeComputeDeltasCpu(refit, pit->second.derived, dframes,
                                   &deltas, &derr)) {
            _diagnostic = derr;
            return false;
        }
        pit->second.actual = refit;
        pit->second.deltas = deltas;
    }
    ++_version;
    ++_mapVersion;
    _dirty |= PomadeDirty_Topology;
    return true;
}

bool PomadeModel::MergeSelected(std::vector<int> const &tubeIds, int *outKept)
{
    if (!outKept || tubeIds.empty()) {
        _diagnostic = "PomadeModel::MergeSelected: want tube ids + outKept";
        return false;
    }
    int const parentId = _tubes.find(tubeIds[0]) == _tubes.end()
                             ? -2
                             : _tubes.find(tubeIds[0])->second.actual.parentTubeId;
    for (int id : tubeIds) {
        auto it = _tubes.find(id);
        if (it == _tubes.end() ||
            it->second.actual.parentTubeId != parentId || parentId < 0) {
            _diagnostic =
                "PomadeModel::MergeSelected: want siblings sharing a parent";
            return false;
        }
    }
    std::vector<PomadeTubeDesc> actuals;
    for (int id : tubeIds) {
        actuals.push_back(_tubes.find(id)->second.actual);
    }
    std::string derr;
    PomadeTubeDesc merged;
    if (!_LaneParentAverage(HasCudaMirror(), actuals, &merged, &derr)) {
        _diagnostic = derr;
        return false;
    }
    int const kept = tubeIds[0];
    auto kit = _tubes.find(kept);
    PomadeTubeDesc refit;
    if (!_FitAggregateForWriteback(merged, kit->second.actual, &refit,
                                   &derr)) {
        _diagnostic = derr;
        return false;
    }
    refit.tubeId = kept;
    refit.regionId = kit->second.actual.regionId;
    refit.level = kit->second.actual.level;
    refit.parentTubeId = parentId;
    refit.childIndex = kit->second.actual.childIndex;
    std::vector<PomadeFrame> dframes;
    if (!PomadeTubeFramesCpu(kit->second.derived, &dframes, &derr)) {
        _diagnostic = derr;
        return false;
    }
    PomadeShapeDeltas deltas;
    if (!PomadeComputeDeltasCpu(refit, kit->second.derived, dframes, &deltas,
                               &derr)) {
        _diagnostic = derr;
        return false;
    }
    _PushUndoLocked();
    kit->second.actual = refit;
    kit->second.deltas = deltas;
    for (size_t i = 1; i < tubeIds.size(); ++i) {
        _tubes.erase(tubeIds[i]);
    }
    _PruneActiveCutLocked();
    *outKept = kept;
    ++_version;
    ++_mapVersion;
    _dirty |= PomadeDirty_Topology;
    return true;
}

bool PomadeModel::RemoveTubes(std::vector<int> const &tubeIds, int *outRemoved)
{
    if (outRemoved) {
        *outRemoved = 0;
    }
    if (tubeIds.empty()) {
        _diagnostic = "PomadeModel::RemoveTubes: no tube ids";
        return false;
    }
    // Validate everything before touching anything: a half-applied delete
    // would be an undo step the artist never asked for.
    std::vector<int> const roots = _L1TubeIdsLocked();
    for (int id : tubeIds) {
        if (id == 0 ||
            std::find(roots.begin(), roots.end(), id) != roots.end()) {
            _diagnostic = "PomadeModel::RemoveTubes: tube " +
                          std::to_string(id) +
                          " is an L1 root; its graph region owns it";
            return false;
        }
        if (_tubes.find(id) == _tubes.end()) {
            _diagnostic = "PomadeModel::RemoveTubes: unknown tube " +
                          std::to_string(id);
            return false;
        }
    }
    _PushUndoLocked();
    std::set<int> before;
    for (auto const &kv : _tubes) {
        before.insert(kv.first);
    }
    for (int id : tubeIds) {
        // An id already taken with an earlier id's subtree is simply gone.
        if (_tubes.find(id) != _tubes.end()) {
            _DropSubtreeLocked(id);
        }
    }
    std::set<int> gone;
    for (int id : before) {
        if (_tubes.find(id) == _tubes.end()) {
            gone.insert(id);
            _intersectFlags.erase(id);
        }
    }
    // A group parent must not keep naming a member that no longer exists:
    // the committer would serialise a dangling member path.
    for (auto &kv : _tubes) {
        std::vector<int> &members = kv.second.members;
        members.erase(std::remove_if(members.begin(), members.end(),
                                     [&gone](int member) {
                                         return gone.count(member) > 0;
                                     }),
                      members.end());
    }
    _selection.PruneTubes(_LiveTubeIdsLocked());
    if (outRemoved) {
        *outRemoved = int(gone.size());
    }
    ++_version;
    ++_mapVersion;
    _dirty |= PomadeDirty_Topology | PomadeDirty_Selection;
    return true;
}

int PomadeModel::GetTubeCount() const
{
    int n = int(_tubes.size());
    if (_host.centerX.size() >= 2) {
        ++n;
    }
    return n;
}

int PomadeModel::GetTubeLevel(int tubeId) const
{
    if (tubeId == 0) {
        return _host.centerX.size() >= 2 ? 1 : -1;
    }
    auto it = _tubes.find(tubeId);
    return it == _tubes.end() ? -1 : it->second.actual.level;
}

std::vector<int> PomadeModel::GetTubeChildren(int tubeId) const
{
    std::vector<int> kids;
    for (auto const &kv : _tubes) {
        if (kv.second.actual.parentTubeId == tubeId) {
            kids.push_back(kv.first);
        }
    }
    return kids;
}

int PomadeModel::GetTubeCenterCount(int tubeId) const
{
    if (tubeId == 0) {
        return _host.centerX.size() >= 2 ? int(_host.centerX.size()) : -1;
    }
    auto it = _tubes.find(tubeId);
    return it == _tubes.end() ? -1 : int(it->second.actual.centerX.size());
}

bool PomadeModel::GetTubeCenterCV(int tubeId, int cv, float *x, float *y,
                                 float *z) const
{
    if (!x || !y || !z) {
        return false;
    }
    if (tubeId == 0) {
        int const n = int(_host.centerX.size());
        if (cv < 0 || cv >= n || n < 2) {
            return false;
        }
        *x = _host.centerX[size_t(cv)];
        *y = _host.centerY[size_t(cv)];
        *z = _host.centerZ[size_t(cv)];
        return true;
    }
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end()) {
        return false;
    }
    int const n = int(it->second.actual.centerX.size());
    if (cv < 0 || cv >= n) {
        return false;
    }
    *x = it->second.actual.centerX[size_t(cv)];
    *y = it->second.actual.centerY[size_t(cv)];
    *z = it->second.actual.centerZ[size_t(cv)];
    return true;
}

bool
PomadeModel::GetTubeCenterHandle(int tubeId, int cv, float *x, float *y,
                                float *z) const
{
    if (!x || !y || !z) {
        return false;
    }
    PomadeTubeDesc desc;
    if (!_TubeDescLocked(tubeId, &desc)) {
        return false;
    }
    std::string err;
    return PomadeCenterHandlePointCpu(desc, cv, x, y, z, &err);
}

std::shared_ptr<PomadeModel::GraphUndoState const>
PomadeModel::_GraphUndoStateLocked() const
{
    if (!_scalp || !_scalp->finalized) {
        return nullptr;
    }
    if (_graphUndoCache && _graphUndoCacheMap == _mapVersion) {
        return _graphUndoCache;  // the graph cannot have changed
    }
    auto state = std::make_shared<GraphUndoState>();
    for (auto const &nd : _graph.Nodes()) {
        if (nd.alive) {
            state->nodes.push_back(nd);
        }
    }
    for (auto const &e : _graph.Edges()) {
        if (e.alive) {
            state->edges.emplace_back(e.a, e.b);
        }
    }
    state->linked = _graph.LinkedPairs();
    state->snapRadius = _snapRadius;
    _graphUndoCache = state;
    _graphUndoCacheMap = _mapVersion;
    return _graphUndoCache;
}

void PomadeModel::_RestoreGraphUndoLocked(
    std::shared_ptr<GraphUndoState const> const &state)
{
    if (!state || !_scalp || !_scalp->finalized) {
        return;
    }
    std::string err;
    if (!_graph.Restore(*_scalp, state->nodes, state->edges, state->linked,
                        &err)) {
        return;  // the diagnostic the caller carries is the one that matters
    }
    _snapRadius = state->snapRadius;
    // The region maps are derived, exactly as RestoreGraph derives them:
    // an undone graph edit has to leave the tint and the bake key right.
    PomadeRasteriseRegionsCpu(*_scalp, _graph, &_maps, &err);
    PomadeFlattenLoops(_graph, &_loops, &err);
    // The live graph IS this state now, so the next snapshot shares it
    // instead of rebuilding an identical copy.
    _graphUndoCache = state;
    _graphUndoCacheMap = _mapVersion;
    _dirty |= PomadeDirty_Graph | PomadeDirty_Regions;
}

PomadeModel::HierarchyRollback
PomadeModel::_SnapshotHierarchyLocked() const
{
    HierarchyRollback snap;
    snap.tubes = _tubes;
    snap.cx = _host.centerX;
    snap.cy = _host.centerY;
    snap.cz = _host.centerZ;
    snap.sections = _sections;
    snap.shape = _shape;
    snap.useSections = _useSections;
    snap.roots = _roots;
    snap.tubeRoots = _tubeRoots;
    snap.guides = _guides;
    snap.generatedCurvesSuppressed = _generatedCurvesSuppressed;
    snap.output = _output;
    snap.fill = _fill;
    snap.graph = _GraphUndoStateLocked();
    snap.regionTube = _regionTube;
    snap.tubeRegionKey = _tubeRegionKey;
    snap.tube0Region = _tubeRegionId;
    snap.tube0RootFramePinned = _rootFramePinned;
    snap.tube0RootFrame = _rootFrame;
    snap.tube0FrameReference = _frameReference;
    return snap;
}

void PomadeModel::_RestoreHierarchyLocked(HierarchyRollback const &snap)
{
    std::string keep = _diagnostic;
    _tubes = snap.tubes;
    _host.centerX = snap.cx;
    _host.centerY = snap.cy;
    _host.centerZ = snap.cz;
    _sections = snap.sections;
    _shape = snap.shape;
    _useSections = snap.useSections;
    _roots = snap.roots;
    _tubeRoots = snap.tubeRoots;
    _guides = snap.guides;
    _generatedCurvesSuppressed = snap.generatedCurvesSuppressed;
    _output = snap.output;
    _fill = snap.fill;
    ++_guideVersion;
    _regionTube = snap.regionTube;
    _tubeRegionKey = snap.tubeRegionKey;
    _tubeRegionId = snap.tube0Region;
    _rootFramePinned = snap.tube0RootFramePinned;
    _rootFrame = snap.tube0RootFrame;
    _frameReference = snap.tube0FrameReference;
    _RestoreGraphUndoLocked(snap.graph);
    // Active-cut expansion is viewport state, not undo history. A restored
    // hierarchy can remove an expanded parent or turn it back into a leaf;
    // discard only those stale ids and preserve expansions in other branches.
    _PruneActiveCutLocked();
    // Only re-tessellate when there IS a tube. A graph-only model reaches
    // this path from V1 on (graph edits are undoable now), and the legacy
    // tessellation reads _shape.rings CVs out of the center columns, which
    // a tube-less model does not have.
    if (!_host.centerX.empty()) {
        Sync();  // best effort; the original error below is what matters
    }
    _diagnostic = keep;
    _dirty |= PomadeDirty_Topology;
}

bool
PomadeModel::_TransportRegionAttachmentsLocked(
    HierarchyRollback const &reference,
    std::vector<int> const &movedNodeIds)
{
    if (!_scalp || !reference.graph) {
        _diagnostic = "PomadeModel::GraphMoveNodes: missing graph support";
        return false;
    }

    // A gesture's node positions are absolute from its press-time baseline.
    // Rebuild that graph locally so each accepted sample transports the
    // original authored descriptors directly, never the last sampled pose.
    PomadeScalpGraph oldGraph;
    std::string err;
    if (!oldGraph.Restore(*_scalp, reference.graph->nodes,
                          reference.graph->edges, reference.graph->linked,
                          &err)) {
        _diagnostic = err.empty()
                          ? "PomadeModel::GraphMoveNodes: restore baseline "
                            "graph failed"
                          : err;
        return false;
    }

    auto sourceDesc = [&](int tubeId, PomadeTubeDesc *out) {
        if (!out) {
            return false;
        }
        if (tubeId == 0) {
            out->centerX = reference.cx;
            out->centerY = reference.cy;
            out->centerZ = reference.cz;
            out->sections = reference.sections;
            out->ringVerts = reference.shape.ringVerts;
            out->tubeId = 0;
            out->regionId = reference.tube0Region;
            out->level = 1;
            out->parentTubeId = -1;
            out->childIndex = -1;
            out->rootFramePinned = reference.tube0RootFramePinned;
            out->rootFrame = reference.tube0RootFrame;
            out->frameReference = reference.tube0FrameReference;
            return out->centerX.size() >= 2;
        }
        auto const it = reference.tubes.find(tubeId);
        if (it == reference.tubes.end()) {
            return false;
        }
        *out = it->second.actual;
        return true;
    };

    auto subtree = [&](int root) {
        std::vector<int> ids;
        std::vector<int> pending(1, root);
        while (!pending.empty()) {
            int const id = pending.back();
            pending.pop_back();
            ids.push_back(id);
            for (auto const &kv : reference.tubes) {
                if (kv.second.actual.parentTubeId == id) {
                    pending.push_back(kv.first);
                }
            }
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    };

    // Existing guide roots are world-space caches. Start each live sample
    // from the same cache as the descriptor baseline so a frozen-root drag
    // cannot retain roots from a previous sample's region support.
    _roots = reference.roots;
    _tubeRoots = reference.tubeRoots;
    _guides = reference.guides;

    std::set<std::vector<int>> affectedKeys;
    for (int nodeId : movedNodeIds) {
        for (int regionId : oldGraph.NodeRegions(nodeId)) {
            if (regionId >= 0 && regionId < oldGraph.RegionCount()) {
                affectedKeys.insert(_StableRegionLoopKey(
                    oldGraph.Regions()[size_t(regionId)]));
            }
        }
        for (int regionId : _graph.NodeRegions(nodeId)) {
            if (regionId >= 0 && regionId < _graph.RegionCount()) {
                affectedKeys.insert(_StableRegionLoopKey(
                    _graph.Regions()[size_t(regionId)]));
            }
        }
    }

    auto transportCachedRoots = [&](std::vector<int> const &ids,
                                    PomadeRigidTransform const &rigid) {
        for (int id : ids) {
            auto rootsIt = _tubeRoots.find(id);
            if (rootsIt == _tubeRoots.end()) {
                continue;
            }
            PomadeTubeDesc desc;
            std::vector<PomadeFrame> frames;
            if (!_TubeDescLocked(id, &desc) ||
                !PomadeTubeFramesCpu(desc, &frames, &err) || frames.empty()) {
                _diagnostic = err.empty()
                                  ? "PomadeModel::GraphMoveNodes: invalid "
                                    "transported guide frame"
                                  : err;
                return false;
            }
            float const radius = PomadeSectionMeanRadius(desc.sections.front());
            if (!(radius > 1e-12f)) {
                _diagnostic = "PomadeModel::GraphMoveNodes: invalid "
                              "transported guide radius";
                return false;
            }
            for (PomadeGuideRoot &root : rootsIt->second) {
                float const p[3] = {root.px, root.py, root.pz};
                float moved[3] = {
                    rigid.rotation[0] * p[0] + rigid.rotation[1] * p[1] +
                        rigid.rotation[2] * p[2] + rigid.translation[0],
                    rigid.rotation[3] * p[0] + rigid.rotation[4] * p[1] +
                        rigid.rotation[5] * p[2] + rigid.translation[1],
                    rigid.rotation[6] * p[0] + rigid.rotation[7] * p[1] +
                        rigid.rotation[8] * p[2] + rigid.translation[2]};
                PomadeHit const hit = PomadeClosestPointCpu(*_scalp, moved);
                if (!hit.hit) {
                    _diagnostic = "PomadeModel::GraphMoveNodes: guide root "
                                  "left scalp support";
                    return false;
                }
                root.faceId = hit.faceId;
                root.u = hit.u;
                root.v = hit.v;
                root.px = hit.px;
                root.py = hit.py;
                root.pz = hit.pz;
                float const d[3] = {hit.px - desc.centerX[0],
                                    hit.py - desc.centerY[0],
                                    hit.pz - desc.centerZ[0]};
                float const n[3] = {frames[0].nx, frames[0].ny,
                                    frames[0].nz};
                float const b[3] = {frames[0].bx, frames[0].by,
                                    frames[0].bz};
                root.ru = PomadeDot3(d, n) / radius;
                root.rv = PomadeDot3(d, b) / radius;
            }
        }
        return true;
    };

    std::map<int, int> movedOwners;
    std::set<int> transformed;
    bool attached = false;
    for (int root : _L1TubeIdsLocked()) {
        PomadeTubeDesc rootSource;
        if (!sourceDesc(root, &rootSource) || rootSource.regionId < 0) {
            continue;
        }
        std::vector<int> key;
        auto const keyIt = reference.tubeRegionKey.find(root);
        if (keyIt != reference.tubeRegionKey.end()) {
            key = keyIt->second;
        } else if (rootSource.regionId < oldGraph.RegionCount()) {
            key = _StableRegionLoopKey(
                oldGraph.Regions()[size_t(rootSource.regionId)]);
        }
        if (key.empty()) {
            continue;  // an unrooted legacy tube is independent of graph edits
        }
        int const oldRegion = _RegionForStableLoop(oldGraph, key);
        int const newRegion = _RegionForStableLoop(_graph, key);
        if (oldRegion < 0 || newRegion < 0) {
            _diagnostic = "PomadeModel::GraphMoveNodes: region loop lost "
                          "during geometry-only move";
            return false;
        }
        if (affectedKeys.count(key) == 0) {
            // Extraction may densely renumber an unchanged loop. Its world
            // shape is independent of this bookkeeping id, but commit and
            // subsequent exact-region fills must name the new slot.
            for (int id : subtree(root)) {
                if (id == 0) {
                    _tubeRegionId = newRegion;
                } else {
                    auto current = _tubes.find(id);
                    if (current != _tubes.end()) {
                        current->second.actual.regionId = newRegion;
                        current->second.derived.regionId = newRegion;
                    }
                }
            }
            movedOwners[newRegion] = root;
            continue;
        }
        bool const importedRoot =
            root != 0 && reference.tubes.count(root) != 0 &&
            reference.tubes.at(root).imported;
        _RegionSupport from, to;
        if (!_RegionSupportFromGraph(*_scalp,
                                     oldGraph.Regions()[size_t(oldRegion)],
                                     &from) ||
            !_RegionSupportFromGraph(*_scalp,
                                     _graph.Regions()[size_t(newRegion)],
                                     &to)) {
            _diagnostic = "PomadeModel::GraphMoveNodes: invalid region "
                          "support frame";
            return false;
        }
        PomadeRigidTransform rigid;
        if (!_RigidBetweenRegionSupports(from, to, &rigid)) {
            _diagnostic = "PomadeModel::GraphMoveNodes: invalid region "
                          "transport";
            return false;
        }
        // A whole-tube artist move can leave an existing region root above or
        // below its support chart.  When this graph edit really changes the
        // footprint, bring the complete subtree back by that normal component
        // before fitting the base.  Never move CV0 alone: it changes K4 and
        // shears upper sculpted sections.  The composed rigid is also used
        // for cached guide roots below.
        _RegionFootprint oldFootprint, newFootprint;
        if (!_RegionCanonicalFootprint(*_scalp, oldGraph, oldRegion,
                                       &oldFootprint, &err) ||
            !_RegionCanonicalFootprint(*_scalp, _graph, newRegion,
                                       &newFootprint, &err)) {
            _diagnostic = err.empty()
                              ? "PomadeModel::GraphMoveNodes: invalid region "
                                "attachment footprint"
                              : err;
            return false;
        }
        if (!importedRoot &&
            !_SameRegionFootprint(oldFootprint, newFootprint)) {
            PomadeTubeDesc transformedRoot;
            if (!PomadeRigidTransformTubeCpu(rootSource, rigid,
                                             &transformedRoot, &err)) {
                _diagnostic = err;
                return false;
            }
            float const dx = to.origin[0] - transformedRoot.centerX[0];
            float const dy = to.origin[1] - transformedRoot.centerY[0];
            float const dz = to.origin[2] - transformedRoot.centerZ[0];
            float const normalOffset = dx * to.normal[0] +
                                       dy * to.normal[1] +
                                       dz * to.normal[2];
            if (std::fabs(normalOffset) > 1e-6f) {
                rigid.translation[0] += to.normal[0] * normalOffset;
                rigid.translation[1] += to.normal[1] * normalOffset;
                rigid.translation[2] += to.normal[2] * normalOffset;
            }
        }
        std::vector<int> const ids = subtree(root);
        for (int id : ids) {
            if (!transformed.insert(id).second) {
                continue;
            }
            if (id == 0) {
                PomadeTubeDesc before;
                if (!sourceDesc(0, &before)) {
                    _diagnostic = "PomadeModel::GraphMoveNodes: invalid "
                                  "tube-0 baseline";
                    return false;
                }
                PomadeTubeDesc after;
                if (!PomadeRigidTransformTubeCpu(before, rigid, &after,
                                                 &err)) {
                    _diagnostic = err;
                    return false;
                }
                after.regionId = newRegion;
                _shape.rings = int(after.centerX.size());
                _shape.ringVerts = after.ringVerts;
                _host.centerX = std::move(after.centerX);
                _host.centerY = std::move(after.centerY);
                _host.centerZ = std::move(after.centerZ);
                _sections = std::move(after.sections);
                _useSections = true;
                _tubeRegionId = newRegion;
                _rootFramePinned = after.rootFramePinned;
                _rootFrame = after.rootFrame;
                _frameReference = after.frameReference;
                continue;
            }
            auto sourceIt = reference.tubes.find(id);
            if (sourceIt == reference.tubes.end()) {
                _diagnostic = "PomadeModel::GraphMoveNodes: subtree baseline "
                              "is incomplete";
                return false;
            }
            HierarchyTube entry = sourceIt->second;
            if (!PomadeRigidTransformTubeCpu(sourceIt->second.actual, rigid,
                                             &entry.actual, &err) ||
                !PomadeRigidTransformTubeCpu(sourceIt->second.derived, rigid,
                                             &entry.derived, &err)) {
                _diagnostic = err;
                return false;
            }
            entry.actual.regionId = newRegion;
            entry.derived.regionId = newRegion;
            // The same proper transform and Q'=R*Q were applied to actual
            // and derived. Their stored coefficients remain in that shared
            // material frame exactly; re-measuring only injects float noise
            // (and can turn a deliberate zero marker into a sculpt delta).
            // A later ordinary K6 parent edit re-computes them in its new
            // derived frame, as it has always done.
            _tubes[id] = std::move(entry);
        }
        // Rigidly transported roots keep their exact residuals for pure
        // support motion.  If the graph boundary itself changed, constrain
        // just the attachment section to its canonical new footprint, then
        // rebase descendants downward without ever fitting the root upward.
        bool attachmentChanged = false;
        if (!importedRoot &&
            !_ConformRegionRootSectionLocked(root, oldGraph, oldRegion,
                                              newRegion, rootSource,
                                              &attachmentChanged)) {
            return false;
        }
        if (attachmentChanged && !_PropagateAttachmentDownLocked(root)) {
            return false;
        }
        if (!transportCachedRoots(ids, rigid)) {
            return false;
        }
        movedOwners[newRegion] = root;
        attached = true;
    }

    // The stable loop identity remains the source of ownership; only dense
    // region ids may change after extraction. Rebuild this cheap index from
    // the reference rather than invoking _SyncRegionTubesLocked, which is
    // intentionally allowed to re-root on topological graph edits.
    _regionTube.clear();
    _tubeRegionKey = reference.tubeRegionKey;
    for (auto const &kv : movedOwners) {
        _regionTube[kv.first] = kv.second;
    }
    if (!_host.centerX.empty() && !Sync()) {
        return false;
    }
    if (!PomadeRasteriseRegionsCpu(*_scalp, _graph, &_maps, &err) ||
        !PomadeFlattenLoops(_graph, &_loops, &err)) {
        _diagnostic = err;
        return false;
    }
    if (attached && !_generatedCurvesSuppressed &&
        !_RefillGuidesLocked(_previewFraction)) {
        return false;
    }
    if (attached && _generatedCurvesSuppressed) {
        ++_guideVersion;
    }
    if (attached) {
        _dirty |= PomadeDirty_Points | PomadeDirty_Guides;
    }
    return true;
}

// Accounted bytes of one undo snapshot (payload only: vector contents
// plus a per-node allowance for the map and section structs; vector
// capacities and allocator overhead are deliberately excluded — the
// budget is approximate by contract, and this side under-reports).
uint64_t
PomadeModel::_UndoSnapshotBytes(HierarchyRollback const &snap)
{
    uint64_t bytes = 0;
    auto sectionBytes = [](std::vector<PomadeTubeSection> const &sections) {
        uint64_t sub = 0;
        for (auto const &s : sections) {
            sub += 64;  // struct + vector shells
            sub += uint64_t(s.u.size() + s.v.size()) * sizeof(float);
        }
        return sub;
    };
    auto descBytes = [&](PomadeTubeDesc const &d) {
        uint64_t sub = 64;  // struct shell + scalars
        sub += uint64_t(d.centerX.size() + d.centerY.size() +
                        d.centerZ.size()) *
               sizeof(float);
        sub += sectionBytes(d.sections);
        return sub;
    };
    for (auto const &kv : snap.tubes) {
        bytes += 64;  // map node + key
        bytes += descBytes(kv.second.actual);
        bytes += descBytes(kv.second.derived);
        bytes += 32;  // deltas shell
        bytes += uint64_t(kv.second.deltas.centerDu.size() +
                          kv.second.deltas.centerDv.size() +
                          kv.second.deltas.centerDw.size()) *
                 sizeof(float);
        bytes += sectionBytes(kv.second.deltas.sections);
        bytes += 32;  // subdivide params + flags + members shell
        bytes += uint64_t(kv.second.members.size()) * sizeof(int);
    }
    bytes += uint64_t(snap.cx.size() + snap.cy.size() + snap.cz.size()) *
             sizeof(float);
    bytes += sectionBytes(snap.sections);
    bytes += sizeof(snap.shape) + sizeof(snap.useSections);
    bytes += uint64_t(snap.roots.size()) * sizeof(PomadeGuideRoot);
    bytes += 32;  // guide set shell
    bytes += uint64_t(snap.guides.points.size()) * sizeof(float);
    bytes += uint64_t(snap.guides.counts.size()) * sizeof(int);
    if (snap.graph) {
        // Counted per step even though consecutive steps share the state:
        // over-reporting evicts a little early, which is the safe side of
        // an approximate budget (and a graph is kilobytes, not megabytes).
        bytes += uint64_t(snap.graph->nodes.size()) * sizeof(PomadeGraphNode);
        bytes += uint64_t(snap.graph->edges.size() +
                          snap.graph->linked.size()) *
                 (2 * sizeof(int));
    }
    bytes += uint64_t(snap.label.size());
    return bytes;
}

void PomadeModel::_EnforceUndoBudgetLocked()
{
    while (!_undoStack.empty() &&
           (int(_undoStack.size()) > _undoMaxDepth ||
            _undoBytes > _undoMaxBytes)) {
        _undoBytes -= _UndoSnapshotBytes(_undoStack.front());
        _undoStack.pop_front();
    }
}

void PomadeModel::_ClearRedoLocked()
{
    _redoStack.clear();
    _redoBytes = 0;
}

void PomadeModel::_PushUndoSnapshotLocked(HierarchyRollback const &snap)
{
    // A new mutation branches the history: whatever redo held is
    // unreachable from here (plan/18 §2.5). This runs before the
    // gesture and budget early-outs, so the rule holds even when undo
    // is switched off.
    _ClearRedoLocked();
    if (_gestureDepth > 0) {
        // Inside a bracket the press-time snapshot IS the undo step
        // (plan/18 §3.2): a 200-sample drag pushes one entry, not 200.
        return;
    }
    if (_undoMaxDepth <= 0) {
        return;  // disabled: no stack, no accounting
    }
    HierarchyRollback labelled = snap;
    labelled.label = _nextUndoLabel.empty() ? std::string("edit")
                                            : _nextUndoLabel;
    _nextUndoLabel.clear();
    _undoBytes += _UndoSnapshotBytes(labelled);
    _undoStack.push_back(std::move(labelled));
    _EnforceUndoBudgetLocked();
}

void PomadeModel::_PushUndoLocked()
{
    _PushUndoSnapshotLocked(_SnapshotHierarchyLocked());
}

void PomadeModel::_PushGraphUndoLocked(
    std::shared_ptr<GraphUndoState const> const &before, char const *label)
{
    HierarchyRollback snap = _SnapshotHierarchyLocked();
    snap.graph = before;
    if (_nextUndoLabel.empty() && label) {
        _nextUndoLabel = label;
    }
    _PushUndoSnapshotLocked(snap);
}

bool PomadeModel::Undo(uint32_t *outDirty)
{
    if (outDirty) {
        *outDirty = PomadeDirty_Clean;
    }
    if (_gestureDepth > 0) {
        // Redo's rule, for the same reason: inside a bracket the top of
        // the stack is the press-time step Begin pushed. Popping it here
        // (Ctrl+Z mid-drag on a dock slider) restored the base under a
        // still-open bracket, and a later CancelGesture then popped an
        // unrelated older step.
        _diagnostic = "PomadeModel::Undo: a gesture is open";
        return false;
    }
    if (_undoStack.empty()) {
        return true;  // nothing to undo: no-op success
    }
    HierarchyRollback snap = std::move(_undoStack.back());
    _undoStack.pop_back();
    _undoBytes -= _UndoSnapshotBytes(snap);
    // What we are about to leave becomes the redo step, under the same
    // label: undoing "sculpt" makes "sculpt" the thing Ctrl+Y redoes.
    HierarchyRollback forward = _SnapshotHierarchyLocked();
    forward.label = snap.label;
    _redoBytes += _UndoSnapshotBytes(forward);
    _redoStack.push_back(std::move(forward));
    while (!_redoStack.empty() &&
           (int(_redoStack.size()) > _undoMaxDepth ||
            _redoBytes > _undoMaxBytes)) {
        _redoBytes -= _UndoSnapshotBytes(_redoStack.front());
        _redoStack.pop_front();
    }
    _RestoreHierarchyLocked(snap);
    ++_version;
    ++_mapVersion;
    uint32_t const bits =
        PomadeDirty_Points | PomadeDirty_Topology | PomadeDirty_Guides;
    _dirty |= bits;
    if (outDirty) {
        *outDirty = _dirty;
    }
    return true;
}

bool PomadeModel::Redo(uint32_t *outDirty)
{
    if (outDirty) {
        *outDirty = PomadeDirty_Clean;
    }
    if (_gestureDepth > 0) {
        _diagnostic = "PomadeModel::Redo: a gesture is open";
        return false;
    }
    if (_redoStack.empty()) {
        return true;  // nothing to redo: no-op success
    }
    HierarchyRollback snap = std::move(_redoStack.back());
    _redoStack.pop_back();
    _redoBytes -= _UndoSnapshotBytes(snap);
    // Push the state we leave straight onto the undo stack: going through
    // _PushUndoSnapshotLocked would clear the redo stack we are walking.
    HierarchyRollback back = _SnapshotHierarchyLocked();
    back.label = snap.label;
    if (_undoMaxDepth > 0) {
        _undoBytes += _UndoSnapshotBytes(back);
        _undoStack.push_back(std::move(back));
        _EnforceUndoBudgetLocked();
    }
    _RestoreHierarchyLocked(snap);
    ++_version;
    ++_mapVersion;
    uint32_t const bits =
        PomadeDirty_Points | PomadeDirty_Topology | PomadeDirty_Guides;
    _dirty |= bits;
    if (outDirty) {
        *outDirty = _dirty;
    }
    return true;
}

int PomadeModel::GetRedoDepth() const
{
    return int(_redoStack.size());
}

bool PomadeModel::GetUndoLabel(int depth, std::string *out) const
{
    if (!out) {
        return false;
    }
    if (depth >= 0) {
        if (depth >= int(_undoStack.size())) {
            return false;
        }
        *out = _undoStack[_undoStack.size() - 1 - size_t(depth)].label;
        return true;
    }
    size_t const back = size_t(-depth) - 1;  // -1 is the next redo
    if (back >= _redoStack.size()) {
        return false;
    }
    *out = _redoStack[_redoStack.size() - 1 - back].label;
    return true;
}

// -- V1: the gesture bracket (plan/18 §3.2) ----------------------------------

bool PomadeModel::BeginGesture(char const *label)
{
    if (_gestureDepth > 0) {
        _diagnostic = "PomadeModel::BeginGesture: a gesture is already open";
        return false;
    }
    _gestureBase = _SnapshotHierarchyLocked();
    _nextUndoLabel = label && *label ? label : "gesture";
    _PushUndoSnapshotLocked(_gestureBase);  // the one step of the drag
    _gestureDepth = 1;
    return true;
}

bool PomadeModel::EndGesture()
{
    if (_gestureDepth <= 0) {
        _diagnostic = "PomadeModel::EndGesture: no gesture is open";
        return false;
    }
    _gestureDepth = 0;
    _gestureBase = HierarchyRollback();
    // A label set by a push that was suppressed while the gesture was
    // open must not leak onto the next step pushed outside any gesture
    // (a stub build after a Connect used to be labelled "connect").
    _nextUndoLabel.clear();
    ++_version;  // one coalescing key for the whole drag
    return true;
}

bool PomadeModel::CancelGesture(uint32_t *outDirty)
{
    if (outDirty) {
        *outDirty = PomadeDirty_Clean;
    }
    if (_gestureDepth <= 0) {
        _diagnostic = "PomadeModel::CancelGesture: no gesture is open";
        return false;
    }
    HierarchyRollback const base = _gestureBase;
    _gestureDepth = 0;
    _gestureBase = HierarchyRollback();
    _nextUndoLabel.clear();  // same leak as EndGesture
    // Drop the step Begin pushed: the drag never happened as far as the
    // history is concerned. Pushes were suppressed while it was open, so
    // the top of the stack is that step.
    if (!_undoStack.empty()) {
        _undoBytes -= _UndoSnapshotBytes(_undoStack.back());
        _undoStack.pop_back();
    }
    _RestoreHierarchyLocked(base);
    // No version bump: the stage never saw the drag (plan/18 §2.5). The
    // VIEWPORT did, so the caller still has to publish what changed back.
    _dirty |= PomadeDirty_Points | PomadeDirty_Topology | PomadeDirty_Guides;
    if (outDirty) {
        *outDirty = _dirty;
    }
    return true;
}

int PomadeModel::GetGestureDepth() const
{
    return _gestureDepth;
}

int PomadeModel::GetUndoDepth() const
{
    return int(_undoStack.size());
}

uint64_t PomadeModel::GetUndoBytes() const
{
    return _undoBytes;
}

bool PomadeModel::SetUndoBudget(int maxDepth, uint64_t maxBytes)
{
    if (maxDepth < 0) {
        _diagnostic = "PomadeModel::SetUndoBudget: depth must be >= 0";
        return false;
    }
    _undoMaxDepth = maxDepth;
    _undoMaxBytes = maxBytes;
    _EnforceUndoBudgetLocked();
    return true;
}

void PomadeModel::_ClearUndoLocked()
{
    _undoStack.clear();
    _undoBytes = 0;
    _ClearRedoLocked();  // a cleared history has nothing to redo either
}

void PomadeModel::ClearUndo()
{
    _ClearUndoLocked();
}

bool PomadeModel::MoveTubeCenterCV(int tubeId, int cv, float dx, float dy,
                                  float dz)
{
    HierarchyRollback const snap = _SnapshotHierarchyLocked();
    auto const rollback = [&](bool ok) {
        if (!ok) {
            _RestoreHierarchyLocked(snap);
        }
        return ok;
    };
    if (tubeId == 0) {
        int const n = int(_host.centerX.size());
        if (cv < 0 || cv >= n || n < 2) {
            _diagnostic = "PomadeModel::MoveTubeCenterCV: bad tube-0 CV";
            return false;
        }
        // Same soft-selection application as MoveCenterCV, minus its
        // version bump (one gesture, one version; propagation below).
        float const centerT = float(cv) / float(n - 1);
        for (int i = 0; i < n; ++i) {
            float const t = float(i) / float(n - 1);
            float const w = PomadeSoftWeight(t, centerT, _softRadius);
            if (w <= 0.0f && i != cv) {
                continue;
            }
            float const k =
                (_softRadius > 0.0f) ? w : (i == cv ? 1.0f : 0.0f);
            _host.centerX[size_t(i)] += dx * k;
            _host.centerY[size_t(i)] += dy * k;
            _host.centerZ[size_t(i)] += dz * k;
        }
        if (!Sync()) {
            return rollback(false);
        }
        _dirty |= PomadeDirty_Points;
    } else {
        auto it = _tubes.find(tubeId);
        if (it == _tubes.end()) {
            _diagnostic = "PomadeModel::MoveTubeCenterCV: unknown tube";
            return false;
        }
        HierarchyTube &entry = it->second;
        int const n = int(entry.actual.centerX.size());
        if (cv < 0 || cv >= n) {
            _diagnostic = "PomadeModel::MoveTubeCenterCV: bad CV index";
            return false;
        }
        // Soft selection along the center is a property of the MOVE, not
        // of tube 0 (plan/17 §5.2, and the §5.2 V0b note that says the
        // per-tube spellings keep tube 0's behaviour "soft selection
        // included"). Before V6 only the tube-0 branch above applied it,
        // so a drag on a child snapped one CV while the same drag on the
        // root feathered. Same PomadeSoftWeight, same radius, same
        // single-CV behaviour at radius 0.
        float const centerT = n > 1 ? float(cv) / float(n - 1) : 0.0f;
        for (int i = 0; i < n; ++i) {
            float const t = n > 1 ? float(i) / float(n - 1) : 0.0f;
            float const w = PomadeSoftWeight(t, centerT, _softRadius);
            if (w <= 0.0f && i != cv) {
                continue;
            }
            float const k =
                (_softRadius > 0.0f) ? w : (i == cv ? 1.0f : 0.0f);
            entry.actual.centerX[size_t(i)] += dx * k;
            entry.actual.centerY[size_t(i)] += dy * k;
            entry.actual.centerZ[size_t(i)] += dz * k;
        }
        std::vector<PomadeFrame> dframes;
        std::string derr;
        if (!PomadeTubeFramesCpu(entry.derived, &dframes, &derr)) {
            _diagnostic = derr;
            return rollback(false);
        }
        if (!PomadeComputeDeltasCpu(entry.actual, entry.derived, dframes,
                                   &entry.deltas, &derr)) {
            _diagnostic = derr;
            return rollback(false);
        }
        _dirty |= PomadeDirty_Points;
    }
    if (!_PropagateDownLocked(tubeId)) {
        return rollback(false);
    }
    if (!_PropagateUpLocked(tubeId, &snap)) {
        return rollback(false);
    }
    _PushUndoSnapshotLocked(snap);
    ++_version;
    return true;
}

bool
PomadeModel::TranslateTube(int tubeId, float dx, float dy, float dz)
{
    HierarchyRollback const snap = _SnapshotHierarchyLocked();
    auto const rollback = [&](bool ok) {
        if (!ok) {
            _RestoreHierarchyLocked(snap);
        }
        return ok;
    };

    if (tubeId == 0) {
        int const n = int(_host.centerX.size());
        if (n < 2) {
            _diagnostic = "PomadeModel::TranslateTube: tube 0 has no center";
            return false;
        }
        for (int i = 0; i < n; ++i) {
            _host.centerX[size_t(i)] += dx;
            _host.centerY[size_t(i)] += dy;
            _host.centerZ[size_t(i)] += dz;
        }
        if (!Sync()) {
            return rollback(false);
        }
        _dirty |= PomadeDirty_Points;
    } else {
        auto it = _tubes.find(tubeId);
        if (it == _tubes.end() || it->second.actual.centerX.size() < 2) {
            _diagnostic = "PomadeModel::TranslateTube: unknown or empty tube";
            return false;
        }
        HierarchyTube &entry = it->second;
        for (size_t i = 0; i < entry.actual.centerX.size(); ++i) {
            entry.actual.centerX[i] += dx;
            entry.actual.centerY[i] += dy;
            entry.actual.centerZ[i] += dz;
        }
        std::vector<PomadeFrame> dframes;
        std::string derr;
        if (!PomadeTubeFramesCpu(entry.derived, &dframes, &derr) ||
            !PomadeComputeDeltasCpu(entry.actual, entry.derived, dframes,
                                   &entry.deltas, &derr)) {
            _diagnostic = derr.empty()
                              ? "PomadeModel::TranslateTube: invalid child frame"
                              : derr;
            return rollback(false);
        }
        _dirty |= PomadeDirty_Points;
    }
    if (!_PropagateDownLocked(tubeId) || !_PropagateUpLocked(tubeId, &snap)) {
        return rollback(false);
    }
    _PushUndoSnapshotLocked(snap);
    ++_version;
    return true;
}

std::vector<float> PomadeModel::ReadTubeDeltas(int tubeId) const
{
    std::vector<float> flat;
    if (tubeId == 0) {
        flat.assign(_host.centerX.size() * 3, 0.0f);
        return flat;
    }
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end()) {
        return flat;
    }
    PomadeShapeDeltas const &d = it->second.deltas;
    flat.reserve(d.centerDu.size() * 3);
    for (size_t i = 0; i < d.centerDu.size(); ++i) {
        flat.push_back(d.centerDu[i]);
        flat.push_back(d.centerDv[i]);
        flat.push_back(d.centerDw[i]);
    }
    return flat;
}

void PomadeModel::SetTubeLockParents(int tubeId, bool on)
{
    if (tubeId < 0) {
        _globalLockParents = on;
        return;
    }
    if (tubeId == 0) {
        _lockParents = on;
        return;
    }
    auto it = _tubes.find(tubeId);
    if (it != _tubes.end()) {
        it->second.lockParents = on;
    }
}

void PomadeModel::SetTubeLockChildren(int tubeId, bool on)
{
    if (tubeId < 0) {
        _globalLockChildren = on;
        return;
    }
    if (tubeId == 0) {
        _lockChildren = on;
        return;
    }
    auto it = _tubes.find(tubeId);
    if (it != _tubes.end()) {
        it->second.lockChildren = on;
    }
}

bool PomadeModel::GroupTubes(std::vector<int> const &tubeIds,
                            bool transientParent, int *outParent)
{
    if (!outParent || tubeIds.empty()) {
        _diagnostic = "PomadeModel::GroupTubes: want tube ids + outParent";
        return false;
    }
    std::vector<PomadeTubeDesc> actuals;
    for (int id : tubeIds) {
        PomadeTubeDesc desc;
        if (!_TubeDescLocked(id, &desc)) {
            _diagnostic = "PomadeModel::GroupTubes: unknown member tube";
            return false;
        }
        actuals.push_back(desc);
    }
    std::string derr;
    PomadeTubeDesc agg;
    if (!_LaneParentAverage(HasCudaMirror(), actuals, &agg, &derr)) {
        _diagnostic = derr;
        return false;
    }
    _PushUndoLocked();
    int const id = _nextGroupId--;
    agg.tubeId = id;
    HierarchyTube entry;
    entry.actual = agg;
    entry.derived = agg;  // self-derived: deltas stay zero until edited
    PomadeClearDeltas(agg, &entry.deltas);
    entry.transientParent = transientParent;
    entry.persistent = false;
    entry.members = tubeIds;
    _tubes[id] = entry;
    *outParent = id;
    ++_version;
    ++_mapVersion;
    _dirty |= PomadeDirty_Topology;
    return true;
}

bool PomadeModel::MakeTubePersistent(int tubeId)
{
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end()) {
        _diagnostic = "PomadeModel::MakeTubePersistent: unknown tube";
        return false;
    }
    _PushUndoLocked();
    it->second.persistent = true;
    ++_version;
    return true;
}

// -- V5: brush shaping (plan/18 section 3.6, section 7 G13) ---------------
//
// The audit found comb/twist/falloff/mirror living in Python or nowhere:
// the ABI took pre-weighted per-CV deltas, so every brush's shape was the
// tool's opinion and no two callers had to agree. These helpers are that
// maths, once, in C++; the loop now passes the stroke.
namespace {

bool
_PomadeIsSculptBrush(std::string const &b)
{
    return b == "grab" || b == "smooth" || b == "comb" || b == "lengthen" ||
           b == "shorten" || b == "twist";
}

// Row-major USD projection, top-left-origin physical pixels: the same
// convention Pomade_Pick and pomadeCamera carry, so a brush radius in
// pixels means the pixels the artist is looking at.
bool
_PomadeProjectPixel(float const *m, int w, int h, float px, float py,
                   float pz, float *outX, float *outY)
{
    double const x = double(px), y = double(py), z = double(pz);
    double const cx = x * m[0] + y * m[4] + z * m[8] + m[12];
    double const cy = x * m[1] + y * m[5] + z * m[9] + m[13];
    double const cw = x * m[3] + y * m[7] + z * m[11] + m[15];
    if (!(cw > 0.0)) {
        return false;  // behind the camera: outside every brush
    }
    double const ndcX = cx / cw;
    double const ndcY = cy / cw;
    *outX = float((ndcX * 0.5 + 0.5) * double(w));
    *outY = float((1.0 - (ndcY * 0.5 + 0.5)) * double(h));
    return true;
}

float
_PomadeSmoothStep(double s)
{
    if (s <= 0.0) {
        return 0.0f;
    }
    if (s >= 1.0) {
        return 1.0f;
    }
    return float(s * s * (3.0 - 2.0 * s));
}

// Screen-disc weight times the t-window weight (plan/17 section 5.5).
// tRadius <= 0 means "no t bound", which is what PomadeToolState's
// brushTRadius = 0 has always documented.
float
_PomadeBrushWeight(double distPx, double radiusPx, double t, double tCenter,
                  double tRadius)
{
    float w = 0.0f;
    if (radiusPx > 0.0) {
        w = _PomadeSmoothStep(1.0 - distPx / radiusPx);
    } else {
        w = (distPx == 0.0) ? 1.0f : 0.0f;
    }
    if (w <= 0.0f || !(tRadius > 0.0)) {
        return w;
    }
    double const d = t - tCenter;
    return w * _PomadeSmoothStep(1.0 - (d < 0.0 ? -d : d) / tRadius);
}

// One stroke shaped into per-CV deltas over the positions the stroke sees
// (the tube's own for the primary pass, mirrored across x = 0 for the
// symmetric pass). Only the CVs that actually move come back.
bool
_PomadeShapeStroke(std::string const &brush, std::vector<float> const &cx,
                  std::vector<float> const &cy, std::vector<float> const &cz,
                  std::vector<float> const *handleX,
                  std::vector<float> const *handleY,
                  std::vector<float> const *handleZ,
                  float const *viewProj, int w, int h, float cursorX,
                  float cursorY, float radiusPx, float const *deltaWorld,
                  float amount, float tCenter, float tRadius,
                  std::vector<int> *outIds, std::vector<float> *outDeltas,
                  std::string *err)
{
    int const n = int(cx.size());
    if (n < 2 || int(cy.size()) != n || int(cz.size()) != n) {
        if (err) {
            *err = "PomadeModel::SculptStrokeShaped: tube has no center curve";
        }
        return false;
    }
    bool const handlesValid = handleX && handleY && handleZ &&
        int(handleX->size()) == n && int(handleY->size()) == n &&
        int(handleZ->size()) == n;
    std::vector<float> weight(size_t(n), 0.0f);
    for (int i = 0; i < n; ++i) {
        float sx = 0.0f, sy = 0.0f;
        float const px = handlesValid ? (*handleX)[size_t(i)] : cx[size_t(i)];
        float const py = handlesValid ? (*handleY)[size_t(i)] : cy[size_t(i)];
        float const pz = handlesValid ? (*handleZ)[size_t(i)] : cz[size_t(i)];
        if (!_PomadeProjectPixel(viewProj, w, h, px, py, pz, &sx, &sy)) {
            continue;
        }
        double const dx = double(sx) - double(cursorX);
        double const dy = double(sy) - double(cursorY);
        double const t = double(i) / double(n - 1);
        weight[size_t(i)] = _PomadeBrushWeight(std::sqrt(dx * dx + dy * dy),
                                              double(radiusPx), t,
                                              double(tCenter),
                                              double(tRadius));
    }
    std::vector<float> delta(size_t(n) * 3, 0.0f);
    double const dwx = double(deltaWorld[0]);
    double const dwy = double(deltaWorld[1]);
    double const dwz = double(deltaWorld[2]);
    double const dwLen = std::sqrt(dwx * dwx + dwy * dwy + dwz * dwz);
    if (brush == "grab") {
        for (int i = 0; i < n; ++i) {
            double const k = double(weight[size_t(i)]);
            delta[size_t(i) * 3 + 0] = float(dwx * k);
            delta[size_t(i) * 3 + 1] = float(dwy * k);
            delta[size_t(i) * 3 + 2] = float(dwz * k);
        }
    } else if (brush == "comb") {
        if (!(dwLen > 1e-12)) {
            outIds->clear();
            outDeltas->clear();
            return true;  // no direction: a no-op, never a NaN
        }
        double const mag = (amount > 0.0f) ? double(amount) : dwLen;
        double const ux = dwx / dwLen, uy = dwy / dwLen, uz = dwz / dwLen;
        for (int i = 0; i < n; ++i) {
            double const k = double(weight[size_t(i)]) * mag;
            delta[size_t(i) * 3 + 0] = float(ux * k);
            delta[size_t(i) * 3 + 1] = float(uy * k);
            delta[size_t(i) * 3 + 2] = float(uz * k);
        }
    } else if (brush == "smooth") {
        double strength = (amount > 0.0f) ? double(amount) : 1.0;
        strength = strength > 1.0 ? 1.0 : strength;
        for (int i = 1; i < n - 1; ++i) {
            double const k = double(weight[size_t(i)]) * strength;
            double const ax = 0.5 * (double(cx[size_t(i - 1)]) +
                                     double(cx[size_t(i + 1)]));
            double const ay = 0.5 * (double(cy[size_t(i - 1)]) +
                                     double(cy[size_t(i + 1)]));
            double const az = 0.5 * (double(cz[size_t(i - 1)]) +
                                     double(cz[size_t(i + 1)]));
            delta[size_t(i) * 3 + 0] = float((ax - double(cx[size_t(i)])) * k);
            delta[size_t(i) * 3 + 1] = float((ay - double(cy[size_t(i)])) * k);
            delta[size_t(i) * 3 + 2] = float((az - double(cz[size_t(i)])) * k);
        }
    } else if (brush == "lengthen" || brush == "shorten") {
        // Root-pinned stretch along the CV's own offset from the root:
        // at weight 1 everywhere this is the uniform rescale by
        // (1 + amount) the length field does, and under a brush it grows
        // only the span the artist is over. "shorten" is the same with a
        // negative amount.
        double const scale = (brush == "shorten" && amount > 0.0f)
                                 ? -double(amount)
                                 : double(amount);
        for (int i = 1; i < n; ++i) {
            double const k = double(weight[size_t(i)]) * scale;
            delta[size_t(i) * 3 + 0] =
                float((double(cx[size_t(i)]) - double(cx[0])) * k);
            delta[size_t(i) * 3 + 1] =
                float((double(cy[size_t(i)]) - double(cy[0])) * k);
            delta[size_t(i) * 3 + 2] =
                float((double(cz[size_t(i)]) - double(cz[0])) * k);
        }
    } else {  // twist
        double ax = double(cx[size_t(n - 1)]) - double(cx[0]);
        double ay = double(cy[size_t(n - 1)]) - double(cy[0]);
        double az = double(cz[size_t(n - 1)]) - double(cz[0]);
        double const alen = std::sqrt(ax * ax + ay * ay + az * az);
        if (!(alen > 1e-12)) {
            outIds->clear();
            outDeltas->clear();
            return true;  // degenerate chord: nothing to rotate about
        }
        ax /= alen;
        ay /= alen;
        az /= alen;
        for (int i = 1; i < n; ++i) {
            double const theta = double(amount) * double(weight[size_t(i)]);
            if (theta == 0.0) {
                continue;
            }
            double const ox = double(cx[size_t(i)]) - double(cx[0]);
            double const oy = double(cy[size_t(i)]) - double(cy[0]);
            double const oz = double(cz[size_t(i)]) - double(cz[0]);
            double const dot = ox * ax + oy * ay + oz * az;
            double const parX = ax * dot, parY = ay * dot, parZ = az * dot;
            double const perX = ox - parX, perY = oy - parY, perZ = oz - parZ;
            // Rodrigues about the root-to-tip chord.
            double const crX = ay * perZ - az * perY;
            double const crY = az * perX - ax * perZ;
            double const crZ = ax * perY - ay * perX;
            double const ct = std::cos(theta), st = std::sin(theta);
            delta[size_t(i) * 3 + 0] =
                float(perX * (ct - 1.0) + crX * st);
            delta[size_t(i) * 3 + 1] =
                float(perY * (ct - 1.0) + crY * st);
            delta[size_t(i) * 3 + 2] =
                float(perZ * (ct - 1.0) + crZ * st);
        }
    }
    outIds->clear();
    outDeltas->clear();
    for (int i = 0; i < n; ++i) {
        float const dx = delta[size_t(i) * 3 + 0];
        float const dy = delta[size_t(i) * 3 + 1];
        float const dz = delta[size_t(i) * 3 + 2];
        if (dx == 0.0f && dy == 0.0f && dz == 0.0f) {
            continue;
        }
        outIds->push_back(i);
        outDeltas->push_back(dx);
        outDeltas->push_back(dy);
        outDeltas->push_back(dz);
    }
    return true;
}

}  // namespace

bool PomadeModel::SculptStroke(int tubeId, const char *brush, int const *cvIds,
                              float const *deltas, int cvCount,
                              bool preserveLength, bool mirrorX)
{
    std::string const b = brush ? brush : "";
    if (!_PomadeIsSculptBrush(b)) {
        _diagnostic = "PomadeModel::SculptStroke: unknown brush";
        return false;
    }
    if (cvCount < 0 || (cvCount > 0 && (!cvIds || !deltas))) {
        _diagnostic = "PomadeModel::SculptStroke: want CV ids + deltas";
        return false;
    }
    if (cvCount == 0) {
        return true;  // empty stroke: valid no-op, no version bump
    }
    HierarchyRollback const snap = _SnapshotHierarchyLocked();
    if (!_SculptApplyLocked(tubeId, b, cvIds, deltas, cvCount,
                            preserveLength, mirrorX, &snap)) {
        _RestoreHierarchyLocked(snap);
        return false;
    }
    _PushUndoSnapshotLocked(snap);
    ++_version;
    return true;
}

// One stroke's worth of model mutation, with the CALLER owning the
// rollback snapshot, the undo push and the version bump: SculptStroke runs
// it once, SculptStrokeShaped runs it twice (the stroke and its mirror)
// inside one bracket, which is what makes a symmetric stroke one undo step
// and one version.
bool PomadeModel::_SculptApplyLocked(int tubeId, std::string const &b,
                                    int const *cvIds, float const *deltas,
                                    int cvCount, bool preserveLength,
                                    bool mirrorX,
                                    HierarchyRollback const *beforeEdit)
{
    // Snapshot the pre-stroke length for preserveLength.
    PomadeTubeDesc before;
    if (!_TubeDescLocked(tubeId, &before)) {
        _diagnostic = "PomadeModel::SculptStroke: unknown tube";
        return false;
    }
    float const preLen = PomadeCenterArcLength(
        before.centerX.data(), before.centerY.data(), before.centerZ.data(),
        int(before.centerX.size()));
    for (int k = 0; k < cvCount; ++k) {
        int const cv = cvIds[k];
        float dx = deltas[size_t(k) * 3 + 0];
        float const dy = deltas[size_t(k) * 3 + 1];
        float const dz = deltas[size_t(k) * 3 + 2];
        if (mirrorX) {
            dx = -dx;
        }
        if (tubeId == 0) {
            int const n = int(_host.centerX.size());
            if (cv < 0 || cv >= n || n < 2) {
                _diagnostic = "PomadeModel::SculptStroke: bad CV index";
                return false;
            }
            _host.centerX[size_t(cv)] += dx;
            _host.centerY[size_t(cv)] += dy;
            _host.centerZ[size_t(cv)] += dz;
        } else {
            auto it = _tubes.find(tubeId);
            if (it == _tubes.end()) {
                _diagnostic = "PomadeModel::SculptStroke: unknown tube";
                return false;
            }
            int const n = int(it->second.actual.centerX.size());
            if (cv < 0 || cv >= n) {
                _diagnostic = "PomadeModel::SculptStroke: bad CV index";
                return false;
            }
            it->second.actual.centerX[size_t(cv)] += dx;
            it->second.actual.centerY[size_t(cv)] += dy;
            it->second.actual.centerZ[size_t(cv)] += dz;
        }
    }
    if (b == "smooth") {
        // One 0.5 relax pass over the stroked CVs (interior only).
        for (int k = 0; k < cvCount; ++k) {
            int const cv = cvIds[k];
            auto relax = [&](std::vector<float> &x, std::vector<float> &y,
                             std::vector<float> &z) {
                int const n = int(x.size());
                if (cv <= 0 || cv >= n - 1) {
                    return;
                }
                x[size_t(cv)] = 0.5f * x[size_t(cv)] +
                                0.25f * (x[size_t(cv - 1)] + x[size_t(cv + 1)]);
                y[size_t(cv)] = 0.5f * y[size_t(cv)] +
                                0.25f * (y[size_t(cv - 1)] + y[size_t(cv + 1)]);
                z[size_t(cv)] = 0.5f * z[size_t(cv)] +
                                0.25f * (z[size_t(cv - 1)] + z[size_t(cv + 1)]);
            };
            if (tubeId == 0) {
                relax(_host.centerX, _host.centerY, _host.centerZ);
            } else {
                auto it = _tubes.find(tubeId);
                relax(it->second.actual.centerX, it->second.actual.centerY,
                      it->second.actual.centerZ);
            }
        }
    }
    std::string derr;
    if (preserveLength) {
        PomadeTubeDesc cur;
        if (!_TubeDescLocked(tubeId, &cur)) {
            return false;
        }
        std::vector<float> ox, oy, oz;
        if (!PomadeRescaleCenterLength(cur.centerX.data(), cur.centerY.data(),
                                      cur.centerZ.data(),
                                      int(cur.centerX.size()), preLen, &ox,
                                      &oy, &oz, &derr)) {
            _diagnostic = derr;
            return false;
        }
        if (tubeId == 0) {
            _host.centerX.swap(ox);
            _host.centerY.swap(oy);
            _host.centerZ.swap(oz);
        } else {
            auto it = _tubes.find(tubeId);
            it->second.actual.centerX.swap(ox);
            it->second.actual.centerY.swap(oy);
            it->second.actual.centerZ.swap(oz);
        }
    }
    if (tubeId == 0) {
        if (!Sync()) {
            return false;
        }
    } else {
        auto it = _tubes.find(tubeId);
        HierarchyTube &entry = it->second;
        std::vector<PomadeFrame> dframes;
        if (!PomadeTubeFramesCpu(entry.derived, &dframes, &derr)) {
            _diagnostic = derr;
            return false;
        }
        if (!PomadeComputeDeltasCpu(entry.actual, entry.derived, dframes,
                                   &entry.deltas, &derr)) {
            _diagnostic = derr;
            return false;
        }
    }
    _dirty |= PomadeDirty_Points;
    if (!_PropagateDownLocked(tubeId)) {
        return false;
    }
    if (!_PropagateUpLocked(tubeId, beforeEdit)) {
        return false;
    }
    return true;
}

bool
PomadeModel::SculptStrokeShaped(int tubeId, const char *brush,
                               float const *viewProj, int w, int h, float x,
                               float y, float radiusPx,
                               float const *deltaWorld, float amount,
                               float tCenter, float tRadius,
                               bool preserveLength, bool mirrorX,
                               int *outTouched)
{
    if (outTouched) {
        *outTouched = 0;
    }
    std::string const b = brush ? brush : "";
    if (!_PomadeIsSculptBrush(b)) {
        _diagnostic = "PomadeModel::SculptStrokeShaped: unknown brush";
        return false;
    }
    if (!viewProj || !deltaWorld) {
        _diagnostic = "PomadeModel::SculptStrokeShaped: want a projection "
                      "and a stroke delta";
        return false;
    }
    if (w < 1 || h < 1) {
        _diagnostic = "PomadeModel::SculptStrokeShaped: want a viewport size";
        return false;
    }
    // Lengthen SETS the length, so it is the one brush the
    // length-preserving default cannot apply to.
    bool const keepLength =
        preserveLength && b != "lengthen" && b != "shorten";
    // The shaping happened here, so the apply must take the deltas RAW:
    // no second smooth pass, no dx negation.
    std::string const kRaw("grab");

    PomadeTubeDesc desc;
    if (!_TubeDescLocked(tubeId, &desc)) {
        _diagnostic = "PomadeModel::SculptStrokeShaped: unknown tube";
        return false;
    }
    // Grab is a drag: the press-time footprint stays put while each move
    // applies its incremental view-plane delta to the CURRENT descriptor.
    // Reprojecting the moving CVs would make a long grab lose its own CV as
    // soon as it crossed the brush radius. Other brushes deliberately keep
    // their current-shape footprint.
    bool const frozenGrab = (b == "grab" && _gestureDepth > 0);
    auto gestureDesc = [&](int id, PomadeTubeDesc *out) {
        if (!out) {
            return false;
        }
        if (id == 0) {
            if (_gestureBase.cx.size() < 2 ||
                _gestureBase.cy.size() != _gestureBase.cx.size() ||
                _gestureBase.cz.size() != _gestureBase.cx.size()) {
                return false;
            }
            out->centerX = _gestureBase.cx;
            out->centerY = _gestureBase.cy;
            out->centerZ = _gestureBase.cz;
            out->sections = _gestureBase.sections;
            out->ringVerts = _gestureBase.shape.ringVerts;
            out->tubeId = 0;
            out->regionId = _gestureBase.tube0Region;
            out->level = 1;
            out->parentTubeId = -1;
            out->childIndex = -1;
            out->rootFramePinned = _gestureBase.tube0RootFramePinned;
            out->rootFrame = _gestureBase.tube0RootFrame;
            out->frameReference = _gestureBase.tube0FrameReference;
            return true;
        }
        auto const it = _gestureBase.tubes.find(id);
        if (it == _gestureBase.tubes.end()) {
            return false;
        }
        *out = it->second.actual;
        return out->centerX.size() >= 2;
    };
    PomadeTubeDesc footprint = desc;
    if (frozenGrab && !gestureDesc(tubeId, &footprint)) {
        _diagnostic = "PomadeModel::SculptStrokeShaped: gesture has no "
                      "press-time tube";
        return false;
    }
    if (footprint.centerX.size() != desc.centerX.size()) {
        _diagnostic = "PomadeModel::SculptStrokeShaped: tube layout changed "
                      "during grab";
        return false;
    }
    std::vector<int> ids;
    std::vector<float> deltas;
    std::string err;
    auto handlePositions = [](PomadeTubeDesc const &source,
                              bool mirror,
                              std::vector<float> *hx,
                              std::vector<float> *hy,
                              std::vector<float> *hz) {
        hx->resize(source.centerX.size());
        hy->resize(source.centerX.size());
        hz->resize(source.centerX.size());
        for (size_t i = 0; i < source.centerX.size(); ++i) {
            float x = source.centerX[i], y = source.centerY[i],
                  z = source.centerZ[i];
            std::string ignored;
            PomadeCenterHandlePointCpu(source, int(i), &x, &y, &z, &ignored);
            (*hx)[i] = mirror ? -x : x;
            (*hy)[i] = y;
            (*hz)[i] = z;
        }
    };
    std::vector<float> footprintX, footprintY, footprintZ;
    handlePositions(footprint, false, &footprintX, &footprintY, &footprintZ);
    if (!_PomadeShapeStroke(b, footprint.centerX, footprint.centerY,
                           footprint.centerZ, &footprintX, &footprintY,
                           &footprintZ,
                           viewProj, w, h, x, y, radiusPx, deltaWorld, amount,
                           tCenter, tRadius, &ids, &deltas, &err)) {
        _diagnostic = err;
        return false;
    }

    // The symmetric half: the tube whose root is the mirror of this one's
    // across x = 0. Sculpted in MIRRORED space (positions and stroke both
    // reflected), then the delta is reflected back, so "mirror-X" means
    // the same stroke on the other side rather than a negated dx here.
    int mirrorTube = 0;
    bool haveMirror = false;
    std::vector<int> mirrorIds;
    std::vector<float> mirrorDeltas;
    if (mirrorX && !footprint.centerX.empty()) {
        float const mx = -footprint.centerX.front();
        float const my = footprint.centerY.front();
        float const mz = footprint.centerZ.front();
        double best = -1.0;
        auto const consider = [&](int id, PomadeTubeDesc const &cand) {
            if (cand.centerX.empty()) {
                return;
            }
            double const dx = double(cand.centerX.front()) - double(mx);
            double const dy = double(cand.centerY.front()) - double(my);
            double const dz = double(cand.centerZ.front()) - double(mz);
            double const d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (best < 0.0 || d < best) {
                best = d;
                mirrorTube = id;
            }
        };
        if (frozenGrab) {
            PomadeTubeDesc base;
            if (gestureDesc(0, &base)) {
                consider(0, base);
            }
            for (auto const &kv : _gestureBase.tubes) {
                consider(kv.first, kv.second.actual);
            }
        } else {
            if (_host.centerX.size() >= 2) {
                PomadeTubeDesc hostDesc;
                if (_TubeDescLocked(0, &hostDesc)) {
                    consider(0, hostDesc);
                }
            }
            for (auto const &kv : _tubes) {
                consider(kv.first, kv.second.actual);
            }
        }
        float const arc = PomadeCenterArcLength(
            footprint.centerX.data(), footprint.centerY.data(),
            footprint.centerZ.data(), int(footprint.centerX.size()));
        double const tolerance = std::max(0.05 * double(arc), 1e-4);
        haveMirror = (best >= 0.0 && best <= tolerance);
    }
    if (haveMirror) {
        PomadeTubeDesc mdesc;
        if (!_TubeDescLocked(mirrorTube, &mdesc)) {
            _diagnostic = "PomadeModel::SculptStrokeShaped: unknown tube";
            return false;
        }
        PomadeTubeDesc mirrorFootprint = mdesc;
        if (frozenGrab && !gestureDesc(mirrorTube, &mirrorFootprint)) {
            _diagnostic = "PomadeModel::SculptStrokeShaped: gesture has no "
                          "press-time mirror tube";
            return false;
        }
        if (mirrorFootprint.centerX.size() != mdesc.centerX.size()) {
            _diagnostic = "PomadeModel::SculptStrokeShaped: mirror layout "
                          "changed during grab";
            return false;
        }
        std::vector<float> rx(mirrorFootprint.centerX.size());
        for (size_t i = 0; i < mirrorFootprint.centerX.size(); ++i) {
            rx[i] = -mirrorFootprint.centerX[i];
        }
        std::vector<float> mirroredHandleX, mirroredHandleY,
            mirroredHandleZ;
        handlePositions(mirrorFootprint, true, &mirroredHandleX,
                        &mirroredHandleY, &mirroredHandleZ);
        if (!_PomadeShapeStroke(b, rx, mirrorFootprint.centerY,
                               mirrorFootprint.centerZ, &mirroredHandleX,
                               &mirroredHandleY, &mirroredHandleZ,
                               viewProj, w, h, x, y,
                               radiusPx, deltaWorld, amount, tCenter, tRadius,
                               &mirrorIds, &mirrorDeltas, &err)) {
            _diagnostic = err;
            return false;
        }
        for (size_t i = 0; i < mirrorDeltas.size(); i += 3) {
            mirrorDeltas[i] = -mirrorDeltas[i];
        }
    }

    int touched = int(ids.size()) + int(mirrorIds.size());
    if (touched == 0) {
        return true;  // nothing under the brush: no version, no undo step
    }
    HierarchyRollback const snap = _SnapshotHierarchyLocked();
    if (!ids.empty() &&
        !_SculptApplyLocked(tubeId, kRaw, ids.data(), deltas.data(),
                            int(ids.size()), keepLength, false, &snap)) {
        _RestoreHierarchyLocked(snap);
        return false;
    }
    if (!mirrorIds.empty() && mirrorTube != tubeId &&
        !_SculptApplyLocked(mirrorTube, kRaw, mirrorIds.data(),
                            mirrorDeltas.data(), int(mirrorIds.size()),
                            keepLength, false, &snap)) {
        _RestoreHierarchyLocked(snap);
        return false;
    }
    if (mirrorTube == tubeId) {
        touched = int(ids.size());  // one tube, one pass: the stroke IS
                                    // its own mirror on the plane
    }
    _PushUndoSnapshotLocked(snap);
    ++_version;
    if (outTouched) {
        *outTouched = touched;
    }
    return true;
}

bool
PomadeModel::SubdivideTubeAlongEdge(int tubeId, float const *worldA,
                                   float const *worldB, int seed,
                                   std::vector<int> *outChildren)
{
    float a = 1.0f, b = 0.0f, c = 0.0f;
    {
        if (!worldA || !worldB) {
            _diagnostic = "PomadeModel::SubdivideTubeAlongEdge: want two "
                          "world points";
            return false;
        }
        PomadeTubeDesc desc;
        if (!_TubeDescLocked(tubeId, &desc)) {
            _diagnostic = "PomadeModel::SubdivideTubeAlongEdge: unknown tube";
            return false;
        }
        std::vector<PomadeFrame> frames;
        std::string derr;
        if (!PomadeTubeFramesCpu(desc, &frames, &derr) ||
            frames.empty()) {
            _diagnostic = derr.empty() ? "PomadeModel::SubdivideTubeAlongEdge:"
                                         " no root frame"
                                       : derr;
            return false;
        }
        // The root frame is the chart K14 partitions the root ring in:
        // u along the frame normal, v along its binormal, origin at
        // center CV 0 (pomadeHierarchy.cpp PlaceRing).
        PomadeFrame const &fr = frames.front();
        double const ox = double(desc.centerX.front());
        double const oy = double(desc.centerY.front());
        double const oz = double(desc.centerZ.front());
        auto const chart = [&](float const *p, double *u, double *v) {
            double const dx = double(p[0]) - ox;
            double const dy = double(p[1]) - oy;
            double const dz = double(p[2]) - oz;
            *u = dx * double(fr.nx) + dy * double(fr.ny) + dz * double(fr.nz);
            *v = dx * double(fr.bx) + dy * double(fr.by) + dz * double(fr.bz);
        };
        double uA = 0.0, vA = 0.0, uB = 0.0, vB = 0.0;
        chart(worldA, &uA, &vA);
        chart(worldB, &uB, &vB);
        double la = vB - vA;
        double lb = -(uB - uA);
        double const norm = std::sqrt(la * la + lb * lb);
        if (!(norm > 1e-9)) {
            _diagnostic = "PomadeModel::SubdivideTubeAlongEdge: the stroke "
                          "is a point in the root plane";
            return false;
        }
        la /= norm;
        lb /= norm;
        a = float(la);
        b = float(lb);
        c = float(-(la * uA + lb * vA));
    }
    return SubdivideTube(tubeId, 2, "edge", seed, a, b, c, outChildren);
}

std::vector<float>
PomadeModel::SmoothnessScores() const
{
    std::vector<float> out;
    int const n = int(_host.centerX.size());
    if (n < 1) {
        return out;
    }
    out.assign(size_t(n), 0.0f);
    std::string derr;
    if (!_LaneSmoothnessScore(HasCudaMirror(), _host.centerX.data(),
                              _host.centerY.data(), _host.centerZ.data(), n,
                              out.data(), &derr)) {
        out.clear();
    }
    return out;
}

bool
PomadeModel::CheckRootIntersections()
{
    std::vector<int> ids;
    std::vector<PomadeTubeDesc> descs;
    if (_host.centerX.size() >= 2) {
        ids.push_back(0);
        descs.push_back(_BuildTubeDescLocked());
        descs.back().tubeId = 0;
    }
    for (auto const &kv : _tubes) {
        ids.push_back(kv.first);
        descs.push_back(kv.second.actual);
    }
    std::vector<int> flags(descs.size(), 0);
    std::string derr;
    if (!_LaneTubeIntersect(HasCudaMirror(), descs.data(),
                            int(descs.size()), flags.data(), &derr)) {
        _diagnostic = derr;
        return false;
    }
    _intersectFlags.clear();
    for (size_t i = 0; i < ids.size(); ++i) {
        _intersectFlags[ids[i]] = flags[i];
    }
    return true;
}

std::vector<int>
PomadeModel::IntersectedTubes() const
{
    std::vector<int> out;
    for (auto const &kv : _intersectFlags) {
        if (kv.second) {
            out.push_back(kv.first);
        }
    }
    return out;
}

std::vector<int>
PomadeModel::TubeIds() const
{
    std::vector<int> out;
    if (_host.centerX.size() >= 2) {
        out.push_back(0);
    }
    for (auto const &kv : _tubes) {
        out.push_back(kv.first);
    }
    return out;
}

bool
PomadeModel::ImportLockedTube(int parentId, float const *cx, float const *cy,
                             float const *cz, int nCv, float const *secT,
                             float const *secU, float const *secV, int nSec,
                             int ringVerts, int *outTubeId)
{
    return _ImportLockedTubeLocked(parentId, cx, cy, cz, nCv, secT, secU,
                                   secV, nSec, ringVerts, outTubeId);
}

bool
PomadeModel::ImportSweptMesh(int parentId, float const *points, int ringCount,
                            int ringVerts, int *outTubeId)
{
    if (!points || !outTubeId) {
        _diagnostic = "PomadeModel::ImportSweptMesh: null input or output";
        return false;
    }
    if (ringCount < 2 || ringVerts < 3 || ringVerts > 32) {
        _diagnostic = "PomadeModel::ImportSweptMesh: need >= 2 rings, "
                      "ringVerts in [3, 32]";
        return false;
    }
    // Ring centroids are the centers (K4 needs >= 2 CVs: guaranteed).
    std::vector<float> cx(static_cast<size_t>(ringCount)),
        cy(static_cast<size_t>(ringCount)),
        cz(static_cast<size_t>(ringCount));
    for (int r = 0; r < ringCount; ++r) {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (int k = 0; k < ringVerts; ++k) {
            size_t const o =
                (size_t(r) * size_t(ringVerts) + size_t(k)) * 3;
            sx += points[o + 0];
            sy += points[o + 1];
            sz += points[o + 2];
        }
        cx[size_t(r)] = float(sx / double(ringVerts));
        cy[size_t(r)] = float(sy / double(ringVerts));
        cz[size_t(r)] = float(sz / double(ringVerts));
    }
    std::vector<PomadeFrame> frames;
    std::string derr;
    if (!PomadeCenterFramesCpu(cx.data(), cy.data(), cz.data(), ringCount,
                              &frames, &derr)) {
        _diagnostic = derr;
        return false;
    }
    // Sections: uniform t, rings projected into the K4 charts.
    PomadeTubeDesc centers;
    centers.centerX = cx;
    centers.centerY = cy;
    centers.centerZ = cz;
    std::vector<float> secT(static_cast<size_t>(ringCount));
    std::vector<float> secU(static_cast<size_t>(ringCount) *
                            static_cast<size_t>(ringVerts));
    std::vector<float> secV(static_cast<size_t>(ringCount) *
                            static_cast<size_t>(ringVerts));
    for (int r = 0; r < ringCount; ++r) {
        float const t = ringCount > 1 ? float(r) / float(ringCount - 1) : 0.0f;
        secT[size_t(r)] = t;
        float cp[3];
        PomadeFrame fr;
        if (!PomadeSampleCenterCpu(centers, frames, t, &cp[0], &cp[1], &cp[2],
                                  &fr, &derr)) {
            _diagnostic = derr;
            return false;
        }
        for (int k = 0; k < ringVerts; ++k) {
            size_t const o =
                (size_t(r) * size_t(ringVerts) + size_t(k)) * 3;
            float const dx = points[o + 0] - cp[0];
            float const dy = points[o + 1] - cp[1];
            float const dz = points[o + 2] - cp[2];
            size_t const s = size_t(r) * size_t(ringVerts) + size_t(k);
            secU[s] = dx * fr.nx + dy * fr.ny + dz * fr.nz;
            secV[s] = dx * fr.bx + dy * fr.by + dz * fr.bz;
        }
    }
    return _ImportLockedTubeLocked(parentId, cx.data(), cy.data(), cz.data(),
                                   ringCount, secT.data(), secU.data(),
                                   secV.data(), ringCount, ringVerts,
                                   outTubeId);
}

bool
PomadeModel::_ImportLockedTubeLocked(int parentId, float const *cx,
                                    float const *cy, float const *cz, int nCv,
                                    float const *secT, float const *secU,
                                    float const *secV, int nSec, int ringVerts,
                                    int *outTubeId)
{
    if (!cx || !cy || !cz || !secT || !secU || !secV || !outTubeId) {
        _diagnostic = "PomadeModel::ImportLockedTube: null input or output";
        return false;
    }
    if (nCv < 2 || nSec < 1 || ringVerts < 3 || ringVerts > 32) {
        _diagnostic = "PomadeModel::ImportLockedTube: need >= 2 CVs, >= 1 "
                      "section, ringVerts in [3, 32]";
        return false;
    }
    for (int s = 0; s < nSec; ++s) {
        if (!(secT[s] >= 0.0f) || !(secT[s] <= 1.0f) ||
            (s > 0 && secT[s] < secT[s - 1])) {
            _diagnostic = "PomadeModel::ImportLockedTube: section t must "
                          "ascend in [0, 1]";
            return false;
        }
    }
    int parentLevel = 0, parentRegion = 0;
    if (parentId == 0) {
        if (_host.centerX.size() < 2) {
            _diagnostic = "PomadeModel::ImportLockedTube: no tube 0";
            return false;
        }
        parentLevel = 1;
    } else {
        auto pit = _tubes.find(parentId);
        if (pit == _tubes.end()) {
            _diagnostic = "PomadeModel::ImportLockedTube: unknown parent";
            return false;
        }
        parentLevel = pit->second.actual.level;
        parentRegion = pit->second.actual.regionId;
    }
    // Next free childIndex under the parent (imports append after any
    // subdivided siblings); the id then cannot collide with them. A
    // fixup loop covers adversarial cases (negative group parents).
    int childIndex = 0;
    for (auto const &kv : _tubes) {
        if (kv.second.actual.parentTubeId == parentId &&
            kv.second.actual.childIndex >= childIndex) {
            childIndex = kv.second.actual.childIndex + 1;
        }
    }
    int id = parentId * 16 + 1 + childIndex;
    while (id == 0 || _tubes.count(id) > 0) {
        ++childIndex;
        id = parentId * 16 + 1 + childIndex;
    }
    PomadeTubeDesc actual;
    actual.centerX.assign(cx, cx + nCv);
    actual.centerY.assign(cy, cy + nCv);
    actual.centerZ.assign(cz, cz + nCv);
    actual.ringVerts = ringVerts;
    actual.tubeId = id;
    actual.regionId = parentRegion;
    actual.level = parentLevel + 1;
    actual.parentTubeId = parentId;
    actual.childIndex = childIndex;
    actual.sections.resize(size_t(nSec));
    for (int s = 0; s < nSec; ++s) {
        PomadeTubeSection &sec = actual.sections[size_t(s)];
        sec.t = secT[s];
        sec.scale = 1.0f;
        sec.twist = 0.0f;
        sec.u.assign(secU + size_t(s) * size_t(ringVerts),
                     secU + (size_t(s) + 1) * size_t(ringVerts));
        sec.v.assign(secV + size_t(s) * size_t(ringVerts),
                     secV + (size_t(s) + 1) * size_t(ringVerts));
    }
    HierarchyTube entry;
    entry.actual = actual;
    entry.derived = actual;  // explicit shape: no derivation to match
    PomadeClearDeltas(actual, &entry.deltas);
    entry.imported = true;
    // An import inherits its parent's fill like a subdivided child would,
    // so an artist who unlocks it later starts from the family's density.
    {
        auto pit = _tubes.find(parentId);
        entry.fill = pit != _tubes.end() ? pit->second.fill : _fill;
    }
    _PushUndoLocked();
    _tubes[id] = entry;
    *outTubeId = id;
    ++_version;
    ++_mapVersion;
    _dirty |= PomadeDirty_Topology;
    return true;
}

int
PomadeModel::GetTubeSectionCount(int tubeId) const
{
    if (tubeId == 0) {
        return int(_sections.size());
    }
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end()) {
        return -1;
    }
    return int(it->second.actual.sections.size());
}

bool
PomadeModel::GetTubeSection(int tubeId, int ring,
                           PomadeTubeSection *section) const
{
    if (!section) {
        return false;
    }
    if (tubeId == 0) {
        if (ring < 0 || ring >= int(_sections.size())) {
            return false;
        }
        *section = _sections[size_t(ring)];
        return true;
    }
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end() || ring < 0 ||
        ring >= int(it->second.actual.sections.size())) {
        return false;
    }
    *section = it->second.actual.sections[size_t(ring)];
    return true;
}

bool
PomadeModel::IsTubeImported(int tubeId) const
{
    auto it = _tubes.find(tubeId);
    return it != _tubes.end() && it->second.imported;
}

// -- V0b: the stage contract (plan/18 §7 G1-G3) ------------------------------

bool
PomadeModel::GetTubeDesc(int tubeId, PomadeTubeDesc *out) const
{
    return _TubeDescLocked(tubeId, out);
}

bool
PomadeModel::GetTubeRecord(int tubeId, TubeRecord *out) const
{
    if (!out) {
        return false;
    }
    *out = TubeRecord();
    if (tubeId == 0) {
        if (_host.centerX.size() < 2 || !_TubeDescLocked(0, &out->actual)) {
            return false;
        }
        out->subdivide = _subdivide;
        out->fill = _fill;
        out->locked = _locked;
        out->lockParents = _lockParents;
        out->lockChildren = _lockChildren;
        out->hasTube = true;
        return true;
    }
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end()) {
        return false;
    }
    HierarchyTube const &entry = it->second;
    out->actual = entry.actual;
    out->derived = entry.derived;
    out->deltas = entry.deltas;
    out->subdivide.count = entry.subdivide.count;
    out->subdivide.seed = entry.subdivide.seed;
    out->subdivide.splitMode =
        entry.subdivide.splitMode == PomadeSplit_Edge ? "edge" : "kmeans";
    out->fill = entry.fill;
    // A store tube carries no independent "locked" bit: an import is the
    // frozen kind (plan/17 §5.7), and that is what the stage's
    // usdGen:pomade:locked records for a child.
    out->locked = entry.imported;
    out->lockParents = entry.lockParents;
    out->lockChildren = entry.lockChildren;
    out->transientParent = entry.transientParent;
    out->persistent = entry.persistent;
    out->imported = entry.imported;
    out->members = entry.members;
    out->hasTube = true;
    return true;
}

void
PomadeModel::ClearHierarchy()
{
    _tubes.clear();
    _PruneActiveCutLocked();
    _intersectFlags.clear();
    _regionTube.clear();
    _tubeRegionKey.clear();
    _nextGroupId = -1;
    _ClearUndoLocked();
    ++_version;
    ++_mapVersion;
    _dirty |= PomadeDirty_Topology;
}

bool
PomadeModel::RestoreTubeRecord(int tubeId, TubeRecord const &record)
{
    if (tubeId == 0) {
        _diagnostic = "PomadeModel::RestoreTubeRecord: tube 0 restores through "
                      "Restore()";
        return false;
    }
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end()) {
        _diagnostic = "PomadeModel::RestoreTubeRecord: unknown tube";
        return false;
    }
    if (record.actual.centerX.size() < 2 ||
        record.actual.sections.size() < 2) {
        _diagnostic = "PomadeModel::RestoreTubeRecord: degenerate shape";
        return false;
    }
    HierarchyTube &entry = it->second;
    // Hydrate subdivides before restoring the saved child.  K14 can choose a
    // different ring count than the persisted actual after a K7 rebase, so
    // normalize the fresh reference to the saved actual layout before
    // installing its residuals.  This is the same fixed-layout contract as
    // K7/K6, and keeps the next parent edit from re-sampling the child.
    PomadeTubeDesc actual = record.actual;
    actual.tubeId = entry.actual.tubeId;
    actual.parentTubeId = entry.actual.parentTubeId;
    actual.childIndex = entry.actual.childIndex;
    actual.level = entry.actual.level;
    if (!record.imported && !_SameRingLayout(entry.derived, actual)) {
        PomadeTubeDesc matched;
        std::string derr;
        if (!PomadeResampleDescRingsCpu(entry.derived, actual.ringVerts,
                                       &matched, &derr)) {
            _diagnostic = derr;
            return false;
        }
        entry.derived = std::move(matched);
    }
    // Legacy layers predate explicit parent-boundary links. Their freshly
    // re-derived, layout-matched reference is the only authoritative source
    // for reconstructing those links; a current layer's empty binding set is
    // still an authored statement and must remain empty.
    if (!record.imported && !record.hasInheritedBoundaryBindings) {
        actual.inheritedBoundaryBindings =
            entry.derived.inheritedBoundaryBindings;
    }
    entry.actual = std::move(actual);
    if (!record.deltas.centerDu.empty()) {
        entry.deltas = record.deltas;
    }
    entry.subdivide.count = record.subdivide.count;
    entry.subdivide.seed = record.subdivide.seed;
    entry.subdivide.splitMode = record.subdivide.splitMode == "edge"
                                    ? PomadeSplit_Edge
                                    : PomadeSplit_KMeans;
    entry.fill = record.fill;
    entry.lockParents = record.lockParents;
    entry.lockChildren = record.lockChildren;
    entry.transientParent = record.transientParent;
    entry.persistent = record.persistent;
    if (record.imported) {
        entry.imported = true;
        entry.derived = entry.actual;
    }
    if (!record.members.empty()) {
        entry.members = record.members;
    }
    ++_version;
    _dirty |= PomadeDirty_Topology | PomadeDirty_Points;
    return true;
}

bool
PomadeModel::SetTubeFillParams(int tubeId, FillParams params)
{
    if (tubeId == 0) {
        return SetFillParams(std::move(params));
    }
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end()) {
        _diagnostic = "PomadeModel::SetTubeFillParams: unknown tube";
        return false;
    }
    if (!(params.density >= 0.0f) || params.cvCount < 2 ||
        params.cvCount > 64 || !(params.edgeBias >= -1.0f) ||
        !(params.edgeBias <= 1.0f) || params.lengthProfile.size() % 2 != 0) {
        _diagnostic = "PomadeModel::SetTubeFillParams: density >= 0, cvCount in "
                      "[2, 64], edgeBias in [-1, 1], paired profile";
        return false;
    }
    it->second.fill = std::move(params);
    ++_version;
    _dirty |= PomadeDirty_Guides;
    return true;
}

bool
PomadeModel::GetTubeFillParams(int tubeId, FillParams *out) const
{
    if (!out) {
        return false;
    }
    if (tubeId == 0) {
        *out = GetFillParams();
        return true;
    }
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end()) {
        return false;
    }
    *out = it->second.fill;
    return true;
}

bool
PomadeModel::IsTubeFillSuspended(int tubeId) const
{
    if (tubeId != 0 && _tubes.find(tubeId) == _tubes.end()) {
        return false;
    }
    return _FillSuspendedLocked(tubeId);
}

bool
PomadeModel::_FillSuspendedLocked(int tubeId) const
{
    for (auto const &kv : _tubes) {
        if (kv.second.actual.parentTubeId == tubeId && !kv.second.imported) {
            return true;
        }
    }
    return false;
}

bool
PomadeModel::_FinishChildEditLocked(int tubeId, bool layoutChanged,
                                   uint32_t dirtyBits,
                                   HierarchyRollback const *beforeEdit)
{
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end()) {
        _diagnostic = "PomadeModel: unknown tube";
        return false;
    }
    HierarchyTube &entry = it->second;
    bool const derivedChild = !entry.imported &&
                              entry.actual.parentTubeId >= 0 &&
                              entry.derived.centerX.size() >= 2;
    if (layoutChanged && derivedChild) {
        _diagnostic = "PomadeModel: the operation changes a derived child's "
                      "layout; edit the parent or re-subdivide instead";
        return false;
    }
    if (derivedChild) {
        std::vector<PomadeFrame> dframes;
        std::string derr;
        if (!PomadeTubeFramesCpu(entry.derived, &dframes, &derr)) {
            _diagnostic = derr;
            return false;
        }
        if (!PomadeComputeDeltasCpu(entry.actual, entry.derived, dframes,
                                   &entry.deltas, &derr)) {
            _diagnostic = derr;
            return false;
        }
    } else {
        // An import or an on-the-fly parent is self-derived: it has no
        // derivation to measure against, so the deltas stay zero and the
        // baseline follows the edit (K6/K7 read it, never write it).
        entry.derived = entry.actual;
        PomadeClearDeltas(entry.actual, &entry.deltas);
    }
    _dirty |= dirtyBits;
    if (!_PropagateDownLocked(tubeId)) {
        return false;
    }
    return _PropagateUpLocked(tubeId, beforeEdit);
}

bool
PomadeModel::_EditChildTube(int tubeId, bool layoutChanged, uint32_t dirtyBits,
                           std::function<bool(PomadeTubeDesc &)> const &edit)
{
    auto it = _tubes.find(tubeId);
    if (it == _tubes.end()) {
        _diagnostic = "PomadeModel: unknown tube";
        return false;
    }
    HierarchyRollback const snap = _SnapshotHierarchyLocked();
    if (!edit(it->second.actual) ||
        !_FinishChildEditLocked(tubeId, layoutChanged, dirtyBits, &snap)) {
        _RestoreHierarchyLocked(snap);
        return false;
    }
    _PushUndoSnapshotLocked(snap);
    ++_version;
    return true;
}

// The per-tube editing surface (plan/18 §7 G2). Tube 0 delegates to the
// single-tube spellings above, which keep their exact behaviour (soft
// selection, the _useSections flip, the legacy shape fields); every other id
// runs the store path plus K6 down / K7 up propagation.

bool
PomadeModel::InsertTubeCenterCV(int tubeId, int atIndex)
{
    if (tubeId == 0) {
        return InsertCenterCV(atIndex);
    }
    return _EditChildTube(
        tubeId, /*layoutChanged*/ true,
        PomadeDirty_Topology | PomadeDirty_Points,
        [&](PomadeTubeDesc &desc) {
            int const n = int(desc.centerX.size());
            if (atIndex < 0 || atIndex > n || n < 2) {
                _diagnostic =
                    "PomadeModel::InsertTubeCenterCV: index out of range";
                return false;
            }
            int i0 = atIndex - 1, i1 = atIndex;
            if (i0 < 0) {
                i0 = 0;
                i1 = 1;
            } else if (i1 >= n) {
                i0 = n - 2;
                i1 = n - 1;
            }
            float const mx =
                0.5f * (desc.centerX[size_t(i0)] + desc.centerX[size_t(i1)]);
            float const my =
                0.5f * (desc.centerY[size_t(i0)] + desc.centerY[size_t(i1)]);
            float const mz =
                0.5f * (desc.centerZ[size_t(i0)] + desc.centerZ[size_t(i1)]);
            desc.centerX.insert(desc.centerX.begin() + atIndex, mx);
            desc.centerY.insert(desc.centerY.begin() + atIndex, my);
            desc.centerZ.insert(desc.centerZ.begin() + atIndex, mz);
            return true;
        });
}

bool
PomadeModel::DeleteTubeCenterCV(int tubeId, int index)
{
    if (tubeId == 0) {
        return DeleteCenterCV(index);
    }
    return _EditChildTube(
        tubeId, /*layoutChanged*/ true,
        PomadeDirty_Topology | PomadeDirty_Points,
        [&](PomadeTubeDesc &desc) {
            int const n = int(desc.centerX.size());
            if (index < 0 || index >= n || n <= 2) {
                _diagnostic =
                    "PomadeModel::DeleteTubeCenterCV: need >= 2 CVs left";
                return false;
            }
            desc.centerX.erase(desc.centerX.begin() + index);
            desc.centerY.erase(desc.centerY.begin() + index);
            desc.centerZ.erase(desc.centerZ.begin() + index);
            return true;
        });
}

bool
PomadeModel::SetTubeLengthFor(int tubeId, float length)
{
    if (tubeId == 0) {
        return SetTubeLength(length);
    }
    return _EditChildTube(
        tubeId, /*layoutChanged*/ false, PomadeDirty_Points,
        [&](PomadeTubeDesc &desc) {
            int const n = int(desc.centerX.size());
            if (!(length > 0.0f) || n < 2) {
                _diagnostic =
                    "PomadeModel::SetTubeLengthFor: need a tube + length > 0";
                return false;
            }
            std::vector<float> ox, oy, oz;
            std::string derr;
            if (!PomadeRescaleCenterLength(desc.centerX.data(),
                                          desc.centerY.data(),
                                          desc.centerZ.data(), n, length, &ox,
                                          &oy, &oz, &derr)) {
                _diagnostic = derr;
                return false;
            }
            desc.centerX = std::move(ox);
            desc.centerY = std::move(oy);
            desc.centerZ = std::move(oz);
            return true;
        });
}

bool
PomadeModel::MatchTubeSurface(int tubeId)
{
    if (tubeId == 0) {
        return MatchSurface();
    }
    return _EditChildTube(
        tubeId, /*layoutChanged*/ false, PomadeDirty_Points,
        [&](PomadeTubeDesc &desc) {
            if (!_scalp || !_scalp->finalized || desc.centerX.empty()) {
                _diagnostic =
                    "PomadeModel::MatchTubeSurface: need a bound scalp + tube";
                return false;
            }
            float const p[3] = {desc.centerX[0], desc.centerY[0],
                                desc.centerZ[0]};
            PomadeHit const hit = PomadeClosestPointCpu(*_scalp, p);
            if (!hit.hit) {
                _diagnostic = "PomadeModel::MatchTubeSurface: no surface point";
                return false;
            }
            desc.centerX[0] = hit.px;
            desc.centerY[0] = hit.py;
            desc.centerZ[0] = hit.pz;
            return true;
        });
}

bool
PomadeModel::SnapTubeRootToScalp(int tubeId)
{
    if (tubeId == 0) {
        return SnapRootToScalp();
    }
    return _EditChildTube(
        tubeId, /*layoutChanged*/ false, PomadeDirty_Points,
        [&](PomadeTubeDesc &desc) {
            if (!_scalp || !_scalp->finalized || desc.centerX.empty()) {
                _diagnostic = "PomadeModel::SnapTubeRootToScalp: need a bound "
                              "scalp + tube";
                return false;
            }
            float const p[3] = {desc.centerX[0], desc.centerY[0],
                                desc.centerZ[0]};
            PomadeHit const hit = PomadeClosestPointCpu(*_scalp, p);
            if (!hit.hit) {
                _diagnostic =
                    "PomadeModel::SnapTubeRootToScalp: no surface point";
                return false;
            }
            float const dx = hit.px - desc.centerX[0];
            float const dy = hit.py - desc.centerY[0];
            float const dz = hit.pz - desc.centerZ[0];
            for (size_t i = 0; i < desc.centerX.size(); ++i) {
                desc.centerX[i] += dx;
                desc.centerY[i] += dy;
                desc.centerZ[i] += dz;
            }
            return true;
        });
}

bool
PomadeModel::RelaxTubeCenter(int tubeId, float strength, int iterations)
{
    if (tubeId == 0) {
        return RelaxCenter(strength, iterations);
    }
    return _EditChildTube(
        tubeId, /*layoutChanged*/ false, PomadeDirty_Points,
        [&](PomadeTubeDesc &desc) {
            int const n = int(desc.centerX.size());
            if (!(strength >= 0.0f) || !(strength <= 1.0f) ||
                iterations < 1 || n < 3) {
                _diagnostic =
                    "PomadeModel::RelaxTubeCenter: strength in [0, 1], "
                    "iterations >= 1, need >= 3 CVs";
                return false;
            }
            float const lo = 0.5f * kPomadeSmoothnessSpike;
            float const span = kPomadeSmoothnessSpike - lo;
            for (int it = 0; it < iterations; ++it) {
                std::vector<float> scores(size_t(n), 0.0f);
                std::string derr;
                if (!_LaneSmoothnessScore(
                        HasCudaMirror(), desc.centerX.data(),
                        desc.centerY.data(), desc.centerZ.data(), n,
                        scores.data(), &derr)) {
                    _diagnostic = derr;
                    return false;
                }
                std::vector<float> nx = desc.centerX, ny = desc.centerY,
                                   nz = desc.centerZ;
                for (int i = 1; i + 1 < n; ++i) {
                    float w = (scores[size_t(i)] - lo) / span;
                    w = w < 0.0f ? 0.0f : (w > 1.0f ? 1.0f : w);
                    float const ax = 0.5f * (desc.centerX[size_t(i - 1)] +
                                             desc.centerX[size_t(i + 1)]);
                    float const ay = 0.5f * (desc.centerY[size_t(i - 1)] +
                                             desc.centerY[size_t(i + 1)]);
                    float const az = 0.5f * (desc.centerZ[size_t(i - 1)] +
                                             desc.centerZ[size_t(i + 1)]);
                    nx[size_t(i)] += (ax - nx[size_t(i)]) * strength * w;
                    ny[size_t(i)] += (ay - ny[size_t(i)]) * strength * w;
                    nz[size_t(i)] += (az - nz[size_t(i)]) * strength * w;
                }
                desc.centerX = std::move(nx);
                desc.centerY = std::move(ny);
                desc.centerZ = std::move(nz);
            }
            return true;
        });
}

bool
PomadeModel::MoveTubeSectionRing(int tubeId, int ring, float du, float dv)
{
    if (tubeId == 0) {
        return MoveSectionRing(ring, du, dv);
    }
    return _EditChildTube(
        tubeId, /*layoutChanged*/ false, PomadeDirty_Points,
        [&](PomadeTubeDesc &desc) {
            if (ring < 0 || ring >= int(desc.sections.size())) {
                _diagnostic =
                    "PomadeModel::MoveTubeSectionRing: ring out of range";
                return false;
            }
            PomadeTubeSection &s = desc.sections[size_t(ring)];
            for (size_t i = 0; i < s.u.size(); ++i) {
                s.u[i] += du;
                s.v[i] += dv;
            }
            return true;
        });
}

bool
PomadeModel::ScaleTubeSectionRing(int tubeId, int ring, float scale)
{
    if (tubeId == 0) {
        return ScaleSectionRing(ring, scale);
    }
    return _EditChildTube(
        tubeId, /*layoutChanged*/ false, PomadeDirty_Points,
        [&](PomadeTubeDesc &desc) {
            if (ring < 0 || ring >= int(desc.sections.size()) ||
                !(scale > 0.0f)) {
                _diagnostic =
                    "PomadeModel::ScaleTubeSectionRing: bad ring or scale";
                return false;
            }
            desc.sections[size_t(ring)].scale *= scale;
            return true;
        });
}

bool
PomadeModel::TwistTubeSectionRing(int tubeId, int ring, float radians)
{
    if (tubeId == 0) {
        return TwistSectionRing(ring, radians);
    }
    return _EditChildTube(
        tubeId, /*layoutChanged*/ false, PomadeDirty_Points,
        [&](PomadeTubeDesc &desc) {
            if (ring < 0 || ring >= int(desc.sections.size())) {
                _diagnostic =
                    "PomadeModel::TwistTubeSectionRing: ring out of range";
                return false;
            }
            desc.sections[size_t(ring)].twist += radians;
            return true;
        });
}

bool
PomadeModel::MoveTubeSectionCV(int tubeId, int ring, int slot, float du,
                              float dv)
{
    if (tubeId == 0) {
        return MoveSectionCV(ring, slot, du, dv);
    }
    return _EditChildTube(
        tubeId, /*layoutChanged*/ false, PomadeDirty_Points,
        [&](PomadeTubeDesc &desc) {
            if (ring < 0 || ring >= int(desc.sections.size()) || slot < 0 ||
                slot >= int(desc.sections[size_t(ring)].u.size())) {
                _diagnostic =
                    "PomadeModel::MoveTubeSectionCV: ring/slot out of range";
                return false;
            }
            PomadeTubeSection &s = desc.sections[size_t(ring)];
            s.u[size_t(slot)] += du;
            s.v[size_t(slot)] += dv;
            return true;
        });
}

bool
PomadeModel::AddTubeSectionRing(int tubeId, float t)
{
    if (tubeId == 0) {
        return AddSectionRing(t);
    }
    return _EditChildTube(
        tubeId, /*layoutChanged*/ true,
        PomadeDirty_Topology | PomadeDirty_Points,
        [&](PomadeTubeDesc &desc) {
            int const nSec = int(desc.sections.size());
            if (nSec < 2) {
                _diagnostic = "PomadeModel::AddTubeSectionRing: no tube";
                return false;
            }
            float const t0 = desc.sections.front().t;
            float const t1 = desc.sections.back().t;
            if (!(t > t0) || !(t < t1)) {
                _diagnostic = "PomadeModel::AddTubeSectionRing: t must sit "
                              "strictly inside the section range";
                return false;
            }
            int k = 0;
            while (k + 1 < nSec - 1 && desc.sections[size_t(k + 1)].t < t) {
                ++k;
            }
            PomadeTubeSection const &s0 = desc.sections[size_t(k)];
            PomadeTubeSection const &s1 = desc.sections[size_t(k + 1)];
            float const span = s1.t - s0.t;
            float const f = span > 0.0f ? (t - s0.t) / span : 0.0f;
            PomadeTubeSection ins;
            ins.t = t;
            ins.scale = s0.scale + (s1.scale - s0.scale) * f;
            ins.twist = s0.twist + (s1.twist - s0.twist) * f;
            ins.u.resize(s0.u.size());
            ins.v.resize(s0.v.size());
            for (size_t i = 0; i < s0.u.size(); ++i) {
                ins.u[i] = s0.u[i] + (s1.u[i] - s0.u[i]) * f;
                ins.v[i] = s0.v[i] + (s1.v[i] - s0.v[i]) * f;
            }
            desc.sections.insert(desc.sections.begin() + (k + 1),
                                 std::move(ins));
            return true;
        });
}

bool
PomadeModel::RemoveTubeSectionRing(int tubeId, int ring)
{
    if (tubeId == 0) {
        return RemoveSectionRing(ring);
    }
    return _EditChildTube(
        tubeId, /*layoutChanged*/ true,
        PomadeDirty_Topology | PomadeDirty_Points,
        [&](PomadeTubeDesc &desc) {
            if (ring < 0 || ring >= int(desc.sections.size()) ||
                desc.sections.size() <= 2) {
                _diagnostic =
                    "PomadeModel::RemoveTubeSectionRing: need >= 2 rings left";
                return false;
            }
            desc.sections.erase(desc.sections.begin() + ring);
            return true;
        });
}

bool
PomadeModel::CopyTubeSectionRing(int tubeId, int src, int dst)
{
    if (tubeId == 0) {
        return CopySectionRing(src, dst);
    }
    return _EditChildTube(
        tubeId, /*layoutChanged*/ false, PomadeDirty_Points,
        [&](PomadeTubeDesc &desc) {
            if (src < 0 || src >= int(desc.sections.size()) || dst < 0 ||
                dst >= int(desc.sections.size())) {
                _diagnostic =
                    "PomadeModel::CopyTubeSectionRing: ring out of range";
                return false;
            }
            float const t = desc.sections[size_t(dst)].t;
            desc.sections[size_t(dst)] = desc.sections[size_t(src)];
            desc.sections[size_t(dst)].t = t;
            return true;
        });
}

// -- V0: viewport publication views + display state (plan/18 §2.1, §2.2) -----

std::vector<PomadeModel::TubeView>
PomadeModel::SnapshotTubes() const
{
    std::vector<TubeView> out;
    out.reserve(_tubes.size() + 1);
    if (_host.centerX.size() >= 2) {
        TubeView view;
        if (_TubeDescLocked(0, &view.desc)) {
            out.push_back(std::move(view));
        }
    }
    for (auto const &kv : _tubes) {
        TubeView view;
        view.desc = kv.second.actual;
        _AttachTubeRootLocked(&view.desc);
        view.desc.tubeId = kv.first;
        view.imported = kv.second.imported;
        view.persistent = kv.second.persistent;
        view.transientParent = kv.second.transientParent;
        out.push_back(std::move(view));
    }
    return out;
}

int
PomadeModel::GetMaxTubeLevel() const
{
    int level = _host.centerX.size() >= 2 ? 1 : 0;
    for (auto const &kv : _tubes) {
        level = std::max(level, kv.second.actual.level);
    }
    return level;
}

bool
PomadeModel::SetLevelDisplay(int level, bool visible, bool xray)
{
    if (level < 1) {
        _diagnostic = "PomadeModel::SetLevelDisplay: level must be >= 1";
        return false;
    }
    LevelDisplay next;
    next.visible = visible;
    next.xray = xray;
    auto it = _levelDisplay.find(level);
    if (it != _levelDisplay.end()) {
        // The two-argument form owns visibility and x-ray only: a ladder
        // that dropped this level to centers-only keeps that until it
        // restores it (plan/18 §3.7), and the policy's x-ray strength and
        // centers flag survive a Levels-panel visibility toggle.
        next.centersOnly = it->second.centersOnly;
        next.xrayOpacity = it->second.xrayOpacity;
        next.centers = it->second.centers;
        next.centerCVDots = it->second.centerCVDots;
        next.ringCVDots = it->second.ringCVDots;
        next.guides = it->second.guides;
        if (it->second.visible == next.visible &&
            it->second.xray == next.xray) {
            return true;
        }
    } else if (next.visible && !next.xray) {
        return true;  // already the default
    }
    _levelDisplay[level] = next;
    _dirty |= PomadeDirty_Display;
    ++_version;
    return true;
}

bool
PomadeModel::SetLevelDrawMode(int level, bool visible, bool xray,
                             bool centersOnly)
{
    if (level < 1) {
        _diagnostic = "PomadeModel::SetLevelDrawMode: level must be >= 1";
        return false;
    }
    LevelDisplay next;
    next.visible = visible;
    next.xray = xray;
    next.centersOnly = centersOnly;
    auto it = _levelDisplay.find(level);
    if (it != _levelDisplay.end()) {
        next.xrayOpacity = it->second.xrayOpacity;
        next.centers = it->second.centers;
        next.centerCVDots = it->second.centerCVDots;
        next.ringCVDots = it->second.ringCVDots;
        next.guides = it->second.guides;
        if (it->second.visible == next.visible &&
            it->second.xray == next.xray &&
            it->second.centersOnly == next.centersOnly) {
            return true;
        }
    } else if (next.visible && !next.xray && !next.centersOnly) {
        return true;  // already the default
    }
    _levelDisplay[level] = next;
    _dirty |= PomadeDirty_Display;
    ++_version;
    return true;
}

bool
PomadeModel::SetLevelDraw(int level, LevelDisplay const &draw)
{
    if (level < 1) {
        _diagnostic = "PomadeModel::SetLevelDraw: level must be >= 1";
        return false;
    }
    LevelDisplay next = draw;
    next.xrayOpacity = std::max(0.0f, std::min(1.0f, next.xrayOpacity));
    auto it = _levelDisplay.find(level);
    if (it != _levelDisplay.end()) {
        // The policy owns the look, the ladder owns centers-only: a policy
        // push in the middle of a heavy drag must not restore the level the
        // ladder just dropped (plan/18 §3.7).
        next.centersOnly = it->second.centersOnly;
        if (it->second.visible == next.visible &&
            it->second.xray == next.xray &&
            it->second.xrayOpacity == next.xrayOpacity &&
            it->second.centers == next.centers &&
            it->second.centerCVDots == next.centerCVDots &&
            it->second.ringCVDots == next.ringCVDots &&
            it->second.guides == next.guides) {
            return true;
        }
    } else {
        LevelDisplay const dflt;
        if (next.visible == dflt.visible && next.xray == dflt.xray &&
            next.xrayOpacity == dflt.xrayOpacity &&
            next.centers == dflt.centers &&
            next.centerCVDots == dflt.centerCVDots &&
            next.ringCVDots == dflt.ringCVDots &&
            next.guides == dflt.guides &&
            !next.centersOnly) {
            return true;  // already the default
        }
    }
    _levelDisplay[level] = next;
    _dirty |= PomadeDirty_Display;
    ++_version;
    return true;
}

bool
PomadeModel::SetRingDisplay(int mode)
{
    if (mode < Rings_Off || mode > Rings_All) {
        _diagnostic = "PomadeModel::SetRingDisplay: mode is 0..2";
        return false;
    }
    if (_ringDisplay == mode) {
        return true;
    }
    _ringDisplay = mode;
    _dirty |= PomadeDirty_Display;
    ++_version;
    return true;
}

int
PomadeModel::GetRingDisplay() const
{
    return _ringDisplay;
}

bool
PomadeModel::SetAmplifiedHair(bool show)
{
    if (_amplifiedHair == show) {
        return true;
    }
    _amplifiedHair = show;
    _dirty |= PomadeDirty_Display;
    ++_version;
    return true;
}

bool
PomadeModel::GetAmplifiedHair() const
{
    return _amplifiedHair;
}

void
PomadeModel::ResolveHairDisplay(bool *tiles, bool *guides) const
{
    bool const gesture = _gestureDepth > 0;
    if (tiles) {
        *tiles = _amplifiedHair && !gesture;
    }
    if (guides) {
        // The preview steps aside only for tiles that are actually up.
        *guides = !(_amplifiedHair && !gesture);
    }
}

PomadeModel::LevelDisplay
PomadeModel::GetLevelDisplay(int level) const
{
    auto it = _levelDisplay.find(level);
    return it == _levelDisplay.end() ? LevelDisplay() : it->second;
}

bool
PomadeModel::SetFocusLevel(int level)
{
    if (level < 0) {
        _diagnostic = "PomadeModel::SetFocusLevel: level must be >= 0";
        return false;
    }
    if (level == _focusLevel) {
        return true;
    }
    _focusLevel = level;
    _dirty |= PomadeDirty_Display;
    ++_version;
    return true;
}

int
PomadeModel::GetFocusLevel() const
{
    return _focusLevel;
}

bool
PomadeModel::SetActiveCutEnabled(bool enabled)
{
    if (_activeCutEnabled == enabled) {
        return true;
    }
    _activeCutEnabled = enabled;
    // This changes the staged tube membership, not authored topology. Mark
    // display for policy consumers and topology so publishers rebuild their
    // per-level membership hashes and scene-index notices.
    _dirty |= PomadeDirty_Display | PomadeDirty_Topology;
    ++_version;
    return true;
}

bool
PomadeModel::GetActiveCutEnabled() const
{
    return _activeCutEnabled;
}

bool
PomadeModel::SetTubeExpanded(int tubeId, bool expanded)
{
    PomadeTubeDesc desc;
    if (!_TubeDescLocked(tubeId, &desc)) {
        _diagnostic = "PomadeModel::SetTubeExpanded: unknown tube";
        return false;
    }
    bool hasChild = false;
    for (auto const &entry : _tubes) {
        if (entry.second.actual.parentTubeId == tubeId) {
            hasChild = true;
            break;
        }
    }
    if (!hasChild) {
        _diagnostic = "PomadeModel::SetTubeExpanded: tube is a leaf";
        return false;
    }
    bool changed = false;
    if (expanded) {
        changed = _expandedTubeIds.insert(tubeId).second;
    } else {
        for (auto it = _expandedTubeIds.begin(); it != _expandedTubeIds.end();) {
            if (*it == tubeId || _IsDescendantOfLocked(*it, tubeId)) {
                it = _expandedTubeIds.erase(it);
                changed = true;
            } else {
                ++it;
            }
        }
    }
    if (changed) {
        _dirty |= PomadeDirty_Display | PomadeDirty_Topology;
        ++_version;
    }
    return true;
}

bool
PomadeModel::GetTubeExpanded(int tubeId) const
{
    return _expandedTubeIds.count(tubeId) != 0;
}

bool
PomadeModel::IsTubeVisibleInActiveCut(int tubeId) const
{
    return _IsTubeVisibleInActiveCutLocked(tubeId);
}

bool
PomadeModel::SetDisplayScale(float worldPerPixel)
{
    if (!(worldPerPixel >= 0.0f) || !(worldPerPixel < 1e30f)) {
        _diagnostic =
            "PomadeModel::SetDisplayScale: want a finite value >= 0";
        return false;
    }
    if (worldPerPixel == _displayScale) {
        return true;
    }
    _displayScale = worldPerPixel;
    _dirty |= PomadeDirty_Display;
    ++_version;
    return true;
}

float
PomadeModel::GetDisplayScale() const
{
    return _displayScale;
}

bool
PomadeModel::SetGroomPath(std::string const &path)
{
    if (_groomPath == path) {
        return true;
    }
    _groomPath = path;
    _dirty |= PomadeDirty_Display;
    ++_version;
    return true;
}

std::string
PomadeModel::GetGroomPath() const
{
    return _groomPath;
}

// -- V0 clump palette (plan/18 §2.4a) ----------------------------------------

namespace {

// The plan's 16 sRGB hex entries, in order.
uint32_t const kPomadePaletteSrgb[16] = {
    0x2F6BFFu, 0xFFD400u, 0xE0248Fu, 0x4CD62Bu,
    0x8A3FFFu, 0xFF7A1Au, 0x00C8D6u, 0xFF3B3Bu,
    0xA6E22Eu, 0xF062F0u, 0x1FA3FFu, 0xFFB000u,
    0x17C77Au, 0xC43CFFu, 0xFF5E9Au, 0x7BD3FFu,
};

float _SrgbToLinear(float c)
{
    return c <= 0.04045f ? c / 12.92f
                         : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

}  // namespace

int
PomadeClumpPaletteSize()
{
    return 16;
}

PomadeRgb
PomadeClumpPaletteEntry(int slot)
{
    int const n = PomadeClumpPaletteSize();
    int const i = ((slot % n) + n) % n;
    uint32_t const hex = kPomadePaletteSrgb[i];
    PomadeRgb out;
    out.r = _SrgbToLinear(float((hex >> 16) & 0xFFu) / 255.0f);
    out.g = _SrgbToLinear(float((hex >> 8) & 0xFFu) / 255.0f);
    out.b = _SrgbToLinear(float(hex & 0xFFu) / 255.0f);
    return out;
}

PomadeRgb
PomadeClumpColor(int regionId, int level, int childIndex)
{
    PomadeRgb base;
    if (regionId >= 0) {
        // Use the graph's persisted region colour for both the scalp patch
        // and its tubes. A separate tube palette changes the colour family
        // and repeats it after 16 regions. Hydra consumes linear RGB.
        float rgb[3];
        PomadeRegionColor(regionId, rgb);
        base = {_SrgbToLinear(rgb[0]), _SrgbToLinear(rgb[1]),
                _SrgbToLinear(rgb[2])};
    } else {
        base = PomadeClumpPaletteEntry(regionId);
    }
    if (level <= 1 || childIndex < 0) {
        return base;
    }
    // Alternating lighten/darken, one step per sibling PAIR.
    //
    // Both operations keep the hue exactly. Lightening is a lerp towards
    // white, so (g-b) and (max-min) are scaled by the same (1-amount) and
    // their ratio — the hue — is unchanged; darkening scales all three
    // channels, which changes nothing but value. That is what "one lock,
    // one hue family" means and what the T0 test checks.
    //
    // The step sizes are the V9 correction. V8 used 0.12/(level-1), which
    // is ±12 % of the remaining headroom at L2 and ±6 % at L3 — below the
    // spread the shader's own key/fill/ambient rig puts across a single
    // curved tube, so siblings were measurably different and visually
    // identical (the 2026-09-18 braid frame). 0.22 with a 0.10 widening
    // per further sibling pair, shrinking by 0.8 per level below 2, keeps
    // an L4 sibling at ±14 % and still separates neighbours on screen.
    int const magnitude = childIndex / 2 + 1;
    float const direction = (childIndex % 2 == 0) ? 1.0f : -1.0f;
    float step = 0.22f + 0.10f * float(magnitude - 1);
    for (int l = 2; l < level; ++l) {
        step *= 0.8f;
    }
    float amount = direction * step;
    amount = std::max(-0.6f, std::min(0.6f, amount));
    float const *src[3] = {&base.r, &base.g, &base.b};
    PomadeRgb out;
    float *dst[3] = {&out.r, &out.g, &out.b};
    for (int c = 0; c < 3; ++c) {
        float const v = *src[c];
        *dst[c] = amount >= 0.0f ? v + (1.0f - v) * amount
                                 : v * (1.0f + amount);
    }
    return out;
}

// -- V9 display policy (plan/18 section 2.4a) --------------------------------
//
// The table itself lives in the header comment; this is the only place it
// is evaluated. Pure functions, no lock, no model: Pomade_SetDisplayPolicy
// walks the levels and hands each answer to PomadeModel::SetLevelDraw.

int
PomadeDisplayModeFromNames(char const *mode, char const *subMode)
{
    if (!mode) {
        return -1;
    }
    std::string const m(mode);
    std::string const s(subMode ? subMode : "");
    if (m == "graph") {
        return PomadeDisplayMode_Graph;
    }
    if (m == "tube") {
        if (s == "tube") {
            return PomadeDisplayMode_TubeObject;
        }
        // Ring and Section both edit the cross-sections, and both want the
        // rings readable against an opaque surface.
        return (s == "ring" || s == "section") ? PomadeDisplayMode_TubeRing
                                               : PomadeDisplayMode_TubeCenter;
    }
    if (m == "fill") {
        return PomadeDisplayMode_Fill;
    }
    if (m == "hierarchy") {
        return PomadeDisplayMode_Hierarchy;
    }
    if (m == "sculpt") {
        return PomadeDisplayMode_Sculpt;
    }
    if (m == "output") {
        return PomadeDisplayMode_Output;
    }
    return -1;
}

PomadeModel::LevelDisplay
PomadePolicyLevelDisplay(int displayMode, int level, int focusLevel)
{
    PomadeModel::LevelDisplay out;
    out.visible = true;
    out.xray = false;
    out.xrayOpacity = PomadeModel::kDefaultXrayOpacity;
    out.centers = true;
    // The interactive guide preview belongs to Fill, where it is being
    // authored. Output renders the committed amplified tiles themselves;
    // its authoring tubes, cage and preview guides must not occlude that
    // interior hair.
    out.guides = displayMode == PomadeDisplayMode_Fill;
    // "Nothing focused" must not mean "everything is a ghost": with no
    // focus every level reads as the focused one.
    bool const focused = focusLevel <= 0 || level == focusLevel;
    switch (displayMode) {
    case PomadeDisplayMode_Graph:
        // Graph mode is the scalp-patch and boundary editing view. Hiding
        // tube/guides leaves the distinct subface patches readable.
        out.visible = false;
        out.centers = false;
        out.guides = false;
        return out;
    case PomadeDisplayMode_Output:
        // Output is a render/result view.  `visible` gates the Pomade tube
        // and guide helpers only; it does not hide the committed amplified
        // tiles supplied by the stage.  Switching back to another policy
        // restores that mode's normal per-level base visibility.
        out.visible = false;
        out.centers = false;
        out.guides = false;
        return out;
    case PomadeDisplayMode_TubeObject:
        out.centers = false;
        out.centerCVDots = false;
        out.ringCVDots = false;
        out.guides = false;
        // Object selection keeps its focused child legible through an
        // enclosing ancestor, like Tube/Ring does for section controls.
        if (!focused) {
            out.xray = true;
            out.xrayOpacity = PomadeModel::kDefaultXrayOpacity;
        }
        return out;
    case PomadeDisplayMode_TubeRing:
        out.centerCVDots = false;
        out.ringCVDots = true;
        if (!focused) {
            out.xray = true;
            out.xrayOpacity = PomadeModel::kDefaultXrayOpacity;
        }
        return out;
    case PomadeDisplayMode_Fill:
        out.xray = true;
        out.xrayOpacity = PomadeModel::kDefaultXrayOpacity;
        return out;
    case PomadeDisplayMode_TubeCenter:
        out.centerCVDots = true;
        out.ringCVDots = false;
    case PomadeDisplayMode_Hierarchy:
    case PomadeDisplayMode_Sculpt:
        out.xray = true;
        out.xrayOpacity = focused ? PomadeModel::kDefaultXrayOpacity
                                  : PomadeModel::kFaintXrayOpacity;
        return out;
    default:
        return out;
    }
}

int
PomadePolicyRingDisplay(int displayMode)
{
    switch (displayMode) {
    case PomadeDisplayMode_Graph:
    case PomadeDisplayMode_Output:
    case PomadeDisplayMode_TubeObject:
        return PomadeModel::Rings_Off;
    case PomadeDisplayMode_TubeRing:
        return PomadeModel::Rings_All;
    default:
        return PomadeModel::Rings_Selected;
    }
}

} // namespace usdGenPomade
