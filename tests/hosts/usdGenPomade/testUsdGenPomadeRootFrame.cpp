// Region-rooted tubes keep their first section on the support plane even
// when an artist bends the first shaft CV.  This is deliberately an end to
// end model test: the host tessellation is the geometry that reaches imaging,
// rather than just an assertion about the descriptor's pinned-frame fields.

#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadeCommit.h"
#include "usdGenPomade/pomadeScalp.h"
#include "usdGenPomade/pomadeTube.h"

#include <algorithm>
#include <cstdint>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, char const *what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    } else {
        std::printf("ok:   %s\n", what);
    }
}

float Dot(float const a[3], float const b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

float Length(float const v[3])
{
    return std::sqrt(Dot(v, v));
}

void Normalize(float v[3])
{
    float const length = Length(v);
    if (length > 0.0f) {
        v[0] /= length;
        v[1] /= length;
        v[2] /= length;
    }
}

float FrameAlignment(usdGenPomade::PomadeFrame const &frame,
                     float const direction[3])
{
    float const tangent[3] = {frame.tx, frame.ty, frame.tz};
    return Dot(tangent, direction);
}

bool SameFrame(usdGenPomade::PomadeFrame const &a,
               usdGenPomade::PomadeFrame const &b, float tolerance = 1e-5f)
{
    float const av[9] = {a.tx, a.ty, a.tz, a.nx, a.ny, a.nz, a.bx, a.by,
                         a.bz};
    float const bv[9] = {b.tx, b.ty, b.tz, b.nx, b.ny, b.nz, b.bx, b.by,
                         b.bz};
    for (int i = 0; i < 9; ++i) {
        if (std::fabs(av[i] - bv[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

bool RootRingPointsOnPlane(std::vector<float> const &positions, int ringVerts,
                           float const center[3], float const normal[3],
                           float tolerance = 2e-4f)
{
    if (ringVerts < 3 || positions.size() < size_t(ringVerts) * 3) {
        return false;
    }
    for (int slot = 0; slot < ringVerts; ++slot) {
        float const *p = positions.data() + size_t(slot) * 3;
        float const offset[3] = {p[0] - center[0], p[1] - center[1],
                                 p[2] - center[2]};
        if (std::fabs(Dot(offset, normal)) > tolerance) {
            return false;
        }
    }
    return true;
}

bool RootRingOnPlane(usdGenPomade::PomadeModel::HostTubeMesh const &mesh,
                     int ringVerts, float const center[3],
                     float const normal[3], float tolerance = 2e-4f)
{
    return RootRingPointsOnPlane(mesh.positions, ringVerts, center, normal,
                                 tolerance);
}

usdGenPomade::PomadeHit Hit(usdGenPomade::PomadeScalpMesh const &mesh, float u,
                          float v, float const normal[3])
{
    using namespace usdGenPomade;
    PomadeHit hit;
    hit.hit = PomadeFacePosition(mesh, 0, u, v, &hit.px, &hit.py, &hit.pz);
    hit.faceId = hit.hit ? 0 : -1;
    hit.ptexFaceId = hit.faceId;
    hit.u = u;
    hit.v = v;
    hit.nx = normal[0];
    hit.ny = normal[1];
    hit.nz = normal[2];
    return hit;
}

usdGenPomade::PomadeHit HitFace(usdGenPomade::PomadeScalpMesh const &mesh,
                              int faceId, float u, float v,
                              float const normal[3])
{
    using namespace usdGenPomade;
    PomadeHit hit;
    hit.hit = PomadeFacePosition(mesh, faceId, u, v, &hit.px, &hit.py,
                                &hit.pz);
    hit.faceId = hit.hit ? faceId : -1;
    hit.ptexFaceId = hit.faceId;
    hit.u = u;
    hit.v = v;
    hit.nx = normal[0];
    hit.ny = normal[1];
    hit.nz = normal[2];
    return hit;
}

bool AddTiltedRegion(usdGenPomade::PomadeModel *model, float const normal[3])
{
    using namespace usdGenPomade;
    std::shared_ptr<PomadeScalpMesh const> const scalp = model->GetScalp();
    if (!scalp) {
        return false;
    }
    // A compact square in the interior of the tilted quad.  Its centroid is
    // (0.5, 0.5, 0.5), and the support plane has normal (0,-1,1)/sqrt(2).
    float const uv[8] = {0.20f, 0.20f, 0.80f, 0.20f,
                          0.80f, 0.80f, 0.20f, 0.80f};
    std::vector<int> nodes;
    for (int i = 0; i < 4; ++i) {
        int const node = model->GraphAddNode(
            Hit(*scalp, uv[i * 2], uv[i * 2 + 1], normal));
        if (node < 0) {
            return false;
        }
        nodes.push_back(node);
    }
    for (int i = 0; i < 4; ++i) {
        if (model->GraphConnect(nodes[i], nodes[(i + 1) % 4]) < 0) {
            return false;
        }
    }
    return model->Rasterise() && model->GetGraph().RegionCount() == 1;
}

// A child has an authored shape of its own after subdivision.  K7 may update
// its parent control cage, but that aggregation must never feed back into the
// other leaf children during the same leaf edit.  Capture the descriptor and
// the actual K5 output, rather than only center CVs, so this catches a sibling
// moving through section frames or ring interpolation too.
struct TubeGeometry {
    usdGenPomade::PomadeTubeDesc desc;
    std::vector<float> positions;
};

bool SameRigidTranslation(TubeGeometry const &before,
                          TubeGeometry const &after, float tolerance = 2e-4f)
{
    if (before.positions.size() != after.positions.size() ||
        before.positions.size() < 3) {
        return false;
    }
    float const delta[3] = {after.positions[0] - before.positions[0],
                            after.positions[1] - before.positions[1],
                            after.positions[2] - before.positions[2]};
    for (size_t i = 0; i < before.positions.size(); ++i) {
        int const axis = int(i % 3);
        if (std::fabs((after.positions[i] - before.positions[i]) -
                      delta[axis]) > tolerance) {
            return false;
        }
    }
    return true;
}

bool SameRigidFrameTransform(TubeGeometry const &before,
                             TubeGeometry const &after,
                             usdGenPomade::PomadeFrame const &oldFrame,
                             usdGenPomade::PomadeFrame const &newFrame,
                             float const oldPivot[3], float const newPivot[3],
                             size_t firstPoint = 0,
                             float tolerance = 3e-4f,
                             float *outMaxError = nullptr)
{
    if (before.positions.size() != after.positions.size() ||
        !oldPivot || !newPivot) {
        return false;
    }
    auto axis = [](usdGenPomade::PomadeFrame const &f, int i) {
        float const values[9] = {f.tx, f.ty, f.tz, f.nx, f.ny, f.nz,
                                 f.bx, f.by, f.bz};
        return std::array<float, 3>{{values[i * 3], values[i * 3 + 1],
                                     values[i * 3 + 2]}};
    };
    size_t const pointCount = before.positions.size() / 3;
    if (firstPoint >= pointCount) {
        return false;
    }
    float maxError = 0.0f;
    for (size_t point = firstPoint; point < pointCount; ++point) {
        float const *p = before.positions.data() + point * 3;
        float const offset[3] = {p[0] - oldPivot[0], p[1] - oldPivot[1],
                                 p[2] - oldPivot[2]};
        float expected[3] = {newPivot[0], newPivot[1], newPivot[2]};
        for (int i = 0; i < 3; ++i) {
            auto const a = axis(oldFrame, i);
            auto const b = axis(newFrame, i);
            float const weight = offset[0] * a[0] + offset[1] * a[1] +
                                 offset[2] * a[2];
            for (int k = 0; k < 3; ++k) {
                expected[k] += weight * b[k];
            }
        }
        float const *got = after.positions.data() + point * 3;
        for (int k = 0; k < 3; ++k) {
            float const error = std::fabs(got[k] - expected[k]);
            if (error > maxError) maxError = error;
        }
    }
    if (outMaxError) *outMaxError = maxError;
    float const cross[3] = {newFrame.ty * newFrame.nz - newFrame.tz * newFrame.ny,
                            newFrame.tz * newFrame.nx - newFrame.tx * newFrame.nz,
                            newFrame.tx * newFrame.ny - newFrame.ty * newFrame.nx};
    float const binormal[3] = {newFrame.bx, newFrame.by, newFrame.bz};
    return maxError <= tolerance && Dot(cross, binormal) > 0.99f;
}

bool SameRigidRootFrameTransform(TubeGeometry const &before,
                                 TubeGeometry const &after,
                                 size_t firstPoint = 0,
                                 float tolerance = 3e-4f)
{
    std::vector<usdGenPomade::PomadeFrame> oldFrames, newFrames;
    std::string error;
    if (before.desc.centerX.empty() || after.desc.centerX.empty() ||
        !usdGenPomade::PomadeTubeFramesCpu(before.desc, &oldFrames, &error) ||
        !usdGenPomade::PomadeTubeFramesCpu(after.desc, &newFrames, &error) ||
        oldFrames.empty() || newFrames.empty()) {
        return false;
    }
    float const oldPivot[3] = {before.desc.centerX[0], before.desc.centerY[0],
                               before.desc.centerZ[0]};
    float const newPivot[3] = {after.desc.centerX[0], after.desc.centerY[0],
                               after.desc.centerZ[0]};
    return SameRigidFrameTransform(before, after, oldFrames.front(),
                                   newFrames.front(), oldPivot, newPivot,
                                   firstPoint, tolerance);
}

bool ProperFrames(usdGenPomade::PomadeTubeDesc const &desc)
{
    std::vector<usdGenPomade::PomadeFrame> frames;
    std::string error;
    if (!usdGenPomade::PomadeTubeFramesCpu(desc, &frames, &error) ||
        frames.empty()) {
        return false;
    }
    for (usdGenPomade::PomadeFrame const &frame : frames) {
        float const cross[3] = {
            frame.ty * frame.nz - frame.tz * frame.ny,
            frame.tz * frame.nx - frame.tx * frame.nz,
            frame.tx * frame.ny - frame.ty * frame.nx};
        float const binormal[3] = {frame.bx, frame.by, frame.bz};
        if (Dot(cross, binormal) < 0.99f) {
            return false;
        }
    }
    return true;
}

bool CaptureTubeGeometry(usdGenPomade::PomadeModel const &model, int tubeId,
                         TubeGeometry *out)
{
    if (!out || !model.GetTubeDesc(tubeId, &out->desc)) {
        return false;
    }
    // K6/K7 and rigid-transport expectations below concern the authored
    // chart. Surface attachment is a derived display correction, tested
    // independently against the scalp by CheckCurvedRootAttachment.
    out->desc.rootSurfaceOffsets.clear();
    std::vector<usdGenPomade::PomadeFrame> frames;
    std::vector<float> normals, ringT;
    std::string error;
    return usdGenPomade::PomadeTubeFramesCpu(out->desc, &frames, &error) &&
           // Four longitudinal samples per authored interval includes the
           // vulnerable near-root K5 spans, not merely center/section knots.
           usdGenPomade::PomadeTessellateCpu(out->desc, frames, 4,
                                           &out->positions, &normals, &ringT,
                                           &error);
}

bool SameRootRingWorld(usdGenPomade::PomadeTubeDesc const &actual,
                       usdGenPomade::PomadeTubeDesc const &derived,
                       float tolerance = 3e-4f)
{
    if (actual.ringVerts != derived.ringVerts || actual.sections.empty() ||
        derived.sections.empty()) {
        return false;
    }
    std::vector<usdGenPomade::PomadeFrame> actualFrames, derivedFrames;
    std::vector<float> actualRing, derivedRing;
    std::string error;
    return usdGenPomade::PomadeTubeFramesCpu(actual, &actualFrames, &error) &&
           usdGenPomade::PomadeTubeFramesCpu(derived, &derivedFrames, &error) &&
           usdGenPomade::PomadeSampleTubeRingCpu(
               actual, actualFrames, actual.sections.front().t, &actualRing,
               &error) &&
           usdGenPomade::PomadeSampleTubeRingCpu(
               derived, derivedFrames, derived.sections.front().t, &derivedRing,
               &error) &&
           actualRing.size() == derivedRing.size() &&
           std::equal(actualRing.begin(), actualRing.end(), derivedRing.begin(),
                      [tolerance](float a, float b) {
                          return std::fabs(a - b) <= tolerance;
                      });
}

bool CaptureDescGeometry(usdGenPomade::PomadeTubeDesc const &desc,
                         TubeGeometry *out)
{
    if (!out) {
        return false;
    }
    out->desc = desc;
    out->desc.rootSurfaceOffsets.clear();
    std::vector<usdGenPomade::PomadeFrame> frames;
    std::vector<float> normals, ringT;
    std::string error;
    return usdGenPomade::PomadeTubeFramesCpu(out->desc, &frames, &error) &&
           usdGenPomade::PomadeTessellateCpu(out->desc, frames, 4,
                                           &out->positions, &normals, &ringT,
                                           &error);
}

// K14 is free to choose a different vertex count for a newly derived child.
// The model keeps the child's authored layout, so reference geometry must be
// matched before the test predicts K6/K7 residual application.
bool MatchDerivedLayout(usdGenPomade::PomadeTubeDesc const &derived,
                        usdGenPomade::PomadeTubeDesc const &actual,
                        usdGenPomade::PomadeTubeDesc *out,
                        std::string *error)
{
    if (!out) {
        return false;
    }
    bool same = derived.ringVerts == actual.ringVerts &&
                derived.sections.size() == actual.sections.size();
    for (size_t i = 0; same && i < derived.sections.size(); ++i) {
        same = derived.sections[i].t == actual.sections[i].t;
    }
    if (same) {
        *out = derived;
        return true;
    }
    return usdGenPomade::PomadeResampleDescRingsCpu(
        derived, actual.ringVerts, out, error);
}

bool SameTubeGeometry(TubeGeometry const &a, TubeGeometry const &b)
{
    if (a.desc.sections.size() != b.desc.sections.size()) {
        return false;
    }
    for (size_t i = 0; i < a.desc.sections.size(); ++i) {
        usdGenPomade::PomadeTubeSection const &as = a.desc.sections[i];
        usdGenPomade::PomadeTubeSection const &bs = b.desc.sections[i];
        if (as.t != bs.t || as.u != bs.u || as.v != bs.v ||
            as.scale != bs.scale || as.twist != bs.twist) {
            return false;
        }
    }
    return a.desc.centerX == b.desc.centerX &&
           a.desc.centerY == b.desc.centerY &&
           a.desc.centerZ == b.desc.centerZ &&
           a.positions == b.positions;
}

bool SameDeltas(usdGenPomade::PomadeShapeDeltas const &a,
                usdGenPomade::PomadeShapeDeltas const &b)
{
    if (a.centerDu != b.centerDu || a.centerDv != b.centerDv ||
        a.centerDw != b.centerDw || a.sections.size() != b.sections.size()) {
        return false;
    }
    for (size_t i = 0; i < a.sections.size(); ++i) {
        usdGenPomade::PomadeTubeSection const &as = a.sections[i];
        usdGenPomade::PomadeTubeSection const &bs = b.sections[i];
        if (as.t != bs.t || as.u != bs.u || as.v != bs.v ||
            as.scale != bs.scale || as.twist != bs.twist) {
            return false;
        }
    }
    return true;
}

bool HasNonRootSculpt(usdGenPomade::PomadeModel::TubeRecord const &record)
{
    int const n = int(record.actual.centerX.size());
    if (n < 2 || record.derived.centerX.size() != size_t(n) ||
        record.actual.centerY.size() != size_t(n) ||
        record.actual.centerZ.size() != size_t(n) ||
        record.derived.centerY.size() != size_t(n) ||
        record.derived.centerZ.size() != size_t(n)) {
        return false;
    }
    for (int cv = 1; cv < n; ++cv) {
        if (!std::isfinite(record.actual.centerX[size_t(cv)]) ||
            !std::isfinite(record.actual.centerY[size_t(cv)]) ||
            !std::isfinite(record.actual.centerZ[size_t(cv)])) {
            return false;
        }
        if (std::fabs(record.actual.centerX[size_t(cv)] -
                      record.derived.centerX[size_t(cv)]) > 1e-5f ||
            std::fabs(record.actual.centerY[size_t(cv)] -
                      record.derived.centerY[size_t(cv)]) > 1e-5f ||
            std::fabs(record.actual.centerZ[size_t(cv)] -
                      record.derived.centerZ[size_t(cv)]) > 1e-5f) {
            return true;
        }
    }
    for (size_t ring = 1; ring < record.actual.sections.size() &&
                           ring < record.derived.sections.size(); ++ring) {
        usdGenPomade::PomadeTubeSection const &actual =
            record.actual.sections[ring];
        usdGenPomade::PomadeTubeSection const &derived =
            record.derived.sections[ring];
        if (!std::isfinite(actual.scale) || !std::isfinite(actual.twist) ||
            actual.u.size() != actual.v.size() ||
            derived.u.size() != derived.v.size()) {
            return false;
        }
        if (actual.scale != derived.scale || actual.twist != derived.twist ||
            actual.u != derived.u || actual.v != derived.v) {
            return true;
        }
    }
    return false;
}

bool SameEdgeSplit(usdGenPomade::PomadeModel::SubdivideParams const &params,
                   int count, int seed, float a, float b, float c)
{
    return params.count == count && params.seed == seed &&
           params.splitMode == "edge" &&
           std::fabs(params.edgeA - a) < 1e-6f &&
           std::fabs(params.edgeB - b) < 1e-6f &&
           std::fabs(params.edgeC - c) < 1e-6f;
}

bool BuildRegionLoop(usdGenPomade::PomadeModel *model,
                     std::vector<std::array<float, 2>> const &uv,
                     float const normal[3], int *outRegion)
{
    using namespace usdGenPomade;
    if (!model || !outRegion || uv.size() < 3 || !model->GetScalp()) {
        return false;
    }
    std::vector<int> nodes;
    for (std::array<float, 2> const &point : uv) {
        int const id = model->GraphAddNode(
            HitFace(*model->GetScalp(), 0, point[0], point[1], normal));
        if (id < 0) {
            return false;
        }
        nodes.push_back(id);
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (model->GraphConnect(nodes[i], nodes[(i + 1) % nodes.size()]) < 0) {
            return false;
        }
    }
    if (!model->Rasterise() || model->GetGraph().RegionCount() != 1) {
        return false;
    }
    *outRegion = model->GetGraph().Regions().front().id;
    return *outRegion >= 0;
}

bool RootSectionWorld(usdGenPomade::PomadeTubeDesc const &desc,
                      std::vector<std::array<float, 3>> *out,
                      usdGenPomade::PomadeFrame *outFrame = nullptr)
{
    using namespace usdGenPomade;
    if (!out || desc.sections.empty() || desc.centerX.empty()) {
        return false;
    }
    std::vector<PomadeFrame> frames;
    std::string error;
    if (!PomadeTubeFramesCpu(desc, &frames, &error) || frames.empty()) {
        return false;
    }
    PomadeTubeSection const &section = desc.sections.front();
    if (section.u.size() != size_t(desc.ringVerts) ||
        section.v.size() != size_t(desc.ringVerts)) {
        return false;
    }
    PomadeFrame const &frame = frames.front();
    float const c = std::cos(section.twist);
    float const s = std::sin(section.twist);
    out->clear();
    out->reserve(size_t(desc.ringVerts));
    for (int slot = 0; slot < desc.ringVerts; ++slot) {
        float const u = section.scale * section.u[size_t(slot)];
        float const v = section.scale * section.v[size_t(slot)];
        float const ru = u * c - v * s;
        float const rv = u * s + v * c;
        out->push_back({{desc.centerX[0] + frame.nx * ru + frame.bx * rv,
                         desc.centerY[0] + frame.ny * ru + frame.by * rv,
                         desc.centerZ[0] + frame.nz * ru + frame.bz * rv}});
    }
    if (outFrame) {
        *outFrame = frame;
    }
    return true;
}

std::array<float, 3> ProjectToRootPlane(usdGenPomade::PomadeFrame const &frame,
                                        usdGenPomade::PomadeTubeDesc const &desc,
                                        float const point[3])
{
    float const offset[3] = {point[0] - desc.centerX[0],
                             point[1] - desc.centerY[0],
                             point[2] - desc.centerZ[0]};
    float const normal[3] = {frame.tx, frame.ty, frame.tz};
    float const depth = Dot(offset, normal);
    return {{point[0] - normal[0] * depth, point[1] - normal[1] * depth,
             point[2] - normal[2] * depth}};
}

bool SamePoint(std::array<float, 3> const &a, std::array<float, 3> const &b,
               float tolerance = 3e-4f)
{
    for (int axis = 0; axis < 3; ++axis) {
        if (std::fabs(a[size_t(axis)] - b[size_t(axis)]) > tolerance) {
            return false;
        }
    }
    return true;
}

bool PointOnSegment(std::array<float, 3> const &point,
                    std::array<float, 3> const &a,
                    std::array<float, 3> const &b,
                    float tolerance = 3e-4f)
{
    float const edge[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    float const fromA[3] = {point[0] - a[0], point[1] - a[1],
                            point[2] - a[2]};
    float const edge2 = Dot(edge, edge);
    if (!(edge2 > 1e-12f)) {
        return false;
    }
    float const t = Dot(fromA, edge) / edge2;
    if (t < -tolerance || t > 1.0f + tolerance) {
        return false;
    }
    std::array<float, 3> const nearest = {{a[0] + edge[0] * t,
                                             a[1] + edge[1] * t,
                                             a[2] + edge[2] * t}};
    return SamePoint(point, nearest, tolerance);
}

bool RootMatchesRegionLoop(usdGenPomade::PomadeModel const &model, int regionId,
                           usdGenPomade::PomadeTubeDesc const &desc,
                           bool requireEverySlotOnBoundary,
                           bool *outColumnsNormal = nullptr)
{
    using namespace usdGenPomade;
    if (regionId < 0 || regionId >= model.GetGraph().RegionCount()) {
        return false;
    }
    PomadeGraphRegion const &region =
        model.GetGraph().Regions()[size_t(regionId)];
    std::vector<std::array<float, 3>> actual;
    PomadeFrame frame;
    if (desc.ringVerts < 3 || !RootSectionWorld(desc, &actual, &frame)) {
        return false;
    }
    std::vector<std::array<float, 3>> projected;
    for (int nodeId : region.loop) {
        PomadeGraphNode const *node = model.GetGraph().FindNode(nodeId);
        if (!node) {
            return false;
        }
        projected.push_back(ProjectToRootPlane(frame, desc, node->p));
    }
    if (projected.size() < 3 || actual.size() < projected.size()) {
        return false;
    }
    // Auto (and an explicit request below the corner count) has no inserted
    // edge samples. The canonical seam is the smallest stable node id, so
    // test the ordered correspondence too, rather than allowing a radial
    // ring to pass by containing the same points in a scrambled order.
    if (actual.size() == projected.size()) {
        size_t seam = 0;
        for (size_t index = 1; index < region.loop.size(); ++index) {
            if (region.loop[index] < region.loop[seam]) {
                seam = index;
            }
        }
        std::vector<std::array<float, 3>> expected;
        expected.reserve(projected.size());
        for (size_t slot = 0; slot < projected.size(); ++slot) {
            expected.push_back(projected[(seam + slot) % projected.size()]);
        }
        // The native fitting orients a loop to its Newell support normal.
        // Graph extraction can spell that orientation forward or backwards,
        // but the stable-id seam is fixed. Accept precisely those two cyclic
        // sequences, never a scrambled set of radial samples.
        std::vector<std::array<float, 3>> reverse = expected;
        std::reverse(reverse.begin() + 1, reverse.end());
        bool forward = true;
        bool backward = true;
        for (size_t slot = 0; slot < actual.size(); ++slot) {
            forward = forward && SamePoint(actual[slot], expected[slot]);
            backward = backward && SamePoint(actual[slot], reverse[slot]);
        }
        if (!forward && !backward) {
            return false;
        }
    }
    // Auto rings must use exactly the loop's corners. Explicit rings retain
    // all corners, though their slots may contain extra points on edges.
    std::vector<bool> used(actual.size(), false);
    for (std::array<float, 3> const &corner : projected) {
        bool found = false;
        for (size_t slot = 0; slot < actual.size(); ++slot) {
            if (!used[slot] && SamePoint(actual[slot], corner)) {
                used[slot] = true;
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    if (requireEverySlotOnBoundary) {
        for (std::array<float, 3> const &slot : actual) {
            bool onBoundary = false;
            for (size_t corner = 0; corner < projected.size(); ++corner) {
                if (PointOnSegment(slot, projected[corner],
                                   projected[(corner + 1) % projected.size()])) {
                    onBoundary = true;
                    break;
                }
            }
            if (!onBoundary) {
                return false;
            }
        }
    }
    if (outColumnsNormal) {
        *outColumnsNormal = false;
        std::vector<PomadeFrame> frames;
        std::vector<float> positions, normals, ringT;
        std::string error;
        // Refine every authored interval. Checking only section knots lets a
        // rolled upper K4 frame pass even though the first interpolated K5
        // span shears the fitted footprint.
        int const refine = 4;
        int const tessellatedRings = PomadeTessellatedRingCount(desc, refine);
        if (!PomadeTubeFramesCpu(desc, &frames, &error) ||
            !PomadeTessellateCpu(desc, frames, refine, &positions, &normals,
                                &ringT, &error) ||
            positions.size() != size_t(tessellatedRings) *
                                    size_t(desc.ringVerts) * 3) {
            return false;
        }
        float const normal[3] = {frame.tx, frame.ty, frame.tz};
        bool columns = true;
        for (int ring = 1; ring < tessellatedRings && columns; ++ring) {
            float reference[3] = {0.0f, 0.0f, 0.0f};
            for (int slot = 0; slot < desc.ringVerts; ++slot) {
                size_t const base = size_t(slot) * 3;
                size_t const at = (size_t(ring) * size_t(desc.ringVerts) +
                                   size_t(slot)) * 3;
                float const delta[3] = {positions[at + 0] - positions[base + 0],
                                        positions[at + 1] - positions[base + 1],
                                        positions[at + 2] - positions[base + 2]};
                if (slot == 0) {
                    reference[0] = delta[0];
                    reference[1] = delta[1];
                    reference[2] = delta[2];
                } else if (std::fabs(delta[0] - reference[0]) > 4e-4f ||
                           std::fabs(delta[1] - reference[1]) > 4e-4f ||
                           std::fabs(delta[2] - reference[2]) > 4e-4f) {
                    columns = false;
                    break;
                }
                float const parallel = Dot(delta, normal);
                float const residual[3] = {delta[0] - parallel * normal[0],
                                           delta[1] - parallel * normal[1],
                                           delta[2] - parallel * normal[2]};
                if (Length(residual) > 4e-4f || !(parallel > 0.0f)) {
                    columns = false;
                    break;
                }
            }
        }
        *outColumnsNormal = columns;
    }
    return true;
}

struct GuideGeometry {
    std::vector<int> counts;
    std::vector<uint64_t> ids;
    std::vector<float> points;
};

bool CaptureTubeGuides(usdGenPomade::PomadeModel const &model, int tubeId,
                       GuideGeometry *out)
{
    if (!out) {
        return false;
    }
    *out = GuideGeometry();
    usdGenPomade::PomadeGuideSet const &guides = model.GetGuides();
    if (guides.counts.size() != size_t(guides.guideCount) ||
        guides.ids.size() != size_t(guides.guideCount) ||
        guides.tubeIds.size() != size_t(guides.guideCount)) {
        return false;
    }
    size_t offset = 0;
    for (int guide = 0; guide < guides.guideCount; ++guide) {
        int const count = guides.counts[size_t(guide)];
        if (count < 0 || offset + size_t(count) * 3 > guides.points.size()) {
            return false;
        }
        if (guides.tubeIds[size_t(guide)] == tubeId) {
            out->counts.push_back(count);
            out->ids.push_back(guides.ids[size_t(guide)]);
            out->points.insert(out->points.end(),
                               guides.points.begin() + offset,
                               guides.points.begin() + offset +
                                   size_t(count) * 3);
        }
        offset += size_t(count) * 3;
    }
    return offset == guides.points.size();
}

bool SameGuideGeometry(GuideGeometry const &a, GuideGeometry const &b)
{
    return a.counts == b.counts && a.ids == b.ids && a.points == b.points;
}

void CheckRegionBoundaryRootSections()
{
    using namespace usdGenPomade;
    // These are deliberately asymmetric: a radial fit could look plausible
    // on a square, but cannot preserve all corners of this triangle, uneven
    // quad, or concave L-shaped artist region.
    struct Shape {
        char const *name;
        std::vector<std::array<float, 2>> uv;
        bool subdivide = false;
    };
    std::vector<Shape> const planarShapes = {
        {"asymmetric triangle", {{0.13f, 0.17f}, {0.86f, 0.31f},
                                  {0.38f, 0.79f}}, false},
        {"uneven quad", {{0.12f, 0.16f}, {0.84f, 0.23f},
                          {0.67f, 0.76f}, {0.21f, 0.69f}}, false},
        {"concave region", {{0.13f, 0.15f}, {0.87f, 0.15f},
                             {0.87f, 0.41f}, {0.52f, 0.41f},
                             {0.52f, 0.79f}, {0.13f, 0.79f}}, true},
    };
    std::vector<float> const planarPoints = {
        0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 2.0f, 0.0f, 0.0f, 2.0f,
    };
    std::vector<int> const quad = {4};
    std::vector<int> const quadIndices = {0, 1, 2, 3};
    float const up[3] = {0.0f, 1.0f, 0.0f};

    for (Shape const &shape : planarShapes) {
        PomadeModel model;
        int region = -1;
        bool const setup = model.BindScalp(planarPoints, quad, quadIndices) &&
                           BuildRegionLoop(&model, shape.uv, up, &region) &&
                           // ringVerts==0 is the artist-facing auto choice:
                           // one base CV for each stable region-loop corner.
                           model.BuildTubeFromRegion(region, 4, 0, 1.5f);
        int const root = setup ? model.TubeForRegion(region) : -1;
        PomadeTubeDesc desc;
        bool columns = false;
        bool const matched = setup && root >= 0 &&
            model.GetTubeDesc(root, &desc) &&
            desc.ringVerts == int(shape.uv.size()) &&
            RootMatchesRegionLoop(model, region, desc,
                                  /*requireEverySlotOnBoundary=*/true,
                                  &columns);
        Check(setup && root >= 0,
              (std::string("region root: ") + shape.name +
               " auto root builds").c_str());
        Check(matched && columns,
              (std::string("region root: ") + shape.name +
               " auto base preserves every canonical corner and normal column")
                  .c_str());
        if (shape.subdivide && setup && root >= 0) {
            std::vector<int> children;
            TubeGeometry parentBefore, parentAfter;
            bool childLayouts = CaptureTubeGeometry(model, root, &parentBefore) &&
                model.SubdivideTube(root, 2, "kmeans", 211, &children) &&
                children.size() == 2 &&
                CaptureTubeGeometry(model, root, &parentAfter) &&
                SameTubeGeometry(parentBefore, parentAfter);
            for (int child : children) {
                TubeGeometry childGeometry;
                std::vector<PomadeFrame> childFrames;
                std::string childError;
                float childCenter[3] = {0.0f, 0.0f, 0.0f};
                float childNormal[3] = {0.0f, 0.0f, 0.0f};
                childLayouts = childLayouts &&
                    CaptureTubeGeometry(model, child, &childGeometry) &&
                    childGeometry.desc.ringVerts >= 3 &&
                    childGeometry.desc.ringVerts <= 32 &&
                    childGeometry.desc.rootFramePinned &&
                    PomadeTubeFramesCpu(childGeometry.desc, &childFrames,
                                       &childError) &&
                    !childFrames.empty();
                if (childLayouts) {
                    bool sectionsValid = !childGeometry.desc.sections.empty();
                    for (PomadeTubeSection const &section :
                         childGeometry.desc.sections) {
                        sectionsValid = sectionsValid &&
                            section.u.size() ==
                                size_t(childGeometry.desc.ringVerts) &&
                            section.v.size() ==
                                size_t(childGeometry.desc.ringVerts) &&
                            PomadeSectionMeanRadius(section) > 1e-6f;
                    }
                    childCenter[0] = childGeometry.desc.centerX[0];
                    childCenter[1] = childGeometry.desc.centerY[0];
                    childCenter[2] = childGeometry.desc.centerZ[0];
                    childNormal[0] = childFrames.front().tx;
                    childNormal[1] = childFrames.front().ty;
                    childNormal[2] = childFrames.front().tz;
                    childLayouts = sectionsValid && RootRingPointsOnPlane(
                        childGeometry.positions,
                        childGeometry.desc.ringVerts, childCenter, childNormal);
                }
            }
            Check(childLayouts,
                  "region root: auto concave footprint survives subdivision while children keep valid clipped K5 rings and the parent stays exact");
        }
    }

    // Eight explicitly requested ring CVs keep the four authored corners
    // exactly and place every added CV on a boundary edge. In particular,
    // none may come from a centroid ray/median fallback inside the quad.
    {
        Shape const uneven = planarShapes[1];
        PomadeModel model;
        int region = -1;
        bool const setup = model.BindScalp(planarPoints, quad, quadIndices) &&
                           BuildRegionLoop(&model, uneven.uv, up, &region) &&
                           model.BuildTubeFromRegion(region, 4, 8, 1.5f);
        int const root = setup ? model.TubeForRegion(region) : -1;
        PomadeTubeDesc desc;
        bool columns = false;
        bool const matched = setup && root >= 0 &&
            model.GetTubeDesc(root, &desc) && desc.ringVerts == 8 &&
            RootMatchesRegionLoop(model, region, desc,
                                  /*requireEverySlotOnBoundary=*/true,
                                  &columns);
        Check(setup && root >= 0,
              "region root: explicit eight-CV uneven root builds");
        Check(matched && columns,
              "region root: explicit eight-CV root retains corners, adds only edge controls and keeps normal columns");
    }

    // This non-axis-aligned plane crosses the historical frame-reference
    // tie. A descriptor whose upper frames roll 90 degrees can still keep
    // the root section plausible, so the refined rail check above reaches
    // into the first span and every later one.
    {
        std::vector<float> const obliquePoints = {
            0.0f, 0.0f, 0.0f, 2.0f, 0.5f, 0.1f,
            2.0f, 0.8f, 2.1f, 0.0f, 0.3f, 2.0f,
        };
        float obliqueNormal[3] = {0.97f, -4.0f, 0.60f};
        Normalize(obliqueNormal);
        Shape const uneven = planarShapes[1];
        PomadeModel model;
        int region = -1;
        bool const setup = model.BindScalp(obliquePoints, quad, quadIndices) &&
                           BuildRegionLoop(&model, uneven.uv, obliqueNormal,
                                           &region) &&
                           model.BuildTubeFromRegion(region, 5, 0, 1.5f);
        int const root = setup ? model.TubeForRegion(region) : -1;
        PomadeTubeDesc desc;
        bool columns = false;
        bool const matched = setup && root >= 0 &&
            model.GetTubeDesc(root, &desc) && desc.ringVerts == 4 &&
            RootMatchesRegionLoop(model, region, desc,
                                  /*requireEverySlotOnBoundary=*/true,
                                  &columns);
        Check(setup && root >= 0,
              "region root: oblique uneven auto root builds");
        Check(matched && columns,
              "region root: oblique refined K5 rails keep every projected corner offset without first-span roll");
    }

    // The editable chart remains planar on a curved support. The derived
    // display boundary is checked separately against the actual scalp.
    {
        std::vector<float> const curvedPoints = {
            0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f,
            2.0f, 0.9f, 2.0f, 0.0f, -0.7f, 2.0f,
        };
        Shape const triangle = {"curved triangle",
                                {{0.16f, 0.17f}, {0.85f, 0.31f},
                                 {0.39f, 0.78f}}, false};
        PomadeModel model;
        int region = -1;
        bool const setup = model.BindScalp(curvedPoints, quad, quadIndices) &&
                           BuildRegionLoop(&model, triangle.uv, up, &region) &&
                           model.BuildTubeFromRegion(region, 4, 0, 1.5f);
        int const root = setup ? model.TubeForRegion(region) : -1;
        PomadeTubeDesc desc;
        bool columns = false;
        bool const got = setup && root >= 0 && model.GetTubeDesc(root, &desc);
        desc.rootSurfaceOffsets.clear();
        bool const matched = got && desc.ringVerts == 3 &&
            RootMatchesRegionLoop(model, region, desc,
                                  /*requireEverySlotOnBoundary=*/true,
                                  &columns);
        Check(setup && root >= 0,
              "region root: curved triangle auto root builds");
        Check(matched && columns,
              "region root: curved auto base matches projected canonical corners and normal columns");
    }
}

void CheckCurvedRootAttachment()
{
    using namespace usdGenPomade;
    PomadeModel model;
    std::vector<float> const points = {
        0, 0, 0, 2, 0, 0, 2, 0.9f, 2, 0, -0.7f, 2};
    float const up[3] = {0, 1, 0};
    int region = -1;
    Check(model.BindScalp(points, {4}, {0, 3, 2, 1}) &&
              BuildRegionLoop(&model, {{0.16f, 0.17f}, {0.85f, 0.31f},
                                      {0.39f, 0.78f}}, up, &region) &&
              model.BuildTubeFromRegion(region, 4, 0, 1.5f),
          "attachment: curved growth surface and four-ring tube build");
    auto const scalp = model.GetScalp();
    int const root = model.TubeForRegion(region);
    auto attached = [&](int id, int segments) {
        PomadeTubeDesc tube;
        std::vector<PomadeFrame> frames;
        std::vector<float> p, n, t;
        std::string error;
        if (!model.GetTubeDesc(id, &tube) ||
            tube.rootSurfaceOffsets.size() != size_t(tube.ringVerts) * 3 ||
            !PomadeTubeFramesCpu(tube, &frames, &error) ||
            !PomadeTessellateCpu(tube, frames, segments, &p, &n, &t, &error))
            return false;
        for (int s = 0; s < tube.ringVerts; ++s) {
            PomadeHit const hit = PomadeClosestPointCpu(*scalp, p.data() + s * 3);
            if (!hit.hit || hit.t > 2e-4f) return false;
        }
        return true;
    };
    bool density = true;
    for (int segments : {1, 2, 4, 8, 16})
        density &= model.SetDisplaySegments(segments) && attached(root, segments);
    Check(density, "attachment: increasing display subdivision never lifts the root from the scalp");
    std::vector<int> children;
    bool split = model.SubdivideTube(root, 2, "kmeans", 211, &children) &&
                 children.size() == 2;
    for (int child : children) split &= attached(child, 8);
    Check(split, "attachment: subdivided child boundaries stay on the growth surface");
    PomadeModel edgeModel;
    int edgeRegion = -1;
    bool const edgeSetup = edgeModel.BindScalp(points, {4}, {0, 3, 2, 1}) &&
        BuildRegionLoop(&edgeModel, {{0.16f, 0.17f}, {0.85f, 0.31f},
                                     {0.39f, 0.78f}}, up, &edgeRegion) &&
        edgeModel.BuildTubeFromRegion(edgeRegion, 4, 0, 1.5f) &&
        edgeModel.ScaleSectionRing(0, 4.0f);
    PomadeTubeDesc edgeTube;
    std::vector<PomadeFrame> edgeFrames;
    std::vector<float> edgeP, edgeN, edgeT;
    std::string edgeError;
    bool edgeAttached = edgeSetup && edgeModel.GetTubeDesc(0, &edgeTube) &&
        PomadeTubeFramesCpu(edgeTube, &edgeFrames, &edgeError) &&
        PomadeTessellateCpu(edgeTube, edgeFrames, 8, &edgeP, &edgeN, &edgeT,
                            &edgeError);
    for (int s = 0; edgeAttached && s < edgeTube.ringVerts; ++s) {
        auto const hit = PomadeClosestPointCpu(*scalp, edgeP.data() + s * 3);
        edgeAttached &= hit.hit && hit.t < 2e-4f;
    }
    Check(edgeAttached, "attachment: an open mesh boundary cannot leave root CVs floating");
}

void CheckPinnedRegionRoot()
{
    using namespace usdGenPomade;

    // Plane p(u,v) = (u,v,v).  It makes a useful regression surface: its
    // normal is neither world-up nor parallel to the later center-CV move.
    std::vector<float> const points = {0.0f, 0.0f, 0.0f,
                                       1.0f, 0.0f, 0.0f,
                                       1.0f, 1.0f, 1.0f,
                                       0.0f, 1.0f, 1.0f};
    std::vector<int> const counts = {4};
    std::vector<int> const indices = {0, 1, 2, 3};
    float planeNormal[3] = {0.0f, -1.0f, 1.0f};
    Normalize(planeNormal);
    float const rootCenter[3] = {0.5f, 0.5f, 0.5f};

    PomadeModel model;
    Check(model.BindScalp(points, counts, indices) &&
              AddTiltedRegion(&model, planeNormal),
          "root frame: tilted-plane region builds");
    Check(model.BuildTubeFromRegion(0, 4, 8, 2.0f),
          "root frame: region builds an authored tube");

    PomadeTubeDesc before;
    std::vector<PomadeFrame> beforeFrames;
    std::string error;
    Check(model.GetTubeDesc(0, &before) && before.rootFramePinned &&
              PomadeTubeFramesCpu(before, &beforeFrames, &error) &&
              beforeFrames.size() == before.centerX.size() &&
              std::fabs(FrameAlignment(beforeFrames.front(), planeNormal)) >
                  0.999f,
          "root frame: region descriptor pins frame zero to the tilted support plane");

    PomadeTubeSection rootSection;
    Check(model.GetTubeSection(0, 0, &rootSection) &&
              rootSection.u.size() == size_t(before.ringVerts) &&
              rootSection.v.size() == size_t(before.ringVerts) &&
              RootRingOnPlane(model.GetHostMesh(), before.ringVerts,
                              rootCenter, planeNormal),
          "root frame: authored root section and final tessellated base are coplanar with the scalp");

    PomadeFrame const pinned = beforeFrames.front();
    // A tangential bend makes raw K4's first tangent differ from the scalp
    // normal. The pinned root may not follow it, while the shaft may bend.
    Check(model.MoveCenterCV(1, 0.85f, 0.0f, 0.0f),
          "root frame: first shaft CV moves tangentially");
    PomadeTubeDesc bent;
    std::vector<PomadeFrame> bentFrames;
    error.clear();
    Check(model.GetTubeDesc(0, &bent) && bent.rootFramePinned &&
              PomadeTubeFramesCpu(bent, &bentFrames, &error) &&
              SameFrame(pinned, bentFrames.front()) &&
              std::fabs(FrameAlignment(bentFrames.front(), planeNormal)) >
                  0.999f &&
              RootRingOnPlane(model.GetHostMesh(), bent.ringVerts,
                              rootCenter, planeNormal),
          "root frame: bending CV 1 preserves the root section plane and final base geometry");

    // Snapshot/hydrate must retain enough authoring state for the same final
    // base to be rebuilt later on a model which has no live scalp binding.
    PomadeModel::TubeSnapshot const snapshot = model.Snapshot();
    PomadeModel restored;
    PomadeTubeDesc hydrated;
    std::vector<PomadeFrame> hydratedFrames;
    error.clear();
    Check(snapshot.rootFramePinned && SameFrame(snapshot.rootFrame, pinned) &&
              restored.Restore(snapshot) && restored.GetTubeDesc(0, &hydrated) &&
              hydrated.rootFramePinned &&
              PomadeTubeFramesCpu(hydrated, &hydratedFrames, &error) &&
              SameFrame(hydratedFrames.front(), pinned) &&
              RootRingOnPlane(restored.GetHostMesh(), hydrated.ringVerts,
                              rootCenter, planeNormal),
          "root frame: snapshot restore preserves the pinned root and rebuilt base geometry");

    // K14 must carry the plane frame to every child.  The child surfaces are
    // tessellated here, not inferred from their descriptors, because store
    // tubes do not share tube zero's host-mesh cache while subdivided.
    std::vector<int> children;
    bool childBases = model.SubdivideTube(0, 2, "kmeans", 23, &children) &&
                      children.size() == 2;
    for (int childId : children) {
        PomadeTubeDesc child;
        std::vector<PomadeFrame> childFrames;
        std::vector<float> positions, normals, ringT;
        std::string childError;
        childBases = childBases && model.GetTubeDesc(childId, &child) &&
                      child.rootFramePinned &&
                      PomadeTubeFramesCpu(child, &childFrames, &childError) &&
                      SameFrame(childFrames.front(), pinned) &&
                      PomadeTessellateCpu(child, childFrames, 1, &positions,
                                          &normals, &ringT, &childError) &&
                      RootRingPointsOnPlane(positions, child.ringVerts,
                                            rootCenter, planeNormal);
    }
    Check(childBases,
          "root frame: subdivision children retain the pinned support plane in their tessellated bases");

    PomadeTubeDesc merged;
    std::vector<PomadeFrame> mergedFrames;
    error.clear();
    Check(model.MergeChildren(0) && model.GetTubeDesc(0, &merged) &&
              merged.rootFramePinned &&
              PomadeTubeFramesCpu(merged, &mergedFrames, &error) &&
              SameFrame(mergedFrames.front(), pinned) &&
              RootRingOnPlane(model.GetHostMesh(), merged.ringVerts,
                              rootCenter, planeNormal),
          "root frame: merging pinned children retains the parent plane and final base");

    // Ordinary descriptors deliberately retain the raw K4 behaviour. This
    // prevents a region-specific root plane from leaking into legacy tubes.
    PomadeTubeDesc legacy = bent;
    legacy.rootFramePinned = false;
    legacy.frameReference = {{1.0f, 0.0f, 0.0f,
                              0.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 1.0f}};
    std::vector<PomadeFrame> rawFrames;
    std::vector<PomadeFrame> genericFrames;
    error.clear();
    Check(PomadeCenterFramesCpu(legacy.centerX.data(), legacy.centerY.data(),
                               legacy.centerZ.data(),
                               int(legacy.centerX.size()), &rawFrames,
                               &error) &&
              PomadeTubeFramesCpu(legacy, &genericFrames, &error) &&
              SameFrame(rawFrames.front(), genericFrames.front()) &&
              std::fabs(FrameAlignment(genericFrames.front(), planeNormal)) <
                  0.99f,
          "root frame: unpinned legacy descriptors retain their tilted raw K4 frame");
}

void CheckTubeZeroEdgeSplitPropagation()
{
    using namespace usdGenPomade;
    int constexpr count = 2;
    int constexpr seed = 613;
    float constexpr edgeA = 0.37f;
    float constexpr edgeB = -0.81f;
    float constexpr edgeC = 0.19f;
    PomadeModel model;
    PomadeTubeShape shape;
    shape.rings = 5;
    shape.ringVerts = 8;
    shape.radius = 0.8f;
    shape.length = 3.5f;
    std::vector<int> children;
    bool const setup = model.BuildTestTube(shape) &&
        model.SubdivideTube(0, count, "edge", seed, edgeA, edgeB, edgeC,
                            &children) && children.size() == 2;
    Check(setup, "edge propagation: tube-0 accepts a nondefault edge split");
    if (!setup) {
        return;
    }
    Check(SameEdgeSplit(model.GetSubdivideParams(), count, seed, edgeA,
                        edgeB, edgeC),
          "edge propagation: tube-0 stores the drawn edge coefficients");

    TubeGeometry parentBeforeK7;
    bool const captured = CaptureTubeGeometry(model, 0, &parentBeforeK7);
    Check(captured, "edge propagation: parent geometry captures before child K7");
    if (!captured) {
        return;
    }
    // A leaf center edit uses K7 to write its aggregate into tube 0. Its
    // untouched sibling makes the edge split line observable: default
    // (1,0,0) would not recognize this nondefault partition as center-only.
    bool const childEdit = model.MoveTubeCenterCV(children[0], 2, 0.11f,
                                                   -0.06f, 0.04f);
    PomadeTubeDesc childA, childB, parentAfterK7, expectedK7;
    std::vector<PomadeFrame> parentFrames;
    std::string error;
    PomadeSubdivideDesc edgeParams;
    edgeParams.count = count;
    edgeParams.seed = seed;
    edgeParams.splitMode = PomadeSplit_Edge;
    edgeParams.edgeA = edgeA;
    edgeParams.edgeB = edgeB;
    edgeParams.edgeC = edgeC;
    bool const expectedK7Ready = childEdit &&
        model.GetTubeDesc(children[0], &childA) &&
        model.GetTubeDesc(children[1], &childB) &&
        model.GetTubeDesc(0, &parentAfterK7) &&
        PomadeMergeTubesCpu({childA, childB}, edgeParams, &parentBeforeK7.desc,
                           &expectedK7, &error);
    TubeGeometry expectedK7Geometry, actualK7Geometry;
    bool const k7Matches = expectedK7Ready &&
        CaptureDescGeometry(expectedK7, &expectedK7Geometry) &&
        CaptureTubeGeometry(model, 0, &actualK7Geometry) &&
        SameTubeGeometry(expectedK7Geometry, actualK7Geometry);
    Check(childEdit && SameEdgeSplit(model.GetSubdivideParams(), count, seed,
                                     edgeA, edgeB, edgeC) && k7Matches,
          "edge propagation: child K7 preserves and uses the original tube-0 edge cut");
    if (!k7Matches) {
        return;
    }

    std::vector<PomadeModel::TubeRecord> records(children.size());
    bool recordsRead = true;
    for (size_t index = 0; index < children.size(); ++index) {
        recordsRead = recordsRead && model.GetTubeRecord(children[index],
                                                          &records[index]);
    }
    Check(recordsRead, "edge propagation: child residual records read before parent K6");
    if (!recordsRead) {
        return;
    }
    bool const parentEdit = model.MoveTubeCenterCV(0, 1, -0.05f, 0.08f,
                                                    0.03f);
    PomadeTubeDesc parentAfterK6;
    std::vector<PomadeTubeDesc> derived;
    bool const expectedK6Ready = parentEdit &&
        model.GetTubeDesc(0, &parentAfterK6) &&
        PomadeTubeFramesCpu(parentAfterK6, &parentFrames, &error) &&
        PomadeSubdivideTubeCpu(parentAfterK6, parentFrames, edgeParams,
                              &derived, &error) &&
        derived.size() == children.size();
    bool k6Matches = expectedK6Ready;
    for (size_t index = 0; index < children.size() && k6Matches; ++index) {
        int const childIndex = records[index].actual.childIndex;
        PomadeTubeDesc fresh, expected;
        PomadeShapeDeltas stored;
        TubeGeometry expectedGeometry, actualGeometry;
        k6Matches = childIndex >= 0 && size_t(childIndex) < derived.size() &&
            MatchDerivedLayout(derived[size_t(childIndex)],
                               records[index].actual, &fresh, &error) &&
            PomadeHierarchicalSculptApplyCpu(
                fresh, records[index].actual,
                records[index].derived, records[index].deltas,
                records[index].lockChildren, /*preserveLength=*/true,
                &expected, &stored, &error) &&
            CaptureDescGeometry(expected, &expectedGeometry) &&
            CaptureTubeGeometry(model, children[index], &actualGeometry) &&
            SameTubeGeometry(expectedGeometry, actualGeometry);
    }
    Check(parentEdit && SameEdgeSplit(model.GetSubdivideParams(), count, seed,
                                      edgeA, edgeB, edgeC) && k6Matches,
          "edge propagation: later tube-0 K6 re-derives both leaves from the original edge cut");
}

void CheckInternalChildCvKeepsCurvedParentExact()
{
    using namespace usdGenPomade;

    // The K14 children of this curved, oblique root start in their own
    // frames.  Editing a clipped-only material slot must therefore be gated
    // against the pre-edit child sample, not a newly derived child frame:
    // no inherited holding corner moved, so K7 must leave the parent exact.
    std::vector<float> const points = {
        0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f,
        2.0f, 0.9f, 2.0f, 0.0f, -0.7f, 2.0f,
    };
    std::vector<int> const counts = {4};
    std::vector<int> const indices = {0, 1, 2, 3};
    float const up[3] = {0.0f, 1.0f, 0.0f};
    PomadeModel model;
    int region = -1;
    std::vector<int> children;
    bool const setup = model.BindScalp(points, counts, indices) &&
        BuildRegionLoop(&model, {{0.14f, 0.16f}, {0.88f, 0.24f},
                                 {0.76f, 0.81f}, {0.19f, 0.72f}},
                        up, &region) &&
        model.BuildTubeFromRegion(region, 5, 8, 1.7f) &&
        model.SubdivideTube(0, 3, "kmeans", 401, &children) &&
        children.size() == 3;
    Check(setup,
          "holding edge: curved oblique root produces clipped K14 children");
    if (!setup) {
        return;
    }

    int target = -1, section = -1, slot = -1;
    for (int childId : children) {
        PomadeTubeDesc child;
        if (!model.GetTubeDesc(childId, &child)) {
            continue;
        }
        for (int s = 1; s + 1 < int(child.sections.size()) && target < 0;
             ++s) {
            std::vector<bool> holding(size_t(child.ringVerts), false);
            for (PomadeParentBoundaryBinding const &binding :
                 child.inheritedBoundaryBindings) {
                if (binding.section == s && binding.childSlot >= 0 &&
                    binding.childSlot < child.ringVerts) {
                    holding[size_t(binding.childSlot)] = true;
                }
            }
            for (int candidate = 0; candidate < child.ringVerts;
                 ++candidate) {
                if (!holding[size_t(candidate)]) {
                    target = childId;
                    section = s;
                    slot = candidate;
                    break;
                }
            }
        }
        if (target >= 0) {
            break;
        }
    }
    Check(target >= 0,
          "holding edge: curved child exposes an internal clipped CV");
    if (target < 0) {
        return;
    }

    TubeGeometry parentBefore, parentAfter;
    bool const unchanged = CaptureTubeGeometry(model, 0, &parentBefore) &&
        model.MoveTubeSectionCV(target, section, slot, 0.071f, -0.049f) &&
        CaptureTubeGeometry(model, 0, &parentAfter) &&
        SameTubeGeometry(parentBefore, parentAfter);
    Check(unchanged,
          "holding edge: curved child internal CV edit leaves parent K7 geometry exact");
}

void CheckAutoRegionSculptPropagation()
{
    using namespace usdGenPomade;

    // This is the default artist path: Region creates an Auto (zero request)
    // quad root, subdivision creates clipped children, then component edits
    // write back through K7 before a deliberate parent K6.  Do not couple the
    // assertion to child ring counts: clipping is allowed to add edge points.
    std::vector<float> const points = {0.0f, 0.0f, 0.0f,
                                       2.0f, 0.0f, 0.0f,
                                       2.0f, 0.0f, 2.0f,
                                       0.0f, 0.0f, 2.0f};
    std::vector<int> const counts = {4};
    std::vector<int> const indices = {0, 1, 2, 3};
    float const up[3] = {0.0f, 1.0f, 0.0f};
    PomadeModel model;
    int region = -1;
    std::vector<int> children;
    bool const setup = model.BindScalp(points, counts, indices) &&
        BuildRegionLoop(&model, {{0.17f, 0.19f}, {0.86f, 0.18f},
                                 {0.83f, 0.78f}, {0.15f, 0.81f}},
                        up, &region) &&
        model.BuildTubeFromRegion(region, 5, /*ringVerts Auto=*/0, 1.6f) &&
        model.SubdivideTube(0, 2, "kmeans", 97, &children) &&
        children.size() == 2;
    Check(setup,
          "auto region propagation: default quad root subdivides into editable children");
    if (!setup) {
        return;
    }

    int const target = children.front();
    PomadeTubeDesc targetDesc;
    bool const targetReady = model.GetTubeDesc(target, &targetDesc) &&
        targetDesc.sections.size() > 1 && targetDesc.ringVerts >= 3;
    Check(targetReady,
          "auto region propagation: selected clipped child has component CVs");
    if (!targetReady) {
        return;
    }

    TubeGeometry beforeSection, afterSection, afterCenter;
    bool const sculpted = CaptureTubeGeometry(model, target, &beforeSection) &&
        model.MoveTubeSectionCV(target, 1, 0, 0.075f, -0.045f) &&
        CaptureTubeGeometry(model, target, &afterSection) &&
        model.MoveTubeCenterCV(target, 2, 0.035f, 0.02f, -0.025f) &&
        CaptureTubeGeometry(model, target, &afterCenter);
    Check(sculpted && !SameTubeGeometry(beforeSection, afterSection) &&
              !SameTubeGeometry(afterSection, afterCenter),
          "auto region propagation: section and center CV edits survive child K7 writeback");
    if (!sculpted) {
        return;
    }

    std::vector<PomadeModel::TubeRecord> records(children.size());
    bool recordsRead = true;
    for (size_t index = 0; index < children.size(); ++index) {
        recordsRead = recordsRead && model.GetTubeRecord(children[index],
                                                          &records[index]);
    }
    Check(recordsRead,
          "auto region propagation: sculpt records capture before parent K6");
    if (!recordsRead) {
        return;
    }

    bool const parentEdit = model.MoveTubeCenterCV(0, 1, -0.04f, 0.055f,
                                                    0.025f);
    PomadeTubeDesc parent;
    std::vector<PomadeFrame> parentFrames;
    std::vector<PomadeTubeDesc> derived;
    std::string error;
    PomadeModel::SubdivideParams const split = model.GetSubdivideParams();
    PomadeSubdivideDesc params;
    params.count = split.count;
    params.seed = split.seed;
    params.splitMode = split.splitMode == "edge" ? PomadeSplit_Edge
                                                  : PomadeSplit_KMeans;
    params.edgeA = split.edgeA;
    params.edgeB = split.edgeB;
    params.edgeC = split.edgeC;
    bool const rederived = parentEdit && model.GetTubeDesc(0, &parent) &&
        PomadeTubeFramesCpu(parent, &parentFrames, &error) &&
        PomadeSubdivideTubeCpu(parent, parentFrames, params, &derived, &error) &&
        derived.size() == children.size();
    bool k6Matches = rederived;
    bool targetResidualSurvived = false;
    for (size_t index = 0; index < children.size() && k6Matches; ++index) {
        int const childIndex = records[index].actual.childIndex;
        PomadeTubeDesc fresh, expected;
        PomadeShapeDeltas stored;
        TubeGeometry expectedGeometry, derivedGeometry, actualGeometry;
        bool const expectedChild = childIndex >= 0 &&
            size_t(childIndex) < derived.size() &&
            MatchDerivedLayout(derived[size_t(childIndex)],
                               records[index].actual, &fresh, &error) &&
            PomadeHierarchicalSculptApplyCpu(
                fresh, records[index].actual,
                records[index].derived, records[index].deltas,
                records[index].lockChildren, /*preserveLength=*/true,
                &expected, &stored, &error) &&
            CaptureDescGeometry(expected, &expectedGeometry) &&
            CaptureDescGeometry(fresh, &derivedGeometry) &&
            CaptureTubeGeometry(model, children[index], &actualGeometry) &&
            SameTubeGeometry(expectedGeometry, actualGeometry);
        k6Matches = k6Matches && expectedChild;
        if (children[index] == target) {
            targetResidualSurvived = expectedChild &&
                !SameTubeGeometry(actualGeometry, derivedGeometry);
        }
    }
    Check(k6Matches && targetResidualSurvived,
          "auto region propagation: parent K6 preserves the child section sculpt");
}

void CheckSubdividedChildIsolation()
{
    using namespace usdGenPomade;

    // This is the artist path: make one region-rooted tube, split it, then
    // move and reshape exactly one selected leaf.  Keeping the support plane
    // in the fixture catches an accidental tube-0 writeback / re-derive too.
    std::vector<float> const points = {0.0f, 0.0f, 0.0f,
                                       1.0f, 0.0f, 0.0f,
                                       1.0f, 1.0f, 1.0f,
                                       0.0f, 1.0f, 1.0f};
    std::vector<int> const counts = {4};
    std::vector<int> const indices = {0, 1, 2, 3};
    float normal[3] = {0.0f, -1.0f, 1.0f};
    Normalize(normal);

    PomadeModel model;
    std::vector<int> children;
    Check(model.BindScalp(points, counts, indices) &&
              AddTiltedRegion(&model, normal) &&
              model.BuildTubeFromRegion(0, 4, 8, 2.0f) &&
              model.SubdivideTube(0, 3, "kmeans", 29, &children) &&
              children.size() == 3 && model.RefillGuides(1.0f),
          "child isolation: region-rooted tube splits and fills");
    if (children.size() != 3) {
        return;
    }

    std::vector<TubeGeometry> before(children.size());
    std::vector<GuideGeometry> guidesBefore(children.size());
    bool captured = true;
    for (size_t i = 0; i < children.size(); ++i) {
        captured = captured && CaptureTubeGeometry(model, children[i],
                                                    &before[i]) &&
                   CaptureTubeGuides(model, children[i], &guidesBefore[i]);
    }
    Check(captured, "child isolation: every leaf geometry and guide output reads");
    if (!captured) {
        return;
    }

    size_t targetIndex = 0;
    while (targetIndex < guidesBefore.size() &&
           guidesBefore[targetIndex].points.empty()) {
        ++targetIndex;
    }
    Check(targetIndex < children.size(),
          "child isolation: at least one selected leaf owns guides");
    if (targetIndex == children.size()) {
        return;
    }
    int const target = children[targetIndex];
    Check(model.MoveTubeCenterCV(target, 1, 0.35f, 0.0f, -0.20f) &&
              model.RefillGuides(1.0f),
          "child isolation: selected child center CV moves");
    std::vector<TubeGeometry> afterCenter(children.size());
    std::vector<GuideGeometry> guidesAfterCenter(children.size());
    bool centerState = true;
    for (size_t i = 0; i < children.size(); ++i) {
        centerState = centerState && CaptureTubeGeometry(model, children[i],
                                                          &afterCenter[i]) &&
                      CaptureTubeGuides(model, children[i],
                                        &guidesAfterCenter[i]);
    }
    Check(centerState &&
              !SameTubeGeometry(before[targetIndex],
                                afterCenter[targetIndex]) &&
              !SameGuideGeometry(guidesBefore[targetIndex],
                                 guidesAfterCenter[targetIndex]),
          "child isolation: selected child and its guides change");
    bool centerSiblingsFixed = centerState;
    for (size_t i = 0; i < children.size(); ++i) {
        if (i == targetIndex) {
            continue;
        }
        centerSiblingsFixed = centerSiblingsFixed &&
                              SameTubeGeometry(before[i], afterCenter[i]) &&
                              SameGuideGeometry(guidesBefore[i],
                                                guidesAfterCenter[i]);
    }
    Check(centerSiblingsFixed,
          "child isolation: center edit leaves sibling curves, sections, meshes and guides exact");

    Check(model.ScaleTubeSectionRing(target, 1, 1.35f) &&
              model.TwistTubeSectionRing(target, 1, 0.29f) &&
              model.RefillGuides(1.0f),
          "child isolation: selected child section ring scales and twists");
    std::vector<TubeGeometry> afterSection(children.size());
    std::vector<GuideGeometry> guidesAfterSection(children.size());
    bool sectionState = true;
    for (size_t i = 0; i < children.size(); ++i) {
        sectionState = sectionState && CaptureTubeGeometry(model, children[i],
                                                           &afterSection[i]) &&
                       CaptureTubeGuides(model, children[i],
                                         &guidesAfterSection[i]);
    }
    Check(sectionState &&
              !SameTubeGeometry(afterCenter[targetIndex],
                                afterSection[targetIndex]) &&
              !SameGuideGeometry(guidesAfterCenter[targetIndex],
                                 guidesAfterSection[targetIndex]),
          "child isolation: selected section and its guides change");
    bool sectionSiblingsFixed = sectionState;
    for (size_t i = 0; i < children.size(); ++i) {
        if (i == targetIndex) {
            continue;
        }
        sectionSiblingsFixed = sectionSiblingsFixed &&
                               SameTubeGeometry(afterCenter[i], afterSection[i]) &&
                               SameGuideGeometry(guidesAfterCenter[i],
                                                 guidesAfterSection[i]);
    }
    Check(sectionSiblingsFixed,
          "child isolation: section edit leaves sibling curves, sections, meshes and guides exact");

    Check(model.MoveTubeSectionCV(target, 1, 0, 0.10f, -0.06f) &&
              model.RefillGuides(1.0f),
          "child isolation: selected child section CV moves");
    std::vector<TubeGeometry> afterSectionCv(children.size());
    std::vector<GuideGeometry> guidesAfterSectionCv(children.size());
    bool sectionCvState = true;
    for (size_t i = 0; i < children.size(); ++i) {
        sectionCvState = sectionCvState &&
                         CaptureTubeGeometry(model, children[i],
                                             &afterSectionCv[i]) &&
                         CaptureTubeGuides(model, children[i],
                                           &guidesAfterSectionCv[i]);
    }
    Check(sectionCvState &&
              !SameTubeGeometry(afterSection[targetIndex],
                                afterSectionCv[targetIndex]) &&
              !SameGuideGeometry(guidesAfterSection[targetIndex],
                                 guidesAfterSectionCv[targetIndex]),
          "child isolation: selected section CV and its guides change");
    bool sectionCvSiblingsFixed = sectionCvState;
    for (size_t i = 0; i < children.size(); ++i) {
        if (i == targetIndex) {
            continue;
        }
        sectionCvSiblingsFixed = sectionCvSiblingsFixed &&
                                SameTubeGeometry(afterSection[i],
                                                 afterSectionCv[i]) &&
                                SameGuideGeometry(guidesAfterSection[i],
                                                  guidesAfterSectionCv[i]);
    }
    Check(sectionCvSiblingsFixed,
          "child isolation: section-CV edit leaves sibling curves, sections, meshes and guides exact");

    // K7 has updated the parent aggregate during the leaf edits above.  A
    // later deliberate parent move must be a normal K6 re-derive: each child
    // is rebuilt from that new parent and its own stored delta, rather than
    // receiving an accidental sibling delta from the earlier K7 writeback.
    std::vector<PomadeModel::TubeRecord> records(children.size());
    bool recordsRead = true;
    for (size_t i = 0; i < children.size(); ++i) {
        recordsRead = recordsRead && model.GetTubeRecord(children[i],
                                                          &records[i]);
    }
    Check(recordsRead, "child isolation: pre-parent-move child records read");
    if (!recordsRead) {
        return;
    }
    // K7 has just written tube 0 from the edited child family.  Every direct
    // child must now hold a reference freshly derived from that written
    // parent, while retaining its unchanged actual and remeasured residual.
    // Otherwise the following K6 applies an old residual to a new parent.
    PomadeTubeDesc parentAfterK7;
    std::vector<PomadeFrame> k7Frames;
    std::vector<PomadeTubeDesc> k7Derived;
    std::string k7Error;
    PomadeModel::SubdivideParams const k7Split = model.GetSubdivideParams();
    PomadeSubdivideDesc k7Params;
    k7Params.count = k7Split.count;
    k7Params.seed = k7Split.seed;
    k7Params.splitMode = k7Split.splitMode == "edge" ? PomadeSplit_Edge
                                                       : PomadeSplit_KMeans;
    k7Params.edgeA = k7Split.edgeA;
    k7Params.edgeB = k7Split.edgeB;
    k7Params.edgeC = k7Split.edgeC;
    bool k7Inputs = model.GetTubeDesc(0, &parentAfterK7) &&
                    PomadeTubeFramesCpu(parentAfterK7, &k7Frames, &k7Error) &&
                    PomadeSubdivideTubeCpu(parentAfterK7, k7Frames, k7Params,
                                           &k7Derived, &k7Error) &&
                    k7Derived.size() == children.size();
    bool k7Rebased = k7Inputs;
    for (size_t i = 0; i < children.size() && k7Rebased; ++i) {
        int const childIndex = records[i].actual.childIndex;
        PomadeTubeDesc fresh;
        std::vector<PomadeFrame> freshFrames;
        PomadeShapeDeltas freshDeltas;
        TubeGeometry freshGeometry, storedGeometry;
        PomadeTubeDesc matched;
        bool const matches = childIndex >= 0 &&
            size_t(childIndex) < k7Derived.size() &&
            ((fresh = k7Derived[size_t(childIndex)]), true) &&
            MatchDerivedLayout(fresh, records[i].actual, &matched,
                               &k7Error) &&
            ((fresh = std::move(matched)), true) &&
            PomadeTubeFramesCpu(fresh, &freshFrames, &k7Error) &&
            PomadeComputeDeltasCpu(records[i].actual, fresh, freshFrames,
                                  &freshDeltas, &k7Error) &&
            CaptureDescGeometry(fresh, &freshGeometry) &&
            CaptureDescGeometry(records[i].derived, &storedGeometry) &&
            SameTubeGeometry(freshGeometry, storedGeometry) &&
            SameDeltas(freshDeltas, records[i].deltas);
        k7Rebased = k7Rebased && matches;
    }
    Check(k7Rebased,
          "child isolation: K7 re-bases every direct child without moving its actual shape");
    Check(model.MoveTubeCenterCV(0, 1, 0.06f, 0.0f, -0.03f) &&
              model.RefillGuides(1.0f),
          "child isolation: deliberate parent center move re-derives leaves");
    PomadeTubeDesc parentAfterMove;
    std::vector<PomadeFrame> parentFrames;
    std::vector<PomadeTubeDesc> derivedAll;
    std::string parentError;
    PomadeModel::SubdivideParams const split = model.GetSubdivideParams();
    PomadeSubdivideDesc params;
    params.count = split.count;
    params.seed = split.seed;
    params.splitMode = split.splitMode == "edge" ? PomadeSplit_Edge
                                                  : PomadeSplit_KMeans;
    params.edgeA = split.edgeA;
    params.edgeB = split.edgeB;
    params.edgeC = split.edgeC;
    bool expectedInputs = model.GetTubeDesc(0, &parentAfterMove) &&
                          PomadeTubeFramesCpu(parentAfterMove, &parentFrames,
                                             &parentError) &&
                          PomadeSubdivideTubeCpu(parentAfterMove, parentFrames,
                                                params, &derivedAll,
                                                &parentError) &&
                          derivedAll.size() == children.size();
    Check(expectedInputs,
          "child isolation: parent move has a complete K6 reference family");
    if (!expectedInputs) {
        return;
    }
    std::vector<TubeGeometry> afterParentMove(children.size());
    std::vector<GuideGeometry> guidesAfterParentMove(children.size());
    bool parentMoveExact = true;
    bool siblingMovedOnlyByParent = true;
    bool targetResidualPreserved = false;
    for (size_t i = 0; i < children.size(); ++i) {
        PomadeTubeDesc fresh, expected;
        PomadeShapeDeltas stored;
        int const childIndex = records[i].actual.childIndex;
        std::string error;
        TubeGeometry expectedGeometry, derivedGeometry;
        PomadeModel::TubeRecord actualRecord;
        bool const expectedChild = childIndex >= 0 &&
                                   size_t(childIndex) < derivedAll.size() &&
                                   MatchDerivedLayout(
                                       derivedAll[size_t(childIndex)],
                                       records[i].actual, &fresh, &error) &&
                                   PomadeHierarchicalSculptApplyCpu(
                                       fresh,
                                       records[i].actual, records[i].derived,
                                       records[i].deltas,
                                       records[i].lockChildren,
                                       /*preserveLength=*/true, &expected,
                                       &stored, &error) &&
                                   CaptureDescGeometry(expected,
                                                       &expectedGeometry) &&
                                   CaptureDescGeometry(fresh,
                                                       &derivedGeometry) &&
                                   CaptureTubeGeometry(model, children[i],
                                                       &afterParentMove[i]) &&
                                   CaptureTubeGuides(
                                       model, children[i],
                                       &guidesAfterParentMove[i]) &&
                                   model.GetTubeRecord(children[i],
                                                       &actualRecord) &&
                                   SameDeltas(stored, actualRecord.deltas);
        parentMoveExact = parentMoveExact && expectedChild &&
                          SameTubeGeometry(expectedGeometry,
                                           afterParentMove[i]);
        if (i == targetIndex) {
            targetResidualPreserved = expectedChild &&
                                      !SameTubeGeometry(afterParentMove[i],
                                                        derivedGeometry);
        } else {
            siblingMovedOnlyByParent = siblingMovedOnlyByParent &&
                                         expectedChild &&
                                         !SameTubeGeometry(afterSectionCv[i],
                                                           afterParentMove[i]);
        }
    }
    Check(parentMoveExact && siblingMovedOnlyByParent &&
              targetResidualPreserved,
          "child isolation: delayed parent K6 moves siblings only by the parent and preserves the edited child residual");

    Check(model.Undo() && model.GetUndoDepth() >= 1,
          "child isolation: parent re-derive undoes");
    bool undoExact = true;
    for (size_t i = 0; i < children.size(); ++i) {
        TubeGeometry geometry;
        GuideGeometry guides;
        undoExact = undoExact && CaptureTubeGeometry(model, children[i], &geometry) &&
                    CaptureTubeGuides(model, children[i], &guides) &&
                    SameTubeGeometry(geometry, afterSectionCv[i]) &&
                    SameGuideGeometry(guides, guidesAfterSectionCv[i]);
    }
    Check(undoExact,
          "child isolation: undo restores every child and guide output exactly");

    Check(model.Redo(nullptr) && model.GetRedoDepth() == 0,
          "child isolation: parent re-derive redoes");
    bool redoExact = true;
    for (size_t i = 0; i < children.size(); ++i) {
        TubeGeometry geometry;
        GuideGeometry guides;
        redoExact = redoExact && CaptureTubeGeometry(model, children[i], &geometry) &&
                    CaptureTubeGuides(model, children[i], &guides) &&
                    SameTubeGeometry(geometry, afterParentMove[i]) &&
                    SameGuideGeometry(guides, guidesAfterParentMove[i]);
    }
    Check(redoExact,
          "child isolation: redo restores every child and guide output exactly");
}

void CheckGeneratedCurveClear()
{
    using namespace usdGenPomade;
    PomadeModel model;
    Check(model.BuildTestTube(),
          "generated curves: test tube builds");
    model.SetFreezeRoots(true);
    Check(model.RefillGuides(1.0f) && model.GetGuides().guideCount > 0,
          "generated curves: frozen-root preview fills before clear");
    int const filled = model.GetGuides().guideCount;
    Check(model.ClearGeneratedCurves() && model.GeneratedCurvesSuppressed() &&
              model.GetGuides().guideCount == 0,
          "generated curves: clear suppresses and empties the live cache");

    // The normal sculpt/tube release path calls RefillGuides. It must not
    // resurrect a user-cleared preview, even with a frozen root prefix.
    Check(model.MoveTubeCenterCV(0, 1, 0.10f, 0.0f, 0.0f) &&
              model.RefillGuides(1.0f) && model.GetGuides().guideCount == 0 &&
              model.GeneratedCurvesSuppressed(),
          "generated curves: automatic refill respects a clear");

    PomadeSnapshot const suppressed = PomadeSnapshotFromModel(model);
    Check(PomadeGuidesFromSnapshot(suppressed).counts.empty(),
          "generated curves: suppressed snapshot commits no guides");
    Check(model.GenerateGuides(1.0f) && !model.GeneratedCurvesSuppressed() &&
              model.GetGuides().guideCount == filled,
          "generated curves: explicit Fill re-enables generation");

    Check(model.ClearGeneratedCurves() && model.Undo() &&
              !model.GeneratedCurvesSuppressed() &&
              model.GetGuides().guideCount == filled,
          "generated curves: undo restores generated curves and suppression state");
    Check(model.Redo(nullptr) && model.GeneratedCurvesSuppressed() &&
              model.GetGuides().guideCount == 0,
          "generated curves: redo restores the cleared suppression state");
}

void CheckCurvedRegionFrameTransport()
{
    using namespace usdGenPomade;
    // A bilinear saddle has a support normal which turns as the loop moves
    // in V.  This catches the old failure where roots moved but the K4/K5
    // frame reference was re-bootstrapped in world axes and sheared the
    // first interpolated span.
    std::vector<float> const points = {0, 0, 0, 1, 0, 0,
                                       1, 1, 1, 0, 1, -1};
    std::vector<int> const counts = {4};
    std::vector<int> const indices = {0, 1, 2, 3};
    float const normal[3] = {0, 0, 1};
    PomadeModel model;
    std::vector<int> nodes;
    bool setup = model.BindScalp(points, counts, indices);
    float const uv[8] = {0.20f, 0.20f, 0.80f, 0.20f,
                         0.80f, 0.40f, 0.20f, 0.40f};
    for (int i = 0; setup && i < 4; ++i) {
        int const node = model.GraphAddNode(
            HitFace(*model.GetScalp(), 0, uv[i * 2], uv[i * 2 + 1], normal));
        setup = node >= 0;
        if (setup) {
            nodes.push_back(node);
        }
    }
    for (int i = 0; setup && i < 4; ++i) {
        setup = model.GraphConnect(nodes[i], nodes[(i + 1) % 4]) >= 0;
    }
    setup = setup && model.Rasterise();
    int const region = setup ? model.RegionAtSurface(0, 0.5f, 0.3f) : -1;
    setup = setup && region >= 0 && model.BuildTubeFromRegion(region, 5, 8, 2.0f);
    int const root = setup ? model.TubeForRegion(region) : -1;
    std::vector<int> children;
    setup = setup && root >= 0 && model.SubdivideTube(root, 2, "kmeans", 131,
                                                       &children) &&
            children.size() == 2 &&
            model.MoveTubeCenterCV(children[0], 2, 0.14f, -0.09f, 0.06f) &&
            model.MoveTubeSectionCV(children[0], 2, 1, 0.07f, -0.05f) &&
            model.ScaleTubeSectionRing(children[0], 2, 1.18f) &&
            model.TwistTubeSectionRing(children[0], 2, 0.21f);
    Check(setup, "curved transport: sculpted subdivided saddle region builds");
    if (!setup) {
        return;
    }
    std::vector<int> affected = {root, children[0], children[1]};
    std::vector<TubeGeometry> before(affected.size()), after(affected.size());
    PomadeModel::TubeRecord sculptRecord;
    bool captured = true;
    for (size_t i = 0; i < affected.size(); ++i) {
        captured = captured && CaptureTubeGeometry(model, affected[i], &before[i]);
    }
    captured = captured && model.GetTubeRecord(children[0], &sculptRecord);
    if (!captured) {
        Check(false, "curved transport: full refined K5 capture reads");
        return;
    }
    std::vector<PomadeHit> targets;
    for (int node : nodes) {
        PomadeGraphNode current;
        captured = captured && model.GraphGetNode(node, &current);
        targets.push_back(HitFace(*model.GetScalp(), current.faceId, current.u,
                                  current.v + 0.20f, normal));
    }
    Check(captured && model.GraphMoveNodes(nodes, targets) && model.Rasterise(),
          "curved transport: graph move changes the saddle support frame");
    bool capturedAfter = true;
    bool properFrames = true;
    bool childRootsConform = true;
    for (size_t i = 0; i < affected.size(); ++i) {
        capturedAfter = capturedAfter &&
            CaptureTubeGeometry(model, affected[i], &after[i]);
        properFrames = properFrames && ProperFrames(after[i].desc);
        if (i > 0) {
            PomadeModel::TubeRecord record;
            childRootsConform = childRootsConform &&
                model.GetTubeRecord(affected[i], &record) &&
                SameRootRingWorld(record.actual, record.derived);
        }
    }
    // Reposition now conforms section zero to the edited support boundary.
    // The root-to-first-upper span is therefore intentionally not one rigid
    // K5 transform. The L1 base must attach exactly, while its upper endpoint
    // still carries the support-frame pose without a roll/shear.
    size_t const rootLastRing = before[0].positions.size() / 3 -
                                size_t(before[0].desc.ringVerts);
    bool const rootConforms = capturedAfter &&
        RootMatchesRegionLoop(model, region, after[0].desc,
                              /*requireEverySlotOnBoundary=*/true);
    bool const rootUpperRigid = capturedAfter &&
        SameRigidRootFrameTransform(before[0], after[0], rootLastRing);
    std::vector<bool> childUpperRigid(affected.size() - 1, false);
    std::vector<float> childUpperError(affected.size() - 1, -1.0f);
    bool upperRingsRigid = rootUpperRigid;
    std::vector<PomadeFrame> oldSupportFrames, newSupportFrames;
    std::string supportFrameError;
    bool const haveSupportPose = capturedAfter &&
        PomadeTubeFramesCpu(before[0].desc, &oldSupportFrames,
                           &supportFrameError) &&
        PomadeTubeFramesCpu(after[0].desc, &newSupportFrames,
                           &supportFrameError) &&
        !oldSupportFrames.empty() && !newSupportFrames.empty();
    float const oldSupportPivot[3] = {before[0].desc.centerX[0],
                                      before[0].desc.centerY[0],
                                      before[0].desc.centerZ[0]};
    float const newSupportPivot[3] = {after[0].desc.centerX[0],
                                      after[0].desc.centerY[0],
                                      after[0].desc.centerZ[0]};
    for (size_t i = 1; i < affected.size(); ++i) {
        size_t const lastRing = before[i].positions.size() / 3 -
                                size_t(before[i].desc.ringVerts);
        childUpperRigid[i - 1] = haveSupportPose &&
            SameRigidFrameTransform(before[i], after[i],
                                    oldSupportFrames.front(),
                                    newSupportFrames.front(), oldSupportPivot,
                                    newSupportPivot, lastRing, 3e-4f,
                                    &childUpperError[i - 1]);
        upperRingsRigid = upperRingsRigid && childUpperRigid[i - 1];
    }
    std::printf("info: curved transport conformance=%d childRoots=%d frames=%d rootUpper=%d"
                " childUpper0=%d(error=%g) childUpper1=%d(error=%g)\n",
                rootConforms ? 1 : 0, childRootsConform ? 1 : 0,
                properFrames ? 1 : 0,
                rootUpperRigid ? 1 : 0,
                childUpperRigid.size() > 0 && childUpperRigid[0] ? 1 : 0,
                childUpperError.size() > 0 ? childUpperError[0] : -1.0f,
                childUpperRigid.size() > 1 && childUpperRigid[1] ? 1 : 0,
                childUpperError.size() > 1 ? childUpperError[1] : -1.0f);
    Check(rootConforms && childRootsConform && upperRingsRigid && properFrames,
          "curved transport: conformed bases retain every root and sculpted child terminal ring under the support pose without a frame flip");
    PomadeModel::TubeRecord afterTransport, siblingTransport, afterParentEdit,
        siblingParentEdit;
    PomadeTubeDesc expectedChild, expectedSibling;
    PomadeShapeDeltas rebasedDeltas, siblingRebasedDeltas;
    TubeGeometry expectedGeometry, actualGeometry, expectedSiblingGeometry,
        actualSiblingGeometry;
    std::string rebaseError;
    bool const gotTransport = model.GetTubeRecord(children[0], &afterTransport) &&
        model.GetTubeRecord(children[1], &siblingTransport);
    bool const transportDeltas = gotTransport &&
        !SameDeltas(sculptRecord.deltas, afterTransport.deltas) &&
        HasNonRootSculpt(afterTransport);
    bool const parentMoved = model.MoveTubeCenterCV(root, 1, 0.03f, -0.02f, 0.01f);
    bool const gotParent = parentMoved &&
        model.GetTubeRecord(children[0], &afterParentEdit) &&
        model.GetTubeRecord(children[1], &siblingParentEdit);
    bool const applied = gotParent && PomadeHierarchicalSculptApplyCpu(
        afterParentEdit.derived, afterTransport.actual,
        afterTransport.derived, afterTransport.deltas,
        afterTransport.lockChildren, /*preserveLength=*/true,
        &expectedChild, &rebasedDeltas, &rebaseError);
    bool const expectedCaptured = applied &&
        CaptureDescGeometry(expectedChild, &expectedGeometry);
    bool const actualCaptured = gotParent &&
        CaptureTubeGeometry(model, children[0], &actualGeometry);
    bool const meshMatches = expectedCaptured && actualCaptured &&
        SameTubeGeometry(expectedGeometry, actualGeometry);
    bool const deltaMatches = applied && gotParent &&
        SameDeltas(rebasedDeltas, afterParentEdit.deltas);
    bool const siblingApplied = gotParent && PomadeHierarchicalSculptApplyCpu(
        siblingParentEdit.derived, siblingTransport.actual,
        siblingTransport.derived, siblingTransport.deltas,
        siblingTransport.lockChildren, /*preserveLength=*/true,
        &expectedSibling, &siblingRebasedDeltas, &rebaseError);
    bool const siblingCaptured = siblingApplied &&
        CaptureDescGeometry(expectedSibling, &expectedSiblingGeometry) &&
        CaptureTubeGeometry(model, children[1], &actualSiblingGeometry);
    bool const siblingMatches = siblingCaptured &&
        SameTubeGeometry(expectedSiblingGeometry, actualSiblingGeometry) &&
        SameDeltas(siblingRebasedDeltas, siblingParentEdit.deltas);
    Check(gotTransport, "curved transport: transported child record reads");
    Check(transportDeltas,
          "curved transport: attachment refit rebases deltas while retaining non-root sculpt");
    Check(parentMoved && gotParent, "curved transport: later parent edit re-derives the child");
    Check(applied, "curved transport: expected K6 residual applies");
    Check(expectedCaptured && actualCaptured,
          "curved transport: expected and actual refined child meshes capture");
    Check(meshMatches,
          "curved transport: later parent edit re-applies the transported sculpt in world geometry");
    Check(deltaMatches,
          "curved transport: later parent edit stores the expected rebased residual");
    Check(siblingMatches,
          "curved transport: later parent K6 keeps the unsculpted sibling free of transported child residuals");
}

void CheckRegionTransportedSubtree()
{
    using namespace usdGenPomade;
    // Two disconnected flat support faces make the affected/unaffected
    // distinction unambiguous.  Moving every node of the left loop by +U is
    // a pure world translation: every K5 point, including the interpolated
    // spans between section knots, must carry that same delta.
    std::vector<float> const points = {
        0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
        3, 0, 0, 4, 0, 0, 4, 1, 0, 3, 1, 0,
    };
    std::vector<int> const counts = {4, 4};
    std::vector<int> const indices = {0, 1, 2, 3, 4, 5, 6, 7};
    float const normal[3] = {0, 0, 1};
    PomadeModel model;
    std::vector<int> leftNodes, rightNodes;
    auto addLoop = [&](int face, std::vector<int> *out) {
        float const uv[8] = {0.20f, 0.20f, 0.80f, 0.20f,
                             0.80f, 0.80f, 0.20f, 0.80f};
        for (int i = 0; i < 4; ++i) {
            int const id = model.GraphAddNode(
                HitFace(*model.GetScalp(), face, uv[i * 2], uv[i * 2 + 1],
                        normal));
            if (id < 0) {
                return false;
            }
            out->push_back(id);
        }
        for (int i = 0; i < 4; ++i) {
            if (model.GraphConnect((*out)[i], (*out)[(i + 1) % 4]) < 0) {
                return false;
            }
        }
        return true;
    };
    bool setup = model.BindScalp(points, counts, indices) &&
                 addLoop(0, &leftNodes) && addLoop(1, &rightNodes) &&
                 model.Rasterise();
    int const leftRegion = setup ? model.RegionAtSurface(0, 0.5f, 0.5f) : -1;
    int const rightRegion = setup ? model.RegionAtSurface(1, 0.5f, 0.5f) : -1;
    setup = setup && leftRegion >= 0 && rightRegion >= 0 &&
            model.BuildTubeFromRegion(leftRegion, 5, 8, 2.0f) &&
            model.BuildTubeFromRegion(rightRegion, 5, 8, 2.0f);
    int const root = setup ? model.TubeForRegion(leftRegion) : -1;
    int const unrelated = setup ? model.TubeForRegion(rightRegion) : -1;
    std::vector<int> children, grandchildren;
    setup = setup && root >= 0 && unrelated >= 0 &&
            model.SubdivideTube(root, 2, "kmeans", 77, &children) &&
            children.size() == 2 &&
            model.SubdivideTube(children[0], 2, "kmeans", 91, &grandchildren) &&
            grandchildren.size() == 2;
    Check(setup, "region transport: two attached roots and a nested subtree build");
    if (!setup) {
        return;
    }
    int const sculpted = grandchildren[0];
    Check(model.MoveTubeCenterCV(sculpted, 2, 0.19f, -0.13f, 0.08f) &&
              model.MoveTubeSectionCV(sculpted, 2, 1, 0.11f, -0.07f),
          "region transport: nested child receives off-axis center and section sculpt");
    std::vector<int> affected = {root, children[0], children[1],
                                 grandchildren[0], grandchildren[1]};
    std::vector<TubeGeometry> before(affected.size()), after(affected.size());
    TubeGeometry unrelatedBefore, unrelatedAfter;
    bool captured = CaptureTubeGeometry(model, unrelated, &unrelatedBefore);
    for (size_t i = 0; i < affected.size(); ++i) {
        captured = captured && CaptureTubeGeometry(model, affected[i], &before[i]);
    }
    Check(captured, "region transport: captures full K5 geometry before graph move");
    if (!captured) {
        return;
    }
    std::vector<PomadeHit> targets;
    for (int node : leftNodes) {
        PomadeGraphNode current;
        if (!model.GraphGetNode(node, &current)) {
            return;
        }
        targets.push_back(HitFace(*model.GetScalp(), current.faceId,
                                  current.u + 0.10f, current.v, normal));
    }
    Check(model.GraphMoveNodes(leftNodes, targets) && model.Rasterise(),
          "region transport: graph release rigidly repositions the attached subtree");
    bool rigid = true;
    for (size_t i = 0; i < affected.size(); ++i) {
        rigid = rigid && CaptureTubeGeometry(model, affected[i], &after[i]) &&
                SameRigidTranslation(before[i], after[i]);
    }
    Check(rigid,
          "region transport: every root, sculpted child and nested descendant K5 mesh follows one rigid delta");
    Check(CaptureTubeGeometry(model, unrelated, &unrelatedAfter) &&
              SameTubeGeometry(unrelatedBefore, unrelatedAfter),
          "region transport: unrelated attached region stays bit-exact");

    // A collapsed drag sample is rejected atomically: it may not leave the
    // graph moved while a descendant remains at its old world location.
    std::vector<PomadeHit> invalid(leftNodes.size(),
        HitFace(*model.GetScalp(), 0, 0.5f, 0.5f, normal));
    std::vector<TubeGeometry> beforeInvalid = after;
    Check(!model.GraphMoveNodes(leftNodes, invalid),
          "region transport: topology-collapsing graph sample is rejected");
    bool invalidExact = true;
    for (size_t i = 0; i < affected.size(); ++i) {
        TubeGeometry now;
        invalidExact = invalidExact && CaptureTubeGeometry(model, affected[i], &now) &&
                       SameTubeGeometry(beforeInvalid[i], now);
    }
    Check(invalidExact,
          "region transport: rejected sample leaves the entire subtree exact");

    Check(model.Undo(), "region transport: graph and subtree motion undo together");
    bool undoExact = true;
    for (size_t i = 0; i < affected.size(); ++i) {
        TubeGeometry now;
        undoExact = undoExact && CaptureTubeGeometry(model, affected[i], &now) &&
                    SameTubeGeometry(before[i], now);
    }
    Check(undoExact, "region transport: undo restores sculpted nested world geometry");
    Check(model.Redo(nullptr), "region transport: graph and subtree motion redo together");
    bool redoRigid = true;
    for (size_t i = 0; i < affected.size(); ++i) {
        TubeGeometry now;
        redoRigid = redoRigid && CaptureTubeGeometry(model, affected[i], &now) &&
                    SameTubeGeometry(after[i], now);
    }
    Check(redoRigid, "region transport: redo restores every transported K5 mesh");
    std::vector<PomadeHit> cancelTargets;
    for (int node : leftNodes) {
        PomadeGraphNode current;
        bool const got = model.GraphGetNode(node, &current);
        if (!got) {
            cancelTargets.clear();
            break;
        }
        cancelTargets.push_back(HitFace(*model.GetScalp(), current.faceId,
                                        current.u + 0.02f, current.v, normal));
    }
    uint32_t cancelDirty = 0;
    Check(cancelTargets.size() == leftNodes.size() &&
              model.BeginGesture("region transport cancel") &&
              model.GraphMoveNodes(leftNodes, cancelTargets) &&
              model.Rasterise() && model.CancelGesture(&cancelDirty),
          "region transport: an in-flight graph/subtree transport cancels atomically");
    bool cancelExact = true;
    for (size_t i = 0; i < affected.size(); ++i) {
        TubeGeometry now;
        cancelExact = cancelExact && CaptureTubeGeometry(model, affected[i], &now) &&
                      SameTubeGeometry(after[i], now);
    }
    Check(cancelExact,
          "region transport: cancel restores every descendant K5 mesh exactly");
}

// Review 2026-09-24 (attachment-refresh root alignment): K14 starts a
// child's root ring on whichever parent corner first falls in the clip, so a
// refreshed root can arrive rotated against the child's transported upper
// sections. The refresh used to renumber only the ROOT to match them and
// store the rotation as a section residual: an unsculpted child turned
// "sculpted", and a later K6 re-split re-applied that full-radius residual
// slot by slot. It must instead bring the upper sections into the fresh
// split's slot order, so a child whose numbering was rotated before the
// move converges to exactly the un-rotated control: same actual, same
// derived, same (sculpt-sized) residual -- and a later parent edit agrees.
void CheckAttachmentRefreshKeepsFreshSlotOrder()
{
    using namespace usdGenPomade;
    std::vector<float> const points = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    std::vector<int> const counts = {4};
    std::vector<int> const indices = {0, 1, 2, 3};
    float const normal[3] = {0, 0, 1};
    struct Rig {
        PomadeModel model;
        std::vector<int> nodes;
        int root = -1;
        std::vector<int> kids;
    };
    auto build = [&](Rig *rig) {
        PomadeModel &model = rig->model;
        if (!model.BindScalp(points, counts, indices)) {
            return false;
        }
        float const uv[8] = {0.15f, 0.20f, 0.85f, 0.20f,
                             0.80f, 0.55f, 0.20f, 0.60f};
        for (int i = 0; i < 4; ++i) {
            int const node = model.GraphAddNode(
                HitFace(*model.GetScalp(), 0, uv[i * 2], uv[i * 2 + 1],
                        normal));
            if (node < 0) {
                return false;
            }
            rig->nodes.push_back(node);
        }
        for (int i = 0; i < 4; ++i) {
            if (model.GraphConnect(rig->nodes[size_t(i)],
                                   rig->nodes[size_t((i + 1) % 4)]) < 0) {
                return false;
            }
        }
        if (!model.Rasterise()) {
            return false;
        }
        int const region = model.RegionAtSurface(0, 0.5f, 0.4f);
        if (region < 0 || !model.BuildTubeFromRegion(region, 5, 8, 2.0f)) {
            return false;
        }
        rig->root = model.TubeForRegion(region);
        return rig->root >= 0 &&
               model.SubdivideTube(rig->root, 2, "kmeans", 131,
                                   &rig->kids) &&
               rig->kids.size() == 2;
    };
    auto move = [&](Rig *rig) {
        // One corner only: the region boundary itself changes, so the
        // refresh re-derives every child root (attachmentChanged).
        PomadeGraphNode current;
        if (!rig->model.GraphGetNode(rig->nodes[1], &current)) {
            return false;
        }
        std::vector<int> const moved = {rig->nodes[1]};
        std::vector<PomadeHit> const targets = {
            HitFace(*rig->model.GetScalp(), current.faceId,
                    current.u + 0.06f, current.v + 0.05f, normal)};
        return rig->model.GraphMoveNodes(moved, targets) &&
               rig->model.Rasterise();
    };
    auto closeTo = [](std::vector<float> const &a, std::vector<float> const &b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (!(std::fabs(a[i] - b[i]) <= 1e-5f)) {
                return false;
            }
        }
        return true;
    };
    auto sameSections = [&](std::vector<PomadeTubeSection> const &a,
                            std::vector<PomadeTubeSection> const &b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t s = 0; s < a.size(); ++s) {
            if (!closeTo(a[s].u, b[s].u) || !closeTo(a[s].v, b[s].v)) {
                return false;
            }
        }
        return true;
    };
    auto maxSectionDelta = [](PomadeShapeDeltas const &d) {
        float worst = 0.0f;
        for (PomadeTubeSection const &sec : d.sections) {
            for (float x : sec.u) worst = std::max(worst, std::fabs(x));
            for (float x : sec.v) worst = std::max(worst, std::fabs(x));
        }
        return worst;
    };

    Rig control, rotated;
    bool const setup = build(&control) && build(&rotated);
    Check(setup, "slot order: two identical subdivided regions build");
    if (!setup) {
        return;
    }
    // Rotate every section of one unsculpted child by one slot, bindings
    // along: the same surface, numbered as if an older K14 had started its
    // ring one corner later.
    int const kid = rotated.kids[0];
    PomadeModel::TubeRecord record;
    bool relabelled = rotated.model.GetTubeRecord(kid, &record) &&
                      record.actual.ringVerts >= 4;
    if (relabelled) {
        int const n = record.actual.ringVerts;
        for (PomadeTubeSection &sec : record.actual.sections) {
            PomadeTubeSection const old = sec;
            for (int i = 0; i < n; ++i) {
                sec.u[size_t(i)] = old.u[size_t((i + 1) % n)];
                sec.v[size_t(i)] = old.v[size_t((i + 1) % n)];
            }
        }
        for (PomadeParentBoundaryBinding &b :
             record.actual.inheritedBoundaryBindings) {
            b.childSlot = (b.childSlot + n - 1) % n;
        }
        record.hasInheritedBoundaryBindings = true;
        relabelled = rotated.model.RestoreTubeRecord(kid, record);
    }
    Check(relabelled, "slot order: one child is renumbered by one slot");
    Check(move(&control) && move(&rotated),
          "slot order: moving one region corner refreshes both hierarchies");

    PomadeModel::TubeRecord want, got;
    bool const read = control.model.GetTubeRecord(control.kids[0], &want) &&
                      rotated.model.GetTubeRecord(kid, &got);
    std::printf("info: slot order max section residual control=%g "
                "rotated=%g\n",
                read ? maxSectionDelta(want.deltas) : -1.0f,
                read ? maxSectionDelta(got.deltas) : -1.0f);
    Check(read && sameSections(got.actual.sections, want.actual.sections) &&
              sameSections(got.derived.sections, want.derived.sections),
          "slot order: the renumbered child converges to the control's "
          "slot order at every section");
    Check(read && sameSections(got.deltas.sections, want.deltas.sections),
          "slot order: and stores the control's residual, not a "
          "rotation-sized one");

    // A later parent edit re-derives both children with K6.
    bool const edited =
        control.model.MoveTubeCenterCV(control.root, 2, 0.02f, -0.01f,
                                       0.01f) &&
        rotated.model.MoveTubeCenterCV(rotated.root, 2, 0.02f, -0.01f,
                                       0.01f);
    Check(edited, "slot order: a parent edit after the refresh is accepted");
    PomadeModel::TubeRecord want2, got2;
    Check(edited && control.model.GetTubeRecord(control.kids[0], &want2) &&
              rotated.model.GetTubeRecord(kid, &got2) &&
              sameSections(got2.actual.sections, want2.actual.sections) &&
              sameSections(got2.deltas.sections, want2.deltas.sections),
          "slot order: and K6 re-derives the renumbered child exactly as "
          "the control");
}

} // namespace

int main()
{
    CheckRegionBoundaryRootSections();
    CheckPinnedRegionRoot();
    CheckCurvedRootAttachment();
    CheckTubeZeroEdgeSplitPropagation();
    CheckInternalChildCvKeepsCurvedParentExact();
    CheckAutoRegionSculptPropagation();
    CheckSubdividedChildIsolation();
    CheckRegionTransportedSubtree();
    CheckCurvedRegionFrameTransport();
    CheckGeneratedCurveClear();
    CheckAttachmentRefreshKeepsFreshSlotOrder();
    std::printf("testUsdGenPomadeRootFrame: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
