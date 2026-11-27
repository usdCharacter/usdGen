// usdGenPomade — the asynchronous commit pipeline (plan/17 section 3, P1).
#include "usdGenPomade/pomadeCommit.h"
#include "usdGenUsd/convert/math.h"
#include "usdGenPomade/pomadeModelState.h"
#include "usdGenPomade/pomadeBake.h"

// The cook the stage swap starts lives on the description's imaging
// session; cancelling it is the plan/17 §3.2 rule 2 half of G7.
#include "usdGenImaging/usdGenImagingSession.h"

#include "usdGen/parkingWord.h"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec2i.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/value.h"
#include "pxr/usd/sdf/attributeSpec.h"
#include "pxr/usd/sdf/changeBlock.h"
#include "pxr/usd/sdf/copyUtils.h"
#include "pxr/usd/sdf/listOp.h"
#include "pxr/usd/sdf/primSpec.h"
#include "pxr/usd/sdf/propertySpec.h"
#include "pxr/usd/sdf/relationshipSpec.h"
#include "pxr/usd/sdf/schema.h"
#include "pxr/usd/sdf/types.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/relationship.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/subset.h"
#include "pxr/usd/usdGeom/tokens.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <limits>
#include <stdexcept>
#include <utility>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenPomade {

std::string
PomadeCommitPaths::TubeName(int tubeId)
{
    char buf[32];
    // '-' is not a legal prim-name character, so the negative ids the
    // on-the-fly parents mint spell as group<n>.
    std::snprintf(buf, sizeof(buf), tubeId < 0 ? "group%d" : "tube%d",
                  tubeId < 0 ? -tubeId : tubeId);
    return buf;
}

SdfPath
PomadeCommitPaths::TubePath(int tubeId) const
{
    return TubesPath().AppendChild(TfToken(TubeName(tubeId)));
}

std::map<int, SdfPath>
PomadeTubePaths(PomadeSnapshot const &snapshot, PomadeCommitPaths const &paths)
{
    std::map<int, SdfPath> out;
    std::map<int, int> parentOf;
    for (PomadeSnapshotTube const &entry : snapshot.tubes) {
        parentOf[entry.tubeId] = entry.parentTubeId;
    }
    // Entries arrive parents-first, so one pass resolves every path. -1 is
    // the "no parent" sentinel, never a reference to the on-the-fly parent
    // that happens to carry id -1, so a root stays a root.
    for (PomadeSnapshotTube const &entry : snapshot.tubes) {
        auto const parent = entry.parentTubeId == -1
                                ? out.end()
                                : out.find(entry.parentTubeId);
        out[entry.tubeId] =
            parent == out.end()
                ? paths.TubePath(entry.tubeId)
                : parent->second.AppendChild(
                      TfToken(PomadeCommitPaths::TubeName(entry.tubeId)));
    }
    return out;
}

SdfPath
PomadeCommitPaths::LevelMapPath(int level) const
{
    if (level <= 1) {
        return RegionMapPath();
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "RegionMapL%d", level);
    return groomPath.AppendChild(TfToken(buf));
}

SdfPath
PomadeCommitPaths::LevelExprPath(int level) const
{
    if (level <= 1) {
        return RegionExprPath();
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "RegionExprL%d", level);
    return groomPath.AppendChild(TfToken(buf));
}

SdfPath
PomadeCommitPaths::InterpOpPath() const
{
    // No description bound — "Bind scalp" on a stage with no usdGen
    // description, which is most of the T3 scripts — means there is no
    // op to fill in. Answer with an empty path: appending to the empty
    // one is an Sdf coding error, and every enqueue raised two of them
    // (here, then again from EnqueuePlan's empty-path retry). On the
    // main thread those are posted TfErrors that usdview's paintGL
    // decorator re-raises. Callers already treat an empty opPath as
    // "nothing to author": PomadePlanGuideInterpolateFill returns early
    // on the same condition without setting createInterpOp.
    if (descriptionPath.IsEmpty()) {
        return SdfPath();
    }
    // NOTE (chain order): v1 chains execute in reverse namespace order, so
    // this name also fixes the op's chain position. The fill-in never moves
    // an artist's stack; arranging operators around the pomade op is the
    // Output panel's job (P5), not the committer's.
    return descriptionPath.AppendChild(TfToken("Ops"))
        .AppendChild(TfToken("pomadeInterp"));
}

namespace {

// -- Sdf authoring helpers (worker-safe: layer-local, no stage) --------------

SdfPrimSpecHandle
_EnsurePrim(SdfLayerHandle const &layer, SdfPath const &path,
            SdfSpecifier spec, char const *typeName)
{
    SdfPrimSpecHandle prim = SdfCreatePrimInLayer(layer, path);
    if (!prim) {
        return SdfPrimSpecHandle();
    }
    prim->SetSpecifier(spec);
    if (typeName) {
        prim->SetTypeName(typeName);
    }
    return prim;
}

template <class T>
bool
_SetAttr(SdfPrimSpecHandle const &prim, char const *name,
         SdfValueTypeName const &typeName, T const &value,
         SdfVariability variability = SdfVariabilityVarying,
         bool custom = false)
{
    SdfAttributeSpecHandle attr =
        SdfAttributeSpec::New(prim, name, typeName, variability, custom);
    if (!attr) {
        return false;
    }
    static_assert(!usdGenUsd::kIsEngineMath<T>, "convert engine math with usdGenUsd::ToHost");
    attr->SetDefaultValue(VtValue(value));
    return true;
}

bool
_SetRelTargets(SdfPrimSpecHandle const &prim, char const *name,
               SdfPathVector const &targets)
{
    SdfRelationshipSpecHandle rel =
        SdfRelationshipSpec::New(prim, name, /*custom*/ false);
    if (!rel) {
        return false;
    }
    SdfTargetsProxy proxies = rel->GetTargetPathList();
    proxies.ClearEditsAndMakeExplicit();
    proxies.GetExplicitItems() = targets;
    return true;
}

bool
_SetConnections(SdfPrimSpecHandle const &prim, char const *name,
                SdfValueTypeName const &typeName, SdfPathVector const &conns)
{
    SdfAttributeSpecHandle attr =
        SdfAttributeSpec::New(prim, name, typeName);
    if (!attr) {
        return false;
    }
    SdfConnectionsProxy proxies = attr->GetConnectionPathList();
    proxies.ClearEditsAndMakeExplicit();
    proxies.GetExplicitItems() = conns;
    return true;
}

bool
_PrependApiSchema(SdfPrimSpecHandle const &prim, char const *schema)
{
    SdfTokenListOp listOp;
    std::string err;
    if (!listOp.SetPrependedItems(TfTokenVector{TfToken(schema)}, &err)) {
        return false;
    }
    prim->SetInfo(TfToken("apiSchemas"), VtValue(listOp));
    return true;
}

bool
_SetInterpolation(SdfPrimSpecHandle const &prim, char const *attrName,
                  char const *interp)
{
    for (SdfAttributeSpecHandle const &attr : prim->GetAttributes()) {
        if (attr->GetName() == attrName) {
            attr->SetInfo(TfToken("interpolation"), VtValue(TfToken(interp)));
            return true;
        }
    }
    return false;
}

// C++ spellings of the relationship/connection getters (out-params, unlike
// the python GetTargets()/GetConnections()).
SdfPathVector
_Targets(UsdRelationship const &rel)
{
    SdfPathVector targets;
    rel.GetTargets(&targets);
    return targets;
}

SdfPathVector
_Connections(UsdAttribute const &attr)
{
    SdfPathVector conns;
    attr.GetConnections(&conns);
    return conns;
}

// The deepest tube whose cell contains `p`, walking down from tube 0 with
// the same K14 partition the map bake uses. Bridge imports are not cells
// (they are not part of a subdivision), so a foreign guide always lands
// under a derived tube.
int
_OwningLeafTube(PomadeModel const &model, float const p[3])
{
    int current = 0;
    for (;;) {
        std::vector<int> cells;
        for (int kid : model.GetTubeChildren(current)) {
            if (!model.IsTubeImported(kid)) {
                cells.push_back(kid);
            }
        }
        if (cells.empty()) {
            return current;
        }
        PomadeTubeDesc parent;
        if (!model.GetTubeDesc(current, &parent) ||
            parent.centerX.size() < 2) {
            return current;
        }
        std::vector<PomadeFrame> frames;
        std::string err;
        if (!PomadeTubeFramesCpu(parent, &frames, &err) || frames.empty()) {
            return current;
        }
        // Ascending id IS ascending childIndex (id = parent * 16 + 1 + k).
        std::vector<float> centers;
        for (int kid : cells) {
            PomadeTubeDesc child;
            if (!model.GetTubeDesc(kid, &child) || child.centerX.empty()) {
                return current;
            }
            centers.push_back(child.centerX[0]);
            centers.push_back(child.centerY[0]);
            centers.push_back(child.centerZ[0]);
        }
        float const root[3] = {parent.centerX[0], parent.centerY[0],
                               parent.centerZ[0]};
        int const cell = PomadeOwningChildCell(root, frames[0], centers.data(),
                                              int(cells.size()), p);
        if (cell < 0) {
            return current;
        }
        current = cells[size_t(cell)];
    }
}

// Hand every mesh-fill face down the hierarchy to the leaf that owns it.
// The snapshot is ordered parents-first, so one forward pass cascades from
// L1 to the deepest level.
void
_PartitionRegionFaces(PomadeSnapshot *snapshot)
{
    for (size_t i = 0; i < snapshot->tubes.size(); ++i) {
        PomadeSnapshotTube &parent = snapshot->tubes[i];
        if (parent.regionFaces.empty()) {
            continue;
        }
        std::vector<size_t> cells;
        for (size_t j = 0; j < snapshot->tubes.size(); ++j) {
            PomadeSnapshotTube const &child = snapshot->tubes[j];
            if (child.parentTubeId == parent.tubeId && !child.imported &&
                !child.tube.centerX.empty()) {
                cells.push_back(j);
            }
        }
        if (cells.empty()) {
            continue;
        }
        std::sort(cells.begin(), cells.end(), [&](size_t a, size_t b) {
            return snapshot->tubes[a].childIndex <
                   snapshot->tubes[b].childIndex;
        });
        PomadeTubeDesc const pd = PomadeTubeDescFromSnapshot(parent.tube);
        std::vector<PomadeFrame> frames;
        std::string err;
        if (pd.centerX.size() < 2 ||
            !PomadeTubeFramesCpu(pd, &frames, &err) || frames.empty()) {
            continue;
        }
        float const rootCenter[3] = {pd.centerX[0], pd.centerY[0],
                                     pd.centerZ[0]};
        std::vector<float> childCenters;
        childCenters.reserve(cells.size() * 3);
        for (size_t j : cells) {
            childCenters.push_back(snapshot->tubes[j].tube.centerX[0]);
            childCenters.push_back(snapshot->tubes[j].tube.centerY[0]);
            childCenters.push_back(snapshot->tubes[j].tube.centerZ[0]);
        }
        std::vector<int> const faces = parent.regionFaces;
        parent.regionFaces.clear();
        for (int f : faces) {
            if (size_t(f) * 3 + 2 >= snapshot->scalp.faceCentroids.size()) {
                continue;
            }
            int const cell = PomadeOwningChildCell(
                rootCenter, frames[0], childCenters.data(), int(cells.size()),
                &snapshot->scalp.faceCentroids[size_t(f) * 3]);
            if (cell >= 0) {
                snapshot->tubes[cells[size_t(cell)]].regionFaces.push_back(f);
            }
        }
    }
}

// -- V6 (plan/18 §7 G6): the per-version dirty set --------------------------
//
// FNV-1a over the bytes the guide fill reads. Floats go in by their bit
// pattern, not their value: the fill is bit-exact, so the hash has to be
// too (and -0.0f must not collide with 0.0f, which is exactly the case a
// value comparison would miss after a drag returns a CV to its start).
uint64_t constexpr kHashSeed = 1469598103934665603ull;

uint64_t _HashBytes(uint64_t h, void const *data, size_t bytes)
{
    auto const *p = static_cast<unsigned char const *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        h ^= uint64_t(p[i]);
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t _HashInt(uint64_t h, int64_t v)
{
    return _HashBytes(h, &v, sizeof(v));
}

uint64_t _HashFloat(uint64_t h, float v)
{
    return _HashBytes(h, &v, sizeof(v));
}

uint64_t _HashFloats(uint64_t h, std::vector<float> const &v)
{
    h = _HashInt(h, int64_t(v.size()));
    return v.empty() ? h : _HashBytes(h, v.data(), v.size() * sizeof(float));
}

uint64_t _HashFillParams(uint64_t h, PomadeModel::FillParams const &fill)
{
    h = _HashFloat(h, fill.density);
    h = _HashInt(h, fill.cvCount);
    h = _HashInt(h, fill.seed);
    h = _HashFloat(h, fill.edgeBias);
    h = _HashFloats(h, fill.lengthProfile);
    h = _HashInt(h, int(fill.sampler));
    return h;
}

// Everything PomadeGuidesFromSnapshot reads off one entry. The scalp is a
// shared input and is hashed once into the snapshot-wide salt instead.
uint64_t _HashSnapshotTube(PomadeSnapshotTube const &entry)
{
    uint64_t h = kHashSeed;
    h = _HashInt(h, entry.tubeId);
    h = _HashInt(h, entry.regionId);
    h = _HashInt(h, entry.level);
    h = _HashInt(h, entry.tube.hasTube ? 1 : 0);
    h = _HashInt(h, entry.fillSuspended ? 1 : 0);
    h = _HashInt(h, entry.imported ? 1 : 0);
    h = _HashInt(h, int64_t(entry.members.size()));
    h = _HashInt(h, entry.tube.shape.rings);
    h = _HashInt(h, entry.tube.shape.ringVerts);
    h = _HashFloat(h, entry.tube.shape.radius);
    h = _HashFloat(h, entry.tube.shape.length);
    h = _HashFloats(h, entry.tube.centerX);
    h = _HashFloats(h, entry.tube.centerY);
    h = _HashFloats(h, entry.tube.centerZ);
    h = _HashBytes(h, entry.tube.frameReference.data(),
                   entry.tube.frameReference.size() * sizeof(float));
    h = _HashInt(h, int64_t(entry.tube.sections.size()));
    for (PomadeTubeSection const &section : entry.tube.sections) {
        h = _HashFloat(h, section.t);
        h = _HashFloat(h, section.scale);
        h = _HashFloat(h, section.twist);
        h = _HashFloats(h, section.u);
        h = _HashFloats(h, section.v);
    }
    h = _HashFloats(h, entry.centerDeltas);
    h = _HashFloats(h, entry.sectionDeltas);
    h = _HashFloats(h, entry.sectionDeltaTransforms);
    h = _HashInt(h, int64_t(entry.inheritedBoundaryBindings.size()));
    if (!entry.inheritedBoundaryBindings.empty()) {
        h = _HashBytes(h, entry.inheritedBoundaryBindings.data(),
                       entry.inheritedBoundaryBindings.size() * sizeof(int));
    }
    h = _HashFillParams(h, entry.tube.fill);
    h = _HashInt(h, int64_t(entry.regionFaces.size()));
    if (!entry.regionFaces.empty()) {
        h = _HashBytes(h, entry.regionFaces.data(),
                       entry.regionFaces.size() * sizeof(int));
    }
    h = _HashFloats(h, entry.regionBoundary);
    h = _HashInt(h, entry.tube.rootFramePinned ? 1 : 0);
    h = _HashFloat(h, entry.tube.rootFrame.tx);
    h = _HashFloat(h, entry.tube.rootFrame.ty);
    h = _HashFloat(h, entry.tube.rootFrame.tz);
    h = _HashFloat(h, entry.tube.rootFrame.nx);
    h = _HashFloat(h, entry.tube.rootFrame.ny);
    h = _HashFloat(h, entry.tube.rootFrame.nz);
    h = _HashFloat(h, entry.tube.rootFrame.bx);
    h = _HashFloat(h, entry.tube.rootFrame.by);
    h = _HashFloat(h, entry.tube.rootFrame.bz);
    return h;
}

// Pomade Fill produces world-space points.  The owned Output source must not
// inherit a transform from a groom container that would apply that transform
// a second time.  This is the Sdf spelling of UsdGeomXformable's
// SetResetXformStack(true); no local op leaves the reset local matrix at I.
bool
_SetResetXformStack(SdfPrimSpecHandle const &prim)
{
    return _SetAttr(prim, "xformOpOrder", SdfValueTypeNames->TokenArray,
                    VtTokenArray{TfToken("!resetXformStack!")},
                    SdfVariabilityUniform);
}

// K10 guides carry N/B/T rows as float-derived doubles.  Those are also the
// physical C3 tangent/binormal/normal axes respectively: K4's radial N is
// the scalp tangent, its B is the binormal, and its tube-axis T is the scalp
// normal.  Re-orthonormalize only the owned output copy at double precision;
// existing Guides retain their historical K10 bytes.
bool
_OutputRootFrameFromK10(double const *src, GfMatrix4d *out)
{
    if (!src || !out) {
        return false;
    }
    for (int i = 0; i < 16; ++i) {
        if (!std::isfinite(src[i])) {
            return false;
        }
    }
    double tx = src[0], ty = src[1], tz = src[2];        // K10 N / C3 T
    double nx = src[8], ny = src[9], nz = src[10];       // K10 T / C3 N
    auto normalize = [](double *x, double *y, double *z) {
        double const length = std::sqrt(*x * *x + *y * *y + *z * *z);
        if (!(length > 1.0e-12) || !std::isfinite(length)) {
            return false;
        }
        *x /= length;
        *y /= length;
        *z /= length;
        return true;
    };
    if (!normalize(&tx, &ty, &tz)) {
        return false;
    }
    // Keep the physical normal, removing only K4 float round-off against T.
    double const normalAlongT = nx * tx + ny * ty + nz * tz;
    nx -= normalAlongT * tx;
    ny -= normalAlongT * ty;
    nz -= normalAlongT * tz;
    if (!normalize(&nx, &ny, &nz)) {
        return false;
    }
    // This restores K4's B: C3 needs T x B = N, so B is N x T.
    double bx = ny * tz - nz * ty;
    double by = nz * tx - nx * tz;
    double bz = nx * ty - ny * tx;
    if (!normalize(&bx, &by, &bz)) {
        return false;
    }
    // Recompute N from the output pair to make the authored determinant and
    // orthogonality meet C3's 1e-8 double validation tolerance.
    nx = ty * bz - tz * by;
    ny = tz * bx - tx * bz;
    nz = tx * by - ty * bx;
    if (!normalize(&nx, &ny, &nz)) {
        return false;
    }
    out->Set(tx, ty, tz, 0.0, bx, by, bz, 0.0, nx, ny, nz, 0.0,
             src[12], src[13], src[14], 1.0);
    return true;
}

std::array<float, 9>
_IdentityFrameReference()
{
    return {{1.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f,
             0.0f, 0.0f, 1.0f}};
}

bool
_IsProperFrameReference(std::array<float, 9> const &q)
{
    // Q is a row-major local-to-world rotation.  Keep the tolerance well
    // below visible frame error while accepting the normal float round-trip
    // through USD.  A reflection is not a frame reference: it reverses the
    // section winding and changes the groom's handedness.
    constexpr float eps = 2.0e-4f;
    for (float value : q) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    auto dot = [&](int a, int b) {
        return q[size_t(a) * 3] * q[size_t(b) * 3] +
               q[size_t(a) * 3 + 1] * q[size_t(b) * 3 + 1] +
               q[size_t(a) * 3 + 2] * q[size_t(b) * 3 + 2];
    };
    if (std::abs(dot(0, 0) - 1.0f) > eps ||
        std::abs(dot(1, 1) - 1.0f) > eps ||
        std::abs(dot(2, 2) - 1.0f) > eps ||
        std::abs(dot(0, 1)) > eps || std::abs(dot(0, 2)) > eps ||
        std::abs(dot(1, 2)) > eps) {
        return false;
    }
    float const determinant =
        q[0] * (q[4] * q[8] - q[5] * q[7]) -
        q[1] * (q[3] * q[8] - q[5] * q[6]) +
        q[2] * (q[3] * q[7] - q[4] * q[6]);
    return std::abs(determinant - 1.0f) <= eps;
}

// The shared fill input: the scalp copy the worker carries. One number for
// the whole mesh, so a re-bind or a re-rasterise drops every cached slice.
uint64_t _HashScalp(PomadeSnapshot const &snapshot)
{
    uint64_t h = kHashSeed;
    h = _HashInt(h, snapshot.hasScalp ? 1 : 0);
    h = _HashFloats(h, snapshot.scalp.points);
    h = _HashInt(h, int64_t(snapshot.scalp.faceVertexCounts.size()));
    if (!snapshot.scalp.faceVertexCounts.empty()) {
        h = _HashBytes(h, snapshot.scalp.faceVertexCounts.data(),
                       snapshot.scalp.faceVertexCounts.size() * sizeof(int));
    }
    h = _HashInt(h, int64_t(snapshot.scalp.faceVertexIndices.size()));
    if (!snapshot.scalp.faceVertexIndices.empty()) {
        h = _HashBytes(h, snapshot.scalp.faceVertexIndices.data(),
                       snapshot.scalp.faceVertexIndices.size() * sizeof(int));
    }
    // A face subset moves every region root it clips (plan/02 §2.20 rule
    // 5: a subset edit is a recapture). Whole-mesh binds hash exactly as
    // they always did.
    if (!snapshot.scalp.activeFaces.empty()) {
        h = _HashInt(h, int64_t(snapshot.scalp.activeFaces.size()));
        h = _HashBytes(h, snapshot.scalp.activeFaces.data(),
                       snapshot.scalp.activeFaces.size() * sizeof(int));
    }
    // Exact polygon support and child-cell ownership are derived from this
    // graph. Include it in the shared cache salt so an in-face graph drag,
    // link edit, or a sibling-cell move cannot reuse roots from yesterday's
    // classifier merely because the coarse scalp faces stayed unchanged.
    h = _HashInt(h, int64_t(snapshot.graph.nodes.size()));
    for (PomadeGraphNode const &node : snapshot.graph.nodes) {
        h = _HashInt(h, node.id);
        h = _HashFloat(h, node.p[0]);
        h = _HashFloat(h, node.p[1]);
        h = _HashFloat(h, node.p[2]);
    }
    h = _HashInt(h, int64_t(snapshot.graph.edges.size()));
    for (auto const &edge : snapshot.graph.edges) {
        h = _HashInt(h, edge.first);
        h = _HashInt(h, edge.second);
    }
    h = _HashInt(h, int64_t(snapshot.graph.linked.size()));
    for (auto const &link : snapshot.graph.linked) {
        h = _HashInt(h, link.first);
        h = _HashInt(h, link.second);
    }
    return h;
}

// The Output owner map is a generated asset, not a mutable live-map alias.
// Name it from exactly the snapshot data that determines categorical root
// ownership so a concurrent graph/hierarchy bake can never replace it under
// an already-authored Output description.  Output density and width are
// deliberately absent: they are runtime CurveSource/Width controls.
uint64_t
_OutputOwnerMapContentId(PomadeSnapshot const &snapshot)
{
    uint64_t h = _HashScalp(snapshot);
    h = _HashInt(h, int64_t(snapshot.graph.mapVersion));
    h = _HashInt(h, snapshot.outputPtexResolution);
    h = _HashInt(h, int64_t(snapshot.graph.nodes.size()));
    for (PomadeGraphNode const &node : snapshot.graph.nodes) {
        h = _HashInt(h, node.id);
        h = _HashInt(h, node.faceId);
        h = _HashFloat(h, node.u);
        h = _HashFloat(h, node.v);
    }
    h = _HashInt(h, int64_t(snapshot.graph.edges.size()));
    for (std::pair<int, int> const &edge : snapshot.graph.edges) {
        h = _HashInt(h, edge.first);
        h = _HashInt(h, edge.second);
    }
    h = _HashInt(h, int64_t(snapshot.tubes.size()));
    for (PomadeSnapshotTube const &entry : snapshot.tubes) {
        // Imported bridges are not K14 cells.  All other live tubes affect
        // a descendant owner walk through id/link/root-frame data.
        if (!entry.tube.hasTube || entry.imported || !entry.members.empty()) {
            continue;
        }
        h = _HashInt(h, entry.tubeId);
        h = _HashInt(h, entry.parentTubeId);
        h = _HashInt(h, entry.childIndex);
        h = _HashInt(h, entry.level);
        h = _HashInt(h, entry.regionId);
        h = _HashFloats(h, entry.tube.centerX);
        h = _HashFloats(h, entry.tube.centerY);
        h = _HashFloats(h, entry.tube.centerZ);
        h = _HashInt(h, entry.tube.rootFramePinned ? 1 : 0);
        h = _HashFloat(h, entry.tube.rootFrame.tx);
        h = _HashFloat(h, entry.tube.rootFrame.ty);
        h = _HashFloat(h, entry.tube.rootFrame.tz);
        h = _HashFloat(h, entry.tube.rootFrame.nx);
        h = _HashFloat(h, entry.tube.rootFrame.ny);
        h = _HashFloat(h, entry.tube.rootFrame.nz);
        h = _HashFloat(h, entry.tube.rootFrame.bx);
        h = _HashFloat(h, entry.tube.rootFrame.by);
        h = _HashFloat(h, entry.tube.rootFrame.bz);
        h = _HashBytes(h, entry.tube.frameReference.data(),
                       entry.tube.frameReference.size() * sizeof(float));
    }
    return h;
}

bool
_BuildOutputOwnerBakeInput(PomadeSnapshot const &snapshot,
                           PomadeBakeInput *out, std::string *err)
{
    if (!out || !snapshot.hasScalp || !snapshot.scalp.finalized) {
        if (err) *err = "Output requires a bound, finalised scalp";
        return false;
    }
    PomadeBakeInput input;
    input.scalp = std::make_shared<PomadeScalpMesh>(snapshot.scalp);
    if (!input.graph.Restore(*input.scalp, snapshot.graph.nodes,
                             snapshot.graph.edges, snapshot.graph.linked,
                             err)) {
        return false;
    }
    input.levelCount = 1;
    input.resOverride = snapshot.outputPtexResolution;
    input.outputOwnerMap = true;
    for (PomadeSnapshotTube const &entry : snapshot.tubes) {
        if (!entry.tube.hasTube || entry.imported || !entry.members.empty()) {
            continue;
        }
        PomadeTubeDesc const desc = PomadeTubeDescFromSnapshot(entry.tube);
        std::vector<PomadeFrame> frames;
        std::string frameErr;
        if (desc.centerX.size() < 2 ||
            !PomadeTubeFramesCpu(desc, &frames, &frameErr) || frames.empty()) {
            if (err) *err = "Output owner map: " + frameErr;
            return false;
        }
        PomadeBakeTube tube;
        tube.tubeId = entry.tubeId;
        tube.parentTubeId = entry.parentTubeId;
        tube.level = entry.level;
        tube.regionId = entry.regionId;
        tube.childIndex = entry.childIndex;
        tube.rootCenter[0] = desc.centerX[0];
        tube.rootCenter[1] = desc.centerY[0];
        tube.rootCenter[2] = desc.centerZ[0];
        tube.rootFrame = frames[0];
        input.tubes.push_back(tube);
    }
    *out = std::move(input);
    return true;
}

bool
_BakeOutputOwnerMap(PomadeSnapshot const &snapshot, uint64_t contentId,
                    std::string *outPath, std::string *err)
{
    if (!outPath) {
        if (err) *err = "Output owner map has no result path";
        return false;
    }
    PomadeBakeInput input;
    if (!_BuildOutputOwnerBakeInput(snapshot, &input, err)) {
        return false;
    }
    namespace fs = std::filesystem;
    fs::path directory;
    if (!snapshot.graph.bakedMapFile.empty()) {
        directory = fs::path(snapshot.graph.bakedMapFile).parent_path();
    }
    if (directory.empty()) {
        directory = fs::temp_directory_path() / "usdGenPomadeOutputMaps";
    }
    std::error_code ec;
    fs::create_directories(directory, ec);
    if (ec) {
        if (err) *err = "cannot create Output map directory " + directory.string();
        return false;
    }
    char name[96];
    std::snprintf(name, sizeof(name), "outputRegionMap.%016llx.ptx",
                  static_cast<unsigned long long>(contentId));
    fs::path const path = directory / name;
    if (!fs::is_regular_file(path, ec)) {
        std::vector<std::vector<float>> cache;
        std::string bakeErr;
        if (!PomadeBakePtex(input, path.string(), nullptr, &cache, nullptr,
                           &bakeErr)) {
            if (err) *err = bakeErr;
            return false;
        }
    }
    *outPath = path.string();
    return true;
}

std::vector<PomadeRootOwnershipCell>
_OwnershipCells(PomadeSnapshot const &snapshot, int tubeId)
{
    std::vector<PomadeRootOwnershipCell> chain;
    int childId = tubeId;
    for (;;) {
        PomadeSnapshotTube const *child = nullptr;
        for (PomadeSnapshotTube const &entry : snapshot.tubes) {
            if (entry.tubeId == childId) {
                child = &entry;
                break;
            }
        }
        if (!child || child->parentTubeId < 0) {
            break;
        }
        std::vector<PomadeSnapshotTube const *> children;
        PomadeSnapshotTube const *parent = nullptr;
        for (PomadeSnapshotTube const &entry : snapshot.tubes) {
            if (entry.tubeId == child->parentTubeId) {
                parent = &entry;
            }
            if (entry.parentTubeId == child->parentTubeId && !entry.imported &&
                !entry.tube.centerX.empty()) {
                children.push_back(&entry);
            }
        }
        if (!parent || children.empty()) {
            break;
        }
        std::sort(children.begin(), children.end(),
                  [](PomadeSnapshotTube const *a, PomadeSnapshotTube const *b) {
                      return a->childIndex != b->childIndex
                                 ? a->childIndex < b->childIndex
                                 : a->tubeId < b->tubeId;
                  });
        PomadeTubeDesc const parentDesc = PomadeTubeDescFromSnapshot(parent->tube);
        std::vector<PomadeFrame> frames;
        std::string err;
        if (parentDesc.centerX.size() < 2 ||
            !PomadeTubeFramesCpu(parentDesc, &frames, &err) || frames.empty()) {
            break;
        }
        PomadeRootOwnershipCell cell;
        cell.rootCenter[0] = parentDesc.centerX[0];
        cell.rootCenter[1] = parentDesc.centerY[0];
        cell.rootCenter[2] = parentDesc.centerZ[0];
        cell.frame = frames[0];
        for (size_t i = 0; i < children.size(); ++i) {
            PomadeTubeDesc const sibling =
                PomadeTubeDescFromSnapshot(children[i]->tube);
            cell.childCenters.push_back(sibling.centerX[0]);
            cell.childCenters.push_back(sibling.centerY[0]);
            cell.childCenters.push_back(sibling.centerZ[0]);
            if (children[i]->tubeId == childId) {
                cell.childIndex = int(i);
            }
        }
        if (cell.childIndex < 0) {
            break;
        }
        chain.push_back(std::move(cell));
        childId = child->parentTubeId;
    }
    return chain;
}

} // namespace

uint64_t
PomadeSnapshotTubeHash(PomadeSnapshotTube const &entry)
{
    return _HashSnapshotTube(entry);
}

void
PomadeHashSnapshot(PomadeSnapshot *snapshot)
{
    if (!snapshot) {
        return;
    }
    for (PomadeSnapshotTube &entry : snapshot->tubes) {
        uint64_t h = _HashSnapshotTube(entry);
        // A child root depends only on its own ancestor cell chain, not on
        // every tube in the groom. Keeping this in the per-entry hash lets a
        // one-tube edit retain unrelated cached guide slices.
        std::vector<PomadeRootOwnershipCell> const cells =
            snapshot->hasScalp && entry.regionId >= 0
                ? _OwnershipCells(*snapshot, entry.tubeId)
                : std::vector<PomadeRootOwnershipCell>();
        h = _HashInt(h, int64_t(cells.size()));
        for (PomadeRootOwnershipCell const &cell : cells) {
            h = _HashFloat(h, cell.rootCenter[0]);
            h = _HashFloat(h, cell.rootCenter[1]);
            h = _HashFloat(h, cell.rootCenter[2]);
            h = _HashFloat(h, cell.frame.tx);
            h = _HashFloat(h, cell.frame.ty);
            h = _HashFloat(h, cell.frame.tz);
            h = _HashFloat(h, cell.frame.nx);
            h = _HashFloat(h, cell.frame.ny);
            h = _HashFloat(h, cell.frame.nz);
            h = _HashFloat(h, cell.frame.bx);
            h = _HashFloat(h, cell.frame.by);
            h = _HashFloat(h, cell.frame.bz);
            h = _HashFloats(h, cell.childCenters);
            h = _HashInt(h, cell.childIndex);
        }
        entry.contentHash = h;
    }
}

PomadeGuideCache::Entry const *
PomadeGuideCache::Find(int tubeId, uint64_t contentHash) const
{
    auto const it = _entries.find(tubeId);
    if (it == _entries.end() || it->second.contentHash != contentHash) {
        return nullptr;
    }
    return &it->second;
}

void
PomadeGuideCache::Store(int tubeId, Entry entry)
{
    _entries[tubeId] = std::move(entry);
}

bool
PomadeGuideCache::NoteScalp(uint64_t scalpHash)
{
    if (scalpHash == _scalpHash) {
        return false;
    }
    _entries.clear();
    _scalpHash = scalpHash;
    return true;
}

size_t
PomadeGuideCache::Reap(std::vector<int> const &live)
{
    std::vector<int> sorted = live;
    std::sort(sorted.begin(), sorted.end());
    size_t dropped = 0;
    for (auto it = _entries.begin(); it != _entries.end();) {
        if (std::binary_search(sorted.begin(), sorted.end(), it->first)) {
            ++it;
        } else {
            it = _entries.erase(it);
            ++dropped;
        }
    }
    return dropped;
}

void
PomadeGuideCache::Clear()
{
    _entries.clear();
    _scalpHash = 0;
    _lastReused = 0;
    _lastRefilled = 0;
}

PomadeSnapshot
PomadeSnapshotFromState(PomadeModelState const &state)
{
    PomadeSnapshot snapshot;
    auto getRecord = [&](int id, PomadeModel::TubeRecord *out) {
        auto it = state.records.find(id);
        if (it == state.records.end()) {
            return false;
        }
        *out = it->second;
        return true;
    };
    auto copyBoundaryBindings = [](PomadeTubeDesc const &desc,
                                   PomadeSnapshotTube *entry) {
        if (!entry) {
            return;
        }
        entry->inheritedBoundaryBindings.clear();
        entry->inheritedBoundaryBindings.reserve(
            desc.inheritedBoundaryBindings.size() * 3);
        for (PomadeParentBoundaryBinding const &binding :
             desc.inheritedBoundaryBindings) {
            entry->inheritedBoundaryBindings.push_back(binding.section);
            entry->inheritedBoundaryBindings.push_back(binding.parentSlot);
            entry->inheritedBoundaryBindings.push_back(binding.childSlot);
        }
    };
    PomadeModel::TubeSnapshot tube = state.tube;
    snapshot.version = tube.version;
    snapshot.generatedCurvesSuppressed = state.generatedCurvesSuppressed;
    PomadeModel::OutputSettings const &output = state.output;
    snapshot.outputEnabled = output.enabled;
    snapshot.outputDensityMultiplier = output.densityMultiplier;
    snapshot.outputWidth = output.width;
    snapshot.outputPtexResolution = output.ptexResolution;
    snapshot.tubeRegionId = state.tubeRegionId;
    if (tube.hasTube) {
        PomadeSnapshotTube entry;
        entry.tubeId = 0;
        // The model's real rooting, -1 included: hydrate replays the
        // same disc-vs-mesh rule from the stored region id, so the two
        // can never disagree (unrooted tubes stay on the disc stream
        // even when a scalp is bound).
        entry.regionId = snapshot.tubeRegionId;
        entry.level = 1;
        entry.tube = std::move(tube);
        snapshot.tubes.push_back(std::move(entry));
    }
    // V6: every further L1 root (one per closed region, plan/18 §7 G14).
    // They are ordinary store tubes with no parent, emitted right after
    // tube 0 so the breadth-first walk below can seed on all of them.
    std::vector<int> const &l1Ids = state.l1TubeIds;
    for (int id : l1Ids) {
        if (id == 0) {
            continue;
        }
        PomadeModel::TubeRecord record;
        if (!getRecord(id, &record) || !record.hasTube) {
            continue;
        }
        PomadeSnapshotTube entry;
        entry.tubeId = id;
        entry.regionId = record.actual.regionId;
        entry.level = 1;
        entry.parentTubeId = -1;
        entry.childIndex = -1;
        entry.tube = PomadeSnapshotFromTubeRecord(record);
        copyBoundaryBindings(record.actual, &entry);
        entry.tube.version = snapshot.version;
        snapshot.tubes.push_back(std::move(entry));
    }
    // -- V0b: the rest of the hierarchy, parents before children ---------
    //
    // Order matters twice over: PomadeTubePaths resolves a child's prim path
    // from its parent's, and hydrate has to subdivide a parent before it can
    // restore that parent's children.
    if (!snapshot.tubes.empty()) {
        std::vector<int> const &ids = state.tubeIds;
        std::map<int, PomadeModel::TubeRecord> records;
        for (int id : ids) {
            if (id == 0) {
                continue;
            }
            PomadeModel::TubeRecord record;
            if (getRecord(id, &record) && record.hasTube) {
                records[id] = std::move(record);
            }
        }
        // Emission order: every root (parent -1) first, then their
        // descendants breadth-first in ascending id order. Tube 0 is
        // already in; on-the-fly parents come last so their member paths
        // resolve. Transient parents are never committed (§5.4).
        std::vector<int> queue;
        for (PomadeSnapshotTube const &entry : snapshot.tubes) {
            queue.push_back(entry.tubeId);  // the L1 roots, already emitted
        }
        std::vector<int> groups;
        for (auto const &kv : records) {
            if (kv.second.actual.parentTubeId < 0 &&
                (!kv.second.members.empty() || kv.second.transientParent)) {
                if (kv.second.transientParent && !kv.second.persistent) {
                    continue;
                }
                groups.push_back(kv.first);
            }
        }
        for (size_t head = 0; head < queue.size(); ++head) {
            int const parentId = queue[head];
            for (auto const &kv : records) {
                if (kv.second.actual.parentTubeId != parentId) {
                    continue;
                }
                if (kv.second.transientParent && !kv.second.persistent) {
                    continue;
                }
                queue.push_back(kv.first);
                PomadeModel::TubeRecord const &record = kv.second;
                PomadeSnapshotTube entry;
                entry.tubeId = kv.first;
                entry.regionId = record.actual.regionId;
                entry.level = record.actual.level;
                entry.parentTubeId = record.actual.parentTubeId;
                entry.childIndex = record.actual.childIndex;
                entry.tube = PomadeSnapshotFromTubeRecord(record);
                copyBoundaryBindings(record.actual, &entry);
                entry.tube.version = snapshot.version;
                entry.transientParent = record.transientParent;
                entry.persistent = record.persistent;
                entry.imported = record.imported;
                entry.members = record.members;
                for (size_t i = 0; i < record.deltas.centerDu.size(); ++i) {
                    entry.centerDeltas.push_back(record.deltas.centerDu[i]);
                    entry.centerDeltas.push_back(record.deltas.centerDv[i]);
                    entry.centerDeltas.push_back(record.deltas.centerDw[i]);
                }
                for (auto const &sec : record.deltas.sections) {
                    entry.sectionDeltaTransforms.push_back(sec.scale);
                    entry.sectionDeltaTransforms.push_back(sec.twist);
                    for (size_t i = 0; i < sec.u.size(); ++i) {
                        entry.sectionDeltas.push_back(sec.u[i]);
                        entry.sectionDeltas.push_back(sec.v[i]);
                    }
                }
                snapshot.tubes.push_back(std::move(entry));
            }
        }
        for (int id : groups) {
            PomadeModel::TubeRecord const &record = records[id];
            PomadeSnapshotTube entry;
            entry.tubeId = id;
            entry.regionId = record.actual.regionId;
            entry.level = record.actual.level;
            entry.parentTubeId = -1;
            entry.childIndex = -1;
            entry.tube = PomadeSnapshotFromTubeRecord(record);
            copyBoundaryBindings(record.actual, &entry);
            entry.tube.version = snapshot.version;
            entry.transientParent = record.transientParent;
            entry.persistent = record.persistent;
            entry.imported = record.imported;
            entry.members = record.members;
            snapshot.tubes.push_back(std::move(entry));
        }
        // Fill suspension (§2.3): a tube with a SUBDIVIDED child stops
        // filling; a bridge import is not a subdivision, so a parent that
        // only carries imports keeps its own fill.
        for (PomadeSnapshotTube &entry : snapshot.tubes) {
            if (entry.tubeId < 0) {
                continue;  // -1 is "no parent", not this group's id
            }
            for (PomadeSnapshotTube const &other : snapshot.tubes) {
                if (other.parentTubeId == entry.tubeId && !other.imported) {
                    entry.fillSuspended = true;
                    break;
                }
            }
        }
    }
    snapshot.graph = state.graph;
    for (PomadeSnapshotTube &entry : snapshot.tubes) {
        if (entry.regionId >= 0 &&
            size_t(entry.regionId) < snapshot.graph.regionBoundaries.size()) {
            entry.regionBoundary =
                snapshot.graph.regionBoundaries[size_t(entry.regionId)];
        }
    }
    // One region map + expression per level the hierarchy actually reaches
    // (plan/17 §4.5: channel k selects level k + 1).
    for (PomadeSnapshotTube const &entry : snapshot.tubes) {
        snapshot.graph.levelCount =
            std::max(snapshot.graph.levelCount, entry.level);
    }
    // Mesh-fill inputs: the scalp copy plus the tube's region faces at
    // its interpolation id (the SnapshotGraph rasterise is always
    // current, so this never reads a stale map).
    std::shared_ptr<PomadeScalpMesh const> const &scalp = state.scalp;
    // Union-find over the linked pairs (min id wins), mirroring
    // PomadeScalpGraph::InterpId. (Sized by resize, not by a single-argument
    // size constructor: `X(size_t(n))` parses as a function declaration —
    // the most vexing parse.) Resolved once for every L1 root, not only
    // tube 0's region (V6, plan/18 §7 G14).
    int const regionCount = int(snapshot.graph.regionLoops.size());
    std::vector<int> regionParent;
    regionParent.resize(size_t(std::max(regionCount, 0)));
    for (int i = 0; i < regionCount; ++i) {
        regionParent[size_t(i)] = i;
    }
    for (auto const &pr : snapshot.graph.linked) {
        if (pr.first < 0 || pr.first >= regionCount || pr.second < 0 ||
            pr.second >= regionCount) {
            continue;
        }
        auto root = [&](int x) {
            while (regionParent[size_t(x)] != x) {
                regionParent[size_t(x)] =
                    regionParent[size_t(regionParent[size_t(x)])];
                x = regionParent[size_t(x)];
            }
            return x;
        };
        int const a = root(pr.first), b = root(pr.second);
        regionParent[size_t(std::max(a, b))] = std::min(a, b);
    }
    auto interpOf = [&](int regionId) {
        if (regionId < 0 || regionId >= regionCount) {
            return -1;
        }
        int x = regionId;
        while (regionParent[size_t(x)] != x) {
            regionParent[size_t(x)] =
                regionParent[size_t(regionParent[size_t(x)])];
            x = regionParent[size_t(x)];
        }
        return x;
    };
    if (scalp && scalp->finalized) {
        snapshot.scalp = *scalp;
        snapshot.hasScalp = true;
        int const interp = interpOf(snapshot.tubeRegionId);
        if (interp >= 0) {
            for (size_t f = 0; f < snapshot.graph.faceRegions.size(); ++f) {
                if (snapshot.graph.faceRegions[f] == interp) {
                    snapshot.tubeRegionFaces.push_back(int(f));
                }
            }
        }
    }
    // Per-tube mesh-fill roots (V0b): the L1 tube owns the region's faces,
    // and every subdivision hands its faces down to the child cell that
    // claims them — the same K14 partition the region map bakes per level
    // (§2.3, §4.5), so a strand's guide and its map channel agree.
    if (!snapshot.tubes.empty() && snapshot.hasScalp) {
        snapshot.tubes[0].regionFaces = snapshot.tubeRegionFaces;
        // Every other L1 root owns the faces of its own region, resolved
        // through the same interpolation id (linked regions share a fill
        // set, §2.2). Channel 0 of the bake and the mesh-fill roots then
        // agree per root instead of only for tube 0.
        for (PomadeSnapshotTube &entry : snapshot.tubes) {
            if (entry.tubeId == 0 || entry.parentTubeId >= 0 ||
                entry.level != 1 || entry.regionId < 0) {
                continue;
            }
            int const interp = interpOf(entry.regionId);
            if (interp < 0) {
                continue;
            }
            for (size_t f = 0; f < snapshot.graph.faceRegions.size(); ++f) {
                if (snapshot.graph.faceRegions[f] == interp) {
                    entry.regionFaces.push_back(int(f));
                }
            }
        }
        _PartitionRegionFaces(&snapshot);
    }
    // Last, because the face partition above rewrites regionFaces: the hash
    // has to see the entry exactly as the guide fill will (plan/18 §7 G6).
    PomadeHashSnapshot(&snapshot);
    return snapshot;
}

PomadeSnapshot
PomadeSnapshotFromModel(PomadeModel const &model)
{
    return PomadeSnapshotFromState(*PomadeCaptureModelState(model));
}

PomadeSnapshotGuides
PomadeGuidesFromSnapshot(PomadeSnapshot const &snapshot)
{
    return PomadeGuidesFromSnapshot(snapshot, nullptr);
}

PomadeSnapshotGuides
PomadeGuidesFromSnapshot(PomadeSnapshot const &snapshot, PomadeGuideCache *cache)
{
    PomadeSnapshotGuides out;
    if (snapshot.generatedCurvesSuppressed) {
        return out;
    }
    uint64_t nextId = 1000;
    std::vector<int> live;
    size_t reused = 0, refilled = 0;
    // Reconstruct the exact graph loops from the immutable snapshot once.
    // `regionFaces` is intentionally only a fallback: two subface polygons
    // can share that same coarse face while requiring different roots.
    PomadeRegionLoops regionLoops;
    bool haveRegionLoops = false;
    if (snapshot.hasScalp && !snapshot.graph.nodes.empty() &&
        !snapshot.graph.edges.empty()) {
        PomadeScalpGraph graph;
        std::string loopErr;
        haveRegionLoops =
            graph.Restore(snapshot.scalp, snapshot.graph.nodes,
                          snapshot.graph.edges, snapshot.graph.linked,
                          &loopErr) &&
            PomadeFlattenLoops(graph, &regionLoops, &loopErr);
    }
    if (cache) {
        live.reserve(snapshot.tubes.size());
        // A changed scalp changes every mesh fill, so it is checked before
        // the first lookup, not after the build (plan/18 §7 G6).
        cache->NoteScalp(_HashScalp(snapshot));
    }
    for (PomadeSnapshotTube const &entry : snapshot.tubes) {
        if (!entry.tube.hasTube || entry.fillSuspended || entry.imported ||
            !entry.members.empty()) {
            // An on-the-fly parent is an aggregate over tubes that fill
            // themselves; filling it too would double every strand.
            continue;
        }
        if (cache) {
            live.push_back(entry.tubeId);
        }
        // The dirty set: an entry whose bytes are unchanged refills to the
        // same guides, so its slice is copied rather than regenerated.
        PomadeGuideCache::Entry const *hit =
            cache ? cache->Find(entry.tubeId, entry.contentHash) : nullptr;
        if (hit) {
            ++reused;
            out.points.insert(out.points.end(), hit->points.begin(),
                              hit->points.end());
            out.counts.insert(out.counts.end(), hit->counts.begin(),
                              hit->counts.end());
            out.frames.insert(out.frames.end(), hit->frames.begin(),
                              hit->frames.end());
            for (int g = 0; g < hit->guideCount; ++g) {
                out.ids.push_back(nextId++);
                out.tubeIds.push_back(entry.tubeId);
                out.levels.push_back(entry.level);
                out.regionIds.push_back(entry.regionId);
            }
            continue;
        }
        // Mesh roots when this tube owns scalp faces, else the root-disc
        // stream. The root hash stream is keyed by the tube id, so siblings
        // that inherited one set of fill params still differ.
        PomadeGuideSet guides;
        if (haveRegionLoops && entry.regionId >= 0) {
            std::vector<PomadeRootOwnershipCell> const ownership =
                _OwnershipCells(snapshot, entry.tubeId);
            guides = PomadeGenerateGuidesOnRegionForTube(
                entry.tube, snapshot.scalp, regionLoops, entry.regionId,
                entry.tubeId, &ownership);
        } else if (snapshot.hasScalp && !entry.regionFaces.empty()) {
            guides = PomadeGenerateGuidesOnScalpForTube(
                entry.tube, snapshot.scalp, entry.regionFaces.data(),
                int(entry.regionFaces.size()), entry.tubeId);
        } else {
            guides = PomadeGenerateGuidesForTube(entry.tube, entry.tubeId);
        }
        ++refilled;
        out.points.insert(out.points.end(), guides.points.begin(),
                          guides.points.end());
        out.counts.insert(out.counts.end(), guides.counts.begin(),
                          guides.counts.end());
        out.frames.insert(out.frames.end(), guides.frames.begin(),
                          guides.frames.end());
        for (int g = 0; g < guides.guideCount; ++g) {
            out.ids.push_back(nextId++);
            out.tubeIds.push_back(entry.tubeId);
            out.levels.push_back(entry.level);
            out.regionIds.push_back(entry.regionId);
        }
        if (cache) {
            PomadeGuideCache::Entry stored;
            stored.contentHash = entry.contentHash;
            stored.points = guides.points;
            stored.counts = guides.counts;
            stored.frames = guides.frames;
            stored.guideCount = guides.guideCount;
            cache->Store(entry.tubeId, std::move(stored));
        }
    }
    if (cache) {
        // Evict the tubes this version dropped (merge, delete, undo) so the
        // cache never outgrows the model.
        cache->Reap(live);
        cache->_lastReused = reused;
        cache->_lastRefilled = refilled;
    }
    return out;
}

struct _OutputCage
{
    std::vector<float> points;
    std::vector<int> counts;
    std::vector<uint64_t> ids;
    std::vector<double> frames;
    std::vector<float> normalizedT;
    std::vector<int> curveTubeIds;
    std::vector<int> curveLevels;
    std::vector<int> curveRegionIds;
    std::vector<int> ownerIds;
    std::vector<float> ownerDensities;
    std::vector<int> ownerSeeds;
    std::vector<int> ownerCvCounts;
    std::vector<float> ownerEdgeBias;
    std::vector<GfVec2f> ownerChartCentroids;
    std::vector<float> ownerChartMeanRadii;
    std::vector<int> ownerProfileOffsets;
    std::vector<GfVec2f> ownerProfile;
    std::vector<GfVec3i> triangles;
    std::vector<int> triangleOwnerIndices;
    std::vector<GfVec2f> triangleRootCharts;
};

// surfaceCage roots are charted in the actual placed root section, rather
// than raw authoring U/V.  Scale and twist are an affine chart transform, so
// the shared slot triangulation remains valid while the runtime receives the
// coordinates that match the raw K5 rails exactly.
bool
_OutputPlacedRootChart(PomadeTubeSection const &section,
                       std::vector<GfVec2f> *out, GfVec2f *centroid,
                       float *meanRadius)
{
    if (!out || !centroid || !meanRadius || section.u.size() < 3 ||
        section.u.size() != section.v.size() ||
        !std::isfinite(section.scale) || !(section.scale > 0.0f) ||
        !std::isfinite(section.twist)) {
        return false;
    }
    float const ct = std::cos(section.twist);
    float const st = std::sin(section.twist);
    out->resize(section.u.size());
    double twiceArea = 0.0;
    double centroidU = 0.0;
    double centroidV = 0.0;
    for (size_t i = 0; i < section.u.size(); ++i) {
        float const u = section.u[i] * section.scale;
        float const v = section.v[i] * section.scale;
        if (!std::isfinite(u) || !std::isfinite(v)) {
            return false;
        }
        (*out)[i] = GfVec2f(u * ct - v * st, u * st + v * ct);
    }
    for (size_t i = 0; i < out->size(); ++i) {
        GfVec2f const &a = (*out)[i];
        GfVec2f const &b = (*out)[(i + 1) % out->size()];
        double const cross = double(a[0]) * double(b[1]) -
                             double(b[0]) * double(a[1]);
        twiceArea += cross;
        centroidU += (double(a[0]) + double(b[0])) * cross;
        centroidV += (double(a[1]) + double(b[1])) * cross;
    }
    if (!std::isfinite(twiceArea) || std::fabs(twiceArea) <= 1e-12) {
        return false;
    }
    *centroid = GfVec2f(float(centroidU / (3.0 * twiceArea)),
                         float(centroidV / (3.0 * twiceArea)));
    double radius = 0.0;
    for (GfVec2f const &point : *out) {
        double const du = double(point[0]) - double((*centroid)[0]);
        double const dv = double(point[1]) - double((*centroid)[1]);
        radius += std::sqrt(du * du + dv * dv);
    }
    *meanRadius = float(radius / double(out->size()));
    return std::isfinite(*meanRadius) && *meanRadius > 0.0f;
}

// Raw K5 boundary rails for the runtime surface cage.  Unlike K9/K10
// preview guides, rails never apply edgeBias or the owner length profile;
// those are applied once by CurveSource when it scatters dense roots.
bool
_OutputCageFromSnapshot(PomadeSnapshot const &snapshot, _OutputCage *out,
                        std::string *err)
{
    if (!out) {
        if (err) *err = "PomadeBuildCommitLayer: null Output cage";
        return false;
    }
    *out = _OutputCage();
    if (!std::isfinite(snapshot.outputDensityMultiplier) ||
        !(snapshot.outputDensityMultiplier > 0.0f) ||
        !std::isfinite(snapshot.outputWidth) || snapshot.outputWidth < 0.0f ||
        snapshot.outputPtexResolution < -1 ||
        snapshot.outputPtexResolution > 12) {
        if (err) *err = "PomadeBuildCommitLayer: invalid Output settings";
        return false;
    }
    out->ownerProfileOffsets.push_back(0);
    for (PomadeSnapshotTube const &entry : snapshot.tubes) {
        if (!entry.tube.hasTube || entry.fillSuspended || entry.imported ||
            !entry.members.empty()) {
            continue;
        }
        PomadeTubeDesc const tube = PomadeTubeDescFromSnapshot(entry.tube);
        std::vector<PomadeFrame> tubeFrames;
        std::string tubeErr;
        // Every rail uses exactly the same normalized-t stations. Preserve
        // every center knot and interval midpoint (up to 127 stations for
        // the supported 64-CV tube) so a strongly sculpted center curve is
        // faithfully represented between its section knots.  The authored
        // section knots are included too; none of this depends on density.
        std::vector<float> stationT;
        if (tube.sections.size() >= 2) {
            float const begin = tube.sections.front().t;
            float const end = tube.sections.back().t;
            int const centerCount = int(tube.centerX.size());
            if (centerCount < 2) {
                if (err) *err = "PomadeBuildCommitLayer: Output cage lacks centers";
                return false;
            }
            // Cubic pinned BasisCurves needs a useful minimum rail span even
            // for a two-center tube; larger tubes retain every knot/midpoint.
            int const uniformCount = std::max(5, 2 * (centerCount - 1) + 1);
            for (int i = 0; i < uniformCount; ++i) {
                stationT.push_back(begin + (end - begin) * float(i) /
                                             float(uniformCount - 1));
            }
            for (PomadeTubeSection const &section : tube.sections) {
                stationT.push_back(section.t);
            }
            std::sort(stationT.begin(), stationT.end());
            stationT.erase(std::unique(stationT.begin(), stationT.end()),
                           stationT.end());
        }
        std::vector<float> mesh;
        if (!PomadeTubeFramesCpu(tube, &tubeFrames, &tubeErr) ||
            stationT.size() < 2 || tube.ringVerts < 3) {
            if (err) *err = "PomadeBuildCommitLayer: Output cage: " + tubeErr;
            return false;
        }
        mesh.reserve(stationT.size() * size_t(tube.ringVerts) * 3);
        for (float station : stationT) {
            std::vector<float> ring;
            if (!PomadeSampleTubeRingCpu(tube, tubeFrames, station, &ring,
                                        &tubeErr) ||
                ring.size() != size_t(tube.ringVerts) * 3) {
                if (err) *err = "PomadeBuildCommitLayer: Output cage: " + tubeErr;
                return false;
            }
            mesh.insert(mesh.end(), ring.begin(), ring.end());
        }
        float const beginT = stationT.front();
        float const endT = stationT.back();
        float const spanT = endT > beginT ? endT - beginT : 1.0f;
        int const owner = int(out->ownerIds.size());
        std::vector<GfVec2f> rootChart;
        GfVec2f rootCentroid;
        float rootMeanRadius = 0.0f;
        if (!_OutputPlacedRootChart(tube.sections.front(), &rootChart,
                                    &rootCentroid, &rootMeanRadius)) {
            if (err) *err = "PomadeBuildCommitLayer: invalid Output root chart";
            return false;
        }
        out->ownerIds.push_back(entry.tubeId);
        out->ownerDensities.push_back(entry.tube.fill.density);
        out->ownerSeeds.push_back(entry.tube.fill.seed);
        out->ownerCvCounts.push_back(entry.tube.fill.cvCount);
        out->ownerEdgeBias.push_back(entry.tube.fill.edgeBias);
        out->ownerChartCentroids.push_back(rootCentroid);
        out->ownerChartMeanRadii.push_back(rootMeanRadius);
        if (entry.tube.fill.lengthProfile.empty()) {
            out->ownerProfile.emplace_back(0.0f, 1.0f);
            out->ownerProfile.emplace_back(1.0f, 1.0f);
        } else if (entry.tube.fill.lengthProfile.size() % 2 == 0) {
            for (size_t p = 0; p < entry.tube.fill.lengthProfile.size(); p += 2) {
                out->ownerProfile.emplace_back(entry.tube.fill.lengthProfile[p],
                                               entry.tube.fill.lengthProfile[p + 1]);
            }
        } else {
            if (err) *err = "PomadeBuildCommitLayer: malformed Output length profile";
            return false;
        }
        out->ownerProfileOffsets.push_back(int(out->ownerProfile.size()));

        size_t const railBase = out->counts.size();
        for (int slot = 0; slot < tube.ringVerts; ++slot) {
            double src[16] = {tubeFrames.front().nx, tubeFrames.front().ny,
                              tubeFrames.front().nz, 0.0,
                              tubeFrames.front().bx, tubeFrames.front().by,
                              tubeFrames.front().bz, 0.0,
                              tubeFrames.front().tx, tubeFrames.front().ty,
                              tubeFrames.front().tz, 0.0,
                              mesh[size_t(slot) * 3], mesh[size_t(slot) * 3 + 1],
                              mesh[size_t(slot) * 3 + 2], 1.0};
            GfMatrix4d frame;
            if (!_OutputRootFrameFromK10(src, &frame)) {
                if (err) *err = "PomadeBuildCommitLayer: invalid Output cage frame";
                return false;
            }
            out->counts.push_back(int(stationT.size()));
            out->ids.push_back((uint64_t(uint32_t(entry.tubeId)) << 32) |
                               uint64_t(uint32_t(slot)));
            out->curveTubeIds.push_back(entry.tubeId);
            out->curveLevels.push_back(entry.level);
            out->curveRegionIds.push_back(entry.regionId);
            for (int r = 0; r < int(stationT.size()); ++r) {
                size_t const at = (size_t(r) * size_t(tube.ringVerts) +
                                   size_t(slot)) * 3;
                out->points.push_back(mesh[at]);
                out->points.push_back(mesh[at + 1]);
                out->points.push_back(mesh[at + 2]);
                out->normalizedT.push_back((stationT[size_t(r)] - beginT) / spanT);
            }
            double const *values = frame.GetArray();
            out->frames.insert(out->frames.end(), values, values + 16);
        }
        std::vector<std::array<int, 3>> topology;
        if (!PomadeTriangulateSectionSlotsCpu(tube.sections.front(), &topology,
                                             &tubeErr)) {
            if (err) *err = "PomadeBuildCommitLayer: Output cage: " + tubeErr;
            return false;
        }
        // This is K9's own deterministic terminal-slot topology.  It keeps
        // collinear authored boundary slots incident to a nondegenerate cage
        // triangle instead of silently dropping a sculpted rail.
        for (std::array<int, 3> const &tri : topology) {
            out->triangles.emplace_back(int(railBase + size_t(tri[0])),
                                        int(railBase + size_t(tri[1])),
                                        int(railBase + size_t(tri[2])));
            out->triangleOwnerIndices.push_back(owner);
            out->triangleRootCharts.push_back(rootChart[size_t(tri[0])]);
            out->triangleRootCharts.push_back(rootChart[size_t(tri[1])]);
            out->triangleRootCharts.push_back(rootChart[size_t(tri[2])]);
        }
    }
    return true;
}

// Validation-only compatibility path for layers authored before subface
// regions had their own root sampler.  Those layers stored the deterministic
// face-list/disc fill, so accepting them requires reproducing that exact
// stream; it must never be used to author a new layer.
static PomadeSnapshotGuides
_LegacyGuidesFromSnapshot(PomadeSnapshot const &snapshot)
{
    PomadeSnapshotGuides out;
    uint64_t nextId = 1000;
    for (PomadeSnapshotTube const &entry : snapshot.tubes) {
        if (!entry.tube.hasTube || entry.fillSuspended || entry.imported ||
            !entry.members.empty()) {
            continue;
        }
        PomadeGuideSet guides;
        if (snapshot.hasScalp && !entry.regionFaces.empty()) {
            guides = PomadeGenerateGuidesOnScalpForTube(
                entry.tube, snapshot.scalp, entry.regionFaces.data(),
                int(entry.regionFaces.size()), entry.tubeId);
        } else {
            guides = PomadeGenerateGuidesForTube(entry.tube, entry.tubeId);
        }
        out.points.insert(out.points.end(), guides.points.begin(),
                          guides.points.end());
        out.counts.insert(out.counts.end(), guides.counts.begin(),
                          guides.counts.end());
        out.frames.insert(out.frames.end(), guides.frames.begin(),
                          guides.frames.end());
        for (int g = 0; g < guides.guideCount; ++g) {
            out.ids.push_back(nextId++);
            out.tubeIds.push_back(entry.tubeId);
            out.levels.push_back(entry.level);
            out.regionIds.push_back(entry.regionId);
        }
    }
    return out;
}

// region-v1 already used exact polygon support, but its root frames came
// from raw K4.  Clear the later support-plane pin on a private copy so a v1
// layer continues to authenticate byte-for-byte after v2 introduced it.
static PomadeSnapshotGuides
_RegionV1GuidesFromSnapshot(PomadeSnapshot const &snapshot)
{
    PomadeSnapshot rawFrames = snapshot;
    for (PomadeSnapshotTube &entry : rawFrames.tubes) {
        entry.tube.rootFramePinned = false;
        entry.tube.rootFrame = PomadeFrame();
    }
    return PomadeGuidesFromSnapshot(rawFrames);
}

bool
PomadeBuildCommitLayer(PomadeSnapshot const &snapshot,
                      PomadeCommitPaths const &paths,
                      SdfLayerRefPtr *outLayer,
                      std::string *err,
                      PomadeGuideCache *cache,
                      PomadeGuideCache *outputCache)
{
    // Sparse Output is immutable snapshot data; it intentionally does not
    // share the interactive Guide cache.
    (void)outputCache;
    auto fail = [&](char const *what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    if (!outLayer) {
        return fail("PomadeBuildCommitLayer: null out-layer");
    }
    if (snapshot.tubes.empty()) {
        return fail("PomadeBuildCommitLayer: snapshot holds no tubes");
    }
    SdfLayerRefPtr layer = SdfLayer::CreateAnonymous("usdGenPomade-commit");
    if (!layer) {
        return fail("PomadeBuildCommitLayer: cannot create anonymous layer");
    }

    // -- groom container ----------------------------------------------------
    SdfPrimSpecHandle groom =
        _EnsurePrim(layer, paths.groomPath, SdfSpecifierDef, "UsdGenPomadeGroom");
    if (!groom) {
        return fail("PomadeBuildCommitLayer: cannot author the groom prim");
    }
    if (!_SetAttr(groom, "usdGen:pomade:version", SdfValueTypeNames->Token,
                  TfToken("1"))) {
        return fail("PomadeBuildCommitLayer: cannot author the groom version");
    }
    // The root sampler changes the deterministic Guide bytes.  Keep that
    // authoring version separate from the structural groom version so old
    // layers can be verified by their legacy generator during hydrate.
    if (!_SetAttr(groom, "usdGen:pomade:rootSampler",
                  SdfValueTypeNames->Token, TfToken("region-v3"))) {
        return fail("PomadeBuildCommitLayer: cannot author the root sampler");
    }
    // K14's child slot order changed on 2026-09-24 (coincident-vertex drop,
    // split tie rule, per-station ring alignment): the same (parent, seed)
    // now numbers a child's ring slots differently. Stored residuals are
    // per slot of the derivation that measured them, so hydrate must know
    // which K14 wrote them (see PomadeHydrateModel).
    if (!_SetAttr(groom, "usdGen:pomade:subdivider",
                  SdfValueTypeNames->Token, TfToken("aligned-v1"))) {
        return fail("PomadeBuildCommitLayer: cannot author the subdivider");
    }
    if (!_SetAttr(groom, "usdGen:pomade:generatedCurvesSuppressed",
                  SdfValueTypeNames->Bool,
                  snapshot.generatedCurvesSuppressed)) {
        return fail("PomadeBuildCommitLayer: cannot author generated-curve state");
    }
    if (!_SetAttr(groom, "usdGen:pomade:output:enabled",
                  SdfValueTypeNames->Bool, snapshot.outputEnabled) ||
        !_SetAttr(groom, "usdGen:pomade:output:densityMultiplier",
                  SdfValueTypeNames->Float, snapshot.outputDensityMultiplier) ||
        !_SetAttr(groom, "usdGen:pomade:output:width",
                  SdfValueTypeNames->Float, snapshot.outputWidth) ||
        !_SetAttr(groom, "usdGen:pomade:output:ptexResolution",
                  SdfValueTypeNames->Int, snapshot.outputPtexResolution)) {
        return fail("PomadeBuildCommitLayer: cannot author Output settings");
    }
    if (!paths.descriptionPath.IsEmpty() &&
        !_SetRelTargets(groom, "usdGen:pomade:description",
                        SdfPathVector{paths.descriptionPath})) {
        return fail("PomadeBuildCommitLayer: cannot author the description link");
    }
    if (!snapshot.scalpPath.IsEmpty() &&
        !_SetRelTargets(groom, "usdGen:pomade:scalp",
                        SdfPathVector{snapshot.scalpPath})) {
        return fail("PomadeBuildCommitLayer: cannot author the scalp link");
    }

    // -- tubes (hierarchy = prim hierarchy, every level) --------------------
    SdfPrimSpecHandle tubes =
        _EnsurePrim(layer, paths.TubesPath(), SdfSpecifierDef, "Scope");
    if (!tubes) {
        return fail("PomadeBuildCommitLayer: cannot author the Tubes scope");
    }
    std::map<int, SdfPath> const tubePaths = PomadeTubePaths(snapshot, paths);
    for (PomadeSnapshotTube const &entry : snapshot.tubes) {
        PomadeModel::TubeSnapshot const &tube = entry.tube;
        if (!tube.hasTube) {
            continue;
        }
        auto const pathIt = tubePaths.find(entry.tubeId);
        if (pathIt == tubePaths.end()) {
            return fail("PomadeBuildCommitLayer: tube has no prim path");
        }
        SdfPrimSpecHandle prim =
            _EnsurePrim(layer, pathIt->second, SdfSpecifierDef, "UsdGenTube");
        if (!prim) {
            return fail("PomadeBuildCommitLayer: cannot author a tube prim");
        }
        int const rings = int(tube.centerX.size());
        VtVec3fArray centers;
        centers.resize(size_t(rings));
        for (int r = 0; r < rings; ++r) {
            centers[size_t(r)] = GfVec3f(tube.centerX[size_t(r)],
                                         tube.centerY[size_t(r)],
                                         tube.centerZ[size_t(r)]);
        }
        // Authored sections (default circles for an untouched test tube,
        // so the commit bytes match the P1 form exactly).
        std::vector<PomadeTubeSection> sections = tube.sections;
        if (sections.size() < 2) {
            sections = PomadeDefaultSections(tube);
        }
        int const ringVerts = tube.shape.ringVerts;
        VtFloatArray sectionT;
        sectionT.resize(sections.size());
        VtVec2fArray sectionCvs;
        sectionCvs.resize(sections.size() * size_t(ringVerts));
        // Keep the legacy baked plane pairs for old readers, and carry the
        // authored representation separately so a hydrate can preserve its
        // exact scale/twist interpolation and later K6 residuals.
        VtVec2fArray sectionRawCvs;
        sectionRawCvs.resize(sections.size() * size_t(ringVerts));
        VtVec2fArray sectionTransforms;
        sectionTransforms.resize(sections.size());
        for (size_t r = 0; r < sections.size(); ++r) {
            sectionT[r] = sections[r].t;
            float const ct = std::cos(sections[r].twist);
            float const st = std::sin(sections[r].twist);
            sectionTransforms[r] =
                GfVec2f(sections[r].scale, sections[r].twist);
            for (int s = 0; s < ringVerts; ++s) {
                sectionRawCvs[r * size_t(ringVerts) + size_t(s)] =
                    GfVec2f(sections[r].u[size_t(s)],
                            sections[r].v[size_t(s)]);
                float const uu =
                    sections[r].u[size_t(s)] * sections[r].scale;
                float const vv =
                    sections[r].v[size_t(s)] * sections[r].scale;
                sectionCvs[r * size_t(ringVerts) + size_t(s)] =
                    GfVec2f(uu * ct - vv * st, uu * st + vv * ct);
            }
        }
        VtFloatArray lengthProfile;
        lengthProfile.resize(tube.fill.lengthProfile.size());
        for (size_t i = 0; i < tube.fill.lengthProfile.size(); ++i) {
            lengthProfile[i] = tube.fill.lengthProfile[i];
        }
        VtFloatArray rootFrame(9);
        rootFrame[0] = tube.rootFrame.tx;
        rootFrame[1] = tube.rootFrame.ty;
        rootFrame[2] = tube.rootFrame.tz;
        rootFrame[3] = tube.rootFrame.nx;
        rootFrame[4] = tube.rootFrame.ny;
        rootFrame[5] = tube.rootFrame.nz;
        rootFrame[6] = tube.rootFrame.bx;
        rootFrame[7] = tube.rootFrame.by;
        rootFrame[8] = tube.rootFrame.bz;
        VtFloatArray frameReference(9);
        for (size_t i = 0; i < frameReference.size(); ++i) {
            frameReference[i] = tube.frameReference[i];
        }
        // Deltas ride the schema's own layout: 3 floats per center CV in
        // the derived frames, 2 per section vertex in the section plane,
        // and one scale/twist residual per section.
        VtVec3fArray centerDeltas;
        centerDeltas.resize(entry.centerDeltas.size() / 3);
        for (size_t i = 0; i < centerDeltas.size(); ++i) {
            centerDeltas[i] = GfVec3f(entry.centerDeltas[i * 3],
                                      entry.centerDeltas[i * 3 + 1],
                                      entry.centerDeltas[i * 3 + 2]);
        }
        VtVec2fArray sectionDeltas;
        sectionDeltas.resize(entry.sectionDeltas.size() / 2);
        for (size_t i = 0; i < sectionDeltas.size(); ++i) {
            sectionDeltas[i] = GfVec2f(entry.sectionDeltas[i * 2],
                                       entry.sectionDeltas[i * 2 + 1]);
        }
        VtVec2fArray sectionDeltaTransforms;
        sectionDeltaTransforms.resize(entry.sectionDeltaTransforms.size() / 2);
        for (size_t i = 0; i < sectionDeltaTransforms.size(); ++i) {
            sectionDeltaTransforms[i] =
                GfVec2f(entry.sectionDeltaTransforms[i * 2],
                        entry.sectionDeltaTransforms[i * 2 + 1]);
        }
        VtIntArray inheritedBoundaryBindings;
        inheritedBoundaryBindings.resize(
            entry.inheritedBoundaryBindings.size());
        for (size_t i = 0; i < inheritedBoundaryBindings.size(); ++i) {
            inheritedBoundaryBindings[i] =
                entry.inheritedBoundaryBindings[i];
        }
        bool attrsOk =
            _SetAttr(prim, "usdGen:pomade:regionId", SdfValueTypeNames->Int,
                     entry.regionId) &&
            _SetAttr(prim, "usdGen:pomade:level", SdfValueTypeNames->Int,
                     entry.level) &&
            _SetAttr(prim, "usdGen:pomade:centerPoints",
                     SdfValueTypeNames->Point3fArray, centers) &&
            _SetAttr(prim, "usdGen:pomade:sectionT",
                     SdfValueTypeNames->FloatArray, sectionT) &&
            _SetAttr(prim, "usdGen:pomade:sectionCvCount",
                     SdfValueTypeNames->Int, ringVerts) &&
            _SetAttr(prim, "usdGen:pomade:sectionCvs",
                     SdfValueTypeNames->Float2Array, sectionCvs) &&
            _SetAttr(prim, "usdGen:pomade:sectionRawCvs",
                     SdfValueTypeNames->Float2Array, sectionRawCvs) &&
            _SetAttr(prim, "usdGen:pomade:sectionTransforms",
                     SdfValueTypeNames->Float2Array, sectionTransforms) &&
            _SetAttr(prim, "usdGen:pomade:rootFramePinned",
                     SdfValueTypeNames->Bool, tube.rootFramePinned) &&
            _SetAttr(prim, "usdGen:pomade:rootFrame",
                     SdfValueTypeNames->FloatArray, rootFrame) &&
            _SetAttr(prim, "usdGen:pomade:frameReference",
                     SdfValueTypeNames->FloatArray, frameReference) &&
            _SetAttr(prim, "usdGen:pomade:childIndex", SdfValueTypeNames->Int,
                     entry.childIndex) &&
            _SetAttr(prim, "usdGen:pomade:centerDeltas",
                     SdfValueTypeNames->Point3fArray, centerDeltas) &&
            _SetAttr(prim, "usdGen:pomade:sectionDeltas",
                     SdfValueTypeNames->Float2Array, sectionDeltas) &&
            _SetAttr(prim, "usdGen:pomade:sectionDeltaTransforms",
                     SdfValueTypeNames->Float2Array, sectionDeltaTransforms) &&
            _SetAttr(prim, "usdGen:pomade:inheritedBoundaryBindings",
                     SdfValueTypeNames->IntArray, inheritedBoundaryBindings) &&
            _SetAttr(prim, "usdGen:pomade:subdivide:count",
                     SdfValueTypeNames->Int, tube.subdivide.count) &&
            _SetAttr(prim, "usdGen:pomade:subdivide:seed",
                     SdfValueTypeNames->Int, tube.subdivide.seed) &&
            _SetAttr(prim, "usdGen:pomade:subdivide:splitMode",
                     SdfValueTypeNames->Token,
                     TfToken(tube.subdivide.splitMode)) &&
            _SetAttr(prim, "usdGen:pomade:fill:density",
                     SdfValueTypeNames->Float, tube.fill.density) &&
            _SetAttr(prim, "usdGen:pomade:fill:cvCount", SdfValueTypeNames->Int,
                     tube.fill.cvCount) &&
            _SetAttr(prim, "usdGen:pomade:fill:lengthProfile",
                     SdfValueTypeNames->FloatArray, lengthProfile) &&
            _SetAttr(prim, "usdGen:pomade:fill:seed", SdfValueTypeNames->Int,
                     tube.fill.seed) &&
            _SetAttr(prim, "usdGen:pomade:fill:edgeBias",
                     SdfValueTypeNames->Float, tube.fill.edgeBias) &&
            _SetAttr(prim, "usdGen:pomade:locked", SdfValueTypeNames->Bool,
                     tube.locked || entry.imported) &&
            _SetAttr(prim, "usdGen:pomade:lockParents", SdfValueTypeNames->Bool,
                     tube.lockParents) &&
            _SetAttr(prim, "usdGen:pomade:lockChildren",
                     SdfValueTypeNames->Bool, tube.lockChildren);
        if (!attrsOk) {
            return fail("PomadeBuildCommitLayer: cannot author tube attributes");
        }
        // An on-the-fly parent the artist kept carries the members it was
        // built from (plan/17 §5.4 "Make persistent"); transient ones never
        // reach the snapshot at all.
        if (!entry.members.empty() || entry.persistent) {
            SdfPathVector members;
            for (int member : entry.members) {
                auto const it = tubePaths.find(member);
                if (it != tubePaths.end()) {
                    members.push_back(it->second);
                }
            }
            if (!_PrependApiSchema(prim, "UsdGenTubeHierarchyAPI") ||
                !_SetRelTargets(prim, "usdGen:pomade:members", members) ||
                !_SetAttr(prim, "usdGen:pomade:persistent",
                          SdfValueTypeNames->Bool, entry.persistent)) {
                return fail("PomadeBuildCommitLayer: cannot author the tube "
                            "hierarchy API");
            }
        }
    }

    // -- scalp graph (P2: the shared planar graph, polygon-style) ---------
    {
        SdfPrimSpecHandle graphPrim =
            _EnsurePrim(layer, paths.ScalpGraphPath(), SdfSpecifierDef,
                        "UsdGenScalpGraph");
        if (!graphPrim) {
            return fail(
                "PomadeBuildCommitLayer: cannot author the scalp graph");
        }
        // Nodes are parallel arrays; edges and region loops name node
        // INDICES into them (0-based), not the model's stable ids: the
        // stage encoding is positional, the stable ids are model-side.
        PomadeModel::GraphSnapshot const &gs = snapshot.graph;
        std::map<int, int> indexOf;
        VtIntArray nodeFaceIds;
        VtVec2fArray nodeUVs;
        nodeFaceIds.reserve(gs.nodes.size());
        nodeUVs.reserve(gs.nodes.size());
        for (size_t i = 0; i < gs.nodes.size(); ++i) {
            indexOf[gs.nodes[i].id] = int(i);
            nodeFaceIds.push_back(gs.nodes[i].faceId);
            nodeUVs.push_back(
                GfVec2f(gs.nodes[i].u, gs.nodes[i].v));
        }
        VtVec2iArray edges;
        edges.reserve(gs.edges.size());
        for (auto const &pr : gs.edges) {
            auto ia = indexOf.find(pr.first);
            auto ib = indexOf.find(pr.second);
            if (ia == indexOf.end() || ib == indexOf.end()) {
                return fail("PomadeBuildCommitLayer: graph edge names an "
                            "absent node");
            }
            edges.push_back(GfVec2i(ia->second, ib->second));
        }
        VtIntArray regionCounts;
        VtIntArray regionIndices;
        regionCounts.reserve(gs.regionLoops.size());
        for (auto const &loop : gs.regionLoops) {
            regionCounts.push_back(int(loop.size()));
            for (int nid : loop) {
                auto it = indexOf.find(nid);
                if (it == indexOf.end()) {
                    return fail("PomadeBuildCommitLayer: region loop names "
                                "an absent node");
                }
                regionIndices.push_back(it->second);
            }
        }
        VtVec3fArray regionColors;
        regionColors.reserve(gs.regionLoops.size());
        for (size_t i = 0; i < gs.regionColors.size() / 3; ++i) {
            regionColors.push_back(
                GfVec3f(gs.regionColors[i * 3], gs.regionColors[i * 3 + 1],
                        gs.regionColors[i * 3 + 2]));
        }
        VtVec2iArray linked;
        linked.reserve(gs.linked.size());
        for (auto const &pr : gs.linked) {
            linked.push_back(GfVec2i(pr.first, pr.second));
        }
        bool graphOk =
            _SetAttr(graphPrim, "usdGen:pomade:nodeFaceIds",
                     SdfValueTypeNames->IntArray, nodeFaceIds) &&
            _SetAttr(graphPrim, "usdGen:pomade:nodeUVs",
                     SdfValueTypeNames->Float2Array, nodeUVs) &&
            _SetAttr(graphPrim, "usdGen:pomade:edges",
                     SdfValueTypeNames->Int2Array, edges) &&
            _SetAttr(graphPrim, "usdGen:pomade:regionNodeCounts",
                     SdfValueTypeNames->IntArray, regionCounts) &&
            _SetAttr(graphPrim, "usdGen:pomade:regionNodeIndices",
                     SdfValueTypeNames->IntArray, regionIndices) &&
            _SetAttr(graphPrim, "usdGen:pomade:regionColors",
                     SdfValueTypeNames->Color3fArray, regionColors) &&
            _SetAttr(graphPrim, "usdGen:pomade:linkedRegions",
                     SdfValueTypeNames->Int2Array, linked) &&
            _SetAttr(graphPrim, "usdGen:pomade:snapRadius",
                     SdfValueTypeNames->Float, gs.snapRadius);
        if (!graphOk) {
            return fail("PomadeBuildCommitLayer: cannot author graph fields");
        }
    }

    // -- live region primvar (P2: the preview map, every swap) -------------
    // A per-face int primvar on the scalp mesh. The Ptex is the contract;
    // this primvar is the live preview the HUD and the bake read from.
    SdfPath const boundScalpPath = snapshot.scalpPath.IsEmpty()
        ? paths.scalpPath : snapshot.scalpPath;
    // Both opinions belong to the Mesh: for a face GeomSubset scalp that is
    // the subset's parent (plan/02 §2.20), and the per-face primvar stays
    // parent-sized because a subset never renumbers faces.
    SdfPath const boundMeshPath = snapshot.scalpPath.IsEmpty()
        ? paths.ScalpMeshPath()
        : (snapshot.scalpMeshPath.IsEmpty() ? snapshot.scalpPath
                                            : snapshot.scalpMeshPath);
    bool const needOutputRest = snapshot.outputEnabled &&
        !boundScalpPath.IsEmpty();
    if (needOutputRest || (!snapshot.scalpPath.IsEmpty() &&
                           !snapshot.graph.faceRegions.empty())) {
        SdfPrimSpecHandle scalp =
            _EnsurePrim(layer, boundMeshPath, SdfSpecifierOver, nullptr);
        if (!scalp) {
            return fail("PomadeBuildCommitLayer: cannot over the scalp mesh");
        }
        // CurveSource requires an immutable rest binding.  RestAPI exposes
        // the bound mesh's Default-time points through the scene index, so it
        // remains a valid rest pose without copying a posed current sample.
        if (needOutputRest && !_PrependApiSchema(scalp, "UsdGenRestAPI")) {
            return fail("PomadeBuildCommitLayer: cannot apply RestAPI to Output scalp");
        }
        if (!snapshot.graph.faceRegions.empty()) {
            VtIntArray regions;
            regions.assign(snapshot.graph.faceRegions.begin(),
                           snapshot.graph.faceRegions.end());
            if (!_SetAttr(scalp, "primvars:usdGen:pomadeRegion",
                          SdfValueTypeNames->IntArray, regions,
                          SdfVariabilityUniform, /*custom*/ true) ||
                !_SetInterpolation(scalp, "primvars:usdGen:pomadeRegion",
                                   "uniform")) {
                return fail("PomadeBuildCommitLayer: cannot author the live "
                            "region primvar");
            }
        }
    }

    // -- committed guides (the existing usdGen guide contract) --------------
    //
    // One fill per LEAF tube (§2.3: a subdivided parent's own fill is
    // suspended, a bridge import carries explicit geometry instead), in the
    // snapshot's parents-first order. Hydrate re-runs the same function over
    // the hydrated model and memcmps the result, so the two cannot drift.
    PomadeSnapshotGuides const guideSet =
        PomadeGuidesFromSnapshot(snapshot, cache);
    VtVec3fArray points;
    points.resize(guideSet.points.size() / 3);
    for (size_t i = 0; i < points.size(); ++i) {
        points[i] = GfVec3f(guideSet.points[i * 3], guideSet.points[i * 3 + 1],
                            guideSet.points[i * 3 + 2]);
    }
    VtIntArray const counts(guideSet.counts.begin(), guideSet.counts.end());
    VtUInt64Array const ids(guideSet.ids.begin(), guideSet.ids.end());
    VtIntArray const tubeIds(guideSet.tubeIds.begin(), guideSet.tubeIds.end());
    VtIntArray const levels(guideSet.levels.begin(), guideSet.levels.end());
    VtIntArray const regionIds(guideSet.regionIds.begin(),
                               guideSet.regionIds.end());
    VtMatrix4dArray frames;
    frames.resize(guideSet.frames.size() / 16);
    for (size_t g = 0; g < frames.size(); ++g) {
        double const *src = &guideSet.frames[g * 16];
        frames[g].Set(src[0], src[1], src[2], src[3], src[4], src[5], src[6],
                      src[7], src[8], src[9], src[10], src[11], src[12],
                      src[13], src[14], src[15]);
    }
    SdfPrimSpecHandle guidesPrim = _EnsurePrim(layer, paths.GuidesPath(),
                                               SdfSpecifierDef, "BasisCurves");
    if (!guidesPrim) {
        return fail("PomadeBuildCommitLayer: cannot author the Guides prim");
    }
    bool guidesOk =
        _PrependApiSchema(guidesPrim, "UsdGenCurveAPI") &&
        _SetAttr(guidesPrim, "type", SdfValueTypeNames->Token, TfToken("cubic"),
                 SdfVariabilityUniform) &&
        _SetAttr(guidesPrim, "basis", SdfValueTypeNames->Token,
                 TfToken("bspline"), SdfVariabilityUniform) &&
        _SetAttr(guidesPrim, "wrap", SdfValueTypeNames->Token, TfToken("pinned"),
                 SdfVariabilityUniform) &&
        _SetAttr(guidesPrim, "curveVertexCounts", SdfValueTypeNames->IntArray,
                 counts) &&
        _SetAttr(guidesPrim, "points", SdfValueTypeNames->Point3fArray,
                 points) &&
        _SetAttr(guidesPrim, "primvars:usdGen:role", SdfValueTypeNames->Token,
                 TfToken("guide"), SdfVariabilityUniform) &&
        _SetAttr(guidesPrim, "primvars:usdGen:curveId",
                 SdfValueTypeNames->UInt64Array, ids,
                 SdfVariabilityUniform) &&
        _SetAttr(guidesPrim, "primvars:usdGen:rootFrame",
                 SdfValueTypeNames->Matrix4dArray, frames,
                 SdfVariabilityUniform) &&
        _SetAttr(guidesPrim, "primvars:tubeId", SdfValueTypeNames->IntArray,
                 tubeIds, SdfVariabilityUniform, /*custom*/ true) &&
        _SetAttr(guidesPrim, "primvars:hierarchyLevel",
                 SdfValueTypeNames->IntArray, levels, SdfVariabilityUniform,
                 /*custom*/ true) &&
        _SetAttr(guidesPrim, "primvars:regionId", SdfValueTypeNames->IntArray,
                 regionIds, SdfVariabilityUniform, /*custom*/ true) &&
        _SetInterpolation(guidesPrim, "primvars:usdGen:curveId", "uniform") &&
        _SetInterpolation(guidesPrim, "primvars:usdGen:rootFrame", "uniform") &&
        _SetInterpolation(guidesPrim, "primvars:tubeId", "uniform") &&
        _SetInterpolation(guidesPrim, "primvars:hierarchyLevel", "uniform") &&
        _SetInterpolation(guidesPrim, "primvars:regionId", "uniform");
    if (!guidesOk) {
        return fail("PomadeBuildCommitLayer: cannot author guide attributes");
    }

    // -- owned sparse Output cage + categorical owner map -----------------
    //
    // The USD source is intentionally compact. CurveSource creates dense
    // hairs transiently from these raw K5 rails and the current immutable
    // owner map; Clear Guides and preview visibility cannot affect it.
    if (snapshot.outputEnabled) {
        SdfPath const outputScalpPath = snapshot.scalpPath.IsEmpty()
            ? paths.scalpPath : snapshot.scalpPath;
        if (outputScalpPath.IsEmpty() || !snapshot.hasScalp) {
            return fail("PomadeBuildCommitLayer: Output requires bound scalp geometry");
        }
        _OutputCage cage;
        std::string outputErr;
        if (!_OutputCageFromSnapshot(snapshot, &cage, &outputErr)) {
            return fail(outputErr.c_str());
        }
        uint64_t const mapGeneration = _OutputOwnerMapContentId(snapshot);
        std::string outputMapFile;
        if (!_BakeOutputOwnerMap(snapshot, mapGeneration, &outputMapFile,
                                 &outputErr)) {
            return fail(("PomadeBuildCommitLayer: OutputRegionMap: " + outputErr).c_str());
        }
        VtVec3fArray outputPoints(cage.points.size() / 3);
        for (size_t i = 0; i < outputPoints.size(); ++i) {
            outputPoints[i] = GfVec3f(cage.points[i * 3], cage.points[i * 3 + 1],
                                      cage.points[i * 3 + 2]);
        }
        VtMatrix4dArray outputFrames(cage.frames.size() / 16);
        for (size_t i = 0; i < outputFrames.size(); ++i) {
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    outputFrames[i][row][column] =
                        cage.frames[i * 16 + size_t(row * 4 + column)];
                }
            }
        }
        VtIntArray const outputCounts(cage.counts.begin(), cage.counts.end());
        VtUInt64Array const outputIds(cage.ids.begin(), cage.ids.end());
        VtIntArray const outputTubeIds(cage.curveTubeIds.begin(), cage.curveTubeIds.end());
        VtIntArray const outputLevels(cage.curveLevels.begin(), cage.curveLevels.end());
        VtIntArray const outputRegionIds(cage.curveRegionIds.begin(),
                                         cage.curveRegionIds.end());
        VtFloatArray const normalizedT(cage.normalizedT.begin(), cage.normalizedT.end());
        VtIntArray const ownerIds(cage.ownerIds.begin(), cage.ownerIds.end());
        VtFloatArray const ownerDensities(cage.ownerDensities.begin(), cage.ownerDensities.end());
        VtIntArray const ownerSeeds(cage.ownerSeeds.begin(), cage.ownerSeeds.end());
        VtIntArray const ownerCvCounts(cage.ownerCvCounts.begin(), cage.ownerCvCounts.end());
        VtFloatArray const ownerEdgeBias(cage.ownerEdgeBias.begin(), cage.ownerEdgeBias.end());
        VtVec2fArray const ownerChartCentroids(cage.ownerChartCentroids.begin(),
                                               cage.ownerChartCentroids.end());
        VtFloatArray const ownerChartMeanRadii(cage.ownerChartMeanRadii.begin(),
                                               cage.ownerChartMeanRadii.end());
        VtIntArray const ownerProfileOffsets(cage.ownerProfileOffsets.begin(), cage.ownerProfileOffsets.end());
        VtVec2fArray const ownerProfile(cage.ownerProfile.begin(), cage.ownerProfile.end());
        VtVec3iArray const triangles(cage.triangles.begin(), cage.triangles.end());
        VtIntArray const triangleOwners(cage.triangleOwnerIndices.begin(),
                                        cage.triangleOwnerIndices.end());
        VtVec2fArray const triangleRootCharts(cage.triangleRootCharts.begin(),
                                               cage.triangleRootCharts.end());

        SdfPrimSpecHandle outputCurves = _EnsurePrim(
            layer, paths.OutputCurvesPath(), SdfSpecifierDef, "BasisCurves");
        SdfPrimSpecHandle outputMap = _EnsurePrim(
            layer, paths.OutputRegionMapPath(), SdfSpecifierDef, "UsdGenPtexMap");
        if (!outputCurves || !outputMap) {
            return fail("PomadeBuildCommitLayer: cannot author Output source or map");
        }
        bool const outputCurvesOk =
            _PrependApiSchema(outputCurves, "UsdGenCurveAPI") &&
            _SetResetXformStack(outputCurves) &&
            _SetAttr(outputCurves, "usdGen:pomade:outputOwned", SdfValueTypeNames->Bool,
                     true, SdfVariabilityUniform, /*custom*/ true) &&
            _SetAttr(outputCurves, "visibility", SdfValueTypeNames->Token,
                     TfToken("invisible"), SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "type", SdfValueTypeNames->Token, TfToken("cubic"),
                     SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "basis", SdfValueTypeNames->Token, TfToken("bspline"),
                     SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "wrap", SdfValueTypeNames->Token, TfToken("pinned"),
                     SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "curveVertexCounts", SdfValueTypeNames->IntArray,
                     outputCounts) &&
            _SetAttr(outputCurves, "points", SdfValueTypeNames->Point3fArray, outputPoints) &&
            _SetAttr(outputCurves, "primvars:usdGen:role", SdfValueTypeNames->Token,
                     TfToken("guide"), SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "primvars:usdGen:curveId", SdfValueTypeNames->UInt64Array,
                     outputIds, SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "primvars:usdGen:rootFrame", SdfValueTypeNames->Matrix4dArray,
                     outputFrames, SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "primvars:tubeId", SdfValueTypeNames->IntArray,
                     outputTubeIds, SdfVariabilityUniform, /*custom*/ true) &&
            _SetAttr(outputCurves, "primvars:hierarchyLevel",
                     SdfValueTypeNames->IntArray, outputLevels,
                     SdfVariabilityUniform, /*custom*/ true) &&
            _SetAttr(outputCurves, "primvars:regionId",
                     SdfValueTypeNames->IntArray, outputRegionIds,
                     SdfVariabilityUniform, /*custom*/ true) &&
            _SetAttr(outputCurves, "usdGen:surfaceCage:normalizedT",
                     SdfValueTypeNames->FloatArray, normalizedT, SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "usdGen:surfaceCage:ownerIds",
                     SdfValueTypeNames->IntArray, ownerIds, SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "usdGen:surfaceCage:ownerDensities",
                     SdfValueTypeNames->FloatArray, ownerDensities, SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "usdGen:surfaceCage:ownerSeeds",
                     SdfValueTypeNames->IntArray, ownerSeeds, SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "usdGen:surfaceCage:ownerCvCounts",
                     SdfValueTypeNames->IntArray, ownerCvCounts, SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "usdGen:surfaceCage:ownerEdgeBias",
                     SdfValueTypeNames->FloatArray, ownerEdgeBias, SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "usdGen:surfaceCage:ownerChartCentroids",
                     SdfValueTypeNames->Float2Array, ownerChartCentroids,
                     SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "usdGen:surfaceCage:ownerChartMeanRadii",
                     SdfValueTypeNames->FloatArray, ownerChartMeanRadii,
                     SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "usdGen:surfaceCage:ownerLengthProfileOffsets",
                     SdfValueTypeNames->IntArray, ownerProfileOffsets, SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "usdGen:surfaceCage:ownerLengthProfile",
                     SdfValueTypeNames->Float2Array, ownerProfile, SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "usdGen:surfaceCage:triangles",
                     SdfValueTypeNames->Int3Array, triangles, SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "usdGen:surfaceCage:triangleOwnerIndices",
                     SdfValueTypeNames->IntArray, triangleOwners, SdfVariabilityUniform) &&
            _SetAttr(outputCurves, "usdGen:surfaceCage:triangleRootCharts",
                     SdfValueTypeNames->Float2Array, triangleRootCharts,
                     SdfVariabilityUniform) &&
            _SetInterpolation(outputCurves, "primvars:usdGen:curveId", "uniform") &&
            _SetInterpolation(outputCurves, "primvars:usdGen:rootFrame", "uniform") &&
            _SetInterpolation(outputCurves, "primvars:tubeId", "uniform") &&
            _SetInterpolation(outputCurves, "primvars:hierarchyLevel", "uniform") &&
            _SetInterpolation(outputCurves, "primvars:regionId", "uniform");
        bool const outputMapOk =
            _SetAttr(outputMap, "usdGen:pomade:outputOwned", SdfValueTypeNames->Bool,
                     true, SdfVariabilityUniform, /*custom*/ true) &&
            _SetAttr(outputMap, "usdGen:map:file", SdfValueTypeNames->Asset,
                     SdfAssetPath(outputMapFile)) &&
            _SetAttr(outputMap, "usdGen:map:filter", SdfValueTypeNames->Token,
                     TfToken("nearest")) &&
            _SetAttr(outputMap, "usdGen:map:blur", SdfValueTypeNames->Float,
                     0.0f) &&
            _SetAttr(outputMap, "usdGen:map:firstChannel", SdfValueTypeNames->Int, 0) &&
            _SetAttr(outputMap, "usdGen:map:channelCount", SdfValueTypeNames->Int, 1) &&
            _SetAttr(outputMap, "usdGen:map:clamp", SdfValueTypeNames->Float2,
                     GfVec2f(0.0f, 0.0f)) &&
            _SetAttr(outputMap, "usdGen:map:textureGeneration",
                     SdfValueTypeNames->UInt64, mapGeneration);
        if (!outputCurvesOk || !outputMapOk) {
            return fail("PomadeBuildCommitLayer: cannot author Output cage fields");
        }

        SdfPrimSpecHandle output = _EnsurePrim(layer, paths.OutputPath(),
                                                SdfSpecifierDef, "UsdGenDescription");
        SdfPrimSpecHandle ops = _EnsurePrim(layer, paths.OutputPath().AppendChild(TfToken("Ops")),
                                             SdfSpecifierDef, "Scope");
        SdfPrimSpecHandle width = _EnsurePrim(layer, paths.OutputPath().AppendChild(TfToken("Ops"))
                                               .AppendChild(TfToken("width")),
                                               SdfSpecifierDef, "UsdGenWidth");
        SdfPrimSpecHandle source = _EnsurePrim(layer, paths.OutputPath().AppendChild(TfToken("Ops"))
                                                .AppendChild(TfToken("source")),
                                                SdfSpecifierDef, "UsdGenCurveSource");
        if (!output || !ops || !width || !source) {
            return fail("PomadeBuildCommitLayer: cannot author Output graph");
        }
        ops->SetNameChildrenOrder(TfTokenVector{TfToken("width"), TfToken("source")});
        bool const outputGraphOk =
            _SetResetXformStack(output) &&
            _SetAttr(output, "usdGen:pomade:outputOwned", SdfValueTypeNames->Bool,
                     true, SdfVariabilityUniform, /*custom*/ true) &&
            _SetRelTargets(output, "usdGen:surface", SdfPathVector{outputScalpPath}) &&
            _SetAttr(width, "usdGen:width", SdfValueTypeNames->Float, snapshot.outputWidth) &&
            _SetAttr(width, "usdGen:replace", SdfValueTypeNames->Bool, true) &&
            _SetRelTargets(source, "usdGen:curves", SdfPathVector{paths.OutputCurvesPath()}) &&
            _SetAttr(source, "usdGen:interpolationMode", SdfValueTypeNames->Token,
                     TfToken("surfaceCage")) &&
            _SetAttr(source, "usdGen:densityMultiplier", SdfValueTypeNames->Float,
                     snapshot.outputDensityMultiplier) &&
            _SetRelTargets(source, "usdGen:regionMap", SdfPathVector{paths.OutputRegionMapPath()}) &&
            _SetAttr(source, "usdGen:regionMapChannel", SdfValueTypeNames->Int, 0) &&
            _SetAttr(source, "usdGen:expectMapGeneration", SdfValueTypeNames->UInt64,
                     mapGeneration) &&
            _SetAttr(source, "usdGen:useRest", SdfValueTypeNames->Bool, false) &&
            _SetAttr(source, "usdGen:resampleTo", SdfValueTypeNames->Int, 0);
        if (!outputGraphOk) {
            return fail("PomadeBuildCommitLayer: cannot author Output graph attributes");
        }
    }

    // -- region maps + expressions (P2: one level; P4: per level) ---------
    // The versioned .ptx is written by the bake worker; the §3.1 build
    // re-authors the CURRENT file (from the snapshot) so a guide swap never
    // wipes the baked reference. The bake swap is what advances it.
    int const levelCount = std::max(snapshot.graph.levelCount, 1);
    for (int level = 1; level <= levelCount; ++level) {
        SdfPath const mapPath = paths.LevelMapPath(level);
        SdfPath const exprPath = paths.LevelExprPath(level);
        SdfPrimSpecHandle regionMap =
            _EnsurePrim(layer, mapPath, SdfSpecifierDef, "UsdGenPtexMap");
        if (!regionMap) {
            return fail(
                "PomadeBuildCommitLayer: cannot author the RegionMap");
        }
        // usdGen:map:clamp is (0, 0): region ids are -1..N, and both the
        // schema and the evaluator default to [0, 1], which would merge
        // uncovered strands into region 0. Equal components disable it.
        bool mapOk =
            _SetAttr(regionMap, "usdGen:map:filter", SdfValueTypeNames->Token,
                     TfToken("nearest")) &&
            _SetAttr(regionMap, "usdGen:map:firstChannel",
                     SdfValueTypeNames->Int, level - 1) &&
            _SetAttr(regionMap, "usdGen:map:channelCount",
                     SdfValueTypeNames->Int, 1) &&
            _SetAttr(regionMap, "usdGen:map:clamp",
                     SdfValueTypeNames->Float2, GfVec2f(0.0f, 0.0f));
        if (!snapshot.graph.bakedMapFile.empty()) {
            mapOk = mapOk &&
                    _SetAttr(regionMap, "usdGen:map:file",
                             SdfValueTypeNames->Asset,
                             SdfAssetPath(snapshot.graph.bakedMapFile));
        }
        if (!mapOk) {
            return fail(
                "PomadeBuildCommitLayer: cannot author RegionMap fields");
        }
        SdfPrimSpecHandle regionExpr = _EnsurePrim(
            layer, exprPath, SdfSpecifierDef, "UsdGenExpression");
        if (!regionExpr) {
            return fail(
                "PomadeBuildCommitLayer: cannot author the RegionExpr");
        }
        SdfAttributeSpecHandle outputsResult = SdfAttributeSpec::New(
            regionExpr, "outputs:result", SdfValueTypeNames->Float,
            SdfVariabilityVarying, /*custom*/ true);
        if (!outputsResult) {
            return fail(
                "PomadeBuildCommitLayer: cannot author outputs:result");
        }
        if (!_SetAttr(regionExpr, "usdGen:expr:source",
                      SdfValueTypeNames->String,
                      std::string("ptex(\"regionMap\")"),
                      SdfVariabilityUniform, /*custom*/ true) ||
            !_SetRelTargets(regionExpr, "input:regionMap",
                            SdfPathVector{mapPath})) {
            return fail(
                "PomadeBuildCommitLayer: cannot author the RegionExpr");
        }
    }

    // -- GuideInterpolate fill-in (only what the plan allows) ---------------
    if (!paths.descriptionPath.IsEmpty() &&
        (snapshot.createInterpOp || snapshot.setInterpGuides ||
         snapshot.setInterpRegion)) {
        SdfPrimSpecHandle op;
        if (snapshot.createInterpOp) {
            if (!_EnsurePrim(layer, paths.descriptionPath, SdfSpecifierOver,
                             nullptr)) {
                return fail("PomadeBuildCommitLayer: cannot over the "
                            "description");
            }
            SdfPath const opsPath =
                paths.descriptionPath.AppendChild(TfToken("Ops"));
            if (!_EnsurePrim(layer, opsPath, SdfSpecifierDef, "Scope")) {
                return fail("PomadeBuildCommitLayer: cannot author the Ops "
                            "scope");
            }
            op = _EnsurePrim(layer, snapshot.interpOpPath, SdfSpecifierDef,
                             "UsdGenGuideInterpolate");
        } else {
            op = _EnsurePrim(layer, snapshot.interpOpPath, SdfSpecifierOver,
                             nullptr);
        }
        if (!op) {
            return fail("PomadeBuildCommitLayer: cannot author the "
                        "GuideInterpolate op");
        }
        if (snapshot.setInterpGuides &&
            !_SetRelTargets(op, "usdGen:guides",
                            SdfPathVector{paths.GuidesPath()})) {
            return fail("PomadeBuildCommitLayer: cannot connect usdGen:guides");
        }
        if (snapshot.setInterpRegion &&
            !_SetConnections(op, "usdGen:region", SdfValueTypeNames->Float,
                            SdfPathVector{paths.RegionExprPath()})) {
            return fail("PomadeBuildCommitLayer: cannot connect usdGen:region");
        }
    }

    *outLayer = layer;
    return true;
}

PomadeFillPlan
PomadePlanGuideInterpolateFill(UsdStagePtr const &stage,
                              PomadeCommitPaths const &paths)
{
    PomadeFillPlan plan;
    plan.opPath = paths.InterpOpPath();
    if (!stage || paths.descriptionPath.IsEmpty()) {
        return plan;
    }
    UsdPrim const desc = stage->GetPrimAtPath(paths.descriptionPath);
    if (!desc) {
        return plan;
    }
    // Any UsdGenGuideInterpolate under <desc>/Ops counts, whatever its name:
    // the tool fills in, it never authors a second op beside an artist's.
    UsdPrim existing;
    UsdPrim const ops =
        stage->GetPrimAtPath(paths.descriptionPath.AppendChild(TfToken("Ops")));
    if (ops) {
        for (UsdPrim const &child : ops.GetChildren()) {
            if (child.GetTypeName() == TfToken("UsdGenGuideInterpolate")) {
                existing = child;
                break;
            }
        }
    }
    if (!existing) {
        plan.createInterpOp = true;
        plan.setInterpGuides = true;
        plan.setInterpRegion = true;
        return plan;
    }
    plan.opPath = existing.GetPath();
    if (_Targets(existing.GetRelationship(TfToken("usdGen:guides"))).empty()) {
        plan.setInterpGuides = true;
    }
    UsdAttribute const region =
        existing.GetAttribute(TfToken("usdGen:region"));
    if (!region.HasAuthoredValueOpinion() &&
        _Connections(region).empty()) {
        plan.setInterpRegion = true;
    }
    return plan;
}

namespace {

// Whether a stored residual carries any authoring at all (the K6 "zero
// sculpt stays zero" rule compares against exact 0.0 the same way).
bool
_AnyShapeDelta(PomadeShapeDeltas const &d)
{
    auto any = [](std::vector<float> const &values) {
        for (float x : values) {
            if (x != 0.0f) {
                return true;
            }
        }
        return false;
    };
    if (any(d.centerDu) || any(d.centerDv) || any(d.centerDw)) {
        return true;
    }
    for (PomadeTubeSection const &sec : d.sections) {
        if (sec.scale != 0.0f || sec.twist != 0.0f || any(sec.u) ||
            any(sec.v)) {
            return true;
        }
    }
    return false;
}

} // namespace

bool
PomadeResolveScalpTarget(UsdStagePtr const &stage, SdfPath const &path,
                        PomadeScalpTarget *out, std::string *err)
{
    auto fail = [&](std::string const &what) {
        if (err) {
            *err = what;
        }
        return false;
    };
    if (!stage || !out) {
        return fail("PomadeResolveScalpTarget: null stage or output");
    }
    *out = PomadeScalpTarget();
    out->targetPath = path;
    UsdPrim const prim = stage->GetPrimAtPath(path);
    if (!prim) {
        return fail("scalp link names no prim at " + path.GetString());
    }
    UsdPrim meshPrim = prim;
    if (prim.IsA<UsdGeomSubset>()) {
        // Rule 1: a face subset under its Mesh, nothing else. The token is
        // read as authored (the schema fallback is "face").
        UsdGeomSubset const subset(prim);
        TfToken elementType;
        subset.GetElementTypeAttr().Get(&elementType);
        if (elementType != UsdGeomTokens->face) {
            return fail("scalp GeomSubset " + path.GetString() +
                        " has elementType \"" + elementType.GetString() +
                        "\"; a scalp subset must be \"face\"");
        }
        meshPrim = prim.GetParent();
        if (!meshPrim || !meshPrim.IsA<UsdGeomMesh>()) {
            return fail("scalp GeomSubset " + path.GetString() +
                        " is not a child of a Mesh");
        }
        VtIntArray indices;
        subset.GetIndicesAttr().Get(&indices);
        if (indices.empty()) {
            return fail("scalp GeomSubset " + path.GetString() +
                        " names no faces");
        }
        out->activeFaces.assign(indices.begin(), indices.end());
        out->isSubset = true;
    } else if (!prim.IsA<UsdGeomMesh>()) {
        return fail("scalp link " + path.GetString() + " names a " +
                    prim.GetTypeName().GetString() +
                    ", not a Mesh or a face GeomSubset");
    }
    out->meshPath = meshPrim.GetPath();
    UsdGeomMesh const mesh(meshPrim);
    VtVec3fArray pts;
    VtIntArray counts, indices;
    if (!mesh.GetPointsAttr().Get(&pts) ||
        !mesh.GetFaceVertexCountsAttr().Get(&counts) ||
        !mesh.GetFaceVertexIndicesAttr().Get(&indices)) {
        return fail("scalp mesh " + out->meshPath.GetString() +
                    " has no readable topology");
    }
    out->points.resize(pts.size() * 3);
    for (size_t i = 0; i < pts.size(); ++i) {
        out->points[i * 3 + 0] = pts[i][0];
        out->points[i * 3 + 1] = pts[i][1];
        out->points[i * 3 + 2] = pts[i][2];
    }
    out->faceVertexCounts.assign(counts.begin(), counts.end());
    out->faceVertexIndices.assign(indices.begin(), indices.end());
    // Rule 2: indices are parent-mesh face ids. Union semantics tolerate a
    // repeat; a face the parent does not have is an error, not a clamp.
    std::vector<int> &active = out->activeFaces;
    std::sort(active.begin(), active.end());
    active.erase(std::unique(active.begin(), active.end()), active.end());
    if (!active.empty() &&
        (active.front() < 0 ||
         size_t(active.back()) >= out->faceVertexCounts.size())) {
        int const bad = active.front() < 0 ? active.front() : active.back();
        return fail("scalp GeomSubset " + path.GetString() + " names face " +
                    std::to_string(bad) + ", but " +
                    out->meshPath.GetString() + " has " +
                    std::to_string(out->faceVertexCounts.size()) + " faces");
    }
    return true;
}

PomadeHydrateResult
PomadeHydrateModel(UsdStagePtr const &stage, SdfPath const &groomPath,
                  PomadeModel *model)
{
    PomadeHydrateResult result;
    auto fail = [&](std::string const &what) {
        result.diagnostic = "PomadeHydrateModel: " + what;
        return result;
    };
    if (!stage || !model) {
        return fail("null stage or model");
    }
    UsdPrim const groom = stage->GetPrimAtPath(groomPath);
    if (!groom || groom.GetTypeName() != TfToken("UsdGenPomadeGroom")) {
        return fail("no UsdGenPomadeGroom at " + groomPath.GetString());
    }
    TfToken version;
    if (!groom.GetAttribute(TfToken("usdGen:pomade:version")).Get(&version) ||
        version != "1") {
        return fail("unsupported groom version (want \"1\")");
    }
    // Missing means the layer predates the subface root sampler. A known
    // marker selects exactly the generator that wrote the guide bytes;
    // unknown markers are not safe to hydrate.
    enum class _RootSampler { LegacyFaceDisc, RegionV1, RegionV2, RegionV3 };
    _RootSampler sampler = _RootSampler::LegacyFaceDisc;
    UsdAttribute const rootSamplerAttr =
        groom.GetAttribute(TfToken("usdGen:pomade:rootSampler"));
    bool const legacyRootSampler = !rootSamplerAttr;
    if (!legacyRootSampler) {
        TfToken rootSampler;
        if (!rootSamplerAttr.Get(&rootSampler)) {
            return fail("unsupported root sampler");
        }
        if (rootSampler == "region-v1") {
            sampler = _RootSampler::RegionV1;
        } else if (rootSampler == "region-v2") {
            sampler = _RootSampler::RegionV2;
        } else if (rootSampler == "region-v3") {
            sampler = _RootSampler::RegionV3;
        } else {
            return fail("unsupported root sampler (want \"region-v1\" or "
                        "\"region-v2\" or \"region-v3\")");
        }
    }
    // Missing means the layer predates K14's ring-slot alignment: its
    // stored child residuals were measured against a derivation whose slot
    // order the current K14 no longer reproduces. Those are re-measured
    // from the stored (authoritative) actual against today's derivation
    // below; a marked layer installs its residuals verbatim (bit-exact).
    UsdAttribute const subdividerAttr =
        groom.GetAttribute(TfToken("usdGen:pomade:subdivider"));
    bool const legacySubdivider = !subdividerAttr;
    if (!legacySubdivider) {
        TfToken subdivider;
        if (!subdividerAttr.Get(&subdivider) || subdivider != "aligned-v1") {
            return fail("unsupported subdivider (want \"aligned-v1\")");
        }
    }
    bool generatedCurvesSuppressed = false;
    UsdAttribute const generatedCurvesAttr =
        groom.GetAttribute(TfToken("usdGen:pomade:generatedCurvesSuppressed"));
    if (generatedCurvesAttr &&
        !generatedCurvesAttr.Get(&generatedCurvesSuppressed)) {
        return fail("cannot read generated-curve state");
    }
    PomadeModel::OutputSettings outputSettings;
    UsdAttribute const outputEnabledAttr =
        groom.GetAttribute(TfToken("usdGen:pomade:output:enabled"));
    UsdAttribute const outputDensityAttr = groom.GetAttribute(
        TfToken("usdGen:pomade:output:densityMultiplier"));
    UsdAttribute const outputWidthAttr =
        groom.GetAttribute(TfToken("usdGen:pomade:output:width"));
    UsdAttribute const outputPtexResolutionAttr = groom.GetAttribute(
        TfToken("usdGen:pomade:output:ptexResolution"));
    if ((outputEnabledAttr && !outputEnabledAttr.Get(&outputSettings.enabled)) ||
        (outputDensityAttr &&
         !outputDensityAttr.Get(&outputSettings.densityMultiplier)) ||
        (outputWidthAttr && !outputWidthAttr.Get(&outputSettings.width)) ||
        (outputPtexResolutionAttr &&
         !outputPtexResolutionAttr.Get(&outputSettings.ptexResolution))) {
        return fail("cannot read Output settings");
    }
    if (!model->SetOutputSettings(outputSettings)) {
        return fail(model->GetDiagnostic());
    }
    UsdPrim const graph =
        stage->GetPrimAtPath(groomPath.AppendChild(TfToken("ScalpGraph")));
    if (!graph || graph.GetTypeName() != TfToken("UsdGenScalpGraph")) {
        return fail("groom has no UsdGenScalpGraph shell");
    }
    // -- P2 graph hydrate -------------------------------------------------
    // The stage encoding is positional (nodes are parallel arrays; edges and
    // loops name indices); the model takes stable ids, so hydrate maps index
    // i to id i and verifies the re-extracted loops, colours and live
    // primvar against the stored opinions bit-exactly.
    VtIntArray nodeFaceIds;
    VtVec2fArray nodeUVs;
    VtVec2iArray storedEdges;
    VtIntArray regionCounts;
    VtIntArray regionIndices;
    VtVec3fArray storedColors;
    VtVec2iArray storedLinked;
    float storedSnapRadius = 0.05f;
    graph.GetAttribute(TfToken("usdGen:pomade:nodeFaceIds")).Get(&nodeFaceIds);
    graph.GetAttribute(TfToken("usdGen:pomade:nodeUVs")).Get(&nodeUVs);
    graph.GetAttribute(TfToken("usdGen:pomade:edges")).Get(&storedEdges);
    graph.GetAttribute(TfToken("usdGen:pomade:regionNodeCounts"))
        .Get(&regionCounts);
    graph.GetAttribute(TfToken("usdGen:pomade:regionNodeIndices"))
        .Get(&regionIndices);
    graph.GetAttribute(TfToken("usdGen:pomade:regionColors")).Get(&storedColors);
    graph.GetAttribute(TfToken("usdGen:pomade:linkedRegions")).Get(&storedLinked);
    graph.GetAttribute(TfToken("usdGen:pomade:snapRadius")).Get(&storedSnapRadius);
    if (nodeUVs.size() != nodeFaceIds.size()) {
        return fail("graph node arrays have mismatched lengths");
    }
    size_t loopTotal = 0;
    for (int c : regionCounts) {
        if (c < 0) {
            return fail("graph region counts are negative");
        }
        loopTotal += size_t(c);
    }
    if (loopTotal != regionIndices.size()) {
        return fail("graph region indices do not match the counts");
    }
    SdfPathVector scalpTargets =
        _Targets(groom.GetRelationship(TfToken("usdGen:pomade:scalp")));
    bool haveScalp = false;
    // The Mesh behind the link: the linked prim itself, or a face
    // GeomSubset's parent, which carries the live primvar (plan/02 §2.20).
    UsdPrim scalpPrim;
    if (!scalpTargets.empty()) {
        PomadeScalpTarget target;
        std::string targetErr;
        if (!PomadeResolveScalpTarget(stage, scalpTargets[0], &target,
                                     &targetErr)) {
            return fail(targetErr);
        }
        scalpPrim = stage->GetPrimAtPath(target.meshPath);
        if (!model->BindScalp(target.points, target.faceVertexCounts,
                              target.faceVertexIndices,
                              target.activeFaces)) {
            return fail(model->GetDiagnostic());
        }
        haveScalp = true;
    }
    if (!nodeFaceIds.empty()) {
        if (!haveScalp) {
            return fail("graph nodes but no scalp link");
        }
        std::shared_ptr<PomadeScalpMesh const> scalp = model->GetScalp();
        PomadeModel::GraphSnapshot gs;
        gs.snapRadius = storedSnapRadius;
        for (size_t i = 0; i < nodeFaceIds.size(); ++i) {
            PomadeGraphNode nd;
            nd.id = int(i);
            nd.faceId = nodeFaceIds[i];
            nd.u = nodeUVs[i][0];
            nd.v = nodeUVs[i][1];
            float px = 0.0f, py = 0.0f, pz = 0.0f;
            if (!PomadeFacePosition(*scalp, nd.faceId, nd.u, nd.v, &px, &py,
                                   &pz)) {
                return fail("graph node addresses no surface point");
            }
            nd.p[0] = px;
            nd.p[1] = py;
            nd.p[2] = pz;
            if (size_t(nd.faceId) >= scalp->faceVertexCounts.size()) {
                return fail("graph node face is out of range");
            }
            nd.n[0] = scalp->faceNormals[size_t(nd.faceId) * 3 + 0];
            nd.n[1] = scalp->faceNormals[size_t(nd.faceId) * 3 + 1];
            nd.n[2] = scalp->faceNormals[size_t(nd.faceId) * 3 + 2];
            gs.nodes.push_back(nd);
        }
        for (auto const &e : storedEdges) {
            if (e[0] < 0 || e[1] < 0 ||
                size_t(e[0]) >= gs.nodes.size() ||
                size_t(e[1]) >= gs.nodes.size()) {
                return fail("graph edge names absent nodes");
            }
            gs.edges.emplace_back(e[0], e[1]);
        }
        for (auto const &pr : storedLinked) {
            if (pr[0] < 0 || pr[1] < 0 ||
                size_t(pr[0]) >= regionCounts.size() ||
                size_t(pr[1]) >= regionCounts.size()) {
                return fail("linked regions name absent regions");
            }
            gs.linked.emplace_back(pr[0], pr[1]);
        }
        if (!model->RestoreGraph(gs)) {
            return fail(model->GetDiagnostic());
        }
        // The graph round-trip asserts: re-extracted loops, colours and the
        // re-rasterised live primvar equal the stored opinions bit-exactly.
        PomadeModel::GraphSnapshot check = model->SnapshotGraph();
        bool loopsEqual =
            check.regionLoops.size() == regionCounts.size();
        size_t flat = 0;
        for (size_t r = 0; loopsEqual && r < check.regionLoops.size(); ++r) {
            loopsEqual = check.regionLoops[r].size() ==
                         size_t(regionCounts[r]);
            for (size_t k = 0; loopsEqual && k < check.regionLoops[r].size();
                 ++k, ++flat) {
                loopsEqual =
                    check.regionLoops[r][k] == regionIndices[flat];
            }
        }
        loopsEqual = loopsEqual && flat == regionIndices.size();
        bool colorsEqual =
            check.regionColors.size() == storedColors.size() * 3;
        for (size_t i = 0; colorsEqual && i < storedColors.size(); ++i) {
            colorsEqual =
                check.regionColors[i * 3 + 0] == storedColors[i][0] &&
                check.regionColors[i * 3 + 1] == storedColors[i][1] &&
                check.regionColors[i * 3 + 2] == storedColors[i][2];
        }
        if (!loopsEqual || !colorsEqual) {
            return fail("re-extracted regions differ from the stored graph "
                        "(foreign or hand-edited graph data)");
        }
        result.graphNodeCount = check.nodes.size();
        result.graphRegionCount = check.regionLoops.size();
    }
    if (haveScalp && bool(scalpPrim)) {
        // The live primvar must equal the re-rasterised map bit-exactly
        // (empty graphs rasterise to all -1, which the committer authors
        // whenever a scalp is bound).
        VtIntArray storedPrimvar;
        bool const hasPrimvar =
            scalpPrim.GetAttribute(TfToken("primvars:usdGen:pomadeRegion"))
                .Get(&storedPrimvar);
        PomadeModel::GraphSnapshot check = model->SnapshotGraph();
        if (hasPrimvar &&
            (storedPrimvar.size() != check.faceRegions.size() ||
             !std::equal(storedPrimvar.begin(), storedPrimvar.end(),
                         check.faceRegions.begin()))) {
            return fail("re-rasterised regions differ from the stored "
                        "primvar (foreign or hand-edited map data)");
        }
    }
    result.graphRoundTrip = true;

    // -- V0b: the whole tube hierarchy (plan/17 §2.5, plan/18 §7 G3) -------
    //
    // Every level is REBUILT, never read as geometry: the L1 tube is
    // restored outright, each deeper level is re-derived from its parent
    // with the stored (tubeId, seed) subdivision and then takes the stored
    // shape + deltas verbatim. Bridge imports come back through
    // ImportLockedTube, on-the-fly parents through their members.
    UsdPrim const tubes =
        stage->GetPrimAtPath(groomPath.AppendChild(TfToken("Tubes")));
    if (!tubes) {
        return fail("groom has no Tubes scope");
    }
    struct _StageTube {
        UsdPrim prim;
        int level = 1;
        int childIndex = -1;
        int regionId = -1;
        bool locked = false;  // a child's locked bit marks a bridge import
        bool persistent = false;
        SdfPathVector members;
        PomadeModel::TubeSnapshot shape;
        PomadeShapeDeltas deltas;
        std::vector<PomadeParentBoundaryBinding> inheritedBoundaryBindings;
        bool hasInheritedBoundaryBindings = false;
    };
    auto readTube = [&](UsdPrim const &prim, _StageTube *out) -> std::string {
        VtVec3fArray centers;
        VtFloatArray sectionT;
        int ringVerts = 0;
        VtVec2fArray sectionCvs;
        VtVec2fArray sectionRawCvs;
        VtVec2fArray sectionTransforms;
        if (!prim.GetAttribute(TfToken("usdGen:pomade:centerPoints"))
                 .Get(&centers) ||
            centers.empty() ||
            !prim.GetAttribute(TfToken("usdGen:pomade:sectionT"))
                 .Get(&sectionT) ||
            !prim.GetAttribute(TfToken("usdGen:pomade:sectionCvCount"))
                 .Get(&ringVerts) ||
            !prim.GetAttribute(TfToken("usdGen:pomade:sectionCvs"))
                 .Get(&sectionCvs)) {
            return "tube is missing center/section attributes";
        }
        if (sectionCvs.empty()) {
            return "tube has no section CVs";
        }
        if (sectionCvs.size() != sectionT.size() * size_t(ringVerts)) {
            return "tube section CVs do not match sectionT x cvCount";
        }
        PomadeModel::TubeSnapshot &snapshot = out->shape;
        snapshot.hasTube = true;
        snapshot.shape.rings = int(centers.size());
        snapshot.shape.ringVerts = ringVerts;
        // Radius is the section-plane extent of the first ring CV; length
        // spans the center column. Both round-trip the committed tube.
        snapshot.shape.radius = std::sqrt(sectionCvs[0][0] * sectionCvs[0][0] +
                                          sectionCvs[0][1] * sectionCvs[0][1]);
        snapshot.centerX.resize(centers.size());
        snapshot.centerY.resize(centers.size());
        snapshot.centerZ.resize(centers.size());
        for (size_t i = 0; i < centers.size(); ++i) {
            snapshot.centerX[i] = centers[i][0];
            snapshot.centerY[i] = centers[i][1];
            snapshot.centerZ[i] = centers[i][2];
        }
        // Pre-frame-reference layers have no custom attribute. Preserve
        // their old K4 interpretation exactly by making that identity, then
        // reject malformed authored data before a child is re-derived or a
        // stored delta is installed against the wrong frame.
        snapshot.frameReference = _IdentityFrameReference();
        UsdAttribute const frameReferenceAttr =
            prim.GetAttribute(TfToken("usdGen:pomade:frameReference"));
        if (frameReferenceAttr) {
            VtFloatArray frameReference;
            if (frameReferenceAttr.Get(&frameReference)) {
                if (frameReference.size() != snapshot.frameReference.size()) {
                    return "tube frameReference must contain 9 floats";
                }
            } else if (frameReferenceAttr.HasAuthoredValueOpinion()) {
                return "tube frameReference must contain 9 floats";
            } else {
                // A property with no value is equivalent to the missing
                // optional legacy attribute, so leave identity installed.
                frameReference.clear();
            }
            for (size_t i = 0; i < frameReference.size(); ++i) {
                snapshot.frameReference[i] = frameReference[i];
            }
            if (!frameReference.empty() &&
                !_IsProperFrameReference(snapshot.frameReference)) {
                return "tube frameReference is not a proper rotation";
            }
        }
        if (sampler == _RootSampler::RegionV2 ||
            sampler == _RootSampler::RegionV3) {
            VtFloatArray rootFrame;
            if (!prim.GetAttribute(TfToken("usdGen:pomade:rootFramePinned"))
                     .Get(&snapshot.rootFramePinned) ||
                !prim.GetAttribute(TfToken("usdGen:pomade:rootFrame"))
                     .Get(&rootFrame) ||
                rootFrame.size() != 9) {
                return "tube is missing the region-v2 root frame";
            }
            snapshot.rootFrame.tx = rootFrame[0];
            snapshot.rootFrame.ty = rootFrame[1];
            snapshot.rootFrame.tz = rootFrame[2];
            snapshot.rootFrame.nx = rootFrame[3];
            snapshot.rootFrame.ny = rootFrame[4];
            snapshot.rootFrame.nz = rootFrame[5];
            snapshot.rootFrame.bx = rootFrame[6];
            snapshot.rootFrame.by = rootFrame[7];
            snapshot.rootFrame.bz = rootFrame[8];
        }
        snapshot.shape.length = centers.back()[1] - centers.front()[1];
        UsdAttribute const rawCvsAttr =
            prim.GetAttribute(TfToken("usdGen:pomade:sectionRawCvs"));
        UsdAttribute const transformsAttr =
            prim.GetAttribute(TfToken("usdGen:pomade:sectionTransforms"));
        bool const hasRawRepresentation = rawCvsAttr && transformsAttr &&
            rawCvsAttr.Get(&sectionRawCvs) &&
            transformsAttr.Get(&sectionTransforms) &&
            sectionRawCvs.size() == sectionCvs.size() &&
            sectionTransforms.size() == sectionT.size();
        // Pre-transform layers have only the baked pairs.  Keep that legacy
        // interpretation rather than attempting a lossy inverse transform.
        snapshot.sections.resize(sectionT.size());
        for (size_t r = 0; r < sectionT.size(); ++r) {
            PomadeTubeSection section;
            section.t = sectionT[r];
            if (hasRawRepresentation) {
                section.scale = sectionTransforms[r][0];
                section.twist = sectionTransforms[r][1];
            }
            section.u.resize(size_t(ringVerts));
            section.v.resize(size_t(ringVerts));
            for (int s = 0; s < ringVerts; ++s) {
                GfVec2f const cv =
                    hasRawRepresentation
                        ? sectionRawCvs[r * size_t(ringVerts) + size_t(s)]
                        : sectionCvs[r * size_t(ringVerts) + size_t(s)];
                section.u[size_t(s)] = cv[0];
                section.v[size_t(s)] = cv[1];
            }
            snapshot.sections[r] = std::move(section);
        }
        prim.GetAttribute(TfToken("usdGen:pomade:regionId")).Get(&out->regionId);
        prim.GetAttribute(TfToken("usdGen:pomade:level")).Get(&out->level);
        prim.GetAttribute(TfToken("usdGen:pomade:childIndex"))
            .Get(&out->childIndex);
        prim.GetAttribute(TfToken("usdGen:pomade:fill:density"))
            .Get(&snapshot.fill.density);
        prim.GetAttribute(TfToken("usdGen:pomade:fill:cvCount"))
            .Get(&snapshot.fill.cvCount);
        prim.GetAttribute(TfToken("usdGen:pomade:fill:seed"))
            .Get(&snapshot.fill.seed);
        prim.GetAttribute(TfToken("usdGen:pomade:fill:edgeBias"))
            .Get(&snapshot.fill.edgeBias);
        VtFloatArray lengthProfile;
        if (prim.GetAttribute(TfToken("usdGen:pomade:fill:lengthProfile"))
                .Get(&lengthProfile)) {
            snapshot.fill.lengthProfile.assign(lengthProfile.begin(),
                                               lengthProfile.end());
        }
        // The committed root sampler is groom-wide.  Old markerless and
        // v1/v2 layers must keep their historical K9 stream for strict
        // hydrate validation; v3 is the new material-aware sampler.
        snapshot.fill.sampler = sampler == _RootSampler::RegionV3
            ? PomadeGuideSampler::RegionV3
            : PomadeGuideSampler::Legacy;
        prim.GetAttribute(TfToken("usdGen:pomade:subdivide:count"))
            .Get(&snapshot.subdivide.count);
        prim.GetAttribute(TfToken("usdGen:pomade:subdivide:seed"))
            .Get(&snapshot.subdivide.seed);
        TfToken splitMode;
        if (prim.GetAttribute(TfToken("usdGen:pomade:subdivide:splitMode"))
                .Get(&splitMode)) {
            snapshot.subdivide.splitMode = splitMode.GetString();
        }
        prim.GetAttribute(TfToken("usdGen:pomade:locked")).Get(&snapshot.locked);
        prim.GetAttribute(TfToken("usdGen:pomade:lockParents"))
            .Get(&snapshot.lockParents);
        prim.GetAttribute(TfToken("usdGen:pomade:lockChildren"))
            .Get(&snapshot.lockChildren);
        out->locked = snapshot.locked;
        prim.GetAttribute(TfToken("usdGen:pomade:persistent"))
            .Get(&out->persistent);
        out->members =
            _Targets(prim.GetRelationship(TfToken("usdGen:pomade:members")));
        VtVec3fArray centerDeltas;
        VtVec2fArray sectionDeltas;
        VtVec2fArray sectionDeltaTransforms;
        VtIntArray inheritedBoundaryBindings;
        prim.GetAttribute(TfToken("usdGen:pomade:centerDeltas"))
            .Get(&centerDeltas);
        prim.GetAttribute(TfToken("usdGen:pomade:sectionDeltas"))
            .Get(&sectionDeltas);
        prim.GetAttribute(TfToken("usdGen:pomade:sectionDeltaTransforms"))
            .Get(&sectionDeltaTransforms);
        UsdAttribute const boundaryBindingsAttr = prim.GetAttribute(
            TfToken("usdGen:pomade:inheritedBoundaryBindings"));
        out->hasInheritedBoundaryBindings = bool(boundaryBindingsAttr) &&
            boundaryBindingsAttr.HasAuthoredValueOpinion();
        if (boundaryBindingsAttr &&
            !boundaryBindingsAttr.Get(&inheritedBoundaryBindings) &&
            boundaryBindingsAttr.HasAuthoredValueOpinion()) {
            return "tube inheritedBoundaryBindings must be an int array";
        }
        if (inheritedBoundaryBindings.size() % 3 != 0) {
            return "tube inheritedBoundaryBindings must contain triples";
        }
        for (size_t i = 0; i < inheritedBoundaryBindings.size(); i += 3) {
            PomadeParentBoundaryBinding binding;
            binding.section = inheritedBoundaryBindings[i + 0];
            binding.parentSlot = inheritedBoundaryBindings[i + 1];
            binding.childSlot = inheritedBoundaryBindings[i + 2];
            if (binding.section < 0 ||
                binding.section >= int(sectionT.size()) ||
                binding.parentSlot < 0 || binding.childSlot < 0 ||
                binding.childSlot >= ringVerts) {
                return "tube inheritedBoundaryBindings has an invalid slot";
            }
            out->inheritedBoundaryBindings.push_back(binding);
        }
        for (auto const &d : centerDeltas) {
            out->deltas.centerDu.push_back(d[0]);
            out->deltas.centerDv.push_back(d[1]);
            out->deltas.centerDw.push_back(d[2]);
        }
        if (!sectionDeltas.empty() &&
            sectionDeltas.size() == sectionT.size() * size_t(ringVerts)) {
            out->deltas.sections.resize(sectionT.size());
            for (size_t r = 0; r < sectionT.size(); ++r) {
                PomadeTubeSection &sec = out->deltas.sections[r];
                sec.t = sectionT[r];
                sec.scale = 0.0f;
                sec.twist = 0.0f;
                if (sectionDeltaTransforms.size() == sectionT.size()) {
                    sec.scale = sectionDeltaTransforms[r][0];
                    sec.twist = sectionDeltaTransforms[r][1];
                }
                sec.u.resize(size_t(ringVerts));
                sec.v.resize(size_t(ringVerts));
                for (int s = 0; s < ringVerts; ++s) {
                    GfVec2f const d =
                        sectionDeltas[r * size_t(ringVerts) + size_t(s)];
                    sec.u[size_t(s)] = d[0];
                    sec.v[size_t(s)] = d[1];
                }
            }
        }
        out->prim = prim;
        return std::string();
    };
    auto tubeChildren = [](UsdPrim const &prim) {
        std::vector<UsdPrim> out;
        for (UsdPrim const &child : prim.GetChildren()) {
            if (child.GetTypeName() == TfToken("UsdGenTube")) {
                out.push_back(child);
            }
        }
        return out;
    };

    std::vector<UsdPrim> const roots = tubeChildren(tubes);
    // V6: one L1 root per region (plan/18 §7 G14). Their ids are carried by
    // the prim names the committer wrote (`tube<id>`), which is what keeps a
    // hydrated groom's guide tubeIds equal to the committed ones.
    std::vector<std::pair<int, UsdPrim>> l1Prims;
    std::vector<UsdPrim> groupPrims;
    for (UsdPrim const &root : roots) {
        SdfPathVector const members =
            _Targets(root.GetRelationship(TfToken("usdGen:pomade:members")));
        if (!members.empty()) {
            groupPrims.push_back(root);
            continue;
        }
        std::string const name = root.GetName().GetString();
        int id = -1;
        if (name.rfind("tube", 0) == 0 && name.size() > 4) {
            char *end = nullptr;
            long const parsed = std::strtol(name.c_str() + 4, &end, 10);
            if (end && *end == 0 && parsed >= 0 && parsed < (1 << 24)) {
                id = int(parsed);
            }
        }
        if (id < 0 || (id % 16) != 0) {
            return fail("an L1 UsdGenTube is not named tube<id> with an L1 "
                        "root id (0, or a multiple of 16)");
        }
        l1Prims.emplace_back(id, root);
    }
    if (l1Prims.empty()) {
        return fail("groom has no L1 UsdGenTube");
    }
    std::sort(l1Prims.begin(), l1Prims.end(),
              [](std::pair<int, UsdPrim> const &a,
                 std::pair<int, UsdPrim> const &b) {
                  return a.first < b.first;
              });
    if (l1Prims.front().first != 0) {
        return fail("the groom's L1 roots do not include tube0");
    }
    for (size_t i = 1; i < l1Prims.size(); ++i) {
        if (l1Prims[i].first == l1Prims[i - 1].first) {
            return fail("two L1 UsdGenTubes carry the same id");
        }
    }
    // Group prims mint negative ids in name order, which is the order the
    // committer wrote them: hydrating the same way re-mints the same ids.
    std::sort(groupPrims.begin(), groupPrims.end(),
              [](UsdPrim const &a, UsdPrim const &b) {
                  return a.GetPath() < b.GetPath();
              });

    UsdPrim const l1Prim = l1Prims.front().second;
    _StageTube l1;
    if (std::string const rerr = readTube(l1Prim, &l1); !rerr.empty()) {
        return fail(rerr);
    }
    model->ClearHierarchy();
    if (!model->Restore(l1.shape)) {
        return fail(model->GetDiagnostic());
    }
    model->SetTubeRegionId(l1.regionId);
    model->SetFreezeRoots(false);
    result.tubeCount = 1;

    // Breadth-first: a parent has to exist (and be subdivided) before its
    // children can be restored.
    std::map<SdfPath, int> idOfPath;
    idOfPath[l1Prim.GetPath()] = 0;
    std::vector<std::pair<UsdPrim, int>> queue{{l1Prim, 0}};
    for (size_t i = 1; i < l1Prims.size(); ++i) {
        _StageTube extra;
        if (std::string const rerr = readTube(l1Prims[i].second, &extra);
            !rerr.empty()) {
            return fail(rerr);
        }
        PomadeModel::TubeRecord install;
        install.actual = PomadeTubeDescFromSnapshot(extra.shape);
        install.actual.regionId = extra.regionId;
        install.subdivide = extra.shape.subdivide;
        install.fill = extra.shape.fill;
        install.lockParents = extra.shape.lockParents;
        install.lockChildren = extra.shape.lockChildren;
        install.hasTube = true;
        if (!model->InstallL1Tube(l1Prims[i].first, install)) {
            return fail(model->GetDiagnostic());
        }
        idOfPath[l1Prims[i].second.GetPath()] = l1Prims[i].first;
        ++result.tubeCount;
        queue.emplace_back(l1Prims[i].second, l1Prims[i].first);
    }
    for (size_t head = 0; head < queue.size(); ++head) {
        UsdPrim const parentPrim = queue[head].first;
        int const parentId = queue[head].second;
        std::vector<UsdPrim> children = tubeChildren(parentPrim);
        if (children.empty()) {
            continue;
        }
        std::vector<_StageTube> records(children.size());
        for (size_t i = 0; i < children.size(); ++i) {
            if (std::string const rerr = readTube(children[i], &records[i]);
                !rerr.empty()) {
                return fail(rerr);
            }
        }
        std::vector<size_t> order(children.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return records[a].childIndex < records[b].childIndex;
        });
        int subdivided = 0;
        for (size_t i : order) {
            if (!records[i].locked) {
                ++subdivided;
            }
        }
        if (subdivided > 0) {
            std::string splitMode = "kmeans";
            int seed = 0;
            int parentCount = 0;
            if (parentId == 0) {
                splitMode = l1.shape.subdivide.splitMode;
                seed = l1.shape.subdivide.seed;
                parentCount = l1.shape.subdivide.count;
            } else {
                PomadeModel::TubeRecord parentRecord;
                if (!model->GetTubeRecord(parentId, &parentRecord)) {
                    return fail("a committed parent tube is missing from the "
                                "model");
                }
                splitMode = parentRecord.subdivide.splitMode;
                seed = parentRecord.subdivide.seed;
                parentCount = parentRecord.subdivide.count;
            }
            if (parentCount != subdivided) {
                return fail("a tube's subdivide:count does not match its "
                            "committed child count (foreign or hand-edited "
                            "hierarchy)");
            }
            std::vector<int> kids;
            if (!model->SubdivideTube(parentId, parentCount, splitMode.c_str(),
                                      seed, &kids)) {
                return fail(model->GetDiagnostic());
            }
        }
        for (size_t i : order) {
            _StageTube const &record = records[i];
            int childId = 0;
            if (record.locked) {
                // A bridge import: explicit geometry, no derivation. It is
                // created after the subdivided siblings, so its childIndex
                // (and therefore its id) continues past them.
                std::vector<float> secT, secU, secV;
                for (auto const &sec : record.shape.sections) {
                    secT.push_back(sec.t);
                    for (size_t s = 0; s < sec.u.size(); ++s) {
                        secU.push_back(sec.u[s]);
                        secV.push_back(sec.v[s]);
                    }
                }
                if (!model->ImportLockedTube(
                        parentId, record.shape.centerX.data(),
                        record.shape.centerY.data(),
                        record.shape.centerZ.data(),
                        int(record.shape.centerX.size()), secT.data(),
                        secU.data(), secV.data(), int(secT.size()),
                        record.shape.shape.ringVerts, &childId)) {
                    return fail(model->GetDiagnostic());
                }
            } else {
                childId = parentId * 16 + 1 + record.childIndex;
            }
            PomadeModel::TubeRecord install;
            if (!model->GetTubeRecord(childId, &install)) {
                return fail("a committed child tube has no re-derived "
                            "counterpart (foreign or hand-edited hierarchy)");
            }
            install.actual = PomadeTubeDescFromSnapshot(record.shape);
            install.actual.inheritedBoundaryBindings =
                record.inheritedBoundaryBindings;
            install.hasInheritedBoundaryBindings =
                record.hasInheritedBoundaryBindings;
            // regionId is authored per tube, not carried by the shape.
            install.actual.regionId = record.regionId;
            install.deltas = record.deltas;
            install.subdivide = record.shape.subdivide;
            install.fill = record.shape.fill;
            install.lockParents = record.shape.lockParents;
            install.lockChildren = record.shape.lockChildren;
            install.persistent = record.persistent;
            install.imported = record.locked;
            if (!model->RestoreTubeRecord(childId, install)) {
                return fail(model->GetDiagnostic());
            }
            if (legacySubdivider && !record.locked &&
                _AnyShapeDelta(record.deltas)) {
                // Pre-alignment K14: the stored residual names the old
                // slot order, and K6 would add it to today's slots (a
                // twisted child, or a refused parent edit). Re-measure it
                // from the stored actual against the derivation just
                // restored. Zero residuals stay exactly zero (untouched).
                PomadeModel::TubeRecord restored;
                std::vector<PomadeFrame> frames;
                std::string derr;
                if (!model->GetTubeRecord(childId, &restored) ||
                    !PomadeTubeFramesCpu(restored.derived, &frames, &derr) ||
                    !PomadeComputeDeltasCpu(restored.actual, restored.derived,
                                           frames, &restored.deltas,
                                           &derr)) {
                    return fail("cannot re-measure a pre-alignment child "
                                "residual" +
                                (derr.empty() ? std::string()
                                              : ": " + derr));
                }
                // The actual already carries its bindings (or the derived
                // ones, for a legacy layer): keep them as installed.
                restored.hasInheritedBoundaryBindings = true;
                if (!model->RestoreTubeRecord(childId, restored)) {
                    return fail(model->GetDiagnostic());
                }
            }
            idOfPath[record.prim.GetPath()] = childId;
            ++result.tubeCount;
            queue.emplace_back(record.prim, childId);
        }
    }

    // On-the-fly parents the artist kept (§5.4 "Make persistent").
    for (UsdPrim const &groupPrim : groupPrims) {
        _StageTube group;
        if (std::string const rerr = readTube(groupPrim, &group); !rerr.empty()) {
            return fail(rerr);
        }
        std::vector<int> members;
        for (SdfPath const &member : group.members) {
            auto const it = idOfPath.find(member);
            if (it == idOfPath.end()) {
                return fail("an on-the-fly parent names a tube that is not "
                            "in the groom");
            }
            members.push_back(it->second);
        }
        int groupId = 0;
        if (!model->GroupTubes(members, /*transientParent*/ false, &groupId)) {
            return fail(model->GetDiagnostic());
        }
        if (group.persistent && !model->MakeTubePersistent(groupId)) {
            return fail(model->GetDiagnostic());
        }
        PomadeModel::TubeRecord install;
        if (!model->GetTubeRecord(groupId, &install)) {
            return fail("the re-created on-the-fly parent is missing");
        }
        install.actual = PomadeTubeDescFromSnapshot(group.shape);
        install.actual.regionId = group.regionId;
        install.subdivide = group.shape.subdivide;
        install.fill = group.shape.fill;
        install.lockParents = group.shape.lockParents;
        install.lockChildren = group.shape.lockChildren;
        install.persistent = group.persistent;
        install.members = members;
        if (!model->RestoreTubeRecord(groupId, install)) {
            return fail(model->GetDiagnostic());
        }
        idOfPath[groupPrim.GetPath()] = groupId;
        ++result.tubeCount;
    }

    // -- guides: regenerate, assert, then import what no tube claims -------
    model->SetGeneratedCurvesSuppressed(generatedCurvesSuppressed);
    UsdPrim const guides =
        stage->GetPrimAtPath(groomPath.AppendChild(TfToken("Guides")));
    VtVec3fArray storedPoints;
    VtIntArray storedCounts;
    VtUInt64Array storedIds;
    VtIntArray storedTubeIds;
    bool const guidesReadable =
        bool(guides) &&
        guides.GetAttribute(TfToken("points")).Get(&storedPoints) &&
        guides.GetAttribute(TfToken("curveVertexCounts")).Get(&storedCounts) &&
        guides.GetAttribute(TfToken("primvars:usdGen:curveId")).Get(&storedIds);
    if (!guidesReadable) {
        return fail("groom has no readable Guides prim");
    }
    guides.GetAttribute(TfToken("primvars:tubeId")).Get(&storedTubeIds);
    PomadeSnapshot const check = PomadeSnapshotFromModel(*model);
    PomadeSnapshotGuides const regen =
        sampler == _RootSampler::RegionV1
            ? _RegionV1GuidesFromSnapshot(check)
            : PomadeGuidesFromSnapshot(check);
    // Stored offsets per curve, and which curves a model tube claims.
    std::vector<size_t> storedOffset(storedCounts.size(), 0);
    size_t running = 0;
    for (size_t i = 0; i < storedCounts.size(); ++i) {
        storedOffset[i] = running;
        running += size_t(std::max(storedCounts[i], 0));
    }
    if (running != storedPoints.size()) {
        return fail("stored Guides counts do not cover its points");
    }
    std::vector<int> knownTubes;
    for (PomadeSnapshotTube const &entry : check.tubes) {
        knownTubes.push_back(entry.tubeId);
    }
    std::vector<size_t> claimed, foreign;
    for (size_t i = 0; i < storedCounts.size(); ++i) {
        int const tubeId =
            i < storedTubeIds.size() ? storedTubeIds[i] : -1;
        bool const known =
            i < storedTubeIds.size() &&
            std::find(knownTubes.begin(), knownTubes.end(), tubeId) !=
                knownTubes.end();
        (known ? claimed : foreign).push_back(i);
    }
    auto guidesEqual = [&](PomadeSnapshotGuides const &candidate) {
        bool equal = claimed.size() == candidate.counts.size() &&
                     candidate.ids.size() == candidate.counts.size() &&
                     candidate.tubeIds.size() == candidate.counts.size();
        size_t candidateOffset = 0;
        for (size_t g = 0; equal && g < claimed.size(); ++g) {
            size_t const s = claimed[g];
            equal = s < storedIds.size() && s < storedTubeIds.size() &&
                    storedCounts[s] == candidate.counts[g] &&
                    storedIds[s] == candidate.ids[g] &&
                    storedTubeIds[s] == candidate.tubeIds[g];
            for (int c = 0; equal && c < candidate.counts[g]; ++c) {
                equal = std::memcmp(
                    &storedPoints[storedOffset[s] + size_t(c)],
                    &candidate.points[(candidateOffset + size_t(c)) * 3],
                    sizeof(float) * 3) == 0;
            }
            candidateOffset += size_t(candidate.counts[g]);
        }
        return equal;
    };
    bool equal = guidesEqual(regen);
    PomadeSnapshotGuides legacyRegen;
    PomadeSnapshotGuides const *verified = &regen;
    if (!equal && legacyRootSampler) {
        // A markerless layer is admitted only when its guide bytes exactly
        // match the old deterministic face-list/disc generator.  This keeps
        // edited legacy guide data rejected just as strictly as new data.
        legacyRegen = _LegacyGuidesFromSnapshot(check);
        equal = guidesEqual(legacyRegen);
        if (equal) {
            verified = &legacyRegen;
        }
    }
    result.guideCount = verified->counts.size();
    result.guidesBitEqual = equal;
    if (!equal) {
        return fail("regenerated guides differ from the stored Guides "
                    "(foreign or hand-edited guide data)");
    }
    // region-v1 guide bytes were generated with raw K4 frames. Once those
    // bytes have authenticated, migrate the live model's region roots to
    // the deterministic support-plane pins used by all subsequent edits and
    // v2 commits. This deliberately happens after the strict v1 comparison.
    if (sampler == _RootSampler::RegionV1 &&
        !model->PinRegionRootFrames()) {
        return fail(model->GetDiagnostic());
    }
    if (sampler != _RootSampler::RegionV3) {
        // Authentication above deliberately ran against the historical root
        // stream.  Once it has succeeded, migrate the live model before any
        // future commit: writing a v3 marker alongside legacy guide bytes
        // would otherwise make the next hydrate reject its own save.
        for (PomadeSnapshotTube const &entry : check.tubes) {
            PomadeModel::FillParams fill;
            if (!model->GetTubeFillParams(entry.tubeId, &fill)) {
                return fail("a hydrated tube has no Fill parameters");
            }
            fill.sampler = PomadeGuideSampler::RegionV3;
            if (!model->SetTubeFillParams(entry.tubeId, std::move(fill))) {
                return fail(model->GetDiagnostic());
            }
        }
        // The guide cache key includes the sampler. Refill the displayed
        // stream after the migration (unless it was explicitly cleared) so
        // no caller observes legacy geometry under a v3 live model.
        if (!model->RefillGuides(1.0f)) {
            return fail(model->GetDiagnostic());
        }
    }
    // Foreign guides (hand-authored, a DCC) are not fill output: they
    // become locked tubes one level under the tube whose region roots them
    // (plan/17 §2.5, §5.7), so the next commit carries them as geometry.
    for (size_t s : foreign) {
        int const nCv = storedCounts[s];
        if (nCv < 2) {
            return fail("a foreign guide has fewer than two CVs");
        }
        std::vector<float> cx, cy, cz;
        for (int c = 0; c < nCv; ++c) {
            GfVec3f const p = storedPoints[storedOffset[s] + size_t(c)];
            cx.push_back(p[0]);
            cy.push_back(p[1]);
            cz.push_back(p[2]);
        }
        float const root[3] = {cx[0], cy[0], cz[0]};
        int parentId = _OwningLeafTube(*model, root);
        PomadeTubeDesc parentDesc;
        if (!model->GetTubeDesc(parentId, &parentDesc)) {
            return fail("a foreign guide has no tube to import under");
        }
        std::vector<float> secT, secU, secV;
        for (auto const &sec : parentDesc.sections) {
            secT.push_back(sec.t);
            for (int v = 0; v < parentDesc.ringVerts; ++v) {
                secU.push_back(sec.u[size_t(v)] * sec.scale);
                secV.push_back(sec.v[size_t(v)] * sec.scale);
            }
        }
        int importedId = 0;
        if (!model->ImportLockedTube(parentId, cx.data(), cy.data(), cz.data(),
                                     nCv, secT.data(), secU.data(),
                                     secV.data(), int(secT.size()),
                                     parentDesc.ringVerts, &importedId)) {
            return fail(model->GetDiagnostic());
        }
        ++result.importedTubeCount;
        ++result.tubeCount;
    }
    result.ok = true;
    return result;
}

bool
PomadeSaveGroom(UsdStagePtr const &stage, SdfLayerHandle const &live,
              std::string const &filePath, std::string *err)
{
    auto fail = [&](std::string const &what) {
        if (err) {
            *err = "PomadeSaveGroom: " + what;
        }
        return false;
    };
    if (!stage || !live) {
        return fail("null stage or live layer");
    }
    if (filePath.size() < 6 ||
        (filePath.compare(filePath.size() - 5, 5, ".usdc") != 0 &&
         filePath.compare(filePath.size() - 5, 5, ".USDC") != 0)) {
        return fail("groom files are .usdc, never .usda (S42)");
    }
    SdfLayerRefPtr file = SdfLayer::FindOrOpen(filePath);
    if (!file) {
        file = SdfLayer::CreateNew(filePath);
    }
    if (!file) {
        return fail("cannot open or create " + filePath);
    }
    file->TransferContent(live);
    if (!file->Save()) {
        return fail("cannot save " + filePath);
    }
    // Re-parent: the live sublayer stays strongest so editing continues
    // there; the saved file sits directly beneath it.
    SdfLayerHandle const session = stage->GetSessionLayer();
    if (!session) {
        return fail("stage has no session layer");
    }
    std::vector<std::string> sublayers;
    sublayers.push_back(live->GetIdentifier());
    sublayers.push_back(file->GetIdentifier());
    for (std::string const &existing : session->GetSubLayerPaths()) {
        if (existing != live->GetIdentifier() &&
            existing != file->GetIdentifier()) {
            sublayers.push_back(existing);
        }
    }
    {
        SdfChangeBlock block;
        session->SetSubLayerPaths(sublayers);
    }
    return true;
}

bool
PomadeBakeSwapMapFile(SdfLayerHandle const &live, SdfPath const &regionMapPath,
                     std::string const &file, std::string *err)
{
    auto fail = [&](std::string const &what) {
        if (err) {
            *err = "PomadeBakeSwapMapFile: " + what;
        }
        return false;
    };
    if (!live) {
        return fail("null live layer");
    }
    if (regionMapPath.IsEmpty() || !regionMapPath.IsAbsolutePath()) {
        return fail("region map path must be absolute");
    }
    if (file.empty()) {
        return fail("no baked file to point at");
    }
    SdfPrimSpecHandle const map = live->GetPrimAtPath(regionMapPath);
    if (!map) {
        return fail("live layer carries no " + regionMapPath.GetString());
    }
    SdfChangeBlock block;
    // The spec usually exists already (the §3.1 build authors it): look it
    // up first, since New() on an existing spec logs a coding error.
    SdfAttributeSpecHandle attr;
    for (SdfAttributeSpecHandle const &cand : map->GetAttributes()) {
        if (cand->GetName() == "usdGen:map:file") {
            attr = cand;
            break;
        }
    }
    if (!attr) {
        attr = SdfAttributeSpec::New(map, "usdGen:map:file",
                                     SdfValueTypeNames->Asset);
    }
    if (!attr) {
        return fail("cannot author usdGen:map:file");
    }
    attr->SetDefaultValue(VtValue(SdfAssetPath(file)));
    return true;
}

PomadeClumpOffer
PomadePlanClumpFill(UsdStagePtr const &stage, PomadeCommitPaths const &paths,
                   int deepestLevel)
{
    PomadeClumpOffer offer;
    offer.exprPath = paths.LevelExprPath(std::max(deepestLevel, 1));
    if (!stage || paths.descriptionPath.IsEmpty()) {
        return offer;
    }
    UsdPrim const ops =
        stage->GetPrimAtPath(paths.descriptionPath.AppendChild(TfToken("Ops")));
    if (!ops) {
        return offer;
    }
    // The first Clump op whose usdGen:clump:map is unconnected gets the
    // offer; connected maps are never touched (the fill-in rule, §2.2).
    for (UsdPrim const &child : ops.GetChildren()) {
        if (child.GetTypeName() != TfToken("UsdGenClump")) {
            continue;
        }
        UsdAttribute const map =
            child.GetAttribute(TfToken("usdGen:clump:map"));
        if (_Connections(map).empty()) {
            offer.opPath = child.GetPath();
            return offer;
        }
    }
    return offer;
}

bool
PomadeSaveGroomAndMaps(UsdStagePtr const &stage, SdfLayerHandle const &live,
                      std::string const &filePath, std::string const &mapFile,
                      PomadeCommitPaths const &paths, std::string *err)
{
    if (!PomadeSaveGroom(stage, live, filePath, err)) {
        return false;
    }
    auto fail = [&](std::string const &what) {
        if (err) {
            *err = "PomadeSaveGroomAndMaps: " + what;
        }
        return false;
    };
    namespace fs = std::filesystem;
    std::error_code ec;
    std::string outputMapFile;
    if (live) {
        SdfAttributeSpecHandle const attr = live->GetAttributeAtPath(
            paths.OutputRegionMapPath().AppendProperty(TfToken("usdGen:map:file")));
        if (attr && attr->GetDefaultValue().IsHolding<SdfAssetPath>()) {
            outputMapFile = attr->GetDefaultValue().UncheckedGet<SdfAssetPath>()
                                .GetAssetPath();
        }
    }
    if (mapFile.empty() && outputMapFile.empty()) {
        return true;
    }
    SdfLayerRefPtr file = SdfLayer::FindOrOpen(filePath);
    if (!file) {
        return fail("cannot reopen " + filePath);
    }
    auto copyAndRepoint = [&](std::string const &source, std::string const &name,
                              SdfPath const &mapPath) {
        if (source.empty()) {
            return true;
        }
        fs::path const dest = fs::path(filePath).parent_path() / name;
        ec.clear();
        bool const sameFile = fs::exists(dest, ec) && !ec &&
            fs::equivalent(fs::path(source), dest, ec) && !ec;
        ec.clear();
        if (!sameFile) {
            fs::copy_file(source, dest, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                return fail("cannot copy " + source + " to " + dest.string());
            }
        }
        SdfPrimSpecHandle map = SdfCreatePrimInLayer(file, mapPath);
        if (!map) {
            return fail("cannot author map in the saved file");
        }
        SdfAttributeSpecHandle attr;
        for (SdfAttributeSpecHandle const &cand : map->GetAttributes()) {
            if (cand->GetName() == "usdGen:map:file") {
                attr = cand;
                break;
            }
        }
        if (!attr) {
            attr = SdfAttributeSpec::New(map, "usdGen:map:file",
                                         SdfValueTypeNames->Asset);
        }
        if (!attr) {
            return fail("cannot author usdGen:map:file in the saved file");
        }
        attr->SetDefaultValue(VtValue(SdfAssetPath("./" + dest.filename().string())));
        return true;
    };
    {
        SdfChangeBlock block;
        if (!copyAndRepoint(mapFile, "regionMap.ptx", paths.RegionMapPath()) ||
            !copyAndRepoint(outputMapFile,
                            fs::path(filePath).stem().string() + ".output-" +
                                fs::path(outputMapFile).filename().string(),
                            paths.OutputRegionMapPath())) {
            return false;
        }
    }
    if (!file->Save()) {
        return fail("cannot save " + filePath);
    }
    return true;
}

// The lane's shared state (see the header).
struct PomadeCommitter::_Lane {
    struct Request : usdGen::UsdGenRefCounted {
        uint64_t version = 0;
        PomadeFillPlan plan;
    };
    struct Diagnostic : usdGen::UsdGenRefCounted {
        std::string text;
    };
    struct Paths : usdGen::UsdGenRefCounted {
        PomadeCommitPaths paths;
    };
    // A built layer on the shelf. Its fate is decided once: the UI claims it
    // for a swap, or a newer build (or Detach) drops it, never both.
    enum : uint32_t { kFresh = 0, kClaimed = 1, kDropped = 2 };
    struct Built : usdGen::UsdGenRefCounted {
        SdfLayerRefPtr layer;  // strong: anonymous layers die with it
        uint64_t version = 0;
        mutable std::atomic<uint32_t> fate{kFresh};
        bool Decide(uint32_t to) const
        {
            uint32_t fresh = kFresh;
            return fate.compare_exchange_strong(fresh, to,
                                                std::memory_order_acq_rel);
        }
    };
    // Newest-wins handoffs; the worker parks on `wake`, reading it BEFORE
    // it checks for work, so a Notify in between makes the park return.
    usdGen::UsdGenNewestWins<Request> pending;
    usdGen::UsdGenAtomicRef<Paths> paths;
    usdGen::UsdGenAtomicRef<Built> ready;
    usdGen::UsdGenRef<Built> partial;  // UI thread: the layer in transfer
    usdGen::UsdGenNewestWins<Diagnostic> diagnostic;
    std::atomic<uint32_t> wake{0};

    void Notify()
    {
        wake.fetch_add(1u, std::memory_order_seq_cst);
        usdGen::UsdGenUnparkAll(wake);
    }

    uint64_t ReadyVersion() const
    {
        usdGen::UsdGenEpochGuard guard;
        Built const *b = ready.Peek();
        return b ? b->version : 0;
    }
    void PublishPaths(PomadeCommitPaths const &p)
    {
        usdGen::UsdGenRef<Paths> next = usdGen::UsdGenMakeRef<Paths>();
        next->paths = p;
        paths.Store(std::move(next));
    }
    void Diagnose(std::string text)
    {
        auto d = usdGen::UsdGenMakeRef<Diagnostic>();
        d->text = std::move(text);
        diagnostic.Publish(std::move(d));
    }
};

PomadeCommitter::PomadeCommitter(PomadeModel *model, PomadeCommitPaths paths)
    : _model(model)
    , _paths(std::move(paths))
    , _lane(new _Lane())
{
    _lane->PublishPaths(_paths);
    // Every member is live before the lane starts (see the header).
    _worker = std::thread(&PomadeCommitter::_WorkerLoop, this);
}

PomadeCommitter::~PomadeCommitter()
{
    _stop.store(true, std::memory_order_seq_cst);
    _lane->Notify();
    if (_worker.joinable()) {
        _worker.join();
    }
    _lane.reset();
    usdGen::UsdGenEpochDrain();
}

void
PomadeCommitter::Enqueue(UsdStagePtr const &stage)
{
    PomadeFillPlan plan;
    if (stage) {
        plan = PomadePlanGuideInterpolateFill(stage, _paths);
    } else {
        plan.opPath = _paths.InterpOpPath();
    }
    EnqueuePlan(plan, stage);
}

SdfPath
PomadeCommitter::_BlockedOutputPath(UsdStagePtr const &stage) const
{
    // Output is a reserved generated path only when its owner marker is
    // present on the sparse cage, categorical owner map, and description.
    // Never let enabling Pomade output replace an artist-owned member of
    // that atomic triple.
    if (!stage || !_model->GetOutputSettings().enabled) {
        return SdfPath();
    }
    SdfPath const guarded[] = {_paths.OutputCurvesPath(),
                               _paths.OutputRegionMapPath(),
                               _paths.OutputPath()};
    for (SdfPath const &path : guarded) {
        UsdPrim const existing = stage->GetPrimAtPath(path);
        if (!existing) {
            continue;
        }
        bool owned = false;
        UsdAttribute const marker = existing.GetAttribute(
            TfToken("usdGen:pomade:outputOwned"));
        if (!marker || !marker.Get(&owned) || !owned) {
            return path;
        }
    }
    return SdfPath();
}

size_t
PomadeCommitter::CancelDescriptionCooks() const
{
    usdGenImaging::UsdGenSessionStore &sessions =
        usdGenImaging::UsdGenSessionStore::GetInstance();
    size_t cancelled = 0;
    if (!_paths.descriptionPath.IsEmpty()) {
        cancelled += sessions.CancelCooks(_paths.descriptionPath);
    }
    // Output has its own description root and may cook independently of the
    // legacy description path. Keep this cancellation separate so a groom
    // with no legacy description still drops stale generated output cooks.
    SdfPath const output = _paths.OutputPath();
    if (!output.IsEmpty() && output != _paths.descriptionPath) {
        cancelled += sessions.CancelCooks(output);
    }
    return cancelled;
}

void
PomadeCommitter::SetScalpPath(SdfPath const &scalpPath, SdfPath const &meshPath)
{
    _paths.scalpPath = scalpPath;
    // A retarget always restates the Mesh: a stale parent from the previous
    // subset must not receive the next scalp's rest binding.
    _paths.scalpMeshPath =
        (scalpPath.IsEmpty() || meshPath == scalpPath) ? SdfPath() : meshPath;
    _lane->PublishPaths(_paths);
}

SdfPath
PomadeCommitter::GetScalpPath() const
{
    return _paths.scalpPath;
}

SdfPath
PomadeCommitter::GetScalpMeshPath() const
{
    return _paths.ScalpMeshPath();
}

void
PomadeCommitter::EnqueuePlan(PomadeFillPlan const &plan,
                            UsdStagePtr const &stage)
{
    // The guard lives here, not in Enqueue(stage), so every enqueue that
    // can see a stage is checked the same way; a refusal is a failure the
    // artist must hear about (TakeDiagnostic + FailedVersion), not a
    // silent drop.
    SdfPath const blocked = _BlockedOutputPath(stage);
    if (!blocked.IsEmpty()) {
        _lane->Diagnose("refusing to overwrite artist-owned output at " +
                        blocked.GetString());
        _failedVersion.store(_model->GetVersion());
        return;
    }
    // An explicit enqueue is a request to try again, even of a version
    // whose build failed.
    _failedVersion.store(0);
    // The worker builds from the newest published state, never the live
    // model, so publish before the request becomes visible.
    _model->PublishState();
    auto request = usdGen::UsdGenMakeRef<_Lane::Request>();
    request->version = _model->GetVersion();
    request->plan = plan;
    if (request->plan.opPath.IsEmpty()) {
        request->plan.opPath = _paths.InterpOpPath();
    }
    _pendingVersionAtomic.store(request->version);
    _lane->pending.Publish(std::move(request));
    _lane->Notify();
}

void
PomadeCommitter::_ClearPartial()
{
    _partialSlots.clear();
    _partialNext = 0;
    _partialVersion = 0;
    _lane->partial.Reset();
}

void
PomadeCommitter::Detach()
{
    // Published before the shelf is cleared: a build finishing now either
    // sees the flag or has its layer cleared below (see _WorkerLoop).
    _detached.store(true);
    if (usdGen::UsdGenRef<_Lane::Built> const old =
            _lane->ready.Exchange(usdGen::UsdGenRef<_Lane::Built>())) {
        // Built for the dead stage: never swappable, same accounting as
        // superseded-on-the-shelf.
        if (old->version > _committedVersion.load() &&
            old->Decide(_Lane::kDropped)) {
            _droppedCount.fetch_add(1);
        }
    }
    _ClearPartial();
    // The pending enqueue (if any) carries the dead stage's fill-in plan;
    // drop it so only a post-Reattach enqueue can build.
    _lane->pending.Take();
    _pendingVersionAtomic.store(0);
}

void
PomadeCommitter::Reattach()
{
    _detached.store(false);
    // Same guarantee as Detach: anything enqueued while detached targeted
    // the dead stage, so the tool must enqueue fresh for the new one.
    _lane->pending.Take();
    _pendingVersionAtomic.store(0);
    // A new live-layer lineage: the fresh live layer carries nothing yet,
    // whatever the old lineage committed.
    _committedVersion.store(0);
    _lane->Notify();
}

bool
PomadeCommitter::IsDetached() const
{
    return _detached.load();
}

void
PomadeCommitter::_WorkerLoop()
{
    for (;;) {
        uint32_t const seen = _lane->wake.load(std::memory_order_seq_cst);
        if (_stop.load(std::memory_order_seq_cst)) {
            return;
        }
        usdGen::UsdGenRef<_Lane::Request> request;
        if (!_paused.load(std::memory_order_seq_cst) && !_detached.load()) {
            request = _lane->pending.Take();
        }
        if (!request) {
            usdGen::UsdGenParkWhileEqual(_lane->wake, seen);
            continue;
        }
        if (request->version <= _lane->ReadyVersion()) {
            continue;  // the shelf already carries it (or newer)
        }
        // Coalescing: the mailbox holds only the latest enqueue.
        uint64_t const version = request->version;
        _building.store(true);
        usdGen::UsdGenRef<_Lane::Paths const> const pathsRef =
            _lane->paths.Load();
        PomadeCommitPaths const &paths = pathsRef->paths;
        // Build from the newest published model state, with no stage.
        // Everything from here to the handover is wrapped: a throw in the
        // snapshot or the layer build must leave the previous live layer,
        // the model and this thread alone (plan/17 §3.4).
        SdfLayerRefPtr built;
        std::string err;
        bool builtOk = false;
        uint64_t builtVersion = version;
        try {
            if (int const pending = _throwBuilds.load(); pending > 0) {
                _throwBuilds.store(pending - 1);
                throw std::runtime_error(
                    "PomadeCommitter: injected build failure (test hook)");
            }
            PomadeModelStateRef const state = PomadeAcquireModelState(*_model);
            if (!state) {
                throw std::runtime_error(
                    "PomadeCommitter: no published model state");
            }
            PomadeSnapshot snapshot = PomadeSnapshotFromState(*state);
            snapshot.scalpPath = paths.scalpPath;
            snapshot.scalpMeshPath = paths.scalpMeshPath;
            snapshot.version = snapshot.tubes.empty()
                                   ? version
                                   : snapshot.tubes[0].tube.version;
            snapshot.createInterpOp = request->plan.createInterpOp;
            snapshot.setInterpGuides = request->plan.setInterpGuides;
            snapshot.setInterpRegion = request->plan.setInterpRegion;
            snapshot.interpOpPath = request->plan.opPath;
            builtVersion = snapshot.version;
            builtOk = PomadeBuildCommitLayer(snapshot, paths, &built, &err,
                                            &_guideCache, &_outputGuideCache);
        } catch (std::exception const &e) {
            _workerThrowCount.fetch_add(1);
            builtOk = false;
            built.Reset();
            err = std::string("PomadeCommitter: worker threw: ") + e.what();
        } catch (...) {
            _workerThrowCount.fetch_add(1);
            builtOk = false;
            built.Reset();
            err = "PomadeCommitter: worker threw an unknown exception";
        }
        _building.store(false);
        if (!builtOk) {
            // A bad snapshot (or a throw) keeps the previous live layer and
            // the previous built layer; the model is unaffected (plan/17
            // §3.4). The version is parked: the request is gone, so only a
            // new enqueue builds again.
            _lane->Diagnose(err);
            _failedVersion.store(version);
            continue;
        }
        _buildCount.fetch_add(1);
        if (_detached.load()) {
            // Detached mid-build: the fill-in plan targets the dead stage,
            // so this layer can never swap. Drop it like a superseded build;
            // the tool re-enqueues after Reattach.
            _droppedCount.fetch_add(1);
            continue;
        }
        if (_pendingVersionAtomic.load() > builtVersion) {
            // Superseded mid-build: abandon this layer (cancellation); the
            // latest request is already in the mailbox.
            _droppedCount.fetch_add(1);
            continue;
        }
        usdGen::UsdGenRef<_Lane::Built> next =
            usdGen::UsdGenMakeRef<_Lane::Built>();
        next->layer = built;
        next->version = builtVersion;
        built.Reset();
        if (usdGen::UsdGenRef<_Lane::Built> const old =
                _lane->ready.Exchange(next)) {
            // A built layer that never swapped: superseded on the shelf.
            if (old->version > _committedVersion.load() &&
                old->Decide(_Lane::kDropped)) {
                _droppedCount.fetch_add(1);
            }
        }
        if (_detached.load() &&
            _lane->ready.CompareExchange(next, usdGen::UsdGenRef<_Lane::Built>()) &&
            next->Decide(_Lane::kDropped)) {
            // Detach ran between the check above and the store: take the
            // layer back off the shelf (Detach clears what it can see).
            _droppedCount.fetch_add(1);
        }
        usdGen::UsdGenEpochDrain();
    }
}

PomadeCommitter::SwapResult
PomadeCommitter::SwapIfIdle(SdfLayerHandle const &live, bool gestureActive)
{
    if (_detached.load()) {
        return Detached;
    }
    if (!live) {
        return NothingPending;
    }
    if (gestureActive) {
        return SkippedGesture;
    }
    for (;;) {
        usdGen::UsdGenRef<_Lane::Built> const ready = _lane->ready.Load();
        uint64_t const readyVersion = ready ? ready->version : 0;
        if (_partialVersion != 0) {
            if (readyVersion > _partialVersion || ready != _lane->partial) {
                // A newer layer landed mid-transfer: abandon the stale
                // partial and start over below.
                _ClearPartial();
            } else {
                return _SwapPartialSlot(live, ready->layer, _partialVersion);
            }
        }
        if (!ready || readyVersion <= _committedVersion.load()) {
            return NothingPending;
        }
        if (_building.load() && _pendingVersionAtomic.load() > readyVersion) {
            // A newer version is already being built; the worker will hand
            // a newer layer soon, so this swap waits.
            return SkippedStale;
        }
        if (_partialMode.load()) {
            _BeginPartial(ready->layer);
            _partialVersion = readyVersion;
            _lane->partial = ready;
            return _SwapPartialSlot(live, ready->layer, readyVersion);
        }
        if (!ready->Decide(_Lane::kClaimed)) {
            continue;  // dropped by a newer build just now: take that one
        }
        return _SwapFull(live, ready->layer, readyVersion);
    }
}

PomadeCommitter::SwapResult
PomadeCommitter::_SwapFull(SdfLayerHandle const &live,
                          SdfLayerHandle const &built, uint64_t version)
{
    auto t0 = std::chrono::steady_clock::now();
    {
        SdfChangeBlock block;
        live->TransferContent(built);
    }
    auto t1 = std::chrono::steady_clock::now();
    double const ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    _lastSwapMs.store(ms);
    _committedVersion.store(version);
    _ClearPartial();
    if (ms > _swapBudgetMs.load()) {
        // Over budget: subsequent swaps go one subtree per idle slot.
        _partialMode.store(true);
    }
    return Swapped;
}

namespace {

// Copy one prim's self (specifier, type, and — unless `shellOnly` — its own
// properties) without children. The Guides self-slot is shell-only: its
// properties arrive in later per-property slots (TN-4), but stale ones are
// pruned here so a removed property never lingers on live.
void
_CopyPrimSelf(SdfLayerHandle const &built, SdfLayerHandle const &live,
              SdfPath const &path, bool shellOnly = false)
{
    SdfPrimSpecHandle const src = built->GetPrimAtPath(path);
    SdfPrimSpecHandle const dst = SdfCreatePrimInLayer(live, path);
    if (!src || !dst) {
        return;
    }
    dst->SetSpecifier(src->GetSpecifier());
    if (!src->GetTypeName().IsEmpty()) {
        dst->SetTypeName(src->GetTypeName().GetString());
    }
    if (!shellOnly) {
        for (SdfAttributeSpecHandle const &attr : src->GetAttributes()) {
            SdfCopySpec(built, attr->GetPath(), live, attr->GetPath());
        }
        for (SdfRelationshipSpecHandle const &rel : src->GetRelationships()) {
            SdfCopySpec(built, rel->GetPath(), live, rel->GetPath());
        }
    }
    std::vector<SdfPropertySpecHandle> staleProps;
    for (SdfAttributeSpecHandle const &attr : dst->GetAttributes()) {
        if (!built->GetAttributeAtPath(attr->GetPath())) {
            staleProps.push_back(attr);
        }
    }
    for (SdfRelationshipSpecHandle const &rel : dst->GetRelationships()) {
        if (!built->GetRelationshipAtPath(rel->GetPath())) {
            staleProps.push_back(rel);
        }
    }
    for (SdfPropertySpecHandle const &prop : staleProps) {
        dst->RemoveProperty(prop);
    }
}

// Drop live children of `parentPath` the built layer no longer carries.
void
_DropStaleChildren(SdfLayerHandle const &built, SdfLayerHandle const &live,
                   SdfPath const &parentPath)
{
    SdfPrimSpecHandle const dst = live->GetPrimAtPath(parentPath);
    if (!dst) {
        return;
    }
    std::vector<SdfPrimSpecHandle> stale;
    for (SdfPrimSpecHandle const &child : dst->GetNameChildren()) {
        if (!built->GetPrimAtPath(child->GetPath())) {
            stale.push_back(child);
        }
    }
    for (SdfPrimSpecHandle const &child : stale) {
        dst->RemoveNameChild(child);
    }
}

// Mirror the built layer's scalp over onto live. It must be a whole-spec
// SdfCopySpec, not _CopyPrimSelf: the over carries UsdGenRestAPI in its
// apiSchemas metadata (the Output CurveSource's rest binding), which a
// self copy of specifier/type/properties silently drops. When the built
// layer no longer overs the scalp, the live over goes too, so a stale
// RestAPI or pomadeRegion preview never outlives the edit that removed it.
void
_SyncScalpOver(SdfLayerHandle const &built, SdfLayerHandle const &live,
               SdfPath const &scalpPath)
{
    if (scalpPath.IsEmpty() || !scalpPath.IsPrimPath()) {
        return;
    }
    if (built->GetPrimAtPath(scalpPath)) {
        SdfPath const parentPath = scalpPath.GetParentPath();
        if (parentPath != SdfPath::AbsoluteRootPath() &&
            !live->GetPrimAtPath(parentPath)) {
            // SdfCopySpec needs the destination parent; the built layer
            // only carries overs above the scalp, so overs suffice here.
            SdfCreatePrimInLayer(live, parentPath);
        }
        SdfPrimSpecHandle const src = built->GetPrimAtPath(scalpPath);
        if (src->GetNameChildren().empty()) {
            SdfCopySpec(built, scalpPath, live, scalpPath);
        } else {
            // The groom (or its description) lives UNDER the scalp prim:
            // a whole-spec copy would re-copy that entire subtree in this
            // one slot (defeating the per-slot budget and the TN-4 Guides
            // split), and those children already have slots of their own.
            // Mirror the scalp's own fields -- apiSchemas included, which
            // _CopyPrimSelf would drop -- and its properties only.
            SdfPrimSpecHandle const dst =
                SdfCreatePrimInLayer(live, scalpPath);
            if (!dst) {
                return;
            }
            SdfSchemaBase const &schema = src->GetSchema();
            for (TfToken const &field : dst->ListFields()) {
                if (!schema.HoldsChildren(field) && !src->HasField(field)) {
                    dst->ClearField(field);
                }
            }
            for (TfToken const &field : src->ListFields()) {
                if (!schema.HoldsChildren(field)) {
                    dst->SetField(field, src->GetField(field));
                }
            }
            for (SdfAttributeSpecHandle const &attr : src->GetAttributes()) {
                SdfCopySpec(built, attr->GetPath(), live, attr->GetPath());
            }
            for (SdfRelationshipSpecHandle const &rel :
                 src->GetRelationships()) {
                SdfCopySpec(built, rel->GetPath(), live, rel->GetPath());
            }
            std::vector<SdfPropertySpecHandle> staleProps;
            for (SdfPropertySpecHandle const &prop : dst->GetProperties()) {
                if (!built->HasSpec(prop->GetPath())) {
                    staleProps.push_back(prop);
                }
            }
            for (SdfPropertySpecHandle const &prop : staleProps) {
                dst->RemoveProperty(prop);
            }
        }
    } else if (SdfPrimSpecHandle const stale = live->GetPrimAtPath(scalpPath)) {
        if (SdfPrimSpecHandle const parent =
                live->GetPrimAtPath(scalpPath.GetParentPath())) {
            parent->RemoveNameChild(stale);
        }
    }
}

} // namespace

void
PomadeCommitter::_BeginPartial(SdfLayerHandle const &built)
{
    _partialSlots.clear();
    _partialNext = 0;
    // Slot 0 is the groom prim itself (type + own properties, no children).
    // Tubes go one per slot (SdfCopySpec per tube): at reference scale the
    // Tubes scope alone is megabytes, so a per-groom-child split would still
    // blow the slot budget. Guides goes as the prim shell plus one slot per
    // property for the same reason. Every other groom child and the
    // description opinions (if any) take one slot each, in name order.
    _partialSlots.push_back(_PartialSlot{_paths.groomPath, true});
    SdfPath const tubesPath = _paths.TubesPath();
    if (SdfPrimSpecHandle const groom = built->GetPrimAtPath(_paths.groomPath)) {
        std::vector<SdfPath> children;
        SdfPath const outputCurvesPath = _paths.OutputCurvesPath();
        SdfPath const outputRegionMapPath = _paths.OutputRegionMapPath();
        SdfPath const outputPath = _paths.OutputPath();
        bool haveOutputTriple = false;
        for (SdfPrimSpecHandle const &child : groom->GetNameChildren()) {
            if (child->GetPath() == outputPath ||
                child->GetPath() == outputRegionMapPath) {
                continue;  // paired with OutputCurves below
            }
            if (child->GetPath() == outputCurvesPath) {
                haveOutputTriple = true;
                continue;
            }
            if (child->GetPath() != tubesPath) {
                children.push_back(child->GetPath());
            }
        }
        std::sort(children.begin(), children.end());
        for (SdfPath const &child : children) {
            if (child == _paths.GuidesPath()) {
                // Guides is the one fat non-tube child (192 K guide CVs +
                // root frames at reference scale): one SdfCopySpec for the
                // whole subtree blows the slot budget, so it goes as the
                // prim self plus one slot per property (TN-4).
                _partialSlots.push_back(_PartialSlot{child, true});
                SdfPrimSpecHandle const guides =
                    built->GetPrimAtPath(child);
                if (guides) {
                    std::vector<SdfPath> props;
                    for (SdfAttributeSpecHandle const &attr :
                         guides->GetAttributes()) {
                        props.push_back(attr->GetPath());
                    }
                    for (SdfRelationshipSpecHandle const &rel :
                         guides->GetRelationships()) {
                        props.push_back(rel->GetPath());
                    }
                    std::sort(props.begin(), props.end());
                    for (SdfPath const &prop : props) {
                        _partialSlots.push_back(_PartialSlot{prop, false});
                    }
                }
            } else {
                _partialSlots.push_back(_PartialSlot{child, false});
            }
        }
        if (haveOutputTriple && built->GetPrimAtPath(outputRegionMapPath) &&
            built->GetPrimAtPath(outputPath)) {
            // A CurveSource must never observe new counts against old points
            // during a budgeted transfer. Keep both owned Output prims in the
            // same ChangeBlock even when their source data is large.
            _partialSlots.push_back(
                _PartialSlot{outputCurvesPath, false, true});
        }
    }
    if (built->GetPrimAtPath(tubesPath)) {
        _partialSlots.push_back(_PartialSlot{tubesPath, true});
        std::vector<SdfPath> tubePaths;
        for (SdfPrimSpecHandle const &child :
             built->GetPrimAtPath(tubesPath)->GetNameChildren()) {
            tubePaths.push_back(child->GetPath());
        }
        std::sort(tubePaths.begin(), tubePaths.end());
        for (SdfPath const &tube : tubePaths) {
            _partialSlots.push_back(_PartialSlot{tube, false});
        }
    }
    if (!_paths.descriptionPath.IsEmpty() &&
        built->GetPrimAtPath(_paths.descriptionPath)) {
        _partialSlots.push_back(_PartialSlot{_paths.descriptionPath, false});
    }
    // The scalp over (UsdGenRestAPI for Output, the pomadeRegion preview
    // primvar) gets its own slot even when the built layer dropped it: the
    // slot then removes the live over (_SyncScalpOver). It sits on the
    // Mesh, which is a face subset's parent (plan/02 §2.20).
    if (!_paths.ScalpMeshPath().IsEmpty()) {
        _partialSlots.push_back(_PartialSlot{_paths.ScalpMeshPath(), false});
    }
}

PomadeCommitter::SwapResult
PomadeCommitter::_SwapPartialSlot(SdfLayerHandle const &live,
                                 SdfLayerHandle const &built, uint64_t version)
{
    // Keep the optional TN-4 diagnosis out of the timed mutation itself.
    // A reference groom has thousands of one-tube slots; when a platform
    // budget regression appears, the path distinguishes a large Guide
    // property from an unexpectedly costly ordinary tube copy without adding
    // a public test/debug API.
    if (_partialNext + 1 >= _partialSlots.size() && _lane->partial &&
        !_lane->partial->Decide(_Lane::kClaimed)) {
        // A newer build dropped this layer before its last slot: abandon
        // the transfer; the next idle slot starts the newer one.
        _ClearPartial();
        return PartialProgress;
    }
    _PartialSlot const &slot = _partialSlots[_partialNext];
    SdfPath const slotPath = slot.path;
    bool const slotSelfOnly = slot.selfOnly;
    bool const slotOutputPair = slot.outputPair;
    size_t const slotIndex = _partialNext;
    size_t const slotCount = _partialSlots.size();
    auto t0 = std::chrono::steady_clock::now();
    {
        SdfChangeBlock block;
        if (_partialNext == 0) {
            // Groom self, plus stale-subtree cleanup so a partial swap
            // never leaves removed tubes/children behind: the live layer is
            // tool-owned, so anything the built layer no longer carries is
            // dropped. Ancestors for the later slots are ensured here too.
            _CopyPrimSelf(built, live, slot.path);
            _DropStaleChildren(built, live, _paths.groomPath);
            _DropStaleChildren(built, live, _paths.TubesPath());
            if (!_paths.descriptionPath.IsEmpty()) {
                if (built->GetPrimAtPath(_paths.descriptionPath)) {
                    SdfCreatePrimInLayer(live, _paths.descriptionPath);
                } else if (live->GetPrimAtPath(_paths.descriptionPath)) {
                    SdfPrimSpecHandle const parent = live->GetPrimAtPath(
                        _paths.descriptionPath.GetParentPath());
                    if (parent) {
                        parent->RemoveNameChild(live->GetPrimAtPath(
                            _paths.descriptionPath));
                    }
                }
            }
        } else if (slot.outputPair) {
            // The Output CurveSource captures against the scalp's rest
            // binding (UsdGenRestAPI on the scalp over), so the over must
            // land in the same ChangeBlock as the Output triple, or the
            // first capture after a partial transfer sees no rest and fails.
            if (built->GetPrimAtPath(_paths.ScalpMeshPath())) {
                _SyncScalpOver(built, live, _paths.ScalpMeshPath());
            }
            SdfCopySpec(built, _paths.OutputCurvesPath(), live,
                        _paths.OutputCurvesPath());
            SdfCopySpec(built, _paths.OutputRegionMapPath(), live,
                        _paths.OutputRegionMapPath());
            SdfCopySpec(built, _paths.OutputPath(), live,
                        _paths.OutputPath());
        } else if (slot.selfOnly) {
            // The Guides self-slot carries the shell only; its properties
            // follow in per-property slots (TN-4).
            _CopyPrimSelf(built, live, slot.path,
                          /*shellOnly*/ slot.path == _paths.GuidesPath());
        } else if (slot.path == _paths.ScalpMeshPath()) {
            _SyncScalpOver(built, live, slot.path);
        } else {
            SdfCopySpec(built, slot.path, live, slot.path);
        }
        ++_partialNext;
    }
    auto t1 = std::chrono::steady_clock::now();
    double const elapsedMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    _lastSwapMs.store(elapsedMs);
    if (std::getenv("USDGEN_POMADE_PARTIAL_SLOT_TIMING") &&
        elapsedMs > 4.0) {
        std::fprintf(stderr,
                     "pomade: partial slot %zu/%zu %s%s%s %.3f ms\n",
                     slotIndex + 1, slotCount, slotPath.GetText(),
                     slotSelfOnly ? " [self]" : "",
                     slotOutputPair ? " [output-pair]" : "", elapsedMs);
    }
    if (_partialNext >= _partialSlots.size()) {
        _committedVersion.store(version);
        _ClearPartial();
        return Swapped;
    }
    return PartialProgress;
}

void
PomadeCommitter::PauseWorker(bool pause)
{
    _paused.store(pause, std::memory_order_seq_cst);
    _lane->Notify();
}

void
PomadeCommitter::SetReadyForTest(SdfLayerRefPtr const &layer, uint64_t version)
{
    usdGen::UsdGenRef<_Lane::Built> next =
        usdGen::UsdGenMakeRef<_Lane::Built>();
    next->layer = layer;
    next->version = version;
    _lane->ready.Store(std::move(next));
}

void
PomadeCommitter::ForcePartialModeForTest(bool on)
{
    _partialMode.store(on);
}

void
PomadeCommitter::ThrowOnNextBuildsForTest(int count)
{
    _throwBuilds.store(count < 0 ? 0 : count);
}

std::string
PomadeCommitter::TakeDiagnostic()
{
    usdGen::UsdGenRef<_Lane::Diagnostic> d = _lane->diagnostic.Take();
    return d ? std::move(d->text) : std::string();
}

uint64_t
PomadeCommitter::FailedVersion() const
{
    return _failedVersion.load();
}

} // namespace usdGenPomade
