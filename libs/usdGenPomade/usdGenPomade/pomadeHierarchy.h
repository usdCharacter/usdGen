// usdGenPomade — P4 hierarchy core: K14 subdivide/merge, K7 parent average
// (plan/17 §2.3, §2.4, §4.1, §6 P4 exit). CPU twins.
//
// Qt-free, no USD, no CUDA types. Free functions over PomadeTubeDesc so the
// T0 test proves the P4 exit without a multi-tube model (the model stays
// single-tube; the multi-tube store is later P4 work). Deterministic:
// same inputs give bit-identical outputs (pure functions, index-ordered
// loops, hash streams keyed by (tubeId, seed)).
//
// Reconstruction notes (ours, per §4.1 — the publications say nothing here):
//   * K14 k-means runs over the placed root ring in the root frame with
//     k-means++ init from the PomadeHash01 stream and Lloyd's to convergence
//     (cap 64, deterministic tie-breaks). Empty clusters re-seed to the
//     farthest point, deterministically.
//   * Child centers are the parent center offset by the sub-region centroid
//     scaled by the local-to-root section radius ratio (§2.3 step 2).
//   * Child sections are the parent sections clipped to the child's cell
//     (Voronoi half-planes, or the drawn line in edge mode) at every t.
//     The root cell is carried to each section by that section's
//     centroid/radius similarity, so a shifted or rescaled ring keeps
//     the same relative partition; an unchanged chart reuses the root
//     cell bit-exactly. Parent arc vertices are kept EXACTLY; only long
//     chord runs are subdivided (points stay on the parent polygon).
//     Child rings store
//     absolute placed coords (scale = 1, twist = 0), so the union of the
//     children's vertex sets equals the parent's up to float clip error and
//     the P4 Hausdorff bar (<= 1e-3 of the tube radius) holds literally on
//     vertex sets. Child ring counts vary per child (parent arcs + cut
//     samples); uniformize-on-demand is an authoring op, not done here.
//   * Edge mode supports count == 2 only (one drawn edge, two halves).
//   * Hint-less K7 averages arc-length-resampled child centers in double
//     precision. A merge with a persistent parent hint keeps that parent's
//     center cage/material frame/layout, extends only its terminal cage CV
//     when descendants outgrow it, and writes a conservative support-plane
//     envelope around every child section sample at every supported K5 density.
//     Hint-less on-the-fly parents retain the historical union refit.
//   * Merge takes the persistent parent as an optional hint (the model
//     always has it: parents persist per §2.3). When every child matches
//     its re-derived shape bit-wise (zero deltas), merge returns the hint
//     copy, which is how subdivide -> merge round-trips bit-exactly.
//     Otherwise a hinted merge is the K7 envelope; a hint-less merge is the
//     historical K7 average.
#ifndef USDGEN_POMADE_HIERARCHY_H
#define USDGEN_POMADE_HIERARCHY_H

#include "usdGenPomade/api.h"
#include "usdGenPomade/pomadeTube.h"

#include <algorithm>
#include <string>
#include <vector>

namespace usdGenPomade {

enum PomadeSplitMode {
    PomadeSplit_KMeans = 0,
    PomadeSplit_Edge = 1,
};

// K14 parameters (mirrors PomadeModel::SubdivideParams without the model
// dependency; the model adapter maps splitMode strings to this enum).
struct PomadeSubdivideDesc {
    int count = 4;      // in [2, 8]
    int seed = 0;
    int splitMode = PomadeSplit_KMeans;
    // Edge mode: the splitting line a*u + b*v + c = 0 in root-plane coords.
    // Child 0 takes the negative side, child 1 the positive side.
    float edgeA = 1.0f;
    float edgeB = 0.0f;
    float edgeC = 0.0f;
};

USDGENPOMADE_API bool PomadeValidateSubdivide(PomadeSubdivideDesc const &params,
                            std::string *err);

// K14 subdivide: parent -> `params.count` children (§2.3). `parentFrames`
// are the K4 frames at the parent center CVs (size nCv). Children inherit
// regionId/fill layout, get level + 1, tubeId = parent*16 + 1 + childIndex,
// and parentTubeId/childIndex links. Zero deltas by construction.
USDGENPOMADE_API bool PomadeSubdivideTubeCpu(PomadeTubeDesc const &parent,
                           std::vector<PomadeFrame> const &parentFrames,
                           PomadeSubdivideDesc const &params,
                           std::vector<PomadeTubeDesc> *children,
                           std::string *err);

// K7 parent average: children -> aggregate parent. Centers are
// arc-length-resampled to the first child's layout and averaged in double
// precision; sections re-fit to the union of the child rings at each of
// the first child's section t values. Used for child-edit refresh (unless
// lockParents), on-the-fly parents, and merge.
USDGENPOMADE_API bool PomadeParentAverageCpu(std::vector<PomadeTubeDesc> const &children,
                           PomadeTubeDesc *parentOut, std::string *err);

// K14 merge: children -> parent. When `roundTripParent` is non-null and
// every child matches its re-derived shape from that parent bit-wise
// (untouched children), the output is an exact copy of the hint. An edited
// hinted child updates only the original parent corners K14 recorded as its
// inherited holding boundary; internal K14 cuts stay child-local. Without a
// hint it uses the K7 average.
// Merging a selected sibling subset works the same way over the subset.
USDGENPOMADE_API bool PomadeMergeTubesCpu(std::vector<PomadeTubeDesc> const &children,
                        PomadeSubdivideDesc const &params,
                        PomadeTubeDesc const *roundTripParent,
                        PomadeTubeDesc *parentOut, std::string *err,
                        std::vector<PomadeTubeDesc> const *priorChildren = nullptr);

// Polyline arc length of center CVs (double accumulation, float result).
USDGENPOMADE_API float PomadeCenterArcLength(float const *cx, float const *cy, float const *cz,
                           int nCv);

// V0b (plan/18 §7 G1/G4): which child cell of a subdivided parent a world
// point falls in. K14 partitions the parent's root ring in the parent's root
// frame and each child's center starts at its cell's centroid (§2.3 step 2),
// so the cells ARE the Voronoi cells of the children's root centers in that
// plane and the test is a nearest-centre search there. Points outside the
// parent's root ring still land in a cell, which is what the region map and
// the per-child mesh fill need (every root belongs to exactly one child).
// `childRootCenters` is 3 floats per child in childIndex order. Ties take the
// lowest index; -1 when childCount <= 0.
USDGENPOMADE_API int PomadeOwningChildCell(float const parentRootCenter[3],
                         PomadeFrame const &parentRootFrame,
                         float const *childRootCenters, int childCount,
                         float const p[3]);

// -- K6: hierarchical sculpt (top-down, length-preserving) ------------------
//
// A child's authored shape is its derived shape plus deltas (§2.3). Deltas
// live in the DERIVED frames: per center CV the world offset expressed in
// the derived K4 frame at that CV, per section vertex the (du, dv) offset
// in the derived section plane (child-relative by construction, so no
// transform is needed for sections). Re-derivation re-applies the same
// local coords in the new derived frames, which is the plan's "offset in
// the parent's local frame" carried through the derivation transform.

// Sculpt deltas for one child (see above). Layout must match the derived
// desc exactly (counts and section t values); mismatches fail loudly
// (topology changed under the sculpt: re-subdivide).
struct PomadeShapeDeltas {
    std::vector<float> centerDu;  // per center CV
    std::vector<float> centerDv;
    std::vector<float> centerDw;
    std::vector<PomadeTubeSection> sections;  // per-vertex (du, dv)
};

inline void PomadeClearDeltas(PomadeTubeDesc const &like,
                             PomadeShapeDeltas *deltas)
{
    deltas->centerDu.assign(like.centerX.size(), 0.0f);
    deltas->centerDv.assign(like.centerX.size(), 0.0f);
    deltas->centerDw.assign(like.centerX.size(), 0.0f);
    deltas->sections = like.sections;
    for (auto &s : deltas->sections) {
        std::fill(s.u.begin(), s.u.end(), 0.0f);
        std::fill(s.v.begin(), s.v.end(), 0.0f);
        s.scale = 0.0f;
        s.twist = 0.0f;
    }
}

// Derive ONE child (shared code path with PomadeSubdivideTubeCpu: runs the
// same subdivision and picks childIndex, so hydrate and K6 re-derive
// bit-identical children).
USDGENPOMADE_API bool PomadeDeriveChildCpu(PomadeTubeDesc const &parent,
                         std::vector<PomadeFrame> const &parentFrames,
                         PomadeSubdivideDesc const &params, int childIndex,
                         PomadeTubeDesc *derived, std::string *err);

// Deltas of actual against derived, in the derived frames.
USDGENPOMADE_API bool PomadeComputeDeltasCpu(PomadeTubeDesc const &actual,
                           PomadeTubeDesc const &derived,
                           std::vector<PomadeFrame> const &derivedFrames,
                           PomadeShapeDeltas *deltas, std::string *err);

// Rescale a center curve about its root to targetLen (uniform along-curve,
// shape-preserving). Used by K6 and sculpt length preservation.
USDGENPOMADE_API bool PomadeRescaleCenterLength(float const *cx, float const *cy,
                              float const *cz, int nCv, float targetLen,
                              std::vector<float> *ox, std::vector<float> *oy,
                              std::vector<float> *oz, std::string *err);

// Arc-length resample of every section ring to ringVerts (child-relative
// coords preserved). Used to refit K7 aggregates to a parent layout.
USDGENPOMADE_API bool PomadeResampleDescRingsCpu(PomadeTubeDesc const &tube, int ringVerts,
                               PomadeTubeDesc *out, std::string *err);

// Adjacent-station slot alignment, the K14 rule PomadeSubdivideTubeCpu
// applies between a child's consecutive sections, for a caller that
// replaces ONE section of an existing tube (the Reposition attachment
// refresh installs a freshly split root under the child's transported
// upper sections).  K14 starts each split's ring at whichever parent corner
// first falls in the clip, so the new ring can arrive rotated against its
// neighbour; K5 interpolates per slot and would twist that span into a
// figure-eight.  Give `section` the neighbour's winding and the cyclic shift
// with the least summed squared slot distance, comparing both rings as K5
// places them (scale, twist) about their own means, normalised by their
// mean radius; near-ties keep the smallest shift, so an aligned ring is
// left bit-identical.  `outFrom` (optional) receives the renumbering, new
// slot i = old slot outFrom[i], when it returns true.  Returns false
// (section untouched) when already aligned or when the ring sizes differ.
USDGENPOMADE_API bool PomadeAlignSectionRingCpu(PomadeTubeSection *section,
                              PomadeTubeSection const &neighbour,
                              std::vector<int> *outFrom);

// K6: re-derive one child from an edited parent and re-apply its sculpt.
// lockChildren rides rigidly (world deltas frozen, stored deltas copied
// bit-exactly); otherwise deltas re-apply in the new derived frames and
// the center rescales to the old actual length when preserveLength holds.
// Exact-zero sculpt stays exact-zero (derivation drift is not authoring),
// so untouched-detection can compare against 0.0; commit persists the
// per-tube length basis separately (hydrate re-derives + re-applies +
// rescales to the committed basis). A moved parent re-partitions, so ring
// counts may change across the call: section deltas resample by arc length
// (centers always match; section-count or t drift still fails loudly).
USDGENPOMADE_API bool PomadeHierarchicalSculptCpu(
    PomadeTubeDesc const &parentNew,
    std::vector<PomadeFrame> const &parentNewFrames,
    PomadeSubdivideDesc const &params, int childIndex,
    PomadeTubeDesc const &oldActual, PomadeTubeDesc const &oldDerived,
    PomadeShapeDeltas const &oldStored, bool lockChildren,
    bool preserveLength, PomadeTubeDesc *outActual,
    PomadeShapeDeltas *outStored, std::string *err);

// Per-child half of the above over an already-derived child: K6 derives
// ALL of a parent's children with one PomadeSubdivideTubeCpu call, then
// applies each (one subdivide per parent per move instead of two per
// child — the TN-1 fix; bit-identical, since the inputs are). Errors
// keep the PomadeHierarchicalSculptCpu spellings.
USDGENPOMADE_API bool PomadeHierarchicalSculptApplyCpu(
    PomadeTubeDesc const &derivedNew, PomadeTubeDesc const &oldActual,
    PomadeTubeDesc const &oldDerived, PomadeShapeDeltas const &oldStored,
    bool lockChildren, bool preserveLength, PomadeTubeDesc *outActual,
    PomadeShapeDeltas *outStored, std::string *err);

}  // namespace usdGenPomade

#endif  // USDGEN_POMADE_HIERARCHY_H
