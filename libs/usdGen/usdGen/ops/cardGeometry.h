// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — hair-card surface grids for UsdGenCardSource.
//
// A card is a spine polyline plus shape parameters. Building it gives one
// rows x cols vertex grid per strip (Ribbon and Tube: one strip, Cross: two
// strips that share the centre line), with per-row texture parameter and
// width. Strands authored in the 2-D tile of a card are then placed on that
// grid with CardSurface().
//
// Construction order (per card, reference behaviour of the imported groom):
//   spine as authored -> random length scale (arc-length resample, linear
//   extrapolation, same point count) -> card frizz -> push-out from the
//   scalp -> Catmull-Rom subdivision x4 when smoothing (4n-3 rows) ->
//   root normal N0 and side reference S0 from the ORIGINAL spine -> sink
//   along N0 near the root -> frames per orientation (Free: transported,
//   Mesh Normal: fixed side, Surface: side from the near-root segment,
//   Conform: side turned toward the scalp tangent) -> per strip twist,
//   width profile, cross-section (ribbon with optional centre ridge, or an
//   elliptic tube) -> span record (row parameter = normalised arc length of
//   the un-sunk spine, row width, flipU, longitudinal bias).
// Mirror twins (a card whose twin id is smaller than its own id) share the
// random length/frizz seed with their twin and take the twin's root normal
// and side reference mirrored about the scalp's bounding-box centre.
//
// Braid cards (type 3) build no grid here: their weave is not reproduced,
// and their hair is placed by UsdGenCardSource directly on the spine.
#ifndef USDGEN_OPS_CARD_GEOMETRY_H
#define USDGEN_OPS_CARD_GEOMETRY_H

#include "usdGen/curveBuffer.h"
#include "usdGen/ops/strandUtil.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"

#include <cstdint>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace card {

enum CardType : int { kRibbon = 0, kCross = 1, kTube = 2, kBraid = 3 };
enum Orientation : int { kFree = 0, kMeshNormal = 1, kSurface = 2, kConform = 3 };

/// One card, with every parameter already in engine units (scene units,
/// degrees, fractions) and clamped to its documented range by Sanitize().
/// Field meanings follow the spine-prim plane contract (card_planes).
struct Card
{
    std::vector<GfVec3f> points;   // the spine as authored, root first
    uint64_t id = 0, twinId = 0;
    int mirrorPlane = 0;           // 0: x, 1: z, 2: y (about the scalp bbox centre)
    int layer = 0, uvId = -1;
    int tile[3] = {-1, -1, -1};    // strip 0 / Cross strip 1 / braid strand 3
    float vBias[3] = {0.5f, 0.5f, 0.5f};
    int cardType = kRibbon, orientation = kSurface, widthSegments = 1;
    int rootTri = -1;
    GfVec3f rootBary{0.0f};
    bool erased = false;
    float distributionMask = 1.0f, lengthMask = 1.0f;
    float width = 0.05f, widthRoot = 0.5f, widthMid = 0.5f, widthMidShift = 0.5f,
          widthTip = 0.5f, crossWidth = 1.0f, tubeFlatten = 0.0f, centerRidge = 0.0f,
          ridgeRootFlatten = 0.0f, sink = 0.0f;
    float twistRoot = 0.0f, twistMid = 0.0f, twistMidShift = 0.5f, twistTip = 0.0f;
    float lengthMin = 0.01f, lengthMax = 1.0f, lengthRandomize = 0.0f;
    float frizzAmount = 0.0f, frizzFrequency = 0.5f, frizzFrequencyZ = 0.5f,
          frizzNoise = 0.0f, frizzRoot = 0.0f, frizzMid = 0.5f, frizzMidShift = 0.5f,
          frizzTip = 1.0f;
    bool pushOutEnabled = false;
    float pushOutRoot = 0.0f, pushOutTip = 0.01f, pushOutSmoothness = 0.5f;

    /// Clamps every parameter to its documented range (idempotent).
    void Sanitize();
    /// True when a mirror twin is named and its id is smaller than ours.
    bool MirrorSecondary() const { return twinId != 0 && twinId < id; }
};

/// One strip of a built card: the grid hair is placed on.
struct Span
{
    uint32_t card = 0;             // index into the card list given to the builder
    int strip = 0;                 // 0, or 1 for the second strip of a Cross
    int uvId = -1, tile = -1;
    uint32_t rows = 0, cols = 0;
    bool flipU = false, tube = false;
    float vBias = 0.5f;
    std::vector<GfVec3f> V, N;     // rows * cols, row-major (vertex (i, j) = i * cols + j)
    std::vector<float> rowParam;   // per row: normalised arc length of the un-sunk spine
    std::vector<float> rowWidth;   // per row: |V[i,last] - V[i,0]| (tube: ring perimeter)
    float length = 0.0f;           // sum of row-to-row distances along column cols / 2
};

/// Inputs shared by every card of a build.
struct Context
{
    /// Closest-point structure over the WHOLE scalp mesh (triangles in fan
    /// order, so Card::rootTri indexes it), or null when no surface is bound.
    strand::TriangleBvh const *scalp = nullptr;
    GfVec3f bboxCentre{0.0f};      // scalp bounding-box centre (mirror plane)
    /// Largest distance between Card::rootTri/rootBary on the scalp and the
    /// spine root for the binding to be trusted (else the closest scalp
    /// point gives the root normal). Scene units.
    float bindTolerance = 1e-4f;
    bool smooth = true;            // Catmull-Rom subdivision (rows 4n-3, tube sides x4)
    /// Every card of the scene (mirror twins are looked up here by id).
    std::vector<Card> const *cards = nullptr;
};

/// The working spine after length scale, frizz and push-out (no smoothing).
std::vector<GfVec3f> ProcessSpine(Card const &card, Context const &ctx);

/// Root normal N0 and side reference S0 of a card (mirrored from its twin
/// for a mirror-secondary card). `twinFound` reports whether the twin was
/// found (it decides the normal sign and the Conform sign).
void RootFrame(Card const &card, Context const &ctx, GfVec3f *n0, GfVec3f *s0,
               bool *twinFound);

/// Builds the spans of card `index` of ctx.cards (empty for braids, cards
/// with fewer than 2 points, and erased / masked / unbound cards when
/// `honourInclude`). Spans whose uvId or tile is negative are still built
/// (callers that need hair skip them, mesh counts keep them).
std::vector<Span> BuildCard(uint32_t index, Context const &ctx, bool honourInclude = true);

/// True when the hair surface builds this card: uvId in 0..9, not erased,
/// distribution mask >= 0.5, at least 2 points, not a braid.
bool Included(Card const &card);

/// Point and normal of the card surface at (u, v) in [0,1]^2 of the tile
/// (u across, v root -> tip), pushed `depth` row widths along the normal.
/// v is un-biased by the span's longitudinal bias; u is mirrored by flipU.
void CardSurface(Span const &span, float u, float v, float depth, GfVec3f *position,
                 GfVec3f *normal);

/// Render-mesh size of a set of spans: sum rows * cols vertices and
/// sum 6 (rows - 1)(cols - 1) triangle indices.
void MeshCounts(std::vector<Span> const &spans, uint64_t *vertices, uint64_t *indices);

// --- reading the reference curve sets (card_planes contract) ---------------------------

/// The uniform plane `name` of a curve set with the given arity, int or float,
/// sized to the curve count; null when absent or malformed.
UsdGenPlane const *CurvePlane(UsdGenCurveBuffer const &set, TfToken const &name, uint8_t arity = 1);

/// A uniform scalar plane read as int or float whichever way it was authored.
struct PlaneColumn
{
    UsdGenPlane const *plane = nullptr;
    PlaneColumn(UsdGenCurveBuffer const &b, char const *name);
    bool Has() const { return plane != nullptr; }
    int Int(size_t c, int fallback) const;
    float Float(size_t c, float fallback) const;
};

/// The card spine set carries a `cardType` plane.
bool IsSpineSet(UsdGenCurveBuffer const &set);
/// The tile strand library carries `tile` and `index` planes and no `cardType`.
bool IsLibrarySet(UsdGenCurveBuffer const &set);

/// Decodes every curve of a card spine set into a sanitized Card (rest points;
/// missing planes take the card defaults).
bool DecodeCards(UsdGenCurveBuffer const &spineSet, std::vector<Card> *cards, std::string *error);

/// The spine curve id hash of a strand grown on strip `strip` of card `cardId`
/// from local tile strand `j`.
inline uint64_t StrandId(uint64_t cardId, uint64_t j, int strip)
{
    return strand::SplitMix64(strand::SplitMix64(cardId) ^ j ^
                              (uint64_t(strip + 1) << 40)) | 1ull;
}

/// Uniform Catmull-Rom subdivision with clamped end neighbours: 4 samples per
/// segment at tau = 0, .25, .5, .75 plus the last point (4n - 3 points).
std::vector<GfVec3f> CatmullRom4(std::vector<GfVec3f> const &p);

}  // namespace card
}  // namespace usdGen

#endif  // USDGEN_OPS_CARD_GEOMETRY_H
