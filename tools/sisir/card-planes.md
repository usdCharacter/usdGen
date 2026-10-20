# Card planes — `UsdGenCardSource` inputs

Status: **binding** for `tools/sisir/cards.py` (authoring) and `ops/cardSource.cpp` (engine).
`tools/sisir/cards.py` already authors exactly this; the importer should call it rather than
re-implement it (see §5).

Transport: every `primvars:usdGen:plane:<name>` primvar on a BasisCurves prim arrives as plane
`<name>` on that curve set's buffer (`usdGen/authoredPlaneConvert.h`). `uniform` interpolation →
one value per curve. Supported value types: `int[]`, `bool[]` (→ int), `float[]`, `double[]`
(→ float32), `float2[]/float3[]/float4[]`. **There is no 64-bit integer plane**, so u64 ids are
split into lo/hi int32 bit patterns.

Units (engine rule): lengths in **scene units** (the importer authors metres, `scale` multiplies
them), angles in **degrees**, everything else a fraction. All cm/percent conversions happen in
`cards.py`, never in the engine.

The two reference curve sets are told apart **by planes, not by relationship order**:
* a curve set carrying the uniform int plane `cardType` is the **card spine set**;
* a curve set without `cardType` that carries uniform int planes `tile` and `index` is the
  **tile strand library**.

The compiler requires the reference count of a node to equal its declared slots, so each
`UsdGenCardSource` binds **exactly one** spine prim (`rel usdGen:cards`) and **exactly one** library
prim (`rel usdGen:cardStrands`).

Both prims must live in the **same space as the bound scalp mesh** (no extra transform between
them; `cards.py` authors them as plain BasisCurves with identity xform). The engine reads their
untransformed points.

---------------------------------------------------------------------------------------------------

## 1. Card spine prim (one prim with ALL cards)

`def BasisCurves "cards"` (default path `<root>/cards`), `type = "linear"`, `wrap = "nonperiodic"`,
`purpose = "guide"`, one curve per scene `cards[]` entry with **≥ 2 points, in scene order** (erased,
masked and unbound cards included; the planes below decide what grows; cards with < 2 points are
omitted — they never build), `curveVertexCounts[c] = len(points)`, `points` = `cards[c].points ×
scale` (world spine as stored, root first). The engine's emission order is (layer, curve order).

Uniform planes (`primvars:usdGen:plane:<name>`, interpolation `uniform`, one value per card):

| plane | type | value (JSON source; clamps applied by cards.py) | engine use |
|---|---|---|---|
| `sceneIndex` | int | index of the card in scene `cards[]` (bookkeeping only) | — |
| `cardIdLo`, `cardIdHi` | int | low / high 32 bits of `cards[].id` (u64) as int32 bit patterns | strand id hash, length/frizz seed |
| `twinIdLo`, `twinIdHi` | int | low / high 32 bits of `mirrorTwinId` (0 = none) | mirror twins (§10 of card_geometry.md) |
| `mirrorPlane` | int | `mirrorPairPlane` (0 = x, 1 = z, 2 = y, mirrored about the mesh bbox centre) | mirror twins |
| `layer` | int | `cards[].layer` | `cardSource:layer` filter, emission order |
| `uvId` | int | `layers[layer].cardUvId`; −1 when the layer has none | build gate (0..9), filter, library match |
| `tile`, `tile2`, `tile3` | int | resolved tiles: `n = len(cardUvSets[uvId].uvTiles)`; `tile = ((uvTile % n) + n) % n`; `tile2/3 = wrap(uvTile2/3)` if ≥ 0 else `tile`; all −1 when `n == 0` or uvId invalid | strip 0 / Cross strip 1 / braid strand 3 tile |
| `vBias`, `vBias2`, `vBias3` | float | `clamp(uvTiles[tile*].longitudinalBias, 0.05, 0.95)` of tile/tile2/tile3 (0.5 if absent, non-finite or tile −1) | `cardSurface` V un-bias |
| `cardType` | int | `params.cardType` (0 Ribbon, 1 Cross, 2 Tube, 3 Braid) | |
| `orientation` | int | `params.orientation` (0 Free, 1 Mesh Normal, 2 Surface, 3 Conform) | frames |
| `widthSegments` | int | `params.widthSegments` clamped [1, 32] | columns / tube sides |
| `rootTri` | int | `cards[].rootTri` (fan-triangle index, OBJ face order; −1 none) | root normal N0 |
| `rootBary` | float3 | `cards[].rootBary` | root normal N0 |
| `erased` | int | 1 if `cards[].erased` | card skipped |
| `distributionMask` | float | layer `cardDistributionVertexMask` at the card root (bary blend of the fan triangle's vertex values /255), 1 if none | card built only if ≥ 0.5 |
| `lengthMask` | float | layer `cardLengthVertexMask` at the card root, 1 if none | multiplies the card length scale |
| `width` | float | `params.shape.width` clamped [0.001, 0.5] × scale | full card width |
| `widthRoot`, `widthMid`, `widthTip` | float | `shape.root/mid/tip` clamped [0, 1] | width profile |
| `widthMidShift` | float | `shape.midShift` clamped [0.1, 0.9] | width profile |
| `crossWidth` | float | `shape.crossWidth` clamped [0.05, 1] | Cross strip-1 width factor |
| `tubeFlatten` | float | `shape.tubeFlatten` clamped [0, 0.98] | Tube ellipse |
| `centerRidge` | float | `shape.centerRidge` clamped [−1, 1] | Ribbon ridge |
| `ridgeRootFlatten` | float | `shape.ridgeRootFlatten` clamped [0, 1] | ridge fade-in |
| `sink` | float | `params.sink` clamped [0, 0.05] × scale | root sink along N0 |
| `twistRoot`, `twistMid`, `twistTip` | float | `twist.root/mid/tip` (degrees) clamped [−180, 180] | twist profile |
| `twistMidShift` | float | `twist.midShift` clamped [0.1, 0.9] | twist profile |
| `lengthMin`, `lengthMax` | float | `length.min/max` clamped [0.01, 2] (fractions) | random length |
| `lengthRandomize` | float | `length.randomize` clamped [0, 1] | random length |
| `frizzAmount` | float | `frizz.amount` clamped [0,100] / 100 (displacement = frizzAmount · 0.25 · L) | card frizz |
| `frizzFrequency`, `frizzFrequencyZ` | float | `frizz.scale`, `frizz.scaleZ` clamped [0,100] / 100 · 10 (waves per card) | card frizz |
| `frizzNoise` | float | `frizz.noise` clamped [0,100] / 100 | card frizz |
| `frizzRoot`, `frizzMid`, `frizzTip` | float | `frizz.root/mid/tip` clamped [0, 1] | card frizz profile |
| `frizzMidShift` | float | `frizz.midShift` clamped [0.1, 0.9] (loader clamp; the frizz Bézier itself clamps 0..1) | card frizz profile |
| `frizzPreserveBraidKnots` | int | `frizz.preserveBraidKnots` | (braid only) |
| `pushOutEnabled` | int | `pushOut.enabled` | card push-out |
| `pushOutRoot`, `pushOutTip` | float | `pushOut.root/tip` clamped [0, 10] **cm → scene units** (× 0.01 × scale) | push-out distance |
| `pushOutSmoothness` | float | `pushOut.smoothness` clamped [0, 1] | push-out filter |
| `braidResolution` | int | `braid.resolution` clamped [16, 16384] | braid samples |
| `braidFrequency`, `braidWidth`, `braidFlatness`, `braidKnotTightness`, `braidFadeIn`, `braidFadeOut`, `braidRootSpread`, `braidTipSpread`, `braidTailLength`, `braidTailRelax`, `braidTailCinch` | float | `braid.*` (the scene file's clamps) | braid (weave not decoded, see §4) |

Card trim curves (`cardTrimCurves`) are **not** transported (trim-surface construction is not
decoded; Autumn has none). `cards.py` warns when a scene has them.
Image card masks (`layers[].cardDistributionMask` / `cardLengthMask` paths) are **not** evaluated
(`cards.py` warns and uses 1); per-vertex card masks are.

## 2. Tile strand library prims (one per `cardHairSources` key)

`def BasisCurves "cardStrands_<16-hex key>"` (default `<root>/cardStrands_<key>`),
`type = "linear"`, `purpose = "guide"`, `customData = { string "usdGen:cardSourceKey" = "<key>" }`.
Points are **tile space, never scaled**: x across the tile in [−0.5, 0.5] (+x ↔ +U), y root (−0.5) →
tip (+0.5), z depth (≈ ±0.13). Curves are emitted tile entry by tile entry in the source's `tiles`
order (first entry wins when two entries name the same `(uvId, tile)`), and within an entry in
strand order; strands with ≤ 1 point are **omitted** (they never grow).

Uniform int planes:

| plane | value |
|---|---|
| `uvId` | tile entry `uvId` |
| `tile` | tile entry `tile` |
| `index` | `j − firstStrand`: the strand's local index inside its tile entry (enters the strand id hash) |
| `tileStrandCount` | the entry's `strandCount` (including omitted ≤ 1-point strands; enters the 2M cap) |

A source without `tiles` gets the synthesized entry `[cardSourceUvId, cardSourceTile, 0, nStrands]`
of the first hair set that references it (legacy scenes without tile tables).

## 3. CardSource node attributes (`UsdGenCardSource`)

| attribute | type, default | from hair set (`card_source_params`) |
|---|---|---|
| `rel usdGen:cards` | → the spine prim | |
| `rel usdGen:cardStrands` | → the set's library prim | `hairSets[].cardSource` key |
| `usdGen:cardSource:layer` | int −1 | `settings.cardSourceLayer` (−1 any) |
| `usdGen:cardSource:uvId` | int −1 | `settings.cardSourceUvId` |
| `usdGen:cardSource:tile` | int −1 | `settings.cardSourceTile` |
| `usdGen:cardSource:braid` | bool false | `settings.cardSourceBraid` |
| `usdGen:cardSource:amount` | float 1 | `clamp01(settings.cardStrandAmount)` |
| `usdGen:cardSource:rootFlatten` | float 0 | `clamp01(settings.cardRootFlatten)` |
| `usdGen:cardSource:smooth` | bool true | `settings.cardSmooth` |
| `usdGen:cardSource:maxStrands` | int 2000000 | constant |
| `usdGen:cardSource:braidWidth` | float 1 | `settings.braidWidth` (braid source only) |
| `usdGen:cardSource:braidFlatness` | float 0.55 | `settings.braidFlatness` (braid source only) |
| `usdGen:cvCount` | int 8, [2, 4096] | resolutionType 0: `clamp(hairResolution, 3, 1000)`; type 1: `min(max(2·ceil(Lmax/d)+1, 8), 1000)`, `d = clamp(resolutionDistance, 0.01, 10)·0.01·scale`, `Lmax` = longest spine of the set's cards × lengthMax [INF: the reference uses a card-texture length] |

Bound surface: the description's `usdGen:surface` = the scalp mesh (roots snap to it; Mesh-Normal /
Surface / Conform frames read it; `rootTri` indexes its fan triangles in face order, so author the
mesh with the reference face order and fan-compatible polygons).

## 4. Outputs of `UsdGenCardSource`

Linear strands of `usdGen:cvCount` CVs (uniform arc length), Morton-sorted by root (curve index is
not generation order). Per curve: `curveId` = strand id `SplitMix64(SplitMix64(cardId) ^ j ^
((strip+1) << 40)) | 1` (u64), `rootPrim`/`rootUV` = scalp face / UV at the snapped root, root frame
(rootN = face normal, rootT = fan edge Gram–Schmidt, rootB = N × T, like Scatter), `rest` = points,
`hairT = i/(n−1)`. Uniform int planes: `strandIndex` (generation order), `strandSeed`
(`SplitMix64(id) & 0x7fffffff`), `cardIndex` (spine curve index of the card that grew it).

Braid cards (`cardType 3`) grow hair only with `usdGen:cardSource:braid = true`; the braid weave is
not decoded, so braid hair follows the length-scaled spine with the documented R2 root spread
(inferred). Mirror twins, push-out (exact closest point instead of the reference SDF grid) and
Conform normals (exact closest point) are implemented from the spec but have no reference sample.

Missing planes take the card defaults (`card::Card` in `ops/cardGeometry.h`); planes may be
authored int or float (the engine reads either).

Validation: on the sample scene the CardSource root counts of the eight card-grown sets (22560, 10120, 3337, 7331, 6911, 5196, 2405, 3212) and the card mesh size (85976 vertices / 331392 indices) equal the source tool's build log exactly.

## 5. Python entry points (`tools/sisir/cards.py`, stdlib + pxr)

* `author_cards(stage, scene, root_path, mesh_path, scale=1.0) -> dict` — authors §1 and §2,
  returns `{'cards': '<root>/cards', '<key>': '<root>/cardStrands_<key>', ...}`.
* `card_source_params(hair_set, scene, scale=1.0) -> dict` — §3 attribute values (`{attr name: value}`).
* `CARD_SOURCE_ATTR_TYPES` — `{attr name: Sdf type name string}` for those attributes.
* `card_source_key(hair_set) -> str | None` — the set's `cardSource` key.
* `author_card_source(stage, op_path, hair_set, scene, card_prims, scale=1.0) -> Usd.Prim` —
  defines a `UsdGenCardSource` prim with the attributes and both relationships.
