# tools/sisir — Sisir groom import (`sir2usd`)

Converts a scene from the grooming tool **Sisir** (`.sir` archive or legacy bare `.json`) into a USD stage that
usdGen cooks: scalp mesh, guide sets, one `UsdGenDescription` per hair set with the reference stage order
as an operator stack, masks, trim/parting/tie data, card spines + tile strand libraries (through `cards.py`),
and `UsdGenHairStrands` materials. The original settings ride along in `customData` so a scene round-trips and
hair presets can be re-applied.

Only this folder names the source tool; the engine operators it targets use neutral names.

| file | role |
|---|---|
| `sir2usd.py` | reader + converter + preset application (CLI and library) |
| `cards.py` | card spines / strand libraries / `UsdGenCardSource` (plane contract: [`card-planes.md`](card-planes.md)) |
| `selftest.py` | synthetic-scene tests + optional sample check |
| `__init__.py` | package exports (`convert`, `apply_preset_to_stage`, …) |

## Running

Python 3.10 with the OpenUSD 26.08 bindings. **Always pass `-S`**: an editable package in that interpreter's
`site-packages` installs an import hook that hijacks `pxr` with a stub. `-S` skips those hooks, so the tools are
stdlib + `pxr` only.

```sh
# $USD is the OpenUSD install prefix; $PY the Python that USD was built against.
export PATH="$USD/bin:$USD/lib:$PATH"
export PYTHONPATH="$USD/lib/python"          # Windows installs: $USD/Lib/site-packages
PY="$PY -S"
# For usdchecker / usdGenCookStats also put the usdGen build resources on
# PXR_PLUGINPATH_NAME (source bin/_env.sh, or see bin/launch_usdview.ps1 -PrintEnv).

$PY tools/sisir/sir2usd.py Autumn.sir out/Autumn.usda                  # all hair sets
$PY tools/sisir/sir2usd.py Autumn.sir out/Autumn.usda --sets 0 "Set 1 Copy" --summary-json out/summary.json
MSYS_NO_PATHCONV=1 $PY tools/sisir/sir2usd.py --preset curly.json --description /World/Groom/Set_1 out/Autumn.usda [--output out/b.usda]
$PY tools/sisir/selftest.py [--sample Autumn.sir] [--out DIR] [--cards-module PATH/cards.py]
usdchecker out/Autumn.usda
```

(`MSYS_NO_PATHCONV=1` stops Git Bash from turning `/World/...` into a Windows path.)

Options: `--no-extract` (do not unpack the archive), `--active-ops-only` (omit stages the reference skips for a
set instead of authoring them disabled; presets then cannot switch them on), `--include-hidden-layers`
(guides/trim/parting curves of hidden layers still drive hair), `--cards-module PATH` (or `$SISIR_CARDS_MODULE`;
default `cards.py` next to `sir2usd.py`), `--shader-defs PATH` (the `shaderDefs.usda` used to detect which
`UsdGenHairStrands` inputs exist), `--quiet`.

Archive safety: entries containing `..`, absolute names, drive letters / `:` or NUL are rejected (reported
as warnings); extraction is confined to `<out>_assets/`; absolute asset paths inside a `.sir` are ignored.
Legacy `.json` scenes reference absolute files; the files that exist are copied into `<out>_assets/`.

The schema plugin: if `usdGenSchema` is not on `PXR_PLUGINPATH_NAME`, the converter registers
`plugin/usdGenSchema/resources` (or `$USDGEN_SCHEMA_PLUGIN`) itself so every attribute gets its declared type
and variability. `usdchecker` needs the schema **and** shader plugins on `PXR_PLUGINPATH_NAME` (use
`sisir_env.sh`); without the shader plugin it reports `UsdGenHairStrands` as missing from the Sdr registry.

## Stage layout

```
/World                      Xform (defaultPrim), metersPerUnit 1, upAxis Y
  <mesh>                    Mesh + UsdGenRestAPI: OBJ x meshImportScale, vertex st/normals, mask primvars
  Camera                    from scene camera (yaw/pitch convention inferred)
  Looks/Hair_<id>           Material -> Shader info:id UsdGenHairStrands
  Looks/<mesh>_skin         UsdPreviewSurface from renderMeshMaterial (see "Mesh material" below)
  Groom                     plain Xform (CPU reference lane; a UsdGenGroom would select CUDA)
    Guides/<Set>            UsdGenGuideSet
      curves                linear BasisCurves, the set's guides only (layerVisibility x visible layers)
    Cards/cards             card spines (cards.py)            Cards/cardStrands_<key>  tile strand libraries
    <Set>                   UsdGenDescription + UsdGenLookAPI + material binding
      Maps/                 UsdGenPaintMap / UsdGenImageMap per non-trivial mask
      TrimCurves            linear BasisCurves (+ per-curve trim planes)
      Ops/                  operators, LAST stage first (bottom sibling runs first)
```

Execution order (bottom → top of `Ops`): `scatter → interp` (guide-driven) or `cards` (card-grown) `→ clump →
braid → twist → length → trim → frizz → tieClump → pushOut → smooth → resample → width`. Every stage is
authored; stages whose reference gate fails get `usdGen:enabled = 0`, so a preset can switch them on.
`usdGen:guides` is authored only where the current settings consume guides (scatter with perGuide/proximity,
interp, clump guides/guideSeeded, enabled braid / tieClump): the compiler rejects a reference on an operator
that declares no slot, and accepts any operator without one.

### Mesh and bindings

* OBJ parsed (`v`, `vt`, `vn`, `f` in all index forms, negative indices); re-indexed to **unique (v, vt, vn)
  corners in first-occurrence face order** — the reference vertex numbering, so 0..255 vertex masks map 1:1
  (Autumn: 19003 positions → 19131 vertices = mask length). Face order and polygons are kept (no triangulation).
  Points `× meshImportScale`; `st` (OBJ convention = USD) and normals are vertex primvars; missing `vn` →
  area-weighted smooth normals per position.
* Guide root binding: reference triangle `t` = fan triangle `(f0, fk, fk+1)` of the polygons in file order;
  `rootBary` clamped/renormalised. Converted to usdGen's face parametrisation: `skinprim` = polygon index,
  `skinprimuv` = triangle barycentric `(b1, b2)`; quad bilinear `(u, v)` (affine start + Gauss-Newton projection
  onto the bilinear patch); n-gon `u = (k − 1) + b1`, `v = b2`. Validated against `points[0]`: if the fan
  position is off by more than 1e-4 m (or `rootTri` is invalid) the root is rebound by closest point.
  `primvars:sisir:rootUV` keeps the mesh UV at the root.
  Autumn: fan error ≤ 3.6e-8 m; `skinprimuv` error ≤ 7.5e-5 m (non-planar quads: the engine's bilinear patch
  is not the reference's triangle pair).

### Guides (`Guides/<Set>/curves`)

| reference | USD | rule |
|---|---|---|
| `curves[]` with `layer` in the set | one linear curve each | eligible = layer bit in `layerVisibility` (all bits for From-Cards sets; legacy single-set scenes: all), layer `visible`, ≥ 2 points, length > 1e-6 |
| `points` | `points` | metres, as stored |
| `id` | `primvars:usdGen:curveId` (uint64, uniform) | `id`; 0 / duplicate → `(1<<40) | (index+1)` |
| `mirrorTwinId`, `id` | `primvars:usdGen:plane:mirrorSign` (int, uniform) | −1 if `twin != 0 && twin < id`, else +1 |
| `layer`, scene index | `primvars:sisir:layer`, `primvars:sisir:curveIndex` | round trip |
| — | `primvars:usdGen:role = "guide"`, `purpose = guide`, per-CV `widths` 0.0002 | display; widths must be per CV for a reference set |

## Conversion table — hair-set settings → operator attributes

Values are first resolved like the reference reader: absent keys take the constructor defaults, legacy keys
(`hairCount`, `clumpTightness`, `root/mid/tipThickness`) are converted, load-time clamps applied (e.g.
`hairResolution` [3,1000], `frizz*` [0,100], `twist*` [−180,180], `hairThickness` ≥ 0.001, `thickness*`
[0.01,1], enums out of range → default). Stage-time clamps are left to the engine.
`clamp01(x)` = clamp to [0, 1]. "op.attr" = `usdGen:attr` on that `Ops` child.

| key(s) | operator.attribute | formula |
|---|---|---|
| `distribution` | `scatter.distribution` | 0 `uvGrid`, 1 `vertex`, 2 `even`, 3 `perGuide`; 4 → `cards` (CardSource) replaces scatter + interp |
| — | `scatter.countMode`, `scatter.seed` | `"surface"`, 1337 |
| `hairDensity` (per m²) | `scatter.density` | `max(0, v)` (scene unit = metre → per unit²) |
| `guideProximityMode`, `guideProximityRadius` (cm **distance**), `guideProximityDistance` (**coverage**) | `scatter.proximity:mode` / `:distance` / `:coverage` | mode 1 → `distance`, distance = `radius·0.01`; mode 0 → `coverage` = `clamp01(distance key)`, `none` if ≥ 0.9999; `none` for dist 3/4 or no guides |
| generation + distribution masks | `scatter.densityMap` → Maps | see Masks |
| `hairResolution`, `resolutionType`, `resolutionDistance` | `interp.cvCount` / `cards.cvCount` | type 0: `clamp(res,3,1000)`; type 1: `min(max(2·ceil(Lmax/d)+1, 8), 1000)`, `d = clamp(v,0.01,10)·0.01`, Lmax = max(guideSettings.defaultLength, longest set guide) (card sets: cards.py value) |
| `resolutionType = 1` | `resample.enabled`, `resample.spacing` | spacing `clamp(v,0.01,10)·0.01` |
| `guideInterpolation` | `interp.weighting` | 0 `nearest`, 1 `inverseDistance`, 2 `gaussian`; dist 3 → `nearest` |
| `guideInterpolationTightness`, `guideBendFollow` | `interp.tightness`, `interp.bendFollow` | as is |
| `guideSettings.defaultLength` | `interp.defaultLength` | metres |
| `partingAmount` / `partingAngle` / `partingRadius` / `partingDistance` | `interp.part:amount` / `:angle` / `:radius` / `:distance` | `radius = v/100` (fraction of the parting-curve length); others as is |
| `partingCurves[]` (set layers, visible) | `interp.part:points`, `part:counts`, `part:normals` | normals from `pointTris/pointBarys` (vertex normals, fan bary) when every point is bound |
| `clumpType` | `clump.enabled`, `clump.clump:centers`, `braid.enabled` | 0 none; 1 `guides` (needs guides); 2 `count`; 3 `guideSeeded`; 4 → braid |
| `clumpRandomCount` | `clump.clump:count` | `max(0, v)` |
| `clumpRoot/Mid/TipTightness`, `clumpMidShift`, `clumpAmount` | `clump.clump:root/mid/tip/midShift`, `clump.clump:amount`, `clump.clump:profile:mode = bezier` | as is (engine: `clamp01(amount·Bezier)`) |
| — | `clump.seed`, `clump.clump:method`, `clump.preserveLength` | 1337, `linearBlend`, 0 |
| `clumpFlyawayPercentage`, `clumpFlyawayAmount` | `clump.clump:flyaway:fraction` / `:amount` (and `braid.braid:flyaway:*`) | as is |
| `subClumpsEnabled`, `subClumpCount`, `subClumpTightness` (UI "spread"), `subClumpAmount`, `generateSubClumpIds` | `clump.clump:sub:enabled/count/spread/amount/ids` | as is |
| `clumpOverride` `[{i,v}]`, `clumpOverrideHairCount` | `clump.clump:overrides` (int[], INT_MIN = keep), `clump.clump:overrideCount` | dense array of length HairCount |
| `braidFrequency/Width/Flatness/KnotTightness/FadeIn/FadeOut/RootSpread/TipSpread/TailLength/TailRelax/TailCinch` | `braid.braid:frequency/width/…/tailCinch` | as is (relative to the reference length) |
| — | `braid.braid:referenceLength` | 0 (= bound-surface bbox diagonal) |
| clump profile keys, `clumpAmount` | `braid.braid:root/mid/midShift/tip/amount` | as is |
| `frizzPreserveBraidKnots && frizzAmount > 1e-4` | `braid.braid:knotMask` | bool |
| `twistAmount`, `twistRoot/Mid/Tip` (deg), `twistMidShift`, `twistNoise`, `twistUseClumpIds` | `twist.twist:amount/root/mid/tip/midShift/noise/useClumpIds` | as is; enabled iff `amount > 0` and any of root/mid/tip ≠ 0 |
| `twistRoot != 0` (scalp sets only) | `twist.twist:resnapRoot` | bool (card sets: false) |
| `lengthMin`, `lengthMax`, `lengthRandomize` (multipliers) | `length.length:min/max/randomize`, `length:mode = randomRange` | as is |
| `lengthUseClumpIds`, `lengthUseInterpolation` | `length.length:shareByClump`, `length.length:followClumpMean` | as is |
| length masks | `length.lengthMap` → Maps | see Masks |
| `trimAngle` (deg), `trimInvertAngle`, `trimDistance` (**raw metres**) | `trim.trim:angle/invert/distance`, `trim:normalSearch = 0.08` | as is; enabled iff the set has trim curves |
| `trimCurves[]` | `trim.trim:curves` → `<Set>/TrimCurves` | curves with `hairSet ∈ {−1, set}`, layer in set and visible (or `trim.allLayers`); if any curve is set-bound, planes `trimAngle/trimInvert/trimDistance` per curve (set-bound: its own `trim{}` or 0/false/0.1; others: the set values) |
| `frizzAmount` (%) | `frizz.frizz:amplitude` | `v/100·0.25` (fraction of the pre-scale length); enabled iff v > 1e-4 |
| `frizzScale`, `frizzScaleZ` | `frizz.frizz:wavesX`, `wavesZ` | `v/10` (waves per reference length) |
| `frizzNoise` | `frizz.frizz:noise` | `v/100` |
| `frizzNoiseSize` (cm) | `frizz.frizz:noiseSize` | `v·0.01` |
| `frizzNoiseSharpness`, `frizzRoot/Mid/MidShift/Tip`, `frizzUseClumpIds`, `frizzPreserveBraidKnots` | `frizz.frizz:sharpness/root/mid/midShift/tip/useClumpIds/preserveKnots` | as is |
| `clumpToHairTies`, `tiesClumpAmount`, `tiesClumpSpread` | `tieClump.enabled`, `tie:amount`, `tie:spread` | enabled iff `clumpToHairTies && amount > 1e-4` and ties bind set guides |
| `hairTies[]` | `tieClump.tie:positions/entryCounts/guideIndices/vertexIndices` | per tie with ≥ 1 binding to a set guide: position (m); entries = (index of the guide in the set's guide curves, `pointIndex`) |
| `pushOutEnabled`, `pushOutRoot`, `pushOutTip` (cm) | `pushOut` (UsdGenCollide) `.enabled`, `clearance:root/tip`, `resolveType = relax`, `relax:iterations = 24` | clearance `max(1e-4, v·0.01)` |
| `smoothness` | `smooth.strength`, `smooth.mode = box`, `smooth.window`, `iterations = 1`, `lockRoot` | `clamp01(v)`; window `max(2, lround(2·steps/steps100))`; enabled iff v > 1e-4 |
| `hairThickness` (radius, m) | `width.width`, description `usdGen:width:default` | `2·v` |
| `thicknessRoot/Mid/MidShift/Tip` | `width.width:root/mid/midShift/tip`, `width:profile:mode = bezier`, `replace = 1` | as is |
| `cardSourceLayer/UvId/Tile/Braid`, `cardStrandAmount`, `cardRootFlatten`, `cardSmooth`, `braidWidth/Flatness` | `cards` (UsdGenCardSource) `cardSource:*` | `cards.card_source_params` (contract §3) |
| set `cardSource` key | `cards.cardStrands` → `Cards/cardStrands_<key>`; `cards.cards` → `Cards/cards` | `cards.author_card_source` |
| set `name`, `visible`, `renderMaterialId` | description name (sanitised), `visibility`, material binding | invisible → `visibility = invisible` |
| set `color`, `layerVisibility`, edits, … | `customData.sisir.hairSet` | round trip |
| whole `settings` block | `customData.sisir.settings` | raw, unclamped (99 keys) |

Inert on a scalp groom (kept in `settings`, listed in `customData.sisir.inertSettings` when non-default):
`showSteps`, `pushOutDebug`, `trimDebugPlane`, `maskBrush*`, `showPaintVertexColors`, `pushOutSmoothness`
(never read), `lengthBreakup(Size)`, `textureDepth`, `textureTileBorder` (texture-module builds only).

### Masks

| reference | USD | notes |
|---|---|---|
| `distributionVertexMask`, `generationDistributionVertexMask` (if initialised) | faceVarying float primvar `primvars:mask:<Set>:<kind>` on the mesh (`value[k] = mask[faceVertexIndices[k]] / 255`) + `UsdGenPaintMap` `<Set>/Maps/<kind>Paint` → `scatter.densityMap` | all-255 masks are not authored; identical masks share one primvar |
| `lengthVertexMask` | same, → `length.lengthMap` | |
| `distributionMaskPath`, `lengthMaskPath`, `generationDistributionMaskPath` | `UsdGenImageMap` (`file` relative to the layer, nearest, clamp, raw, channel luminance) | no flip: the reference stores mesh UVs as (u, 1−v) at import (`FUN_14003dec0`) and reads row `v·(h−1)` from the top, which is USD's bottom-origin lookup |

### Materials (`renderMaterials[renderMaterialId]` → `Looks/Hair_<id>`)

| reference | `UsdGenHairStrands` input | formula |
|---|---|---|
| `Roughness`, `LongitudialRoughness` (β) | `roughness` | `clamp(√(2·v·Roughness), 1/255, 1)`, `v = (0.726β + 0.812β² + 3.7β²⁰)²` (R-lobe match) |
| `SpecularScale` | `specular` | `v/2` |
| `UsePresetColorSettings` on | `baseColor`, `tipColor`, `colorRamp`, `useMelanin = 0` | highest-weight gradient: root/tip = sRGB→linear of the gradient at t = 0/1; `colorRamp` = least-squares fit of `mix(root, tip, t^k)` to the 3-stop smoothstep gradient |
| preset off: `Melanin`, `MelaninRedness` | `useMelanin = 1`, `melanin`, `redness` | direct (coefficient sets differ) |
| `Hue_Random` | `randomHue` | `clamp01` (reference is one-sided) |
| `Opacity` | `opacity` | `clamp01` |
| `MultipleScattering` dual / not | `selfShadow`, `scatter` | dual: 0.45 / 0; else 0 / `FakeMultipleScatteringFactor` |
| `ColorMap` | `baseColorMap` ← UsdUVTexture on `st` | sRGB |
| — | `backlit = 1`, `randomValue = 0`, `randomRoughness = 0` | (`RandomRoughness` has no effect in the reference) |
| colours | `usdGen:look:rootColor/tipColor/rampExponent` on the description | displayColor bake |

Extended `UsdGenHairStrands` inputs (authored when `shaderDefs.usda` declares them, which it does from this
version on; otherwise kept in `customData.sisir.pendingShaderInputs`):

| reference key | input | conversion |
|---|---|---|
| `Roughness`, `LongitudialRoughness` (β) | `primaryRoughness`, `longitudinalRoughness` | `√clamp(2·v·Roughness, 1/255, 1)`, `√v` |
| `AzimuthalRoughness` | `azimuthalRoughness` | direct |
| `CuticleTiltAngle` (rad) | `cuticleTilt` | `degrees(v)/4` |
| `BackScatterScale` | `backScatter` | direct |
| `UsePresetColorSettings` | `usePresetColors`, `presetSrgb`, `jitterMode` | 1 when on |
| `ColorX_1..3`, `ColorPointX` | `presetColorX1..3`, `presetPointsX` | sRGB values (converted in the shader) |
| `ColorA/B/C_Weight`, `RandomColor_Offset` | `presetWeights` | (A, B, C, offset) |
| `Saturate_Random` | `randomSaturation` | `clamp01` |
| `RootBlendRate`, `RootScalpBlend` | `rootBlendRate`, `rootFade` | `clamp01` |
| `IndirectLightingMultiplier` | `ambientFill` (and `indirectMultiplier = 0`) | strand ambient without an environment term |
| `IOR` | `ior` | `max(1, v)`; read by the Marschner build only |
| `HairAOIntensity`, `ShowHairAO` | `aoIntensity` | `v` when shown, else 0; read by the Marschner build only |

### Hair shading: build the Marschner model

The table above authors the same inputs whichever strand model usdGen was built with. The default Karis build
approximates the reference look (different lobes, colour read through `pow(albedo, k)`, the sky through the
BSDF). For shading that follows the reference, configure usdGen with

```sh
cmake -S . -B build -DUSDGEN_HAIR_SHADING_MODEL=marschner
```

That build compiles the reference's own strand model into `UsdGenHairStrands`: the Chiang-variance lobe widths,
absorption along the refracted path, its TT azimuthal table, its dual-scattering composition and table, the
flat albedo ambient and the `exp(-HairAOIntensity * density)` hair occlusion (from the `furDensity` primvar the
same build bakes). [docs/storm-fur.md](../../docs/storm-fur.md#strand-models) lists what is ported, what is
kept on purpose (the table built at the first material's defaults, the brightening absorption-axis indexing,
an unnormalised TT lobe) and where Storm forces a difference (baked hair count instead of a density ray march,
per-fragment shading, no AgX display transform).

The port was written from the formulas of the reference's shipped strand shaders and the table generator in
its executable. No reference source file is included.

## Validation (Autumn.sir)

```sh
$PY tools/sisir/sir2usd.py Autumn.sir out/Autumn.usda && usdchecker out/Autumn.usda   # Success
usdGenCookStats out/Autumn.usda --hash                                              # headless cook (test build)
$PY tools/sisir/selftest.py --sample Autumn.sir
```

* Mesh 19131 vertices (= mask length) / 18922 faces; root targets `round(A·density)` = 5716 / 2287 (log
  values); guide bindings 6 + 6, fan error ≤ 3.6e-8 m; painted distribution vertices 38 / 28.
* Headless cook: the eight card-grown sets give 22560, 10120, 3337, 7331, 6911, 5196, 2405 and 3212 strands at
  40 points, equal to the build log; Set 1 → 153 strands (log 152), Set 1 Copy → 94 (log 88 = 95 − 7
  `deletedStrands`, which have no slot).

## Presets

`--preset <file> --description <path>` reads `{"version": 2, "module": "hair"|"texture", "settings": {…}}`,
filters keys like the reference (class 0 always; class 1 — density, proximity, ties, parting, push-out,
breakup, texture keys, `frizzNoiseSize` — only for `module: "hair"`; class 2 — `cardSource*`, `card*`,
`trim*`, mask paths, UI keys — never; unknown keys never), refuses a switch into/out of From-Cards, merges
into `customData.sisir.settings`, re-derives every operator attribute, `enabled` flag and guide relationship
from the new settings plus the stored `customData.sisir.context`, and appends to `sisir.presetHistory`.
Scene data (masks, curves, ties, card params) is not touched.

## Recorded, not converted

Per description, `customData.sisir.unsupported` (no engine slot) and `customData.sisir.approximations`:

* `deletedStrands` — index-keyed deletions; no slot, strands are not removed (Autumn "Set 1 Copy": 7 of 2287).
* `clumpOverride` on braid sets; braid-tail sub-clumps — the Braid operator has no slot.
* generation/distribution masks on card-grown sets — CardSource has no density-map input.
* approximations: generation × distribution masks multiplied (reference: one hash against both, i.e. min);
  distribution mask thins at scatter (reference keeps the strand numbering and filters at output); clump type 3
  decoded only for guideInterpolation 0; push-out exact closest point vs the reference SDF; card sets' final
  root re-snap; card distance-resolution length.

Scene level (`/World/Groom` `customData.sisir.unsupported`): blend shape, reference mesh, mannequin, comb
curves, transplant markers, card trim curves, render/viewer state, the texture-module scene (its baked strands
arrive through `cardHairSources`). Card sets are skipped with a warning (and their settings kept in
`customData.sisir.skippedHairSets`) when `cards.py` is unavailable.

### Mesh material

`renderMeshMaterial` becomes `Looks/<mesh>_skin`, following the reference mesh shader (`PS_Mesh`):

| Setting | UsdPreviewSurface |
|---|---|
| albedo texture × `albedoColor` | `diffuseColor` ← UsdUVTexture (sRGB, `scale` = colour). The texture is the scene mesh texture when `useSceneMeshTextureAsAlbedo` is set, else `albedoTexturePath`; an empty slot falls back to the other. No texture: `albedoColor`. |
| `roughness` × roughness texture `.r` | `roughness` ← UsdUVTexture `r` (`scale` = slider), or the slider (min 0.045) |
| `metalness` × metalness texture `.r` | `metallic`, same way |
| normal map, `normalStrength`, `normalFlipY` | `normal` ← UsdUVTexture, `scale`/`bias` = (2s, ±2s, 2) / (−s, ∓s, −1). `normalFlipY` marks a +Y-up (OpenGL-style) map, USD's convention, so only unflagged maps get Y negated. Strength ≠ 1 or a negated Y makes usdchecker warn about a non-standard normal-map scale; Storm applies it. |

Listed in the material's `customData.sisir.unsupported`: the detail normal map (UsdPreviewSurface has one normal
input), the cavity map (it darkens direct specular only) and subsurface scattering. A scene without
`renderMeshMaterial` gets the viewport `meshTexturePath` only, as before.
