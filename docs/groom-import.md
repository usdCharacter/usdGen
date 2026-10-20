# Imported-groom operators

usdGen can reproduce grooms authored in external grooming tools whose generators work per hair set: roots
scattered on a scalp, strands blended from guides or grown on hair cards, then clumped, braided, twisted,
length-randomised, trimmed, frizzed, tied, pushed out and smoothed. This page lists the operators, modes and
data channels that make that possible. The importer for one such tool lives in
[the groom importer](../tools/sisir/README.md).

Everything here runs on the CPU reference lane. The CUDA and Vulkan planners refuse a graph that uses these
modes rather than silently running the default behaviour.

## Building blocks

| Piece | What it adds |
|---|---|
| `usdGen:curve:basis = "linear"` | Tiles publish polylines (`type = linear`), so strands keep their exact points. C3 sources accept linear curves. |
| CV cap | Up to 4096 CVs per strand on the CPU lane (`kUsdGenMaxCvsPerCurve`). The GPU lanes keep their own limits. |
| `primvars:usdGen:plane:<name>` | Any such primvar on a BasisCurves prim reaches operators as the plane `<name>` (uniform, vertex or constant). Used for per-card parameters, tile ids and per-guide flags. |
| Root-sampled mask maps | `rel usdGen:densityMap` (Scatter) and `rel usdGen:lengthMap` (Length) take `UsdGenPaintMap` / `UsdGenImageMap` targets, multiplied, sampled at each root. |

### Shared planes

Operators hand per-strand state to each other through these planes.

| Plane | Type | Written by | Meaning |
|---|---|---|---|
| `strandIndex` | uniform int | Scatter (non-random modes), CardSource | Generation order. Curves are Morton-sorted, so order-dependent operators key by this. |
| `strandSeed` | uniform int | Scatter, CardSource | Per-strand random key. |
| `primaryGuide` | uniform int | GuideInterpolate | Dominant blend guide, -1 for none. |
| `nearestGuide` | uniform int | GuideInterpolate | Parting-aware nearest guide by root distance. |
| `partId` | uniform int | GuideInterpolate | Parting side key (`curve * 2 + side`), -1 when unparted. |
| `mirrorSign` | uniform int | GuideInterpolate | +1/-1 forwarded from the guide set's `mirrorSign` plane. |
| `clumpId_0` | uniform int | Clump, Braid | Clump id, -1 unclumped or flyaway. |
| `restLength` | uniform float | Length (`randomRange`) | Strand length before length scaling. |
| `braidKnotMask` | vertex float | Braid | 1 - knot weight; Frizz uses it to keep crossings clean. |
| `collideUnresolved` | uniform float | Collide (`relax`) | 1 when a strand still touches a collider. |

## Profiles

Clump, Twist, Frizz, Braid and Width accept a root/mid/tip profile: a quadratic Bezier through the control
points (0, root), (midShift, mid), (1, tip). `mid` is a control value, not a value the curve passes through,
and `midShift` places it along the strand (clamped to [0.1, 0.9], Frizz [0, 1]).

## Operators

### UsdGenScatter (extended)

| Attribute | Default | Notes |
|---|---|---|
| `usdGen:distribution` | `random` | `random`, `even` (Poisson-disk darts with relaxation), `uvGrid` (deterministic grid in UV space), `vertex` (one root per unique vertex position, shuffled), `perGuide` (one root per guide root) |
| `usdGen:countMode` | `perFace` | `surface`: round(density x surface area) roots over the whole surface; the non-random distributions always count this way |
| `rel usdGen:densityMap` | | Masks thin roots stochastically; the target count is unchanged |
| `rel usdGen:guides` | | For `perGuide` and proximity |
| `usdGen:proximity:mode` | `none` | `distance` (within `usdGen:proximity:distance` of a guide root) or `coverage` (the `usdGen:proximity:coverage` share of the surface nearest the guides); roots pack into the kept domain |

`usdGen:seed` seeds the random stream (`std::mt19937`).

### UsdGenGuideInterpolate (extended)

| Attribute | Default | Notes |
|---|---|---|
| `usdGen:weighting` | `falloff` | `falloff` (the original), `nearest`, `inverseDistance` (3 nearest), `gaussian` (all guides, sigma from the guide-root spacing) |
| `usdGen:tightness` | 3 | Gaussian exponent, [0.25, 8] |
| `usdGen:bendFollow` | false | Root offsets ride the guide's rotation-minimising frame |
| `usdGen:defaultLength` | 0.01 | Straight strand along the root normal when there is no guide |
| `usdGen:part:points` / `counts` / `normals` | | Parting curves (attributes, not a relationship) |
| `usdGen:part:amount` / `angle` / `radius` / `distance` | 0 / 80 / 0.5 / 1 | Strands rotate away from the parting line; guides across it are suppressed |

### UsdGenClump (extended)

| Attribute | Default | Notes |
|---|---|---|
| `usdGen:clump:centers` | `density` | `count` (farthest-point seeding of `usdGen:clump:count` centres, each strand joins its nearest), `guides` (pull toward the guide curves), `guideSeeded` |
| `usdGen:clump:profile:mode` | `ramp` | `bezier` uses `usdGen:clump:root/mid/midShift/tip` |
| `usdGen:clump:flyaway:fraction` / `amount` | 0 / 0 | Escaping strands; amount 0 is fully free |
| `usdGen:clump:sub:*` | off | Angular sub-clumps: `enabled`, `count`, `spread`, `amount`, `ids` |
| `usdGen:clump:overrides`, `overrideCount` | | Per-strand overrides in generation order |
| `rel usdGen:guides` | | For `guides` / `guideSeeded` |

The reference behaviour pulls with an absolute blend: author `usdGen:preserveLength = 0` and
`usdGen:clump:method = "linearBlend"`.

### UsdGenBraid (new)

Weaves the strands of each guide into a three-lane braid. Controls `usdGen:braid:frequency`, `width`,
`flatness`, `knotTightness`, `fadeIn`, `fadeOut`, `rootSpread`, `tipSpread`, `tailLength`, `tailRelax`,
`tailCinch`, `referenceLength` (0 = bound surface's rest bounding-box diagonal), the pull profile
`root/mid/midShift/tip/amount`, `flyaway:fraction/amount`, and `knotMask`. Writes `clumpId_0` (3 x guide + lane)
and `braidKnotMask`.

### UsdGenTwist (new)

Rotates every point about the mean curve of its group (`primaryGuide`, or `clumpId_0` with
`usdGen:twist:useClumpIds`) by `usdGen:twist:amount` x the Bezier profile in degrees
(`root/mid/midShift/tip`), with per-strand `noise`; `mirrorSign` flips mirrored guides. `resnapRoot` puts a
moved root back on the surface.

### UsdGenLength (extended)

`usdGen:length:mode = "randomRange"`: scale = `(min + u (max - min)) randomize + (1 - randomize) max`, times
the `usdGen:lengthMap` masks; `shareByClump` gives a clump one random length, `followClumpMean` makes strands
follow their clump's mean curve. Strands are resampled uniformly and extended straight past the tip. Writes
`restLength`.

### UsdGenTrim (new)

Each curve of `rel usdGen:trim:curves` stands a blade of height `usdGen:trim:distance`, tilted by
`usdGen:trim:angle` (`invert` flips it). A strand is cut at its first crossing and resampled. Per-curve
overrides: `primvars:usdGen:plane:trimAngle`, `trimInvert`, `trimDistance`.

### UsdGenFrizz (new)

Two perpendicular sine waves plus 1-D value noise in a transport frame: `usdGen:frizz:amplitude` (fraction of
the strand's reference length, `restLength` when present), `wavesX`, `wavesZ` (waves per reference length),
`noise`, `noiseSize`, `sharpness` (0 smooth bumps, 1 sharp corners), the Bezier profile, `useClumpIds`
(clump-mates frizz together) and `preserveKnots` (reads `braidKnotMask`). The root never moves.

### UsdGenTieClump (new)

Hair ties as attributes: `usdGen:tie:positions`, `entryCounts`, `guideIndices`, `vertexIndices`, with
`rel usdGen:guides`. Strands of tied guides (by `primaryGuide`) are pulled toward each tie with
`usdGen:tie:amount` and a falloff along the strand set by `usdGen:tie:spread`.

### UsdGenCollide (extended)

`usdGen:resolveType = "relax"`: clearance blends from `usdGen:clearance:root` to `usdGen:clearance:tip`
along the strand; `usdGen:relax:iterations` sweeps push penetrating points out, smooth the correction and
restore segment lengths from the fixed root. All resolve types now search colliders through a bounding-volume
hierarchy.

### UsdGenSmooth, UsdGenResample, UsdGenWidth (extended)

- Smooth `usdGen:mode = "box"` with `usdGen:window`: one box-average pass, both ends fixed.
- Resample `usdGen:spacing > 0`: each strand gets `ceil(length / spacing) + 1` points (ragged).
- Width `usdGen:width:profile:mode = "bezier"`: width x clamp01(profile) from `usdGen:width:root/mid/midShift/tip`.

### UsdGenCardSource (new generator)

Grows strands on hair cards. `rel usdGen:cards` is a linear BasisCurves prim with one spine per card and the
card parameters as `primvars:usdGen:plane:*`; `rel usdGen:cardStrands` is a tile-space strand library with
`uvId` / `tile` planes. Every card is meshed into a surface grid (ribbon, cross, tube or braid cards; Free,
Mesh Normal, Surface or Conform orientation; twist and width profiles; optional subdivision with
`usdGen:cardSource:smooth`), and its tile's strands are mapped onto it and rooted on the scalp. Filters
`usdGen:cardSource:layer/uvId/tile`, `amount`, `rootFlatten`, `maxStrands`, and `usdGen:cvCount`. The full list of card planes is in
[the card-plane contract](../tools/sisir/card-planes.md).

## Hair shading

`UsdGenHairStrands` gained inputs for imported looks: separate primary and longitudinal roughness, azimuthal
roughness, cuticle tilt, backscatter, per-strand A/B/C three-stop colour gradients, saturation and offset
jitter, root blend and fade, an ambient fill, `ior` and `aoIntensity`. They default to the original look; see
[storm-fur.md](storm-fur.md).

The material can be built with either of two strand models (`-DUSDGEN_HAIR_SHADING_MODEL=karis|marschner`,
[storm-fur.md, *Strand models*](storm-fur.md#strand-models)). `marschner` is the absorption-based Marschner
model real-time groom tools ship, with their multiple scattering, flat ambient and hair occlusion, so imported
grooms shade as they did at the source. That build also bakes the `furDensity` primvar the occlusion reads.

## Checking a groom

`usdGenCookStats` (built with the tests) cooks every description of a stage on the CPU lane and prints
per-description strand and CV counts; `--bake out.usda` writes the result as plain BasisCurves and `--hash`
prints the reference output hash of the imported tool for comparison with its build log.
