# Storm fur shading and self-shadowing

The implementation works with stock OpenUSD 26.08 Storm/OpenGL. It provides
geometry-derived self-shadowing, R/TT/TRT strand highlights, absorption-colored
transmission, pixel-footprint highlight filtering, and bounded colored fill.
It does **not** establish quality or performance parity with a host renderer.

Two materials ship. `UsdGenHairPreview*` (three variants, C5-frozen inputs) is
the original approximate model described below. `UsdGenHairStrands` /
`UsdGenHairStrandsTranslucent` is a port of a host renderer's strand-hair shading and is
what a usdGen tile binds by default; see **Strand-hair shading** at the end.

## Progressive Storm viewport tiles

Launch the interactive Storm viewport through the repository launcher:

```powershell
.\bin\launch_usdview.ps1 examples\head-hair-closeup.usda
```

With no renderer argument, the launcher selects Storm (`--renderer GL`) and
adds OpenUSD's `--allow-async` flag. The viewer then polls completed scene work
while the CPU evaluator publishes finished hair tiles, so the first completed
tile can appear before the full groom finishes. Pass an explicit non-Storm
renderer name to leave asynchronous scene processing disabled.

Tile preparation and generation run on workers. Hydra input capture, source
notices, and `asyncPoll` remain serialized on the frontend thread, which is
the safe boundary for the stage scene index. A host other than usdview must set
`allowAsynchronousSceneProcessing` when it creates its imaging engine and poll
for asynchronous updates itself. This is currently a CPU publication path;
there is no stock-Storm GPU-resident tile handoff.

An active diagnostic preview waits for its whole-output colour pass, and
whole-groom occlusion plus the scalp-shadow cap settle with the final complete
generation. Progressive tiles therefore expose completed curve geometry early
without presenting those whole-groom results as partial data.

## Findings

The procedural groom publishes tiled BasisCurves, not one draw primitive per
hair. Its former self-shadow was a smoothstep of `hairT`, independent of strand
density, nearby curves, and light direction. The previous specular/transmission
functions also normalized zero-length projected directions for axial views.

The old `tools/fur_gen.py` used cone prototypes, `points` instead of PointInstancer
`positions`, no `protoIndices`, repeated IDs, and invalid visibility/camera
attributes. The generator now produces valid native instanced cubic curves,
vertex taper/root-to-tip coordinates, unique IDs, per-instance colors, a bound
hair material, and a combed coat. Existing files in `renders/fur` are historical;
regenerate them into a new output path using the tool below.

## Implementation

`furOcclusion.cpp` voxelizes the entire groom into one world-space grid. The
grid is anisotropic and derived from the groom's bounds at a target world voxel
size — 0.3 units, a host renderer's `Voxelization.Virtual.VoxelWorldSize` — never coarser
than the historical 48-cube fitted to the longest axis and never more than 128
voxels on an axis. `UsdGenFurOcclusionParams::voxelSize` moves the target and
`::resolution` pins the old cube instead. Every curve is sampled along the
evaluated cubic that Storm draws (pinned bspline or catmullRom, with the
extrapolated phantom end control points), not along its control polygon, and
each sub-sample deposits projected fibre area over the cell cross-section for
the three axes with a trilinear splat. Six prefix sweeps integrate optical depth
toward positive/negative X/Y/Z into one interleaved fixed-point grid (x1000, the
quantisation a host renderer's voxel pages use), and trilinear gathers publish two float3
vertex primvars, `furTauP` and `furTauN`.

The published number is a host renderer's `FHairTransmittanceMask::HairCount`: the
expected number of fibre crossings from the CV to the edge of the volume,
**including** the receiver's own half-voxel, exactly as a host renderer's voxel ray march
does — so the consumer applies a host renderer's `HairCount = max(0, tau - 1)` shift
before `Tf = pow(A_front, HairCount)`. These are expectations over a voxel
cross-section, not strand counts: a single fibre of width `w` in a voxel of edge
`h` contributes `w/h`, around 0.03 for real hair, while a dense groom reads tens.
Values saturate at 64 (`usdGen::UsdGenFurTauClamp`). Directional depth is
reconstructed with a normalised cosine-power-4 blend of the six stored depths,

    vec3 d = mix(furTauN, furTauP, step(vec3(0), L));
    vec3 w = L*L; w *= w;
    float tau = dot(w, d) / (w.x + w.y + w.z);

whose divisor is at least 1/9 for a unit `L`, so no epsilon is needed.

The groom's emitting surfaces are injected as opaque blockers, a host renderer's
`Voxelization.InjectOpaqueDepth`: without them light reaches hair straight
through the scalp and the unlit side of a head stays lit. Each triangle marks a
saturated shell two voxels beneath the surface and four voxels thick (a host renderer's
`InjectOpaque.BiasCount` / `MarkCount`), so one crossing reaches the ceiling
while roots resting on the surface keep their sideways and upward light. Which
side of a triangle is solid is decided by where the fur is, not by the winding
order, so a reversed or left-handed mesh bakes the same shell. The volume is
grown to contain that shell — a ground-plane emitter is the case that needs it —
but only as far as the occluder actually reaches, so a room-sized floor cannot
cost the groom its resolution, and only the part of an occluder inside the
volume is voxelized.

The shell stops at the far side of the solid. Two to six voxels is deeper than
an ear is thick (5-10 mm against 11-33 mm on the imported Autumn head), so the
shell under the back of the ear used to come straight out of its front, into
the hair lying over it: those CVs read a saturated count toward every light and
the hair rendered pitch black in the outline of the ear. Each sample the shell
is marked from now casts a short ray along the same inward normal to the next
surface crossing, and only layers that end inside that thickness are marked;
an ear, an eyelid or a lip, thinner than the two-voxel bias, gets no shell and
lets light through, which costs nothing visible. `testUsdGenFurOcclusion`
holds a zero-width probe above a 4 mm fin to it: 0.00 crossings up and
sideways, while light through the fin itself is still blocked.

All tiles of a procedural description share the volume. Width, point, topology,
basis and transform edits rebuild it; affected receiving tiles get primvar
dirties even when their own geometry did not change. Unchanged geometry shares
the preceding immutable arrays. The normal generation cache also retains these
planes. Occluder meshes and grid parameters are invisible in the tiles, so the
caller carries a `volumeKey` the bake reads and rewrites; the session cooker
keeps one per description. Light and camera motion perform no bake. Other
descriptions are separate volumes.

`TF_DEBUG=USDGEN_FUR` prints one line per bake: grid dimensions, voxel size,
cells, tiles, work jobs, CVs and occluder count.

The shader interpolates directional optical depth with squared light-direction
weights and evaluates Beer-Lambert transmittance. Environment occlusion averages
six directional transmittances. Reflection/refraction lobes use longitudinal
Gaussians, azimuthal terms, Schlick Fresnel, and absorption derived from albedo.
Artist gains share a bounded budget; this is an approximate real-time model,
not a normalized path-traced hair BSDF. The material still multiplies host Storm
shadow maps when enabled. Opaque alpha-to-coverage and translucent OIT variants
share the same scattering code. All existing input names/defaults are retained.

Depth is packed into vector attributes because twelve independent instance
scalar buffers exceed the GL per-stage SSBO limit on the tested RTX 4090.

### Why six directions and not more

Six axis sweeps are a low-frequency angular approximation, so the obvious upgrade
is to store more directions. Measured against a brute-force ray march of a
head-shaped field (combed tangents over an ellipsoid shell plus the opaque scalp
shell), with every candidate reconstructing through the same normalised
cosine-power blend, RMS error in optical depth relative to the reference was:

| stored directions | floats/CV | best RMS | RMS of `Tf = A_front^HairCount` (A 0.35) |
|---|---|---|---|
| exact sweep per direction (unreachable) | — | 5.8% | 0.059 |
| 6 axes, power 2 (the old formula) | 6 | 53% | 0.170 |
| 6 axes, power 4 (shipped) | 6 | 57% | 0.141 |
| 14, axes + cube corners | 14 | 40% | 0.145 |
| 18, axes + cube edges | 18 | 38.5% | 0.151 |
| 26, axes + edges + corners | 26 | 35.5% | 0.158 |
| L1 spherical harmonics | 4 | 47% | 0.353 |
| L2 spherical harmonics | 9 | 31.5% | 0.308 |

Tripling the storage, the gather and the vertex-primvar bandwidth buys about a
quarter of the depth error and nothing at all on the transmittance the shader
actually evaluates, because the remaining error is angular bandwidth — the hard
terminator where the scalp occludes — that no smooth blend over even 26 stored
directions recovers. Raising the cosine power of the existing six from 2 to 4
buys more than fourteen directions did, for free: it keeps lit and shadowed
directions apart instead of averaging them. Spherical harmonics win on depth and
lose badly on transmittance, because they ring across that same terminator.
`testUsdGenFurOcclusion` repeats the comparison in-tree against a rotated-bake
reference and fails if power 2 ever beats power 4 on transmittance.

## Use

Procedural CPU grooms automatically receive the density data and bind the
default material at high/veryhigh complexity — `UsdGenHairStrands` since the
strand-hair shading below; `UsdGenHairPreview*` remain available and are
bound by naming them on the description's material. There is no new
renderer plugin or OpenUSD patch. CUDA-to-stock-Storm publication remains
unimplemented in this checkout; this change does not add that handoff.

For native PointInstancer scenes, use the offline bake utility. It accepts one
BasisCurves strand per prototype, with constant or vertex widths, default-time
transforms, and instance masks. It includes all instancers in a shared volume,
then stores root/tip optical depth as four float3 **instance** primvars. The
viewport geometry stays instanced. Prototype `hairT` interpolates those values.
Rebake after moving/deforming/replacing instances; rotating lights and cameras
works without rebaking. Nested/mesh prototypes and animated bakes are not handled.

From an environment configured like `bin/launch_usdview.ps1`:

```powershell
.\bin\build_usdgen.ps1
python tools/fur_gen.py 200000 build/fur_200k.usda
.\build\usdGenBakeFur.exe build/fur_200k.usda build/fur_200k_shadowed.usda
.\bin\launch_usdview.ps1 build/fur_200k_shadowed.usda
```

Select High or Very High complexity. `inputs:selfOcclusion = 0.5` applies the
baked density; 0 disables it, 1 doubles extinction. Tune root/tip colors, lobe
widths/gains, and transmission through the existing material inputs. Every Mesh
on the stage is baked in as an opaque blocker, the way the procedural lane
treats a groom's emitting surfaces. The optional third argument pins a cube of
that many voxels (clamped to 8–128) instead of deriving the grid from the
groom's bounds.

The generator itself needs only Python. The bake executable needs the USD/core
DLLs on PATH (the standard project launcher environment). A typical Windows
prefix here is `<usdrig-src>/usd-install`.

## Verification and measured limits

`testUsdGenFurOcclusion` tests cross-tile occlusion, light-direction asymmetry,
width response, transform changes, occluder removal, no-op COW reuse, finite
depth, malformed geometry rejection, dirty notices, and typed Hydra transport.
It also calibrates the published scale (a slab of N parallel fibre layers of
known coverage must report N x coverage crossings, to within 8%, at three
different voxel sizes), checks that an opaque emitter blocks light to geometry
behind it while roots on it keep their sideways and upward light, checks that
reversing the emitter's winding changes nothing, checks that the volume key
catches an occluder edit the tiles cannot show, checks determinism bit for bit,
and scores the angular reconstruction against a rotated-bake reference over a
hair shell on an opaque core.
`checkC5.py` validates all three shader definitions and defaults through Sdr.

`usdGenShaders/test/validate_fur.py` renders all three material variants, checks
that shadows darken actual pixels, and verifies direct-light relighting. It
times 60 steady frames after 10 warmups with `glFinish`, excluding image export
and the bake. Example:

```powershell
python usdGenShaders/test/validate_fur.py build/fur_200k_shadowed.usda build/fur_validation_200k --usdrecord <usdrig-src>/usd-install/bin/usdrecord --width 1600
```

On the RTX 4090 in this workspace, the 200,000-instance / 1.6-million-CV scene at
1600x1200 measured 6.27 ms median with shadows and 5.95 ms with extinction disabled.
The shadowed p95 was 16.17 ms (desktop/scheduling variability); these are synchronized
CPU wall times, not isolated GPU timer queries. Baking and exporting the native
instanced scene took 1.61 seconds. JSON and rendered evidence are under
`build/fur_validation_200k`.

Bake cost, on the same box (`bin/trace_playback.ps1`, the engine's own TRACE
scopes; the scheduler's arena is eight threads by default):

| workload | before | after |
|---|---|---|
| `examples/head-hair-closeup.usda`, 474k strands / 5.7M CVs | 95 ms at a 48-cube (h 0.55) | 122–134 ms at 78x85x83 (h 0.30) with the head injected |
| `testUsdGenFurOcclusion` bench, 50k strands / 100k CVs, single thread | 16.2 ms at a 48-cube | 13.5–15.3 ms at 88x8x8 |

The head bake resolves 5x the voxels, walks the evaluated cubic instead of the
control polygon, and voxelizes 8800 scalp triangles, for 33% more time: the
splat, the occluder shell and the final merge were re-cut into ~370 curve-range
jobs instead of 62 tiles, the merge and the primvar allocation moved off the
commit thread, and the gathered volume became interleaved fixed point. The split
is roughly 7 ms positions, 70 ms splat, 13 ms occluders, 3 ms sweep and
interleave, 28 ms gather. The splat, sweep and occluder terms fall with the cube
of `voxelSize`; the positions and gather terms follow CV count and do not move.
A groom whose bake must be cheaper should raise `voxelSize` rather than lower
`maxDimension`, which does nothing until the groom is larger than 128 voxels.

The focused fur suite and Sdr checks passed. A serial
`ctest -L 'T0|T1'` over all 91 tests passed with the eleven usual
platform/tooling skips (2026-09-16, CUDA-off build tree). Under `--parallel` a
rotating test or two still fails on exit-time timing and passes when rerun
alone; that is unrelated to the fur path.

## What remains below a full engine hair renderer

Six-axis depth is a low-frequency angular approximation: it can leak or
over-occlude at oblique directions, at around 50% RMS against a per-direction
reference (see the table above, which is also the argument for not spending more
storage on it). Point-light depth is integrated beyond the light, so nearby
lights are approximate. Nonuniform transforms use conservative maximum width
scaling. Opaque blockers are a shell, not a solid fill, so a mesh thinner than
the bias distance does not block, and only the part of a blocker inside the
groom's own (shell-expanded) bounds is voxelized. Native instances interpolate
only root/tip depth, which loses interior detail on long or tightly curved hair.
The bake does not account for material opacity.

There is no new GPU density update pass, deep opacity shadow atlas, temporal
reconstruction, shadow LOD, or scene-wide mesh/hair mutual shadow pipeline.
Stock host shadow support is still needed for fur shadows on the scalp/other
objects and mesh shadows on fur. CPU bake costs are unsuitable for large animated
grooms every frame. Metal/Vulkan are untested, and the variant-A tangent still
depends on Storm's GL curve-patch layout.

A host renderer's groom pipeline uses density voxelization and optional dedicated deep
shadow maps, with view-dependent voxel sizing and specialized visibility and
composition passes. Achieving or exceeding that full feature/quality envelope
requires renderer integration and matched scenes/hardware benchmarks, beyond
the stock material/scene-index path implemented here.

References: [the host vendor groom pipeline and performance](https://dev.epicgames.com/documentation/unreal-engine/groom-scalability-and-performance-with-unreal-engine),
[Pixar volumetric hair methods](https://graphics.pixar.com/library/Hair/paper.pdf).

## Strand-hair shading

`usdGenShaders/resources/shaders/usdGenHairStrands.glslfx` and
`...Translucent.glslfx` share `usdGenHairStrandsBsdf.glslfx` and are registered
as the fifth and sixth defs in `shaderDefs.usda`. They are outside the C5
freeze, which covers the `inputs:` block of the three `UsdGenHairPreview*`
files and nothing else; the three frozen files are unchanged.

The model is a Marschner R/TT/TRT lobe plus dual scattering, in
`usdGenHairStrandsBsdf.glslfx`:

| the host renderer | Here |
|---|---|
| `HairBsdf.ush HairShading()` — Karis 2016 R/TT/TRT | `UsdGenStrandsShading()`, line for line, the host renderer's constants |
| `ComputeDualScatteringTerms()` — Zinke 2008, driven by a hair count | `UsdGenStrandsDualScattering()` |
| `A_front`/`A_back` from a 64x64x16 LUT (`HairStrandsLUT.usf`) | polynomial fits of the same half-sphere integrals |
| `KajiyaKayDiffuseAttenuation()` — the `Scatter` fake | fallback for curves with no `furTau*` primvars, as the host renderer uses it for cards |
| `Coverage = saturate(r / max(r, PixelRadius) * CoverageScale)` | same, from `widths` and the projection Storm used |
| `EvaluateEnvHair()` — the BSDF against the sky, `Area = 0.2` | 8-tap fibre-frame quadrature normalised by the analytic albedo |
| `GetHairColorFromMelanin()`, Chiang absorption | `UsdGenStrandsColorFromMelanin()` |

Inputs are the host renderer's names and defaults: `baseColor`, `tipColor`, `colorRamp`,
`useMelanin`, `melanin`, `redness`, `dyeColor`, `roughness`, `specular`,
`scatter`, `backlit`, `opacity`, `randomHue`, `randomValue`,
`randomRoughness`, `selfShadow`, `hairCoverageScale`, `minPixelWidth` and the
`baseColorMap` texture, plus the matching controls of § *Matching an imported
look* below, all of which default to the unchanged look. Output is linear radiance: the host renderer's BSDF carries no extra
`NoL`, and `light.diffuse.rgb` is radiance in the same convention Storm's own
`previewSurface.glslfx` uses, so a light that correctly exposes a
`UsdPreviewSurface` correctly exposes hair. There is no artist budget and no
tonemap, so a grazing highlight can exceed 1 and clip where the host renderer would roll it
off in its filmic curve.

`UsdGenTilePublisher::BuildDefaultMaterialDataSource()` binds the opaque
`UsdGenHairStrands` (`UsdGenTilePublisher::DefaultMaterialIdentifier()`; see
the next section for why) and carries the description's `usdGen:look:*` into its
parameters. Before this the default material received no look at all: the tile
bakes `usdGen:look:rootColor` into `displayColor`, which the shader reads as
the root albedo, but the tip colour, ramp exponent and hue/value jitter had no
route to Storm, so an authored dark-brown look rendered with the shader's own
light-brown tip. A multi-stop `usdGen:look:colorRamp` cannot be expressed in
the shader's single lerp; its first and last stops are used.

### Strand models

`UsdGenHairStrands` and `UsdGenHairStrandsTranslucent` can be built with either
of two strand models. The CMake cache variable `USDGEN_HAIR_SHADING_MODEL`
picks one:

```sh
cmake -S . -B build -DUSDGEN_HAIR_SHADING_MODEL=marschner   # or karis, the default
```

| | `karis` (default) | `marschner` |
|---|---|---|
| Lobes | Karis 2016, as above, plus the matching controls | Absorption-based Marschner: widths from a Chiang longitudinal variance, a TT azimuthal term read from the model's table, TRT narrowed by azimuthal roughness |
| Colour in the lobes | `pow(baseColor, k / cosθd)` | absorption `σ` from the albedo (Chiang's inverse), attenuated along the refracted path `exp(−2σ(1 + cos 2γt))` with Bravais' index; F0 and the index follow `ior` |
| Multiple scattering | Zinke as the Karis engine composes it, `a_f`/`a_b` from closed-form fits | Zinke as the Marschner model composes it: `exp(−n)` extinction of the direct BSDF, forward- and back-scatter lobes, × cos θL, `a_f`/`a_b` from its own table |
| Sky | through the BSDF (eight fibre-frame taps), optional flat fill | flat albedo-tinted ambient times hair ambient occlusion |
| Extra data | none | the `furDensity` primvar (local hair density) for the occlusion |
| Source | `usdGenHairStrandsBsdf.glslfx` | `usdGenHairStrandsMarschner.glslfx` |

Storm compiles glslfx when it draws, so the selection is a file:
`shaders/usdGenHairModel.glslfx` holds one `#define USDGEN_STRANDS_MODEL_KARIS`
or `..._MARSCHNER` line. CMake writes the build tree's and the install's copy
from the cache variable; the source copy selects `karis` so a source checkout's
shader directory still parses. Choosing `marschner` also defines
`USDGEN_FUR_LOCAL_DENSITY` on `sessionCooker.cpp`, which turns on the fur
bake's `furDensity` output. The `USDGEN_FUR_DENSITY` environment setting
overrides that at run time either way. Switching models takes a re-configure,
which recompiles that one source file and relinks.

Both builds declare the same inputs with the same defaults, so a material
authored for one loads in the other. Colour (root/tip, melanin, presets,
jitter), coverage, root fade and the stochastic sample mask are shared code;
only the lighting loop in `surfaceShader()` differs. The two inputs the Karis
lobes have no use for, `ior` and `aoIntensity`, are declared in both builds
and ignored by `karis`. `testUsdGenHairStrandsDefs` fails when the build's
shader copy selects a different model than the cache, or when `shaders/`
changed without a re-configure (CMake copies the directory at configure time).

#### The Marschner model

This is an absorption-based Marschner strand model for imported grooms, ported
so an imported groom shades as it did at the source (see
[the groom importer](../tools/sisir/README.md)). The shared inputs mean, in this build:

| input | Marschner meaning |
|---|---|
| `roughness` | Both lobe families when the split controls are unauthored: `B_R = r²` and longitudinal variance `v = r²`, i.e. the Karis widths. |
| `primaryRoughness` | `B_R = clamp(p², 1/255, 1)`, with no `√2·cos(φ/2)` scaling. A source `Roughness R` at variance `v` is `p = √(2vR)`. |
| `longitudinalRoughness` | The variance `v = l²`: `B_TT = v/2`, `B_TRT = 2v`. A source longitudinal roughness `β` gives `v = (0.726β + 0.812β² + 3.7β²⁰)²`. |
| `azimuthalRoughness` | TRT narrowing `sr·exp(sr(17cosφ − 16.78))`, `sr = saturate(1.5(1 − a))`, and the width of the TT azimuthal lobe. Floored at 0.02; unauthored (−1) is 0.35. |
| `cuticleTilt` | Shift `s = 4·radians(cuticleTilt)`, lobes at {−s, s/2, 3s/2} in `sinθL + sinθV`: the same R shift as the Karis lobes for the same value. Unauthored is `s = 0.03` rad. |
| `specular`, `backlit` | R × `2·specular`; TT × `backlit`. |
| `ior` | F0 = `((1 − n)/(1 + n))²` and Bravais' index. |
| `selfShadow` | Scales the baked hair count, as in `karis`. 0 switches dual scattering off; about 0.45 maps the bake onto the source's density units. |
| `scatter`, `backScatter` | Without dual scattering, the model's colourised wrap term and a weak `0.05·albedo` back-scatter fill. With it, `backScatter` scales the back-scatter lobes. |
| `indirectMultiplier` + `ambientFill` | Their sum is the gain of the flat ambient `albedo × mean sky radiance`. The sky never goes through the BSDF. |
| `aoIntensity` | Hair occlusion on that ambient: `exp(−aoIntensity · selfShadow · furDensity)`. |

The TT azimuthal term is, in the source, a 64 × 64 table built on the CPU. Its
rows and columns are closed-form (`R = 1/(max(0.05, 2(y/63)²)·√(2π))`,
`G = 0.5 − x/63`, lobe `R·exp(−(cosφ + G)²)`, sampled at
`(0.5 − 0.5 sinθL, √azimuthal)`), so the shader evaluates it exactly, filtering
included. The lobe is broad and is not normalised. White hair's directional
albedo is about 2.1 at azimuthal roughness 0.35, 1.5 at 0.48 and 28 at the
0.02 floor. Dark hair absorbs most of TT, so imported looks are unaffected,
but light hair on this model is brighter than the Karis model's. That is how
the model ships, and the port keeps it. `check_hair_energy.py` covers the
Karis lobes only.

The multiple-scattering averages are, in the source, a 32 × 16 × 16
half-float texture (`|sinθL|`, roughness, absorption) filled on the GPU by
Monte Carlo. Storm cannot bind a texture to this material, so
`usdGenShaders/test/tabulate_hair_marschner.py` recomputes it by quadrature
from a NumPy reference of the model. It resamples the result onto 12
incidence nodes (uniform in angle) × 5 roughness nodes × the 16 absorption
texels and prints the GLSL table. Against the full lattice, between nodes,
`a_f` is within 0.012 absolute, `a_b` within 0.003 and the forward columns
within 0.002. `testUsdGenHairMarschnerTable` re-derives the table and fails if
the glslfx no longer holds it. Two properties of the source are kept:

- The table is built at IOR 1.55, shift 0.03 rad and azimuthal roughness 0.35.
  The source builds it from its first material, so a material's own IOR,
  tilt and azimuthal roughness reach the direct lobes but not multiple
  scattering. Moving them changes `a_b` by up to 40–57%.
- The absorption axis is built at `σ = coord²` and read at `coord = σ`, which
  brightens multiple scattering in dark hair.

Where the port departs from the source:

- The hair count is usdGen's baked directional depth (`furTauP`/`furTauN`)
  times `selfShadow`, shifted by one crossing as on the Karis path, not a
  per-frame density ray march.
- Hair occlusion reads `furDensity`. It is the fur bake's hair-only density,
  in crossings per voxel along an axis, averaged over the three axes and
  gathered trilinearly at each CV. It is published only when the bake's
  `localDensity` is on (Marschner builds, or `USDGEN_FUR_DENSITY=1`), and not
  for instanced fur. Without it the occlusion is 1.
- Shading is per fragment, not per vertex, and the direct lobes get the Karis
  path's pixel-footprint widening.
- Every Storm light is evaluated, not one key light plus ambient.
- No display transform. The source applies AgX and gamma 2.2; Storm writes
  linear radiance through its sRGB correction.

### Matching an imported look

Grooms imported from real-time grooming tools carry a strand material that the
Karis model above cannot express with one roughness and one root/tip lerp: a
sharp white primary highlight over almost flat coloured lobes, an azimuthal
roughness, and per-strand colour presets with saturation jitter. Both shipped
strand files take these extra inputs. Every default selects the expression the
material evaluated before the input existed, so **an unauthored material
renders bit-identically** (see *Keeping the default bit-identical* below).
They are material inputs only: `usdGen:look` does not carry them, so author a
`UsdGenHairStrands` material and bind it to use them.

Lobes (`UsdGenStrandsLobes` in `usdGenHairStrandsBsdf.glslfx`):

| input | type | default | meaning |
|---|---|---|---|
| `primaryRoughness` | float | -1 | R lobe roughness alone, in `roughness` units: `B_R = p²`, clamped to [1/255, 1]. < 0 follows `roughness`. |
| `longitudinalRoughness` | float | -1 | TT and TRT roughness: `B_TT = l²/2`, `B_TRT = 2l²`, clamped to [1/255, **4**] so a Chiang longitudinal variance `v` is reachable as `l = √v`. Also indexes the dual-scattering fits and the dome albedo (clamped to their fitted 0..1). < 0 follows `roughness`. |
| `azimuthalRoughness` | float | -1 | TRT azimuthal broadening `Np = s·exp(s·(17cosφ − 16.78))`, `s = saturate(1.5(1 − a))`. Every `a ≤ 1/3`, and the unauthored −1, is Karis' TRT exactly. TT keeps its fixed azimuthal fit. |
| `cuticleTilt` | float (degrees) | -1 | Karis' cuticle shift: lobes centred at {−2, +1, +4} × tilt in `sinθL + sinθV` (R through Karis' `ShiftR`); dual scattering follows. < 0 keeps the built-in 0.035 rad (≈ 2.005°). |
| `backScatter` | float | 1 | Gain on the dual-scattering back-scatter (`LocalScattering`) lobe. |

`randomRoughness` jitters all three roughnesses by the same per-strand factor.

Colour, applied in this order after the root/tip lerp and the melanin blend,
then `baseColorMap`, then the jitter:

| input | type | default | meaning |
|---|---|---|---|
| `rootBlendRate` | float | 0 | > 0: the root/tip blend is `smoothstep(0, rate, hairT)` (tip colour reached at that fraction of the strand) before `colorRamp`. 0: linear in `hairT`, as before. |
| `usePresetColors` | float | 0 | Blend (0..1) from the root/tip/melanin albedo to the strand's preset gradient. |
| `presetColorA1/A2/A3`, `…B1–B3`, `…C1–C3` | color3f | browns / light / near-black | Three stops of gradients A, B, C. |
| `presetPointsA/B/C` | float4 | (0, .33, .66, 1) | `mix(mix(c1, c2, smoothstep(x, y, t)), c3, smoothstep(z, w, t))`; clamped to [0,1] and forced ascending (each ≥ previous + 1e-4). |
| `presetWeights` | float4 | (1, 0, 0, 0) | xyz: weights of A, B, C (normalised); w: selector offset. A strand takes A when `fract(hairId + w) < x/Σ`, B below `(x+y)/Σ`, else C. Only the picked gradient is evaluated. |
| `presetSrgb` | float | 0 | 1: the stops are sRGB-encoded, blended as authored and linearised afterwards (clamped to [0,1] first) — how display-referred presets are defined. 0: linear stops. |
| `jitterMode` | float | 0 | 0: today's symmetric jitter (`randomHue` rotates ±H/2 turn about the grey axis, `randomValue` scales by 1 ± V, `randomSaturation` adds ±S, each on its own hash of `hairId`), then the offsets in HSV. 1: one-sided HSV keyed on `hairId` itself: `h = fract(h + hueOffset + id·randomHue)`, `s = sat(s + saturationOffset + id·randomSaturation)`, `v = sat(v + valueOffset + id·randomValue)` — correlated with the preset pick, as preset models define it. |
| `randomSaturation` | float | 0 | Per-strand saturation jitter (HSV, additive). |
| `hueOffset`, `saturationOffset`, `valueOffset` | float | 0 | Constant HSV offsets (hue in turns). `saturationOffset = -1` makes the hair grey. |

The one-sided HSV path and the preset pick both need `hairId`; without it every
strand uses id 0.

Lighting and coverage:

| input | type | default | meaning |
|---|---|---|---|
| `indirectMultiplier` | float | 1 | Multiplies the dome as the BSDF sees it (lobes, dual scattering, scatter fake). |
| `ambientFill` | float | 0 | Flat albedo-tinted ambient: `albedo × mean dome radiance × fill × visibility`, the visibility being the mean over the eight dome taps of `exp(−max(0, hairCount − 1))` (the `hairT` canopy without a density bake). Needs a DomeLight. |
| `rootFade` | float | 0 | `alpha *= 1 − fade·(1 − smoothstep(0, 0.18, hairT))`: the hairline blends into the scalp. Spent through the stochastic sample mask like any other alpha. |

#### Converting an imported material

Formulas for a material described the way real-time grooming tools commonly do
(the importer applies them; the engine never sees the source units):

- **Roughness.** With a Chiang longitudinal roughness `β` and a primary scale
  `R`, such tools use `v = (0.726β + 0.812β² + 3.7β²⁰)²` and lobe widths
  `{clamp(2vR, 1/255, 1), v/2, 2v}`. Author `longitudinalRoughness = √v` and
  `primaryRoughness = √clamp(2vR, 1/255, 1)`. Example: β = 0.916, R = 0.015
  gives v = 3.945, `longitudinalRoughness` 1.986, `primaryRoughness` 0.344.
- **Azimuthal roughness** maps directly. Its TRT effect is the same formula;
  its TT effect (a roughness-indexed table in such tools) is not reproduced.
- **Cuticle tilt.** A tool that centres its lobes at {−s, s/2, 3s/2} in
  `sinθL + sinθV` (s in radians) matches the R lobe — the only lobe narrow
  enough for the shift to show — with `cuticleTilt = degrees(s)/4`; TT and TRT
  then sit at s/4 and s instead of s/2 and 3s/2, which is invisible at their
  widths.
- **Colour presets** map one to one: gradients A/B/C → `presetColor*`, their
  points → `presetPoints*`, the three weights and the random offset →
  `presetWeights`, `presetSrgb = 1` for display-encoded colours,
  `usePresetColors = 1`, `useMelanin = 0`, `jitterMode = 1`, hue/saturation
  randomness → `randomHue`/`randomSaturation`.
- **Ambient.** A flat `albedo × ambient × multiplier` model is
  `indirectMultiplier = 0`, `ambientFill = multiplier`, with a DomeLight of the
  ambient colour and intensity.
- **Back-scatter gain** → `backScatter`; **root/scalp blend** → `rootFade`;
  **root blend rate** → `rootBlendRate` (such tools apply it to the root/tip
  tint of the non-preset path).

#### What is approximated

- `a_f`/`a_b` are fits of the Karis BSDF at ONE roughness. A split material
  indexes them by `min(longitudinalRoughness, 1)`, since TT and TRT carry the
  colour multiple scattering accumulates; the R width and the TRT azimuthal
  broadening reach them only through that roughness. Exact again at the
  defaults.
- The Karis lobes ignore `ior`. The literal F0 is folded by the driver into
  the lobes' Fresnel factors, and every authorable form tried (a uniform F0, an
  exact-zero correction on top) moved default pixels; Karis' fitted refraction
  `n'` is for 1.55 anyway, which is also what imported materials usually carry.
  The input exists for the Marschner build (§ *Strand models*).
- `ambientFill`'s visibility uses the directional hair count, not a local
  density. A tool's scalar hair AO (`exp(−k·ρ_local)`) is the Marschner
  build's `aoIntensity`, which reads the `furDensity` primvar that build bakes.
- The TT azimuthal term ignores `azimuthalRoughness` (see the table above).
- For a faithful match of an imported groom, build the Marschner model instead
  (§ *Strand models*); everything in this section is the Karis approximation.

#### Keeping the default bit-identical

"Same value" is not enough on a GPU: the driver constant-folds and contracts
literals, so replacing a literal by a uniform of the same value, or adding an
exact-zero correction to an expression it had folded, can move a pixel. Each
control is therefore wired so that its default runs the original instructions:
a uniform branch that keeps Karis' literal expression whole (the R lobe's
`ShiftR` and the Gaussian argument it feeds), a correction that is an exact
`× 1` on top of the untouched original (TRT azimuthal: `s·exp((s−1)·…)`), or a
branch around a block that did not exist before (presets, HSV, root blend,
fill, fade). Measured, not assumed: the render check's frames and six extra
frames with the existing jitter, melanin, scatter, ramp, coverage and opacity
inputs away from their defaults are pixel-identical (opaque variant, which
renders deterministically) to the shaders before these inputs existed. The
bisection that found the three offenders is the reason the Karis lobes do
not read `ior`.

### How the coverage is spent, and why the default is the opaque variant

the host renderer spends its coverage through a per-sample visibility buffer. Stock Storm has
no such pass, but a material CAN drive the sample mask: Storm declares
`hd_SampleMask` as a stage output only for `PRIM_POINTS`
(`codeGen.cpp:2931-2941`), yet `gl_SampleMask` is a built-in fragment output in
GLSL 4.0+ that needs no declaration, and a glslfx surface shader is plain GLSL.
`usdGenHairStrands.glslfx` writes it directly. Three options were measured on
`examples/head-hair-closeup.usda` (1280x960, complexity veryhigh, 60 frames
after 10 warmups with `glFinish`, RTX 4090):

| | TempleCam | HeadCam | quality |
|---|---|---|---|
| plain alpha-to-coverage | 12.74 ms | 48.79 ms | coat reads as one strand thick; scalp shows through |
| binary hashed alpha | 13.82 ms | 50.69 ms | correct density, visible pepper grain |
| **hashed `gl_SampleMask`** | **14.61 ms** | **57.87 ms** | correct density, no grain |
| translucent / OIT | 20.16 ms | 61.05 ms | smooth, but loses most of the coat on HeadCam |

**Why plain alpha-to-coverage fails.** Its sample mask comes from the alpha
value and the pixel position alone, so N overlapping strands at coverage `a`
select the *same* samples and composite to `a` rather than `1-(1-a)^N`. And in
usdview it does not run at all: `UsdImagingGLRenderParams::enableSampleAlphaToCoverage`
defaults to **false** (`usdImagingGL/renderParams.h:119`) and usdview only ever
sets it for the pick pass (`Usdviewq/stageView.py:2259`), so a material that
spends its coverage through alpha alone has no mechanism there whatsoever.
Driving `gl_SampleMask` works in both viewers because it does not depend on the
alpha-to-coverage state.

**What the shader does instead.** `multisampleCount` is bound as a
render-pass-state param on every draw (`hdSt/renderPassState.cpp:454-467`), so
the material reads the real sample count — 4 by default
(`HDX_MSAA_SAMPLE_COUNT`, `hdx/taskControllerSceneIndex.cpp:78`). The fragment
builds a mask whose popcount is `alpha * samples` with stochastic rounding, so
the expectation is exact, rotates the run of set bits by a second hash so two
strands with the same popcount rarely take the same samples, writes it to
`gl_SampleMask[0]` and outputs alpha 1 so alpha-to-coverage's own mask is
all-ones and ANDs away. The mask is also ANDed with the rasterizer's geometric
coverage, which is what a silhouette wants. Both hashes are keyed on the
**strand** (`hairId`) where it exists, not the pixel: a per-pixel key breaks
each strand into a dashed line and speckles the coat. With one sample there is
no room for a fraction and it falls back to keeping or dropping the strand.
The cost over plain A2C is early-Z: writing `gl_SampleMask` disables it, which
is most of the +14% on HeadCam.

**Why not OIT.** Storm's OIT pool is `8 * width * height` fragments for the
**whole frame**, handed out by one atomic counter, and a fragment past the end
is silently dropped (`hdx/shaders/renderPass.glslfx RenderOutputImpl`,
`HdxOitResolveTask::_PrepareOitBuffers`). This groom exhausts it and loses the
crown on HeadCam; at a quarter of the density the same frame is clean, which is
the test that identifies the cause. The loss is by draw order, so it moves as
the camera moves. Bind `UsdGenHairStrandsTranslucent` by hand for a hero still
of a groom that fits the budget.

One thing here is still not the host renderer's. the host renderer's default is `RasterizationScale = 0.5`
with 8x MSAA, i.e. a strand is snapped to **one MSAA sample**, 1/8 of a pixel,
and the geometry carries most of the sub-pixel coverage. A usdGen tile
publishes `minScreenSpaceWidths = 1` (C2:35), the host renderer's *stable rasterization* mode,
which maximises the coverage that has to be paid back in the mask — measured
mean coverage over the hair pixels of the HeadCam frame is 0.54. the host renderer's own
comment recommends 1.325 where there is no TAA, which is what we have; the
value we use is 1 because it is the contract's, not because it is the best one.
Lowering it is a C2 change, not a shader one.

### Sample count, and measuring anti-aliasing at all

Storm's MSAA sample count comes from `HDX_MSAA_SAMPLE_COUNT`, which
`HdxTaskController` reads; the stock default is 4. The strand material's sample
mask scales with whatever it is given, so the sample count is also the number
of coverage levels a sub-pixel strand has to spend — at 4 it has five, and the
coat reads as speckle and broken dashes. `bin/launch_usdview.ps1` and
`bin/record_usd.ps1` therefore default it to **8**, guarded so a caller's own
value wins; `record_usd.ps1 -Msaa N` overrides per render. This is launcher
configuration, not an OpenUSD patch.

head-hair-closeup at 1280, against a 4x supersampled reference:

| | 4x | 8x | 16x | reference |
|---|---|---|---|---|
| frame time | 14.84 ms | 16.71 ms | 24.52 ms | -- |
| isolated-pixel spike score | 0.00631 | 0.00536 | 0.00530 | 0.00176 |
| character-asset frame time | 3.25 ms | 3.63 ms | 6.03 ms | -- |

8 buys nearly all of the improvement for +12.6%; 16 costs +65% for almost
nothing more and is indistinguishable in the crops
(`renders/ue-parity/aa1_sheet_*.png`). Raising the count does not close the gap
to the reference — one shading sample per pixel and a finite mask cannot — but
it is the cheapest large step.

`record_usd.ps1 -Supersample N` and `render_ue_parity.ps1 -Supersample N`
render at N x and box-downsample in linear light (`bin/downsample_linear.py`).
That is the converged image, so it separates "the shader is wrong" from "this
needs more samples", and it is how every number above was judged. Cost goes as
N^2.

It is also the offline route for stills. There is **no size ceiling in
practice**: at 1280 wide, N = 8 renders 10240 x 7680 and completes, and the
script's own `ValidateRange(1, 8)` is the only limit. usdrecord prints an
`HdStVBOSimpleMemoryManager can't reassign ranges` warning at every N
including 1, which is benign — the image is correct. **N = 4 is the practical
choice**, because the character-asset TempleCam spike score converges there and
stops moving:

| | 1x (8x MSAA) | SS 4 | SS 6 | SS 8 |
|---|---|---|---|---|
| spike score | 0.000562 | 0.000184 | 0.000172 | 0.000169 |
| mean luminance | 0.41503 | 0.41405 | 0.41390 | 0.41384 |

`renders/ue-parity/final1_ss4_*` is the current converged set for all five
cameras; `final1_*` is the same five at the 1x shipping defaults.

**When comparing a supersampled reference to a 1x frame, compare `mean(alpha)`
or `mean(RGB * alpha)`, never `mean(RGB)`.** usdrecord writes RGBA and on a
scene with no backdrop the hair's alpha is its coverage; the two images encode
the uncovered pixels differently, and comparing raw RGB makes a correct
renderer look like it is losing half its energy. `downsample_linear.py` carries
the same warning.

### Why the mask is a rotated run of bits, and not something cleverer

The mask lights *exactly* `k = alpha * samples` of the N samples, stochastically
rounded, then rotates that run per strand. Two more obvious designs were tried
against the converged reference on head-hair-closeup and both are worse:

| | spike score | cost |
|---|---|---|
| shipped: stratified run of k bits | **0.00536** | 16.71 ms |
| cross-strand coverage profile | 0.00588 | 16.71 ms |
| true per-sample visibility (`gl_SampleID`) | 0.00603 | 50.49 ms |

The reason is stratification. Lighting exactly k samples is a stratified
estimate of the coverage; letting each sample decide for itself is plain
Bernoulli with the same mean and higher variance. the host renderer can afford per-sample
because its visibility buffer resolves coverage analytically rather than
stochastically — the mechanism being "more correct" does not survive the
variance arithmetic here.

The cross-strand profile fails for a second reason worth remembering: a strand
wider than a pixel already gets *exact* edge coverage from the rasterizer's
MSAA, so any shader-side profile across the ribbon softens an edge that was
already right. The same double-counting argument applies to raising
`minScreenSpaceWidths`, and is why that stays at 1.0 until someone measures it
on a real groom.

What is left between the 1x frame and the reference (0.00536 against 0.00176)
is one shading sample per pixel and a finite sample count. Nothing inside a
stock-Storm fragment shader closes it: it needs more samples (`-Msaa 16`,
`-Supersample`) or a temporal filter, and Storm has no temporal filter.

### Strands with no `hairId`

The sample mask is keyed on `hairId` so that strands decorrelate from each
other while each stays solid along its length. A native `BasisCurves` prim
bound to `UsdGenHairStrands` directly has no such primvar — only usdGen tiles
publish it — and keying on `gl_FragCoord` alone is not enough: two strands
crossing the same pixel hash to the same rotation and the same popcount, take
an identical mask and collide on every sample, so the coverage of a stack never
builds past one strand's alpha. Measured on a patch of quarter-pixel native
curves, mean coverage was 0.251 against a converged 0.353. The fallback key is
`gl_PrimitiveID`, which restores it to 0.325 (0.92x). It changes per segment
rather than per curve, so a strand may pick different samples across a segment
boundary; that is the lesser artefact. Grooms published as tiles are unaffected
— they take the `hairId` branch, and their coverage already matches the
converged reference to 0.1% (0.3110 against 0.3113 on `styled-fur-plane`).

### Picking and the ID pass

Writing `gl_SampleMask` raises the question of whether a partially covered
strand stays pickable. It does, and the reason is structural rather than lucky.
The pick pass composes the material's shader like any other pass, so
`surfaceShader()` does run and does write the mask — but `HdxPickTask` allocates
its AOVs single-sampled (`pickTask.cpp:316`, `Allocate(dimensions, format,
/*multiSampled*/ false)`), and a fragment shader's sample-mask output has no
effect on a non-multisample framebuffer. The material's own fallback writes an
all-ones mask whenever `multisampleCount` is 1, so it would be safe even if that
changed.

Measured, on native `BasisCurves` in usdview through `testusdview`, 16 picks
over the frame:

| | `/World/Hair` | `/World/Scalp` | nothing |
|---|---|---|---|
| `UsdGenHairPreview` (no mask write) | 11 | 4 | 1 |
| `UsdGenHairStrands` | 11 | 4 | 1 |
| `UsdGenHairStrands`, forced `gl_SampleMask[0] = 0` | 11 | 4 | 1 |

Bit-identical, including with every sample killed — which is the proof that the
ID pass ignores the mask.

Separately, and NOT caused by this material: usdGen's synthetic **tiles** are
not pickable in usdview at all, with or without the mask write (1 of 9 picks
hits the head mesh, the rest return nothing, identically both ways). That is a
property of the synthetic-prim path, not of the shader.

### Measured composited opacity

The acceptance question for a groom is whether the dense mass reads opaque.
`I = a*H + (1-a)*B` is solved per pixel from two renders of the groom alone
(head hidden) over a black and a white backdrop, which needs no shader edit and
assumes nothing about the hair's colour. TempleCam, complexity Very High,
`examples/head-hair-closeup-render.usda`:

| | hair pixels | mean alpha | dense mass mean | dense mass fully opaque |
|---|---|---|---|---|
| usdrecord 1280x960 | 756 592 | 0.856 | **0.976** | 93.1% |
| usdview 478x432 @ dpr 1.25 | 209 400 | 0.847 | **0.968** | 89.7% |
| usdview, translucent/OIT bound instead | 219 814 | 0.773 | 0.948 | **20.0%** |

The two viewers agree, which also rules out a viewport or HiDPI mismatch
between them: both `UsdAppUtilsFrameRecorder` and usdview's `StageView` call
`SetFraming`, so `GetViewport()` is the render-buffer size in both. The OIT row
is what "the whole hair mass is transparent, roots included" looks like as a
number: nothing is missing, every pixel is merely short of opaque.

The usdview figures come from `testusdview` driving usdview's own `StageView`,
so they are that viewer's real output, not a reconstruction.

### Verification

- `usdGenShaders/test/fit_hair_scattering.py` integrates the ported BSDF and
  prints the `a_f`/`a_b` constants the shader carries. Re-run it rather than
  editing those constants.
- `usdGenShaders/test/check_hair_energy.py` (`testUsdGenHairEnergy`, T1)
  integrates the BSDF over the sphere and asserts it creates no energy: the
  white furnace is 1.00074 cos-weighted over incidence and 0.9999 at normal
  incidence, and falls monotonically with roughness. Karis' model is **not**
  energy-conserving per view direction — each lobe's azimuthal profile is
  normalised separately and R then takes near-total grazing Fresnel — so the
  per-direction peak reaches 1.15 near 78 degrees at the lowest roughness; the
  script asserts that headroom explicitly rather than hiding it.
- `usdGenShaders/test/check_hair_strands_defs.py` (`testUsdGenHairStrandsDefs`,
  T1) is the C5 check's counterpart for the two new defs, and asserts the three
  frozen defs are still discovered.
- `usdGenShaders/test/check_hair_strands_render.py`
  (`testUsdGenHairStrandsRender`, T2) draws through Storm and requires the
  pixels to move when `baseColor` changes (a failed glslfx compile falls back
  silently, so every check is differential), when the key light rotates, when
  `selfShadow` changes the hair count, and when `hairCoverageScale` clears
  coverage. On the opaque variant it also checks every matching control
  (§ *Matching an imported look*): each moves the pixels the way its
  definition says (preset weights split the strands, preset points move the
  stops, `saturationOffset -1` greys red hair, `rootFade` lowers coverage, …),
  and the neutral settings — split roughnesses equal to `roughness`,
  `azimuthalRoughness` 0.3, `cuticleTilt` authored as 0.035 rad — reproduce
  the default frame to within one LSB.

### The `scatter` pin, and scenes with no lights

`HairShading()` ends, under `HAIR_COMPONENT_MULTISCATTER`, with

    S  = EvaluateHairMultipleScattering(...);   // GlobalScattering*(Fs+LocalScattering)
    S += KajiyaKayDiffuseAttenuation(...);      // the wrapped-Lambert fake

The Kajiya term is not a fallback for missing dual-scattering data: it is
always added, gated by `GBuffer.Metallic`, which `MaterialAttributeDefinitionMap`
maps to the hair label **Scatter**. the host renderer's strands path is dark there only because
`HairSampleToGBufferData` hardcodes `Out.Metallic = 0`. This material keeps it
as the input it is, added in the same place, and defaults it to **0** — the host renderer's own
default and what its strands path forces.

That matters for a viewport. Every Marschner lobe needs the light near the
specular cone: with the light co-located with the camera the Gaussian argument
is `sinThetaL + sinThetaV = 2*sinThetaV`, so only strands perpendicular to the
view fire at all. Measured on `examples/styled-fur-plane.usda` (roughness 0.35,
baseColor 0.02, strands vertical, camera ~35 degrees up): `S_R ~ 5e-6`,
`S_TRT ~ 2e-4`, and `TT` is dead because its azimuthal term is
`exp(-3.65*cosPhi - 3.98) = 4.8e-4` at `cosPhi = 1`. The coat renders black,
and that is correct, not a bug: forcing the hair count to zero leaves it black
too. Any scene lit only by `usdrecord`/usdview's camera light needs either a
real light or a non-zero `scatter`; 0.3 reads as a plain dark-brown coat.
`examples/styled-fur-plane.usda` and `examples/scatter-grow-plane.usda` author
no lights at all; adding a dome light, which is what usdview's own "Enable
Default Dome Light" does, makes them read as a plain dark-brown coat at
scatter 0 with nothing artificial added.

The default is 0 and not something larger because the term is added on top of
dual scattering and is not attenuated by it, and `sqrt(BaseColor)` is about six
times `BaseColor` for dark hair. Rendered side by side on the relit parity
scene (`renders/ue-parity/ws1_scatter0_temple.png` against
`ws1_scatter1_temple.png`), `scatter = 1` replaces a dark brown coat with deep
self-shadowing and visible specular bands with a flat pale khaki one: it does
not merely lift the shadows, it overrides the authored look entirely. There is
also no way for a scene to turn it down again -- `UsdGenLookAPI` has no
`scatter`, so a description cannot override it without authoring a whole
material -- which is the other reason the safe value is the default.

`ScatterTint` is `Shadow < 1 ? pow(BaseColor/luma, 1 - Shadow) : 1`, and the host renderer's
`Shadow` there is `FShadowTerms.TransmissionShadow` — the shadow map, not the
dual-scattering transmittance. Feeding it the latter pins the tint at full
saturation everywhere inside a dark coat and turns the fill orange.

### Tangent direction

`HairBsdf.ush` says of its `N`: *"the vector parallel to hair pointing toward
the root"*. Storm's `inData.Neye` is the derivative with respect to the curve
parameter, which runs root to tip, so the material negates it. The sign is not
cosmetic: `sinThetaL` and `sinThetaV` flip together, so the Gaussian argument
`sinThetaL + sinThetaV - Alpha_p` tilts the cuticle shift the wrong way and the
primary highlight sits toward the tip instead of the root. `CosPhi`,
`cosThetaD` and the Kajiya term are invariant; the R lobe's `ShiftR`, the
dual-scattering `theta_h` and the signed `a_f`/`a_b` fits are not.

### Binding your own material used to lose half the look

The description's `usdGen:look:*` used to reach the shader by two different
routes, and only one of them survived an explicitly bound material:

- `rootColor` (plus the ramp, hue/value jitter and any colour map) is **baked
  by the tiles into `displayColor`**, a primvar. It arrives whatever material
  is bound.
- `tipColor` is a **material input**. The publisher copies it onto the
  *synthetic* default `UsdGenHairStrands` material it builds from the look, so
  the default binding carries both ends of the ramp.

Bind your own `UsdGenHairStrands` material and the synthetic one is replaced,
so `tipColor` silently falls back to its Sdr default `(0.21, 0.115, 0.045)` — a
light blond — while `rootColor` keeps arriving. The albedo is
`mix(displayColor, tipColor, pow(t, colorRamp))`, so you get a half-applied
look, and on a groom viewed down the strands the tip end is most of what you
see. Measured on a converted groom that is not part of this tree: with a bound material,
halving the look's `rootColor` moved the lit coat by 7.5% and setting the look
to pure black still read a shader albedo of `0.2086` in R. Removing the binding
took the same halving to 24% and the black-look albedo to 0.0015.

**Fixed by making the look material-independent.** The tile now publishes the
rest of the look as CONSTANT primvars alongside the baked `displayColor`:

| primvar | type | from |
|---|---|---|
| `hairTipColor` | `color3f` | `usdGen:look:tipColor`, or the last ramp stop |
| `hairColorRamp` | `float` | `usdGen:look:rampExponent` |
| `hairRandomHue` | `float` | `usdGen:look:hueJitter` |
| `hairRandomValue` | `float` | `usdGen:look:valueJitter` |

The shader prefers the primvar where it exists (`#ifdef HD_HAS_hairTipColor`)
and the material input otherwise, so a bound material inherits the look and an
authored `inputs:tipColor` on the *default* material still wins nothing it did
not win before. `rootColor` is deliberately NOT repeated: it already travels as
`displayColor`, and a second channel for it would only be a way for the two to
disagree.

Each primvar is published only where the look RESOLVES it to something other
than its schema fallback, for the reason the `usdGen:look` precedence rule
exists at all: a description that merely applies
`UsdGenLookAPI` has a full set of schema-default look values, and publishing
those would be indistinguishable from the user asking for them. So an
unauthored look publishes none of the four, every `#ifdef` is false, and the
default-material path is bit-identical to before. It also means a forwarded
asset `displayColor` — one flat colour per curve — never acquires an invented
tip.

Cost: four constant values per tile. Constants are one value each rather than
one per CV, so this is nowhere near the per-stage SSBO limit that forced
`furTauP`/`furTauN` to be packed vec3s (§ *Why six directions and not more*),
and it does not
touch the instanced lane at all — `usdGenBakeFur` publishes *instance*
primvars on a PointInstancer, which is a different path.

A look-only edit moves no point and changes no topology, so the look is now
mixed into the generation's `colorDigest`, the same digest `usdGen:preview`
uses to force a recolour of every tile. Without that the strands would keep
their previous colour while the synthetic material updated underneath them.

Setting `usdGen:look:bakeMode = "perCV"` is still not a fix for anything here —
it bakes the resolved ramp into `displayColor` per vertex, but the shader has
no way to know that and blends toward the tip on top of it, applying the tip
twice.

Pinned by `tests/testUsdGenLookPrimvars.cpp`.

### What is approximated, beyond the list above

- `a_f`/`a_b` are 39-coefficient closed forms rather than the host renderer's 64x64x16 LUT.
  They keep the incidence angle, as the host renderer's does. Measured strictly between the
  fitted nodes: `a_f` max 10.0% / rms 4.4% within +-60 degrees and 15.5% / 5.0%
  to +-75; `a_b` 12.1% / 3.0% and 12.1% / 3.6%. Keeping the angle costs about
  8% of the frame on the TempleCam (13.82 ms against 12.76) and is worth it,
  because `a_f` enters as `Tf = a_f^n`: the theta-averaged fit the script also
  prints is max 38% / rms 15% on `a_f` inside +-60 degrees, and at ten
  crossings 15% on `a_f` is a factor of four in transmittance.
- `delta_b` and `sigma_b` follow Zinke's published equations. the host renderer's shipped
  lines multiply where the paper adds, and repeat one square root; both are
  small shifts of an already broad back-scatter lobe.
- The hair count is the six-axis baked optical depth (`furTauP`/`furTauN`)
  interpolated with squared direction weights, not a voxel ray march or a deep
  opacity map, so it keeps every limit listed under "What remains below a full
  engine hair renderer" above.
- The dome is eight taps of `HdGet_domeLightIrradiance`, which is
  cosine-convolved and therefore over-smooth for a lobe; the quadrature is
  normalised by the analytic directional albedo so it is exact for a constant
  environment and the taps only carry direction and occlusion. A DomeLight with
  no `inputs:texture:file` gives Storm no irradiance texture at all, and the
  shader then treats the dome as a uniform environment of the light's own
  colour rather than dropping it, which is what `previewSurface.glslfx` does.
- A Hydra-instanced tile (native PointInstancer prototypes) has no
  `ApplyInstanceTransform` in scope in a material, so coverage uses the prim
  transform only and a per-instance scale is not accounted for.

## Hair shadowing the scalp

The largest remaining difference from the host renderer reference, once the strand
material was in, was not the hair: it was the skin. Every gap between clumps
showed fully lit scalp, so the temple fade read as hairs pasted onto a bright
surface rather than as hair growing out of a shadowed one.

### Storm casts no shadows, from any light

Authoring `inputs:shadow:enable` on the scene's DistantLight does nothing at
all, and the reason is structural, not a missing parameter:

* `HdStLight::_PrepareSimpleLight` (`hdSt/light.cpp`) asks for
  `HdLightTokens->hasShadow`, whose string is `"hasShadow"`.
  `UsdImagingPrimAdapter::GetLightParamValue` resolves that to the USD
  attribute `inputs:hasShadow`, which no UsdLux light has. `shadowEnable`
  (`"shadow:enable"`) is declared two lines away in `hd/light.h` and never
  read here, so `SetHasShadow(false)` is unconditional.
* `HdxSimpleLightTask` forces `SetHasShadow(false)` again unless
  `HdxShadowParams::enabled`, and nothing in `pxr/usdImaging` ever produces
  `HdLightTokens->shadowParams`.
* `HdxSimpleLightTaskParams::enableShadows` defaults false and
  `HdxTaskController::SetEnableShadows()` has no callers anywhere in pxr, so
  `HdxShadowTask` is dropped from the task list outright.
* The only `HdxShadowMatrixComputation` subclass in OpenUSD is the test-only
  one in `hdx/unitTestDelegate.cpp`. There is no shadow frustum to size, which
  is also why `inputs:shadow:distance` changes nothing.

Measured rather than only read: overriding `/World/Key` to
`inputs:shadow:enable = 1` and re-rendering both cameras moves 56 pixels of
1 228 800 on HeadCam (max delta 12) and 5 on TempleCam -- hair antialiasing
noise, no shadow. A reported hard rectangular clipping patch under that
attribute did not reproduce in either camera or with either attribute type; if
it returns, look at the tile publisher's extents or the dome background draw
rather than at Hydra shadows. Practical advice: leave `inputs:shadow:enable`
off, because toggling it costs a light resync and a groom recook and buys
nothing. Shadow maps are reachable only by an application that drives
`HdxTaskController` itself, installs its own `HdxShadowMatrixComputation` and
calls `SetEnableShadows(true)` -- not by usdview or usdrecord.

### So the shadow is published as geometry

`UsdGenGroomSceneIndex` adds one synthetic Mesh per emitting surface,
`<description>/__usdGenRender/scalpShadow` on the first (by path) of the
descriptions that grow from it: the haired part of the surface, lifted a
hair's breadth along its own smooth normal and carrying the same
`furTauP`/`furTauN` vertex primvars a strand carries, summed over every
description (see [One cap per surface](#one-cap-per-surface)). It binds
`<description>/__usdGenRender/material_scalpShadow`, the seventh shipped
glslfx (`UsdGenScalpShadow`, outside C5), which writes **black with alpha
`1 - T`** under the `translucent` tag -- so the pass multiplies whatever skin
material the user authored underneath instead of replacing it. Because the
shadow comes from the density volume and not from a light, it works under
usdview's default lighting, where a shadow map would have nothing to offer.
Beside it sits an opaque sample mask that carries the whole steps of the same
darkening ([Hair edges over shadowed skin](#hair-edges-over-shadowed-skin)).

`T` is the fraction of the incident light that survived the coat:

    T = SUM_i w_i * a_f(hairColour, roughness, theta_i) ^ HairCount(L_i)
        / SUM_i w_i,     w_i = radiance_i * attenuation_i * max(0, N . L_i)

with `a_f` the same 39-term fit of a host renderer's scattering LUT the strand material
uses (`#import`ed from `usdGenHairStrandsBsdf.glslfx`, not copied) and
`HairCount` the same normalised cosine-power-4 blend of the six stored depths.
Weighting each light by what it actually contributes to the skin is what lets
one scalar multiply stand in for a per-light shadow. The dome is five
cosine-weighted taps about the normal rather than the three axes a bare
`max(0, N . axis)` would select.

Two deliberate differences from the strand shader:

* **No self shift.** a host renderer's `max(0, HairCount - 1)` removes the *shading
  strand's* own crossing. The receiver here is skin, so every crossing between
  it and the light is real and the count goes in as it stands.
* **Hair-only depth.** The bake runs the six sweeps twice: once over hair
  alone, for the cap, and once with the opaque shell injected, for the strands.
  Folding the shell into the cap would darken the unlit side of the head twice,
  because the skin material's own `N . L` already does that.

### The parts that needed care

* **Where the depth is sampled.** One voxel out along the surface normal: far
  enough that the opaque shell, which sits two voxels *in*, cannot reach the
  gather, close enough that it is the light the roots actually see.
* **Tessellation.** One refinement level for the whole mesh, chosen from the
  90th-percentile edge against 1/128 of the surface's bounding diagonal (about
  a voxel on a head) and then bounded by a triangle ceiling. It depends on the
  surface alone, so every description over it samples the same lattice (see
  [One cap per surface](#one-cap-per-surface)). Per-face refinement would
  crack: neighbouring faces would split a shared edge differently and leave
  T-junctions. The head's 8800 scalp triangles refine 3x into 79 200, of which
  the cull keeps about 58 000.
* **Culling, and why the threshold is small.** A triangle with no hair over it
  is dropped, which is what keeps bare skin untouched, but transmittance falls
  steeply from zero crossings -- `a_f` is around 0.2 for a dark coat, so 0.05
  crossings is already 6% of the light -- and dropping a triangle at that level
  draws a hard polygonal edge across the skin, one source face wide. The
  threshold is 0.002 crossings, a quarter of one 8-bit level. It is also asked
  the right question: the depth toward any *single* axis is useless as a test,
  because from anywhere on a head the horizontal axes run through the hair mass
  on the far side, so the cull evaluates the same cosine-weighted hemisphere
  the shader will.
* **Z-fighting and the silhouette.** glslfx has no polygon offset, so the cap
  is lifted 0.08 of that lattice edge along the normal in the bake (about
  0.3 mm on a head; the sample mask's depth sits half that behind it) -- still an order of
  magnitude above the depth buffer's resolution at this scale. That lift also
  pushes the cap's silhouette a little outside the skin's, which painted a dark
  rim on the background; the shader fades the cap as the surface turns away
  (`smoothstep(0.05, 0.35, N . V)`), which costs nothing real because at
  grazing angles what the camera sees is hair, not scalp.
* **Draw order.** The strand default is the opaque tag, so hair writes depth in
  the opaque pass and the cap -- drawn later, translucent, depth-tested -- is
  hidden wherever a strand already covered the pixel and darkens only the gaps.
  That needs no sorting, but the translucent pass makes that decision once per
  pixel, not per sample, which is what the sample mask below is for.
* **Dirtying.** The cap is diffed data source by data source like a tile, never
  dirtied universally: a deforming groom rebakes it every frame, and a
  universal dirty there would cost a full mesh re-sync per frame.
  `testUsdGenRbfDeform` holds it to the same locator discipline as a tile.

### Limits

* **The darkening is a scalar.** The light that survives a blonde coat is warm,
  so the physical transmittance is a colour, but `over` blending carries one
  alpha and stock Storm has no multiply blend mode: `dst * T` per channel would
  need to read `dst`. The luminance of `T` is used, which keeps the brightness
  exact and loses only the hue shift -- nothing at all for dark hair, where `T`
  is near zero anyway.
* **It is an expectation, not a ray.** Under a *camera headlight* the light
  path and the view path coincide, so a gap the view ray found is a gap the
  light ray found too, and darkening it is double-counting. The voxel count
  cannot know that. It is right for any light that is not at the camera, which
  is the case the comparison scene renders.
* **Detail is bounded by the emitter's tessellation and the voxel size**, so
  the cap shows a soft density shadow rather than individual strand shadows.
* The cap carries no `primOrigin` and takes no selection overlay, so picking
  passes straight through it to whatever the user authored underneath. The
  mask draws nothing in a one-sample pass, so it is never picked either.
* `USDGEN_SCALP_SHADOW=0` publishes no cap at all; nothing else changes.
  `USDGEN_SCALP_SHADOW_MASK=0` keeps the cap but publishes no mask, which is
  the behaviour before the mask existed.

### One cap per surface

Each description bakes its own cap, but the scene index shows one per
emitting surface, summed over every visible description that grows from it.
A cap per description failed twice over on a real groom. The imported Autumn
head has ten descriptions over one head mesh, and on its face five to eight
caps overlapped -- past Storm's OIT resolve, which keeps eight fragments per
pixel and silently drops the rest (about 9 800 pixels of that frame hit the
limit). And the sample mask below needs the *total* darkening in one place:
masks from separate caps would union their samples instead of multiplying.

Summing is exact for the skin. The light that reaches it has crossed every
description's hair in turn, so per light the transmittance is
`a_f ^ (n_1 + n_2 + ...)`, and the per-light weighting then averages that. Stacking
separate caps instead multiplied the light-weighted *averages*, which
over-darkens wherever the lights disagree -- the summed cap leaves Autumn's
face a little lighter than the stack did, and that is the correction.

The bake makes the sum possible by sampling every cap on the same lattice:
the refinement, the lift and the solid side now come from the emitting
surface alone (an edge target of 1/128 of its bounding diagonal, under the
same triangle ceiling) rather than from the description's voxel grid. Each
cap records a `latticeKey` and, per point, its lattice slot, so caps that
dropped different triangles still line up. `UsdGenMergeScalpShadows`
(`furOcclusion.h`) keeps every triangle any part kept, adds the depths point
for point, and -- only when the descriptions' hair colours differ -- adds a
`furCapAlbedo` vertex primvar weighting each colour by how much of that hair
the point sees. The shown cap belongs to the first description by path, so it
does not move when another joins; a hidden or dormant description's hair is
not drawn, so its shadow is left out. The scene index caches the sum by its
inputs, so a snapshot in which no cap changed reuses it.

### Hair edges over shadowed skin

With the cap on, every strand that crossed shadowed skin -- the hairline, the
lashes, a lock hanging across the cheek -- grew a bright, skin-coloured
outline. Storm resolves the multisampled depth buffer to one sample per pixel
before its translucent (OIT) pass (`HdxTaskController` sets
`useAovMultiSample = false` for it, and the resolve is a `GL_NEAREST`
`glBlitFramebuffer`), so the cap's depth test reads one sample. Where a strand
covers part of a pixel and owns that sample, the cap is hidden from the whole
pixel, and the skin between the strand's samples keeps its full light while
the skin around it is shadowed. Measured on Autumn: the chance a partly covered
pixel lost the cap rose with the strand's coverage -- 18% under a fifth of the
pixel, 48% at a third, 74% at a half, every pixel above 60%.

Nothing in stock Storm can multiply per sample: the opaque pass can only
replace samples and the additive pass can only add. So the darkening is split:

* **`scalpShadowMask`** (`UsdGenScalpShadowMask`, opaque) is the same cap,
  its depth pushed back by half the lift, depth-tested per sample, writing
  black to `steps` of every `maskLevels` samples of the pixel, where
  `steps = floor(darkening * maskLevels)`.
* **`scalpShadow`** (translucent, as before) takes only what is left:
  `alpha = 1 - (1 - darkening) / (1 - steps / maskLevels)`, so on bare skin the
  two compose to the cap's darkening exactly.

The two halves must agree on `steps` at every pixel. A first version drew
the mask at half the lift, and the offset was enough to shift where each pixel
read the cap: about 900 skin pixels of the Autumn frame, along every step
contour, came out a step too dark or too light. So the mask shares the cap's
points exactly and only pushes its depth back in its shader (`gl_FragDepth`,
half the lift along the view ray -- the skin is at least a whole lift away
along it), and both compute the darkening at the skin point they read back
from the `usdGenCapLift` primvar. The mask
takes its black samples from samples 1..n-1 and leaves sample 0 free: that is
the sample the resolve keeps, so where the translucent half is hidden, sample
0 is hair and every skin sample in the pixel is one the mask can blacken.

`maskLevels` follows Storm's MSAA sample count (`HDX_MSAA_SAMPLE_COUNT`,
default 4) unless `USDGEN_SCALP_SHADOW_MASK` says otherwise. The mask draws
nothing in a pass with one sample per pixel (an id pass, a host without
MSAA); a host that renders without MSAA but leaves that setting alone should
set `USDGEN_SCALP_SHADOW_MASK=0`, or the translucent half will leave out the
mask's steps.

| sub-pixel black strips over the haired half (`check_scalp_shadow.py`, linear) | mean | above the shadow they sit in |
|---|---|---|
| ideal: strips over skin darkened like bare skin | 0.0028 | -- |
| translucent cap alone | 0.0462 | +0.0435 |
| with the mask, black samples rotated over all four | 0.0152 | +0.0124 (29%) |
| with the mask, sample 0 left free | **0.0049** | **+0.0021 (5%)** |

Leaving sample 1, 2 or 3 free instead left 41%, 29% and 37%: sample 0 is the
one the resolve keeps on this driver, and a driver that resolves another sample
loses only that refinement, not the mask. On bare skin the halves compose to
0.0030 against the cap's own 0.0030. On Autumn, pixels brighter than their
neighbourhood inside the shadowed face fell from 1 848 to 655 (most of what
remains is skin showing between strands, which is meant to be there), the
lashes render dark instead of copper, and not one pixel of hair over the
background changed.

**Limits.** Where the resolve keeps another sample, a hidden pixel misses at
most one step of the darkening, `1/maskLevels` of the skin's light;
`HDX_MSAA_SAMPLE_COUNT=8` halves that. The mask is opaque black in the pixels
it covers, so at a cap's grazing silhouette it fades with the translucent
half's `N.V` fade, which it inherits through the shared darkening.

### Verification

`testUsdGenFurOcclusion` builds a combed patch over half a quad and requires
the cap to exist only where it should: mean depth 1.33 over the haired half
against 0.000 over the bare half, triangles, unit normals pointing out of the
solid, a lift under a strand's length, vertex float3 depths, a digest that is
stable across a rebake and changes when the hair does, and COW sharing on the
reuse path. (Note that the vertical strands the rest of that file uses have
*zero* projected area against a ray straight up -- which is the first assertion
in the file -- so the cap test combs them over.)

`usdGenShaders/test/check_scalp_shadow.py` validates the Sdr def and then, on a
real GL context, renders a grey plane with the cap over it, dense hair on one
half and none on the other:

| | under dense hair | bare |
|---|---|---|
| `strength = 0` | 216.0 | 216.0 |
| `strength = 1` | **90.7** | **216.0** |
| `strength = 0.4` | 179.7 | 216.0 |
| `strength = 1`, pale coat | 117.7 | 216.0 |

The last row is the compile check: only a shader that really evaluates
`a_f(baseColor)` darkens less for a pale coat, and Storm falls back silently
when a glslfx fails to compile. The same script then checks the mask: that the
two halves compose exactly on bare skin, and that under sub-pixel black strips
they keep at least half the shadow the translucent cap alone loses (the table
above). `testUsdGenFurOcclusion` checks that caps over one surface share a
lattice and that the merge sums them point for point, keeps every triangle
once and mixes hair colours only when they differ; `testUsdGenScenePublication`
checks that two descriptions over one surface publish one cap and one mask, on
the first, carrying both descriptions' hair.

On `examples/head-hair-closeup.usda`, against the same frame without the cap,
the skin brightens nowhere and darkens where the hair is: on HeadCam the
120-200 band of the crop under the hairline moves -17.5 of 255 and the 60-120
band -10.7, while the background moves 0.00 and the hair itself -0.05. On
TempleCam the same bands move -27.6 and -12.1. `renders/ue-parity/ws3_*` are
the before/after set.

## Supersampling the viewport

`record_usd.ps1 -Supersample N` renders offline at N x and box-downsamples in
linear light, and that converged image is what every anti-aliasing number in
this file is judged against. The same thing can be done interactively, at a
price that is nowhere near N^2:

    .\bin\launch_usdview.ps1 -Supersample 2 examples\head-hair-closeup.usda

or `USDGEN_USDVIEW_SUPERSAMPLE=2` in the environment, or the **usdGen >
Viewport Supersampling** menu (Off / 2x / 4x) once usdview is up. It is off
unless asked for. No OpenUSD source is patched; the whole thing is
`plugin/usdGenTools/python/usdGenTools/supersample.py`, which patches
`StageView._paintGLWithRenderer` when the plugin container loads.

**Why it needs a framebuffer of its own.** Asking the engine for a bigger
render buffer is one line -- `StageView._paintGLWithRenderer` already calls
`SetRenderBufferSize` and `SetFraming` (`stageView.py:1715-1720`) -- but the
present step then draws the AOV at that size *into the window*. With a valid
framing the task controller sets

    dstRegion = (0, 0, renderBufferSize[0], renderBufferSize[1])

(`hdx/taskControllerSceneIndex.cpp:2636`, `hdx/taskController.cpp:2148`), and
`HgiInteropOpenGL::CompositeToInterop` does `glViewport(*dstRegion)` before its
fullscreen triangle (`hgiInterop/opengl.cpp:246`). Nothing in
`UsdImagingGLEngine` exposes that region, so an N x render buffer composites
N x into the widget and all but the lower-left 1/N^2 falls off the edge. The
smallest OpenUSD-side fix would be to let the application set the present
region (an `HdxPresentTaskParams::dstRegion` that the app owns, reached through
something like `UsdImagingGLEngine::SetPresentationRegion`, and Python-wrapped);
`SetEnablePresentation`, `SetPresentationOutput`, `GetAovTexture` and
`GetAovRenderBuffer` are all C++-only today (`usdImagingGL/wrapEngine.cpp`), so
a Python plugin cannot take the AOV out of the engine either.

What the application *can* choose is the framebuffer that is composited into:
the present task's `dstFramebuffer` is empty in usdview, and HgiInterop then
composites into whatever is bound (`hgiInterop/opengl.cpp:152`). So the plugin
binds its own N x FBO for the whole paint -- the render, the axis, the camera
guides, the mask and the reticles all land in it at N x, because everything
they use comes from `GetPhysicalWindowSize`, which the patch scales for the
duration of the paint -- and then resolves that texture down into the widget's
framebuffer with an exact N x N box filter. The HUD is held back and drawn
after the resolve, at window resolution, so its text stays crisp.

**The resolve is in linear light.** `HdxColorCorrectionTask` has already
applied the sRGB transfer function by the time the pixels reach the present
step, so averaging those code values is not averaging the light: it loses
energy exactly where the strands are. The resolve shader therefore decodes to
linear, averages, and re-encodes, which is what `bin/downsample_linear.py` does
offline. Measured on the TempleCam frame at 4x, averaging the code values
instead darkens the frame by 5.5% of mean linear luminance and moves individual
strand-edge pixels by up to 63 code values. `USDGEN_USDVIEW_SUPERSAMPLE_LINEAR=0`
selects that (wrong, but cheaper-looking) behaviour; `=1` forces the decode when
usdview's colour correction is off.

head-hair-closeup TempleCam, 1280x960, `HDX_MSAA_SAMPLE_COUNT=8`, RTX 4090,
12 timed frames each (wall time around `update()` + `glFinish()`), spike score
as defined in the sample-count section above:

| | 1x | 2x | 4x | offline 4x reference |
|---|---|---|---|---|
| frame time (min) | 19.1 ms | 29.5 ms | 63.4 ms | -- |
| cost over 1x | -- | 1.55x | 3.3x | -- |
| isolated-pixel spike score | 0.00522 | 0.00295 | 0.00165 | 0.00176 |
| neighbour-gradient energy | 0.0582 | 0.0384 | 0.0260 | 0.0289 |

Two things to read off that. The cost is well under N^2 (1.55x for 2x, 3.3x for
4x) because this frame is not purely fill-bound. And 4x in the viewport reaches
the offline reference: the same speckle metric the offline `-Supersample 4`
reference scores 0.00176 on, the live viewport scores 0.00165 on. 2x is the
interactive setting -- it removes 44% of the speckle for half a frame.

**Limits.**

- At 4x on a 1280x960 window the render buffer is 5120x3840, and Storm's OIT
  fragment pool (`8 * width * height` elements) then exceeds `HdSt`'s maximum
  buffer array size: `HdStVBOSimpleMemoryManager` posts *"Number of elements in
  the buffer array range (0x9600000) is larger than the maximum ... 0x5600000
  bytes of data will be skipped"* on the resize. The opaque strand variant this
  file recommends does not read that pool, and the frames come out correct, but
  the errors are real and a translucent-bound groom would lose fragments. 2x on
  the same window stays inside the limit.
- Memory goes as N^2 as well, and the MSAA colour target multiplies it by the
  sample count: 4x at 1280x960 with 8 samples is a little over a gigabyte of
  AOV.
- Picking is untouched: it goes through a narrowed frustum built from
  normalised window coordinates (`stageView.py computePickFrustum`) and
  `HdxPickTask`'s own buffers, not the render buffer. The test asserts the pick
  frustums and hits are identical at 1x and 2x.

`plugin/usdGenTools/testenv/testUsdGenToolsSupersample.py` is the test (a
testusdview script, like the expression editor's): it checks the resolved frame
is the window size and matches the 1x frame quadrant by quadrant -- which is
what fails if the present ever magnifies and crops again -- that switching back
to 1x restores the stock frame, that the HUD draws over the resolved image
without disturbing it, that the camera mask still crops, and the picking
invariant above. `plugin/usdGenTools/testenv/shotUsdGenToolsSupersample.py` is
the photographer that produced the table.

### Turning it on and off at runtime

The **usdGen > Viewport Supersampling** submenu is checkable and mutually
exclusive (Off / 2x / 4x, the factor in force shown checked), and a change
applies to the next frame with no restart. The menu callback deliberately makes
**no GL call at all** -- it sets the factor, updates the check marks and asks
for a repaint; every GL call, including allocating the N x target for the new
factor, happens inside the paint, where usdview's context is current. Nothing
is persisted across sessions: the plugin has no settings mechanism of its own
and usdview's state file is not ours to add keys to, so `USDGEN_USDVIEW_SUPERSAMPLE`
(or `launch_usdview.ps1 -Supersample N`) is how you start somewhere other than
Off.

One trap, worth writing down because it cost a user-visible bug: **under
QOpenGLWidget the widget's framebuffer is not 0.** An early version of the
resolve ended its target allocation with `glBindFramebuffer(GL_FRAMEBUFFER, 0)`
and then read the "window" framebuffer back with
`glGetIntegerv(GL_FRAMEBUFFER_BINDING)` -- so on exactly the frames that
reallocate (every factor change) it captured 0, and that frame's resolve and
HUD went to the window-system framebuffer instead of the widget's. The frame
was lost, GL was left with an error pending, and PyOpenGL blamed it on the
first checked call of the *next* frame:
`GLError(err = 1280, baseOperation = glGetIntegerv, pyArgs = (GL_FRAMEBUFFER_BINDING, ...))`
-- one error per toggle, after which moving the camera "fixed" it, because the
next frame took the early-out in the resize path and bound the right target.
The fix is to ask Qt (`StageView.defaultFramebufferObject()`) before any GL
call of ours, and never to bind 0. The test toggles Off -> 2x -> 4x -> Off -> 2x
through the menu commands and fails if a toggle logs a rendering error or if the
frame straight after it differs from the one after that.

Picking is unchanged by any of this, but note what the assertion can and cannot
say: usdGen's synthetic tiles are not pickable in usdview at all (see **Picking
and the ID pass** above), so on a groom-only scene every pick returns nothing at
every factor. What the test checks is that the pick frustums are bit-identical
and the answer does not move with the factor.
