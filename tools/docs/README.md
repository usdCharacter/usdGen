<!-- Copyright (c) 2026 Nick Burkard -->
<!-- SPDX-License-Identifier: MIT -->

# Visual manual build guide

The browser manual is generated from source data in `docs/reference/`. Build it
from the repository root with Python 3:

```sh
python tools/build_manual.py
python tests/checks/check_manual.py
```

When only one operator's media is ready, use
`python tools/build_manual.py --op UsdGenWind` to update just its page.
Repeat `--op` for more pages; omit it for the complete manual build.

Open [the manual](../../docs/site/index.html) in a browser. The build writes a page for each
operator under `docs/site/operators/`, plus the getting started, character
groom, and gallery pages. It requires no Python packages beyond the standard
library. The checker compares the operator inventory with
`libs/hosts/usdGenUsd/schema/schema.usda` and `libs/usdGen/usdGen/opRegistry.cpp`, checks
operator parameter and source references, and checks the generated pages and
local media paths. It does not evaluate a groom or judge image quality.

The usdview operator walkthrough uses the localized Noodles editor. In an
eligible OpenUSD build, `USDGEN_BUILD_USDNOODLES` defaults ON when the
`python`, `garch`, and `glf` CMake targets are present and the build is not
cross-compiling; otherwise enable it explicitly after those dependencies are
available. Native noodles is found through `NOODLES_ROOT` (default `$USD`) or
fetched at the pinned revision. The build stages the Python module under
`build/python` and its plugin resource under `build/usd/usdNoodles/resources`;
`bin/launch_usdview.ps1` adds that resource to `PXR_PLUGINPATH_NAME`. In
usdview, open **Window > Noodles Editor** (`N`); the separate **Window > Layer
Editor** uses `L`. The plugin is optional and does not require usdRig.
In the Noodles editor, select an `Ops` scope before pressing `Tab` to create a
schema-backed operator. To inspect an existing node, select its prim in the
usdview Prim Tree and press `A` in Noodles; the graph opens empty by default.
Inline value editing covers scalar and small-vector fields. Arrays and
relationships do not have an inline value editor, and connected values are
read-only. WidthBlend has a runtime kernel but no typed creation entry;
Instance has schema controls but no current kernel.

[operators.json](../../docs/reference/operators.json) contains the human-readable operator content.
[media.json](../../docs/reference/media.json) records which local image or animation belongs to
each operator, its caption, renderer, scene, and provenance. Curated media lives
in `docs/site/media/`; raw capture frames belong in ignored `renders/docs/`.
Keep the media record honest when a scene cannot be rendered. An illustration
or a source image must not be described as a live usdGen render.

The `Part`, `WidthBlend`, and `Freeze` diagnostic renders come from a separate
native fixture exporter. Build `usdGenUtilityFixtures` with the standalone
[`CMakeLists.txt`](CMakeLists.txt) against an existing usdGen build and OpenUSD
installation, then run it once per mode:

```sh
renders/docs/baker-build/usdGenUtilityFixtures width-blend renders/docs/width-blend-native.usda
renders/docs/baker-build/usdGenUtilityFixtures part renders/docs/part-native.usda
renders/docs/baker-build/usdGenUtilityFixtures freeze renders/docs/freeze-native.usda
```

On Windows with the Visual Studio generator, invoke
`renders/docs/baker-build/Release/usdGenUtilityFixtures.exe` instead.

[`utility_fixtures.cpp`](utility_fixtures.cpp) cooks each graph with the engine
compiler and scheduler. The `WidthBlend` stage carries its two ordered input
groups and the actual weight-0.5 output. The `Part` stage colors strands from
the cooked uniform `partId` plane and retains that plane as a USD primvar; Part
itself leaves positions unchanged. The `Freeze` stage runs the same graph twice,
edits the upstream buffer between cooks, and exports the updated live input
beside the unchanged held output. These diagnostic stages contain ordinary USD
curves; a camera and lighting stage can reference them for presentation.
After running all three fixture modes, `python tools/render_utility_fixtures.py`
copies the cooked layers into `examples/docs/fixtures/`, authors their studio
layers, and renders the corresponding primary manual images through MoonRay.

To render the sample fur patch through the installed OpenUSD and usdGen runtime,
run:

```sh
python tools/render_operator_examples.py --sample
```

This writes a USD stage and PNG into ignored `renders/docs/operators/`. It
requires a working OpenUSD Python and Storm environment; the manual build and
static checker do not. Review captures before committing them to the manual.

To regenerate the operator scenes and their live captures, run
`python tools/render_operator_examples.py --all --width 1536`, then
`python tools/render_operator_diagrams.py` for source diagrams, followed by the
manual build and checker above. `--op UsdGenClump` limits the render to one operator;
`--scenes-only` authors scenes without rendering. Operators without a typed
USD scene may use an explicitly labeled source diagram in the media registry.
Collide and Wind are excluded from this generic generator, including explicit
`--op` requests: their dedicated comparison generators below own their practice
scenes and media.

The optional looping parameter sweeps use seven live scene-index captures per
operator. With `ffmpeg` on `PATH`, run
`python tools/render_operator_animations.py --all --width 960` to regenerate
the GIFs and their per-clip provenance in `media.json`. Individual clips can be
selected with `--op UsdGenClump`. Check the resulting GIFs visually before
rebuilding the manual.
Wind is excluded from this generic sweep so its native billowing animation stays intact.

For offline beauty captures, first build the main project, then configure the
standalone baker against that build and the same OpenUSD installation:

```sh
cmake -S tools/docs -B renders/docs/baker-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DUSD_INSTALL_DIR="$USD" \
  -DUSDGEN_BUILD_DIR="$PWD/build"
cmake --build renders/docs/baker-build --target usdGenBakeGroom usdGenCollideFarOracle usdGenUtilityFixtures
```

On Windows, a Visual Studio 2022 generator works from a regular PowerShell:

```powershell
cmake -S tools/docs -B renders/docs/baker-build -G "Visual Studio 17 2022" -A x64 `
  -DUSD_INSTALL_DIR="$env:USD" -DUSDGEN_BUILD_DIR="$((Resolve-Path build).Path)"
cmake --build renders/docs/baker-build --config Release --target usdGenBakeGroom usdGenCollideFarOracle usdGenUtilityFixtures
```

The baker links the existing `usdGenTestUtilsHd` support library from the main
build. That target is defined even with `USDGEN_BUILD_TESTS=OFF`; the main build
must have compiled it. For the Windows MoonRay debug delegate workflow, install
Pillow in the Python environment and run
`python tools/docs/render_pathtraced.py --batch-operators --width 1200`.
This path requires the delegate to be installed and discoverable by
`bin/launch_usdview.ps1 -PrintEnv`. The baker itself does not need MoonRay.
After the standard batch, render the two existing guide examples with their
special studio lighting. Deform explicitly bakes time 24; the native bake
compared its rest and posed curve geometry to verify that the result moved:

```sh
python tools/docs/render_pathtraced.py examples/rbf-guides-plane.usda --description /World/Groom/Hair --time 24 --camera /World/Cam --studio-special --output docs/site/media/deform.png --width 1200
python tools/docs/render_pathtraced.py examples/guide-interpolate-plane.usda --description /World/Groom/Hair --camera /World/Cam --studio-special --output docs/site/media/guide-interpolate.png --width 1200
```

For the native Collide and Wind demonstrations, run:

```sh
python -S tools/render_collide_diagnostic.py --animation --batch
python tools/render_wind_variants.py
```

Each Wind practice scene has density 6500,
100 actual cooked frames (0–99 at 24 fps), and the same elevated camera, seeds,
lighting and Wind controls. Native Clump's enabled state is the only scene
difference. The manual displays their labeled GIFs in separate full-width rows.
Shared motion follows native clump rest anchors and effective cohesion;
no expression or renderer substitutes for the operator effect.

These dedicated generators bake each state of the checked-in practice scenes
through the actual usdGen compiler and scheduler before MoonRay renders it.
Collide shows a Catmull-Clark sphere starting clear of dense clumped fur,
descending to its contact depth by frame 27, then sliding in Z through frame
99. Native collision uses the sphere's authored subdivision limit and the
polygon boundary of an invisible, closed scalp volume. The baker logs explicit
polygon-fallback warnings for `/World/Patch` and `/World/ScalpVolume`; a warning
for `/World/Shield` fails the media gate. `--animation --batch` validates 100 actual
cooks and renders the sequence with one recorder; it also checks sampled frames
against independent same-time renders. All MoonRay captures request
`--complexity veryhigh`, so the visible sphere is subdivided too. The Wind pair shows native low- and
high-frequency billowing from 100 real cooked frames per variant, 0–99.
No expression substitutes for either effect. Wind's labels and direction arrow
are presentation aids; they do not change the cooked strands. The generic
MoonRay operator batch excludes these two demonstrations and the Part diagnostic.
Both generators require OpenUSD Python, the docs baker, Pillow and
the MoonRay delegate. Review their printed geometric measurements and captures
before updating the Collide/Wind records in `media.json` and rebuilding the
manual.
For the Collide geometry gate alone, use
`python -S tools/render_collide_diagnostic.py --bake-only --animation`.
It checks every integer frame 0–99. Its measurements concern CVs and their
sampled control-polygon segments against an independent OpenSubdiv Far
tessellation of the Shield limit, plus exact polygon half-space checks against
the closed scalp volume. The Far proof reports its 64-to-128 sample convergence
error and distinguishes Shield contact from the coarse cage. It does not certify
the exact mathematical limit, every cubic B-spline sample, or the rendered tube
surface. The native Collide unit tests provide the tighter Bfr patch checks.

After publishing both animations, run the separate Pillow regression gate:

```sh
python tests/checks/check_operator_gifs.py
```

It requires exactly 100 encoded GIF frames, 100 distinct decoded frame images
and authored scene timelines 0–99 for both operators. Image uniqueness includes
any labels; native geometry uniqueness comes from the bake validation reports.
Use `--op UsdGenWind` or `--op UsdGenCollide` to check one completed animation.
When Wind's media record includes variants, its gate checks each variant's
linked scene and GIF independently.

After changing either practice scene, refresh its real Noodles screenshot:

```powershell
tools/docs/capture_all_operator_ui.ps1 -Operators UsdGenCollide,UsdGenWind
```

This runs the established usdview TestScript capture and records the actual
operator node and its authored values. For the final Collide held-frame and
cut-control captures, first accept the native runtime and the canonical
`collide.usda` / matching `collide-slide.usda` scenes. Then run the normal
synchronous TestScript launch, pinning the exact input and its content hash:

```powershell
$env:USDGEN_UI_EXPECT_SCENE = (Resolve-Path examples/docs/operators/collide.usda).Path
$env:USDGEN_UI_EXPECT_SCENE_SHA256 = (Get-FileHash $env:USDGEN_UI_EXPECT_SCENE -Algorithm SHA256).Hash
$env:USDGEN_UI_CAPTURE_DIR = "renders/docs/collide-ui-review"
bin/launch_usdview.ps1 -TestScript tools/docs/capture_collide_slide_ui.py $env:USDGEN_UI_EXPECT_SCENE
```

The helper rejects `--allow-async`, checks the flexible `cutThenCollide`
preset (threshold 0.01, blend 0.02, 128 iterations, push 1 and offset 0), and
waits for each synchronous cook and converged Storm presentation. It captures
held frames 0/27/99, both collider targets, Catmull-Clark at Very High display
complexity, translate authoring and the Noodles cut controls. Set
`USDGEN_UI_CUT_CONTROLS_ONLY=1` for only the last capture. Clear that variable
for a complete run. Inspect these review captures before copying accepted
images into the manual and promoting `ui.json`'s `pendingCaptures` metadata;
a pending record is not published evidence.

The manual's practice steps use the
visual usdview and Noodles controls; the generation commands above are for
maintaining the documentation artifacts.

The character workflow uses the ALab stoat sample. With an OpenUSD 26.08
installation available, prepare it from the repository root:

```sh
python tools/prepare_alab.py prepare --usdcat "$USD/bin/usdcat"
```

On Windows PowerShell, use
`python tools/prepare_alab.py prepare --usdcat "$env:USD\bin\usdcat.exe"`.

The tool selectively fetches the ALab structure and Stoat techvar members,
then writes `out/alab/stoat01.usdc`, texture and preview-card sidecars,
`out/alab/provenance.json`, a copy of the ALab license, and
`out/alab/stoat01.usdz`. The USDC layer consolidates the rig, geometry, and
material scene description while its textures remain external; the USDZ packs
those assets inside. The tool requires `usdzip` beside `usdcat` for packaging.
These downloaded files are ignored by Git and retain ALab's
ASWF Digital Assets License. See [the ALab example guide](../../examples/alab/README.md) for source versions,
the upstream texture repair, and rig caveats.

After preparing the asset, regenerate the guide-driven coat and its bare and
guide-only comparison stages:

```sh
python -S tools/make_stoat_groom.py
```

The command writes `examples/alab/stoat-groom.usda`, `stoat-bare.usda`, and
`stoat-guides.usda`. They reference the local consolidated ALab USDC, so that
asset must stay in `out/alab/` when opening the example. The scene descriptions
are small project-authored recipes; the ALab character and textures remain
separately licensed local data. The example guides are static; they are not
skinned to the character's skeleton. For the coat study, the groom layer hides
the outfit and backpack and substitutes a fitted scalp for the native tail
tuft. The consolidated ALab asset is unchanged.
