# Third-party notices

usdGen is MIT, Copyright (c) 2026 Nick Burkard (`LICENSE`). The components
below keep their own licenses. This file is the index; `NOTICE` is the
short form installed to `share/usdGen/NOTICE`.

## Build dependencies (not in this tree)

| Component | License | Where the terms live |
| --- | --- | --- |
| OpenUSD | Tomorrow Open Source Technology License 1.0 | https://openusd.org/license |
| OpenSubdiv | Apache-2.0 with a modified trademarks section | upstream `LICENSE.txt` |
| MaterialX | Apache-2.0 | upstream license |
| oneTBB | Apache-2.0 | upstream license |

No OpenUSD source is vendored here, so there is no Pixar/TOST header to
relicense. Link against an OpenUSD build you supply (`USD` or
`-DUSD_INSTALL_DIR`).

## Vendored libraries

| Path | Upstream | License |
| --- | --- | --- |
| `thirdparty/seexpr`, `thirdparty/seexprFrontend` | SeExpr | Apache-2.0 with Section 6 (Trademarks) replaced. Upstream copyright notices stay. Terms are in each header and in `thirdparty/seexprFrontend/LICENSE`. `thirdparty/seexpr` has no separate `LICENSE` file; the grant is the header block, which points at http://www.apache.org/licenses/LICENSE-2.0 |
| `thirdparty/ptex` | Ptex | BSD-3-Clause with an upstream non-endorsement clause. `thirdparty/ptex/LICENSE` |
| `thirdparty/nanoflann/nanoflann.hpp` | nanoflann | BSD-2-Clause, in the header |
| `thirdparty/zlib` | zlib | zlib License. `thirdparty/zlib/LICENSE` |

## RTXCR hair evaluator

`usdGenShaders/resources/shaders/usdGenHairStrandsRtxcr.glslfx` contains a
GLSL adaptation of `HairFarFieldBCSDF.hlsli` and selected functions from
`HairMaterial.hlsli`, `utils/RtxcrMath.hlsli`, and
`utils/RtxcrHairBsdfHelper.hlsli`, and `utils/RtxcrBsdf.hlsli` in
[NVIDIA-RTX/RTXCR-Material-Library](https://github.com/NVIDIA-RTX/RTXCR-Material-Library/tree/eb184f9637f81a58a620d0e3cb4356acbc45cbe6/shaders/include/rtxcr),
pinned to revision `eb184f9637f81a58a620d0e3cb4356acbc45cbe6`.

Those source files carry explicit MIT permission notices, retained in the
adaptation, with Copyright (c) 2024-2026, NVIDIA CORPORATION (2024-2025 for
the helper files). This attribution concerns those shader files, not the
whole SDK: the upstream repository also has an NVIDIA RTX SDK `License.txt`.
This integration does not vendor that SDK, path tracer, or denoiser.

Changes include GLSL syntax, removal of sampling/PDF code, optional independent
R roughness and R/TT gains, and a separate Storm adapter for lighting,
pixel-footprint filtering, existing hair-count visibility and ambient fill.
The adapter and project changes are Copyright (c) 2026 Nick Burkard, MIT.

## `usdNoodles/`

The directory is a copy of [facebookexperimental/noodles](https://github.com/facebookexperimental/noodles)
plus a few usdGen files.

Upstream files keep `Copyright (c) Meta Platforms, Inc. and affiliates`
and `usdNoodles/LICENSE.txt` (MIT). That includes `core/`, `render/`,
`spatial/`, `undo/`, `assets/`, `tests/` except the files named below,
and the upstream docs (`README.md`, `BUILDING.md`, `CODE_OF_CONDUCT.md`,
`CONTRIBUTING.md`, `SECURITY.md`, `THIRD_PARTY_LICENSES.txt`).

Poppins atlases under `usdNoodles/assets/fonts/Poppins-*` are SIL OFL 1.1.
The Baskerville atlas is described as public domain in
`usdNoodles/THIRD_PARTY_LICENSES.txt`. GLEW, stb_image, and RapidJSON are
FetchContent dependencies, not files in this repository; their notices are
in that same third-party license file.

These paths are **not** in the public noodles tree. They are usdGen code
under the root MIT license (`SPDX-License-Identifier: MIT`), not Meta's:

- `usdNoodles/.github/workflows/ci.yml`
- `usdNoodles/cmake/noodlesConfig.cmake.in`
- `usdNoodles/core/AttributeConnectionDrag.cpp`
- `usdNoodles/core/AttributeConnectionDrag.h`
- `usdNoodles/render/noodles_gl.h`
- `usdNoodles/tests/AttributeConnectionDragTest.cpp`
- `usdNoodles/tests/ExpressionConnectionsTest.cpp`
- `usdNoodles/tests/package-consumer/CMakeLists.txt`
- `usdNoodles/tests/package-consumer/main.cpp`
- `usdNoodles/tests/package/CMakeLists.txt`
- `usdNoodles/tests/package/main.cpp`
- `usdNoodles/usd/ExpressionConnections.cpp`
- `usdNoodles/usd/ExpressionConnections.h`
- `usdNoodles/usd/api.h`

## `plugin/usdNoodles/` usdview editor

This localized usdview Noodles plugin was ported from
[usdRig commit `8dafce7d6d6299e6f79f9c4fb22c17145d96359e`](https://github.com/n-burk/usdRig/commit/8dafce7d6d6299e6f79f9c4fb22c17145d96359e),
which adapts [OpenUSD PR #4156](https://github.com/PixarAnimationStudios/OpenUSD/pull/4156).
The Python, binding, and editor code retains Meta/OpenUSD copyright headers
and the Tomorrow Open Source Technology License 1.0 in
[`plugin/usdNoodles/LICENSE.txt`](plugin/usdNoodles/LICENSE.txt). The native
noodles code and copied GLSL retain Meta's MIT license in
[`plugin/usdNoodles/NOODLES_LICENSE.txt`](plugin/usdNoodles/NOODLES_LICENSE.txt).
The native source is pinned to
[`facebookexperimental/noodles` commit `ff5d473f10e8c37ceaf0da11ea7cb80805bc8314`](https://github.com/facebookexperimental/noodles/commit/ff5d473f10e8c37ceaf0da11ea7cb80805bc8314).

The imported Poppins font atlases remain under SIL Open Font License 1.1;
their terms and the other dependency notices are in
[`plugin/usdNoodles/THIRD_PARTY_LICENSES.txt`](plugin/usdNoodles/THIRD_PARTY_LICENSES.txt).
The local port removes usdRig-only TouchPose references and adapts build,
installation, and launcher paths for usdGen. These changes do not alter the
licenses or copyright notices on the upstream files.

## Optional host renderer

Scripts and comments can talk to a MoonRay Hydra plugin (`hdMoonray`,
`moonray:sceneVariable`). That renderer is not vendored. The plugin names
are the host API, so they are unchanged.

## ALab-derived manual images

`docs/site/media/stoat-*.png` and `docs/site/media/ui/alab-*.png` show the ALab
stoat from the ASWF Digital Production Example Library. These images are derived from assets under the
ASWF Digital Assets License v1.1, not usdGen's MIT license. The full license,
required Netflix Animation Studios ALab copyright notice, source versions,
and modification summary are in
[`docs/site/media/README.md`](docs/site/media/README.md) and
[`docs/site/media/ALab-LICENSE.md`](docs/site/media/ALab-LICENSE.md).
