# usdGen

usdGen generates and grooms hair and fur inside an OpenUSD pipeline. A groom is ordinary USD: a description, an operator stack, guides, maps, and looks. A Hydra scene index cooks that graph and publishes curves for Storm and other render delegates. usdview plugins author the groom and edit expressions on the stage.

The intended result is a set of plugins and tools a studio pipeline can build, install, and load next to OpenUSD. Artists and TDs use the usdview tools. Engineers use the libraries, the schema, and the tests.

## What ships in this repository

- **Engine** (`libs/usdGen`, `libs/usdGenMath`). A stage-free operator graph. Example scenes exercise scatter, grow, noise, length, width, clumping, guide interpolation, curve import, and deformation.
- **Hydra** (`libs/usdGenImaging`, `plugin/usdGenImaging`). A groom scene index that publishes curve tiles into Hydra, including Storm shading and optional instanced fur.
- **Schema** (`libs/usdGenSchema`, `plugin/usdGenSchema`). Codeless `usdGen:` types. The checked-in resources are generated from `libs/usdGenSchema/schema.usda`.
- **Pomade** (`libs/usdGenPomade`, `plugin/usdGenPomadeTools`). A usdview tool for hierarchical groom authoring: tubes, fills, guides, and a commit back to USD. See [docs/pomade-tool.md](docs/pomade-tool.md).
- **Expression editor** (`plugin/usdGenTools`). A usdview dock for SeExpr on `usdGen:expr:source`.
- **Shaders** (`usdGenShaders`). Storm materials for strand shading and self-shadowing. See [docs/storm-fur.md](docs/storm-fur.md).
- **Examples** in [examples/](examples/README.md).

Pomade is this repository's authoring tool. It is not a separate product name from an outside DCC.

## How it sits next to OpenUSD and usdRig

usdGen is built against **OpenUSD 26.08** and uses Hydra 2 scene indexes. OpenUSD is a build dependency. It is not vendored here. Its license is the Tomorrow Open Source Technology License; see [NOTICE](NOTICE).

**usdRig** is an optional sibling. The groom index is meant to run after deformation (UsdSkel or RigExec) in the Hydra chain. Configure `-DUSDGEN_WITH_RIGEXEC=ON` and `RIGEXEC_INSTALL_DIR` only when you want to link `rigExecMath`. A normal usdGen build does not need usdRig.

This repository does not integrate usdOrchestrator.

## Quick start

You need CMake 3.26 or newer, a C++17 compiler, and an OpenUSD 26.08 install. Ninja is recommended. CUDA 12.8 or newer and Vulkan 1.2 are optional execution backends. A cook that asks for a backend omitted from the build is refused at runtime.

From the repository root:

```sh
export USD=/path/to/OpenUSD
export VENV=/path/to/venv   # optional; Python that can import pxr
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSD_INSTALL_DIR="$USD"
ninja -C build
```

`bin/build_usdgen.sh` wraps configure and build. On Windows, `bin/build_usdgen.ps1` does the same job. The OpenUSD prefix must match the compiler you are using.

The Storm hair material has two strand models. The default, `-DUSDGEN_HAIR_SHADING_MODEL=karis`, is the Karis R/TT/TRT model. `-DUSDGEN_HAIR_SHADING_MODEL=marschner` compiles the absorption-based Marschner model that imported real-time grooms were authored against, with its own multiple scattering and hair ambient occlusion. Switching takes a re-configure and recompiles one source file: the choice is a generated shader file plus one compile definition on the fur-bake call site. See [docs/storm-fur.md](docs/storm-fur.md#strand-models).

Load the build tree and run a flatten smoke test:

```sh
source bin/_env.sh
usdcat --flatten tests/scenes/scene_empty_groom.usda
```

`bin/_env.sh` sets `USD`, `GEN`, `PY`, and `PXR_PLUGINPATH_NAME` from the checkout location. Override `USD` or `VENV` before sourcing it when the defaults (a sibling `OpenUSD_26_08` directory, and `python3` on `PATH`) are wrong.

Open an example in usdview:

```sh
source bin/_env.sh
"$PY" "$USD/bin/usdview" examples/scatter-grow-plane.usda
```

On Windows, `bin/launch_usdview.ps1` is the equivalent launcher.

Install:

```sh
cmake --install build --prefix /path/to/usdGen-install
```

That writes the libraries, headers, and plugin resources under the prefix.

Turn the groom scene index off for one process with `USDGEN_ENABLE=0`.

## Vulkan execution and CUDA parity

Build the Vulkan Session runtime with a Vulkan SDK, `glslangValidator` (or
`glslc`), and `spirv-val` available:

```sh
cmake -S . -B build-vulkan -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DUSD_INSTALL_DIR="$USD" \
    -DUSDGEN_ENABLE_VULKAN_RUNTIME=ON -DUSDGEN_BUILD_VULKAN_TESTS=ON \
    -DUSDGEN_BUILD_TESTS=ON
cmake --build build-vulkan
export GENBUILD="$PWD/build-vulkan"
source bin/_env.sh
ctest --test-dir "$GENBUILD" -L '^vulkan$' --output-on-failure
```

On Windows, install the [Vulkan SDK](https://vulkan.lunarg.com/doc/view/latest/windows/getting_started.html)
and open a new PowerShell so its `VULKAN_SDK` environment variable is available:

```powershell
.\bin\build_usdgen.ps1 -Build build-vulkan -UsdInstallDir $env:USD -Vulkan -NoCuda -Test
```

`-Vulkan` enables the runtime even in a previously configured build. With
`-Test`, it also enables and runs the Vulkan tests alongside T0/T1. Omit
`-NoCuda` to keep the script's CUDA detection, or use `-Cuda` to require both
backends. `-NoVulkan` disables the Vulkan runtime and its tests. With neither
Vulkan switch, CMake retains its cached choice or detects the SDK on the first
configure.

The runtime finds its shader bundle relative to the library, including after
installation under `share/usdGen/vulkan`. `USDGEN_VULKAN_SHADER_DIR` overrides
that location. Native Vulkan consumers link the exported
`usdGen::usdGenVulkanNative` target and use `GetVulkanGenerationAccess` to hold
the generation and its queue owner while consuming device buffers.

The default Vulkan memory pool follows CUDA's policy: use the initial available
device-local memory and reserve the greater of 1 GiB or 20% as headroom.
`VK_EXT_memory_budget`, when available, supplies the driver budget minus usage;
otherwise the initial estimate uses heap capacity. External allocations can
still race that estimate, so native allocation failures remain authoritative.
Sessions share the first pool for each device UUID. Applications can establish
an explicit pool configuration through `DeviceFactory::ResolveAsync` before
creating a default provider; later automatic snapshots reuse that configuration.

The software Vulkan CI job checks Vulkan execution and validation diagnostics.
It sets `USDGEN_VULKAN_ALLOW_SOFTWARE=1` to permit a CPU Vulkan driver; normal
device selection prefers hardware GPUs and excludes software drivers.
To compare actual CUDA and Vulkan execution on a machine with both backends,
also configure `-DUSDGEN_ENABLE_CUDA=ON`, build, and run:

```sh
bin/check_cuda_vulkan_parity.sh build-vulkan
```

This hardware gate requires `vulkaninfo` and the Khronos validation layer,
enables synchronization validation, and fails if a selected comparison skips.
Tests read back results for comparison; production Session geometry stays on
the device. GPU Sessions require a device-aware consumer. Stock Storm's scene
index path currently lacks that consumer for both CUDA and Vulkan.

## More documentation

| | |
|---|---|
| [docs/README.md](docs/README.md) | Index of user-facing docs and historical notes |
| [docs/pomade-tool.md](docs/pomade-tool.md) | Pomade workspace, modes, and hotkeys |
| [docs/storm-fur.md](docs/storm-fur.md) | Storm hair shading and self-shadowing |
| [docs/moonray-fur.md](docs/moonray-fur.md) | Instanced fur through the MoonRay Hydra delegate |
| [examples/README.md](examples/README.md) | Scenes and how to view them |
| [plan/README.md](plan/README.md) | Design record. It is not the build guide |

Regenerating schema resources, test tiers, and coding conventions are in [AGENTS.md](AGENTS.md).

## License

usdGen is [MIT](LICENSE), Copyright (c) 2026 Nick Burkard.

Third-party code keeps its own license. OpenUSD is a build dependency under the [Tomorrow Open Source Technology License](https://openusd.org/license). SeExpr, Ptex, nanoflann, and zlib are vendored under `thirdparty/` with their upstream notices. `usdNoodles/` includes the noodles project (MIT, Meta Platforms), SIL OFL Poppins atlases, and a Baskerville atlas. Files in `usdNoodles/` that are not in the public noodles tree are project MIT. See [NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md).
