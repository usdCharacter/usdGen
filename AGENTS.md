# AGENTS.md

Instructions for coding agents working in this repository. Humans should start at [README.md](README.md).

## Purpose

usdGen is an OpenUSD 26.08 plugin that generates and grooms hair and fur and publishes the result through a Hydra scene index. Pomade (`usdGenPomade` / `usdGenPomadeTools`) is the in-tree hierarchical authoring tool. usdRig is an optional link (`USDGEN_WITH_RIGEXEC`). This tree does not reference usdOrchestrator; do not add that integration unless asked.

## Layout

| Path | What it is |
|---|---|
| `libs/core/usdGen`, `libs/base/usdGenMath` | Stage-free groom engine. Links no pxr library. `usdGen/api/` (with the extension SDK in `api/ext/`) is the public surface; `registries/`, `compile/` and `exec/` are private |
| `libs/core/primitives/usdGen{Surface,Expr,Curves}`, `libs/core/backends/usdGen{Cuda,Vulkan}` | Primitives and execution backends, loaded by `usdGen` as extensions (`usdGenExtensions.txt`) |
| `libs/hosts/usdGenHydra/primitives/curves` | Hydra adapter for curves (`usdGenHydraCurves`) and the hair shaders |
| `libs/hosts/usdGenHydra/usdGenImaging`, `plugin/usdGenImaging` | Hydra scene index, tile publisher, brush and expression C ABI |
| `libs/hosts/usdGenUsd/usdGenUsd` | USD host: SdfPath to node keys, `Lower`, the stage builder (`walk/`) and image maps (`resolve/`) |
| `libs/hosts/usdGenUsd/schema` | Hand-written `schema.usda` and the checked-in generated resources (`resources/`) |
| `libs/usdGenPomade`, `plugin/usdGenPomade`, `plugin/usdGenPomadeTools` | Pomade model, scene index, and usdview tool |
| `plugin/usdGenTools` | SeExpr editor usdview plugin |
| `usdGenShaders` | Storm glslfx materials |
| `examples/` | Scenes. See `examples/README.md` |
| `tests/` | C++ ctest sources and `tests/checks/` Python checks |
| `bin/` | `build_usdgen.sh`, `build_usdgen.ps1`, `_env.sh`, `gen_schema.sh`, launchers |
| `docs/` | User docs (`pomade-tool.md`, `storm-fur.md`, `moonray-fur.md`) and historical notes (`review/`, `prework/`, `m1/`, `freezes/`) |
| `plan/` | Design record. Read before a large behavior change. Not a status dashboard |
| `tools/` | Scene generators and capture helpers |
| `thirdparty/` | SeExpr, Ptex, nanoflann, zlib. Do not relicense |
| `usdNoodles/` | Vendored noodles plus a few project files. Upstream files stay under their MIT header |

## Build, test, schema

```sh
export USD=/path/to/OpenUSD
export VENV=/path/to/venv   # optional
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSD_INSTALL_DIR="$USD"
ninja -C build
source bin/_env.sh
ctest --test-dir build --output-on-failure
```

Useful cache variables: `USDGEN_BUILD_TESTS` (default ON on Unix), `USDGEN_ENABLE_CUDA`, `USDGEN_WITH_RIGEXEC` (default OFF), `USDGEN_FP_CONTRACT`.

There is no pytest suite. C++ tests are ctest. Python checks are plain scripts (`tests/checks/check_pomade_docs.py`, `plugin/usdGenPomadeTools/testenv/testUsdGenPomadeToolsGizmoScreen.py`) that do not import `pxr`. T0/T1 labels are the headless gates; `ctest -L '^T[01]$'` is the usual agent run when a build exists. `.github/workflows/usdgen.yml` is the CI definition. Do not treat milestone prose in `plan/` as the list of jobs that exist today.

Schema, only after editing `libs/hosts/usdGenUsd/schema/schema.usda`:

```sh
source bin/_env.sh
bin/gen_schema.sh
```

That runs OpenUSD's `usdGenSchema` (the tool in `$USD/bin`, not the `usdGenSchema` CMake target) and `libs/hosts/usdGenUsd/schema/restore_generated_metadata.py`. Review the diff in `libs/hosts/usdGenUsd/schema/resources/` before committing. If `usdGenSchema` is not installed, do not hand-edit generated files unless you are applying the same rename already made in `schema.usda`, and say so.

`source bin/_env.sh` is how to get `usdcat` and `PXR_PLUGINPATH_NAME`. Launch `usdview` as `"$PY" "$USD/bin/usdview"`. Do not rely on the shebang inside the OpenUSD prefix.

## Conventions

- Naming: the authoring tool is Pomade (`Pomade`, `pomade`, `POMADE`). The scale helper is `scaleFactor`, not a DCC-prefixed name.
- Stay MIT-consistent. New project files get `Copyright (c) 2026 Nick Burkard` and `SPDX-License-Identifier: MIT`. Do not relicense `thirdparty/` or upstream noodles files. See `NOTICE` and `THIRD_PARTY.md`.
- Do not change `attributeBake.*` logic unless the task explicitly says so.
- No workstation paths, home directories, Tailscale hostnames, or private model names in docs, comments, scenes, or scripts. Use `$USD`, `$VENV`, `$GEN`, `$USDGEN_SRC`, or a relative path.
- Do not reintroduce DCC marketing names or studio design-target names that the open-source pass removed. MoonRay / `hdMoonray` stay where they are the Hydra delegate API. Navigation preset labels that the Pomade UI actually shows stay as those UI strings.
- Prefer editing the design in `plan/` only when the task is a design change. Do not delete historical plans; mark them historical instead.

## Do not commit

Build trees (`build/`, `workusdGenbuild*/`), `*.o`, `*.os`, `*.obj`, `*.a`, `*.so`, `*.dylib`, `*.dll`, `*.pyc`, `renders/ue-parity/`, secrets, and local capture dumps. `.gitignore` lists the usual artifacts.

## Read before a large change

1. [README.md](README.md) and [docs/README.md](docs/README.md)
2. [plan/README.md](plan/README.md), then the overlay that owns the area (`plan/13-codebase-alignment.md` for plan-versus-code, `plan/15-resource-aware-execution.md` for the CUDA execution overlay, `plan/17-pomade-authoring-tool.md` for Pomade)
3. The user doc for the surface you are changing (`docs/pomade-tool.md`, `docs/storm-fur.md`)
4. `tests/checks/check_pomade_docs.py` if you edit Pomade hotkeys or `docs/pomade-tool.md`

`plan/research/ENVIRONMENT.md` is a neutralized snapshot of one old measurement host. It is not a setup guide.
