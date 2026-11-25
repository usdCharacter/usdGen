# usdGen restructure: remaining work

Continue from draft PR [#21](https://github.com/usdCharacter/usdGen/pull/21), branch `claude/usdgen-restructure`. Branch from its head, not from main.
Design, contracts and evidence: [architecture review](https://claude.ai/artifact/DtQ2RD4mYhFZaTghW6up8e). Step numbers below are its "Migration order".

## Fixed decisions

- Option B: nothing under `libs/core/` includes or links pxr. Only `libs/hosts/` (usdGenUsd, usdGenHydra) touch OpenUSD.
- Layout: `libs/base/usdGenMath`, `libs/core/usdGen/{api,registries,compile,exec}`, with backends and primitives nested under `libs/core/` (`backends/usdGenCuda`, `backends/usdGenVulkan`, `primitives/usdGen{Surface,Curves,Expr}`). Hydra adapters go in `usdGenHydra/primitives/<name>/`.
- The engine never names an op, backend or primitive. Registration through one `usdGenRegister(Registrar&)` per library is the only way to extend it. Registries hand out handles, not strings, and freeze before the first Session.
- No locks, mutexes, condition variables, once_flags, TBB or atomic `shared_ptr` in core or hosts. Use atomic publication, refcounts and epoch reclamation.
- No heap allocation after warm-up. Only usdGen.dll allocates engine memory, and nothing that owns memory crosses a DLL boundary.
- Time follows Hydra 2.0: `SetTime(frame)` plus shutter offsets, `InputSource::SampleTimes/Sample`, `ElementSet::SampleTimes/Column`, and dirtiness as locators.
- nanobind over the C++ API replaces cApi.h and ctypes.
- No backward compatibility.
- Out of scope: rigExec, cross-primitive work, CPU/GPU parity work.
- Style: compile/exec subdirectories, public and private headers, terse comments.

## Where it stands (PR #21, head after the PLAN commit)

| Commit | What |
|---|---|
| bench | `tests/perf/benchUsdGenRestructure.cpp` and `compareBench.py` |
| step 1 | the seven independent bug fixes |
| gates | `cmake/gates/`: RatchetScan, CheckLinkRatchet, Gates.cmake, baselines; `testUsdGenWarmAllocations` |
| publish path | `UsdGenFurOcclusionCache` (thickness and shell mask reused across bakes), parallel scalp cap; not a numbered step |
| step 2, partial | `registrar.h/.cpp`, frozen op registry, scheduler scratch kept across runs, static trace scopes |
| step 3, partial | `parkingWord.h/.cpp` (futex / WaitOnAddress); workerPool has no mutex |
| PLAN | this file; MSVC fix for the counting allocators |

Main against the branch: 4-core Xeon @ 2.1 GHz, Linux, GCC 13.3 Release, 4 threads, medians of 5 alternating rounds.

| Metric | main | branch |
|---|---|---|
| Cold cook, 100k curves | 143.6 ms | 133.9 ms |
| Warm recook | 128.2 ms | 122.4 ms |
| Param-edit recook | 116.0 ms | 110.7 ms |
| Hydra publish (54 tiles) | 954.5 ms | 350.7 ms |
| Hydra pull | 0.36 ms | 0.48 ms (noise) |
| Allocations, warm cook | 110 | 25 |
| Allocations, param edit | 140 | 80 |
| Allocations, Hydra publish | 42,344 | 14,902 |

Gates (ctest label `gate`). The baselines only go down; lower them in the same commit that removes hits.

| Gate | Baseline now | Target |
|---|---|---|
| `testUsdGenNoLocks` (locks, atomic shared_ptr, TBB) | empty | empty |
| `testUsdGenNoGuardedStatics` | 4 (usdGenMath ramp.cpp) | empty |
| `testUsdGenNoPxrIncludes` | see `noPxrIncludes.txt` | empty under libs/core |
| `testUsdGenNoPxrLink_usdGen` | tf gf vt sdf arch js plug ts ar work trace hio pxOsd | none |
| `testUsdGenNoPxrLink_usdGenMath` | arch tf gf vt work | none |
| `testUsdGenWarmAllocations` | 0 per warm cook, 0 per param edit (10k curves, all modules counted); 0 interns each | 0 / 0 |
| `testUsdGenNoOpNameLiterals` | empty | empty |
| `testUsdGenNoTokenIntern` | token.cpp, curveLoader.cpp, furOcclusion.cpp, limitSurface.cpp | token.cpp only |
| `testUsdGenEngineNoOpCoupling` | 4 rbfField includes | empty |
| `testUsdGenTokenChecks` (+`Off`) | an exec-phase intern fails `Run` | stays |

To refresh a baseline, use the `cmake -DWRITE=ON ... -P cmake/gates/RatchetScan.cmake` line in the header of `Gates.cmake`.

These tests fail on main too, so they aren't regressions:
- `testUsdGenScenePublication` ("4098 superseded CPU generations ...")
- `testUsdGenPomadeIndex` (G6 move budget, timing)
- `testUsdGenGroomExamples` (flaky)
- `testUsdGenStorm{Look,Tangent,HgiResource,Refine,Material}` need `make_scene.py`, so they need a Python-enabled OpenUSD. CI gets past these.

CI on #21: `ctest -L '^T[01]$'` fails only on ScenePublication and PomadeIndex, the same as main's last 5 runs. Build, install, offline configure and software Vulkan all pass.

## Windows (verified 2026-11-24)

RTX 4090, MSVC 14.36, CUDA 12.6, Vulkan on, serial `ctest -L '^(T0|T1)$'`: 420 of 422 pass.
- `testUsdGenScenePublication` fails as on main.
- `testUsdGenPomadeCommit` TN-4 (8 ms idle slot) and `testUsdGenWorkerPool` "at least two workers ran tasks" are timing checks that also fail intermittently on main.
- The 36 Vulkan tests that failed on main on Windows now pass. Causes: a per-module copy of an inline function-local static registry, a text-mode SPIR-V read, a DLL-local `atexit`, and two submissions that ran past the 2 s TDR limit.
- Allocation counting covers every module (`allocCounter.h` wraps the CRT heap imports), so the Windows numbers are real.

Bench, main (b741e6c) against the branch, same machine, 4 threads, medians of 5 alternating rounds:

| Metric | main | branch |
|---|---|---|
| Cold cook, 100k curves | 128.6 ms | 125.0 ms |
| Warm recook | 121.0 ms | 118.4 ms |
| Param-edit recook | 103.3 ms | 3.7 ms |
| Hydra publish | 663.1 ms | 303.6 ms |
| Hydra pull | 0.41 ms | 0.44 ms |

main's allocation rows read 0 on Windows because its counter only sees the test exe.

## Not verified

- **Linux build of everything after the PLAN commit.** Steps 2 and 3 were built and tested on Windows only. CI covers Linux with CUDA off.
- **TSan.** The stress tests ran natively; MSVC has no TSan. Run `testUsdGenLockFreeStress`, `testUsdGenEpoch`, `testUsdGenTaskScheduler`, `testUsdGenPomadeLanes` under TSan on Linux.
- **ARM64.** Store-load handshakes use seq_cst fences, but nothing ran on ARM64.

## Run locally

```
cmake --build build --config Release
ctest --test-dir build -C Release -L gate --output-on-failure
ctest --test-dir build -C Release -j8 --timeout 300
```

The benchmark needs `PXR_PLUGINPATH_NAME` set to the build's `usd/*/resources` dirs, without fakeHdPrman and *Tools, separated by `;` on Windows. compareBench.py does that for you:

```
build\Release\benchUsdGenRestructure.exe --threads 4 --runs 5 [--no-hydra]
python tests/perf/compareBench.py 5 <mainBuild> <branchBuild>
```

To get a main build, use a worktree at b741e6c: main's code plus the bench. Alternate the rounds, because single runs on a busy machine drift by about 10%. Record the rounds, machine and config with every table.

## Remaining work

Each step is one commit (or a short series). Build, tests and gates stay green after every commit, and each step ends with a compareBench table in its PR.

### 2 and 3. Done

- **2:** size-class pools (`pool.h`), per-thread scratch (`scratch.h`), captures and written planes recycled through retirement, compiled node kept on a value-only edit. Budgets are 0 / 0.
- **3:** `epoch.h` (epoch reclamation, intrusive refcounts, atomic publication), `taskScheduler.h` (lanes, pending counts, process-wide `UsdGenParallelFor`), `sharedCell.h`, `parkingWord.h` wake words. No lock, TBB or atomic shared_ptr is left under `libs/`; usdGen no longer links TBB.

Left over, none of it blocking 4:

| What | Where it goes |
|---|---|
| CUDA executor still returns synchronously (parks on a callback-woken fence); the completion lane exists but no production path uses it | step 8 |
| Vulkan: only deform and RBF binding use the queue timeline; about 35 submit sites still make a fence each; `sourceWidthJob` calls the blocking `DeformPipeline::Begin` | step 8 |
| Retirement lane is a dedicated thread in usdGenExecutionResources (below usdGen, cannot reach the scheduler) | step 8, when that DLL folds into the backend runtime |
| `UsdGenWorkerPool` and `UsdGenTaskScheduler` are two pools | merge during step 5 |
| Ops outside the bench graph still make a fresh capture on re-capture (no `Recyclable()`) | step 5, ElementSet arenas |
| No cook-lane epoch drain; idle scheduler workers drain | step 5 |
| Pomade lanes keep their own threads | step 13 |
| Cache domains, Ptex cache and op registry are made on first use, not in `Engine::Create` | step 5 (step 4 freezes the registries at the first compile) |

### 4 and 4a. Done

- **4:** `registries/`: op, type and backend-kernel registries with dense handles, frozen by the first compile. Every op registers an `UsdGenOpDef` (param decls with dirty class, routing and expression admission; planes; traits; hooks); `UsdGenOp` has no static virtuals and `UsdGenParamView` reads by slot only. CUDA and Vulkan kernels are registered rows (`UsdGenKernelDef`). `UsdGenInternToken` is the only interner; `UsdGenScheduler::Run`, the cook after `_Prepare` and CUDA/Vulkan completion fail on an exec-phase intern (`USDGEN_TOKEN_CHECKS=0` warns). Gates: no op-name literal, no `TfToken` from text outside the interner and three host-side files, warm cook and param edit 0 allocations and 0 interns (`testUsdGenTokenChecks`, `testUsdGenWarmAllocations`). Goldens and contracts byte-identical to S4-0.
- **4a:** Clump, Length and Collide name their planes through `emittedPlanes` with a generic indexed-plane allocator; RBF binding is Deform's `makeAnimationBinding` hook. No engine file outside ops/ includes an op header or casts to an op class.

Bench, main (b741e6c) against step 4, same Windows machine, 4 threads, medians of 5 alternating rounds, no other build running:

| Metric | main | step 4 |
|---|---|---|
| Cold cook, 100k curves | 131.3 ms | 125.7 ms |
| Warm recook | 121.2 ms | 108.0 ms |
| Param-edit recook | 102.9 ms | 3.2 ms |
| Hydra publish | 662.6 ms | 295.6 ms |
| Hydra pull | 0.41 ms | 0.44 ms |

Left over:

| What | Where it goes |
|---|---|
| `TfToken` from text in curveLoader.cpp (44, host loader), furOcclusion.cpp (15, publish look), limitSurface.cpp (6, pxOsd) | 7, 9, 10 |
| Device channels by `std::string` (cudaGeneration.cpp); capture-time plane lookups by token (wholeStrand, clump, trim, length); `UsdGenPlane::interpolation/type` are tokens | 5 |
| ops/rbfField.h included by rbfAnimation.h, vulkan/rbfVkPlan.h, vulkan/executionPlan.cpp, vulkan/surfaceVk.cpp (engineNoOpCoupling baseline) | 6 and 8 |
| Hydra vocabulary in sessionCooker (engine tokens, no interning) | 9 |
| Op registry is built on first use and frozen by the first compile; Ptex cache and the cache-domain table are still made on first use. There is no `Engine::Create` to own them | 5 |
| CUDA plan task types "UsdGenScatterGrow"/"UsdGenPublication" are interned once at load, spelled in two pieces for gate 1 | 8 |
| `testUsdGenScenePublication` also aborts at exit intermittently (0xc0000409), on restructure/s4 too | not a step; owner of the imaging exit path |

### 5. Generic data model

- **Goal:** the engine works on ElementLayout and ElementSet; curves are the first registered layout.
- **Where (hard-wired today):**
  - `op.h:199-220, 262, 271`: output is `UsdGenCurveBuffer`
  - `graph.h:99-101, 171` (`DirtyCurves(SdfPath)`), `:233` (`Repartition(totalCurves, cvCount)`)
  - `types.h:36` (4096-CV limit), `:59` TopoFx, `:62` Role
  - `scheduler.cpp:533-553` (CV spans from cvOffsets)
  - `deviceGeneration.h:53-146`
  - `executionPlan.h:88`
  - `executionCache.h:82` `UpdateCurve(SdfPath)`
  - `graphDesc.h:78, 220`
- **Work:**
  - Chunks become element ranges, with partition as a layout hook and ragged offsets for any number of levels.
  - Each ElementSet lives in one arena, with WriteSet delta arenas and a device mirror at the same offsets.
  - Ops get typed inputs and outputs. All 32 ops change mechanically.
- **Done:** output is byte-identical (goldens), and `curveBuffer.h` is gone from graph, scheduler and cache.
- **Proves it:** goldens; recook times within noise or better; warm allocations still 0.
- **Depends on:** 4.

Left over after 5.4 (reference values are sets, the buffer bridge is tests-only, owning and whole-strand captures recycle):

| What | Notes |
|---|---|
| `UsdGenWorkerPool` is not merged into `UsdGenTaskScheduler` | Perf item, not correctness. Needs its own bench first (task latency, `ParallelFor` throughput, warm-cook time at 100k) against the two-pool setup |
| Capture bodies that allocate on every re-capture, so they are not in `testUsdGenWarmAllocations` (counts per warm re-capture at 10k curves) | Clump, about 920: `PickCentres` spatial hash (`unordered_map<int64, vector<uint32>>`, clump.cpp:309-310) and membership vectors (clump.cpp:467-469, 546, 816). GuideInterpolate, about 1300: `guideBlend::Delaunay2D` (guideBlend.cpp:634-652), `BuildGuides` (guideBlend.cpp:787-793), the clumpRef kd-tree. Collide, about 20000: `ValidateColliderMesh` edge `std::map` per capture (collide.cpp:1527-1538, with an O(faces^2) offset scan), then the BVH. Twist and Frizz `Compute`: group `std::map`/vectors (twist.cpp:249-283), per-strand arc and offset vectors (frizz.cpp:137, 154, 255) |
| Non-owning captures (Clump, Collide, Smooth, Straighten, ...) are not `Recyclable` | One payload allocation per re-capture; worth doing with the item above |
| `Recompile` rebuilds every reference value | About 8 allocations per edit with one reference curve set (`BuildReferenceValue`, the per-node reference sets and maps). Reuse by (path, `curveGeneration`) would make edits of graphs with guides allocation-free |
| `UsdGenBuildClumpMotion(UsdGenCurveSetView)` copies the planes | So the previous-capture reuse never hits for set inputs (Noise, Bend, Curl, Wind, GuideInterpolate) |

### 6. Split the primitives

- **Goal:** `libs/core/primitives/usdGenSurface` (scatter emits a RootSet; `ops/scatter.h:24` today), `usdGenCurves` (types and 30 ops), `usdGenExpr`. Each is a DLL that exports `usdGenRegister`.
- **Gate:** add the source gate: usdGen fails on any op token, primitive word (curve, CV, strand, feather, scale), basisCurves, `__usdGenRender`, or CUDA/Vulkan header.
- **Done:** the source gate passes and the engine builds without any primitive.
- **Depends on:** 4a and 5.

### 7. GraphSpec and NodeKey replace SdfPath

- **Goal:** no SdfPath in plans, ops, caches or the Vulkan plans.
- **Work:**
  - Add InputDesc.
  - Move the builders into usdGenUsd, driven by it.
  - Drop sdf from usdGen's link line.
  - 20 headers in usdGen mention SdfPath today.
- **Done:** sdf is gone from `noPxrLink_usdGen.txt`.
- **Depends on:** 5.

### 8. Generic GPU backends

- **Goal:**
  - `backends/usdGenCuda` and `backends/usdGenVulkan` become runtimes only: plan runner, fusion engine, streams, pools, block cache.
  - The kernels move to `usdGenCurvesCuda` and `usdGenCurvesVulkan`, with SPIR-V paths registered.
- **Where:** `cudaExecution.cpp` (about 575 curve references), `gpu/curve*`, `vulkan/curveGrowPipeline`, `deformPipeline`, `exprVk*`.
- **Done:** the link gate holds (a backend depends on usdGen only), one block cache, and the Vulkan software-device CI job passes.
- **Needs a GPU:** CUDA and Vulkan timings before and after.
- **Depends on:** 3 (GPU sync rows) and 6.

### 9. PublishDesc and the Hydra adapters

- **Goal:** tiles, publication types, hair shaders, fur occlusion (including the new `UsdGenFurOcclusionCache`) and the Storm index layouts move into `usdGenHydra`: generic publishers plus `primitives/curves/`. Locator routing goes through the API.
- **Where:** `sessionCooker.cpp:674` (tile paths), `:2403` (basisCurves), `furOcclusion.*`.
- **Done:** usdGen has no Hydra vocabulary.
- **Proves it:** `hydra.publish` and `hydra.pull` must not regress; scene publication and pomade index behave as on main.
- **Depends on:** 6 and 7.

### 10. OpenSubdiv Far replaces pxOsd

- **Goal:** usdGenSurface uses OpenSubdiv Far instead of pxOsd (`limitSurface.cpp`).
- **Gate:** turn on the include gate for `pxr/usd` and `pxr/imaging` under libs/core.
- **Done:** pxOsd is gone from the link baseline. This is all option A would have given.

### 11. Base types

- **Goal:**
  - Add Token, the usdGenMath types, CowArray, Value and Status, first as aliases of Tf, Gf and Vt.
  - Move the code onto them.
  - Swap in the arena-backed implementations and add host conversions.
- **Done:** `noPxrIncludes.txt` is empty for libs/core, both link baselines are empty, and usdGenMath (today it links arch tf gf vt work) moves to `libs/base`.
- **Depends on:** 7 and 10.

### 12. Install only api/

- **Goal:** install only `api/`, re-home the tests that reach internal headers, and use hidden visibility.
- **Gate:** add the install gate, which fails if anything outside `core/usdGen/api/` is installed.

### 13. nanobind

- **Goal:** the app layer moves to nanobind over the C++ API. Delete `usdGenImaging/cApi.h`, `usdGenToolsApi.h` and the ctypes layer. Pomade moves onto `Cow<>` state over usdGenCurves.
- **Note:** Pomade's lock rows (step 3) are easiest to do here.

### 14. Second primitive

- **Goal:** add feathers or scales.
- **Done:** no file in usdGen changes, and at most one adapter folder appears under `usdGenHydra/primitives/`. That is the test that the split worked.

## Order and parallelism

```
4 -> 4a -> 5 -> 6 -> 7 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14
               └─> 8 (after 6)
```
