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
| `testUsdGenNoLocks` | 553 hits in 50 files (`noLocks.txt`) | empty |
| `testUsdGenNoPxrIncludes` | 262 (`noPxrIncludes.txt`) | empty under libs/core |
| `testUsdGenNoPxrLink_usdGen` | tf gf vt sdf arch js plug ts ar work trace hio pxOsd | none |
| `testUsdGenNoPxrLink_usdGenMath` | arch tf gf vt work | none |
| `testUsdGenWarmAllocations` | 23 per warm cook, 76 per param edit (10k curves) | 0 / 0 |

To refresh a baseline, use the `cmake -DWRITE=ON ... -P cmake/gates/RatchetScan.cmake` line in the header of `Gates.cmake`.

These tests fail on main too, so they aren't regressions:
- `testUsdGenScenePublication` ("4098 superseded CPU generations ...")
- `testUsdGenPomadeIndex` (G6 move budget, timing)
- `testUsdGenGroomExamples` (flaky)
- `testUsdGenStorm{Look,Tangent,HgiResource,Refine,Material}` need `make_scene.py`, so they need a Python-enabled OpenUSD. CI gets past these.

CI on #21: `ctest -L '^T[01]$'` fails only on ScenePublication and PomadeIndex, the same as main's last 5 runs. Build, install, offline configure and software Vulkan all pass.

## Not verified

- **Windows / MSVC build.** Never compiled. Watch these:
  - `parkingWord.cpp`: the WaitOnAddress path and linking `Synchronization.lib`
  - `registrar.cpp`: LoadLibrary / GetProcAddress, and `usdGenRegister_usdGenCurves` exported extern "C"
  - the `_aligned_malloc` branch in the two counting allocators
- **Allocation counting on Windows.** Replacing `operator new` in the test exe only counts the exe's own allocations, because usdGen.dll has its own CRT `operator new`. On MSVC, `testUsdGenWarmAllocations` will read low. Fix: count in usdGen's own pools (step 2's proof in the review), or use a `_CrtSetAllocHook` debug build.
- **Parking under contention on Windows.** On Linux, `testUsdGenWorkerPool` and `testUsdGenScheduling` each passed 40 runs with `USDGEN_POOL_SPIN_US=0`. Repeat that on Windows.
- **CUDA and Vulkan timings.** No GPU, so none were measured. The Vulkan software-device CI job passes.

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

### 2 (rest). Engine allocation through usdGen pools

- **Goal:** warm cook and param edit at 0 allocations.
- **Work:**
  - Size-class pools in usdGen.dll, with per-thread magazines and a depot stack (review: lock-free table, "Size-class pools").
  - A per-worker bump scratch, reset per chunk, replacing chunk-local `std::vector`s.
  - Slot tables and ParamBlocks built once at compile.
  - Arenas recycled through retirement.
- **Find the 23 / 76:** run `testUsdGenWarmAllocations` under heaptrack, or break on `operator new` while counting is on. The param-edit path is `UsdGenCompiler::Recompile` (compiler.cpp).
- **Done:** both budgets are 0. On Windows the count comes from the pool counters (see Not verified).
- **Proves it:** `testUsdGenWarmAllocations`; `allocs.cook.*` in the bench.
- **Depends on:** nothing. Can run in parallel with 3.

### 3 (rest). Locks out of core and hosts

- **Goal:** `noLocks.txt` empty for core and hosts.
- **Done:** the no-lock gate covers libs/core and libs/hosts with no baseline, and a 64-thread Submit/Visible stress test passes under TSan.
- **Proves it:** `testUsdGenNoLocks`; warm and param-edit recook must not regress.
- **Order:** engine first, then Vulkan/CUDA, then imaging, then Pomade.
- **Current hits:** usdGen 243 in 32 files, usdGenImaging 60 in 7, usdGenPomade 250 in 11 (pomadeModel.cpp alone has 186).

| Today | Replacement |
|---|---|
| `executionTaskGraph.cpp:89` mutex, workerReady/drained CVs (15 hits) | engine scheduler; drain = per-graph pending count hitting 0, parked on a `parkingWord` |
| TBB flow graph, task_arena, parallel_for, concurrent_unordered_map (sessionCooker, scheduler, executionPipeline, retirement, scatter, Vulkan deform; 21 files) | engine scheduler and `ParallelFor`; frozen or open-addressing tables |
| `std::atomic_load/store` on shared_ptr (cudaExecution 62, groomSceneIndexPlugin 20, testHook 9, deviceGenerationAdapter 7, executionPipeline 7, sessionCooker 5, session 4, ...) | atomic raw pointer plus intrusive refcount; readers take an epoch guard (load acquire, AddRef, leave); writers store release and retire the old one through epochs |
| `session.cpp:63` CoalescedControl mutex | cache entry state word plus a Treiber follower stack; the leader exchanges it for CLOSED |
| `executionCache.h:298` `mutex_` (cache and LRU) | entry table by atomic pointer, CAS insert, CLOCK reference bits; the cook lane advances the hand |
| `executionCache.h:741` registry mutex | cache domains made in `Engine::Create`, then frozen |
| `executionCache.h:791` in-flight coalescing | entry state word and continuation stack |
| `executionResources.cpp:57` permits | one atomic byte budget: Consume CAS loop, Release fetch_add |
| `executionBackend.cpp:23` AdapterRegistry | BackendRegistry, frozen (reuse `UsdGenFreezeRegistries`) |
| `sessionCooker.cpp:2097` progressMutex | tile slots in the cook arena, fetch_add sequence, release store of the count |
| `cudaExecution.cpp:827` width-blend graph cache | owned by the session cook lane; readers by atomic pointer |
| `vulkan/deformPipeline.cpp:283,302` factor/cache mutex (fence wait inside) | immutable entry by atomic pointer; proof phase on the cook lane; wait becomes a timeline check |
| `deformPipeline.cpp:334-339` heteroPool once/mutexes | made at backend registration on the shared scheduler |
| `deformPipeline.cpp:390` scratchMutex | per-worker scratch |
| `rbfVkBinding.h:98,142` | immutable entry by atomic pointer, relaxed counters |
| `vulkan/fencePool.h:47` | one timeline semaphore per queue; delete the fence pool |
| `vulkan/externalRetirement` mutexes and worker thread | completion lane, SPSC rings, non-blocking queries |
| `ops/rbfField.cpp:733` static mutex | per-worker pool |
| `maps/attributeBake.h:126` mutex, CV, `sleep_for(5ms)` | I/O lane ring plus parking, no polling |
| `maps/ptexMap.cpp:80` once_flag | done in `Engine::Create` |
| `gpu/scatterGrow.cu:1330` copy thread | `cudaMemcpyAsync` on a copy stream plus an event |
| `sessionBackendExecutor.cpp:116` thread per request | a scheduler task |
| ~150 `cuda*Synchronize` | events polled by the completion lane; blocking only at shutdown, device loss and in tests |
| `cudaMalloc`/`vkAllocateMemory` per buffer (`deviceBuffer.cu` 8) | size-class device pools that grow only on topology change |
| `usdGenImagingSession.cpp:141` ProgressRelay | newest-wins atomic pointer |
| `imageMapCache.cpp:33,37` decode once | CAS-inserted loading entry, waiters as continuations, decode on the I/O lane |
| `attributePreviewSceneIndex.h:137,165,169` | copy-on-write tables by atomic pointer, written only on the frontend thread |
| Pomade: pomadeModel, pomadeRegistry, pomadeSceneIndex mutexes; pomadeCommit and pomadeBake CV workers with sleep polling; `pomadeKernels.cu:3296` | model as a Cow snapshot by atomic pointer; registry frozen; rings to the commit and bake lanes with parking; one kernel context per device |
| function-local statics with dynamic init | made in `Engine::Create`; ban them in the gate pattern |

- **Needs first:** an epoch reclamation module (`exec/epoch.h`): global epoch, per-thread slot, retire lists drained by the cook lane. Every atomic shared_ptr replacement depends on it.

### 4. Registries and name tables into OpDef

- **Goal:** exec never looks anything up by name.
- **Work:**
  - Add the type, op and backend registries (`registries/`) with handles.
  - Move these name tables into OpDef traits and hooks: digest params, expression admission, clump plane names, RBF, rest ownership.
  - Resolve parameter and channel names to slots at compile.
- **Where:**
  - `compiler.cpp:18` includes `ops/clump.h`; `:641-673` holds the hard-coded op and clump tokens; `:708` clump plane naming
  - digest and param tables in compiler, session, sessionCooker, executionTaskGraph, cudaExecution and cudaGeneration
  - admission in `expressions/cpuEvaluator.h`
  - `TfToken("UsdGenDeform")` and `dynamic_cast<UsdGenDeformOp const*>` in `sessionCooker.cpp:2786-2809` (RBF animation binding)
- **Done:**
  - no `TfToken("UsdGen...")` literal outside an op's registration
  - the Token interner's debug counter fails any cook that interns
  - output unchanged
- **Proves it:** golden tests; `testUsdGenWarmAllocations`; recook times.
- **Depends on:** 2.

### 4a. Untangle compiler from clump and sessionCooker from deform (before 6)

- **Why:** step 6 moves ops into a separate DLL, so the engine can't include op headers.
- **Work:**
  - Clump plane naming becomes a Clump OpDef hook (`EmittedPlanes(params)`), so compiler.cpp drops `ops/clump.h`.
  - RBF animation binding becomes an OpDef hook (`MakeAnimationBinding`) that Deform registers. sessionCooker asks the terminal op's def instead of `dynamic_cast`, and drops `ops/deform.h`.
- **Done:** `grep -rn 'usdGen/ops/' libs/usdGen/usdGen/*.cpp` finds only ops/ itself.

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
2 rest ─┐
3 rest ─┼─> 4 -> 4a -> 5 -> 6 -> 7 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14
        └──────────────────────> 8 (after 3 GPU rows and 6)
```
